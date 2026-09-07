#include "localization/level1_coarse_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>

#include "localization/global_localization_gravity.hpp"
#include "localization/level0_distance_field.hpp"

namespace localization {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct TranslationOffset {
  double x = 0.0;
  double y = 0.0;
};

bool finitePoint(const pcl::PointXYZI& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
      std::isfinite(point.z);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr finiteCloud(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& input) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr output(
      new pcl::PointCloud<pcl::PointXYZI>());
  if (!input) {
    return output;
  }
  output->points.reserve(input->size());
  for (const pcl::PointXYZI& point : *input) {
    if (finitePoint(point)) {
      output->push_back(point);
    }
  }
  return output;
}

std::vector<TranslationOffset> makeTranslationOffsets(const GLParams& params) {
  std::vector<TranslationOffset> offsets;
  if (!std::isfinite(params.level1_search_radius_xy) ||
      !std::isfinite(params.level1_seed_stride_xy) ||
      params.level1_search_radius_xy <= 0.0 ||
      params.level1_seed_stride_xy <= 0.0) {
    return offsets;
  }
  const int steps = static_cast<int>(std::floor(
      params.level1_search_radius_xy / params.level1_seed_stride_xy));
  for (int y_index = -steps; y_index <= steps; ++y_index) {
    for (int x_index = -steps; x_index <= steps; ++x_index) {
      const double x = static_cast<double>(x_index) *
          params.level1_seed_stride_xy;
      const double y = static_cast<double>(y_index) *
          params.level1_seed_stride_xy;
      if (std::hypot(x, y) <= params.level1_search_radius_xy + 1e-9) {
        offsets.push_back({x, y});
      }
    }
  }
  std::sort(offsets.begin(), offsets.end(), [](const TranslationOffset& first,
                                               const TranslationOffset& second) {
    const double first_radius = std::hypot(first.x, first.y);
    const double second_radius = std::hypot(second.x, second.y);
    if (first_radius != second_radius) {
      return first_radius < second_radius;
    }
    if (first.x != second.x) {
      return first.x < second.x;
    }
    return first.y < second.y;
  });
  return offsets;
}

double percentile90(std::vector<double>& residuals) {
  if (residuals.empty()) {
    return 1e9;
  }
  std::sort(residuals.begin(), residuals.end());
  const size_t index = static_cast<size_t>(
      std::ceil(0.9 * static_cast<double>(residuals.size()))) - 1U;
  return residuals[std::min(index, residuals.size() - 1U)];
}

bool passesCoarseMetrics(const GlobalPoseCandidate& candidate,
                         const GLParams& params) {
  return candidate.converged && std::isfinite(candidate.coarse_score) &&
      candidate.overlap_ratio >= params.coarse_min_overlap &&
      candidate.inlier_count >= params.coarse_min_inliers &&
      candidate.p90_residual <= params.coarse_max_p90_residual;
}

bool coarseCandidateLess(const GlobalPoseCandidate& first,
                         const GlobalPoseCandidate& second) {
  if (first.overlap_ratio != second.overlap_ratio) {
    return first.overlap_ratio > second.overlap_ratio;
  }
  if (first.p90_residual != second.p90_residual) {
    return first.p90_residual < second.p90_residual;
  }
  if (first.coarse_score != second.coarse_score) {
    return first.coarse_score < second.coarse_score;
  }
  if (first.grid_cell_id != second.grid_cell_id) {
    return first.grid_cell_id < second.grid_cell_id;
  }
  if (first.yaw_bin != second.yaw_bin) {
    return first.yaw_bin < second.yaw_bin;
  }
  return first.support_group < second.support_group;
}

}  // namespace

bool makeLevel1YawOffsets(const GLParams& params,
                          std::vector<double>& offsets_deg) {
  offsets_deg.clear();
  if (params.level0_yaw_samples <= 0 ||
      !std::isfinite(params.level1_yaw_step_deg) ||
      params.level1_yaw_step_deg <= 0.0) {
    return false;
  }
  const double half_range = 180.0 /
      static_cast<double>(params.level0_yaw_samples);
  offsets_deg.push_back(-half_range);
  for (double value = -half_range + params.level1_yaw_step_deg;
       value < half_range - 1e-9;
       value += params.level1_yaw_step_deg) {
    offsets_deg.push_back(value);
  }
  offsets_deg.push_back(0.0);
  offsets_deg.push_back(half_range);
  std::sort(offsets_deg.begin(), offsets_deg.end());
  offsets_deg.erase(
      std::unique(offsets_deg.begin(), offsets_deg.end(),
                  [](double first, double second) {
                    return std::abs(first - second) <= 1e-9;
                  }),
      offsets_deg.end());
  return !offsets_deg.empty();
}

bool generateLevel1Seeds(
    const std::vector<Level0Hypothesis>& regions,
    const Eigen::Matrix3d& R_level_base,
    int prior_count,
    const std::string& evidence_id,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& seeds,
    Level1GenerationStats& stats,
    GLSummaryCounts& summary) {
  seeds.clear();
  stats = Level1GenerationStats{};
  const int per_batch_budget = params.max_candidates_total - prior_count;
  const std::vector<TranslationOffset> translation_offsets =
      makeTranslationOffsets(params);
  std::vector<double> yaw_offsets;
  if (regions.empty() || !R_level_base.allFinite() ||
      per_batch_budget <= 0 || translation_offsets.size() < 2U ||
      !makeLevel1YawOffsets(params, yaw_offsets)) {
    return false;
  }

  stats.region_count = static_cast<int>(regions.size());
  stats.translation_template_count_per_region =
      static_cast<int>(translation_offsets.size());
  stats.yaw_count_per_template = static_cast<int>(yaw_offsets.size());
  const size_t seed_count = regions.size() * translation_offsets.size() *
      yaw_offsets.size();
  if (seed_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  seeds.reserve(seed_count);

  int seed_index = 0;
  for (size_t region_index = 0; region_index < regions.size(); ++region_index) {
    const Level0Hypothesis& region = regions[region_index];
    if (!region.score.valid || !region.T_map_level.allFinite()) {
      return false;
    }
    for (size_t translation_index = 0;
         translation_index < translation_offsets.size(); ++translation_index) {
      const TranslationOffset& offset = translation_offsets[translation_index];
      std::ostringstream support_group;
      support_group << "grid:" << region.grid_cell_id << ":"
                    << region.yaw_bin << ":xy:" << translation_index;
      for (double yaw_offset : yaw_offsets) {
        Eigen::Matrix4d T_map_level = region.T_map_level;
        const double yaw_deg = region.yaw_deg + yaw_offset;
        T_map_level.block<3, 3>(0, 0) = Eigen::AngleAxisd(
            yaw_deg * kPi / 180.0,
            Eigen::Vector3d::UnitZ()).toRotationMatrix();
        T_map_level(0, 3) += offset.x;
        T_map_level(1, 3) += offset.y;

        GlobalPoseCandidate candidate;
        if (!composeMapBaseSeed(T_map_level, R_level_base,
                                candidate.seed_pose)) {
          return false;
        }
        candidate.final_pose = candidate.seed_pose;
        applyLevel0ScoreToCandidate(region.score, candidate);
        candidate.source_type = "grid";
        candidate.grid_cell_id = region.grid_cell_id;
        candidate.yaw_bin = region.yaw_bin;
        candidate.rotation_batch = seed_index / per_batch_budget;
        candidate.support_group = support_group.str();
        candidate.evidence_id = evidence_id;
        candidate.evidence_role = "anchor_recall";
        seeds.push_back(candidate);
        ++seed_index;
      }
    }
  }

  stats.seed_count = static_cast<int>(seeds.size());
  stats.batch_count =
      (stats.seed_count + per_batch_budget - 1) / per_batch_budget;
  summary.level1_seed_count = stats.seed_count;
  summary.level1_batch_count = stats.batch_count;
  return true;
}

bool runLevel1CoarseAlignment(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& global_map,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& candidates,
    Level1CoarseStats& stats) {
  stats = Level1CoarseStats{};
  if (!source || source->empty() || !global_map || global_map->empty() ||
      candidates.empty() || params.coarse_leaf_src <= 0.0 ||
      params.coarse_leaf_map <= 0.0 ||
      params.coarse_max_iter <= 0 || params.coarse_roi_radius <= 0.0 ||
      params.coarse_max_corr_dist <= 0.0) {
    return false;
  }

  const pcl::PointCloud<pcl::PointXYZI>::Ptr finite_source = finiteCloud(source);
  const pcl::PointCloud<pcl::PointXYZI>::Ptr finite_map = finiteCloud(global_map);
  if (finite_source->empty() || finite_map->empty()) {
    return false;
  }
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_coarse(
      new pcl::PointCloud<pcl::PointXYZI>());
  pcl::VoxelGrid<pcl::PointXYZI> voxel;
  voxel.setInputCloud(finite_source);
  const float leaf = static_cast<float>(params.coarse_leaf_src);
  voxel.setLeafSize(leaf, leaf, leaf);
  voxel.filter(*source_coarse);
  pcl::PointCloud<pcl::PointXYZI>::Ptr map_coarse(
      new pcl::PointCloud<pcl::PointXYZI>());
  voxel.setInputCloud(finite_map);
  const float map_leaf = static_cast<float>(params.coarse_leaf_map);
  voxel.setLeafSize(map_leaf, map_leaf, map_leaf);
  voxel.filter(*map_coarse);
  if (source_coarse->empty() || map_coarse->empty()) {
    return false;
  }

  for (GlobalPoseCandidate& candidate : candidates) {
    ++stats.evaluated;
    candidate.converged = false;
    candidate.coarse_score = 1e9;
    candidate.overlap_ratio = 0.0;
    candidate.inlier_count = 0;
    candidate.p90_residual = 1e9;
    const double center_x = candidate.seed_pose(0, 3);
    const double center_y = candidate.seed_pose(1, 3);
    pcl::PointCloud<pcl::PointXYZI>::Ptr target_roi(
        new pcl::PointCloud<pcl::PointXYZI>());
    target_roi->points.reserve(map_coarse->size());
    for (const pcl::PointXYZI& point : *map_coarse) {
      if (std::hypot(static_cast<double>(point.x) - center_x,
                     static_cast<double>(point.y) - center_y) <=
          params.coarse_roi_radius) {
        target_roi->push_back(point);
      }
    }
    if (target_roi->size() < 3U) {
      continue;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*source_coarse, *transformed,
                             candidate.seed_pose.cast<float>());
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    icp.setInputSource(transformed);
    icp.setInputTarget(target_roi);
    icp.setMaximumIterations(params.coarse_max_iter);
    icp.setMaxCorrespondenceDistance(params.coarse_max_corr_dist);
    icp.setTransformationEpsilon(1e-6);
    pcl::PointCloud<pcl::PointXYZI> aligned;
    icp.align(aligned);
    const Eigen::Matrix4f correction = icp.getFinalTransformation();
    candidate.converged = icp.hasConverged() && correction.allFinite();
    if (!candidate.converged) {
      continue;
    }
    ++stats.converged;
    candidate.coarse_score = icp.getFitnessScore();
    candidate.final_pose = correction.cast<double>() * candidate.seed_pose;

    pcl::KdTreeFLANN<pcl::PointXYZI> tree;
    tree.setInputCloud(target_roi);
    std::vector<double> residuals;
    residuals.reserve(aligned.size());
    std::vector<int> nearest_index(1);
    std::vector<float> nearest_squared_distance(1);
    const double max_squared_distance =
        params.coarse_max_corr_dist * params.coarse_max_corr_dist;
    for (const pcl::PointXYZI& point : aligned) {
      if (tree.nearestKSearch(point, 1, nearest_index,
                              nearest_squared_distance) > 0 &&
          nearest_squared_distance[0] <= max_squared_distance) {
        residuals.push_back(std::sqrt(nearest_squared_distance[0]));
      }
    }
    candidate.inlier_count = static_cast<int>(residuals.size());
    candidate.overlap_ratio = residuals.size() /
        static_cast<double>(std::max<size_t>(1U, aligned.size()));
    candidate.p90_residual = percentile90(residuals);
    if (passesCoarseMetrics(candidate, params)) {
      ++stats.passed_metrics;
    }
  }
  return true;
}

bool selectLevel1RefineCandidates(
    const std::vector<GlobalPoseCandidate>& coarse_candidates,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& selected,
    Level1CoarseStats& stats) {
  selected.clear();
  if (params.max_refine_candidates <= 0) {
    return false;
  }
  std::vector<GlobalPoseCandidate> passing;
  for (const GlobalPoseCandidate& candidate : coarse_candidates) {
    if (passesCoarseMetrics(candidate, params)) {
      passing.push_back(candidate);
    }
  }
  std::sort(passing.begin(), passing.end(), coarseCandidateLess);

  std::set<std::pair<int, int>> selected_regions;
  std::vector<bool> used(passing.size(), false);
  for (size_t index = 0; index < passing.size() &&
       static_cast<int>(selected.size()) < params.max_refine_candidates; ++index) {
    const std::pair<int, int> region(
        passing[index].grid_cell_id, passing[index].yaw_bin);
    if (selected_regions.insert(region).second) {
      selected.push_back(passing[index]);
      used[index] = true;
    }
  }
  for (size_t index = 0; index < passing.size() &&
       static_cast<int>(selected.size()) < params.max_refine_candidates; ++index) {
    if (!used[index]) {
      selected.push_back(passing[index]);
    }
  }
  stats.selected_for_refine = static_cast<int>(selected.size());
  return true;
}

}  // namespace localization
