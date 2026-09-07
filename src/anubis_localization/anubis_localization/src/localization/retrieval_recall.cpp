#include "localization/retrieval_recall.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <limits>
#include <utility>

#include <pcl/point_cloud.h>

#include <Eigen/LU>

#include "localization/global_localization_gravity.hpp"
#include "scan_descriptor/point_adapter.hpp"

namespace localization {
namespace {
constexpr double kPi = 3.14159265358979323846;

bool isRigidTransform(const Eigen::Matrix4d& transform) {
  if (!transform.allFinite()) return false;
  const Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
  const Eigen::Matrix3d orthogonality =
      rotation.transpose() * rotation - Eigen::Matrix3d::Identity();
  return orthogonality.cwiseAbs().maxCoeff() <= 1e-6 &&
      std::abs(rotation.determinant() - 1.0) <= 1e-6 &&
      transform.row(3).isApprox(
          Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-9);
}

std::vector<std::pair<double, double>> translationOffsets(
    const RetrievalRecallParams& params) {
  std::vector<std::pair<double, double>> offsets;
  if (!std::isfinite(params.search_radius_xy) ||
      !std::isfinite(params.seed_stride_xy) || params.search_radius_xy <= 0.0 ||
      params.seed_stride_xy <= 0.0 || params.search_radius_xy < params.seed_stride_xy) {
    return offsets;
  }
  for (double x = -params.search_radius_xy; x <= params.search_radius_xy + 1e-9;
       x += params.seed_stride_xy) {
    for (double y = -params.search_radius_xy; y <= params.search_radius_xy + 1e-9;
         y += params.seed_stride_xy) {
      if (std::hypot(x, y) <= params.search_radius_xy + 1e-9) offsets.emplace_back(x, y);
    }
  }
  return offsets;
}
}  // namespace

bool runRetrievalRecall(
    const scan_descriptor::DescriptorDatabase& database,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& scan_base,
    const Eigen::Matrix3d& R_level_base,
    const Eigen::Matrix4d& T_base_lidar,
    double ground_z_in_level,
    const RetrievalRecallParams& params,
    std::vector<Level0Hypothesis>& top_regions,
    RetrievalRecallStats& stats) {
  top_regions.clear();
  stats = RetrievalRecallStats{};
  stats.db_size = static_cast<int>(database.size());
  if (database.empty() || !scan_base || scan_base->empty() ||
      !R_level_base.allFinite() || !isRigidTransform(T_base_lidar) ||
      params.topk <= 0 ||
      params.min_points <= 0 || !std::isfinite(params.max_distance) ||
      params.max_distance < 0.0 || !std::isfinite(ground_z_in_level)) {
    return false;
  }
  scan_descriptor::DescriptorConfig config = database.config();
  const auto points = scan_descriptor::adaptPoints(
      *scan_base, R_level_base, ground_z_in_level, config);
  stats.valid_points = static_cast<int>(points.size());
  if (stats.valid_points < params.min_points) return false;
  const auto descriptor = scan_descriptor::makeDescriptor(points, config);
  // SCDB v1 persists T_map_lidar, while the descriptor/query points are in
  // base_link. Query a small over-complete pool, convert every pose with the
  // installation extrinsic, then perform NMS in the actual base frame.
  const size_t retrieval_pool = std::min(
      database.size(), static_cast<size_t>(params.topk) * 5U);
  const auto matches = database.queryTopK(
      descriptor, retrieval_pool, params.max_distance,
      std::numeric_limits<uint64_t>::max(), 0.0);
  const Eigen::Matrix4d T_lidar_base = T_base_lidar.inverse();
  for (const auto& candidate : matches) {
    const auto& record = database.at(candidate.record_index);
    const Eigen::Matrix4d T_map_base =
        record.T_map_lidar * T_lidar_base;
    bool suppressed = false;
    if (params.nms_xy > 0.0) {
      const Eigen::Vector2d position = T_map_base.block<2, 1>(0, 3);
      for (const auto& selected : top_regions) {
        const Eigen::Vector2d selected_position =
            selected.T_map_level.block<2, 1>(0, 3);
        if ((position - selected_position).norm() < params.nms_xy) {
          suppressed = true;
          break;
        }
      }
    }
    if (suppressed) continue;
    Level0Hypothesis region;
    const Eigen::Matrix3d T_map_query_rotation =
        T_map_base.block<3, 3>(0, 0) * Eigen::AngleAxisd(
            candidate.yaw_rad, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d R_map_level =
        T_map_query_rotation * R_level_base.transpose();
    const double map_yaw = std::atan2(R_map_level(1, 0), R_map_level(0, 0));
    region.T_map_level = Eigen::Matrix4d::Identity();
    region.T_map_level.block<3, 3>(0, 0) = Eigen::AngleAxisd(
        map_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    region.T_map_level.block<3, 1>(0, 3) = T_map_base.block<3, 1>(0, 3);
    region.yaw_deg = map_yaw * 180.0 / kPi;
    region.grid_cell_id = static_cast<int>(record.id);
    region.yaw_bin = 0;
    region.rotation_batch = 0;
    region.score.valid = true;
    region.score.score = candidate.distance;
    region.score.mean_truncated_distance = candidate.distance;
    region.score.valid_projection_ratio = 1.0;
    region.score.input_point_count = stats.valid_points;
    region.score.valid_projection_points = stats.valid_points;
    top_regions.push_back(region);
    stats.best_distance = std::min(stats.best_distance, candidate.distance);
    if (top_regions.size() == static_cast<size_t>(params.topk)) break;
  }
  stats.match_count = static_cast<int>(top_regions.size());
  stats.complete = !top_regions.empty();
  return stats.complete;
}

bool generateRetrievalSeeds(
    const std::vector<Level0Hypothesis>& regions,
    const Eigen::Matrix3d& R_level_base,
    const RetrievalRecallParams& retrieval,
    int prior_count,
    const std::string& evidence_id,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& seeds,
    GLSummaryCounts& summary) {
  seeds.clear();
  const auto offsets = translationOffsets(retrieval);
  if (std::isfinite(retrieval.seed_stride_xy) &&
      std::isfinite(params.support_seed_xy) &&
      retrieval.seed_stride_xy <= params.support_seed_xy) {
    return false;
  }
  if (regions.empty() || !R_level_base.allFinite() || offsets.empty() ||
      prior_count >= params.max_candidates_total || !std::isfinite(retrieval.yaw_half_range_deg) ||
      !std::isfinite(retrieval.yaw_step_deg) || retrieval.yaw_half_range_deg <= 0.0 ||
      retrieval.yaw_step_deg <= 0.0 || retrieval.yaw_step_deg > retrieval.yaw_half_range_deg ||
      retrieval.yaw_half_range_deg > 45.0) {
    return false;
  }
  std::vector<double> yaws;
  for (double yaw = -retrieval.yaw_half_range_deg;
       yaw <= retrieval.yaw_half_range_deg + 1e-9; yaw += retrieval.yaw_step_deg) {
    yaws.push_back(yaw);
  }
  const int budget = params.max_candidates_total - prior_count;
  int index = 0;
  for (const auto& region : regions) {
    if (!region.score.valid) return false;
    for (size_t offset_index = 0; offset_index < offsets.size(); ++offset_index) {
      for (double yaw : yaws) {
        if (static_cast<int>(seeds.size()) >= budget) break;
        Eigen::Matrix4d T_map_level = region.T_map_level;
        T_map_level.block<3, 3>(0, 0) = Eigen::AngleAxisd(
            (region.yaw_deg + yaw) * kPi / 180.0,
            Eigen::Vector3d::UnitZ()).toRotationMatrix();
        T_map_level(0, 3) += offsets[offset_index].first;
        T_map_level(1, 3) += offsets[offset_index].second;
        GlobalPoseCandidate candidate;
        if (!composeMapBaseSeed(T_map_level, R_level_base, candidate.seed_pose)) return false;
        candidate.final_pose = candidate.seed_pose;
        applyLevel0ScoreToCandidate(region.score, candidate);
        candidate.source_type = "retrieval";
        candidate.grid_cell_id = region.grid_cell_id;
        candidate.yaw_bin = 0;
        candidate.rotation_batch = index / std::max(1, budget);
        std::ostringstream group;
        group << "grid:kf" << region.grid_cell_id << ":xy:" << offset_index;
        candidate.support_group = group.str();
        candidate.evidence_id = evidence_id;
        candidate.evidence_role = "anchor_recall";
        seeds.push_back(candidate);
        ++index;
      }
    }
  }
  summary.level1_seed_count = static_cast<int>(seeds.size());
  summary.level1_batch_count = (summary.level1_seed_count + budget - 1) / budget;
  return !seeds.empty();
}

}  // namespace localization
