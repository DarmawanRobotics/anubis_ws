#include "localization/global_localization_refinement.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

#include "localization/global_localization_math.hpp"

namespace localization {
namespace {

constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

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

double yawDeg(const Eigen::Matrix3d& rotation) {
  return std::atan2(rotation(1, 0), rotation(0, 0)) * kRadiansToDegrees;
}

double rollDeg(const Eigen::Matrix3d& rotation) {
  return std::atan2(rotation(2, 1), rotation(2, 2)) * kRadiansToDegrees;
}

double pitchDeg(const Eigen::Matrix3d& rotation) {
  const double sin_pitch = std::clamp(-rotation(2, 0), -1.0, 1.0);
  return std::asin(sin_pitch) * kRadiansToDegrees;
}

void updateCandidatePoseMetrics(GlobalPoseCandidate& candidate) {
  const Eigen::Vector3d translation_delta =
      candidate.final_pose.block<3, 1>(0, 3) -
      candidate.seed_pose.block<3, 1>(0, 3);
  candidate.correction_xy = std::hypot(translation_delta.x(), translation_delta.y());
  candidate.correction_z = std::abs(translation_delta.z());

  const Eigen::Matrix3d seed_rotation = candidate.seed_pose.block<3, 3>(0, 0);
  const Eigen::Matrix3d final_rotation = candidate.final_pose.block<3, 3>(0, 0);
  candidate.final_yaw_deg = yawDeg(final_rotation);
  candidate.yaw_delta_deg = global_localization_math::wrapAngleDeg(
      candidate.final_yaw_deg - yawDeg(seed_rotation));
  candidate.roll_delta_deg = global_localization_math::wrapAngleDeg(
      rollDeg(final_rotation) - rollDeg(seed_rotation));
  candidate.pitch_delta_deg = global_localization_math::wrapAngleDeg(
      pitchDeg(final_rotation) - pitchDeg(seed_rotation));
}

}  // namespace

bool runDoubleIcpRefinement(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& target,
    const Eigen::Matrix4d& alignment_start_pose,
    const GLParams& params,
    GlobalPoseCandidate& candidate) {
  candidate.final_pose = alignment_start_pose;
  candidate.converged = false;
  candidate.fitness_score = 1e9;
  if (!source || source->empty() || !target || target->empty() ||
      !candidate.seed_pose.allFinite() || !alignment_start_pose.allFinite() ||
      params.refine_max_iter <= 0 ||
      !std::isfinite(params.refine_fitness_threshold) ||
      params.refine_fitness_threshold <= 0.0) {
    return false;
  }
  updateCandidatePoseMetrics(candidate);

  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed(
      new pcl::PointCloud<pcl::PointXYZI>());
  pcl::transformPointCloud(
      *source, *transformed, alignment_start_pose.cast<float>());

  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  icp.setInputTarget(target);
  icp.setInputSource(transformed);
  icp.setMaximumIterations(params.refine_max_iter);
  icp.setTransformationEpsilon(1e-6);

  pcl::PointCloud<pcl::PointXYZI> aligned;
  icp.align(aligned);
  const Eigen::Matrix4f correction_first = icp.getFinalTransformation();
  candidate.converged = icp.hasConverged() && correction_first.allFinite();
  candidate.fitness_score = icp.getFitnessScore();
  if (!candidate.converged) {
    return true;
  }

  candidate.final_pose = correction_first.cast<double>() * alignment_start_pose;
  updateCandidatePoseMetrics(candidate);
  if (!std::isfinite(candidate.fitness_score) ||
      candidate.fitness_score >= params.refine_fitness_threshold) {
    return true;
  }

  pcl::transformPointCloud(*transformed, *transformed, correction_first);
  icp.setInputSource(transformed);
  icp.align(aligned);
  const Eigen::Matrix4f correction_second = icp.getFinalTransformation();
  candidate.converged = icp.hasConverged() && correction_second.allFinite();
  candidate.fitness_score = icp.getFitnessScore();
  if (!candidate.converged) {
    return true;
  }

  candidate.final_pose = global_localization_math::composeAbsolutePose(
      correction_second.cast<double>(), correction_first.cast<double>(),
      alignment_start_pose);
  updateCandidatePoseMetrics(candidate);
  return true;
}

bool runLevel1Refinement(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& global_map,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& candidates,
    GlobalRefinementStats& stats) {
  stats = GlobalRefinementStats{};
  if (!source || source->empty() || !global_map || global_map->empty() ||
      candidates.empty() || !std::isfinite(params.refine_leaf_src) ||
      params.refine_leaf_src <= 0.0 || params.refine_max_iter <= 0 ||
      !std::isfinite(params.refine_fitness_threshold) ||
      params.refine_fitness_threshold <= 0.0) {
    return false;
  }

  const pcl::PointCloud<pcl::PointXYZI>::Ptr finite_source = finiteCloud(source);
  const pcl::PointCloud<pcl::PointXYZI>::Ptr target = finiteCloud(global_map);
  if (finite_source->empty() || target->empty()) {
    return false;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr source_refine(
      new pcl::PointCloud<pcl::PointXYZI>());
  pcl::VoxelGrid<pcl::PointXYZI> voxel;
  voxel.setInputCloud(finite_source);
  const float leaf = static_cast<float>(params.refine_leaf_src);
  voxel.setLeafSize(leaf, leaf, leaf);
  voxel.filter(*source_refine);
  if (source_refine->empty()) {
    return false;
  }

  for (GlobalPoseCandidate& candidate : candidates) {
    const Eigen::Matrix4d alignment_start_pose = candidate.final_pose;
    ++stats.attempted;
    if (!runDoubleIcpRefinement(
            source_refine, target, alignment_start_pose, params, candidate)) {
      continue;
    }
    if (candidate.converged) {
      ++stats.converged;
      if (std::isfinite(candidate.fitness_score) &&
          candidate.fitness_score < params.refine_fitness_threshold) {
        ++stats.completed;
      }
    }
  }
  return true;
}

}  // namespace localization
