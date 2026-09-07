#ifndef GLOBAL_LOCALIZATION_REFINEMENT_HPP
#define GLOBAL_LOCALIZATION_REFINEMENT_HPP

#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "localization/global_localization_types.hpp"

namespace localization {

struct GlobalRefinementStats {
  int attempted = 0;
  int converged = 0;
  int completed = 0;
};

bool runDoubleIcpRefinement(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& target,
    const Eigen::Matrix4d& alignment_start_pose,
    const GLParams& params,
    GlobalPoseCandidate& candidate);

bool runLevel1Refinement(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& global_map,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& candidates,
    GlobalRefinementStats& stats);

}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_REFINEMENT_HPP
