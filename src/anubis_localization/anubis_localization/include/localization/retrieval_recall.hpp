#ifndef LOCALIZATION_RETRIEVAL_RECALL_HPP
#define LOCALIZATION_RETRIEVAL_RECALL_HPP

#include <vector>
#include <string>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "localization/global_localization_types.hpp"
#include "localization/level0_global_search.hpp"
#include "scan_descriptor/database.hpp"

namespace localization {

struct RetrievalRecallParams {
  int topk = 3;
  double nms_xy = 4.0;
  double max_distance = 0.40;
  int min_points = 300;
  double yaw_half_range_deg = 10.0;
  double yaw_step_deg = 5.0;
  double search_radius_xy = 1.5;
  double seed_stride_xy = 1.5;
};

struct RetrievalRecallStats {
  int db_size = 0;
  int match_count = 0;
  double best_distance = 1e9;
  int valid_points = 0;
  bool complete = false;
};

bool runRetrievalRecall(
    const scan_descriptor::DescriptorDatabase& database,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& scan_base,
    const Eigen::Matrix3d& R_level_base,
    const Eigen::Matrix4d& T_base_lidar,
    double ground_z_in_level,
    const RetrievalRecallParams& params,
    std::vector<Level0Hypothesis>& top_regions,
    RetrievalRecallStats& stats);

bool generateRetrievalSeeds(
    const std::vector<Level0Hypothesis>& regions,
    const Eigen::Matrix3d& R_level_base,
    const RetrievalRecallParams& retrieval,
    int prior_count,
    const std::string& evidence_id,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& seeds,
    GLSummaryCounts& summary);

}  // namespace localization

#endif  // LOCALIZATION_RETRIEVAL_RECALL_HPP
