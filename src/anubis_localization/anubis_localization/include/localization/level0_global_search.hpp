#ifndef LEVEL0_GLOBAL_SEARCH_HPP
#define LEVEL0_GLOBAL_SEARCH_HPP

#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "localization/global_localization_types.hpp"
#include "localization/level0_distance_field.hpp"

namespace localization {

struct Level0Hypothesis {
  Eigen::Matrix4d T_map_level = Eigen::Matrix4d::Identity();
  Level0Score score;
  int grid_cell_id = -1;
  int yaw_bin = -1;
  int rotation_batch = -1;
  double yaw_deg = 0.0;
};

struct Level0SearchStats {
  bool complete = false;
  int grid_cell_count = 0;
  int batch_count = 0;
  uint64_t hypothesis_count = 0;
  uint64_t lookup_count_upper_bound = 0;
  int valid_hypothesis_count = 0;
  int invalid_hypothesis_count = 0;
};

bool selectLevel0TopRegions(
    const std::vector<Level0Hypothesis>& hypotheses,
    const GLParams& params,
    std::vector<Level0Hypothesis>& top_regions);

bool runLevel0GlobalSearch(
    const Level0DistanceField& distance_field,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& scan_level,
    const GLParams& params,
    std::vector<Level0Hypothesis>& top_regions,
    Level0SearchStats& stats,
    GLSummaryCounts& summary);

}  // namespace localization

#endif  // LEVEL0_GLOBAL_SEARCH_HPP
