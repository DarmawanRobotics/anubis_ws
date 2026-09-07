#ifndef LEVEL1_COARSE_SEARCH_HPP
#define LEVEL1_COARSE_SEARCH_HPP

#include <string>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "localization/global_localization_types.hpp"
#include "localization/level0_global_search.hpp"

namespace localization {

struct Level1GenerationStats {
  int region_count = 0;
  int translation_template_count_per_region = 0;
  int yaw_count_per_template = 0;
  int seed_count = 0;
  int batch_count = 0;
};

struct Level1CoarseStats {
  int evaluated = 0;
  int converged = 0;
  int passed_metrics = 0;
  int selected_for_refine = 0;
};

bool makeLevel1YawOffsets(const GLParams& params,
                          std::vector<double>& offsets_deg);

bool generateLevel1Seeds(
    const std::vector<Level0Hypothesis>& regions,
    const Eigen::Matrix3d& R_level_base,
    int prior_count,
    const std::string& evidence_id,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& seeds,
    Level1GenerationStats& stats,
    GLSummaryCounts& summary);

bool runLevel1CoarseAlignment(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& global_map,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& candidates,
    Level1CoarseStats& stats);

bool selectLevel1RefineCandidates(
    const std::vector<GlobalPoseCandidate>& coarse_candidates,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& selected,
    Level1CoarseStats& stats);

}  // namespace localization

#endif  // LEVEL1_COARSE_SEARCH_HPP
