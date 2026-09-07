#ifndef GLOBAL_LOCALIZATION_SPATIAL_HPP
#define GLOBAL_LOCALIZATION_SPATIAL_HPP

#include <vector>

#include "localization/global_localization_types.hpp"

namespace localization {

struct GlobalSpatialDecision {
  GLStatus status = GLStatus::NoCandidate;
  int best_cluster_index = -1;
  int second_cluster_index = -1;
};

bool filterRefinedCandidates(
    const std::vector<GlobalPoseCandidate>& refined_candidates,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& passed_candidates,
    GLSummaryCounts& summary);

bool clusterCandidates(
    const std::vector<GlobalPoseCandidate>& candidates,
    const GLParams& params,
    std::vector<GlobalCluster>& clusters,
    GLSummaryCounts& summary);

GlobalSpatialDecision selectBestCluster(
    const std::vector<GlobalCluster>& clusters,
    const GLParams& params,
    GLSummaryCounts& summary);

}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_SPATIAL_HPP
