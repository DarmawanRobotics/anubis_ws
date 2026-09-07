#include "localization/global_localization_spatial.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "localization/global_localization_math.hpp"

namespace localization {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadiansToDegrees = 180.0 / kPi;
constexpr double kComparisonEpsilon = 1e-12;

class DisjointSet {
public:
  explicit DisjointSet(size_t size) : parent_(size), rank_(size, 0) {
    std::iota(parent_.begin(), parent_.end(), 0U);
  }

  size_t find(size_t value) {
    if (parent_[value] != value) {
      parent_[value] = find(parent_[value]);
    }
    return parent_[value];
  }

  bool unite(size_t first, size_t second) {
    first = find(first);
    second = find(second);
    if (first == second) {
      return false;
    }
    if (rank_[first] < rank_[second]) {
      std::swap(first, second);
    }
    parent_[second] = first;
    if (rank_[first] == rank_[second]) {
      ++rank_[first];
    }
    return true;
  }

private:
  std::vector<size_t> parent_;
  std::vector<int> rank_;
};

double yawDeg(const Eigen::Matrix4d& pose) {
  return std::atan2(pose(1, 0), pose(0, 0)) * kRadiansToDegrees;
}

double rollDeg(const Eigen::Matrix4d& pose) {
  return std::atan2(pose(2, 1), pose(2, 2)) * kRadiansToDegrees;
}

double pitchDeg(const Eigen::Matrix4d& pose) {
  const double sin_pitch = std::clamp(-pose(2, 0), -1.0, 1.0);
  return std::asin(sin_pitch) * kRadiansToDegrees;
}

double yawDistanceDeg(double first, double second) {
  return std::abs(global_localization_math::wrapAngleDeg(first - second));
}

double poseXyDistance(const Eigen::Matrix4d& first,
                      const Eigen::Matrix4d& second) {
  return std::hypot(first(0, 3) - second(0, 3),
                    first(1, 3) - second(1, 3));
}

bool lessDouble(double first, double second) {
  if (first < second) {
    return true;
  }
  return false;
}

bool finitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

bool candidateDeterministicLess(const GlobalPoseCandidate& first,
                                const GlobalPoseCandidate& second) {
  const double first_values[] = {
      first.final_pose(0, 3), first.final_pose(1, 3), first.final_pose(2, 3),
      yawDeg(first.final_pose), first.seed_pose(0, 3), first.seed_pose(1, 3),
      first.fitness_score, -first.overlap_ratio, first.correction_xy};
  const double second_values[] = {
      second.final_pose(0, 3), second.final_pose(1, 3), second.final_pose(2, 3),
      yawDeg(second.final_pose), second.seed_pose(0, 3), second.seed_pose(1, 3),
      second.fitness_score, -second.overlap_ratio, second.correction_xy};
  for (size_t index = 0; index < sizeof(first_values) / sizeof(double); ++index) {
    if (lessDouble(first_values[index], second_values[index])) {
      return true;
    }
    if (lessDouble(second_values[index], first_values[index])) {
      return false;
    }
  }
  if (first.support_group != second.support_group) {
    return first.support_group < second.support_group;
  }
  if (first.source_type != second.source_type) {
    return first.source_type < second.source_type;
  }
  if (first.grid_cell_id != second.grid_cell_id) {
    return first.grid_cell_id < second.grid_cell_id;
  }
  return first.yaw_bin < second.yaw_bin;
}

bool adjacent(const GlobalPoseCandidate& first,
              const GlobalPoseCandidate& second,
              const GLParams& params) {
  return poseXyDistance(first.final_pose, second.final_pose) < params.cluster_xy &&
      yawDistanceDeg(yawDeg(first.final_pose), yawDeg(second.final_pose)) <
          params.cluster_yaw_deg;
}

double normalizedPoseDistance(const GlobalPoseCandidate& first,
                              const GlobalPoseCandidate& second,
                              const GLParams& params) {
  return poseXyDistance(first.final_pose, second.final_pose) / params.cluster_xy +
      yawDistanceDeg(yawDeg(first.final_pose), yawDeg(second.final_pose)) /
          params.cluster_yaw_deg;
}

bool withinClusterDiameter(const std::vector<GlobalPoseCandidate>& candidates,
                           const std::vector<int>& members,
                           const GLParams& params) {
  for (size_t first = 0; first < members.size(); ++first) {
    for (size_t second = first + 1; second < members.size(); ++second) {
      const GlobalPoseCandidate& a = candidates[members[first]];
      const GlobalPoseCandidate& b = candidates[members[second]];
      if (poseXyDistance(a.final_pose, b.final_pose) >
              2.0 * params.cluster_xy + kComparisonEpsilon ||
          yawDistanceDeg(yawDeg(a.final_pose), yawDeg(b.final_pose)) >
              2.0 * params.cluster_yaw_deg + kComparisonEpsilon) {
        return false;
      }
    }
  }
  return true;
}

struct WeightedEdge {
  size_t first = 0;
  size_t second = 0;
  double weight = 0.0;
};

bool edgeLess(const WeightedEdge& first, const WeightedEdge& second) {
  if (first.weight != second.weight) {
    return first.weight < second.weight;
  }
  if (first.first != second.first) {
    return first.first < second.first;
  }
  return first.second < second.second;
}

void splitOversizedComponent(
    const std::vector<GlobalPoseCandidate>& candidates,
    const std::vector<int>& members,
    const GLParams& params,
    std::vector<std::vector<int>>& output) {
  if (members.size() <= 1U || withinClusterDiameter(candidates, members, params)) {
    output.push_back(members);
    return;
  }

  std::vector<WeightedEdge> edges;
  for (size_t first = 0; first < members.size(); ++first) {
    for (size_t second = first + 1; second < members.size(); ++second) {
      edges.push_back({first, second, normalizedPoseDistance(
          candidates[members[first]], candidates[members[second]], params)});
    }
  }
  std::sort(edges.begin(), edges.end(), edgeLess);

  DisjointSet mst_set(members.size());
  std::vector<WeightedEdge> mst;
  for (const WeightedEdge& edge : edges) {
    if (mst_set.unite(edge.first, edge.second)) {
      mst.push_back(edge);
      if (mst.size() + 1U == members.size()) {
        break;
      }
    }
  }
  if (mst.size() + 1U != members.size()) {
    for (int member : members) {
      output.push_back({member});
    }
    return;
  }

  size_t longest_index = 0;
  for (size_t index = 1; index < mst.size(); ++index) {
    if (edgeLess(mst[longest_index], mst[index])) {
      longest_index = index;
    }
  }

  DisjointSet split_set(members.size());
  for (size_t index = 0; index < mst.size(); ++index) {
    if (index != longest_index) {
      split_set.unite(mst[index].first, mst[index].second);
    }
  }
  std::map<size_t, std::vector<int>> pieces_by_root;
  for (size_t index = 0; index < members.size(); ++index) {
    pieces_by_root[split_set.find(index)].push_back(members[index]);
  }
  if (pieces_by_root.size() < 2U) {
    for (int member : members) {
      output.push_back({member});
    }
    return;
  }
  for (const auto& entry : pieces_by_root) {
    splitOversizedComponent(candidates, entry.second, params, output);
  }
}

bool isGridCandidate(const GlobalPoseCandidate& candidate) {
  return candidate.source_type == "grid" || candidate.source_type == "retrieval" ||
      candidate.support_group.rfind("grid:", 0) == 0;
}

std::string canonicalNonGridSupport(const GlobalPoseCandidate& candidate,
                                    size_t deterministic_rank) {
  if (candidate.source_type == "ukf" ||
      candidate.source_type == "last_confirmed" ||
      candidate.source_type == "odom" ||
      candidate.support_group == "local_state") {
    return "local_state";
  }
  if (candidate.source_type == "rviz" || candidate.support_group == "rviz") {
    return "rviz";
  }
  if (!candidate.support_group.empty()) {
    return candidate.support_group;
  }
  if (!candidate.source_type.empty()) {
    return "source:" + candidate.source_type;
  }
  return "candidate:" + std::to_string(deterministic_rank);
}

int countSupportGroups(const std::vector<GlobalPoseCandidate>& candidates,
                       const std::vector<int>& members,
                       const GLParams& params) {
  std::vector<int> grid_members;
  std::set<std::string> non_grid_groups;
  for (size_t rank = 0; rank < members.size(); ++rank) {
    const GlobalPoseCandidate& candidate = candidates[members[rank]];
    if (isGridCandidate(candidate)) {
      grid_members.push_back(members[rank]);
    } else {
      non_grid_groups.insert(canonicalNonGridSupport(candidate, rank));
    }
  }

  int grid_group_count = 0;
  if (!grid_members.empty()) {
    DisjointSet grid_set(grid_members.size());
    for (size_t first = 0; first < grid_members.size(); ++first) {
      for (size_t second = first + 1; second < grid_members.size(); ++second) {
        const Eigen::Matrix4d& first_seed = candidates[grid_members[first]].seed_pose;
        const Eigen::Matrix4d& second_seed = candidates[grid_members[second]].seed_pose;
        if (poseXyDistance(first_seed, second_seed) < params.support_seed_xy) {
          grid_set.unite(first, second);
        }
      }
    }
    std::set<size_t> roots;
    for (size_t index = 0; index < grid_members.size(); ++index) {
      roots.insert(grid_set.find(index));
    }
    grid_group_count = static_cast<int>(roots.size());
  }
  return grid_group_count + static_cast<int>(non_grid_groups.size());
}

bool medoidTieBreakLess(const GlobalPoseCandidate& first,
                        const GlobalPoseCandidate& second) {
  if (first.fitness_score != second.fitness_score) {
    return first.fitness_score < second.fitness_score;
  }
  if (first.overlap_ratio != second.overlap_ratio) {
    return first.overlap_ratio > second.overlap_ratio;
  }
  if (first.correction_xy != second.correction_xy) {
    return first.correction_xy < second.correction_xy;
  }
  return candidateDeterministicLess(first, second);
}

GlobalCluster makeCluster(const std::vector<GlobalPoseCandidate>& candidates,
                          std::vector<int> members,
                          const GLParams& params) {
  std::sort(members.begin(), members.end(), [&](int first, int second) {
    return candidateDeterministicLess(candidates[first], candidates[second]);
  });

  GlobalCluster cluster;
  cluster.member_indices = members;
  cluster.size = static_cast<int>(members.size());
  cluster.best_score = std::numeric_limits<double>::infinity();
  double score_sum = 0.0;
  double sin_sum = 0.0;
  double cos_sum = 0.0;
  for (int member : members) {
    const GlobalPoseCandidate& candidate = candidates[member];
    cluster.best_score = std::min(cluster.best_score, candidate.fitness_score);
    score_sum += candidate.fitness_score;
    cluster.mean_pos += candidate.final_pose.block<3, 1>(0, 3);
    const double yaw_rad = yawDeg(candidate.final_pose) * kPi / 180.0;
    sin_sum += std::sin(yaw_rad);
    cos_sum += std::cos(yaw_rad);
  }
  cluster.mean_pos /= static_cast<double>(members.size());
  cluster.mean_score = score_sum / static_cast<double>(members.size());
  cluster.mean_yaw_deg = global_localization_math::wrapAngleDeg(
      std::atan2(sin_sum, cos_sum) * kRadiansToDegrees);
  for (int member : members) {
    const double difference =
        candidates[member].fitness_score - cluster.mean_score;
    cluster.score_variance += difference * difference;
  }
  cluster.score_variance /= static_cast<double>(members.size());
  cluster.support_group_count = countSupportGroups(candidates, members, params);

  int medoid_member = members.front();
  double best_distance_sum = std::numeric_limits<double>::infinity();
  for (int candidate_index : members) {
    double distance_sum = 0.0;
    for (int other_index : members) {
      distance_sum += normalizedPoseDistance(
          candidates[candidate_index], candidates[other_index], params);
    }
    if (distance_sum < best_distance_sum - kComparisonEpsilon ||
        (std::abs(distance_sum - best_distance_sum) <= kComparisonEpsilon &&
         medoidTieBreakLess(
             candidates[candidate_index], candidates[medoid_member]))) {
      best_distance_sum = distance_sum;
      medoid_member = candidate_index;
    }
  }
  cluster.medoid_pose = candidates[medoid_member].final_pose;
  return cluster;
}

bool clusterDecisionLess(const GlobalCluster& first,
                         const GlobalCluster& second) {
  // Quality first: support_group_count is a confirmation gate, not a ranking
  // key. Ranking by support lets a repeated-structure cluster with many
  // correlated seeds permanently outrank a better-scoring correct cluster
  // (measured on the P2 dataset: wrong cluster support=4/score=0.022061 vs
  // correct cluster support=1/score=0.020859).
  const double first_score = std::isfinite(first.best_score)
      ? first.best_score : std::numeric_limits<double>::max();
  const double second_score = std::isfinite(second.best_score)
      ? second.best_score : std::numeric_limits<double>::max();
  if (first_score != second_score) {
    return first_score < second_score;
  }
  if (first.support_group_count != second.support_group_count) {
    return first.support_group_count > second.support_group_count;
  }
  if (first.size != second.size) {
    return first.size > second.size;
  }
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      if (first.medoid_pose(row, column) != second.medoid_pose(row, column)) {
        return first.medoid_pose(row, column) < second.medoid_pose(row, column);
      }
    }
  }
  return false;
}

}  // namespace

bool filterRefinedCandidates(
    const std::vector<GlobalPoseCandidate>& refined_candidates,
    const GLParams& params,
    std::vector<GlobalPoseCandidate>& passed_candidates,
    GLSummaryCounts& summary) {
  passed_candidates.clear();
  summary.refined = static_cast<int>(refined_candidates.size());
  summary.passed = 0;
  summary.rejected_conv = 0;
  summary.rejected_score = 0;
  summary.rejected_xy = 0;
  summary.rejected_z = 0;
  summary.rejected_yaw = 0;
  summary.rejected_rp = 0;
  if (!finitePositive(params.refine_fitness_threshold) ||
      !finitePositive(params.max_correction_xy) ||
      !finitePositive(params.max_correction_z) ||
      !finitePositive(params.max_yaw_delta_deg) ||
      !finitePositive(params.max_rp_delta_deg) ||
      !finitePositive(params.max_abs_roll_deg) ||
      !finitePositive(params.max_abs_pitch_deg)) {
    return false;
  }

  for (const GlobalPoseCandidate& candidate : refined_candidates) {
    if (!candidate.converged || !candidate.final_pose.allFinite()) {
      ++summary.rejected_conv;
    } else if (!std::isfinite(candidate.fitness_score) ||
               candidate.fitness_score < 0.0 ||
               candidate.fitness_score >= params.refine_fitness_threshold) {
      ++summary.rejected_score;
    } else if (!std::isfinite(candidate.correction_xy) ||
               candidate.correction_xy < 0.0 ||
               candidate.correction_xy >= params.max_correction_xy) {
      ++summary.rejected_xy;
    } else if (!std::isfinite(candidate.correction_z) ||
               candidate.correction_z < 0.0 ||
               candidate.correction_z >= params.max_correction_z) {
      ++summary.rejected_z;
    } else if (!std::isfinite(candidate.yaw_delta_deg) ||
               std::abs(candidate.yaw_delta_deg) >= params.max_yaw_delta_deg) {
      ++summary.rejected_yaw;
    } else if (!std::isfinite(candidate.roll_delta_deg) ||
               !std::isfinite(candidate.pitch_delta_deg) ||
               std::abs(candidate.roll_delta_deg) > params.max_rp_delta_deg ||
               std::abs(candidate.pitch_delta_deg) > params.max_rp_delta_deg ||
               std::abs(rollDeg(candidate.final_pose)) > params.max_abs_roll_deg ||
               std::abs(pitchDeg(candidate.final_pose)) > params.max_abs_pitch_deg) {
      ++summary.rejected_rp;
    } else {
      passed_candidates.push_back(candidate);
      ++summary.passed;
    }
  }
  return true;
}

bool clusterCandidates(
    const std::vector<GlobalPoseCandidate>& candidates,
    const GLParams& params,
    std::vector<GlobalCluster>& clusters,
    GLSummaryCounts& summary) {
  clusters.clear();
  summary.cluster_count = 0;
  summary.best_cluster_size = 0;
  summary.best_support_group_count = 0;
  if (!finitePositive(params.cluster_xy) ||
      !finitePositive(params.cluster_yaw_deg) ||
      !finitePositive(params.support_seed_xy)) {
    return false;
  }
  if (candidates.empty()) {
    return true;
  }
  for (const GlobalPoseCandidate& candidate : candidates) {
    if (!candidate.final_pose.allFinite() || !candidate.seed_pose.allFinite() ||
        !std::isfinite(candidate.fitness_score)) {
      return false;
    }
  }

  std::vector<int> stable_indices(candidates.size());
  std::iota(stable_indices.begin(), stable_indices.end(), 0);
  std::sort(stable_indices.begin(), stable_indices.end(), [&](int first, int second) {
    return candidateDeterministicLess(candidates[first], candidates[second]);
  });

  DisjointSet connected_set(candidates.size());
  for (size_t first = 0; first < stable_indices.size(); ++first) {
    for (size_t second = first + 1; second < stable_indices.size(); ++second) {
      if (adjacent(candidates[stable_indices[first]],
                   candidates[stable_indices[second]], params)) {
        connected_set.unite(first, second);
      }
    }
  }
  std::map<size_t, std::vector<int>> components_by_root;
  for (size_t rank = 0; rank < stable_indices.size(); ++rank) {
    components_by_root[connected_set.find(rank)].push_back(stable_indices[rank]);
  }

  std::vector<std::vector<int>> split_components;
  for (const auto& entry : components_by_root) {
    splitOversizedComponent(candidates, entry.second, params, split_components);
  }
  for (std::vector<int>& members : split_components) {
    clusters.push_back(makeCluster(candidates, std::move(members), params));
  }
  std::sort(clusters.begin(), clusters.end(), clusterDecisionLess);
  summary.cluster_count = static_cast<int>(clusters.size());
  return true;
}

GlobalSpatialDecision selectBestCluster(
    const std::vector<GlobalCluster>& clusters,
    const GLParams& params,
    GLSummaryCounts& summary) {
  GlobalSpatialDecision decision;
  summary.cluster_count = static_cast<int>(clusters.size());
  summary.best_cluster_size = 0;
  summary.best_support_group_count = 0;
  if (!finitePositive(params.refine_fitness_threshold) ||
      !finitePositive(params.ambiguous_pose_distance) ||
      !std::isfinite(params.ambiguous_score_ratio) ||
      params.ambiguous_score_ratio <= 1.0 || params.min_support_groups <= 0) {
    return decision;
  }
  if (clusters.empty()) {
    return decision;
  }

  std::vector<int> order(clusters.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int first, int second) {
    return clusterDecisionLess(clusters[first], clusters[second]);
  });
  decision.best_cluster_index = order.front();
  if (order.size() > 1U) {
    decision.second_cluster_index = order[1];
  }

  const GlobalCluster& best = clusters[decision.best_cluster_index];
  summary.best_cluster_size = best.size;
  summary.best_support_group_count = best.support_group_count;
  const bool best_quality_ok =
      std::isfinite(best.best_score) &&
      best.best_score < params.refine_fitness_threshold;

  // Ambiguity must be evaluated before the support gate: when the best
  // cluster lacks independent support but a distant cluster scores nearly as
  // well, downgrading to LowConsensus would hide the fact that two distinct
  // locations are still competing.
  if (best_quality_ok && decision.second_cluster_index >= 0) {
    const GlobalCluster& second = clusters[decision.second_cluster_index];
    double score_ratio = std::numeric_limits<double>::infinity();
    if (best.best_score <= kComparisonEpsilon) {
      if (second.best_score <= kComparisonEpsilon) {
        score_ratio = 1.0;
      }
    } else if (std::isfinite(second.best_score)) {
      score_ratio = second.best_score / best.best_score;
    }
    // Two clusters whose medoids are farther apart than
    // ambiguous_pose_distance describe materially different conclusions;
    // closer pairs are same-site neighbors where either choice stays within
    // the temporal-vote tolerance. The competitor deliberately does not need
    // min_support_groups: a single-support cluster scoring this close is
    // still evidence the scene cannot be disambiguated spatially.
    if (poseXyDistance(best.medoid_pose, second.medoid_pose) >
            params.ambiguous_pose_distance &&
        score_ratio < params.ambiguous_score_ratio) {
      decision.status = GLStatus::Ambiguous;
      return decision;
    }
  }

  if (best.support_group_count < params.min_support_groups ||
      !best_quality_ok) {
    decision.status = GLStatus::LowConsensus;
    return decision;
  }

  decision.status = GLStatus::PendingVote;
  return decision;
}

}  // namespace localization
