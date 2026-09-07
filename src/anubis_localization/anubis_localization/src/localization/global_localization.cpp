#include "localization/global_localization.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/LU>

#include "localization/global_localization_refinement.hpp"
#include "localization/global_localization_gravity.hpp"
#include "localization/global_localization_spatial.hpp"
#include "localization/global_localization_validation.hpp"
#include "localization/level0_distance_field.hpp"
#include "localization/level0_global_search.hpp"
#include "localization/level1_coarse_search.hpp"
#include "localization/retrieval_recall.hpp"

namespace localization {

GlobalLocalization::GlobalLocalization() = default;

GlobalLocalization::~GlobalLocalization() = default;

void GlobalLocalization::init(const Eigen::Matrix4d& initial_pose) {
  initial_pose_ = initial_pose;
}

void GlobalLocalization::setParams(const GLParams& params) {
  const std::vector<std::string> errors = validateGLParams(params);
  if (!errors.empty()) {
    std::ostringstream message;
    message << "Invalid global localization parameters";
    for (const std::string& error : errors) {
      message << "; " << error;
    }
    throw std::invalid_argument(message.str());
  }
  params_ = params;
}

void GlobalLocalization::setBaseFromLidar(
    const Eigen::Matrix4d& T_base_lidar) {
  const Eigen::Matrix3d rotation = T_base_lidar.block<3, 3>(0, 0);
  const Eigen::Matrix3d orthogonality =
      rotation.transpose() * rotation - Eigen::Matrix3d::Identity();
  if (!T_base_lidar.allFinite() ||
      orthogonality.cwiseAbs().maxCoeff() > 1e-6 ||
      std::abs(rotation.determinant() - 1.0) > 1e-6 ||
      !T_base_lidar.row(3).isApprox(
          Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-9)) {
    throw std::invalid_argument("T_base_lidar must be a finite rigid transform");
  }
  base_from_lidar_ = T_base_lidar;
}

void GlobalLocalization::setDescriptorDatabase(
    const std::shared_ptr<const scan_descriptor::DescriptorDatabase>& database) {
  descriptor_database_ = database;
}

bool GlobalLocalization::performGlobalLocalization(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& global_map,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& current_cloud,
    const Eigen::Matrix4d& initial_trans,
    Eigen::Matrix4d& final_pose) {
  (void)global_map;
  (void)current_cloud;
  (void)initial_trans;
  (void)final_pose;

  RCLCPP_WARN_ONCE(
    logger_,
    "Legacy global localization API is disabled: dynamic gravity and temporal "
    "evidence are required before a pose can be confirmed");
  return false;
}

void GlobalLocalization::performGlobalLocalization(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& global_map,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& current_cloud,
    const Eigen::Matrix4d& initial_trans,
    const Eigen::Matrix4d* ukf_pose,
    const Eigen::Matrix4d* odom_pose,
    const GravityAlignmentSample* gravity_sample,
    GlobalLocalizationResult& result,
    GLSummaryCounts* summary) {
  (void)odom_pose;

  result = GlobalLocalizationResult{};
  GLSummaryCounts unused_summary;
  GLSummaryCounts& counts = summary == nullptr ? unused_summary : *summary;
  counts = GLSummaryCounts{};
  if (gravity_sample == nullptr || !gravity_sample->valid) {
    result.anchor_reject_reason = GLAnchorRejectReason::GravityUnavailable;
    accumulateGravityGateSummary(nullptr, GravityGateResult{}, counts);
    return;
  }
  if (!initial_trans.allFinite()) {
    return;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr source(new pcl::PointCloud<pcl::PointXYZI>());
  if (!preprocessCloud(current_cloud, source)) {
    result.anchor_reject_reason = GLAnchorRejectReason::TooFewValidPoints;
    ++counts.anchor_rejected_too_few_points;
    return;
  }

  if (!global_map || global_map->empty()) {
    return;
  }

  const GravityGateResult gravity_gate = evaluateGravitySample(
      gravity_sample->stamp, gravity_sample, params_);
  if (!gravity_gate.accepted) {
    result.anchor_reject_reason = gravity_gate.reject_reason;
    accumulateGravityGateSummary(gravity_sample, gravity_gate, counts);
    return;
  }
  accumulateGravityGateSummary(gravity_sample, gravity_gate, counts);

  std::vector<Level0Hypothesis> top_regions;
  bool retrieval_complete = false;
  if (params_.gl_recall_source == "retrieval" && descriptor_database_) {
    RetrievalRecallParams retrieval_params;
    retrieval_params.topk = params_.retrieval_topk;
    retrieval_params.nms_xy = params_.retrieval_nms_xy;
    retrieval_params.max_distance = params_.retrieval_max_distance;
    retrieval_params.min_points = params_.retrieval_min_points;
    retrieval_params.yaw_half_range_deg = params_.retrieval_yaw_half_range_deg;
    retrieval_params.yaw_step_deg = params_.retrieval_yaw_step_deg;
    retrieval_params.search_radius_xy = params_.retrieval_search_radius_xy;
    retrieval_params.seed_stride_xy = params_.retrieval_seed_stride_xy;
    RetrievalRecallStats retrieval_stats;
    retrieval_complete = runRetrievalRecall(
        *descriptor_database_, current_cloud, gravity_gate.R_level_base,
        base_from_lidar_, -params_.level0_base_ground_offset,
        retrieval_params, top_regions, retrieval_stats);
    counts.retrieval_db_size = retrieval_stats.db_size;
    counts.retrieval_match_count = retrieval_stats.match_count;
    counts.retrieval_best_distance = retrieval_stats.best_distance;
    counts.level0_input_points = retrieval_stats.valid_points;
    counts.level0_valid_points = retrieval_stats.valid_points;
  }
  if (!retrieval_complete) {
    if (params_.gl_recall_source == "retrieval" && !params_.retrieval_fallback_to_grid) {
      return;
    }
    Level0DistanceField distance_field;
    if (!distance_field.build(global_map, params_)) return;
    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_level(
        new pcl::PointCloud<pcl::PointXYZI>());
    Level0ScanPreparationStats preparation_stats;
    if (!distance_field.prepareScan(
            current_cloud, gravity_gate.R_level_base,
            params_.level0_base_ground_offset, params_, scan_level,
            preparation_stats)) {
      result.anchor_reject_reason = GLAnchorRejectReason::TooFewValidPoints;
      ++counts.anchor_rejected_too_few_points;
      return;
    }
    Level0SearchStats level0_stats;
    if (!runLevel0GlobalSearch(
            distance_field, scan_level, params_, top_regions, level0_stats,
            counts) || !level0_stats.complete || top_regions.empty()) return;
  }
  result.level0_candidates.reserve(top_regions.size());
  for (const Level0Hypothesis& region : top_regions) {
    GlobalPoseCandidate candidate;
    if (!composeMapBaseSeed(
            region.T_map_level, gravity_gate.R_level_base,
            candidate.seed_pose)) {
      result.level0_candidates.clear();
      return;
    }
    candidate.final_pose = candidate.seed_pose;
    candidate.coarse_score = region.score.score;
    candidate.level0_mean_truncated_distance =
        region.score.mean_truncated_distance;
    candidate.level0_valid_projection_ratio =
        region.score.valid_projection_ratio;
    candidate.level0_input_points = region.score.input_point_count;
    candidate.level0_valid_points = region.score.valid_projection_points;
    candidate.source_type = retrieval_complete ? "retrieval_region" : "level0_region";
    candidate.grid_cell_id = region.grid_cell_id;
    candidate.yaw_bin = region.yaw_bin;
    candidate.rotation_batch = region.rotation_batch;
    candidate.support_group = (retrieval_complete ? "retrieval:" : "level0:") +
        std::to_string(region.grid_cell_id) + ":" +
        std::to_string(region.yaw_bin);
    candidate.evidence_id = "anchor";
    candidate.evidence_role = "anchor_recall";
    result.level0_candidates.push_back(std::move(candidate));
  }
  accumulateGravityGateSummary(gravity_sample, gravity_gate, counts);

  std::vector<GlobalPoseCandidate> priors;
  auto append_prior = [&priors](const Eigen::Matrix4d& pose,
                                const std::string& source_type) {
    for (const GlobalPoseCandidate& existing : priors) {
      if (existing.seed_pose.isApprox(pose, 1e-6)) {
        return;
      }
    }
    GlobalPoseCandidate prior;
    prior.seed_pose = pose;
    prior.final_pose = pose;
    prior.source_type = source_type;
    prior.support_group = "local_state";
    prior.evidence_id = "anchor";
    prior.evidence_role = "anchor_recall";
    priors.push_back(prior);
  };
  append_prior(initial_trans, "initial_prior");
  if (ukf_pose != nullptr && ukf_pose->allFinite()) {
    append_prior(*ukf_pose, "ukf_prior");
  }

  std::vector<GlobalPoseCandidate> grid_seeds;
  Level1GenerationStats generation_stats;
  bool seeds_generated = false;
  if (retrieval_complete) {
    RetrievalRecallParams retrieval_params;
    retrieval_params.topk = params_.retrieval_topk;
    retrieval_params.nms_xy = params_.retrieval_nms_xy;
    retrieval_params.max_distance = params_.retrieval_max_distance;
    retrieval_params.min_points = params_.retrieval_min_points;
    retrieval_params.yaw_half_range_deg = params_.retrieval_yaw_half_range_deg;
    retrieval_params.yaw_step_deg = params_.retrieval_yaw_step_deg;
    retrieval_params.search_radius_xy = params_.retrieval_search_radius_xy;
    retrieval_params.seed_stride_xy = params_.retrieval_seed_stride_xy;
    seeds_generated = generateRetrievalSeeds(
        top_regions, gravity_gate.R_level_base, retrieval_params,
        static_cast<int>(priors.size()), "anchor", params_, grid_seeds, counts);
  } else {
    seeds_generated = generateLevel1Seeds(
        top_regions, gravity_gate.R_level_base,
        static_cast<int>(priors.size()), "anchor", params_, grid_seeds,
        generation_stats, counts);
  }
  if (!seeds_generated) {
    return;
  }
  std::vector<GlobalPoseCandidate> coarse_candidates;
  coarse_candidates.reserve(priors.size() + grid_seeds.size());
  coarse_candidates.insert(
      coarse_candidates.end(), priors.begin(), priors.end());
  coarse_candidates.insert(
      coarse_candidates.end(), grid_seeds.begin(), grid_seeds.end());
  counts.seed_total = static_cast<int>(coarse_candidates.size());
  result.candidate_count = counts.seed_total;

  Level1CoarseStats coarse_stats;
  if (!runLevel1CoarseAlignment(
          source, global_map, params_, coarse_candidates, coarse_stats)) {
    return;
  }
  result.seed_candidates = coarse_candidates;
  std::vector<GlobalPoseCandidate> refine_candidates;
  if (!selectLevel1RefineCandidates(
          coarse_candidates, params_, refine_candidates, coarse_stats)) {
    return;
  }
  counts.coarse_kept = static_cast<int>(refine_candidates.size());
  result.coarse_kept_count = counts.coarse_kept;
  if (refine_candidates.empty()) {
    return;
  }

  GlobalRefinementStats refinement_stats;
  if (!runLevel1Refinement(
          source, global_map, params_, refine_candidates,
          refinement_stats)) {
    return;
  }
  result.candidates = refine_candidates;
  result.refined_count = static_cast<int>(refine_candidates.size());

  std::vector<GlobalPoseCandidate> passed_candidates;
  if (!filterRefinedCandidates(
          refine_candidates, params_, passed_candidates, counts)) {
    return;
  }
  result.passed_filter_count = static_cast<int>(passed_candidates.size());
  result.passed_candidates = passed_candidates;
  if (passed_candidates.empty()) {
    return;
  }

  std::vector<GlobalCluster> clusters;
  if (!clusterCandidates(passed_candidates, params_, clusters, counts)) {
    return;
  }
  result.cluster_count = static_cast<int>(clusters.size());
  result.clusters = clusters;
  const GlobalSpatialDecision decision = selectBestCluster(
      clusters, params_, counts);
  result.status = decision.status;
  result.best_cluster_index = decision.best_cluster_index;
  result.ambiguous = decision.status == GLStatus::Ambiguous;
  result.best_cluster_size = counts.best_cluster_size;
  if (decision.best_cluster_index >= 0 &&
      decision.best_cluster_index < static_cast<int>(clusters.size())) {
    const GlobalCluster& best = clusters[decision.best_cluster_index];
    result.best_score = best.best_score;
    // [2026-08-14] Publish the winning medoid as final_pose whenever the
    // spatial decision accepted it (PendingVote = geometry + support +
    // discriminative gates all passed). The motion-driven vote used to be the
    // only writer of final_pose; static confirmation now consumes it directly,
    // so leaving it unset would silently break the whole landing chain.
    if (decision.status == GLStatus::PendingVote &&
        best.medoid_pose.allFinite()) {
      result.final_pose = best.medoid_pose;
    }
  }
}

bool GlobalLocalization::preprocessCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& output) const {
  if (!input || input->empty() || params_.refine_leaf_src <= 0.0) {
    return false;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr finite(new pcl::PointCloud<pcl::PointXYZI>(*input));
  finite->is_dense = false;
  std::vector<int> kept_indices;
  pcl::removeNaNFromPointCloud(*finite, *finite, kept_indices);
  if (finite->empty()) {
    return false;
  }

  if (!output) {
    output.reset(new pcl::PointCloud<pcl::PointXYZI>());
  }
  output->clear();

  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
  voxel_filter.setInputCloud(finite);
  const float leaf_size = static_cast<float>(params_.refine_leaf_src);
  voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_filter.filter(*output);
  return !output->empty();
}

void GlobalLocalization::runSingleSeedIcp(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& target,
    GlobalPoseCandidate& candidate) const {
  runDoubleIcpRefinement(
      source, target, candidate.seed_pose, params_, candidate);
}

}  // namespace localization
