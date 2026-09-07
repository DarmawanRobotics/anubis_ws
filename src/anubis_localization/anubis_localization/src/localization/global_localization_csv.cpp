#include "localization/global_localization_csv.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

namespace localization {
namespace {

constexpr double kRadiansToDegrees =
    180.0 / 3.14159265358979323846;

std::string csvEscape(const std::string& value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos) {
    return value;
  }
  std::string escaped;
  escaped.reserve(value.size() + 2U);
  escaped.push_back('"');
  for (char character : value) {
    if (character == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

std::string finiteNumber(double value) {
  if (!std::isfinite(value)) {
    return {};
  }
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

std::string integerNumber(int64_t value) {
  return std::to_string(value);
}

double poseYawDeg(const Eigen::Matrix4d& pose) {
  return std::atan2(pose(1, 0), pose(0, 0)) * kRadiansToDegrees;
}

const char* statusName(GLStatus status) {
  switch (status) {
    case GLStatus::Confirmed:
      return "Confirmed";
    case GLStatus::PendingVote:
      return "PendingVote";
    case GLStatus::Ambiguous:
      return "Ambiguous";
    case GLStatus::NoCandidate:
      return "NoCandidate";
    case GLStatus::LowConsensus:
      return "LowConsensus";
  }
  return "Unknown";
}

const char* rejectReasonName(GLAnchorRejectReason reason) {
  switch (reason) {
    case GLAnchorRejectReason::None:
      return "None";
    case GLAnchorRejectReason::GravityUnavailable:
      return "GravityUnavailable";
    case GLAnchorRejectReason::GravityStale:
      return "GravityStale";
    case GLAnchorRejectReason::GravityUncertain:
      return "GravityUncertain";
    case GLAnchorRejectReason::TiltExceeded:
      return "TiltExceeded";
    case GLAnchorRejectReason::AngularVelocityExceeded:
      return "AngularVelocityExceeded";
    case GLAnchorRejectReason::AccelerationResidualExceeded:
      return "AccelerationResidualExceeded";
    case GLAnchorRejectReason::MotionWindowExceeded:
      return "MotionWindowExceeded";
    case GLAnchorRejectReason::TooFewValidPoints:
      return "TooFewValidPoints";
  }
  return "Unknown";
}

std::string renderRow(const std::vector<std::string>& fields) {
  std::ostringstream output;
  for (size_t index = 0; index < fields.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << csvEscape(fields[index]);
  }
  output << '\n';
  return output.str();
}

void appendSummaryFields(std::vector<std::string>& fields,
                         const GlobalLocalizationResult& result,
                         const GLSummaryCounts& summary) {
  fields.push_back(integerNumber(summary.seed_total));
  fields.push_back(integerNumber(summary.coarse_kept));
  fields.push_back(integerNumber(summary.refined));
  fields.push_back(integerNumber(summary.passed));
  fields.push_back(integerNumber(summary.cluster_count));
  fields.push_back(integerNumber(summary.best_cluster_size));
  fields.push_back(integerNumber(summary.best_support_group_count));
  fields.push_back(integerNumber(summary.level0_input_points));
  fields.push_back(integerNumber(summary.level0_valid_points));
  fields.push_back(integerNumber(summary.level0_boundary_unknown_points));
  fields.push_back(integerNumber(summary.level0_invalid_projection_points));
  fields.emplace_back(rejectReasonName(result.anchor_reject_reason));
  fields.push_back(finiteNumber(summary.anchor_gravity_age_ms));
  fields.push_back(finiteNumber(summary.anchor_gravity_clock_residual_ms));
  fields.push_back(finiteNumber(summary.anchor_gravity_uncertainty_deg));
  fields.push_back(finiteNumber(summary.anchor_roll_deg));
  fields.push_back(finiteNumber(summary.anchor_pitch_deg));
  fields.push_back(finiteNumber(summary.anchor_angular_velocity_rps));
  fields.push_back(finiteNumber(summary.anchor_accel_norm_residual_mps2));
  fields.push_back(integerNumber(summary.retrieval_db_size));
  fields.push_back(integerNumber(summary.retrieval_match_count));
  fields.push_back(finiteNumber(summary.retrieval_best_distance));
  fields.push_back(result.budget_exceeded ? "1" : "0");
}

std::vector<std::string> baseFields(const GLCandidateCsvContext& context,
                                    const std::string& stage,
                                    int rank,
                                    const std::string& status,
                                    const Eigen::Matrix4d* evaluated_pose) {
  std::vector<std::string> fields;
  fields.reserve(68U);
  fields.push_back(context.episode_id);
  fields.push_back(stage);
  fields.push_back(integerNumber(rank));
  fields.emplace_back();  // region: resolved offline from independent polygons
  if (evaluated_pose != nullptr && evaluated_pose->allFinite()) {
    fields.push_back(finiteNumber((*evaluated_pose)(0, 3)));
    fields.push_back(finiteNumber((*evaluated_pose)(1, 3)));
    fields.push_back(finiteNumber(poseYawDeg(*evaluated_pose)));
  } else {
    fields.resize(fields.size() + 3U);
  }
  fields.push_back(status);
  fields.push_back(integerNumber(context.scan_stamp_ns));
  fields.push_back(finiteNumber(context.processing_ms));
  return fields;
}

std::string candidateRow(const GLCandidateCsvContext& context,
                         const std::string& stage,
                         int rank,
                         const GlobalPoseCandidate& candidate,
                         const Eigen::Matrix4d& evaluated_pose,
                         const GlobalLocalizationResult& result,
                         const GLSummaryCounts& summary,
                         const std::string& row_status,
                         int cluster_id,
                         const GlobalCluster* cluster) {
  std::vector<std::string> fields =
      baseFields(context, stage, rank, row_status, &evaluated_pose);
  fields.push_back(candidate.source_type);
  fields.push_back(integerNumber(candidate.grid_cell_id));
  fields.push_back(integerNumber(candidate.yaw_bin));
  fields.push_back(integerNumber(candidate.rotation_batch));
  fields.push_back(finiteNumber(candidate.seed_pose(0, 3)));
  fields.push_back(finiteNumber(candidate.seed_pose(1, 3)));
  fields.push_back(finiteNumber(candidate.seed_pose(2, 3)));
  fields.push_back(finiteNumber(poseYawDeg(candidate.seed_pose)));
  fields.push_back(finiteNumber(candidate.final_pose(0, 3)));
  fields.push_back(finiteNumber(candidate.final_pose(1, 3)));
  fields.push_back(finiteNumber(candidate.final_pose(2, 3)));
  fields.push_back(finiteNumber(poseYawDeg(candidate.final_pose)));
  fields.push_back(finiteNumber(candidate.fitness_score));
  fields.push_back(finiteNumber(candidate.coarse_score));
  fields.push_back(finiteNumber(candidate.level0_mean_truncated_distance));
  fields.push_back(finiteNumber(candidate.level0_valid_projection_ratio));
  fields.push_back(integerNumber(candidate.level0_input_points));
  fields.push_back(integerNumber(candidate.level0_valid_points));
  fields.push_back(finiteNumber(candidate.overlap_ratio));
  fields.push_back(integerNumber(candidate.inlier_count));
  fields.push_back(finiteNumber(candidate.p90_residual));
  fields.push_back(finiteNumber(candidate.correction_xy));
  fields.push_back(finiteNumber(candidate.correction_z));
  fields.push_back(finiteNumber(candidate.yaw_delta_deg));
  fields.push_back(finiteNumber(candidate.roll_delta_deg));
  fields.push_back(finiteNumber(candidate.pitch_delta_deg));
  fields.push_back(candidate.support_group);
  fields.push_back(candidate.evidence_id);
  fields.push_back(candidate.evidence_role);
  if (cluster != nullptr && cluster_id >= 0) {
    fields.push_back(integerNumber(cluster_id));
    fields.push_back(integerNumber(cluster->size));
    fields.push_back(integerNumber(cluster->support_group_count));
    fields.push_back(finiteNumber(cluster->best_score));
    fields.push_back(finiteNumber(cluster->mean_score));
    fields.push_back(finiteNumber(cluster->score_variance));
  } else {
    fields.resize(fields.size() + 6U);
  }
  appendSummaryFields(fields, result, summary);
  return renderRow(fields);
}

int findPassedCandidateIndex(const GlobalPoseCandidate& candidate,
                             const GlobalLocalizationResult& result) {
  for (size_t index = 0; index < result.passed_candidates.size(); ++index) {
    const GlobalPoseCandidate& passed = result.passed_candidates[index];
    if (candidate.source_type == passed.source_type &&
        candidate.support_group == passed.support_group &&
        candidate.seed_pose.isApprox(passed.seed_pose, 1e-9) &&
        candidate.final_pose.isApprox(passed.final_pose, 1e-9)) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

std::string clusterRow(const GLCandidateCsvContext& context,
                       int rank,
                       const GlobalCluster& cluster,
                       const GlobalLocalizationResult& result,
                       const GLSummaryCounts& summary) {
  std::vector<std::string> fields =
      baseFields(context, "cluster", rank, {}, &cluster.medoid_pose);
  fields.emplace_back("cluster_medoid");
  fields.resize(fields.size() + 28U);  // candidate-only fields after source_type
  fields.push_back(integerNumber(rank - 1));
  fields.push_back(integerNumber(cluster.size));
  fields.push_back(integerNumber(cluster.support_group_count));
  fields.push_back(finiteNumber(cluster.best_score));
  fields.push_back(finiteNumber(cluster.mean_score));
  fields.push_back(finiteNumber(cluster.score_variance));
  appendSummaryFields(fields, result, summary);
  return renderRow(fields);
}

std::string decisionRow(const GLCandidateCsvContext& context,
                        const GlobalLocalizationResult& result,
                        const GLSummaryCounts& summary) {
  const Eigen::Matrix4d* pose = nullptr;
  if (result.status == GLStatus::Confirmed && result.success &&
      result.final_pose.allFinite()) {
    pose = &result.final_pose;
  }
  std::vector<std::string> fields = baseFields(
      context, "decision", 1, statusName(result.status), pose);
  fields.emplace_back("decision");
  fields.resize(fields.size() + 34U);  // remaining candidate and cluster fields
  appendSummaryFields(fields, result, summary);
  return renderRow(fields);
}

}  // namespace

GLCandidateCsvWriter::GLCandidateCsvWriter(std::filesystem::path output_path)
    : output_path_(std::move(output_path)) {}

bool GLCandidateCsvWriter::appendProbe(
    const GLCandidateCsvContext& context,
    const GlobalLocalizationResult& result,
    const GLSummaryCounts& summary,
    bool include_decision,
    std::string& error) {
  std::lock_guard<std::mutex> lock(mutex_);
  error.clear();
  if (context.episode_id.empty()) {
    error = "candidate CSV episode_id is empty";
    return false;
  }
  if (context.scan_stamp_ns <= 0) {
    error = "candidate CSV scan_stamp_ns must be > 0";
    return false;
  }

  const bool write_probe =
      probe_episodes_.find(context.episode_id) == probe_episodes_.end();
  const bool write_decision = include_decision &&
      decision_episodes_.find(context.episode_id) == decision_episodes_.end();
  if (!write_probe && !write_decision) {
    return true;
  }

  std::string payload;
  if (write_probe) {
    std::vector<int> cluster_by_passed(
        result.passed_candidates.size(), -1);
    for (size_t cluster_index = 0;
         cluster_index < result.clusters.size(); ++cluster_index) {
      for (int member_index : result.clusters[cluster_index].member_indices) {
        if (member_index >= 0 &&
            member_index < static_cast<int>(cluster_by_passed.size())) {
          cluster_by_passed[member_index] = static_cast<int>(cluster_index);
        }
      }
    }
    int rank = 1;
    for (const GlobalPoseCandidate& candidate : result.seed_candidates) {
      if (candidate.seed_pose.allFinite()) {
        payload += candidateRow(
            context, "seed", rank++, candidate, candidate.seed_pose,
            result, summary, {}, -1, nullptr);
      }
    }
    rank = 1;
    for (const GlobalPoseCandidate& candidate : result.level0_candidates) {
      if (candidate.final_pose.allFinite()) {
        payload += candidateRow(
            context, "coarse", rank++, candidate, candidate.final_pose,
            result, summary, {}, -1, nullptr);
      }
    }
    rank = 1;
    for (const GlobalPoseCandidate& candidate : result.candidates) {
      if (candidate.final_pose.allFinite()) {
        const int passed_index = findPassedCandidateIndex(candidate, result);
        const int cluster_id = passed_index >= 0 ?
            cluster_by_passed[passed_index] : -1;
        const GlobalCluster* cluster = cluster_id >= 0 ?
            &result.clusters[cluster_id] : nullptr;
        payload += candidateRow(
            context, "refined", rank++, candidate, candidate.final_pose,
            result, summary, passed_index >= 0 ? "Passed" : "Rejected",
            cluster_id, cluster);
      }
    }
    rank = 1;
    for (const GlobalCluster& cluster : result.clusters) {
      if (cluster.medoid_pose.allFinite()) {
        payload += clusterRow(context, rank++, cluster, result, summary);
      }
    }
  }
  if (write_decision) {
    if (result.status == GLStatus::Confirmed &&
        (!result.success || !result.final_pose.allFinite())) {
      error = "Confirmed candidate CSV decision has no finite usable pose";
      return false;
    }
    payload += decisionRow(context, result, summary);
  }

  if (!payload.empty() && !appendPayloadLocked(payload, error)) {
    return false;
  }
  if (write_probe) {
    probe_episodes_.insert(context.episode_id);
  }
  if (write_decision) {
    decision_episodes_.insert(context.episode_id);
  }
  return true;
}

bool GLCandidateCsvWriter::appendDecision(
    const GLCandidateCsvContext& context,
    const GlobalLocalizationResult& result,
    const GLSummaryCounts& summary,
    std::string& error) {
  std::lock_guard<std::mutex> lock(mutex_);
  error.clear();
  if (context.episode_id.empty()) {
    error = "candidate CSV episode_id is empty";
    return false;
  }
  if (context.scan_stamp_ns <= 0) {
    error = "candidate CSV scan_stamp_ns must be > 0";
    return false;
  }
  if (decision_episodes_.find(context.episode_id) != decision_episodes_.end()) {
    return true;
  }
  if (result.status == GLStatus::Confirmed &&
      (!result.success || !result.final_pose.allFinite())) {
    error = "Confirmed candidate CSV decision has no finite usable pose";
    return false;
  }
  if (!appendPayloadLocked(decisionRow(context, result, summary), error)) {
    return false;
  }
  decision_episodes_.insert(context.episode_id);
  return true;
}

const std::filesystem::path& GLCandidateCsvWriter::outputPath() const {
  return output_path_;
}

const char* GLCandidateCsvWriter::header() {
  return "episode_id,stage,rank,region,x,y,yaw_deg,status,scan_stamp_ns,processing_ms,source_type,grid_cell_id,yaw_bin,rotation_batch,seed_x,seed_y,seed_z,seed_yaw_deg,final_x,final_y,final_z,final_yaw_deg,fitness_score,coarse_score,level0_mean_truncated_distance,level0_valid_projection_ratio,level0_input_points,level0_valid_points,overlap_ratio,inlier_count,p90_residual,correction_xy,correction_z,yaw_delta_deg,roll_delta_deg,pitch_delta_deg,support_group,evidence_id,evidence_role,cluster_id,cluster_size,cluster_support_group_count,cluster_best_score,cluster_mean_score,cluster_score_variance,seed_total,coarse_kept,refined,passed,cluster_count,best_cluster_size,best_support_group_count,level0_input_points_total,level0_valid_points_total,level0_boundary_unknown_points,level0_invalid_projection_points,anchor_reject_reason,anchor_gravity_age_ms,anchor_gravity_clock_residual_ms,anchor_gravity_uncertainty_deg,anchor_roll_deg,anchor_pitch_deg,anchor_angular_velocity_rps,anchor_accel_norm_residual_mps2,retrieval_db_size,retrieval_match_count,retrieval_best_distance,budget_exceeded";
}

bool GLCandidateCsvWriter::appendPayloadLocked(
    const std::string& payload, std::string& error) {
  if (output_path_.empty()) {
    error = "candidate CSV output path is empty";
    return false;
  }

  std::error_code filesystem_error;
  const std::filesystem::path parent = output_path_.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, filesystem_error);
    if (filesystem_error) {
      error = "failed to create candidate CSV directory " + parent.string() +
          ": " + filesystem_error.message();
      return false;
    }
  }

  bool write_header = true;
  const bool exists = std::filesystem::exists(output_path_, filesystem_error);
  if (filesystem_error) {
    error = "failed to inspect candidate CSV " + output_path_.string() +
        ": " + filesystem_error.message();
    return false;
  }
  if (exists) {
    const uintmax_t size = std::filesystem::file_size(
        output_path_, filesystem_error);
    if (filesystem_error) {
      error = "failed to inspect candidate CSV size " + output_path_.string() +
          ": " + filesystem_error.message();
      return false;
    }
    if (size > 0U) {
      std::ifstream input(output_path_);
      std::string existing_header;
      if (!input || !std::getline(input, existing_header)) {
        error = "failed to read candidate CSV header " + output_path_.string();
        return false;
      }
      if (!existing_header.empty() && existing_header.back() == '\r') {
        existing_header.pop_back();
      }
      if (existing_header != header()) {
        error = "candidate CSV header mismatch: " + output_path_.string();
        return false;
      }
      write_header = false;
    }
  }

  std::ofstream output(output_path_, std::ios::out | std::ios::app);
  if (!output) {
    error = "failed to open candidate CSV for append: " + output_path_.string();
    return false;
  }
  if (write_header) {
    output << header() << '\n';
  }
  output << payload;
  output.flush();
  if (!output) {
    error = "failed to write candidate CSV: " + output_path_.string();
    return false;
  }
  return true;
}

}  // namespace localization
