#include "localization/global_localization_integration.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

#include <Eigen/LU>

#include "localization/global_localization_validation.hpp"

namespace localization {
namespace {

const char* anchorRejectReasonName(GLAnchorRejectReason reason) {
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

bool usableConfirmedResult(const GlobalLocalizationResult& result) {
  if (result.status != GLStatus::Confirmed || !result.success ||
      result.budget_exceeded || !result.final_pose.allFinite()) {
    return false;
  }

  constexpr double kHomogeneousTolerance = 1e-6;
  constexpr double kRotationTolerance = 1e-3;
  const Eigen::Vector4d expected_bottom_row(0.0, 0.0, 0.0, 1.0);
  if (!result.final_pose.row(3).transpose().isApprox(
          expected_bottom_row, kHomogeneousTolerance)) {
    return false;
  }

  const Eigen::Matrix3d rotation = result.final_pose.block<3, 3>(0, 0);
  return (rotation.transpose() * rotation).isApprox(
             Eigen::Matrix3d::Identity(), kRotationTolerance) &&
      std::abs(rotation.determinant() - 1.0) <= kRotationTolerance;
}

}  // namespace

bool automaticApprovalContractSatisfied(const GLParams& params) {
  return params.auto_confirm_enabled && validateGLParams(params).empty() &&
      validateAutoConfirmContract(params).empty();
}

const char* glStatusName(GLStatus status) {
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

const char* glEpisodePhaseName(GLEpisodePhase phase) {
  switch (phase) {
    case GLEpisodePhase::Idle:
      return "Idle";
    case GLEpisodePhase::RecallAnchor:
      return "RecallAnchor";
    case GLEpisodePhase::AwaitingPrior:
      return "AwaitingPrior";
  }
  return "Unknown";
}

GLIntegrationAction decideGLIntegrationAction(
    const GlobalLocalizationResult& result,
    const GLParams& params,
    bool episode_active,
  bool episode_timed_out) {
  GLIntegrationAction action;

  if (result.status == GLStatus::Confirmed) {
    if (!usableConfirmedResult(result)) {
      action.invalid_confirmed_result = true;
    } else if (!episode_active || episode_timed_out) {
      action.stale_confirmed_result = true;
    } else if (automaticApprovalContractSatisfied(params)) {
      action.accept_pose_for_init = true;
      action.freeze_pose_estimator = false;
      action.run_init_verification = true;
      return action;
    } else {
      action.decision_shadow = true;
    }
  }

  action.keep_episode_active = episode_active && !episode_timed_out;
  action.request_initialpose = episode_timed_out;
  return action;
}

GLOutputGate decideGLOutputGate(bool episode_active,
                                bool init_verification_complete) {
  GLOutputGate gate;
  gate.localization_valid = !episode_active && init_verification_complete;
  gate.publish_localization_odom = gate.localization_valid;
  gate.publish_map_tf = gate.localization_valid;
  return gate;
}

GLOutputGate decideGLOutputGate(const GLEpisodeState& episode,
                                bool init_verification_complete) {
  // AwaitingPrior is terminal for the current attempt. It must remain
  // unlocalized even if a stale init-verification flag was left set.
  const bool episode_blocks_output = episode.active ||
      episode.phase == GLEpisodePhase::AwaitingPrior;
  return decideGLOutputGate(episode_blocks_output, init_verification_complete);
}

bool shouldDropScanAfterGL(const rclcpp::Time& now,
                           const rclcpp::Time& scan_stamp,
                           const GLParams& params) {
  if (now.get_clock_type() != scan_stamp.get_clock_type() ||
      now.nanoseconds() < 0 || scan_stamp.nanoseconds() < 0 ||
      !std::isfinite(params.max_scan_age_after_gl_s) ||
      params.max_scan_age_after_gl_s <= 0.0 || now < scan_stamp) {
    return true;
  }
  return (now - scan_stamp).seconds() > params.max_scan_age_after_gl_s;
}

std::string formatGLSummary(const GlobalLocalizationResult& result,
                            const GLSummaryCounts& summary,
                            const GLEpisodeState& episode,
                            double cost_ms) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(3)
         << "[GL] episode=" << episode.episode_id
         << " phase=" << glEpisodePhaseName(episode.phase)
         << " status=" << glStatusName(result.status)
         << " seed=" << summary.seed_total
         << " coarse_kept=" << summary.coarse_kept
         << " refined=" << summary.refined
         << " passed=" << summary.passed
         << " clusters=" << summary.cluster_count
         << " best_size=" << summary.best_cluster_size
         << " best_support=" << summary.best_support_group_count
         << " best_score=" << result.best_score
         << " reject_reason=" << anchorRejectReasonName(result.anchor_reject_reason)
         << " bank=" << summary.candidate_bank_size
         << " rejected=(conv:" << summary.rejected_conv
         << ",score:" << summary.rejected_score
         << ",xy:" << summary.rejected_xy
         << ",z:" << summary.rejected_z
         << ",yaw:" << summary.rejected_yaw
         << ",rp:" << summary.rejected_rp << ')'
         << " confirm=(probes:" << summary.static_confirm_probe_count
         << ",agree:" << summary.static_confirm_agree_count << ')'
         << " stale_dropped=" << summary.stale_scans_dropped
         << " waiting_low_dynamic=" << (result.waiting_low_dynamic ? 1 : 0)
         << " anchor_invalid=" << (result.anchor_invalid ? 1 : 0)
         << " anchor_motion_exceeded="
         << (result.anchor_motion_exceeded ? 1 : 0)
         << " budget=" << (result.budget_exceeded ? 1 : 0)
         << " processing_ms=" << summary.processing_ms
         << " await_ms=" << summary.await_evidence_ms
         << " cost_ms=" << cost_ms;

  if (usableConfirmedResult(result)) {
    output << " final=(" << result.final_pose(0, 3) << ','
           << result.final_pose(1, 3) << ',' << result.final_pose(2, 3) << ')';
  }
  return output.str();
}

}  // namespace localization
