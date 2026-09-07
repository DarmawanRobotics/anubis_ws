#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include "localization/global_localization_integration.hpp"

namespace localization {
namespace {

constexpr rcl_clock_type_t kClock = RCL_ROS_TIME;

rclcpp::Time timeNs(int64_t nanoseconds) {
  return rclcpp::Time(nanoseconds, kClock);
}

GlobalLocalizationResult confirmedResult() {
  GlobalLocalizationResult result;
  result.status = GLStatus::Confirmed;
  result.success = true;
  result.final_pose = Eigen::Matrix4d::Identity();
  result.final_pose(0, 3) = 1.25;
  result.final_pose(1, 3) = -2.5;
  return result;
}

GLParams approvedParams() {
  GLParams params;
  params.auto_confirm_enabled = true;
  params.flat_single_level_map = true;
  params.level0_base_ground_offset = 0.45;
  // Syntax-only fixture for integration behavior; it is not a formal M2b approval.
  params.m2b_approval_id =
      "m2b:unit-fixture:map-a1b2c3d4:params-e5f6a7b8:manifest-01020304";
  return params;
}

TEST(GlobalLocalizationIntegration, EveryNonConfirmedStatusBlocksInit) {
  const std::array<GLStatus, 4> statuses{
      GLStatus::PendingVote,
      GLStatus::Ambiguous,
      GLStatus::NoCandidate,
      GLStatus::LowConsensus,
  };
  const GLParams params = approvedParams();
  for (const GLStatus status : statuses) {
    GlobalLocalizationResult result;
    result.status = status;
    const GLIntegrationAction action = decideGLIntegrationAction(
        result, params, true, false);
    EXPECT_FALSE(action.accept_pose_for_init) << glStatusName(status);
    EXPECT_FALSE(action.run_init_verification) << glStatusName(status);
    EXPECT_TRUE(action.freeze_pose_estimator) << glStatusName(status);
    EXPECT_TRUE(action.keep_episode_active) << glStatusName(status);
  }
}

TEST(GlobalLocalizationIntegration, ApprovedConfirmedPoseOnlyEntersInitVerification) {
  const GLIntegrationAction action = decideGLIntegrationAction(
      confirmedResult(), approvedParams(), true, false);
  EXPECT_TRUE(action.accept_pose_for_init);
  EXPECT_TRUE(action.run_init_verification);
  EXPECT_FALSE(action.freeze_pose_estimator);
  EXPECT_FALSE(action.keep_episode_active);
  EXPECT_FALSE(action.decision_shadow);

  const GLOutputGate gate = decideGLOutputGate(false, false);
  EXPECT_FALSE(gate.localization_valid);
  EXPECT_FALSE(gate.publish_localization_odom);
  EXPECT_FALSE(gate.publish_map_tf);
}

TEST(GlobalLocalizationIntegration, ConfirmationWithoutApprovalRemainsShadow) {
  const GLIntegrationAction action = decideGLIntegrationAction(
      confirmedResult(), GLParams{}, true, false);
  EXPECT_FALSE(action.accept_pose_for_init);
  EXPECT_FALSE(action.run_init_verification);
  EXPECT_TRUE(action.freeze_pose_estimator);
  EXPECT_TRUE(action.keep_episode_active);
  EXPECT_TRUE(action.decision_shadow);
}

TEST(GlobalLocalizationIntegration, InvalidApprovalKeepsEpisodeInDecisionShadow) {
  GLParams params = approvedParams();
  params.m2b_approval_id = "DEBUG NOT APPROVED";

  const GLIntegrationAction action = decideGLIntegrationAction(
      confirmedResult(), params, true, false);
  EXPECT_FALSE(automaticApprovalContractSatisfied(params));
  EXPECT_TRUE(action.decision_shadow);
  EXPECT_FALSE(action.accept_pose_for_init);
  EXPECT_FALSE(action.run_init_verification);
  EXPECT_TRUE(action.freeze_pose_estimator);
  EXPECT_TRUE(action.keep_episode_active);
}

TEST(GlobalLocalizationIntegration, TimedOutOrInactiveConfirmationIsDiscarded) {
  GLIntegrationAction action = decideGLIntegrationAction(
      confirmedResult(), approvedParams(), true, true);
  EXPECT_TRUE(action.stale_confirmed_result);
  EXPECT_TRUE(action.request_initialpose);
  EXPECT_FALSE(action.accept_pose_for_init);
  EXPECT_TRUE(action.freeze_pose_estimator);

  action = decideGLIntegrationAction(
      confirmedResult(), approvedParams(), false, false);
  EXPECT_TRUE(action.stale_confirmed_result);
  EXPECT_FALSE(action.accept_pose_for_init);
}

TEST(GlobalLocalizationIntegration, MalformedConfirmedResultFailsClosed) {
  GlobalLocalizationResult result = confirmedResult();
  result.success = false;
  GLIntegrationAction action = decideGLIntegrationAction(
      result, approvedParams(), true, false);
  EXPECT_TRUE(action.invalid_confirmed_result);
  EXPECT_FALSE(action.accept_pose_for_init);
  EXPECT_TRUE(action.freeze_pose_estimator);

  result.success = true;
  result.final_pose(0, 0) = std::numeric_limits<double>::quiet_NaN();
  action = decideGLIntegrationAction(result, approvedParams(), true, false);
  EXPECT_TRUE(action.invalid_confirmed_result);
  EXPECT_FALSE(action.accept_pose_for_init);

  result = confirmedResult();
  result.final_pose(0, 0) = 2.0;
  action = decideGLIntegrationAction(result, approvedParams(), true, false);
  EXPECT_TRUE(action.invalid_confirmed_result);
  EXPECT_FALSE(action.accept_pose_for_init);

  result = confirmedResult();
  result.final_pose(3, 0) = 0.1;
  action = decideGLIntegrationAction(result, approvedParams(), true, false);
  EXPECT_TRUE(action.invalid_confirmed_result);
  EXPECT_FALSE(action.accept_pose_for_init);
}

TEST(GlobalLocalizationIntegration, BudgetExceededConfirmationFailsClosed) {
  GlobalLocalizationResult result = confirmedResult();
  result.budget_exceeded = true;

  const GLIntegrationAction action = decideGLIntegrationAction(
      result, approvedParams(), true, false);
  EXPECT_TRUE(action.invalid_confirmed_result);
  EXPECT_FALSE(action.accept_pose_for_init);
  EXPECT_FALSE(action.run_init_verification);
  EXPECT_TRUE(action.freeze_pose_estimator);
}

TEST(GlobalLocalizationIntegration, OutputRequiresInactiveEpisodeAndVerifiedInit) {
  GLOutputGate gate = decideGLOutputGate(true, true);
  EXPECT_FALSE(gate.localization_valid);
  EXPECT_FALSE(gate.publish_localization_odom);
  EXPECT_FALSE(gate.publish_map_tf);

  gate = decideGLOutputGate(false, false);
  EXPECT_FALSE(gate.localization_valid);

  gate = decideGLOutputGate(false, true);
  EXPECT_TRUE(gate.localization_valid);
  EXPECT_TRUE(gate.publish_localization_odom);
  EXPECT_TRUE(gate.publish_map_tf);
}

TEST(GlobalLocalizationIntegration, AwaitingPriorAlwaysBlocksOutput) {
  GLEpisodeState episode;
  episode.phase = GLEpisodePhase::AwaitingPrior;
  episode.active = false;
  const GLOutputGate gate = decideGLOutputGate(episode, true);
  EXPECT_FALSE(gate.localization_valid);
  EXPECT_FALSE(gate.publish_localization_odom);
  EXPECT_FALSE(gate.publish_map_tf);
}

TEST(GlobalLocalizationIntegration, EpisodeTimeoutRequestsExternalPrior) {
  GlobalLocalizationResult result;
  result.status = GLStatus::Ambiguous;
  const GLIntegrationAction action = decideGLIntegrationAction(
      result, GLParams{}, true, true);
  EXPECT_TRUE(action.freeze_pose_estimator);
  EXPECT_FALSE(action.keep_episode_active);
  EXPECT_TRUE(action.request_initialpose);
  EXPECT_FALSE(action.accept_pose_for_init);
}

TEST(GlobalLocalizationIntegration, StaleScanGateFailsClosed) {
  GLParams params;
  params.max_scan_age_after_gl_s = 0.20;
  const rclcpp::Time scan = timeNs(1'000'000'000LL);
  EXPECT_FALSE(shouldDropScanAfterGL(
      scan + rclcpp::Duration::from_seconds(0.20), scan, params));
  EXPECT_TRUE(shouldDropScanAfterGL(
      scan + rclcpp::Duration::from_seconds(0.201), scan, params));
  EXPECT_TRUE(shouldDropScanAfterGL(
      scan - rclcpp::Duration::from_seconds(0.01), scan, params));
  EXPECT_TRUE(shouldDropScanAfterGL(
      rclcpp::Time(int64_t{1'100'000'000LL}, RCL_SYSTEM_TIME), scan, params));
}


TEST(GlobalLocalizationIntegration, SummaryUsesStableStatusAndCounterNames) {
  GlobalLocalizationResult result = confirmedResult();
  result.best_score = 0.11;
  GLSummaryCounts summary;
  summary.seed_total = 80;
  summary.coarse_kept = 12;
  summary.refined = 12;
  summary.passed = 7;
  summary.cluster_count = 3;
  summary.best_cluster_size = 4;
  summary.best_support_group_count = 2;
  summary.rejected_z = 5;
  summary.static_confirm_probe_count = 3;
  summary.static_confirm_agree_count = 3;
  GLEpisodeState episode;
  episode.episode_id = 42;
  episode.phase = GLEpisodePhase::RecallAnchor;

  const std::string line = formatGLSummary(result, summary, episode, 812.0);
  EXPECT_NE(line.find("[GL] episode=42"), std::string::npos);
  EXPECT_NE(line.find("phase=RecallAnchor"), std::string::npos);
  EXPECT_NE(line.find("confirm=(probes:3,agree:3)"), std::string::npos);
  EXPECT_NE(line.find("status=Confirmed"), std::string::npos);
  EXPECT_NE(line.find("seed=80 coarse_kept=12 refined=12 passed=7"),
            std::string::npos);
  EXPECT_NE(line.find("rejected=(conv:0,score:0,xy:0,z:5,yaw:0,rp:0)"),
            std::string::npos);
  EXPECT_NE(line.find("final=(1.250,-2.500,0.000)"), std::string::npos);
}

}  // namespace
}  // namespace localization
