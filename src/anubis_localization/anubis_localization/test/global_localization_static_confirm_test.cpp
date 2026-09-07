#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "localization/global_localization_static_confirm.hpp"

namespace {

using localization::GLStatus;
using localization::GLSummaryCounts;
using localization::GlobalLocalizationResult;
using localization::StaticConfirmOutcome;
using localization::StaticConfirmParams;
using localization::StaticConfirmState;

constexpr double kPi = 3.14159265358979323846;

rclcpp::Time stampAt(double seconds) {
  return rclcpp::Time(static_cast<int64_t>(seconds * 1e9), RCL_ROS_TIME);
}

Eigen::Matrix4d poseAt(double x, double y, double yaw_deg) {
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  pose.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(yaw_deg * kPi / 180.0, Eigen::Vector3d::UnitZ())
          .toRotationMatrix();
  pose(0, 3) = x;
  pose(1, 3) = y;
  return pose;
}

GlobalLocalizationResult goodResult(const Eigen::Matrix4d& pose) {
  GlobalLocalizationResult result;
  result.status = GLStatus::PendingVote;  // geometry + discriminative gates passed
  result.final_pose = pose;
  return result;
}

StaticConfirmParams defaultParams() {
  StaticConfirmParams params;
  params.confirm_frames = 3;
  params.confirm_xy_tol = 0.30;
  params.confirm_yaw_tol_deg = 5.0;
  params.probe_period_s = 0.5;
  params.attempt_timeout_s = 30.0;
  return params;
}

}  // namespace

TEST(StaticConfirmTest, ConfirmsAfterConsecutiveAgreements) {
  StaticConfirmState state;
  GLSummaryCounts summary;
  const StaticConfirmParams params = defaultParams();
  const Eigen::Matrix4d pose = poseAt(3.0, -1.0, 42.0);

  EXPECT_EQ(ingestStaticProbeResult(state, goodResult(pose), stampAt(1.0), params, summary),
            StaticConfirmOutcome::Pending);
  EXPECT_EQ(state.agree_count, 1);
  EXPECT_EQ(ingestStaticProbeResult(state, goodResult(pose), stampAt(1.6), params, summary),
            StaticConfirmOutcome::Pending);
  EXPECT_EQ(ingestStaticProbeResult(state, goodResult(pose), stampAt(2.2), params, summary),
            StaticConfirmOutcome::Confirmed);
  EXPECT_EQ(state.agree_count, params.confirm_frames);
  EXPECT_EQ(summary.static_confirm_probe_count, 3);
  EXPECT_EQ(summary.static_confirm_agree_count, 3);
}

TEST(StaticConfirmTest, SmallJitterWithinToleranceStillConfirms) {
  StaticConfirmState state;
  GLSummaryCounts summary;
  const StaticConfirmParams params = defaultParams();

  ingestStaticProbeResult(state, goodResult(poseAt(0.0, 0.0, 0.0)), stampAt(1.0), params, summary);
  ingestStaticProbeResult(state, goodResult(poseAt(0.10, 0.10, 3.0)), stampAt(1.6), params, summary);
  EXPECT_EQ(ingestStaticProbeResult(state, goodResult(poseAt(0.20, 0.05, -2.0)),
                                    stampAt(2.2), params, summary),
            StaticConfirmOutcome::Confirmed);
}

TEST(StaticConfirmTest, PoseJumpRestartsTheChain) {
  StaticConfirmState state;
  GLSummaryCounts summary;
  const StaticConfirmParams params = defaultParams();

  ingestStaticProbeResult(state, goodResult(poseAt(0.0, 0.0, 0.0)), stampAt(1.0), params, summary);
  ingestStaticProbeResult(state, goodResult(poseAt(0.0, 0.0, 0.0)), stampAt(1.6), params, summary);
  ASSERT_EQ(state.agree_count, 2);
  // A different place: the chain restarts from this pose rather than confirming.
  EXPECT_EQ(ingestStaticProbeResult(state, goodResult(poseAt(9.0, 4.0, 90.0)),
                                    stampAt(2.2), params, summary),
            StaticConfirmOutcome::Pending);
  EXPECT_EQ(state.agree_count, 1);
}

TEST(StaticConfirmTest, YawOutsideToleranceRestartsTheChain) {
  StaticConfirmState state;
  GLSummaryCounts summary;
  const StaticConfirmParams params = defaultParams();

  ingestStaticProbeResult(state, goodResult(poseAt(0.0, 0.0, 0.0)), stampAt(1.0), params, summary);
  EXPECT_EQ(ingestStaticProbeResult(state, goodResult(poseAt(0.0, 0.0, 30.0)),
                                    stampAt(1.6), params, summary),
            StaticConfirmOutcome::Pending);
  EXPECT_EQ(state.agree_count, 1);
}

TEST(StaticConfirmTest, AmbiguousOrLowConsensusZeroesTheChain) {
  StaticConfirmState state;
  GLSummaryCounts summary;
  const StaticConfirmParams params = defaultParams();
  const Eigen::Matrix4d pose = poseAt(1.0, 1.0, 10.0);

  ingestStaticProbeResult(state, goodResult(pose), stampAt(1.0), params, summary);
  ingestStaticProbeResult(state, goodResult(pose), stampAt(1.6), params, summary);
  ASSERT_EQ(state.agree_count, 2);

  GlobalLocalizationResult ambiguous;
  ambiguous.status = GLStatus::Ambiguous;
  EXPECT_EQ(ingestStaticProbeResult(state, ambiguous, stampAt(2.2), params, summary),
            StaticConfirmOutcome::Pending);
  EXPECT_EQ(state.agree_count, 0);

  // Two more good probes are not enough: the chain must build up again.
  EXPECT_EQ(ingestStaticProbeResult(state, goodResult(pose), stampAt(2.8), params, summary),
            StaticConfirmOutcome::Pending);
  EXPECT_EQ(ingestStaticProbeResult(state, goodResult(pose), stampAt(3.4), params, summary),
            StaticConfirmOutcome::Pending);
  EXPECT_EQ(ingestStaticProbeResult(state, goodResult(pose), stampAt(4.0), params, summary),
            StaticConfirmOutcome::Confirmed);
}

TEST(StaticConfirmTest, NonFinitePoseIsRejected) {
  StaticConfirmState state;
  GLSummaryCounts summary;
  const StaticConfirmParams params = defaultParams();

  GlobalLocalizationResult broken = goodResult(poseAt(0.0, 0.0, 0.0));
  broken.final_pose(0, 3) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(ingestStaticProbeResult(state, broken, stampAt(1.0), params, summary),
            StaticConfirmOutcome::Pending);
  EXPECT_EQ(state.agree_count, 0);
}

TEST(StaticConfirmTest, TimeoutReportedWhenNoAgreementInTime) {
  StaticConfirmState state;
  GLSummaryCounts summary;
  StaticConfirmParams params = defaultParams();
  params.attempt_timeout_s = 5.0;

  GlobalLocalizationResult no_candidate;
  no_candidate.status = GLStatus::NoCandidate;
  EXPECT_EQ(ingestStaticProbeResult(state, no_candidate, stampAt(1.0), params, summary),
            StaticConfirmOutcome::Pending);
  EXPECT_EQ(ingestStaticProbeResult(state, no_candidate, stampAt(7.0), params, summary),
            StaticConfirmOutcome::Timeout);
}

TEST(StaticConfirmTest, ConfirmationWinsOverTimeoutOnTheSameProbe) {
  StaticConfirmState state;
  GLSummaryCounts summary;
  StaticConfirmParams params = defaultParams();
  params.confirm_frames = 2;
  params.attempt_timeout_s = 3.0;
  const Eigen::Matrix4d pose = poseAt(2.0, 2.0, 0.0);

  ingestStaticProbeResult(state, goodResult(pose), stampAt(1.0), params, summary);
  // Past the timeout, but the agreement completes on this very probe.
  EXPECT_EQ(ingestStaticProbeResult(state, goodResult(pose), stampAt(9.0), params, summary),
            StaticConfirmOutcome::Confirmed);
}

TEST(StaticConfirmTest, ProbeThrottleRespectsPeriod) {
  StaticConfirmState state;
  GLSummaryCounts summary;
  const StaticConfirmParams params = defaultParams();

  EXPECT_TRUE(shouldRunStaticProbe(state, stampAt(1.0), params));
  ingestStaticProbeResult(state, goodResult(poseAt(0.0, 0.0, 0.0)), stampAt(1.0), params, summary);
  EXPECT_FALSE(shouldRunStaticProbe(state, stampAt(1.2), params));
  EXPECT_TRUE(shouldRunStaticProbe(state, stampAt(1.5), params));
  // A backwards clock step must not wedge the probe forever.
  EXPECT_TRUE(shouldRunStaticProbe(state, stampAt(0.5), params));
}

TEST(StaticConfirmTest, ResetClearsEverything) {
  StaticConfirmState state;
  GLSummaryCounts summary;
  const StaticConfirmParams params = defaultParams();

  ingestStaticProbeResult(state, goodResult(poseAt(1.0, 1.0, 0.0)), stampAt(1.0), params, summary);
  ASSERT_EQ(state.agree_count, 1);
  resetStaticConfirm(state);
  EXPECT_EQ(state.agree_count, 0);
  EXPECT_EQ(state.probe_count, 0);
  EXPECT_FALSE(state.has_pose);
  EXPECT_TRUE(shouldRunStaticProbe(state, stampAt(1.1), params));
}

TEST(StaticConfirmTest, InvalidParamsAreRejected) {
  StaticConfirmParams params = defaultParams();
  EXPECT_TRUE(validStaticConfirmParams(params));
  params.confirm_frames = 0;
  EXPECT_FALSE(validStaticConfirmParams(params));
  params = defaultParams();
  params.probe_period_s = 0.0;
  EXPECT_FALSE(validStaticConfirmParams(params));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
