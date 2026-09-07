#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include <Eigen/Geometry>

#include "localization/global_localization_gravity.hpp"

namespace localization {
namespace {

constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;

GravityAlignmentSample validSample() {
  GravityAlignmentSample sample;
  sample.stamp = rclcpp::Time(1'000'000'000LL, RCL_ROS_TIME);
  sample.uncertainty_deg = 0.5;
  sample.angular_velocity_rps = 0.1;
  sample.accel_norm_residual_mps2 = 0.2;
  sample.source = "test";
  sample.valid = true;
  return sample;
}

TEST(GlobalLocalizationGravity, VerticalUpProducesIdentityRotation) {
  Eigen::Matrix3d R_level_base;

  ASSERT_TRUE(makeLevelRotationFromUp(Eigen::Vector3d::UnitZ(), R_level_base));
  EXPECT_TRUE(R_level_base.isApprox(Eigen::Matrix3d::Identity(), 1e-12));
}

TEST(GlobalLocalizationGravity, ShortestRotationLevelsTiltedUpDirection) {
  const Eigen::Matrix3d R_world_base =
      (Eigen::AngleAxisd(25.0 * kDegreesToRadians, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(12.0 * kDegreesToRadians, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(-8.0 * kDegreesToRadians, Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
  const Eigen::Vector3d up_base =
      R_world_base.transpose() * Eigen::Vector3d::UnitZ();
  Eigen::Matrix3d R_level_base;

  ASSERT_TRUE(makeLevelRotationFromUp(up_base, R_level_base));
  EXPECT_TRUE((R_level_base * up_base).isApprox(Eigen::Vector3d::UnitZ(), 1e-12));
  EXPECT_TRUE((R_level_base.transpose() * R_level_base)
                  .isApprox(Eigen::Matrix3d::Identity(), 1e-12));
  EXPECT_NEAR(R_level_base.determinant(), 1.0, 1e-12);
}

TEST(GlobalLocalizationGravity, InvalidUpDirectionFailsClosed) {
  Eigen::Matrix3d R_level_base;
  EXPECT_FALSE(makeLevelRotationFromUp(Eigen::Vector3d::Zero(), R_level_base));
  EXPECT_TRUE(R_level_base.isIdentity());

  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(makeLevelRotationFromUp(Eigen::Vector3d(nan, 0.0, 1.0),
                                       R_level_base));
  EXPECT_TRUE(R_level_base.isIdentity());
}

TEST(GlobalLocalizationGravity, RawImuFilterLevelsStableTiltExactlyOnce) {
  RawImuGravityFilter filter;
  filter.configure(0.30, 30, 0.10, 1.5);
  const Eigen::Vector3d up_base =
      Eigen::AngleAxisd(4.5 * kDegreesToRadians, Eigen::Vector3d::UnitY()) *
      Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d acceleration = up_base * 9.80665;
  GravityAlignmentSample sample;

  for (int i = 0; i <= 30; ++i) {
    const bool valid = filter.addSample(
        rclcpp::Time(1'000'000'000LL + i * 10'000'000LL, RCL_ROS_TIME),
        acceleration, Eigen::Vector3d::Zero(), sample);
    EXPECT_EQ(valid, i == 30);
  }

  ASSERT_TRUE(sample.valid);
  EXPECT_EQ(sample.source, "raw_imu_gravity_filter");
  EXPECT_TRUE((sample.R_level_base * up_base)
                  .isApprox(Eigen::Vector3d::UnitZ(), 1e-12));
  EXPECT_NEAR(
      Eigen::AngleAxisd(sample.R_level_base).angle() / kDegreesToRadians,
      4.5, 1e-10);
  EXPECT_NEAR(std::abs(sample.pitch_deg), 4.5, 1e-10);
  EXPECT_LT(sample.uncertainty_deg, 1e-5);
}

TEST(GlobalLocalizationGravity, RawImuFilterHighAngularVelocityClearsWindow) {
  RawImuGravityFilter filter;
  filter.configure(0.10, 5, 0.10, 1.5);
  GravityAlignmentSample sample;

  for (int i = 0; i <= 10; ++i) {
    filter.addSample(
        rclcpp::Time(1'000'000'000LL + i * 10'000'000LL, RCL_ROS_TIME),
        Eigen::Vector3d(0.0, 0.0, 9.80665),
        Eigen::Vector3d::Zero(), sample);
  }
  ASSERT_GT(filter.sampleCount(), 0U);

  EXPECT_FALSE(filter.addSample(
      rclcpp::Time(1'110'000'000LL, RCL_ROS_TIME),
      Eigen::Vector3d(0.0, 0.0, 9.80665),
      Eigen::Vector3d(0.11, 0.0, 0.0), sample));
  EXPECT_EQ(filter.sampleCount(), 0U);
  EXPECT_FALSE(sample.valid);
}

TEST(GlobalLocalizationGravity, RawImuFilterAbnormalAccelerationClearsWindow) {
  RawImuGravityFilter filter;
  filter.configure(0.10, 5, 0.10, 1.5);
  GravityAlignmentSample sample;

  ASSERT_FALSE(filter.addSample(
      rclcpp::Time(1'000'000'000LL, RCL_ROS_TIME),
      Eigen::Vector3d(0.0, 0.0, 9.80665),
      Eigen::Vector3d::Zero(), sample));
  ASSERT_EQ(filter.sampleCount(), 1U);

  EXPECT_FALSE(filter.addSample(
      rclcpp::Time(1'010'000'000LL, RCL_ROS_TIME),
      Eigen::Vector3d(0.0, 0.0, 12.0),
      Eigen::Vector3d::Zero(), sample));
  EXPECT_EQ(filter.sampleCount(), 0U);
  EXPECT_FALSE(sample.valid);
}

TEST(GlobalLocalizationGravity, RawImuFilterRejectsOutOfOrderTimestamp) {
  RawImuGravityFilter filter;
  filter.configure(0.10, 5, 0.10, 1.5);
  GravityAlignmentSample sample;
  const Eigen::Vector3d acceleration(0.0, 0.0, 9.80665);

  ASSERT_FALSE(filter.addSample(
      rclcpp::Time(1'010'000'000LL, RCL_ROS_TIME), acceleration,
      Eigen::Vector3d::Zero(), sample));
  ASSERT_EQ(filter.sampleCount(), 1U);

  EXPECT_FALSE(filter.addSample(
      rclcpp::Time(1'000'000'000LL, RCL_ROS_TIME), acceleration,
      Eigen::Vector3d::Zero(), sample));
  EXPECT_EQ(filter.sampleCount(), 0U);
  EXPECT_FALSE(sample.valid);
}

TEST(GlobalLocalizationGravity, OrientationSamplePreservesDynamicTiltAndQuality) {
  const Eigen::Quaterniond world_from_sensor(
      Eigen::AngleAxisd(10.0 * kDegreesToRadians,
                        Eigen::Vector3d::UnitX()));
  GravityAlignmentSample sample;

  ASSERT_TRUE(makeGravitySampleFromOrientation(
      rclcpp::Time(1'000'000'000LL, RCL_ROS_TIME), world_from_sensor,
      Eigen::Vector3d(0.0001, 0.0004, 0.0009),
      Eigen::Matrix3d::Identity(),
      Eigen::Vector3d(0.0, 0.0, 9.80665),
      Eigen::Vector3d(0.1, 0.2, 0.3), "imu_orientation", sample));

  const Eigen::Vector3d up_sensor =
      world_from_sensor.toRotationMatrix().transpose() *
      Eigen::Vector3d::UnitZ();
  EXPECT_TRUE((sample.R_level_base * up_sensor)
                  .isApprox(Eigen::Vector3d::UnitZ(), 1e-12));
  EXPECT_NEAR(sample.roll_deg, 10.0, 1e-12);
  EXPECT_NEAR(sample.pitch_deg, 0.0, 1e-12);
  EXPECT_NEAR(sample.uncertainty_deg,
              std::sqrt(0.0004) / kDegreesToRadians, 1e-12);
  EXPECT_NEAR(sample.angular_velocity_rps,
              Eigen::Vector3d(0.1, 0.2, 0.3).norm(), 1e-12);
  EXPECT_NEAR(sample.accel_norm_residual_mps2, 0.0, 1e-12);
  EXPECT_EQ(sample.source, "imu_orientation");
  EXPECT_TRUE(sample.valid);
}

TEST(GlobalLocalizationGravity, UnknownOrientationCovarianceFailsClosed) {
  GravityAlignmentSample sample = validSample();
  ASSERT_FALSE(makeGravitySampleFromOrientation(
      rclcpp::Time(1'000'000'000LL, RCL_ROS_TIME),
      Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(),
      Eigen::Matrix3d::Identity(),
      Eigen::Vector3d(0.0, 0.0, 9.80665), Eigen::Vector3d::Zero(),
      "imu_orientation", sample));
  EXPECT_FALSE(sample.valid);
  EXPECT_TRUE(sample.R_level_base.isIdentity());
}

TEST(GlobalLocalizationGravity, SlerpTreatsQuaternionSignsAsEquivalent) {
  const Eigen::Quaterniond rotation(
      Eigen::AngleAxisd(20.0 * kDegreesToRadians, Eigen::Vector3d::UnitY()));
  Eigen::Quaterniond same_rotation_opposite_sign = rotation;
  same_rotation_opposite_sign.coeffs() *= -1.0;
  Eigen::Matrix3d interpolated;

  ASSERT_TRUE(slerpShortestRotation(rotation, same_rotation_opposite_sign, 0.5,
                                    interpolated));
  EXPECT_TRUE(interpolated.isApprox(rotation.toRotationMatrix(), 1e-12));
}

TEST(GlobalLocalizationGravity, HistorySlerpUsesBracketingSamples) {
  GLParams params;
  GravityAlignmentSample before = validSample();
  GravityAlignmentSample after = validSample();
  before.stamp = rclcpp::Time(980'000'000LL, RCL_ROS_TIME);
  after.stamp = rclcpp::Time(1'020'000'000LL, RCL_ROS_TIME);
  before.R_level_base = Eigen::AngleAxisd(
      -10.0 * kDegreesToRadians,
      Eigen::Vector3d::UnitX()).toRotationMatrix();
  after.R_level_base = Eigen::AngleAxisd(
      10.0 * kDegreesToRadians,
      Eigen::Vector3d::UnitX()).toRotationMatrix();
  before.roll_deg = -10.0;
  after.roll_deg = 10.0;
  before.uncertainty_deg = 0.4;
  after.uncertainty_deg = 0.8;
  GravityAlignmentSample selected;

  ASSERT_TRUE(selectGravitySampleForAnchor(
      {before, after}, rclcpp::Time(1'000'000'000LL, RCL_ROS_TIME),
      params, selected));
  EXPECT_TRUE(selected.R_level_base.isApprox(Eigen::Matrix3d::Identity(), 1e-12));
  EXPECT_NEAR(selected.roll_deg, 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(selected.uncertainty_deg, 0.8);
  EXPECT_EQ(selected.stamp.nanoseconds(), 1'000'000'000LL);
}

TEST(GlobalLocalizationGravity, HistorySelectionAppliesConfiguredTimeOffset) {
  GLParams params;
  params.level0_gravity_time_offset_ms = 20.0;
  GravityAlignmentSample before = validSample();
  GravityAlignmentSample after = validSample();
  before.stamp = rclcpp::Time(960'000'000LL, RCL_ROS_TIME);
  after.stamp = rclcpp::Time(1'000'000'000LL, RCL_ROS_TIME);
  GravityAlignmentSample selected;
  const rclcpp::Time anchor_stamp(1'000'000'000LL, RCL_ROS_TIME);

  ASSERT_TRUE(selectGravitySampleForAnchor(
      {before, after}, anchor_stamp, params, selected));
  EXPECT_EQ(selected.stamp.nanoseconds(), 980'000'000LL);
  EXPECT_TRUE(evaluateGravitySample(anchor_stamp, &selected, params).accepted);
}

TEST(GlobalLocalizationGravity, HistoryFallsBackOnlyToFreshNearestSample) {
  GLParams params;
  GravityAlignmentSample sample = validSample();
  sample.stamp = rclcpp::Time(975'000'000LL, RCL_ROS_TIME);
  GravityAlignmentSample selected;
  const rclcpp::Time anchor_stamp(1'000'000'000LL, RCL_ROS_TIME);

  EXPECT_TRUE(selectGravitySampleForAnchor(
      {sample}, anchor_stamp, params, selected));
  EXPECT_EQ(selected.stamp, sample.stamp);

  sample.stamp = rclcpp::Time(969'000'000LL, RCL_ROS_TIME);
  EXPECT_FALSE(selectGravitySampleForAnchor(
      {sample}, anchor_stamp, params, selected));
  EXPECT_FALSE(selected.valid);
}

TEST(GlobalLocalizationGravity, HistoryRejectsUnorderedOrMixedClockSamples) {
  const GLParams params;
  GravityAlignmentSample first = validSample();
  GravityAlignmentSample second = validSample();
  first.stamp = rclcpp::Time(1'000'000'000LL, RCL_ROS_TIME);
  second.stamp = rclcpp::Time(990'000'000LL, RCL_ROS_TIME);
  GravityAlignmentSample selected;
  const rclcpp::Time anchor_stamp(1'000'000'000LL, RCL_ROS_TIME);

  EXPECT_FALSE(selectGravitySampleForAnchor(
      {first, second}, anchor_stamp, params, selected));

  first.stamp = rclcpp::Time(990'000'000LL, RCL_SYSTEM_TIME);
  EXPECT_FALSE(selectGravitySampleForAnchor(
      {first}, anchor_stamp, params, selected));
}

TEST(GlobalLocalizationGravity, TimeOffsetIsAppliedBeforeAgeGate) {
  GLParams params;
  params.level0_gravity_time_offset_ms = 35.0;
  GravityAlignmentSample sample = validSample();
  const rclcpp::Time anchor_stamp(1'035'000'000LL, RCL_ROS_TIME);

  const GravityGateResult gate =
      evaluateGravitySample(anchor_stamp, &sample, params);

  EXPECT_TRUE(gate.accepted);
  EXPECT_EQ(gate.reject_reason, GLAnchorRejectReason::None);
  EXPECT_DOUBLE_EQ(gate.age_ms, 0.0);
}

TEST(GlobalLocalizationGravity, GateRejectsStaleAndUncertainSamples) {
  const GLParams params;
  const rclcpp::Time anchor_stamp(1'031'000'000LL, RCL_ROS_TIME);
  GravityAlignmentSample sample = validSample();

  GravityGateResult gate = evaluateGravitySample(anchor_stamp, &sample, params);
  EXPECT_FALSE(gate.accepted);
  EXPECT_EQ(gate.reject_reason, GLAnchorRejectReason::GravityStale);

  sample.stamp = anchor_stamp;
  sample.uncertainty_deg = params.level0_gravity_max_uncertainty_deg + 0.01;
  gate = evaluateGravitySample(anchor_stamp, &sample, params);
  EXPECT_FALSE(gate.accepted);
  EXPECT_EQ(gate.reject_reason, GLAnchorRejectReason::GravityUncertain);
}

TEST(GlobalLocalizationGravity, GateUsesDocumentedQualityOrderAndBoundaries) {
  GLParams params;
  GravityAlignmentSample sample = validSample();
  const rclcpp::Time anchor_stamp = sample.stamp;
  sample.uncertainty_deg = params.level0_gravity_max_uncertainty_deg;
  sample.roll_deg = params.level0_anchor_max_abs_rp_deg;
  sample.angular_velocity_rps = params.level0_anchor_max_angular_velocity_rps;
  sample.accel_norm_residual_mps2 =
      params.level0_gravity_max_accel_residual_mps2;
  EXPECT_TRUE(evaluateGravitySample(anchor_stamp, &sample, params).accepted);

  sample.roll_deg += 0.01;
  sample.angular_velocity_rps += 0.01;
  sample.accel_norm_residual_mps2 += 0.01;
  GravityGateResult gate = evaluateGravitySample(anchor_stamp, &sample, params);
  EXPECT_EQ(gate.reject_reason, GLAnchorRejectReason::TiltExceeded);

  sample.roll_deg = 0.0;
  gate = evaluateGravitySample(anchor_stamp, &sample, params);
  EXPECT_EQ(gate.reject_reason, GLAnchorRejectReason::AngularVelocityExceeded);

  sample.angular_velocity_rps = 0.0;
  gate = evaluateGravitySample(anchor_stamp, &sample, params);
  EXPECT_EQ(gate.reject_reason,
            GLAnchorRejectReason::AccelerationResidualExceeded);
}

TEST(GlobalLocalizationGravity, DisabledAlignmentUsesStricterTiltAndIdentity) {
  GLParams params;
  params.level0_gravity_align_enabled = false;
  GravityAlignmentSample sample = validSample();
  sample.roll_deg = params.level0_unaligned_max_abs_rp_deg;

  GravityGateResult gate =
      evaluateGravitySample(sample.stamp, &sample, params);
  EXPECT_TRUE(gate.accepted);
  EXPECT_TRUE(gate.R_level_base.isIdentity());

  sample.roll_deg += 0.01;
  gate = evaluateGravitySample(sample.stamp, &sample, params);
  EXPECT_FALSE(gate.accepted);
  EXPECT_EQ(gate.reject_reason, GLAnchorRejectReason::TiltExceeded);
}

TEST(GlobalLocalizationGravity, RejectReasonUpdatesMatchingSummaryCounter) {
  const GLParams params;
  GravityAlignmentSample sample = validSample();
  const rclcpp::Time anchor_stamp(1'031'000'000LL, RCL_ROS_TIME);
  const GravityGateResult gate =
      evaluateGravitySample(anchor_stamp, &sample, params);
  GLSummaryCounts summary;

  accumulateGravityGateSummary(&sample, gate, summary);

  EXPECT_DOUBLE_EQ(summary.anchor_gravity_age_ms, 31.0);
  EXPECT_EQ(summary.anchor_rejected_gravity_stale, 1);
  EXPECT_EQ(summary.anchor_rejected_gravity_unavailable, 0);
  EXPECT_DOUBLE_EQ(summary.anchor_gravity_uncertainty_deg, 0.5);
}

TEST(GlobalLocalizationGravity, MapBaseSeedUsesLevelThenBaseCompositionOrder) {
  const Eigen::Matrix3d R_level_base =
      Eigen::AngleAxisd(-10.0 * kDegreesToRadians,
                        Eigen::Vector3d::UnitX()).toRotationMatrix();
  Eigen::Matrix4d T_map_level = Eigen::Matrix4d::Identity();
  T_map_level.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(60.0 * kDegreesToRadians,
                        Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T_map_level.block<3, 1>(0, 3) = Eigen::Vector3d(4.0, -2.0, 0.5);
  Eigen::Matrix4d T_map_base;

  ASSERT_TRUE(composeMapBaseSeed(T_map_level, R_level_base, T_map_base));

  const Eigen::Vector4d point_base(1.0, 2.0, 3.0, 1.0);
  Eigen::Matrix4d T_level_base = Eigen::Matrix4d::Identity();
  T_level_base.block<3, 3>(0, 0) = R_level_base;
  EXPECT_TRUE((T_map_base * point_base)
                  .isApprox(T_map_level * T_level_base * point_base, 1e-12));
}

}  // namespace
}  // namespace localization
