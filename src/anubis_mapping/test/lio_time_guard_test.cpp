#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "lio_time_guard.h"

namespace
{
using anubis_mapping::LioTimeGuardConfig;
using anubis_mapping::evaluate_lio_time_guard;

LioTimeGuardConfig config()
{
    return LioTimeGuardConfig{true, 0.25, 0.10, 0.002};
}
}  // namespace

TEST(LioTimeGuardTest, AcceptsFirstAndContinuousFrame)
{
    const auto first = evaluate_lio_time_guard(
        10.0, 10.1, -1.0, {10.0, 10.005, 10.010}, config());
    EXPECT_TRUE(first.accepted);
    // The subtraction of decimal timestamps is not bit-identical across
    // architectures (Jetson reports 0.0050000000000007816 here).
    EXPECT_NEAR(first.imu_dt_max, 0.005, 1e-12);

    const auto next = evaluate_lio_time_guard(
        10.2, 10.3, 10.1, {10.2, 10.205, 10.210}, config());
    EXPECT_TRUE(next.accepted);
    EXPECT_NEAR(next.scan_gap, 0.1, 1e-12);
}

TEST(LioTimeGuardTest, RejectsLargeScanGap)
{
    const auto decision = evaluate_lio_time_guard(
        10.36, 10.46, 10.1, {10.36, 10.365}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "scan_gap");
}

TEST(LioTimeGuardTest, AcceptsOnlyBoundedLidarWindowOverlap)
{
    auto decision = evaluate_lio_time_guard(
        10.100, 10.200, 10.101, {10.100, 10.105}, config());
    EXPECT_TRUE(decision.accepted);
    EXPECT_NEAR(decision.scan_gap, -0.001, 1e-12);

    decision = evaluate_lio_time_guard(
        10.100, 10.200, 10.103, {10.100, 10.105}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "lidar_out_of_order");
    EXPECT_NEAR(decision.scan_gap, -0.003, 1e-12);
}

TEST(LioTimeGuardTest, RejectsReversedOrNonFiniteLidarTime)
{
    auto decision = evaluate_lio_time_guard(
        10.1, 10.0, 10.0, {10.0, 10.01}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "lidar_timestamp_reversed");

    decision = evaluate_lio_time_guard(
        std::numeric_limits<double>::quiet_NaN(), 10.1, -1.0,
        {10.0, 10.01}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "non_finite_lidar_timestamp");
}

TEST(LioTimeGuardTest, RejectsOutOfOrderAndZeroImuSamples)
{
    auto decision = evaluate_lio_time_guard(
        10.0, 10.1, -1.0, {10.0, 10.005, 10.004}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "imu_timestamp_out_of_order");

    decision = evaluate_lio_time_guard(
        10.0, 10.1, -1.0, {10.0, 10.005, 10.005}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "zero_imu_dt");
}

TEST(LioTimeGuardTest, RejectsLargeAndNonFiniteImuIntervals)
{
    auto decision = evaluate_lio_time_guard(
        10.0, 10.1, -1.0, {10.0, 10.2}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "imu_dt");
    EXPECT_NEAR(decision.imu_dt_max, 0.2, 1e-12);

    decision = evaluate_lio_time_guard(
        10.0, 10.1, -1.0,
        {10.0, std::numeric_limits<double>::quiet_NaN()}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "non_finite_imu_timestamp");

    // A malformed IMU batch must win over a coincident large LiDAR gap; the
    // caller must not treat the frame as recoverable re-anchor data.
    decision = evaluate_lio_time_guard(
        11.0, 11.1, 10.0,
        {11.0, std::numeric_limits<double>::quiet_NaN()}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "non_finite_imu_timestamp");
}

TEST(LioTimeGuardTest, ChecksImuGapAcrossFrameBoundaries)
{
    const auto continuous = evaluate_lio_time_guard(
        10.2, 10.3, 10.1, {10.2, 10.205}, config(), 10.195);
    EXPECT_TRUE(continuous.accepted);
    EXPECT_NEAR(continuous.imu_cross_gap, 0.005, 1e-12);
    EXPECT_NEAR(continuous.imu_dt_max, 0.005, 1e-12);

    const auto blackout = evaluate_lio_time_guard(
        10.2, 10.3, 10.1, {10.2, 10.205}, config(), 10.0);
    EXPECT_FALSE(blackout.accepted);
    EXPECT_STREQ(blackout.rejection_reason, "imu_cross_frame_gap");
    EXPECT_NEAR(blackout.imu_cross_gap, 0.2, 1e-12);
    EXPECT_NEAR(blackout.imu_dt_max, 0.2, 1e-12);
}

TEST(LioTimeGuardTest, RejectsCrossFrameImuRollbackZeroAndBadAnchor)
{
    auto decision = evaluate_lio_time_guard(
        10.2, 10.3, 10.1, {10.2, 10.205}, config(), 10.21);
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "imu_timestamp_out_of_order");

    decision = evaluate_lio_time_guard(
        10.2, 10.3, 10.1, {10.2, 10.205}, config(), 10.2);
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "zero_imu_dt");

    decision = evaluate_lio_time_guard(
        10.2, 10.3, 10.1, {10.2}, config(),
        std::numeric_limits<double>::quiet_NaN());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(
        decision.rejection_reason, "non_finite_previous_imu_timestamp");
}

TEST(LioTimeGuardTest, RejectsMissingSamplesAndZeroScanDuration)
{
    auto decision = evaluate_lio_time_guard(
        10.0, 10.0, -1.0, {}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "zero_lidar_duration");

    decision = evaluate_lio_time_guard(
        10.0, 10.1, -1.0, {}, config());
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "missing_imu_samples");
}

TEST(LioTimeGuardTest, InvalidThresholdsFailClosed)
{
    auto invalid = config();
    invalid.max_scan_gap = 0.0;
    auto decision = evaluate_lio_time_guard(
        10.0, 10.1, -1.0, {10.0, 10.01}, invalid);
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "invalid_configuration");

    invalid = config();
    invalid.max_imu_dt = std::numeric_limits<double>::infinity();
    decision = evaluate_lio_time_guard(
        10.0, 10.1, -1.0, {}, invalid);
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "invalid_configuration");

    invalid = config();
    invalid.max_scan_overlap = -0.001;
    decision = evaluate_lio_time_guard(
        10.0, 10.1, -1.0, {10.0, 10.01}, invalid);
    EXPECT_FALSE(decision.accepted);
    EXPECT_STREQ(decision.rejection_reason, "invalid_configuration");
}

TEST(LioTimeGuardTest, DisabledGuardPreservesCallerPath)
{
    auto disabled = config();
    disabled.enabled = false;
    const auto decision = evaluate_lio_time_guard(
        std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0,
        {1.0, 0.0}, disabled);
    EXPECT_TRUE(decision.accepted);
    EXPECT_EQ(decision.rejection_reason, nullptr);
}

TEST(LioTimeGuardTest, InvalidatedAnchorDisablesCrossEpochImuPrepend)
{
    EXPECT_TRUE(anubis_mapping::can_reuse_imu_time_anchor(true, true));
    EXPECT_FALSE(anubis_mapping::can_reuse_imu_time_anchor(false, true));
    EXPECT_FALSE(anubis_mapping::can_reuse_imu_time_anchor(true, false));
    EXPECT_FALSE(anubis_mapping::can_reuse_imu_time_anchor(false, false));
}

TEST(LioTimeGuardTest, RejectedFrameDropsAndReAnchorsWithoutFailingSession)
{
    anubis_mapping::LioTimeGuardSessionState session;
    const auto accepted = evaluate_lio_time_guard(
        10.0, 10.1, -1.0, {10.0, 10.005}, config());
    EXPECT_TRUE(anubis_mapping::commit_lio_time_guard_frame(
        session, accepted, 10.1, 10.005));
    EXPECT_FALSE(session.failed);
    EXPECT_NEAR(session.previous_imu_end, 10.005, 1e-12);

    const auto rejected = evaluate_lio_time_guard(
        10.2, 10.3, session.previous_lidar_end,
        {10.01, 10.015}, config(), session.previous_imu_end);
    // The cross-frame sample gap is intentionally oversized.
    const auto bad = evaluate_lio_time_guard(
        10.2, 10.3, session.previous_lidar_end,
        {10.2, 10.205}, config(), 9.0);
    EXPECT_TRUE(rejected.accepted);
    EXPECT_FALSE(anubis_mapping::commit_lio_time_guard_frame(
        session, bad, 10.3, 10.205));
    // A rejected frame is counted and the cursor re-anchors to it, but the
    // session survives: a transient gap must not discard the whole mapping
    // run nor block saving.
    EXPECT_FALSE(session.failed);
    EXPECT_EQ(session.rejected_frames, 1U);
    EXPECT_STREQ(session.failure_reason.c_str(), "imu_cross_frame_gap");
    EXPECT_DOUBLE_EQ(session.previous_lidar_end, 10.3);
    EXPECT_DOUBLE_EQ(session.previous_imu_end, 10.205);
    // The next continuous frame is now evaluated against the re-anchored
    // timeline and is accepted again: syncData consumes IMU continuously, so
    // the next batch starts right after the dropped batch (5 ms here).
    const auto next = evaluate_lio_time_guard(
        10.35, 10.45, session.previous_lidar_end,
        {10.21, 10.215}, config(), session.previous_imu_end);
    EXPECT_TRUE(next.accepted);
    EXPECT_TRUE(anubis_mapping::commit_lio_time_guard_frame(
        session, next, 10.45, 10.215));

    session.reset();
    EXPECT_FALSE(session.failed);
    EXPECT_EQ(session.rejected_frames, 0U);
    EXPECT_EQ(session.previous_lidar_end, -1.0);
    EXPECT_TRUE(anubis_mapping::commit_lio_time_guard_frame(
        session, accepted, 10.1, 10.005));
}
