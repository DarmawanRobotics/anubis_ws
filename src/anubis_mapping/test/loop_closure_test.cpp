#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "loop_closure.h"

namespace {
using PoseVector = std::vector<Eigen::Matrix4d,
                               Eigen::aligned_allocator<Eigen::Matrix4d>>;

double endpointError(const PoseVector& poses) {
    return (poses.back().block<3, 1>(0, 3) -
            poses.front().block<3, 1>(0, 3)).norm();
}

Eigen::Matrix4d poseWithYaw(double x, double y, double z, double yaw) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    pose.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
    return pose;
}
}  // namespace

TEST(LoopClosureTest, NoLoopPreservesOdometry) {
    PoseVector poses(3, Eigen::Matrix4d::Identity());
    poses[1](0, 3) = 1.0;
    poses[2](0, 3) = 2.0;
    anubis_mapping::LoopClosureConfig config;
    const auto optimized = anubis_mapping::optimizePoseGraph(poses, {}, config);
    ASSERT_EQ(optimized.size(), poses.size());
    for (size_t index = 0; index < poses.size(); ++index) {
        EXPECT_TRUE(optimized[index].isApprox(poses[index], 1e-5));
    }
}

TEST(LoopClosureTest, ReportsSuccessfulSanityCheck) {
    PoseVector poses(2, Eigen::Matrix4d::Identity());
    poses[1](0, 3) = 1.0;
    anubis_mapping::LoopClosureConfig config;
    bool sanity_accepted = false;
    std::string sanity_reason;

    const auto optimized = anubis_mapping::optimizePoseGraph(
        poses, {}, config, &sanity_accepted, &sanity_reason);

    EXPECT_TRUE(sanity_accepted);
    EXPECT_EQ(sanity_reason, "accepted");
    ASSERT_EQ(optimized.size(), poses.size());
}

TEST(LoopClosureTest, EmptyTrajectoryReportsRejectedSanity) {
    anubis_mapping::LoopClosureConfig config;
    bool sanity_accepted = true;
    std::string sanity_reason;

    const auto optimized = anubis_mapping::optimizePoseGraph(
        {}, {}, config, &sanity_accepted, &sanity_reason);

    EXPECT_FALSE(sanity_accepted);
    EXPECT_EQ(sanity_reason, "empty_trajectory");
    EXPECT_TRUE(optimized.empty());
}

TEST(LoopClosureTest, SquareLoopReducesEndpointDrift) {
    PoseVector poses;
    for (int index = 0; index < 20; ++index) {
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        const int side = index / 5;
        const double step = index % 5;
        if (side == 0) { pose(0, 3) = step; }
        else if (side == 1) { pose(0, 3) = 5.0; pose(1, 3) = step; }
        else if (side == 2) { pose(0, 3) = 5.0 - step; pose(1, 3) = 5.0; }
        else { pose(1, 3) = 5.0 - step; }
        pose(0, 3) += 0.03 * index;
        pose(1, 3) -= 0.02 * index;
        poses.push_back(pose);
    }
    anubis_mapping::LoopConstraint loop;
    loop.first = 0;
    loop.second = poses.size() - 1;
    loop.T_first_second = Eigen::Matrix4d::Identity();
    anubis_mapping::LoopClosureConfig config;
    config.loop_translation_sigma = 0.02;
    // This test isolates optimizer convergence with a deliberately large
    // synthetic drift. Production safety limits are covered separately.
    config.max_optimized_xy_deviation = 1.0;
    const auto optimized = anubis_mapping::optimizePoseGraph(poses, {loop}, config);
    EXPECT_LT(endpointError(optimized), endpointError(poses) / 5.0);
}

TEST(LoopClosureTest, TinyCorrectionOnTiltedPoseDoesNotJumpEulerBranch) {
    constexpr double kPitch = 17.0 * M_PI / 180.0;
    constexpr double kTinyNegativeYaw = -1e-12;
    const Eigen::Matrix3d reference =
        Eigen::AngleAxisd(kPitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Matrix3d candidate =
        reference * Eigen::AngleAxisd(
            kTinyNegativeYaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    const Eigen::Matrix3d relative = candidate * reference.transpose();
    const Eigen::Vector3d ambiguous = relative.eulerAngles(0, 1, 2);
    EXPECT_GT(std::max(std::abs(ambiguous.x()), std::abs(ambiguous.y())), 3.0);
    EXPECT_LT(
        anubis_mapping::detail::rollPitchDeviationRadians(reference, candidate),
        1e-9);
}

TEST(LoopClosureTest, GenuinePitchDeviationStillTripsSanityThreshold) {
    constexpr double kPitch = 17.0 * M_PI / 180.0;
    constexpr double kDeviation = 0.12;
    const Eigen::Matrix3d reference =
        Eigen::AngleAxisd(kPitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Matrix3d candidate = Eigen::AngleAxisd(
        kPitch + kDeviation, Eigen::Vector3d::UnitY()).toRotationMatrix();

    EXPECT_NEAR(
        anubis_mapping::detail::rollPitchDeviationRadians(reference, candidate),
        kDeviation,
        1e-12);
    EXPECT_GT(kDeviation, 0.10);
}

TEST(LoopClosureTest, ProjectsBoundedIcpRoundoffToRigidPose) {
    Eigen::Matrix4d input = poseWithYaw(1.2, -0.4, 0.3, 0.6);
    input(0, 0) += 2e-5;
    input(1, 1) -= 2e-5;
    Eigen::Matrix4d output = Eigen::Matrix4d::Zero();

    ASSERT_TRUE(anubis_mapping::detail::projectNearRigidPose(input, &output));
    const Eigen::Matrix3d projected_rotation = output.block<3, 3>(0, 0);
    const Eigen::Vector3d projected_translation = output.block<3, 1>(0, 3);
    const Eigen::Vector3d input_translation = input.block<3, 1>(0, 3);
    EXPECT_TRUE((projected_rotation.transpose() * projected_rotation)
                    .isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_NEAR(projected_rotation.determinant(), 1.0, 1e-12);
    EXPECT_TRUE(projected_translation.isApprox(input_translation, 0.0));
}

TEST(LoopClosureTest, RejectsMateriallyNonRigidIcpResults) {
    Eigen::Matrix4d scaled = Eigen::Matrix4d::Identity();
    scaled(0, 0) = 1.01;
    Eigen::Matrix4d output = Eigen::Matrix4d::Identity();
    EXPECT_FALSE(anubis_mapping::detail::projectNearRigidPose(scaled, &output));

    Eigen::Matrix4d sheared = Eigen::Matrix4d::Identity();
    sheared(0, 1) = 0.01;
    EXPECT_FALSE(anubis_mapping::detail::projectNearRigidPose(sheared, &output));

    Eigen::Matrix4d non_finite = Eigen::Matrix4d::Identity();
    non_finite(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(anubis_mapping::detail::projectNearRigidPose(
        non_finite, &output));
}

TEST(LoopClosureTest, MeasuresConstraintInnovationAgainstOdometry) {
    constexpr double kDegrees = M_PI / 180.0;
    const Eigen::Matrix4d odometry = poseWithYaw(2.0, 1.0, 0.10, 5.0 * kDegrees);
    const Eigen::Matrix4d loop = poseWithYaw(1.5, 1.4, 0.15, -2.0 * kDegrees);

    const auto innovation =
        anubis_mapping::detail::measureLoopInnovation(odometry, loop);

    EXPECT_NEAR(innovation.odom_xy_distance, std::sqrt(5.0), 1e-12);
    EXPECT_NEAR(innovation.correction_xy, std::sqrt(0.41), 1e-12);
    EXPECT_NEAR(innovation.correction_z, 0.05, 1e-12);
    EXPECT_NEAR(innovation.correction_yaw_rad, 7.0 * kDegrees, 1e-12);
}

TEST(LoopClosureTest, ClassifiesUnsafeConstraintInnovation) {
    anubis_mapping::LoopClosureConfig config;
    anubis_mapping::detail::LoopInnovation innovation;

    EXPECT_EQ(
        anubis_mapping::detail::loopInnovationRejectionReason(innovation, config),
        nullptr);

    innovation.odom_xy_distance = config.max_odom_xy_distance + 0.01;
    EXPECT_STREQ(
        anubis_mapping::detail::loopInnovationRejectionReason(innovation, config),
        "odom_xy_distance");

    innovation = {};
    innovation.correction_xy = config.max_constraint_xy_correction + 0.01;
    EXPECT_STREQ(
        anubis_mapping::detail::loopInnovationRejectionReason(innovation, config),
        "constraint_xy_correction");

    innovation = {};
    innovation.correction_z = config.max_constraint_z_correction + 0.01;
    EXPECT_STREQ(
        anubis_mapping::detail::loopInnovationRejectionReason(innovation, config),
        "constraint_z_correction");

    innovation = {};
    innovation.correction_yaw_rad =
        (config.max_constraint_yaw_correction_deg + 0.1) * M_PI / 180.0;
    EXPECT_STREQ(
        anubis_mapping::detail::loopInnovationRejectionReason(innovation, config),
        "constraint_yaw_correction");
}

TEST(LoopClosureTest, LaterDescriptorCandidateCanPassUnchangedOdomGate) {
    anubis_mapping::LoopClosureConfig config;
    const Eigen::Matrix4d current = poseWithYaw(5.0, 0.0, 0.1, 0.0);
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
        candidates{
            poseWithYaw(0.0, 0.0, 0.0, 0.0),
            poseWithYaw(3.5, 0.0, 0.0, 0.0)};

    const auto first = anubis_mapping::detail::measureLoopInnovation(
        candidates[0].inverse() * current,
        candidates[0].inverse() * current);
    const auto second = anubis_mapping::detail::measureLoopInnovation(
        candidates[1].inverse() * current,
        candidates[1].inverse() * current);

    EXPECT_STREQ(
        anubis_mapping::detail::loopInnovationRejectionReason(first, config),
        "odom_xy_distance");
    EXPECT_EQ(
        anubis_mapping::detail::loopInnovationRejectionReason(second, config),
        nullptr);
    EXPECT_DOUBLE_EQ(config.max_odom_xy_distance, 3.5);
}

TEST(LoopClosureTest, RejectsPoseGraphWithLargeXYDeviation) {
    PoseVector poses{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(1.0, 0.0, 0.0, 0.0),
        poseWithYaw(2.0, 0.0, 0.0, 0.0)};
    anubis_mapping::LoopConstraint false_loop;
    false_loop.first = 0;
    false_loop.second = 2;
    false_loop.T_first_second = Eigen::Matrix4d::Identity();
    anubis_mapping::LoopClosureConfig config;
    config.loop_translation_sigma = 1e-3;
    config.max_optimized_xy_deviation = 0.10;
    bool sanity_accepted = true;
    std::string sanity_reason;

    const auto optimized = anubis_mapping::optimizePoseGraph(
        poses, {false_loop}, config, &sanity_accepted, &sanity_reason);

    EXPECT_FALSE(sanity_accepted);
    EXPECT_EQ(sanity_reason, "xy_deviation");
    ASSERT_EQ(optimized.size(), poses.size());
    for (size_t index = 0; index < poses.size(); ++index) {
        EXPECT_TRUE(optimized[index].isApprox(poses[index], 1e-12));
    }
}

TEST(LoopClosureTest, RejectsPoseGraphWithLargeYawDeviation) {
    constexpr double kDegrees = M_PI / 180.0;
    PoseVector poses{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(1.0, 0.0, 0.0, 30.0 * kDegrees)};
    anubis_mapping::LoopConstraint false_loop;
    false_loop.first = 0;
    false_loop.second = 1;
    false_loop.T_first_second = poseWithYaw(1.0, 0.0, 0.0, 0.0);
    anubis_mapping::LoopClosureConfig config;
    config.odom_rotation_sigma = 0.20;
    config.loop_rotation_sigma = 1e-3;
    config.max_optimized_yaw_deviation_deg = 2.0;
    bool sanity_accepted = true;
    std::string sanity_reason;

    const auto optimized = anubis_mapping::optimizePoseGraph(
        poses, {false_loop}, config, &sanity_accepted, &sanity_reason);

    EXPECT_FALSE(sanity_accepted);
    EXPECT_EQ(sanity_reason, "yaw_deviation");
    ASSERT_EQ(optimized.size(), poses.size());
    EXPECT_TRUE(optimized[1].isApprox(poses[1], 1e-12));
}

TEST(LoopClosureTest, RejectsNonRigidOdometryBeforeOptimizer) {
    PoseVector poses{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(1.0, 0.0, 0.0, 0.0)};
    poses[1](0, 0) = 2.0;  // finite, but no longer a rotation matrix
    anubis_mapping::LoopClosureConfig config;
    bool sanity_accepted = true;

    const auto optimized = anubis_mapping::optimizePoseGraph(
        poses, {}, config, &sanity_accepted);

    EXPECT_FALSE(sanity_accepted);
    ASSERT_EQ(optimized.size(), poses.size());
    EXPECT_TRUE(optimized[1].isApprox(poses[1], 1e-12));
}

TEST(LoopClosureTest, RejectsNonRigidLoopConstraintBeforeOptimizer) {
    PoseVector poses{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(1.0, 0.0, 0.0, 0.0)};
    anubis_mapping::LoopConstraint loop;
    loop.first = 0;
    loop.second = 1;
    loop.T_first_second(0, 0) = 2.0;
    anubis_mapping::LoopClosureConfig config;
    bool sanity_accepted = true;

    const auto optimized = anubis_mapping::optimizePoseGraph(
        poses, {loop}, config, &sanity_accepted);

    EXPECT_FALSE(sanity_accepted);
    ASSERT_EQ(optimized.size(), poses.size());
    EXPECT_TRUE(optimized[1].isApprox(poses[1], 1e-12));
}

TEST(LoopClosureTest, ScalesOnlyXYAndYawSanityLimitsWithPathLength) {
    PoseVector odometry{
         poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(20.0, 0.0, 0.0, 0.0)};
    PoseVector optimized = odometry;
    optimized[1](0, 3) += 0.75;
    optimized[1](2, 3) += 0.30;
    anubis_mapping::LoopClosureConfig config;
    config.max_optimized_xy_deviation = 0.5;
    config.max_optimized_z_deviation = 0.45;
    config.sanity_path_reference_length_m = 10.0;
    config.sanity_path_scale_max = 4.0;

    const auto report = anubis_mapping::detail::evaluatePoseGraphSanity(
        odometry, optimized, config);

    EXPECT_TRUE(report.accepted);
    EXPECT_DOUBLE_EQ(report.effective_xy_limit, 1.0);
    EXPECT_NEAR(report.max_z_deviation, 0.30, 1e-12);
}

TEST(LoopClosureTest, VerticalDriftDoesNotRelaxHorizontalSanityLimit) {
    PoseVector odometry{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(0.0, 0.0, 20.0, 0.0),
        poseWithYaw(0.0, 0.0, 40.0, 0.0)};
    PoseVector optimized = odometry;
    optimized.back()(0, 3) += 0.75;
    anubis_mapping::LoopClosureConfig config;
    config.max_optimized_xy_deviation = 0.5;
    config.max_optimized_z_deviation = 100.0;
    config.sanity_path_reference_length_m = 10.0;
    config.sanity_path_scale_max = 4.0;

    const auto report = anubis_mapping::detail::evaluatePoseGraphSanity(
        odometry, optimized, config);

    EXPECT_FALSE(report.accepted);
    EXPECT_EQ(report.offending_index, 2U);
    EXPECT_DOUBLE_EQ(report.path_length_m, 0.0);
    EXPECT_DOUBLE_EQ(report.effective_xy_limit, 0.5);
    EXPECT_EQ(report.rejection_reason, "xy_deviation");
}

TEST(LoopClosureTest, AcceptsZAndRollPitchInsideAbsoluteLimits) {
    PoseVector odometry{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(1.0, 0.0, 0.0, 0.0)};
    PoseVector optimized = odometry;
    optimized[1](2, 3) = 0.44;
    optimized[1].block<3, 3>(0, 0) =
        Eigen::AngleAxisd(6.9 * M_PI / 180.0,
                          Eigen::Vector3d::UnitX()).toRotationMatrix();

    const auto report = anubis_mapping::detail::evaluatePoseGraphSanity(
        odometry, optimized, anubis_mapping::LoopClosureConfig{});

    EXPECT_TRUE(report.accepted);
    EXPECT_NEAR(report.max_z_deviation, 0.44, 1e-12);
    EXPECT_NEAR(report.max_roll_pitch_deviation_rad,
                6.9 * M_PI / 180.0, 1e-12);
}

TEST(LoopClosureTest, RejectsZBeyondAbsoluteLimitWithReason) {
    PoseVector odometry{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(1.0, 0.0, 0.0, 0.0)};
    PoseVector optimized = odometry;
    optimized[1](2, 3) = 0.46;

    const auto report = anubis_mapping::detail::evaluatePoseGraphSanity(
        odometry, optimized, anubis_mapping::LoopClosureConfig{});

    EXPECT_FALSE(report.accepted);
    EXPECT_EQ(report.offending_index, 1U);
    EXPECT_EQ(report.rejection_reason, "z_deviation");
}

TEST(LoopClosureTest, RejectsRollPitchBeyondAbsoluteLimitWithReason) {
    PoseVector odometry{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(1.0, 0.0, 0.0, 0.0)};
    PoseVector optimized = odometry;
    optimized[1].block<3, 3>(0, 0) =
        Eigen::AngleAxisd(7.1 * M_PI / 180.0,
                          Eigen::Vector3d::UnitY()).toRotationMatrix();

    const auto report = anubis_mapping::detail::evaluatePoseGraphSanity(
        odometry, optimized, anubis_mapping::LoopClosureConfig{});

    EXPECT_FALSE(report.accepted);
    EXPECT_EQ(report.offending_index, 1U);
    EXPECT_EQ(report.rejection_reason, "roll_pitch_deviation");
}

TEST(LoopClosureTest, SanityRejectsNonRigidPoseEvenWhenFinite) {
    PoseVector odometry{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(1.0, 0.0, 0.0, 0.0)};
    PoseVector optimized = odometry;
    optimized.back()(0, 0) = 1.01;

    const auto report = anubis_mapping::detail::evaluatePoseGraphSanity(
        odometry, optimized, anubis_mapping::LoopClosureConfig{});

    EXPECT_FALSE(report.accepted);
    EXPECT_EQ(report.offending_index, 1U);
    EXPECT_EQ(report.rejection_reason, "non_finite_pose");
}

TEST(LoopClosureTest, SanityRejectsInvalidSafetyConfiguration) {
    PoseVector poses{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(1.0, 0.0, 0.0, 0.0)};
    anubis_mapping::LoopClosureConfig config;
    config.max_optimized_z_deviation =
        std::numeric_limits<double>::quiet_NaN();

    const auto report = anubis_mapping::detail::evaluatePoseGraphSanity(
        poses, poses, config);

    EXPECT_FALSE(report.accepted);
    EXPECT_EQ(report.offending_index, 0U);
    EXPECT_EQ(report.rejection_reason, "invalid_sanity_configuration");
}

TEST(LoopClosureTest, FinalPoseSafetyGateIsIndependentOfGeometryPolicy) {
    anubis_mapping::detail::PoseGraphSanityReport accepted;
    anubis_mapping::detail::PoseGraphSanityReport rejected;
    rejected.accepted = false;
    rejected.rejection_reason = "z_deviation";

    EXPECT_TRUE(anubis_mapping::detail::acceptsFinalPoseSafetyGate(
        accepted, accepted, accepted));
    EXPECT_FALSE(anubis_mapping::detail::acceptsFinalPoseSafetyGate(
        rejected, accepted, accepted));
    EXPECT_FALSE(anubis_mapping::detail::acceptsFinalPoseSafetyGate(
        accepted, rejected, accepted));
    EXPECT_FALSE(anubis_mapping::detail::acceptsFinalPoseSafetyGate(
        accepted, accepted, rejected));
}

TEST(LoopClosureTest, AcceptsSmallConsistentCorrection) {
    PoseVector poses{
        poseWithYaw(0.0, 0.0, 0.0, 0.0),
        poseWithYaw(0.10, 0.0, 0.0, 1.0 * M_PI / 180.0)};
    anubis_mapping::LoopConstraint loop;
    loop.first = 0;
    loop.second = 1;
    loop.T_first_second = Eigen::Matrix4d::Identity();
    anubis_mapping::LoopClosureConfig config;
    config.loop_translation_sigma = 0.02;
    config.loop_rotation_sigma = 0.02;
    bool sanity_accepted = false;

    const auto optimized = anubis_mapping::optimizePoseGraph(
        poses, {loop}, config, &sanity_accepted);

    EXPECT_TRUE(sanity_accepted);
    ASSERT_EQ(optimized.size(), poses.size());
    EXPECT_LT(endpointError(optimized), endpointError(poses));
}
