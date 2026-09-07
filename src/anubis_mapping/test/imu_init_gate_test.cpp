#include <gtest/gtest.h>

#include <limits>

#include <Eigen/Geometry>

#include "common/const_value.h"
#include "process/imu_init_gate.h"

TEST(ImuInitGate, AcceptsRecordedStationaryWindow)
{
    const Eigen::Vector3d acc_std(0.006050680, 0.017762100, 0.011956507);
    const Eigen::Vector3d gyr_std(0.019025700, 0.004430743, 0.006762520);
    const Eigen::Vector3d mean_gyr(-0.014716400, 0.000139534, -0.011522200);

    EXPECT_TRUE(anubis_mapping::is_imu_init_window_stationary(
        acc_std, gyr_std, mean_gyr, 0.05, 0.05, 0.05));
}

TEST(ImuInitGate, RejectsRecordedMovingWindow)
{
    const Eigen::Vector3d acc_std(0.175064291, 0.078738752, 0.224464812);
    const Eigen::Vector3d gyr_std(0.083196539, 0.175913099, 0.031178771);
    const Eigen::Vector3d mean_gyr(-0.001986516, 0.020059980, -0.009096134);

    EXPECT_FALSE(anubis_mapping::is_imu_init_window_stationary(
        acc_std, gyr_std, mean_gyr, 0.05, 0.05, 0.05));
}

TEST(ImuInitGate, RejectsConstantRotationAndInvalidInput)
{
    const Eigen::Vector3d quiet_std(0.001, 0.001, 0.001);
    EXPECT_FALSE(anubis_mapping::is_imu_init_window_stationary(
        quiet_std, quiet_std, Eigen::Vector3d(0.06, 0.0, 0.0),
        0.05, 0.05, 0.05));

    Eigen::Vector3d invalid = quiet_std;
    invalid.x() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(anubis_mapping::is_imu_init_window_stationary(
        invalid, quiet_std, Eigen::Vector3d::Zero(),
        0.05, 0.05, 0.05));
}

TEST(ImuInitGate, RejectsAccelerometerScaleError)
{
    EXPECT_TRUE(anubis_mapping::is_imu_acc_norm_valid(0.9935, 0.05));
    EXPECT_FALSE(anubis_mapping::is_imu_acc_norm_valid(0.90, 0.05));
    EXPECT_FALSE(anubis_mapping::is_imu_acc_norm_valid(std::nan(""), 0.05));
}

TEST(ImuInitGate, PreservesCorrectedAccelerometerNormWithoutBias)
{
    const Eigen::Vector3d mean_acc_g(0.0, 0.0, -0.9935);
    const Eigen::Vector3d zero_bias_g = Eigen::Vector3d::Zero();
    Eigen::Vector3d corrected_acc_g;
    Eigen::Vector3d state_ba_m_s2;
    double corrected_norm_g = 0.0;
    double scale_m_s2_per_g = 0.0;

    ASSERT_TRUE(anubis_mapping::derive_imu_accel_initialization_terms(
        mean_acc_g, zero_bias_g, anubis_mapping::G_m_s2,
        corrected_acc_g, state_ba_m_s2, corrected_norm_g,
        scale_m_s2_per_g));
    EXPECT_NEAR(corrected_norm_g, 0.9935, 1e-12);
    EXPECT_NEAR((corrected_acc_g - mean_acc_g).norm(), 0.0, 1e-12);
    EXPECT_NEAR(state_ba_m_s2.norm(), 0.0, 1e-12);
    EXPECT_NEAR(scale_m_s2_per_g,
        anubis_mapping::G_m_s2 / 0.9935, 1e-12);
    EXPECT_TRUE(anubis_mapping::is_imu_acc_norm_valid(
        mean_acc_g, zero_bias_g, 0.05));
}

TEST(ImuInitGate, SubtractsConfiguredAccelerometerBiasInG)
{
    const Eigen::Vector3d mean_acc_g(0.12, -0.03, -0.96);
    const Eigen::Vector3d initial_ba_g(0.02, -0.01, -0.06);
    const Eigen::Vector3d expected_corrected_g(0.10, -0.02, -0.90);
    Eigen::Vector3d corrected_acc_g;
    Eigen::Vector3d state_ba_m_s2;
    double corrected_norm_g = 0.0;
    double scale_m_s2_per_g = 0.0;

    ASSERT_TRUE(anubis_mapping::derive_imu_accel_initialization_terms(
        mean_acc_g, initial_ba_g, anubis_mapping::G_m_s2,
        corrected_acc_g, state_ba_m_s2, corrected_norm_g,
        scale_m_s2_per_g));
    EXPECT_NEAR((corrected_acc_g - expected_corrected_g).norm(), 0.0, 1e-12);
    EXPECT_NEAR(corrected_norm_g, expected_corrected_g.norm(), 1e-12);
    EXPECT_NEAR((state_ba_m_s2 - initial_ba_g * scale_m_s2_per_g).norm(),
        0.0, 1e-12);
    EXPECT_GT((state_ba_m_s2 - initial_ba_g * anubis_mapping::G_m_s2).norm(),
        1e-6);
    EXPECT_TRUE(anubis_mapping::is_imu_acc_norm_valid(
        mean_acc_g, initial_ba_g, 0.2));

    Eigen::Vector3d direct_corrected_g;
    double direct_norm_g = 0.0;
    ASSERT_TRUE(anubis_mapping::compute_corrected_imu_accel_mean_g(
        mean_acc_g, initial_ba_g, direct_corrected_g, direct_norm_g));
    EXPECT_NEAR((direct_corrected_g - expected_corrected_g).norm(), 0.0,
        1e-12);
    EXPECT_NEAR(direct_norm_g, expected_corrected_g.norm(), 1e-12);
}

TEST(ImuInitGate, KeepsGyroBiasInRadPerSecond)
{
    const Eigen::Vector3d mean_gyr_rad_s(0.010, -0.020, 0.003);
    const Eigen::Vector3d initial_bg_rad_s(0.004, 0.005, -0.001);
    Eigen::Vector3d state_bg_rad_s;

    ASSERT_TRUE(anubis_mapping::compose_imu_gyro_bias_rad_s(
        mean_gyr_rad_s, initial_bg_rad_s, state_bg_rad_s));
    const Eigen::Vector3d expected = mean_gyr_rad_s + initial_bg_rad_s;
    EXPECT_NEAR((state_bg_rad_s - expected).norm(), 0.0, 1e-12);
    EXPECT_NEAR(state_bg_rad_s.x(), 0.014, 1e-12);
    EXPECT_NEAR(state_bg_rad_s.y(), -0.015, 1e-12);
    EXPECT_NEAR(state_bg_rad_s.z(), 0.002, 1e-12);
    EXPECT_NE(state_bg_rad_s.x(), expected.x() * anubis_mapping::G_m_s2);
}

TEST(ImuInitGate, RejectsNonFiniteGyroBiasCorrection)
{
    const Eigen::Vector3d mean_gyr_rad_s = Eigen::Vector3d::Zero();
    Eigen::Vector3d invalid_bias = Eigen::Vector3d::Zero();
    invalid_bias.z() = std::numeric_limits<double>::infinity();
    Eigen::Vector3d state_bg_rad_s;

    EXPECT_FALSE(anubis_mapping::compose_imu_gyro_bias_rad_s(
        mean_gyr_rad_s, invalid_bias, state_bg_rad_s));
}

TEST(ImuInitGate, UsesPositiveZForGravityAlignmentTarget)
{
    const Eigen::Vector3d corrected_mean_acc_g(-0.31, -0.02, 0.94);
    Eigen::Vector3d gravity_target_g;

    ASSERT_TRUE(anubis_mapping::make_imu_gravity_alignment_target_g(
        corrected_mean_acc_g, gravity_target_g));
    EXPECT_NEAR(gravity_target_g.x(), 0.0, 1e-12);
    EXPECT_NEAR(gravity_target_g.y(), 0.0, 1e-12);
    EXPECT_GT(gravity_target_g.z(), 0.0);
    EXPECT_NEAR(gravity_target_g.z(), corrected_mean_acc_g.norm(), 1e-12);

    const Eigen::Matrix3d alignment =
        Eigen::Quaterniond::FromTwoVectors(
            corrected_mean_acc_g, gravity_target_g).toRotationMatrix();
    EXPECT_NEAR((alignment * corrected_mean_acc_g - gravity_target_g).norm(),
        0.0, 1e-12);
}

TEST(ImuInitGate, RejectsNonFiniteOrDegenerateBiasInputs)
{
    const Eigen::Vector3d valid_acc_g(0.0, 0.0, -1.0);
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    Eigen::Vector3d corrected_acc_g;
    Eigen::Vector3d state_ba_m_s2;
    double corrected_norm_g = 0.0;
    double scale_m_s2_per_g = 0.0;

    Eigen::Vector3d nan_bias = zero;
    nan_bias.x() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(anubis_mapping::derive_imu_accel_initialization_terms(
        valid_acc_g, nan_bias, anubis_mapping::G_m_s2,
        corrected_acc_g, state_ba_m_s2, corrected_norm_g,
        scale_m_s2_per_g));

    EXPECT_FALSE(anubis_mapping::derive_imu_accel_initialization_terms(
        zero, zero, anubis_mapping::G_m_s2,
        corrected_acc_g, state_ba_m_s2, corrected_norm_g,
        scale_m_s2_per_g));

    EXPECT_FALSE(anubis_mapping::make_imu_gravity_alignment_target_g(
        zero, corrected_acc_g));

    Eigen::Vector3d nan_gyr = zero;
    nan_gyr.y() = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d state_bg_rad_s;
    EXPECT_FALSE(anubis_mapping::compose_imu_gyro_bias_rad_s(
        nan_gyr, zero, state_bg_rad_s));
}
