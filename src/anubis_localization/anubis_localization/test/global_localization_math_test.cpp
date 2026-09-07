#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "localization/global_localization_math.hpp"

namespace localization {
namespace {

Eigen::Matrix4d makePose(double x, double y, double z, double yaw_deg) {
  constexpr double kPi = 3.14159265358979323846;
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  const double yaw_rad = yaw_deg * kPi / 180.0;
  pose.block<3, 3>(0, 0) =
    Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  pose.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
  return pose;
}

void expectMatrixNear(const Eigen::Matrix4d& actual,
                      const Eigen::Matrix4d& expected,
                      double tolerance = 1e-12) {
  EXPECT_TRUE(actual.isApprox(expected, tolerance))
    << "actual:\n" << actual << "\nexpected:\n" << expected;
}

TEST(GlobalLocalizationMath, IdentityCorrectionsPreserveSeedPose) {
  const Eigen::Matrix4d seed_pose = makePose(12.0, -4.5, 0.3, 135.0);

  const Eigen::Matrix4d actual = global_localization_math::composeAbsolutePose(
    Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), seed_pose);

  expectMatrixNear(actual, seed_pose);
}

TEST(GlobalLocalizationMath, CorrectionsAreAppliedToNonOriginSeedInCloudOrder) {
  const Eigen::Matrix4d seed_pose = makePose(10.0, 3.0, 0.2, 90.0);
  const Eigen::Matrix4d first_correction = makePose(0.5, -0.2, 0.0, 10.0);
  const Eigen::Matrix4d second_correction = makePose(-0.1, 0.4, 0.0, -3.0);
  const Eigen::Matrix4d expected = second_correction * first_correction * seed_pose;

  const Eigen::Matrix4d actual = global_localization_math::composeAbsolutePose(
    second_correction, first_correction, seed_pose);

  expectMatrixNear(actual, expected);
  EXPECT_FALSE(actual.isApprox(second_correction * first_correction, 1e-12));
}

TEST(GlobalLocalizationMath, CompositionOrderMatchesSequentialPointTransforms) {
  const Eigen::Matrix4d seed_pose = makePose(5.0, -2.0, 0.0, 45.0);
  const Eigen::Matrix4d first_correction = makePose(0.3, 0.1, 0.0, -8.0);
  const Eigen::Matrix4d second_correction = makePose(-0.2, 0.5, 0.0, 4.0);
  const Eigen::Vector4d point_base(1.2, -0.7, 0.4, 1.0);

  const Eigen::Vector4d sequential =
    second_correction * (first_correction * (seed_pose * point_base));
  const Eigen::Vector4d composed = global_localization_math::composeAbsolutePose(
    second_correction, first_correction, seed_pose) * point_base;

  EXPECT_TRUE(composed.isApprox(sequential, 1e-12));
}

TEST(GlobalLocalizationMath, WrapAngleDegUsesSignedShortestDifference) {
  EXPECT_DOUBLE_EQ(global_localization_math::wrapAngleDeg(0.0), 0.0);
  EXPECT_DOUBLE_EQ(global_localization_math::wrapAngleDeg(181.0), -179.0);
  EXPECT_DOUBLE_EQ(global_localization_math::wrapAngleDeg(-181.0), 179.0);
  EXPECT_DOUBLE_EQ(global_localization_math::wrapAngleDeg(540.0), -180.0);
}

}  // namespace
}  // namespace localization
