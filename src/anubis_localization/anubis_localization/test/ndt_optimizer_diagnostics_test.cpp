#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <localization/ndt_optimizer_diagnostics.hpp>

namespace localization {
namespace {

TEST(NdtOptimizerDiagnostics, ReportsConditionAndWeakestDirection) {
  Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
  hessian.diagonal() << -100.0, -50.0, -25.0, -10.0, -5.0, -1.0;

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses(3);
  poses[0] = Eigen::Matrix4f::Identity();
  poses[1] = Eigen::Matrix4f::Identity();
  poses[1](0, 3) = 0.1f;
  poses[2] = Eigen::Matrix4f::Identity();
  poses[2](0, 3) = 0.2f;

  const NdtOptimizerDiagnostics diagnostics = analyzeNdtOptimizer(hessian, poses);

  ASSERT_TRUE(diagnostics.hessian_valid);
  EXPECT_EQ(diagnostics.hessian_negative, 6);
  EXPECT_EQ(diagnostics.hessian_positive, 0);
  EXPECT_EQ(diagnostics.hessian_near_zero, 0);
  EXPECT_NEAR(diagnostics.hessian_abs_condition, 100.0, 1e-9);
  EXPECT_NEAR(std::abs(diagnostics.weakest_direction[5]), 1.0, 1e-9);

  ASSERT_TRUE(diagnostics.trajectory_valid);
  EXPECT_NEAR(diagnostics.iteration_translation_path_m, 0.2, 1e-6);
  EXPECT_NEAR(diagnostics.net_translation_m, 0.2, 1e-6);
  EXPECT_NEAR(diagnostics.translation_path_ratio, 1.0, 1e-6);
  EXPECT_NEAR(diagnostics.final_step_translation_m, 0.1, 1e-6);
  EXPECT_EQ(diagnostics.transformation_count, 3U);
  EXPECT_NEAR(diagnostics.initial_xyz[0], 0.0, 1e-6);
  EXPECT_NEAR(diagnostics.final_xyz[0], 0.2, 1e-6);
  EXPECT_NEAR(diagnostics.correction_xyz_local[0], 0.2, 1e-6);
  EXPECT_NEAR(diagnostics.correction_rpy_deg[2], 0.0, 1e-6);
}

TEST(NdtOptimizerDiagnostics, ReportsSixDegreeCorrection) {
  Eigen::Matrix<double, 6, 6> hessian =
    -Eigen::Matrix<double, 6, 6>::Identity();
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses(2);
  poses[0] = Eigen::Matrix4f::Identity();
  poses[0](0, 3) = 10.0f;
  poses[1] = poses[0];
  poses[1].block<3, 3>(0, 0) =
    Eigen::AngleAxisf(
      static_cast<float>(10.0 * 3.14159265358979323846 / 180.0),
      Eigen::Vector3f::UnitZ()).toRotationMatrix();
  poses[1](1, 3) = 0.25f;

  const NdtOptimizerDiagnostics diagnostics = analyzeNdtOptimizer(hessian, poses);

  ASSERT_TRUE(diagnostics.trajectory_valid);
  EXPECT_NEAR(diagnostics.initial_xyz[0], 10.0, 1e-6);
  EXPECT_NEAR(diagnostics.final_xyz[1], 0.25, 1e-6);
  EXPECT_NEAR(diagnostics.correction_xyz_local[1], 0.25, 1e-6);
  EXPECT_NEAR(diagnostics.correction_rpy_deg[2], 10.0, 1e-4);
}

TEST(NdtOptimizerDiagnostics, MarksSingularHessian) {
  Eigen::Matrix<double, 6, 6> hessian = -Eigen::Matrix<double, 6, 6>::Identity();
  hessian(2, 2) = 0.0;

  const NdtOptimizerDiagnostics diagnostics = analyzeNdtOptimizer(hessian, {});

  ASSERT_TRUE(diagnostics.hessian_valid);
  EXPECT_EQ(diagnostics.hessian_near_zero, 1);
  EXPECT_TRUE(std::isinf(diagnostics.hessian_abs_condition));
  EXPECT_NEAR(std::abs(diagnostics.weakest_direction[2]), 1.0, 1e-9);
  EXPECT_FALSE(diagnostics.trajectory_valid);
}

}  // namespace
}  // namespace localization
