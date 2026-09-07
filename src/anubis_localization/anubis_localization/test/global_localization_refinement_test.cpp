#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "localization/global_localization_refinement.hpp"

namespace localization {
namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr makeAsymmetricCloud() {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZI>());
  for (int x = 0; x < 4; ++x) {
    for (int y = 0; y < 3; ++y) {
      pcl::PointXYZI point;
      point.x = static_cast<float>(0.4 * x + 0.03 * y * y);
      point.y = static_cast<float>(0.35 * y + 0.05 * x);
      point.z = static_cast<float>(0.08 * x * y + 0.02 * x);
      point.intensity = static_cast<float>(x + y);
      cloud->push_back(point);
    }
  }
  return cloud;
}

TEST(GlobalLocalizationRefinement, UsesCoarsePoseButMeasuresFromOriginalSeed) {
  const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = makeAsymmetricCloud();
  GLParams params;
  params.refine_max_iter = 20;
  params.refine_fitness_threshold = 0.25;

  GlobalPoseCandidate candidate;
  candidate.seed_pose(0, 3) = 10.0;
  const Eigen::Matrix4d coarse_pose = Eigen::Matrix4d::Identity();

  ASSERT_TRUE(runDoubleIcpRefinement(
      cloud, cloud, coarse_pose, params, candidate));
  ASSERT_TRUE(candidate.converged);
  EXPECT_TRUE(candidate.final_pose.isApprox(Eigen::Matrix4d::Identity(), 1e-5));
  EXPECT_NEAR(candidate.correction_xy, 10.0, 1e-5);
  EXPECT_NEAR(candidate.correction_z, 0.0, 1e-5);
  EXPECT_NEAR(candidate.yaw_delta_deg, 0.0, 1e-4);
}

TEST(GlobalLocalizationRefinement, BatchRefinementStartsFromEachCoarseFinalPose) {
  const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = makeAsymmetricCloud();
  GLParams params;
  params.refine_leaf_src = 0.01;
  params.refine_max_iter = 20;

  GlobalPoseCandidate candidate;
  candidate.seed_pose(0, 3) = 8.0;
  candidate.final_pose = Eigen::Matrix4d::Identity();
  std::vector<GlobalPoseCandidate> candidates = {candidate};
  GlobalRefinementStats stats;

  ASSERT_TRUE(runLevel1Refinement(cloud, cloud, params, candidates, stats));
  ASSERT_EQ(stats.attempted, 1);
  ASSERT_EQ(stats.converged, 1);
  ASSERT_EQ(stats.completed, 1);
  EXPECT_TRUE(candidates.front().final_pose.isApprox(
      Eigen::Matrix4d::Identity(), 1e-5));
  EXPECT_NEAR(candidates.front().correction_xy, 8.0, 1e-5);
}

TEST(GlobalLocalizationRefinement, InvalidStartPoseFailsClosed) {
  const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = makeAsymmetricCloud();
  Eigen::Matrix4d invalid_pose = Eigen::Matrix4d::Identity();
  invalid_pose(0, 0) = std::numeric_limits<double>::quiet_NaN();
  GlobalPoseCandidate candidate;

  EXPECT_FALSE(runDoubleIcpRefinement(
      cloud, cloud, invalid_pose, GLParams{}, candidate));
  EXPECT_FALSE(candidate.converged);
}

TEST(GlobalLocalizationRefinement, InvalidThresholdFailsClosed) {
  const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = makeAsymmetricCloud();
  GLParams params;
  params.refine_fitness_threshold =
      std::numeric_limits<double>::quiet_NaN();
  GlobalPoseCandidate candidate;

  EXPECT_FALSE(runDoubleIcpRefinement(
      cloud, cloud, Eigen::Matrix4d::Identity(), params, candidate));
  EXPECT_FALSE(candidate.converged);
}

}  // namespace
}  // namespace localization
