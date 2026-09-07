#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

#include <Eigen/Geometry>

#include "localization/level1_coarse_search.hpp"

namespace localization {
namespace {

pcl::PointXYZI makePoint(float x, float y, float z) {
  pcl::PointXYZI point;
  point.x = x;
  point.y = y;
  point.z = z;
  point.intensity = 0.0f;
  return point;
}

Level0Hypothesis makeRegion(int cell, int yaw_bin, double yaw_deg, double x) {
  Level0Hypothesis region;
  region.grid_cell_id = cell;
  region.yaw_bin = yaw_bin;
  region.yaw_deg = yaw_deg;
  region.T_map_level(0, 3) = x;
  region.score.valid = true;
  region.score.score = 0.2;
  region.score.mean_truncated_distance = 0.2;
  region.score.valid_projection_ratio = 0.9;
  region.score.input_point_count = 100;
  region.score.valid_projection_points = 90;
  return region;
}

GLParams level1Params() {
  GLParams params;
  params.level0_yaw_samples = 4;
  params.level1_search_radius_xy = 2.0;
  params.level1_seed_stride_xy = 1.5;
  params.level1_yaw_step_deg = 15.0;
  params.max_candidates_total = 80;
  params.max_refine_candidates = 4;
  return params;
}

GlobalPoseCandidate coarseCandidate(int cell,
                                    int yaw_bin,
                                    double overlap,
                                    double p90,
                                    double score) {
  GlobalPoseCandidate candidate;
  candidate.grid_cell_id = cell;
  candidate.yaw_bin = yaw_bin;
  candidate.support_group = std::to_string(cell) + ":" + std::to_string(score);
  candidate.converged = true;
  candidate.overlap_ratio = overlap;
  candidate.inlier_count = 200;
  candidate.p90_residual = p90;
  candidate.coarse_score = score;
  return candidate;
}

TEST(Level1CoarseSearch, DefaultYawOffsetsCoverBothBinEdgesAndZero) {
  std::vector<double> offsets;
  ASSERT_TRUE(makeLevel1YawOffsets(level1Params(), offsets));
  const std::vector<double> expected = {-45.0, -30.0, -15.0, 0.0,
                                         15.0,  30.0,  45.0};
  EXPECT_EQ(offsets, expected);
}

TEST(Level1CoarseSearch, SeedGenerationKeepsAllRegionsAndDynamicBatches) {
  GLParams params = level1Params();
  std::vector<GlobalPoseCandidate> seeds;
  Level1GenerationStats stats;
  GLSummaryCounts summary;
  const std::vector<Level0Hypothesis> regions = {
      makeRegion(2, 0, 0.0, 0.0), makeRegion(7, 1, 90.0, 10.0)};

  ASSERT_TRUE(generateLevel1Seeds(
      regions, Eigen::Matrix3d::Identity(), 70, "anchor:1", params,
      seeds, stats, summary));

  EXPECT_EQ(stats.translation_template_count_per_region, 5);
  EXPECT_EQ(stats.yaw_count_per_template, 7);
  EXPECT_EQ(stats.seed_count, 70);
  EXPECT_EQ(stats.batch_count, 7);
  EXPECT_EQ(summary.level1_seed_count, 70);
  EXPECT_EQ(summary.level1_batch_count, 7);
  ASSERT_EQ(seeds.size(), 70U);
  EXPECT_EQ(seeds.front().source_type, "grid");
  EXPECT_EQ(seeds.front().evidence_id, "anchor:1");
  EXPECT_EQ(seeds.front().evidence_role, "anchor_recall");
  EXPECT_EQ(seeds.front().grid_cell_id, 2);
  EXPECT_DOUBLE_EQ(seeds.front().coarse_score, 0.2);
  EXPECT_EQ(seeds.back().rotation_batch, 6);

  std::set<std::string> support_groups;
  for (const GlobalPoseCandidate& seed : seeds) {
    support_groups.insert(seed.support_group);
  }
  EXPECT_EQ(support_groups.size(), 10U);
}

TEST(Level1CoarseSearch, BaseSeedRestoresGravityTiltInDocumentedOrder) {
  const GLParams params = level1Params();
  const Eigen::Matrix3d R_level_base = Eigen::AngleAxisd(
      -0.1, Eigen::Vector3d::UnitX()).toRotationMatrix();
  std::vector<GlobalPoseCandidate> seeds;
  Level1GenerationStats stats;
  GLSummaryCounts summary;
  const Level0Hypothesis region = makeRegion(0, 0, 0.0, 3.0);

  ASSERT_TRUE(generateLevel1Seeds(
      {region}, R_level_base, 0, "anchor:2", params,
      seeds, stats, summary));
  ASSERT_GE(seeds.size(), 4U);
  const Eigen::Matrix3d actual_rotation =
      seeds[3].seed_pose.block<3, 3>(0, 0);
  const Eigen::Matrix3d expected_rotation =
      region.T_map_level.block<3, 3>(0, 0) * R_level_base;
  EXPECT_TRUE(actual_rotation.isApprox(expected_rotation, 1e-12));
}

TEST(Level1CoarseSearch, CoarseSelectionPreservesRegionDiversity) {
  GLParams params = level1Params();
  params.coarse_min_overlap = 0.5;
  params.coarse_min_inliers = 100;
  params.coarse_max_p90_residual = 1.0;
  params.max_refine_candidates = 3;
  const std::vector<GlobalPoseCandidate> candidates = {
      coarseCandidate(0, 0, 0.95, 0.2, 0.1),
      coarseCandidate(0, 0, 0.94, 0.2, 0.1),
      coarseCandidate(1, 0, 0.80, 0.3, 0.2),
      coarseCandidate(2, 1, 0.70, 0.4, 0.3)};
  std::vector<GlobalPoseCandidate> selected;
  Level1CoarseStats stats;

  ASSERT_TRUE(selectLevel1RefineCandidates(
      candidates, params, selected, stats));
  ASSERT_EQ(selected.size(), 3U);
  std::set<std::pair<int, int>> regions;
  for (const GlobalPoseCandidate& candidate : selected) {
    regions.emplace(candidate.grid_cell_id, candidate.yaw_bin);
  }
  EXPECT_EQ(regions.size(), 3U);
  EXPECT_EQ(stats.selected_for_refine, 3);
}

TEST(Level1CoarseSearch, IdentityCloudPassesCoarseMetrics) {
  GLParams params = level1Params();
  params.coarse_leaf_src = 0.1;
  params.coarse_max_iter = 10;
  params.coarse_roi_radius = 10.0;
  params.coarse_max_corr_dist = 1.0;
  params.coarse_min_overlap = 0.9;
  params.coarse_min_inliers = 5;
  params.coarse_max_p90_residual = 0.01;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZI>());
  for (int index = 0; index < 10; ++index) {
    cloud->push_back(makePoint(static_cast<float>(index % 5),
                               static_cast<float>(index / 5),
                               static_cast<float>((index % 3) * 0.2)));
  }
  std::vector<GlobalPoseCandidate> candidates(1);
  candidates.front().seed_pose = Eigen::Matrix4d::Identity();
  candidates.front().final_pose = Eigen::Matrix4d::Identity();
  Level1CoarseStats stats;

  ASSERT_TRUE(runLevel1CoarseAlignment(
      cloud, cloud, params, candidates, stats));
  EXPECT_TRUE(candidates.front().converged);
  EXPECT_EQ(candidates.front().inlier_count, 10);
  EXPECT_NEAR(candidates.front().overlap_ratio, 1.0, 1e-12);
  EXPECT_NEAR(candidates.front().p90_residual, 0.0, 1e-6);
  EXPECT_EQ(stats.passed_metrics, 1);
}

}  // namespace
}  // namespace localization
