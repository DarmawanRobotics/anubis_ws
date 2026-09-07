#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "localization/level0_distance_field.hpp"

namespace localization {
namespace {

GLParams testParams() {
  GLParams params;
  params.flat_single_level_map = true;
  params.level0_min_valid_points = 3;
  params.level0_max_scan_points = 10;
  params.level0_min_range = 0.1;
  params.level0_max_range = 20.0;
  return params;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr makeMap() {
  pcl::PointCloud<pcl::PointXYZI>::Ptr map(new pcl::PointCloud<pcl::PointXYZI>());
  const std::vector<Eigen::Vector2f> obstacles = {
      {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
  for (const auto& point : obstacles) {
    pcl::PointXYZI ground;
    ground.x = point.x();
    ground.y = point.y();
    ground.z = 0.0f;
    map->push_back(ground);
    pcl::PointXYZI obstacle = ground;
    obstacle.z = 0.5f;
    map->push_back(obstacle);
  }
  return map;
}

pcl::PointXYZI makePoint(float x, float y, float z) {
  pcl::PointXYZI point;
  point.x = x;
  point.y = y;
  point.z = z;
  point.intensity = 0.0f;
  return point;
}

TEST(Level0DistanceField, BuildsFlatMapAndScoresMatchingHypothesis) {
  const GLParams params = testParams();
  Level0DistanceField field;
  ASSERT_TRUE(field.build(makeMap(), params));
  EXPECT_FALSE(field.empty());
  EXPECT_NEAR(field.floorZ(), 0.0, 1e-12);

  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  for (const auto& point : std::vector<Eigen::Vector2f>{{0.2f, 0.2f}, {1.0f, 0.0f},
                                                        {0.0f, 1.0f}}) {
    scan->push_back(makePoint(point.x(), point.y(), 0.5f));
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr scan_level;
  Level0ScanPreparationStats preparation;
  ASSERT_TRUE(field.prepareScan(scan, Eigen::Matrix3d::Identity(), 0.0, params,
                                scan_level, preparation));
  EXPECT_EQ(preparation.sampled_points, 3);
  EXPECT_EQ(scan_level->width, 3U);
  EXPECT_EQ(scan_level->height, 1U);

  Level0Score score;
  ASSERT_TRUE(field.scoreHypothesis(scan_level, Eigen::Matrix4d::Identity(), params, score));
  EXPECT_TRUE(score.valid);
  EXPECT_EQ(score.valid_projection_points, 3);
  EXPECT_EQ(score.invalid_projection_points, 0);
  EXPECT_EQ(score.boundary_unknown_points, 0);
  EXPECT_NEAR(score.score, 0.0, 1e-12);
}

TEST(Level0DistanceField, HeightAndRangeFiltersRejectInvalidScanPoints) {
  const GLParams params = testParams();
  Level0DistanceField field;
  ASSERT_TRUE(field.build(makeMap(), params));
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  scan->push_back(makePoint(1.0f, 0.0f, 0.5f));
  scan->push_back(makePoint(1.0f, 0.0f, 2.0f));
  scan->push_back(makePoint(0.01f, 0.0f, 0.5f));
  pcl::PointXYZI nan_point = makePoint(NAN, 0.0f, 0.5f);
  scan->push_back(nan_point);

  pcl::PointCloud<pcl::PointXYZI>::Ptr scan_level;
  Level0ScanPreparationStats preparation;
  EXPECT_FALSE(field.prepareScan(scan, Eigen::Matrix3d::Identity(), 0.0, params,
                                 scan_level, preparation));
  EXPECT_EQ(preparation.input_points, 4);
  EXPECT_EQ(preparation.finite_points, 3);
  EXPECT_EQ(preparation.height_points, 2);
  EXPECT_EQ(preparation.range_points, 1);
  EXPECT_EQ(preparation.sampled_points, 1);
}

TEST(Level0DistanceField, SamplingIsDeterministicAndBounded) {
  GLParams params = testParams();
  params.level0_max_scan_points = 5;
  Level0DistanceField field;
  ASSERT_TRUE(field.build(makeMap(), params));
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  for (int index = 0; index < 20; ++index) {
    const double angle = 2.0 * 3.14159265358979323846 * index / 20.0;
    scan->push_back(makePoint(
        static_cast<float>(2.0 * std::cos(angle)),
        static_cast<float>(2.0 * std::sin(angle)), 0.5f));
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr first;
  pcl::PointCloud<pcl::PointXYZI>::Ptr second;
  Level0ScanPreparationStats first_stats;
  Level0ScanPreparationStats second_stats;
  ASSERT_TRUE(field.prepareScan(scan, Eigen::Matrix3d::Identity(), 0.0, params,
                                first, first_stats));
  ASSERT_TRUE(field.prepareScan(scan, Eigen::Matrix3d::Identity(), 0.0, params,
                                second, second_stats));
  ASSERT_EQ(first->size(), 5U);
  ASSERT_EQ(second->size(), first->size());
  for (size_t index = 0; index < first->size(); ++index) {
    EXPECT_FLOAT_EQ((*first)[index].x, (*second)[index].x);
    EXPECT_FLOAT_EQ((*first)[index].y, (*second)[index].y);
    EXPECT_FLOAT_EQ((*first)[index].z, (*second)[index].z);
  }
}

TEST(Level0DistanceField, BoundaryUnknownAndInvalidPointsAreSeparated) {
  GLParams params = testParams();
  params.level0_boundary_padding_m = 1.0;
  params.level0_max_boundary_unknown_ratio = 0.9;
  Level0DistanceField field;
  ASSERT_TRUE(field.build(makeMap(), params));
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  scan->push_back(makePoint(0.0f, 0.0f, 0.5f));
  scan->push_back(makePoint(1.5f, 1.0f, 0.5f));
  scan->push_back(makePoint(4.0f, 1.0f, 0.5f));

  Level0Score score;
  ASSERT_TRUE(field.scoreHypothesis(scan, Eigen::Matrix4d::Identity(), params, score));
  EXPECT_EQ(score.valid_projection_points, 1);
  EXPECT_EQ(score.boundary_unknown_points, 1);
  EXPECT_EQ(score.invalid_projection_points, 1);
  EXPECT_EQ(score.lookup_count, static_cast<uint64_t>(3));
  EXPECT_FALSE(score.valid);
}

TEST(Level0DistanceField, KnownDomainUsesFiniteMapAabbNotObstacleAabb) {
  GLParams params = testParams();
  params.level0_min_valid_points = 1;
  pcl::PointCloud<pcl::PointXYZI>::Ptr map = makeMap();
  map->push_back(makePoint(5.0f, 5.0f, 0.0f));
  Level0DistanceField field;
  ASSERT_TRUE(field.build(map, params));
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  scan->push_back(makePoint(4.0f, 4.0f, 0.5f));
  Level0Score score;

  ASSERT_TRUE(field.scoreHypothesis(scan, Eigen::Matrix4d::Identity(), params,
                                    score));
  EXPECT_EQ(score.valid_projection_points, 1);
  EXPECT_EQ(score.boundary_unknown_points, 0);
  EXPECT_EQ(score.invalid_projection_points, 0);
}

TEST(Level0DistanceField, ValidScorePopulatesCandidateAndSummary) {
  Level0Score score;
  score.valid = true;
  score.input_point_count = 10;
  score.valid_projection_points = 7;
  score.boundary_unknown_points = 2;
  score.invalid_projection_points = 1;
  score.lookup_count = 10;
  score.valid_projection_ratio = 0.875;
  score.mean_truncated_distance = 0.25;
  score.score = 0.625;

  GlobalPoseCandidate candidate;
  GLSummaryCounts summary;
  EXPECT_TRUE(applyLevel0ScoreToCandidate(score, candidate));
  accumulateLevel0Summary(score, summary);

  EXPECT_DOUBLE_EQ(candidate.coarse_score, 0.625);
  EXPECT_DOUBLE_EQ(candidate.level0_mean_truncated_distance, 0.25);
  EXPECT_DOUBLE_EQ(candidate.level0_valid_projection_ratio, 0.875);
  EXPECT_EQ(candidate.level0_input_points, 10);
  EXPECT_EQ(candidate.level0_valid_points, 7);
  EXPECT_EQ(summary.level0_input_points, 10);
  EXPECT_EQ(summary.level0_valid_points, 7);
  EXPECT_EQ(summary.level0_boundary_unknown_points, 2);
  EXPECT_EQ(summary.level0_invalid_projection_points, 1);
  EXPECT_EQ(summary.level0_lookup_count, 10U);
  EXPECT_EQ(summary.level0_invalid_hypotheses, 0);
}

TEST(Level0DistanceField, InvalidScoreCannotEnterCandidateRanking) {
  Level0Score score;
  score.input_point_count = 5;
  score.valid_projection_points = 1;
  score.boundary_unknown_points = 1;
  score.invalid_projection_points = 3;
  score.lookup_count = 5;
  score.valid_projection_ratio = 0.25;
  score.mean_truncated_distance = 0.1;
  score.score = 0.2;

  GlobalPoseCandidate candidate;
  GLSummaryCounts summary;
  EXPECT_FALSE(applyLevel0ScoreToCandidate(score, candidate));
  accumulateLevel0Summary(score, summary);

  EXPECT_DOUBLE_EQ(candidate.coarse_score, 1e9);
  EXPECT_DOUBLE_EQ(candidate.level0_mean_truncated_distance, 0.1);
  EXPECT_EQ(candidate.level0_input_points, 5);
  EXPECT_EQ(candidate.level0_valid_points, 1);
  EXPECT_EQ(summary.level0_lookup_count, 5U);
  EXPECT_EQ(summary.level0_invalid_hypotheses, 1);
}

}  // namespace
}  // namespace localization
