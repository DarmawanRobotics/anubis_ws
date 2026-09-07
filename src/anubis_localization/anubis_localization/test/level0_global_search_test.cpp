#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "localization/level0_global_search.hpp"

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

pcl::PointCloud<pcl::PointXYZI>::Ptr makeSearchMap() {
  pcl::PointCloud<pcl::PointXYZI>::Ptr map(new pcl::PointCloud<pcl::PointXYZI>());
  map->push_back(makePoint(0.0f, 0.0f, 0.0f));
  map->push_back(makePoint(4.0f, 0.0f, 0.0f));
  map->push_back(makePoint(0.0f, 4.0f, 0.0f));
  map->push_back(makePoint(4.0f, 4.0f, 0.0f));
  map->push_back(makePoint(1.0f, 0.0f, 0.5f));
  map->push_back(makePoint(0.0f, 2.0f, 0.5f));
  map->push_back(makePoint(3.0f, 1.0f, 0.5f));
  return map;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr makeSearchScan() {
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  scan->push_back(makePoint(1.0f, 0.0f, 0.5f));
  scan->push_back(makePoint(0.0f, 2.0f, 0.5f));
  scan->push_back(makePoint(3.0f, 1.0f, 0.5f));
  return scan;
}

GLParams searchParams() {
  GLParams params;
  params.flat_single_level_map = true;
  params.level0_grid_stride_xy = 2.0;
  params.level0_yaw_samples = 4;
  params.level0_top_regions = 6;
  params.level0_distance_field_resolution = 0.25;
  params.level0_min_valid_points = 3;
  params.level0_min_valid_projection_ratio = 1.0;
  params.level0_boundary_padding_m = 0.0;
  params.max_candidates_total = 10;
  return params;
}

Level0Hypothesis makeHypothesis(double score,
                                double x,
                                double y,
                                double yaw_deg,
                                int grid_cell_id,
                                int yaw_bin) {
  Level0Hypothesis hypothesis;
  hypothesis.score.valid = true;
  hypothesis.score.score = score;
  hypothesis.score.valid_projection_ratio = 1.0;
  hypothesis.T_map_level(0, 3) = x;
  hypothesis.T_map_level(1, 3) = y;
  hypothesis.yaw_deg = yaw_deg;
  hypothesis.grid_cell_id = grid_cell_id;
  hypothesis.yaw_bin = yaw_bin;
  return hypothesis;
}

TEST(Level0GlobalSearch, CoversFullAabbAndReportsLookupBound) {
  GLParams params = searchParams();
  Level0DistanceField distance_field;
  ASSERT_TRUE(distance_field.build(makeSearchMap(), params));
  std::vector<Level0Hypothesis> top_regions;
  Level0SearchStats stats;
  GLSummaryCounts summary;

  ASSERT_TRUE(runLevel0GlobalSearch(distance_field, makeSearchScan(), params,
                                    top_regions, stats, summary));

  EXPECT_TRUE(stats.complete);
  EXPECT_EQ(stats.grid_cell_count, 9);
  EXPECT_EQ(stats.hypothesis_count, 36U);
  EXPECT_EQ(stats.lookup_count_upper_bound, 108U);
  EXPECT_EQ(stats.batch_count, 4);
  EXPECT_EQ(stats.valid_hypothesis_count + stats.invalid_hypothesis_count, 36);
  EXPECT_EQ(summary.covered_cells, 9);
  EXPECT_EQ(summary.total_cells, 9);
  EXPECT_EQ(summary.level0_lookup_count, 108U);
  EXPECT_EQ(summary.batch_index, 3);
  EXPECT_LE(top_regions.size(), 6U);
}

TEST(Level0GlobalSearch, MatchingOriginHypothesisIsBest) {
  GLParams params = searchParams();
  params.level0_region_nms_xy = 0.5;
  Level0DistanceField distance_field;
  ASSERT_TRUE(distance_field.build(makeSearchMap(), params));
  std::vector<Level0Hypothesis> top_regions;
  Level0SearchStats stats;
  GLSummaryCounts summary;

  ASSERT_TRUE(runLevel0GlobalSearch(distance_field, makeSearchScan(), params,
                                    top_regions, stats, summary));
  ASSERT_FALSE(top_regions.empty());
  EXPECT_EQ(top_regions.front().grid_cell_id, 0);
  EXPECT_EQ(top_regions.front().yaw_bin, 0);
  EXPECT_DOUBLE_EQ(top_regions.front().score.score, 0.0);
}

TEST(Level0GlobalSearch, NmsRequiresBothSpatialAndYawProximity) {
  GLParams params = searchParams();
  params.level0_top_regions = 4;
  params.level0_region_nms_xy = 2.0;
  params.level0_region_nms_yaw_deg = 30.0;
  const std::vector<Level0Hypothesis> hypotheses = {
      makeHypothesis(0.1, 0.0, 0.0, 0.0, 0, 0),
      makeHypothesis(0.2, 1.0, 0.0, 10.0, 1, 0),
      makeHypothesis(0.3, 1.0, 0.0, 90.0, 1, 1),
      makeHypothesis(0.4, 5.0, 0.0, 0.0, 2, 0)};
  std::vector<Level0Hypothesis> selected;

  ASSERT_TRUE(selectLevel0TopRegions(hypotheses, params, selected));
  ASSERT_EQ(selected.size(), 3U);
  EXPECT_DOUBLE_EQ(selected[0].score.score, 0.1);
  EXPECT_DOUBLE_EQ(selected[1].score.score, 0.3);
  EXPECT_DOUBLE_EQ(selected[2].score.score, 0.4);
}

TEST(Level0GlobalSearch, RankingIsDeterministicAcrossInputOrder) {
  const GLParams params = searchParams();
  std::vector<Level0Hypothesis> hypotheses = {
      makeHypothesis(0.2, 4.0, 0.0, 0.0, 2, 0),
      makeHypothesis(0.2, 0.0, 0.0, 90.0, 0, 1),
      makeHypothesis(0.2, 0.0, 0.0, 0.0, 0, 0)};
  std::vector<Level0Hypothesis> first;
  std::vector<Level0Hypothesis> second;
  ASSERT_TRUE(selectLevel0TopRegions(hypotheses, params, first));
  std::reverse(hypotheses.begin(), hypotheses.end());
  ASSERT_TRUE(selectLevel0TopRegions(hypotheses, params, second));

  ASSERT_EQ(first.size(), second.size());
  for (size_t index = 0; index < first.size(); ++index) {
    EXPECT_EQ(first[index].grid_cell_id, second[index].grid_cell_id);
    EXPECT_EQ(first[index].yaw_bin, second[index].yaw_bin);
  }
}

TEST(Level0GlobalSearch, UncalibratedGroundOffsetRemainsShadowDiagnostic) {
  GLParams params = searchParams();
  params.level0_base_ground_offset = -1.0;
  Level0DistanceField distance_field;
  ASSERT_TRUE(distance_field.build(makeSearchMap(), params));
  std::vector<Level0Hypothesis> top_regions;
  Level0SearchStats stats;
  GLSummaryCounts summary;

  ASSERT_TRUE(runLevel0GlobalSearch(distance_field, makeSearchScan(), params,
                                    top_regions, stats, summary));
  EXPECT_TRUE(summary.level0_ground_unavailable);
}

}  // namespace
}  // namespace localization
