#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "localization/retrieval_recall.hpp"

namespace {
constexpr double kPi = 3.14159265358979323846;

scan_descriptor::Descriptor makeDescriptor(float phase) {
  scan_descriptor::DescriptorConfig config;
  std::vector<Eigen::Vector3f> points;
  for (int index = 0; index < 500; ++index) {
    const float angle = phase + static_cast<float>(index) * 0.07f;
    const float radius = 1.0f + static_cast<float>(index % 20) * 0.35f;
    points.emplace_back(radius * std::cos(angle), radius * std::sin(angle),
                        0.2f + static_cast<float>(index % 8) * 0.1f);
  }
  return scan_descriptor::makeDescriptor(points, config);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr makeScan(float phase) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(
      new pcl::PointCloud<pcl::PointXYZI>());
  scan->reserve(500);
  for (int index = 0; index < 500; ++index) {
    const float angle = phase + static_cast<float>(index) * 0.07f;
    const float radius = 1.0f + static_cast<float>(index % 20) * 0.35f;
    pcl::PointXYZI point;
    point.x = radius * std::cos(angle);
    point.y = radius * std::sin(angle);
    point.z = 0.2f + static_cast<float>(index % 8) * 0.1f;
    scan->push_back(point);
  }
  return scan;
}

scan_descriptor::KeyframeRecord record(uint64_t id, double x, float phase) {
  scan_descriptor::KeyframeRecord value;
  value.id = id;
  value.T_map_lidar(0, 3) = x;
  value.desc = makeDescriptor(phase);
  return value;
}
}  // namespace

TEST(RetrievalRecallTest, NmsKeepsDistinctKeyframeLocations) {
  scan_descriptor::DescriptorDatabase database;
  ASSERT_TRUE(database.add(record(0, 0.0, 0.0f)));
  ASSERT_TRUE(database.add(record(1, 0.5, 0.0f)));
  ASSERT_TRUE(database.add(record(2, 8.0, 0.3f)));
  const auto matches = database.queryTopK(makeDescriptor(0.0f), 3, 0.5,
                                           std::numeric_limits<uint64_t>::max(), 4.0);
  ASSERT_EQ(matches.size(), 2U);
  EXPECT_EQ(matches[0].id, 0U);
  EXPECT_EQ(matches[1].id, 2U);
}

TEST(RetrievalRecallTest, SeedsCarryRetrievalSupportGroup) {
  localization::GLParams params;
  params.max_candidates_total = 80;
  params.support_seed_xy = 1.2;
  localization::RetrievalRecallParams retrieval;
  localization::Level0Hypothesis region;
  region.score.valid = true;
  region.score.score = 0.1;
  region.grid_cell_id = 42;
  std::vector<localization::GlobalPoseCandidate> seeds;
  localization::GLSummaryCounts summary;
  ASSERT_TRUE(localization::generateRetrievalSeeds(
      {region}, Eigen::Matrix3d::Identity(), retrieval, 0, "anchor",
      params, seeds, summary));
  ASSERT_FALSE(seeds.empty());
  EXPECT_EQ(seeds.front().source_type, "retrieval");
  EXPECT_EQ(seeds.front().support_group.rfind("grid:kf42:", 0), 0U);
}

TEST(RetrievalRecallTest, RejectsTooFewPoints) {
  scan_descriptor::DescriptorDatabase database;
  ASSERT_TRUE(database.add(record(0, 0.0, 0.0f)));
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  scan->resize(2);
  localization::RetrievalRecallParams params;
  std::vector<localization::Level0Hypothesis> regions;
  localization::RetrievalRecallStats stats;
  EXPECT_FALSE(localization::runRetrievalRecall(
      database, scan, Eigen::Matrix3d::Identity(), Eigen::Matrix4d::Identity(),
      0.0, params, regions, stats));
}

TEST(RetrievalRecallTest, ConvertsPersistedLidarPoseToBasePose) {
  scan_descriptor::DescriptorDatabase database;
  Eigen::Matrix4d T_base_lidar = Eigen::Matrix4d::Identity();
  // Keep this conversion contract aligned with localization/config/config.yaml
  // (the production matrix is deliberately tested at its serialized values).
  T_base_lidar.block<3, 3>(0, 0) <<
      0.943265351, 0.146966729, 0.297743610,
      -0.147711412, 0.988825639, -0.020129442,
      -0.297374874, -0.024992724, 0.954433627;
  T_base_lidar.block<3, 1>(0, 3) = Eigen::Vector3d(0.22, 0.0, 0.085);
  Eigen::Matrix4d T_map_base = Eigen::Matrix4d::Identity();
  T_map_base.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(30.0 * kPi / 180.0, Eigen::Vector3d::UnitZ())
          .toRotationMatrix();
  T_map_base.block<3, 1>(0, 3) = Eigen::Vector3d(3.0, -2.0, 0.4);
  auto stored = record(0, 0.0, 0.0f);
  stored.T_map_lidar = T_map_base * T_base_lidar;
  ASSERT_TRUE(database.add(stored));

  localization::RetrievalRecallParams params;
  params.topk = 1;
  params.nms_xy = 0.0;
  params.max_distance = 0.01;
  std::vector<localization::Level0Hypothesis> regions;
  localization::RetrievalRecallStats stats;
  ASSERT_TRUE(localization::runRetrievalRecall(
      database, makeScan(0.0f), Eigen::Matrix3d::Identity(), T_base_lidar,
      0.0, params, regions, stats));
  ASSERT_EQ(regions.size(), 1U);
  EXPECT_TRUE((regions.front().T_map_level.block<3, 1>(0, 3).isApprox(
      T_map_base.block<3, 1>(0, 3), 1e-9)));
  EXPECT_NEAR(regions.front().yaw_deg, 30.0, 1e-9);
}
