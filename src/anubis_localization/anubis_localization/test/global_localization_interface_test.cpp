#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "localization/global_localization.hpp"

namespace localization {
namespace {

TEST(GlobalLocalizationInterface, LegacyApiFailsClosedWithoutChangingOutput) {
  GlobalLocalization localization;
  pcl::PointCloud<pcl::PointXYZI>::Ptr map(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  const Eigen::Matrix4d seed_pose = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d output_pose = Eigen::Matrix4d::Constant(7.0);
  const Eigen::Matrix4d expected_output = output_pose;

  EXPECT_FALSE(localization.performGlobalLocalization(map, scan, seed_pose, output_pose));
  EXPECT_TRUE(output_pose.isApprox(expected_output));
}

TEST(GlobalLocalizationInterface, ProductionApiRejectsMissingGravity) {
  GlobalLocalization localization;
  pcl::PointCloud<pcl::PointXYZI>::Ptr map(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  GlobalLocalizationResult result;
  result.status = GLStatus::Confirmed;
  result.success = true;

  localization.performGlobalLocalization(
    map, scan, Eigen::Matrix4d::Identity(), nullptr, nullptr, nullptr, result);

  EXPECT_EQ(result.status, GLStatus::NoCandidate);
  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.anchor_reject_reason, GLAnchorRejectReason::GravityUnavailable);
  EXPECT_TRUE(result.final_pose.isIdentity());
  EXPECT_TRUE(result.candidates.empty());
}

TEST(GlobalLocalizationInterface, ProductionApiRejectsEmptyScanWithValidGravity) {
  GlobalLocalization localization;
  pcl::PointCloud<pcl::PointXYZI>::Ptr map(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  GravityAlignmentSample gravity;
  gravity.valid = true;
  GlobalLocalizationResult result;

  localization.performGlobalLocalization(
    map, scan, Eigen::Matrix4d::Identity(), nullptr, nullptr, &gravity, result);

  EXPECT_EQ(result.status, GLStatus::NoCandidate);
  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.anchor_reject_reason, GLAnchorRejectReason::TooFewValidPoints);
  EXPECT_TRUE(result.candidates.empty());
}

TEST(GlobalLocalizationInterface, SetParamsRejectsInvalidConfiguration) {
  GlobalLocalization localization;
  EXPECT_NO_THROW(localization.setParams(GLParams{}));

  GLParams params;
  params.refine_leaf_src = 0.0;

  EXPECT_THROW(localization.setParams(params), std::invalid_argument);
}

TEST(GlobalLocalizationInterface, ProductionApiRunsMulticandidateSpatialPipeline) {
  constexpr double kPi = 3.14159265358979323846;
  pcl::PointCloud<pcl::PointXYZI>::Ptr map(
      new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(
      new pcl::PointCloud<pcl::PointXYZI>());
  for (int index = 0; index < 32; ++index) {
    const double angle = 2.0 * kPi * static_cast<double>(index) / 32.0;
    const double radius = 2.5 + 0.35 * std::sin(3.0 * angle) +
        0.15 * std::cos(5.0 * angle);
    pcl::PointXYZI obstacle;
    obstacle.x = static_cast<float>(radius * std::cos(angle));
    obstacle.y = static_cast<float>(radius * std::sin(angle));
    obstacle.z = 0.5f;
    obstacle.intensity = static_cast<float>(index);
    scan->push_back(obstacle);
    map->push_back(obstacle);

    pcl::PointXYZI floor = obstacle;
    floor.z = 0.0f;
    map->push_back(floor);
  }

  GLParams params;
  params.flat_single_level_map = true;
  params.level0_base_ground_offset = 0.0;
  params.level0_grid_stride_xy = 2.0;
  params.level0_yaw_samples = 4;
  params.level0_top_regions = 1;
  params.level0_distance_field_resolution = 0.25;
  params.level0_min_range = 0.1;
  params.level0_max_range = 10.0;
  params.level0_min_valid_points = 16;
  params.level0_min_valid_projection_ratio = 0.40;
  params.level0_max_boundary_unknown_ratio = 0.99;
  params.level0_boundary_padding_m = 1.0;
  params.level0_region_nms_xy = 1.0;
  params.level0_region_nms_yaw_deg = 20.0;
  params.level1_search_radius_xy = 1.5;
  params.level1_seed_stride_xy = 1.3;
  params.level1_yaw_step_deg = 15.0;
  params.max_candidates_total = 20;
  params.max_refine_candidates = 5;
  params.coarse_leaf_map = 0.05;
  params.coarse_leaf_src = 0.05;
  params.coarse_max_iter = 20;
  params.coarse_roi_radius = 8.0;
  params.coarse_max_corr_dist = 2.0;
  params.coarse_min_overlap = 0.15;
  params.coarse_min_inliers = 4;
  params.coarse_max_p90_residual = 2.0;
  params.refine_leaf_src = 0.05;
  params.refine_max_iter = 20;
  params.refine_fitness_threshold = 1.0;
  params.max_correction_xy = 3.0;
  params.max_correction_z = 2.0;
  params.max_yaw_delta_deg = 90.0;
  params.cluster_xy = 0.4;
  params.cluster_yaw_deg = 20.0;
  params.support_seed_xy = 0.9;  // must exceed 2*cluster_xy (0.8) per validation
  params.min_support_groups = 1;

  GravityAlignmentSample gravity;
  gravity.stamp = rclcpp::Time(int64_t{1'000'000'000LL}, RCL_ROS_TIME);
  gravity.uncertainty_deg = 0.1;
  gravity.angular_velocity_rps = 0.0;
  gravity.accel_norm_residual_mps2 = 0.0;
  gravity.source = "test";
  gravity.valid = true;

  GlobalLocalization localization;
  ASSERT_NO_THROW(localization.setParams(params));
  GlobalLocalizationResult result;
  GLSummaryCounts summary;
  localization.performGlobalLocalization(
      map, scan, Eigen::Matrix4d::Identity(), nullptr, nullptr, &gravity,
      result, &summary);

  EXPECT_GT(result.candidate_count, 1);
  EXPECT_EQ(result.candidate_count, summary.seed_total);
  EXPECT_GT(summary.level1_seed_count, 1);
  ASSERT_EQ(result.level0_candidates.size(), 1U);
  EXPECT_EQ(result.level0_candidates.front().source_type, "level0_region");
  EXPECT_EQ(result.seed_candidates.size(),
            static_cast<size_t>(result.candidate_count));
  EXPECT_GT(result.coarse_kept_count, 0);
  EXPECT_NE(result.status, GLStatus::Confirmed);
  EXPECT_FALSE(result.success);
}

TEST(GlobalLocalizationInterface, RetrievalModeFeedsExistingSpatialPipeline) {
  constexpr double kPi = 3.14159265358979323846;
  pcl::PointCloud<pcl::PointXYZI>::Ptr map(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan(new pcl::PointCloud<pcl::PointXYZI>());
  for (int index = 0; index < 64; ++index) {
    const double angle = 2.0 * kPi * index / 64.0;
    pcl::PointXYZI point;
    point.x = static_cast<float>((2.0 + 0.2 * (index % 5)) * std::cos(angle));
    point.y = static_cast<float>((2.0 + 0.2 * (index % 5)) * std::sin(angle));
    point.z = 0.3f + 0.1f * (index % 7);
    scan->push_back(point);
    map->push_back(point);
  }
  auto database = std::make_shared<scan_descriptor::DescriptorDatabase>();
  scan_descriptor::KeyframeRecord record;
  record.id = 7;
  std::vector<Eigen::Vector3f> descriptor_points;
  for (const auto& point : *scan) descriptor_points.emplace_back(point.x, point.y, point.z);
  record.desc = scan_descriptor::makeDescriptor(descriptor_points, database->config());
  ASSERT_TRUE(database->add(record));

  GLParams params;
  params.gl_recall_source = "retrieval";
  params.retrieval_fallback_to_grid = false;
  params.retrieval_topk = 1;
  params.retrieval_min_points = 16;
  params.retrieval_max_distance = 0.1;
  params.level0_base_ground_offset = 0.0;
  params.level0_min_range = 0.1;
  params.max_candidates_total = 40;
  params.max_refine_candidates = 5;
  params.coarse_leaf_map = 0.05;
  params.coarse_leaf_src = 0.05;
  params.coarse_min_inliers = 4;
  params.coarse_min_overlap = 0.1;
  params.coarse_max_p90_residual = 2.0;
  params.refine_leaf_src = 0.05;
  params.refine_fitness_threshold = 1.0;
  params.min_support_groups = 1;
  GravityAlignmentSample gravity;
  gravity.stamp = rclcpp::Time(int64_t{1'000'000'000LL}, RCL_ROS_TIME);
  gravity.uncertainty_deg = 0.1;
  gravity.angular_velocity_rps = 0.0;
  gravity.accel_norm_residual_mps2 = 0.0;
  gravity.valid = true;

  GlobalLocalization localization;
  ASSERT_NO_THROW(localization.setParams(params));
  localization.setDescriptorDatabase(database);
  GlobalLocalizationResult result;
  GLSummaryCounts summary;
  localization.performGlobalLocalization(
      map, scan, Eigen::Matrix4d::Identity(), nullptr, nullptr, &gravity,
      result, &summary);
  EXPECT_EQ(summary.retrieval_db_size, 1);
  EXPECT_EQ(summary.retrieval_match_count, 1);
  ASSERT_EQ(result.level0_candidates.size(), 1U);
  EXPECT_EQ(result.level0_candidates.front().source_type, "retrieval_region");
  EXPECT_TRUE(std::any_of(
      result.seed_candidates.begin(), result.seed_candidates.end(),
      [](const GlobalPoseCandidate& candidate) {
        return candidate.source_type == "retrieval";
      }));
}

}  // namespace
}  // namespace localization
