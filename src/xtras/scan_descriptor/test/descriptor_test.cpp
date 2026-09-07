#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include "scan_descriptor/descriptor.hpp"

namespace {
constexpr double kPi = 3.14159265358979323846;

std::vector<Eigen::Vector3f> makeScene() {
  std::vector<Eigen::Vector3f> points;
  for (int sector = 0; sector < 60; ++sector) {
    const double angle = sector * 2.0 * kPi / 60.0;
    for (int ring = 2; ring < 18; ++ring) {
      const float radius = 0.5f + ring * 0.45f;
      const float height = 0.2f + 0.03f * ring + 0.6f * (sector % 7 == 0);
      points.emplace_back(radius * std::cos(angle), radius * std::sin(angle), height);
    }
  }
  return points;
}

std::vector<Eigen::Vector3f> rotateScene(
    const std::vector<Eigen::Vector3f>& input, double yaw) {
  std::vector<Eigen::Vector3f> output;
  const Eigen::Matrix3f rotation = Eigen::AngleAxisf(
      static_cast<float>(yaw), Eigen::Vector3f::UnitZ()).toRotationMatrix();
  for (const auto& point : input) output.push_back(rotation * point);
  return output;
}
}  // namespace

TEST(DescriptorTest, RecoversDocumentedYawConvention) {
  scan_descriptor::DescriptorConfig config;
  const auto keyframe = makeScene();
  for (double yaw_deg : {6.0, -30.0, 90.0, 179.0}) {
    // A sensor rotated +yaw observes the old scene rotated -yaw locally.
    const auto query = rotateScene(keyframe, -yaw_deg * kPi / 180.0);
    const auto result = scan_descriptor::match(
        scan_descriptor::makeDescriptor(query, config),
        scan_descriptor::makeDescriptor(keyframe, config), config);
    ASSERT_TRUE(result.valid);
    EXPECT_NEAR(result.yaw_rad, yaw_deg * kPi / 180.0, 2.0 * kPi / 60.0 + 1e-6);
  }
}

TEST(DescriptorTest, AppliesRangeAndHeightGates) {
  scan_descriptor::DescriptorConfig config;
  std::vector<Eigen::Vector3f> points = {
      {0.1f, 0.0f, 1.0f}, {2.0f, 0.0f, -0.3f},
      {2.0f, 0.0f, 0.8f}, {11.0f, 0.0f, 0.8f}};
  const auto descriptor = scan_descriptor::makeDescriptor(points, config);
  EXPECT_FLOAT_EQ(descriptor.ring_key.sum(), 1.0f / config.sector_count);
}

TEST(DescriptorTest, DegenerateInputDoesNotMatch) {
  scan_descriptor::DescriptorConfig config;
  const auto empty = scan_descriptor::makeDescriptor({}, config);
  EXPECT_FALSE(scan_descriptor::match(empty, empty, config).valid);
}

TEST(DescriptorTest, RingKeyTracksRadialContent) {
  scan_descriptor::DescriptorConfig config;
  const auto base = makeScene();
  auto nearby = base;
  for (auto& point : nearby) point.x() += 0.2f;
  auto far = base;
  for (auto& point : far) point *= 0.25f;
  const auto base_desc = scan_descriptor::makeDescriptor(base, config);
  EXPECT_LT(scan_descriptor::ringKeyDistance(
                base_desc.ring_key,
                scan_descriptor::makeDescriptor(nearby, config).ring_key),
            scan_descriptor::ringKeyDistance(
                base_desc.ring_key,
                scan_descriptor::makeDescriptor(far, config).ring_key));
}

TEST(DescriptorTest, TranslationToleranceAndLargeTranslationSeparation) {
  scan_descriptor::DescriptorConfig config;
  const auto base = makeScene();
  auto near = base;
  auto far = base;
  for (auto& point : near) point.x() += 1.0f;
  for (auto& point : far) point.x() += 8.0f;
  const auto base_descriptor = scan_descriptor::makeDescriptor(base, config);
  const auto near_match = scan_descriptor::match(
      scan_descriptor::makeDescriptor(near, config), base_descriptor, config);
  const auto far_match = scan_descriptor::match(
      scan_descriptor::makeDescriptor(far, config), base_descriptor, config);
  ASSERT_TRUE(near_match.valid);
  ASSERT_TRUE(far_match.valid);
  EXPECT_LT(near_match.distance, far_match.distance);
}
