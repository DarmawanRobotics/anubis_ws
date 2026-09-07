#include <filesystem>
#include <fstream>
#include <cmath>

#include <gtest/gtest.h>

#include "scan_descriptor/database.hpp"

namespace {
scan_descriptor::Descriptor makeDescriptor(float height) {
  scan_descriptor::DescriptorConfig config;
  std::vector<Eigen::Vector3f> points;
  for (int index = 0; index < 100; ++index) {
    const float angle = static_cast<float>(index) * 0.1f;
    points.emplace_back(3.0f * std::cos(angle), 3.0f * std::sin(angle),
                        height + 0.01f * (index % 9));
  }
  return scan_descriptor::makeDescriptor(points, config);
}

scan_descriptor::KeyframeRecord makeRecord(uint64_t id, double x, float height) {
  scan_descriptor::KeyframeRecord record;
  record.id = id;
  record.stamp = static_cast<double>(id);
  record.T_map_lidar(0, 3) = x;
  record.desc = makeDescriptor(height);
  return record;
}
}  // namespace

TEST(DatabaseTest, SaveLoadRoundTripAndNms) {
  scan_descriptor::DescriptorDatabase database;
  ASSERT_TRUE(database.add(makeRecord(0, 0.0, 0.5f)));
  ASSERT_TRUE(database.add(makeRecord(1, 0.5, 0.5f)));
  ASSERT_TRUE(database.add(makeRecord(2, 8.0, 0.6f)));
  const auto path = std::filesystem::temp_directory_path() / "scan_descriptor_test.bin";
  std::string error;
  ASSERT_TRUE(database.save(path.string(), &error)) << error;
  scan_descriptor::DescriptorDatabase loaded;
  ASSERT_TRUE(loaded.load(path.string(), &error)) << error;
  ASSERT_EQ(loaded.size(), 3U);
  EXPECT_TRUE(loaded.at(1).T_map_lidar.isApprox(database.at(1).T_map_lidar));
  const auto matches = loaded.queryTopK(makeDescriptor(0.5f), 3, 0.5, 3, 4.0);
  ASSERT_EQ(matches.size(), 2U);
  EXPECT_EQ(matches.front().id, 0U);
  std::filesystem::remove(path);
}

TEST(DatabaseTest, RejectsBadMagic) {
  const auto path = std::filesystem::temp_directory_path() / "bad_scan_descriptor.bin";
  { std::ofstream stream(path, std::ios::binary); stream << "BAD!"; }
  scan_descriptor::DescriptorDatabase database;
  std::string error;
  EXPECT_FALSE(database.load(path.string(), &error));
  EXPECT_FALSE(error.empty());
  std::filesystem::remove(path);
}

TEST(DatabaseTest, RejectsUnsupportedVersion) {
  const auto path = std::filesystem::temp_directory_path() / "bad_version_scan_descriptor.bin";
  {
    std::ofstream stream(path, std::ios::binary);
    stream.write("SCDB", 4);
    const uint32_t version = 999;
    stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
  }
  scan_descriptor::DescriptorDatabase database;
  std::string error;
  EXPECT_FALSE(database.load(path.string(), &error));
  EXPECT_FALSE(error.empty());
  std::filesystem::remove(path);
}

TEST(DatabaseTest, AppliesRecencyExclusionAndDistanceThreshold) {
  scan_descriptor::DescriptorDatabase database;
  ASSERT_TRUE(database.add(makeRecord(0, 0.0, 0.5f)));
  ASSERT_TRUE(database.add(makeRecord(20, 8.0, 1.5f)));
  const auto matches = database.queryTopK(makeDescriptor(0.5f), 2, 0.1, 10, 0.0);
  ASSERT_EQ(matches.size(), 1U);
  EXPECT_EQ(matches.front().id, 0U);
}
