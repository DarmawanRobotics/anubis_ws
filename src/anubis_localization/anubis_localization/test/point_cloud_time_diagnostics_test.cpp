#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include <localization/point_cloud_time_diagnostics.hpp>

namespace localization {
namespace {

sensor_msgs::msg::PointCloud2 makeCloud(const double first, const double second) {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.height = 1;
  cloud.width = 2;
  cloud.is_bigendian = false;
  cloud.point_step = 26;
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.resize(cloud.row_step);

  sensor_msgs::msg::PointField field;
  field.name = "timestamp";
  field.offset = 18;
  field.datatype = sensor_msgs::msg::PointField::FLOAT64;
  field.count = 1;
  cloud.fields.push_back(field);

  std::memcpy(cloud.data.data() + field.offset, &first, sizeof(double));
  std::memcpy(cloud.data.data() + cloud.point_step + field.offset, &second, sizeof(double));
  return cloud;
}

TEST(PointCloudTimeDiagnostics, ReadsLivoxAbsolutePointTimestamps) {
  const auto cloud = makeCloud(1.0e9, 1.1e9);
  const PointCloudTimeDiagnostics diagnostics = analyzePointCloudTimestamps(cloud);

  ASSERT_TRUE(diagnostics.field_present);
  ASSERT_TRUE(diagnostics.field_supported);
  ASSERT_TRUE(diagnostics.valid);
  EXPECT_EQ(diagnostics.valid_points, 2U);
  EXPECT_EQ(diagnostics.timestamp_regressions, 0U);
  EXPECT_DOUBLE_EQ(diagnostics.first_stamp_ns, 1.0e9);
  EXPECT_DOUBLE_EQ(diagnostics.last_stamp_ns, 1.1e9);
  EXPECT_DOUBLE_EQ(diagnostics.min_stamp_ns, 1.0e9);
  EXPECT_DOUBLE_EQ(diagnostics.max_stamp_ns, 1.1e9);
}

TEST(PointCloudTimeDiagnostics, CountsTimestampRegression) {
  const auto cloud = makeCloud(1.1e9, 1.0e9);
  const PointCloudTimeDiagnostics diagnostics = analyzePointCloudTimestamps(cloud);

  ASSERT_TRUE(diagnostics.valid);
  EXPECT_EQ(diagnostics.timestamp_regressions, 1U);
  EXPECT_DOUBLE_EQ(diagnostics.max_timestamp_regression_ns, 1.0e8);
  EXPECT_DOUBLE_EQ(diagnostics.min_stamp_ns, 1.0e9);
  EXPECT_DOUBLE_EQ(diagnostics.max_stamp_ns, 1.1e9);
}

TEST(PointCloudTimeDiagnostics, ReportsMissingField) {
  sensor_msgs::msg::PointCloud2 cloud;
  const PointCloudTimeDiagnostics diagnostics = analyzePointCloudTimestamps(cloud);

  EXPECT_FALSE(diagnostics.field_present);
  EXPECT_FALSE(diagnostics.valid);
}

}  // namespace
}  // namespace localization
