#ifndef LOCALIZATION__POINT_CLOUD_TIME_DIAGNOSTICS_HPP_
#define LOCALIZATION__POINT_CLOUD_TIME_DIAGNOSTICS_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace localization {

struct PointCloudTimeDiagnostics {
  bool field_present = false;
  bool field_supported = false;
  bool valid = false;
  size_t valid_points = 0;
  size_t nonfinite_points = 0;
  size_t timestamp_regressions = 0;
  double max_timestamp_regression_ns = 0.0;
  double max_forward_step_ns = 0.0;
  double first_stamp_ns = 0.0;
  double last_stamp_ns = 0.0;
  double min_stamp_ns = 0.0;
  double max_stamp_ns = 0.0;
};

inline PointCloudTimeDiagnostics analyzePointCloudTimestamps(
    const sensor_msgs::msg::PointCloud2& cloud,
    const std::string& field_name = "timestamp") {
  PointCloudTimeDiagnostics diagnostics;
  const sensor_msgs::msg::PointField* timestamp_field = nullptr;
  for (const auto& field : cloud.fields) {
    if (field.name == field_name) {
      timestamp_field = &field;
      break;
    }
  }
  if (timestamp_field == nullptr) {
    return diagnostics;
  }

  diagnostics.field_present = true;
  if (timestamp_field->datatype != sensor_msgs::msg::PointField::FLOAT64 ||
      timestamp_field->count < 1 ||
      cloud.point_step < timestamp_field->offset + sizeof(double)) {
    return diagnostics;
  }
  diagnostics.field_supported = true;

  bool have_previous = false;
  double previous = 0.0;
  for (uint32_t row = 0; row < cloud.height; ++row) {
    for (uint32_t column = 0; column < cloud.width; ++column) {
      const size_t offset = static_cast<size_t>(row) * cloud.row_step +
        static_cast<size_t>(column) * cloud.point_step + timestamp_field->offset;
      if (offset + sizeof(double) > cloud.data.size()) {
        return diagnostics;
      }

      std::array<uint8_t, sizeof(double)> bytes{};
      std::memcpy(bytes.data(), cloud.data.data() + offset, sizeof(double));
      if (cloud.is_bigendian) {
        std::reverse(bytes.begin(), bytes.end());
      }
      double timestamp = 0.0;
      std::memcpy(&timestamp, bytes.data(), sizeof(double));
      if (!std::isfinite(timestamp)) {
        ++diagnostics.nonfinite_points;
        continue;
      }

      if (diagnostics.valid_points == 0) {
        diagnostics.first_stamp_ns = timestamp;
        diagnostics.min_stamp_ns = timestamp;
        diagnostics.max_stamp_ns = timestamp;
      } else {
        diagnostics.min_stamp_ns = std::min(diagnostics.min_stamp_ns, timestamp);
        diagnostics.max_stamp_ns = std::max(diagnostics.max_stamp_ns, timestamp);
      }
      if (have_previous) {
        const double delta_ns = timestamp - previous;
        if (delta_ns < 0.0) {
          ++diagnostics.timestamp_regressions;
          diagnostics.max_timestamp_regression_ns = std::max(
            diagnostics.max_timestamp_regression_ns, -delta_ns);
        } else {
          diagnostics.max_forward_step_ns = std::max(
            diagnostics.max_forward_step_ns, delta_ns);
        }
      }
      previous = timestamp;
      have_previous = true;
      diagnostics.last_stamp_ns = timestamp;
      ++diagnostics.valid_points;
    }
  }

  diagnostics.valid = diagnostics.valid_points > 0;
  return diagnostics;
}

}  // namespace localization

#endif  // LOCALIZATION__POINT_CLOUD_TIME_DIAGNOSTICS_HPP_
