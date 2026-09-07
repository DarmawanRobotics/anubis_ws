#ifndef SCAN_DESCRIPTOR_POINT_ADAPTER_HPP
#define SCAN_DESCRIPTOR_POINT_ADAPTER_HPP

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "scan_descriptor/descriptor.hpp"

namespace scan_descriptor {

// Converts any iterable point container whose value type exposes x/y/z.
// ground_z_in_level is the ground height in the leveled sensor-centred frame;
// output z is always height above that ground, as recorded in the DB header.
template <typename PointRange>
std::vector<Eigen::Vector3f> adaptPoints(
    const PointRange& points, const Eigen::Matrix3d& R_level_sensor,
    double ground_z_in_level, const DescriptorConfig& config) {
  std::vector<Eigen::Vector3f> output;
  if (!validConfig(config) || !R_level_sensor.allFinite() ||
      !std::isfinite(ground_z_in_level)) {
    return output;
  }
  output.reserve(points.size());
  const double min_range_squared = config.min_range * config.min_range;
  const double max_range_squared = config.max_range * config.max_range;
  for (const auto& point : points) {
    Eigen::Vector3d transformed = R_level_sensor * Eigen::Vector3d(
        static_cast<double>(point.x), static_cast<double>(point.y),
        static_cast<double>(point.z));
    transformed.z() -= ground_z_in_level;
    const double range_squared = transformed.x() * transformed.x() +
                                 transformed.y() * transformed.y();
    if (!transformed.allFinite() || range_squared < min_range_squared ||
        range_squared > max_range_squared || transformed.z() < config.z_min ||
        transformed.z() > config.z_max) {
      continue;
    }
    output.push_back(transformed.cast<float>());
  }
  return output;
}

}  // namespace scan_descriptor

#endif  // SCAN_DESCRIPTOR_POINT_ADAPTER_HPP
