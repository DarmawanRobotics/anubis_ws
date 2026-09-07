#ifndef SCAN_DESCRIPTOR_DESCRIPTOR_HPP
#define SCAN_DESCRIPTOR_DESCRIPTOR_HPP

#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>

namespace scan_descriptor {

enum class HeightNormalization : uint32_t {
  GroundRelative = 1,
};

struct DescriptorConfig {
  int ring_count = 20;
  int sector_count = 60;
  double max_range = 10.0;
  double min_range = 0.3;
  double z_min = -0.2;
  double z_max = 2.0;
  HeightNormalization height_normalization = HeightNormalization::GroundRelative;
};

struct Descriptor {
  Eigen::MatrixXf grid;
  Eigen::VectorXf ring_key;
};

struct MatchResult {
  double distance = std::numeric_limits<double>::infinity();
  int sector_shift = 0;
  double yaw_rad = 0.0;
  bool valid = false;
};

bool validConfig(const DescriptorConfig& config);
bool validDescriptor(const Descriptor& descriptor, const DescriptorConfig& config);
Descriptor makeDescriptor(const std::vector<Eigen::Vector3f>& leveled_ground_points,
                          const DescriptorConfig& config);
MatchResult match(const Descriptor& query, const Descriptor& candidate,
                  const DescriptorConfig& config);
double ringKeyDistance(const Eigen::VectorXf& first,
                       const Eigen::VectorXf& second);

}  // namespace scan_descriptor

#endif  // SCAN_DESCRIPTOR_DESCRIPTOR_HPP
