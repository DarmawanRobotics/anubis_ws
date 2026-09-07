#ifndef SCAN_DESCRIPTOR_DATABASE_HPP
#define SCAN_DESCRIPTOR_DATABASE_HPP

#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "scan_descriptor/descriptor.hpp"

namespace scan_descriptor {

struct KeyframeRecord {
  uint64_t id = 0;
  double stamp = 0.0;
  Eigen::Matrix4d T_map_lidar = Eigen::Matrix4d::Identity();
  Descriptor desc;
};

struct DatabaseMatch {
  size_t record_index = 0;
  uint64_t id = 0;
  double distance = std::numeric_limits<double>::infinity();
  int sector_shift = 0;
  double yaw_rad = 0.0;
};

class DescriptorDatabase {
public:
  explicit DescriptorDatabase(const DescriptorConfig& config = DescriptorConfig{});

  bool add(const KeyframeRecord& record);
  const KeyframeRecord& at(size_t index) const;
  bool setPose(size_t index, const Eigen::Matrix4d& pose);
  size_t size() const { return records_.size(); }
  bool empty() const { return records_.empty(); }
  void clear() { records_.clear(); }
  const DescriptorConfig& config() const { return config_; }

  std::vector<DatabaseMatch> queryTopK(
      const Descriptor& query, size_t k, double max_distance,
      uint64_t max_id_exclusive = std::numeric_limits<uint64_t>::max(),
      double nms_radius_xy = 0.0) const;

  bool save(const std::string& path, std::string* error = nullptr) const;
  bool load(const std::string& path, std::string* error = nullptr);

private:
  DescriptorConfig config_;
  std::vector<KeyframeRecord> records_;
};

}  // namespace scan_descriptor

#endif  // SCAN_DESCRIPTOR_DATABASE_HPP
