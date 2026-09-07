#include "scan_descriptor/database.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <utility>

namespace scan_descriptor {
namespace {
constexpr char kMagic[4] = {'S', 'C', 'D', 'B'};
constexpr uint32_t kVersion = 1;
constexpr uint64_t kMaxRecords = 1000000;

template <typename T>
bool writeValue(std::ofstream& stream, const T& value) {
  stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
  return static_cast<bool>(stream);
}

template <typename T>
bool readValue(std::ifstream& stream, T& value) {
  stream.read(reinterpret_cast<char*>(&value), sizeof(T));
  return static_cast<bool>(stream);
}

void setError(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}
}  // namespace

DescriptorDatabase::DescriptorDatabase(const DescriptorConfig& config)
    : config_(config) {}

bool DescriptorDatabase::add(const KeyframeRecord& record) {
  if (!validConfig(config_) || !record.T_map_lidar.allFinite() ||
      !std::isfinite(record.stamp) || !validDescriptor(record.desc, config_)) {
    return false;
  }
  if (!records_.empty() && record.id <= records_.back().id) {
    return false;
  }
  records_.push_back(record);
  return true;
}

const KeyframeRecord& DescriptorDatabase::at(size_t index) const {
  return records_.at(index);
}

bool DescriptorDatabase::setPose(size_t index, const Eigen::Matrix4d& pose) {
  if (index >= records_.size() || !pose.allFinite()) {
    return false;
  }
  records_[index].T_map_lidar = pose;
  return true;
}

std::vector<DatabaseMatch> DescriptorDatabase::queryTopK(
    const Descriptor& query, size_t k, double max_distance,
    uint64_t max_id_exclusive, double nms_radius_xy) const {
  std::vector<DatabaseMatch> output;
  if (k == 0 || !std::isfinite(max_distance) || max_distance < 0.0 ||
      !std::isfinite(nms_radius_xy) || nms_radius_xy < 0.0 ||
      !validDescriptor(query, config_)) {
    return output;
  }
  struct RingCandidate { size_t index; double distance; };
  std::vector<RingCandidate> ring_candidates;
  for (size_t index = 0; index < records_.size(); ++index) {
    if (records_[index].id >= max_id_exclusive) {
      continue;
    }
    const double distance = ringKeyDistance(
        query.ring_key, records_[index].desc.ring_key);
    if (std::isfinite(distance)) {
      ring_candidates.push_back({index, distance});
    }
  }
  std::sort(ring_candidates.begin(), ring_candidates.end(),
            [this](const RingCandidate& first, const RingCandidate& second) {
    if (first.distance != second.distance) return first.distance < second.distance;
    return records_[first.index].id < records_[second.index].id;
  });
  const size_t prefilter_count = std::min(
      ring_candidates.size(), std::max(k, k * static_cast<size_t>(5)));
  std::vector<DatabaseMatch> matches;
  matches.reserve(prefilter_count);
  for (size_t rank = 0; rank < prefilter_count; ++rank) {
    const size_t index = ring_candidates[rank].index;
    const MatchResult result = match(query, records_[index].desc, config_);
    if (result.valid && result.distance <= max_distance) {
      matches.push_back({index, records_[index].id, result.distance,
                         result.sector_shift, result.yaw_rad});
    }
  }
  std::sort(matches.begin(), matches.end(),
            [](const DatabaseMatch& first, const DatabaseMatch& second) {
    if (first.distance != second.distance) return first.distance < second.distance;
    return first.id < second.id;
  });
  for (const DatabaseMatch& candidate : matches) {
    bool suppressed = false;
    if (nms_radius_xy > 0.0) {
      const Eigen::Vector2d position =
          records_[candidate.record_index].T_map_lidar.block<2, 1>(0, 3);
      for (const DatabaseMatch& selected : output) {
        const Eigen::Vector2d selected_position =
            records_[selected.record_index].T_map_lidar.block<2, 1>(0, 3);
        if ((position - selected_position).norm() < nms_radius_xy) {
          suppressed = true;
          break;
        }
      }
    }
    if (!suppressed) {
      output.push_back(candidate);
      if (output.size() == k) break;
    }
  }
  return output;
}

bool DescriptorDatabase::save(const std::string& path, std::string* error) const {
  if (!validConfig(config_)) {
    setError(error, "invalid descriptor configuration");
    return false;
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    setError(error, "cannot open database for writing: " + path);
    return false;
  }
  stream.write(kMagic, sizeof(kMagic));
  const uint32_t rings = static_cast<uint32_t>(config_.ring_count);
  const uint32_t sectors = static_cast<uint32_t>(config_.sector_count);
  const uint32_t normalization = static_cast<uint32_t>(config_.height_normalization);
  const uint64_t count = static_cast<uint64_t>(records_.size());
  if (!writeValue(stream, kVersion) || !writeValue(stream, rings) ||
      !writeValue(stream, sectors) || !writeValue(stream, config_.min_range) ||
      !writeValue(stream, config_.max_range) || !writeValue(stream, config_.z_min) ||
      !writeValue(stream, config_.z_max) || !writeValue(stream, normalization) ||
      !writeValue(stream, count)) {
    setError(error, "failed to write database header");
    return false;
  }
  for (const KeyframeRecord& record : records_) {
    if (!record.T_map_lidar.allFinite() ||
        !validDescriptor(record.desc, config_)) {
      setError(error, "database contains an invalid record");
      return false;
    }
    if (!writeValue(stream, record.id) || !writeValue(stream, record.stamp)) {
      setError(error, "failed to write keyframe metadata");
      return false;
    }
    for (int row = 0; row < 4; ++row) {
      for (int column = 0; column < 4; ++column) {
        if (!writeValue(stream, record.T_map_lidar(row, column))) return false;
      }
    }
    const uint64_t grid_size = static_cast<uint64_t>(record.desc.grid.size());
    const uint64_t key_size = static_cast<uint64_t>(record.desc.ring_key.size());
    if (!writeValue(stream, grid_size) || !writeValue(stream, key_size)) return false;
    stream.write(reinterpret_cast<const char*>(record.desc.grid.data()),
                 static_cast<std::streamsize>(grid_size * sizeof(float)));
    stream.write(reinterpret_cast<const char*>(record.desc.ring_key.data()),
                 static_cast<std::streamsize>(key_size * sizeof(float)));
    if (!stream) {
      setError(error, "failed to write keyframe descriptor");
      return false;
    }
  }
  return true;
}

bool DescriptorDatabase::load(const std::string& path, std::string* error) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    setError(error, "cannot open database for reading: " + path);
    return false;
  }
  char magic[4] = {};
  stream.read(magic, sizeof(magic));
  uint32_t version = 0, rings = 0, sectors = 0, normalization = 0;
  DescriptorConfig loaded;
  uint64_t count = 0;
  if (!stream || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
      !readValue(stream, version) || version != kVersion ||
      !readValue(stream, rings) || !readValue(stream, sectors) ||
      !readValue(stream, loaded.min_range) || !readValue(stream, loaded.max_range) ||
      !readValue(stream, loaded.z_min) || !readValue(stream, loaded.z_max) ||
      !readValue(stream, normalization) || !readValue(stream, count)) {
    setError(error, "invalid magic, version, or truncated database header");
    return false;
  }
  loaded.ring_count = static_cast<int>(rings);
  loaded.sector_count = static_cast<int>(sectors);
  loaded.height_normalization = static_cast<HeightNormalization>(normalization);
  if (!validConfig(loaded) || count > kMaxRecords) {
    setError(error, "unsupported descriptor configuration or record count");
    return false;
  }
  std::vector<KeyframeRecord> loaded_records;
  loaded_records.reserve(static_cast<size_t>(count));
  const uint64_t expected_grid = static_cast<uint64_t>(rings) * sectors;
  for (uint64_t record_index = 0; record_index < count; ++record_index) {
    KeyframeRecord record;
    if (!readValue(stream, record.id) || !readValue(stream, record.stamp)) {
      setError(error, "truncated keyframe metadata");
      return false;
    }
    for (int row = 0; row < 4; ++row) {
      for (int column = 0; column < 4; ++column) {
        if (!readValue(stream, record.T_map_lidar(row, column))) {
          setError(error, "truncated keyframe pose");
          return false;
        }
      }
    }
    uint64_t grid_size = 0, key_size = 0;
    if (!readValue(stream, grid_size) || !readValue(stream, key_size) ||
        grid_size != expected_grid || key_size != rings) {
      setError(error, "invalid descriptor dimensions");
      return false;
    }
    record.desc.grid.resize(static_cast<int>(rings), static_cast<int>(sectors));
    record.desc.ring_key.resize(static_cast<int>(rings));
    stream.read(reinterpret_cast<char*>(record.desc.grid.data()),
                static_cast<std::streamsize>(grid_size * sizeof(float)));
    stream.read(reinterpret_cast<char*>(record.desc.ring_key.data()),
                static_cast<std::streamsize>(key_size * sizeof(float)));
    if (!stream || !record.T_map_lidar.allFinite() ||
        !validDescriptor(record.desc, loaded)) {
      setError(error, "truncated or invalid descriptor payload");
      return false;
    }
    if (!loaded_records.empty() && record.id <= loaded_records.back().id) {
      setError(error, "keyframe ids are not strictly increasing");
      return false;
    }
    loaded_records.push_back(std::move(record));
  }
  config_ = loaded;
  records_ = std::move(loaded_records);
  return true;
}

}  // namespace scan_descriptor
