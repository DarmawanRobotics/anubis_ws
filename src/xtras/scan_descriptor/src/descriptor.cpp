#include "scan_descriptor/descriptor.hpp"

#include <algorithm>
#include <cmath>

namespace scan_descriptor {
namespace {
constexpr double kPi = 3.14159265358979323846;
}

bool validConfig(const DescriptorConfig& config) {
  return config.ring_count > 0 && config.sector_count > 0 &&
      std::isfinite(config.min_range) && std::isfinite(config.max_range) &&
      config.min_range >= 0.0 && config.max_range > config.min_range &&
      std::isfinite(config.z_min) && std::isfinite(config.z_max) &&
      config.z_max > config.z_min &&
      config.height_normalization == HeightNormalization::GroundRelative;
}

bool validDescriptor(const Descriptor& descriptor,
                     const DescriptorConfig& config) {
  return validConfig(config) && descriptor.grid.rows() == config.ring_count &&
      descriptor.grid.cols() == config.sector_count &&
      descriptor.ring_key.size() == config.ring_count;
}

Descriptor makeDescriptor(
    const std::vector<Eigen::Vector3f>& points,
    const DescriptorConfig& config) {
  Descriptor descriptor;
  if (!validConfig(config)) {
    return descriptor;
  }
  descriptor.grid = Eigen::MatrixXf::Constant(
      config.ring_count, config.sector_count,
      std::numeric_limits<float>::quiet_NaN());
  descriptor.ring_key = Eigen::VectorXf::Zero(config.ring_count);
  const double radial_span = config.max_range - config.min_range;
  for (const Eigen::Vector3f& point : points) {
    if (!point.allFinite()) {
      continue;
    }
    const double range = std::hypot(static_cast<double>(point.x()),
                                    static_cast<double>(point.y()));
    if (range < config.min_range || range > config.max_range ||
        point.z() < config.z_min || point.z() > config.z_max) {
      continue;
    }
    int ring = static_cast<int>(
        (range - config.min_range) / radial_span * config.ring_count);
    ring = std::min(config.ring_count - 1, std::max(0, ring));
    double angle = std::atan2(static_cast<double>(point.y()),
                              static_cast<double>(point.x()));
    if (angle < 0.0) {
      angle += 2.0 * kPi;
    }
    int sector = static_cast<int>(angle / (2.0 * kPi) * config.sector_count);
    sector = std::min(config.sector_count - 1, std::max(0, sector));
    float& cell = descriptor.grid(ring, sector);
    if (!std::isfinite(cell) || point.z() > cell) {
      cell = point.z();
    }
  }
  for (int ring = 0; ring < config.ring_count; ++ring) {
    int occupied = 0;
    for (int sector = 0; sector < config.sector_count; ++sector) {
      occupied += std::isfinite(descriptor.grid(ring, sector)) ? 1 : 0;
    }
    descriptor.ring_key(ring) = static_cast<float>(occupied) /
        static_cast<float>(config.sector_count);
  }
  return descriptor;
}

MatchResult match(const Descriptor& query, const Descriptor& candidate,
                  const DescriptorConfig& config) {
  MatchResult best;
  if (!validDescriptor(query, config) ||
      !validDescriptor(candidate, config)) {
    return best;
  }
  for (int shift = 0; shift < config.sector_count; ++shift) {
    double dot = 0.0;
    double query_norm = 0.0;
    double candidate_norm = 0.0;
    int overlap = 0;
    int query_occupied = 0;
    int candidate_occupied = 0;
    for (int ring = 0; ring < config.ring_count; ++ring) {
      for (int candidate_sector = 0;
           candidate_sector < config.sector_count; ++candidate_sector) {
        int query_sector = candidate_sector - shift;
        if (query_sector < 0) {
          query_sector += config.sector_count;
        }
        const float q = query.grid(ring, query_sector);
        const float c = candidate.grid(ring, candidate_sector);
        query_occupied += std::isfinite(q) ? 1 : 0;
        candidate_occupied += std::isfinite(c) ? 1 : 0;
        if (!std::isfinite(q) || !std::isfinite(c)) {
          continue;
        }
        dot += static_cast<double>(q) * c;
        query_norm += static_cast<double>(q) * q;
        candidate_norm += static_cast<double>(c) * c;
        ++overlap;
      }
    }
    if (overlap < std::max(3, config.ring_count / 4) ||
        query_norm <= 1e-12 || candidate_norm <= 1e-12) {
      continue;
    }
    const double similarity = dot / std::sqrt(query_norm * candidate_norm);
    const double cosine_distance =
        1.0 - std::max(-1.0, std::min(1.0, similarity));
    const double occupancy_overlap = static_cast<double>(overlap) /
        static_cast<double>(std::max(query_occupied, candidate_occupied));
    const double distance = 0.5 * cosine_distance +
        0.5 * (1.0 - occupancy_overlap);
    if (!best.valid || distance < best.distance) {
      best.valid = true;
      best.distance = distance;
      best.sector_shift = shift;
      best.yaw_rad = 2.0 * kPi * static_cast<double>(shift) /
          static_cast<double>(config.sector_count);
      if (best.yaw_rad > kPi) {
        best.yaw_rad -= 2.0 * kPi;
        best.sector_shift -= config.sector_count;
      }
    }
  }
  return best;
}

double ringKeyDistance(const Eigen::VectorXf& first,
                       const Eigen::VectorXf& second) {
  if (first.size() == 0 || first.size() != second.size() ||
      !first.allFinite() || !second.allFinite()) {
    return std::numeric_limits<double>::infinity();
  }
  return (first - second).norm() / std::sqrt(static_cast<double>(first.size()));
}

}  // namespace scan_descriptor
