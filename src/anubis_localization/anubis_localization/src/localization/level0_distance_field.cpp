#include "localization/level0_distance_field.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include <pcl/filters/filter.h>

namespace localization {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kAngleBuckets = 360;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct QueueEntry {
  double distance;
  int index;

  bool operator>(const QueueEntry& other) const {
    return distance > other.distance;
  }
};

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  if (lower == upper) {
    return values[lower];
  }
  const double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

bool finitePoint(const pcl::PointXYZI& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

}  // namespace

bool applyLevel0ScoreToCandidate(const Level0Score& score,
                                 GlobalPoseCandidate& candidate) {
  candidate.level0_mean_truncated_distance = score.mean_truncated_distance;
  candidate.level0_valid_projection_ratio = score.valid_projection_ratio;
  candidate.level0_input_points = score.input_point_count;
  candidate.level0_valid_points = score.valid_projection_points;
  candidate.coarse_score = score.valid ? score.score : 1e9;
  return score.valid;
}

void accumulateLevel0Summary(const Level0Score& score,
                             GLSummaryCounts& summary) {
  summary.level0_input_points += score.input_point_count;
  summary.level0_valid_points += score.valid_projection_points;
  summary.level0_boundary_unknown_points += score.boundary_unknown_points;
  summary.level0_invalid_projection_points += score.invalid_projection_points;
  summary.level0_lookup_count += score.lookup_count;
  if (!score.valid) {
    ++summary.level0_invalid_hypotheses;
  }
}

bool Level0DistanceField::build(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& global_map,
    const GLParams& params) {
  distance_cells_.clear();
  width_ = 0;
  height_ = 0;
  if (!global_map || global_map->empty() || !params.flat_single_level_map ||
      params.level0_distance_field_resolution <= 0.0 ||
      params.level0_min_point_height >= params.level0_max_point_height) {
    return false;
  }

  std::vector<pcl::PointXYZI> finite_points;
  finite_points.reserve(global_map->size());
  std::vector<double> z_values;
  z_values.reserve(global_map->size());
  for (const auto& point : *global_map) {
    if (finitePoint(point)) {
      finite_points.push_back(point);
      z_values.push_back(static_cast<double>(point.z));
    }
  }
  if (finite_points.empty()) {
    return false;
  }

  const double percentile_fraction = std::clamp(params.grid_z_percentile, 0.0, 1.0);
  floor_z_ = percentile(std::move(z_values), percentile_fraction);

  std::vector<pcl::PointXYZI> obstacle_points;
  obstacle_points.reserve(finite_points.size());
  for (const auto& point : finite_points) {
    const double height = static_cast<double>(point.z) - floor_z_;
    if (height >= params.level0_min_point_height &&
        height <= params.level0_max_point_height) {
      obstacle_points.push_back(point);
    }
  }
  if (obstacle_points.empty()) {
    return false;
  }

  min_x_ = max_x_ = finite_points.front().x;
  min_y_ = max_y_ = finite_points.front().y;
  for (const auto& point : finite_points) {
    min_x_ = std::min(min_x_, static_cast<double>(point.x));
    max_x_ = std::max(max_x_, static_cast<double>(point.x));
    min_y_ = std::min(min_y_, static_cast<double>(point.y));
    max_y_ = std::max(max_y_, static_cast<double>(point.y));
  }

  resolution_ = params.level0_distance_field_resolution;
  boundary_padding_m_ = std::max(0.0, params.level0_boundary_padding_m);
  width_ = static_cast<int>(std::ceil((max_x_ - min_x_) / resolution_)) + 1;
  height_ = static_cast<int>(std::ceil((max_y_ - min_y_) / resolution_)) + 1;
  if (width_ <= 0 || height_ <= 0 ||
      static_cast<size_t>(width_) * static_cast<size_t>(height_) > 100000000U) {
    width_ = 0;
    height_ = 0;
    return false;
  }

  const size_t cell_count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  std::vector<uint8_t> occupied(cell_count, 0U);
  distance_cells_.assign(cell_count, kInfinity);
  auto cellIndex = [this](double x, double y) {
    const int ix = static_cast<int>(std::floor((x - min_x_) / resolution_));
    const int iy = static_cast<int>(std::floor((y - min_y_) / resolution_));
    if (ix < 0 || iy < 0 || ix >= width_ || iy >= height_) {
      return -1;
    }
    return iy * width_ + ix;
  };

  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;
  for (const auto& point : obstacle_points) {
    const int index = cellIndex(point.x, point.y);
    if (index >= 0 && !occupied[static_cast<size_t>(index)]) {
      occupied[static_cast<size_t>(index)] = 1U;
      distance_cells_[static_cast<size_t>(index)] = 0.0;
      queue.push({0.0, index});
    }
  }
  if (queue.empty()) {
    distance_cells_.clear();
    width_ = 0;
    height_ = 0;
    return false;
  }

  const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  const double step[8] = {
      resolution_, resolution_, resolution_, resolution_,
      resolution_ * std::sqrt(2.0), resolution_ * std::sqrt(2.0),
      resolution_ * std::sqrt(2.0), resolution_ * std::sqrt(2.0)};
  while (!queue.empty()) {
    const QueueEntry current = queue.top();
    queue.pop();
    if (current.distance > distance_cells_[static_cast<size_t>(current.index)]) {
      continue;
    }
    const int x = current.index % width_;
    const int y = current.index / width_;
    for (int direction = 0; direction < 8; ++direction) {
      const int next_x = x + dx[direction];
      const int next_y = y + dy[direction];
      if (next_x < 0 || next_y < 0 || next_x >= width_ || next_y >= height_) {
        continue;
      }
      const int next_index = next_y * width_ + next_x;
      const double next_distance = current.distance + step[direction];
      if (next_distance < distance_cells_[static_cast<size_t>(next_index)]) {
        distance_cells_[static_cast<size_t>(next_index)] = next_distance;
        queue.push({next_distance, next_index});
      }
    }
  }
  return true;
}

bool Level0DistanceField::prepareScan(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& scan_base,
    const Eigen::Matrix3d& R_level_base,
    double base_ground_offset,
    const GLParams& params,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& scan_level,
    Level0ScanPreparationStats& stats) const {
  stats = Level0ScanPreparationStats{};
  if (!scan_base || scan_base->empty() || !R_level_base.allFinite() ||
      !std::isfinite(base_ground_offset)) {
    return false;
  }
  stats.input_points = static_cast<int>(scan_base->size());

  std::vector<pcl::PointXYZI> bucket_points(kAngleBuckets);
  std::vector<uint8_t> bucket_valid(kAngleBuckets, 0U);
  for (const auto& point : *scan_base) {
    if (!finitePoint(point)) {
      continue;
    }
    ++stats.finite_points;
    const Eigen::Vector3d level =
        R_level_base * Eigen::Vector3d(point.x, point.y, point.z);
    const double height = level.z() + base_ground_offset;
    if (height < params.level0_min_point_height ||
        height > params.level0_max_point_height) {
      continue;
    }
    ++stats.height_points;
    const double range = std::hypot(level.x(), level.y());
    if (range < params.level0_min_range || range > params.level0_max_range) {
      continue;
    }
    ++stats.range_points;

    double angle = std::atan2(level.y(), level.x());
    if (angle < 0.0) {
      angle += 2.0 * kPi;
    }
    int bucket = static_cast<int>(std::floor(angle / (2.0 * kPi) * kAngleBuckets));
    bucket = std::clamp(bucket, 0, kAngleBuckets - 1);
    if (!bucket_valid[static_cast<size_t>(bucket)] ||
        range < std::hypot(bucket_points[static_cast<size_t>(bucket)].x,
                           bucket_points[static_cast<size_t>(bucket)].y)) {
      bucket_points[static_cast<size_t>(bucket)].x = static_cast<float>(level.x());
      bucket_points[static_cast<size_t>(bucket)].y = static_cast<float>(level.y());
      bucket_points[static_cast<size_t>(bucket)].z = static_cast<float>(level.z());
      bucket_points[static_cast<size_t>(bucket)].intensity = point.intensity;
      bucket_valid[static_cast<size_t>(bucket)] = 1U;
    }
  }

  std::vector<pcl::PointXYZI> candidates;
  candidates.reserve(stats.range_points);
  for (int bucket = 0; bucket < kAngleBuckets; ++bucket) {
    if (bucket_valid[static_cast<size_t>(bucket)]) {
      candidates.push_back(bucket_points[static_cast<size_t>(bucket)]);
    }
  }

  const int max_points = std::max(0, params.level0_max_scan_points);
  if (max_points > 0 && static_cast<int>(candidates.size()) > max_points) {
    std::vector<pcl::PointXYZI> sampled;
    sampled.reserve(static_cast<size_t>(max_points));
    for (int index = 0; index < max_points; ++index) {
      const size_t source_index = static_cast<size_t>(
          (static_cast<long long>(index) * candidates.size()) / max_points);
      sampled.push_back(candidates[source_index]);
    }
    candidates.swap(sampled);
  }

  if (!scan_level) {
    scan_level.reset(new pcl::PointCloud<pcl::PointXYZI>());
  }
  scan_level->clear();
  scan_level->points.assign(candidates.begin(), candidates.end());
  scan_level->width = static_cast<uint32_t>(scan_level->points.size());
  scan_level->height = 1U;
  scan_level->is_dense = true;
  stats.sampled_points = static_cast<int>(scan_level->size());
  stats.valid = stats.sampled_points >= params.level0_min_valid_points;
  return stats.valid;
}

int Level0DistanceField::indexFor(double x, double y) const {
  if (distance_cells_.empty()) {
    return -1;
  }
  const int ix = static_cast<int>(std::floor((x - min_x_) / resolution_));
  const int iy = static_cast<int>(std::floor((y - min_y_) / resolution_));
  if (ix < 0 || iy < 0 || ix >= width_ || iy >= height_) {
    return -1;
  }
  return iy * width_ + ix;
}

Level0ProjectionClass Level0DistanceField::classify(double x, double y) const {
  if (x >= min_x_ && x <= max_x_ && y >= min_y_ && y <= max_y_) {
    return Level0ProjectionClass::Valid;
  }
  if (x >= min_x_ - boundary_padding_m_ &&
      x <= max_x_ + boundary_padding_m_ &&
      y >= min_y_ - boundary_padding_m_ &&
      y <= max_y_ + boundary_padding_m_) {
    return Level0ProjectionClass::BoundaryUnknown;
  }
  return Level0ProjectionClass::Invalid;
}

double Level0DistanceField::distanceAt(int index) const {
  if (index < 0 || static_cast<size_t>(index) >= distance_cells_.size()) {
    return kInfinity;
  }
  return distance_cells_[static_cast<size_t>(index)];
}

bool Level0DistanceField::scoreHypothesis(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& scan_level,
    const Eigen::Matrix4d& T_map_level,
    const GLParams& params,
    Level0Score& score) const {
  score = Level0Score{};
  if (!scan_level || scan_level->empty() || distance_cells_.empty() ||
      !T_map_level.allFinite()) {
    return false;
  }

  score.input_point_count = static_cast<int>(scan_level->size());
  double distance_sum = 0.0;
  for (const auto& point : *scan_level) {
    const Eigen::Vector4d level_point(point.x, point.y, point.z, 1.0);
    const Eigen::Vector4d map_point = T_map_level * level_point;
    const Level0ProjectionClass projection = classify(map_point.x(), map_point.y());
    if (projection == Level0ProjectionClass::Valid) {
      const int index = indexFor(map_point.x(), map_point.y());
      const double distance = distanceAt(index);
      if (std::isfinite(distance)) {
        ++score.valid_projection_points;
        distance_sum += std::min(distance, params.level0_max_obstacle_distance);
      } else {
        ++score.invalid_projection_points;
      }
    } else if (projection == Level0ProjectionClass::BoundaryUnknown) {
      ++score.boundary_unknown_points;
    } else {
      ++score.invalid_projection_points;
    }
  }

  score.lookup_count = static_cast<uint64_t>(score.input_point_count);
  const int scorable_count = score.valid_projection_points +
                             score.invalid_projection_points;
  score.valid_projection_ratio = score.valid_projection_points /
      static_cast<double>(std::max(1, scorable_count));
  score.invalid_projection_ratio = score.invalid_projection_points /
      static_cast<double>(std::max(1, scorable_count));
  score.boundary_unknown_ratio = score.boundary_unknown_points /
      static_cast<double>(std::max(1, score.input_point_count));
  if (score.valid_projection_points > 0) {
    score.mean_truncated_distance = distance_sum /
        static_cast<double>(score.valid_projection_points);
  }
  score.score = score.mean_truncated_distance +
      params.level0_invalid_projection_penalty * score.invalid_projection_ratio;
  score.valid = score.valid_projection_points >= params.level0_min_valid_points &&
      score.valid_projection_ratio >= params.level0_min_valid_projection_ratio &&
      score.boundary_unknown_ratio <= params.level0_max_boundary_unknown_ratio;
  return true;
}

}  // namespace localization
