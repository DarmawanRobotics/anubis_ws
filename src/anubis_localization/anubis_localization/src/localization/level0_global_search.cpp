#include "localization/level0_global_search.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Geometry>

namespace localization {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<double> makeAxisSamples(double minimum,
                                    double maximum,
                                    double stride) {
  std::vector<double> samples;
  if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
      !std::isfinite(stride) || stride <= 0.0 || maximum < minimum) {
    return samples;
  }

  const double span = maximum - minimum;
  const size_t regular_steps = static_cast<size_t>(std::floor(span / stride));
  samples.reserve(regular_steps + 2U);
  for (size_t index = 0; index <= regular_steps; ++index) {
    samples.push_back(minimum + static_cast<double>(index) * stride);
  }
  const double tolerance = 1e-9 * std::max(1.0, std::abs(maximum));
  if (maximum - samples.back() > tolerance) {
    samples.push_back(maximum);
  } else {
    samples.back() = maximum;
  }
  return samples;
}

double circularYawDistanceDeg(double first, double second) {
  return std::abs(std::remainder(first - second, 360.0));
}

bool hypothesisLess(const Level0Hypothesis& first,
                    const Level0Hypothesis& second) {
  if (first.score.score != second.score.score) {
    return first.score.score < second.score.score;
  }
  if (first.score.valid_projection_ratio !=
      second.score.valid_projection_ratio) {
    return first.score.valid_projection_ratio >
        second.score.valid_projection_ratio;
  }
  if (first.grid_cell_id != second.grid_cell_id) {
    return first.grid_cell_id < second.grid_cell_id;
  }
  return first.yaw_bin < second.yaw_bin;
}

bool suppressedBy(const Level0Hypothesis& candidate,
                  const Level0Hypothesis& selected,
                  const GLParams& params) {
  const double dx = candidate.T_map_level(0, 3) - selected.T_map_level(0, 3);
  const double dy = candidate.T_map_level(1, 3) - selected.T_map_level(1, 3);
  return std::hypot(dx, dy) <= params.level0_region_nms_xy &&
      circularYawDistanceDeg(candidate.yaw_deg, selected.yaw_deg) <=
          params.level0_region_nms_yaw_deg;
}

bool checkedMultiply(uint64_t first, uint64_t second, uint64_t& product) {
  if (first != 0U && second > std::numeric_limits<uint64_t>::max() / first) {
    return false;
  }
  product = first * second;
  return true;
}

}  // namespace

bool selectLevel0TopRegions(
    const std::vector<Level0Hypothesis>& hypotheses,
    const GLParams& params,
    std::vector<Level0Hypothesis>& top_regions) {
  top_regions.clear();
  if (params.level0_top_regions <= 0 ||
      !std::isfinite(params.level0_region_nms_xy) ||
      params.level0_region_nms_xy <= 0.0 ||
      !std::isfinite(params.level0_region_nms_yaw_deg) ||
      params.level0_region_nms_yaw_deg <= 0.0) {
    return false;
  }

  std::vector<Level0Hypothesis> sorted;
  sorted.reserve(hypotheses.size());
  for (const Level0Hypothesis& hypothesis : hypotheses) {
    if (hypothesis.score.valid && std::isfinite(hypothesis.score.score) &&
        hypothesis.T_map_level.allFinite()) {
      sorted.push_back(hypothesis);
    }
  }
  std::sort(sorted.begin(), sorted.end(), hypothesisLess);

  for (const Level0Hypothesis& candidate : sorted) {
    bool suppressed = false;
    for (const Level0Hypothesis& selected : top_regions) {
      if (suppressedBy(candidate, selected, params)) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      top_regions.push_back(candidate);
      if (static_cast<int>(top_regions.size()) >= params.level0_top_regions) {
        break;
      }
    }
  }
  return true;
}

bool runLevel0GlobalSearch(
    const Level0DistanceField& distance_field,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& scan_level,
    const GLParams& params,
    std::vector<Level0Hypothesis>& top_regions,
    Level0SearchStats& stats,
    GLSummaryCounts& summary) {
  top_regions.clear();
  stats = Level0SearchStats{};
  summary = GLSummaryCounts{};
  if (distance_field.empty() || !scan_level || scan_level->empty() ||
      params.level0_grid_stride_xy <= 0.0 ||
      params.level0_yaw_samples <= 0 || params.max_candidates_total <= 0) {
    return false;
  }

  const std::vector<double> x_samples = makeAxisSamples(
      distance_field.minX(), distance_field.maxX(),
      params.level0_grid_stride_xy);
  const std::vector<double> y_samples = makeAxisSamples(
      distance_field.minY(), distance_field.maxY(),
      params.level0_grid_stride_xy);
  if (x_samples.empty() || y_samples.empty()) {
    return false;
  }

  uint64_t grid_cell_count = 0U;
  uint64_t hypothesis_count = 0U;
  uint64_t lookup_count_upper_bound = 0U;
  if (!checkedMultiply(static_cast<uint64_t>(x_samples.size()),
                       static_cast<uint64_t>(y_samples.size()),
                       grid_cell_count) ||
      !checkedMultiply(grid_cell_count,
                       static_cast<uint64_t>(params.level0_yaw_samples),
                       hypothesis_count) ||
      !checkedMultiply(hypothesis_count,
                       static_cast<uint64_t>(scan_level->size()),
                       lookup_count_upper_bound) ||
      grid_cell_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }

  stats.grid_cell_count = static_cast<int>(grid_cell_count);
  stats.hypothesis_count = hypothesis_count;
  stats.lookup_count_upper_bound = lookup_count_upper_bound;
  stats.batch_count = static_cast<int>(
      (hypothesis_count + static_cast<uint64_t>(params.max_candidates_total) - 1U) /
      static_cast<uint64_t>(params.max_candidates_total));
  summary.level = 0;
  summary.batch_count = stats.batch_count;
  summary.total_cells = stats.grid_cell_count;
  summary.level0_ground_unavailable = params.level0_base_ground_offset < 0.0;

  std::vector<Level0Hypothesis> valid_hypotheses;
  valid_hypotheses.reserve(static_cast<size_t>(std::min<uint64_t>(
      hypothesis_count, static_cast<uint64_t>(1000000U))));
  const double yaw_step_deg = 360.0 /
      static_cast<double>(params.level0_yaw_samples);
  const double map_base_z = distance_field.floorZ() +
      std::max(0.0, params.level0_base_ground_offset);
  uint64_t hypothesis_index = 0U;
  int grid_cell_id = 0;
  for (double y : y_samples) {
    for (double x : x_samples) {
      for (int yaw_bin = 0; yaw_bin < params.level0_yaw_samples; ++yaw_bin) {
        Level0Hypothesis hypothesis;
        hypothesis.grid_cell_id = grid_cell_id;
        hypothesis.yaw_bin = yaw_bin;
        hypothesis.rotation_batch = static_cast<int>(
            hypothesis_index / static_cast<uint64_t>(params.max_candidates_total));
        hypothesis.yaw_deg = static_cast<double>(yaw_bin) * yaw_step_deg;
        hypothesis.T_map_level.block<3, 3>(0, 0) = Eigen::AngleAxisd(
            hypothesis.yaw_deg * kPi / 180.0,
            Eigen::Vector3d::UnitZ()).toRotationMatrix();
        hypothesis.T_map_level(0, 3) = x;
        hypothesis.T_map_level(1, 3) = y;
        hypothesis.T_map_level(2, 3) = map_base_z;
        if (!distance_field.scoreHypothesis(
                scan_level, hypothesis.T_map_level, params, hypothesis.score)) {
          return false;
        }
        accumulateLevel0Summary(hypothesis.score, summary);
        if (hypothesis.score.valid) {
          valid_hypotheses.push_back(hypothesis);
          ++stats.valid_hypothesis_count;
        } else {
          ++stats.invalid_hypothesis_count;
        }
        ++hypothesis_index;
      }
      ++grid_cell_id;
    }
  }

  if (!selectLevel0TopRegions(valid_hypotheses, params, top_regions)) {
    return false;
  }
  summary.covered_cells = stats.grid_cell_count;
  summary.batch_index = std::max(0, stats.batch_count - 1);
  stats.complete = true;
  return true;
}

}  // namespace localization
