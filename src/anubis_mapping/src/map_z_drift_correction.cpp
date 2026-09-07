#include "map_z_drift_correction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

namespace
{

double median(std::vector<double> values)
{
    if (values.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    return values.size() % 2U == 0U
        ? 0.5 * (values[middle - 1U] + values[middle])
        : values[middle];
}

std::string invalidRanges(const std::vector<bool>& used)
{
    std::ostringstream output;
    bool first_range = true;
    std::size_t index = 0U;
    while (index < used.size())
    {
        if (used[index])
        {
            ++index;
            continue;
        }
        const std::size_t begin = index;
        while (index + 1U < used.size() && !used[index + 1U])
        {
            ++index;
        }
        if (!first_range)
        {
            output << ';';
        }
        first_range = false;
        output << begin;
        if (index != begin)
        {
            output << '-' << index;
        }
        ++index;
    }
    return output.str();
}

bool validConfig(const anubis_mapping::MapZDriftCorrectionConfig& config)
{
    return std::isfinite(config.max_local_tilt_deg) &&
        config.max_local_tilt_deg > 0.0 &&
        config.max_local_tilt_deg <= 90.0 &&
        std::isfinite(config.max_local_residual_p95) &&
        config.max_local_residual_p95 > 0.0 &&
        std::isfinite(config.min_valid_fraction) &&
        config.min_valid_fraction > 0.0 && config.min_valid_fraction <= 1.0 &&
        std::isfinite(config.max_unobserved_gap_m) &&
        config.max_unobserved_gap_m > 0.0 &&
        std::isfinite(config.outlier_threshold_m) &&
        config.outlier_threshold_m > 0.0 && config.outlier_window > 0U &&
        std::isfinite(config.smoothness) && config.smoothness >= 0.0 &&
        std::isfinite(config.max_abs_correction_m) &&
        config.max_abs_correction_m > 0.0 &&
        std::isfinite(config.max_correction_slope) &&
        config.max_correction_slope > 0.0 &&
        std::isfinite(config.max_corrected_floor_span_m) &&
        config.max_corrected_floor_span_m > 0.0;
}

}  // namespace

namespace anubis_mapping
{

MapZDriftCorrectionResult solveMapZDriftCorrection(
    const std::vector<MapZDriftObservation>& observations,
    const MapZDriftCorrectionConfig& config)
{
    MapZDriftCorrectionResult result;
    const auto fail = [&](const std::string& reason)
    {
        result.success = false;
        result.failure_reason = reason;
        return result;
    };
    if (!validConfig(config) || observations.size() < 3U)
    {
        return fail("invalid configuration or too few observations");
    }
    for (std::size_t index = 0U; index < observations.size(); ++index)
    {
        if (!std::isfinite(observations[index].path_s) ||
            (index > 0U && observations[index].path_s <
                observations[index - 1U].path_s))
        {
            return fail("path coordinates are non-finite or not monotonic");
        }
    }

    result.used_observations.resize(observations.size(), false);
    for (std::size_t index = 0U; index < observations.size(); ++index)
    {
        const auto& observation = observations[index];
        result.used_observations[index] = observation.valid &&
            std::isfinite(observation.floor_z) &&
            std::isfinite(observation.local_tilt_deg) &&
            observation.local_tilt_deg <= config.max_local_tilt_deg &&
            std::isfinite(observation.residual_p95) &&
            observation.residual_p95 <= config.max_local_residual_p95 &&
            observation.independent_source_count >=
                config.min_independent_sources &&
            observation.sample_cells >= config.min_sample_cells;
        result.input_valid_count += result.used_observations[index] ? 1U : 0U;
    }

    const std::vector<bool> initially_valid = result.used_observations;
    for (std::size_t index = 0U; index < observations.size(); ++index)
    {
        if (!initially_valid[index])
        {
            continue;
        }
        const std::size_t begin = index > config.outlier_window
            ? index - config.outlier_window : 0U;
        const std::size_t end = std::min(
            observations.size(), index + config.outlier_window + 1U);
        std::vector<double> neighborhood;
        for (std::size_t neighbor = begin; neighbor < end; ++neighbor)
        {
            if (initially_valid[neighbor])
            {
                neighborhood.push_back(observations[neighbor].floor_z);
            }
        }
        if (neighborhood.size() >= 3U &&
            std::abs(observations[index].floor_z - median(neighborhood)) >
                config.outlier_threshold_m)
        {
            result.used_observations[index] = false;
            ++result.rejected_outlier_count;
        }
    }

    std::vector<std::size_t> valid_indices;
    std::vector<double> valid_floors;
    for (std::size_t index = 0U; index < observations.size(); ++index)
    {
        if (result.used_observations[index])
        {
            valid_indices.push_back(index);
            valid_floors.push_back(observations[index].floor_z);
        }
    }
    result.used_valid_count = valid_indices.size();
    result.invalid_index_ranges = invalidRanges(result.used_observations);
    result.valid_fraction = static_cast<double>(result.used_valid_count) /
        static_cast<double>(observations.size());
    const double total_path_length =
        observations.back().path_s - observations.front().path_s;
    result.total_path_length_m = total_path_length;
    for (std::size_t index = 1U; index < observations.size(); ++index)
    {
        result.max_adjacent_path_step_m = std::max(
            result.max_adjacent_path_step_m,
            observations[index].path_s - observations[index - 1U].path_s);
    }

    // Compute support as a union of the intervals actually used to form each
    // observation.  Source-aware mapping supplies one or more path-adjacent
    // Voronoi cells in support_intervals.  Keep those cells separate until
    // the union: a scalar min/max over sources would fill a path jump or a
    // revisit gap that has no ground evidence.  Legacy callers that only
    // provide scalar bounds retain the old single-interval fallback.
    struct SupportInterval
    {
        double begin = 0.0;
        double end = 0.0;
    };
    std::vector<SupportInterval> support_intervals;
    support_intervals.reserve(valid_indices.size());
    for (const std::size_t index : valid_indices)
    {
        const auto& observation = observations[index];
        const bool has_explicit_intervals = observation.has_explicit_support &&
            !observation.support_intervals.empty();
        if (has_explicit_intervals)
        {
            for (const auto& explicit_interval : observation.support_intervals)
            {
                double begin = explicit_interval.begin;
                double end = explicit_interval.end;
                begin = std::max(observations.front().path_s, begin);
                end = std::min(observations.back().path_s, end);
                if (std::isfinite(begin) && std::isfinite(end) &&
                    end >= begin)
                {
                    support_intervals.push_back({begin, end});
                }
            }
        }
        if (!has_explicit_intervals)
        {
            const bool has_explicit_interval =
                observation.has_explicit_support &&
                std::isfinite(observation.support_s_min) &&
                std::isfinite(observation.support_s_max) &&
                observation.support_s_max >= observation.support_s_min;
            double begin = 0.0;
            double end = 0.0;
            if (has_explicit_interval)
            {
                begin = observation.support_s_min;
                end = observation.support_s_max;
            }
            else
            {
                begin = index == 0U
                    ? observations.front().path_s
                    : 0.5 * (observations[index - 1U].path_s +
                        observations[index].path_s);
                end = index + 1U == observations.size()
                    ? observations.back().path_s
                    : 0.5 * (observations[index].path_s +
                        observations[index + 1U].path_s);
            }
            begin = std::max(observations.front().path_s, begin);
            end = std::min(observations.back().path_s, end);
            if (std::isfinite(begin) && std::isfinite(end) && end >= begin)
            {
                support_intervals.push_back({begin, end});
            }
        }
    }
    std::sort(support_intervals.begin(), support_intervals.end(),
              [](const SupportInterval& lhs, const SupportInterval& rhs) {
                  return lhs.begin < rhs.begin;
              });
    std::vector<SupportInterval> merged_intervals;
    for (const auto& interval : support_intervals)
    {
        if (merged_intervals.empty() ||
            interval.begin > merged_intervals.back().end)
        {
            merged_intervals.push_back(interval);
        }
        else
        {
            merged_intervals.back().end = std::max(
                merged_intervals.back().end, interval.end);
        }
    }
    result.max_unobserved_gap_m = 0.0;
    if (total_path_length > std::numeric_limits<double>::epsilon())
    {
        double supported_path_length = 0.0;
        double cursor = observations.front().path_s;
        for (const auto& interval : merged_intervals)
        {
            result.max_unobserved_gap_m = std::max(
                result.max_unobserved_gap_m, interval.begin - cursor);
            supported_path_length += std::max(0.0, interval.end - interval.begin);
            cursor = std::max(cursor, interval.end);
        }
        result.max_unobserved_gap_m = std::max(
            result.max_unobserved_gap_m,
            observations.back().path_s - cursor);
        result.path_coverage_fraction = std::clamp(
            supported_path_length / total_path_length, 0.0, 1.0);
    }
    else
    {
        result.path_coverage_fraction = result.valid_fraction;
        result.max_unobserved_gap_m = 0.0;
    }
    if (result.valid_fraction < config.min_valid_fraction ||
        valid_indices.size() < 3U)
    {
        return fail("valid ground observation fraction is below the configured minimum");
    }
    if (result.path_coverage_fraction < config.min_valid_fraction)
    {
        return fail("valid ground observation path coverage is below the configured minimum");
    }
    if (result.max_unobserved_gap_m > config.max_unobserved_gap_m)
    {
        return fail("ground observation gap exceeds the configured path limit");
    }

    result.target_floor_z = median(valid_floors);
    if (!std::isfinite(result.target_floor_z))
    {
        return fail("target floor is non-finite");
    }

    const std::size_t count = observations.size();
    std::vector<double> diagonal(count, 0.0);
    std::vector<double> lower(count > 0U ? count - 1U : 0U, 0.0);
    std::vector<double> rhs(count, 0.0);
    for (std::size_t index = 0U; index < count; ++index)
    {
        if (result.used_observations[index])
        {
            diagonal[index] += 1.0;
            rhs[index] = result.target_floor_z - observations[index].floor_z;
        }
    }
    for (std::size_t index = 1U; index < count; ++index)
    {
        const double distance = std::max(
            0.10, observations[index].path_s - observations[index - 1U].path_s);
        const double weight = config.smoothness / distance;
        diagonal[index - 1U] += weight;
        diagonal[index] += weight;
        lower[index - 1U] = -weight;
    }
    for (std::size_t index = 1U; index < count; ++index)
    {
        if (!std::isfinite(diagonal[index - 1U]) ||
            diagonal[index - 1U] <= std::numeric_limits<double>::epsilon())
        {
            return fail("vertical correction system is singular");
        }
        const double factor = lower[index - 1U] / diagonal[index - 1U];
        diagonal[index] -= factor * lower[index - 1U];
        rhs[index] -= factor * rhs[index - 1U];
    }
    result.corrections.assign(count, 0.0);
    if (!std::isfinite(diagonal.back()) ||
        diagonal.back() <= std::numeric_limits<double>::epsilon())
    {
        return fail("vertical correction system is singular");
    }
    result.corrections.back() = rhs.back() / diagonal.back();
    for (std::size_t reverse = count - 1U; reverse > 0U; --reverse)
    {
        const std::size_t index = reverse - 1U;
        result.corrections[index] =
            (rhs[index] - lower[index] * result.corrections[index + 1U]) /
            diagonal[index];
    }
    const double gauge = median(result.corrections);
    for (double& correction : result.corrections)
    {
        correction -= gauge;
        if (!std::isfinite(correction))
        {
            return fail("vertical correction is non-finite");
        }
        result.max_abs_correction_m = std::max(
            result.max_abs_correction_m, std::abs(correction));
    }
    if (result.max_abs_correction_m > config.max_abs_correction_m)
    {
        return fail("vertical correction exceeds the configured absolute limit");
    }
    for (std::size_t index = 1U; index < count; ++index)
    {
        const double distance = std::max(
            0.10, observations[index].path_s - observations[index - 1U].path_s);
        result.max_correction_slope = std::max(
            result.max_correction_slope,
            std::abs(result.corrections[index] -
                result.corrections[index - 1U]) / distance);
    }
    if (result.max_correction_slope > config.max_correction_slope)
    {
        return fail("vertical correction slope exceeds the configured limit");
    }

    result.corrected_floor_min = std::numeric_limits<double>::max();
    result.corrected_floor_max = std::numeric_limits<double>::lowest();
    for (const std::size_t index : valid_indices)
    {
        const double corrected = observations[index].floor_z +
            result.corrections[index];
        result.corrected_floor_min = std::min(
            result.corrected_floor_min, corrected);
        result.corrected_floor_max = std::max(
            result.corrected_floor_max, corrected);
    }
    if (!std::isfinite(result.corrected_floor_min) ||
        !std::isfinite(result.corrected_floor_max) ||
        result.corrected_floor_max - result.corrected_floor_min >
            config.max_corrected_floor_span_m)
    {
        return fail("corrected per-keyframe floor span exceeds the configured limit");
    }

    result.success = true;
    result.failure_reason.clear();
    return result;
}

}  // namespace anubis_mapping
