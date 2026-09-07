#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace anubis_mapping
{

// A path interval backed by one concrete source (normally the source's
// Voronoi cell between adjacent keyframes).  Keeping the intervals separate
// is important: a scalar min/max span would fill a path jump or a revisit
// gap that has no ground evidence.
struct MapZSupportInterval
{
    double begin = 0.0;
    double end = 0.0;
};

struct MapZDriftObservation
{
    double path_s = 0.0;
    double floor_z = 0.0;
    double local_tilt_deg = 0.0;
    double residual_p95 = 0.0;
    bool valid = false;
    std::string failure_reason;
    // Save-time audit fields.  A valid observation may be formed from a
    // bounded temporal neighborhood when one Mid-360 frame is sparse; these
    // fields make the actual support explicit instead of hiding it in a
    // scalar floor estimate.
    std::size_t selected_window = 0U;
    std::size_t candidate_points = 0U;
    std::size_t sample_cells = 0U;
    std::size_t inlier_cells = 0U;
    std::size_t source_count = 0U;
    std::size_t independent_source_count = 0U;
    // Legacy aggregate bounds retained for CSV/readability.  The solver uses
    // support_intervals when present so disjoint source cells are not merged
    // into one artificial min/max span.
    double support_s_min = 0.0;
    double support_s_max = 0.0;
    bool has_explicit_support = false;
    std::string source_ids;
    std::string window_diagnostics;
    std::vector<MapZSupportInterval> support_intervals;
};

struct MapZDriftCorrectionConfig
{
    // Optional defense-in-depth gates.  MappingAlg sets these to its
    // source-aware local-fit requirements; zero disables count gates for
    // callers that only provide scalar observations.
    std::size_t min_independent_sources = 0U;
    std::size_t min_sample_cells = 0U;
    double max_local_tilt_deg = 90.0;
    double max_local_residual_p95 = 1.0;
    double min_valid_fraction = 0.80;
    double max_unobserved_gap_m = 2.0;
    double outlier_threshold_m = 0.12;
    std::size_t outlier_window = 3U;
    double smoothness = 0.03;
    double max_abs_correction_m = 0.45;
    double max_correction_slope = 0.25;
    double max_corrected_floor_span_m = 0.04;
};

struct MapZDriftCorrectionResult
{
    bool success = false;
    std::string failure_reason;
    std::vector<double> corrections;
    std::vector<bool> used_observations;
    std::size_t input_valid_count = 0U;
    std::size_t used_valid_count = 0U;
    std::size_t rejected_outlier_count = 0U;
    double valid_fraction = 0.0;
    double path_coverage_fraction = 0.0;
    double max_unobserved_gap_m = 0.0;
    double target_floor_z = 0.0;
    double max_abs_correction_m = 0.0;
    double max_correction_slope = 0.0;
    double corrected_floor_min = 0.0;
    double corrected_floor_max = 0.0;
    double total_path_length_m = 0.0;
    double max_adjacent_path_step_m = 0.0;
    std::string invalid_index_ranges;
};

MapZDriftCorrectionResult solveMapZDriftCorrection(
    const std::vector<MapZDriftObservation>& observations,
    const MapZDriftCorrectionConfig& config);

}  // namespace anubis_mapping
