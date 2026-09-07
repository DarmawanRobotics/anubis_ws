#ifndef GLOBAL_LOCALIZATION_VALIDATION_HPP
#define GLOBAL_LOCALIZATION_VALIDATION_HPP

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "localization/global_localization_types.hpp"

namespace localization {

// The old name is kept as a launch/configuration compatibility alias.  It is
// deliberately resolved here, rather than by OR-ing two booleans in the node:
// an explicit disagreement must never enable an automatic episode.
struct GLEpisodeRuntimeResolution {
  bool enabled = false;
  bool legacy_parameter_used = false;
  bool conflict = false;
};

inline GLEpisodeRuntimeResolution resolveGLEpisodeRuntimeEnabled(
    const std::optional<bool>& official,
    const std::optional<bool>& legacy) {
  GLEpisodeRuntimeResolution resolution;
  resolution.legacy_parameter_used = legacy.has_value();
  if (official.has_value() && legacy.has_value()) {
    if (*official != *legacy) {
      resolution.conflict = true;
      // Fail closed.  In particular, new=false + old=true must stay false.
      return resolution;
    }
    resolution.enabled = *official;
    return resolution;
  }
  if (official.has_value()) {
    resolution.enabled = *official;
  } else if (legacy.has_value()) {
    resolution.enabled = *legacy;
  }
  return resolution;
}

inline std::string trimAsciiWhitespace(const std::string& value) {
  size_t first = 0;
  while (first < value.size() &&
         std::isspace(static_cast<unsigned char>(value[first]))) {
    ++first;
  }
  size_t last = value.size();
  while (last > first &&
         std::isspace(static_cast<unsigned char>(value[last - 1]))) {
    --last;
  }
  return value.substr(first, last - first);
}

inline std::string lowerAscii(const std::string& value) {
  std::string lowered;
  lowered.reserve(value.size());
  for (const unsigned char character : value) {
    lowered.push_back(static_cast<char>(std::tolower(character)));
  }
  return lowered;
}

// This is intentionally a syntax gate only.  It does not prove that the map,
// parameter set, manifest, and report hashes actually match this identifier;
// that binding belongs to the M2b evidence/manifest verification workflow.
inline bool isUsableM2bApprovalId(const std::string& approval_id) {
  const std::string value = trimAsciiWhitespace(approval_id);
  constexpr size_t kMinLength = 11;  // "m2b:a:b:c:d"
  constexpr size_t kMaxLength = 256;
  if (value.size() < kMinLength || value.size() > kMaxLength ||
      value.rfind("m2b:", 0) != 0) {
    return false;
  }

  // Require an approval record plus map, parameter, and manifest fields.  The
  // field contents remain deliberately opaque; only their safe transport
  // syntax is checked here.
  size_t field_count = 0;
  size_t field_start = 4;
  bool field_has_alphanumeric = false;
  bool field_has_nonzero_alphanumeric = false;
  if (field_start >= value.size()) {
    return false;
  }
  for (size_t index = field_start; index <= value.size(); ++index) {
    const bool at_end = index == value.size();
    if (at_end || value[index] == ':') {
      if (index == field_start || !field_has_alphanumeric ||
          !field_has_nonzero_alphanumeric) {
        return false;
      }
      ++field_count;
      field_start = index + 1;
      field_has_alphanumeric = false;
      field_has_nonzero_alphanumeric = false;
      if (at_end) {
        break;
      }
      continue;
    }
    const char character = value[index];
    const unsigned char byte = static_cast<unsigned char>(character);
    const bool safe_character =
        (byte >= static_cast<unsigned char>('a') &&
         byte <= static_cast<unsigned char>('z')) ||
        (byte >= static_cast<unsigned char>('A') &&
         byte <= static_cast<unsigned char>('Z')) ||
        (byte >= static_cast<unsigned char>('0') &&
         byte <= static_cast<unsigned char>('9')) ||
        character == '-' || character == '_' || character == '.';
    if (!safe_character) {
      return false;
    }
    field_has_alphanumeric = field_has_alphanumeric ||
        ((byte >= static_cast<unsigned char>('a') &&
          byte <= static_cast<unsigned char>('z')) ||
         (byte >= static_cast<unsigned char>('A') &&
          byte <= static_cast<unsigned char>('Z')) ||
         (byte >= static_cast<unsigned char>('0') &&
          byte <= static_cast<unsigned char>('9')));
    field_has_nonzero_alphanumeric = field_has_nonzero_alphanumeric ||
        ((byte >= static_cast<unsigned char>('a') &&
          byte <= static_cast<unsigned char>('z')) ||
         (byte >= static_cast<unsigned char>('A') &&
          byte <= static_cast<unsigned char>('Z')) ||
         (byte >= static_cast<unsigned char>('1') &&
          byte <= static_cast<unsigned char>('9')));
  }
  if (field_count < 4) {
    return false;
  }

  // Reject known placeholders as tokens, including variants such as
  // "debug_not_approved".  Token-level matching keeps a legitimate opaque
  // value such as "unit-fixture" usable in syntax-only unit tests.
  const std::string lowered = lowerAscii(value);
  std::string token;
  const auto is_token_separator = [](char character) {
    return character == ':' || character == '-' || character == '_' ||
           character == '.';
  };
  std::vector<std::string> tokens;
  for (const char character : lowered) {
    if (is_token_separator(character)) {
      if (!token.empty()) {
        tokens.push_back(token);
        token.clear();
      }
    } else {
      token.push_back(character);
    }
  }
  if (!token.empty()) {
    tokens.push_back(token);
  }
  constexpr const char* kPlaceholderTokens[] = {
      "todo",       "tbd",       "pending",  "changeme", "placeholder",
      "example",    "sample",    "test",     "dummy",    "none",
      "null",       "unknown",   "unapproved"};
  for (const std::string& candidate : tokens) {
    for (const char* placeholder : kPlaceholderTokens) {
      if (candidate == placeholder) {
        return false;
      }
    }
    if (candidate == "notapproved" || candidate == "debugnotapproved") {
      return false;
    }
  }
  // The phrase is checked after separators have become spaces, so common
  // underscore/hyphen/case variants are rejected consistently.
  std::string phrase;
  for (const char character : lowered) {
    phrase.push_back(is_token_separator(character) ? ' ' : character);
  }
  if (phrase.find("not approved") != std::string::npos ||
      phrase.find("debug not approved") != std::string::npos) {
    return false;
  }
  return true;
}

inline std::vector<std::string> validateAutoConfirmContract(
    const GLParams& params) {
  std::vector<std::string> errors;
  if (!params.auto_confirm_enabled) {
    return errors;
  }
  const auto require = [&errors](bool condition, const char* message) {
    if (!condition) {
      errors.emplace_back(message);
    }
  };
  require(params.flat_single_level_map,
          "auto_confirm_enabled requires flat_single_level_map=true");
  require(!params.allow_degraded_fallback,
          "auto_confirm_enabled requires allow_degraded_fallback=false");
  require(std::isfinite(params.level0_base_ground_offset) &&
              params.level0_base_ground_offset >= 0.0,
          "auto_confirm_enabled requires a calibrated base ground offset");
  require(isUsableM2bApprovalId(params.m2b_approval_id),
          "auto_confirm_enabled requires a usable m2b_approval_id syntax");
  return errors;
}

inline std::vector<std::string> validateGLParams(const GLParams& params) {
  constexpr double kStrictMargin = 1e-6;
  std::vector<std::string> errors;
  const auto require = [&errors](bool condition, const char* message) {
    if (!condition) {
      errors.emplace_back(message);
    }
  };
  const auto positive = [&require](double value, const char* name) {
    require(std::isfinite(value) && value > 0.0, name);
  };

  positive(params.level0_grid_stride_xy, "level0_grid_stride_xy must be finite and > 0");
  require(params.gl_recall_source == "grid" || params.gl_recall_source == "retrieval",
          "gl_recall_source must be grid or retrieval");
  require(params.retrieval_topk > 0, "retrieval_topk must be > 0");
  positive(params.retrieval_nms_xy, "retrieval_nms_xy must be finite and > 0");
  require(std::isfinite(params.retrieval_max_distance) && params.retrieval_max_distance >= 0.0,
          "retrieval_max_distance must be finite and >= 0");
  require(params.retrieval_min_points > 0, "retrieval_min_points must be > 0");
  require(std::isfinite(params.retrieval_yaw_step_deg) &&
          params.retrieval_yaw_step_deg > 0.0 &&
          params.retrieval_yaw_step_deg <= params.retrieval_yaw_half_range_deg &&
          params.retrieval_yaw_half_range_deg <= 45.0,
          "retrieval yaw limits must satisfy 0 < step <= half_range <= 45");
  require(std::isfinite(params.retrieval_search_radius_xy) &&
          params.retrieval_search_radius_xy >= params.retrieval_seed_stride_xy &&
          params.retrieval_seed_stride_xy > params.support_seed_xy,
          "retrieval radius/stride must satisfy radius >= stride > support_seed_xy");
  require(params.level0_yaw_samples > 0, "level0_yaw_samples must be > 0");
  require(params.level0_top_regions > 0, "level0_top_regions must be > 0");
  positive(params.level0_distance_field_resolution,
           "level0_distance_field_resolution must be finite and > 0");
  positive(params.level0_max_obstacle_distance,
           "level0_max_obstacle_distance must be finite and > 0");
  require(std::isfinite(params.level0_base_ground_offset),
          "level0_base_ground_offset must be finite");
  require(std::isfinite(params.level0_min_point_height) &&
          std::isfinite(params.level0_max_point_height) &&
          params.level0_min_point_height > 0.0 &&
          params.level0_min_point_height < params.level0_max_point_height,
          "point height limits must satisfy 0 < min < max");
  require(std::isfinite(params.level0_min_range) &&
          std::isfinite(params.level0_max_range) &&
          params.level0_min_range > 0.0 &&
          params.level0_min_range < params.level0_max_range,
          "range limits must satisfy 0 < min < max");
  require(params.level0_min_valid_points > 0 &&
          params.level0_max_scan_points >= params.level0_min_valid_points,
          "scan point limits must satisfy max >= min > 0");
  positive(params.level0_raw_imu_gravity_window_s,
           "level0_raw_imu_gravity_window_s must be finite and > 0");
  require(params.level0_raw_imu_gravity_min_samples >= 2,
          "level0_raw_imu_gravity_min_samples must be >= 2");
  require(std::isfinite(params.level0_gravity_time_offset_ms),
          "level0_gravity_time_offset_ms must be finite");
  positive(params.level0_gravity_max_age_ms,
           "level0_gravity_max_age_ms must be finite and > 0");
  positive(params.level0_gravity_max_uncertainty_deg,
           "level0_gravity_max_uncertainty_deg must be finite and > 0");
  require(std::isfinite(params.level0_unaligned_max_abs_rp_deg) &&
          std::isfinite(params.level0_anchor_max_abs_rp_deg) &&
          params.level0_unaligned_max_abs_rp_deg > 0.0 &&
          params.level0_unaligned_max_abs_rp_deg < params.level0_anchor_max_abs_rp_deg &&
          params.level0_anchor_max_abs_rp_deg <=
            std::min(params.max_abs_roll_deg, params.max_abs_pitch_deg),
          "RP limits must satisfy 0 < unaligned < anchor <= absolute limits");
  positive(params.level0_anchor_max_angular_velocity_rps,
           "level0_anchor_max_angular_velocity_rps must be finite and > 0");
  positive(params.level0_gravity_max_accel_residual_mps2,
           "level0_gravity_max_accel_residual_mps2 must be finite and > 0");
  require(params.level0_anchor_recapture_limit > 0,
          "level0_anchor_recapture_limit must be > 0");

  positive(params.level0_anchor_max_linear_velocity_mps,
           "level0_anchor_max_linear_velocity_mps must be finite and > 0");
  if (params.level0_raw_imu_gravity_filter_enabled) {
    require(std::isfinite(params.level0_raw_imu_gravity_max_angular_velocity_rps) &&
            params.level0_raw_imu_gravity_max_angular_velocity_rps > 0.0 &&
            params.level0_raw_imu_gravity_max_angular_velocity_rps <
                params.level0_anchor_max_angular_velocity_rps,
            "level0_raw_imu_gravity_max_angular_velocity_rps must be > 0 and below level0_anchor_max_angular_velocity_rps");
  }

  require(std::isfinite(params.level0_min_valid_projection_ratio) &&
          params.level0_min_valid_projection_ratio > 0.0 &&
          params.level0_min_valid_projection_ratio <= 1.0,
          "level0_min_valid_projection_ratio must be in (0, 1]");
  require(std::isfinite(params.level0_invalid_projection_penalty) &&
          params.level0_invalid_projection_penalty >= 0.0,
          "level0_invalid_projection_penalty must be finite and >= 0");
  require(std::isfinite(params.level0_boundary_padding_m) &&
          params.level0_boundary_padding_m >= 0.0,
          "level0_boundary_padding_m must be finite and >= 0");
  require(std::isfinite(params.level0_max_boundary_unknown_ratio) &&
          params.level0_max_boundary_unknown_ratio >= 0.0 &&
          params.level0_max_boundary_unknown_ratio < 1.0,
          "level0_max_boundary_unknown_ratio must be in [0, 1)");
  positive(params.level0_region_nms_xy,
           "level0_region_nms_xy must be finite and > 0");
  positive(params.level0_region_nms_yaw_deg,
           "level0_region_nms_yaw_deg must be finite and > 0");

  positive(params.level1_search_radius_xy,
           "level1_search_radius_xy must be finite and > 0");
  require(std::isfinite(params.level1_seed_stride_xy) &&
          params.level1_seed_stride_xy > params.support_seed_xy + kStrictMargin,
          "level1_seed_stride_xy must exceed support_seed_xy by more than epsilon");
  require(std::isfinite(params.level1_search_radius_xy) &&
          params.level1_search_radius_xy >= params.level1_seed_stride_xy,
          "level1_search_radius_xy must be >= level1_seed_stride_xy");
  require(std::isfinite(params.level1_yaw_step_deg) &&
          params.level1_yaw_step_deg > 0.0 && params.level1_yaw_step_deg <= 15.0,
          "level1_yaw_step_deg must be in (0, 15]");
  require(params.max_candidates_total > 0, "max_candidates_total must be > 0");
  require(params.max_refine_candidates > 0 &&
          params.max_refine_candidates <= params.max_candidates_total,
          "max_refine_candidates must be in (0, max_candidates_total]");
  positive(params.seed_dedup_xy, "seed_dedup_xy must be finite and > 0");
  positive(params.seed_dedup_yaw_deg, "seed_dedup_yaw_deg must be finite and > 0");
  require(std::isfinite(params.grid_z_percentile) &&
          params.grid_z_percentile >= 0.0 && params.grid_z_percentile <= 1.0,
          "grid_z_percentile must be in [0, 1]");

  positive(params.coarse_leaf_map, "coarse_leaf_map must be finite and > 0");
  positive(params.coarse_leaf_src, "coarse_leaf_src must be finite and > 0");
  require(params.coarse_max_iter > 0, "coarse_max_iter must be > 0");
  positive(params.coarse_roi_radius, "coarse_roi_radius must be finite and > 0");
  positive(params.coarse_max_corr_dist,
           "coarse_max_corr_dist must be finite and > 0");
  require(std::isfinite(params.coarse_min_overlap) &&
          params.coarse_min_overlap > 0.0 && params.coarse_min_overlap <= 1.0,
          "coarse_min_overlap must be in (0, 1]");
  require(params.coarse_min_inliers > 0, "coarse_min_inliers must be > 0");
  positive(params.coarse_max_p90_residual,
           "coarse_max_p90_residual must be finite and > 0");

  positive(params.refine_leaf_src, "refine_leaf_src must be finite and > 0");
  require(params.refine_max_iter > 0, "refine_max_iter must be > 0");
  positive(params.refine_fitness_threshold,
           "refine_fitness_threshold must be finite and > 0");
  positive(params.max_correction_xy, "max_correction_xy must be finite and > 0");
  positive(params.max_correction_z, "max_correction_z must be finite and > 0");
  positive(params.max_yaw_delta_deg, "max_yaw_delta_deg must be finite and > 0");
  positive(params.max_rp_delta_deg, "max_rp_delta_deg must be finite and > 0");
  positive(params.max_abs_roll_deg, "max_abs_roll_deg must be finite and > 0");
  positive(params.max_abs_pitch_deg, "max_abs_pitch_deg must be finite and > 0");

  positive(params.cluster_xy, "cluster_xy must be finite and > 0");
  positive(params.cluster_yaw_deg, "cluster_yaw_deg must be finite and > 0");
  require(std::isfinite(params.support_seed_xy) &&
          params.support_seed_xy > 2.0 * params.cluster_xy + kStrictMargin,
          "support_seed_xy must exceed twice cluster_xy by more than epsilon");
  require(params.min_support_groups > 0, "min_support_groups must be > 0");
  require(params.max_refine_candidates >= params.min_support_groups,
          "max_refine_candidates must be >= min_support_groups");
  positive(params.ambiguous_pose_distance,
           "ambiguous_pose_distance must be finite and > 0");
  require(std::isfinite(params.ambiguous_score_ratio) &&
          params.ambiguous_score_ratio > 1.0,
          "ambiguous_score_ratio must be finite and > 1");

  // [2026-08-14] Static confirmation replaced the motion-driven vote.
  require(params.gl_confirm_frames > 0, "gl_confirm_frames must be > 0");
  positive(params.gl_confirm_xy_tol, "gl_confirm_xy_tol must be finite and > 0");
  positive(params.gl_confirm_yaw_tol_deg,
           "gl_confirm_yaw_tol_deg must be finite and > 0");
  positive(params.gl_probe_period_s, "gl_probe_period_s must be finite and > 0");
  positive(params.gl_attempt_timeout_s,
           "gl_attempt_timeout_s must be finite and > 0");
  require(std::isfinite(params.gl_attempt_timeout_s) &&
          params.gl_attempt_timeout_s >
              params.gl_probe_period_s * params.gl_confirm_frames,
          "gl_attempt_timeout_s must allow at least gl_confirm_frames probes");
  require(std::isfinite(params.min_retry_interval_s) && params.min_retry_interval_s >= 0.0,
          "min_retry_interval_s must be finite and >= 0");

  positive(params.per_call_deadline_ms,
           "per_call_deadline_ms must be finite and > 0");
  positive(params.episode_timeout_s, "episode_timeout_s must be finite and > 0");
  positive(params.max_scan_age_after_gl_s,
           "max_scan_age_after_gl_s must be finite and > 0");
  positive(params.shadow_gl_budget_ms, "shadow_gl_budget_ms must be finite and > 0");
  positive(params.shadow_gl_period_s, "shadow_gl_period_s must be finite and > 0");
  positive(params.auto_confirm_verify_timeout_s,
           "auto_confirm_verify_timeout_s must be finite and > 0");

  if (params.dump_candidates_csv) {
    require(!params.candidates_csv_path.empty(),
            "dump_candidates_csv requires a non-empty candidates_csv_path");
    require(!params.candidates_csv_episode_id.empty(),
            "dump_candidates_csv requires a non-empty candidates_csv_episode_id");
  }

  if (params.advisory_suggestion_enabled) {
    require(std::isfinite(params.advisory_min_confidence) &&
            params.advisory_min_confidence > 0.0 &&
            params.advisory_min_confidence <= 1.0,
            "advisory_min_confidence must be in (0, 1] when advisory is enabled");
    require(!params.advisory_calibration_id.empty(),
            "advisory mode requires a non-empty advisory_calibration_id");
  } else {
    require(params.advisory_min_confidence == -1.0 ||
            (std::isfinite(params.advisory_min_confidence) &&
             params.advisory_min_confidence > 0.0 &&
             params.advisory_min_confidence <= 1.0),
            "disabled advisory_min_confidence must be -1 or a valid ignored value");
  }

  return errors;
}

}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_VALIDATION_HPP
