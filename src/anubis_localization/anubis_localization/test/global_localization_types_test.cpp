#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "localization/global_localization_types.hpp"
#include "localization/global_localization_validation.hpp"

namespace localization {
namespace {

TEST(GlobalLocalizationTypes, DefaultParametersKeepAutomaticApplicationDisabled) {
  const GLParams params;

  EXPECT_FALSE(params.auto_confirm_enabled);
  EXPECT_FALSE(params.flat_single_level_map);
  EXPECT_FALSE(params.allow_degraded_fallback);
  EXPECT_FALSE(params.advisory_suggestion_enabled);
  EXPECT_DOUBLE_EQ(params.auto_confirm_verify_timeout_s, 5.0);
  EXPECT_EQ(params.gl_confirm_frames, 3);
  EXPECT_DOUBLE_EQ(params.gl_confirm_xy_tol, 0.30);
  EXPECT_DOUBLE_EQ(params.gl_probe_period_s, 0.5);
  EXPECT_DOUBLE_EQ(params.level0_base_ground_offset, -1.0);
  EXPECT_TRUE(params.level0_raw_imu_gravity_filter_enabled);
  EXPECT_DOUBLE_EQ(params.level0_raw_imu_gravity_window_s, 0.30);
  EXPECT_EQ(params.level0_raw_imu_gravity_min_samples, 30);
  EXPECT_TRUE(params.m2b_approval_id.empty());
}

TEST(GlobalLocalizationTypes, DefaultResultDoesNotExposeAUsablePose) {
  const GlobalLocalizationResult result;

  EXPECT_EQ(result.status, GLStatus::NoCandidate);
  EXPECT_FALSE(result.success);
  EXPECT_FALSE(result.ambiguous);
  EXPECT_EQ(result.anchor_reject_reason, GLAnchorRejectReason::None);
  EXPECT_DOUBLE_EQ(result.best_score, -1.0);
  EXPECT_TRUE(result.final_pose.isIdentity());
  EXPECT_TRUE(result.candidates.empty());
}

TEST(GlobalLocalizationTypes, CandidateDefaultsAreInvalidUntilMeasured) {
  const GlobalPoseCandidate candidate;

  EXPECT_FALSE(candidate.converged);
  EXPECT_DOUBLE_EQ(candidate.fitness_score, 1e9);
  EXPECT_DOUBLE_EQ(candidate.coarse_score, 1e9);
  EXPECT_EQ(candidate.grid_cell_id, -1);
  EXPECT_EQ(candidate.yaw_bin, -1);
  EXPECT_TRUE(candidate.seed_pose.isIdentity());
  EXPECT_TRUE(candidate.final_pose.isIdentity());
}

TEST(GlobalLocalizationTypes, GravitySampleDefaultsToUnavailable) {
  const GravityAlignmentSample sample;

  EXPECT_FALSE(sample.valid);
  EXPECT_DOUBLE_EQ(sample.uncertainty_deg, 1e9);
  EXPECT_DOUBLE_EQ(sample.angular_velocity_rps, 1e9);
  EXPECT_TRUE(sample.R_level_base.isIdentity());
}

TEST(GlobalLocalizationTypes, SummaryCountersStartFromNeutralSentinels) {
  const GLSummaryCounts counts;

  EXPECT_EQ(counts.seed_total, 0);
  EXPECT_EQ(counts.level0_lookup_count, 0U);
  EXPECT_EQ(counts.anchor_recapture_count, 0);
  EXPECT_DOUBLE_EQ(counts.anchor_gravity_age_ms, -1.0);
  EXPECT_DOUBLE_EQ(counts.processing_ms, 0.0);
  EXPECT_FALSE(counts.level0_ground_unavailable);
}

TEST(GlobalLocalizationValidation, DefaultShadowParametersAreValid) {
  EXPECT_TRUE(validateGLParams(GLParams{}).empty());
}

TEST(GlobalLocalizationValidation, CandidateCsvRequiresExplicitDatasetIdentity) {
  GLParams params;
  params.dump_candidates_csv = true;

  EXPECT_FALSE(validateGLParams(params).empty());

  params.candidates_csv_path = "/tmp/gl_predictions.csv";
  params.candidates_csv_episode_id = "fixture_room_a_001";
  EXPECT_TRUE(validateGLParams(params).empty());
}

TEST(GlobalLocalizationValidation, RejectsSupportAndLevel1BoundaryCollisions) {
  GLParams params;
  params.support_seed_xy = 2.0 * params.cluster_xy;
  params.level1_seed_stride_xy = params.support_seed_xy;

  const std::vector<std::string> errors = validateGLParams(params);

  EXPECT_GE(errors.size(), 2U);
}

TEST(GlobalLocalizationValidation, Level1SearchMustFitIndependentTemplates) {
  GLParams params;
  params.level1_search_radius_xy = params.level1_seed_stride_xy - 0.01;

  EXPECT_FALSE(validateGLParams(params).empty());
}

TEST(GlobalLocalizationValidation, RejectsInvalidPointAndRangeLimits) {
  GLParams params;
  params.level0_max_point_height = params.level0_min_point_height;
  params.level0_max_range = params.level0_min_range;
  params.level0_min_valid_points = params.level0_max_scan_points + 1;

  const std::vector<std::string> errors = validateGLParams(params);

  EXPECT_GE(errors.size(), 3U);
}

TEST(GlobalLocalizationValidation, RejectsInvalidRawImuGravityWindow) {
  GLParams params;
  params.level0_raw_imu_gravity_window_s = 0.0;
  EXPECT_FALSE(validateGLParams(params).empty());

  params = GLParams{};
  params.level0_raw_imu_gravity_min_samples = 1;
  EXPECT_FALSE(validateGLParams(params).empty());
}

TEST(GlobalLocalizationValidation, RawImuFilterRequiresLowDynamicGyroLimit) {
  GLParams params;
  params.level0_raw_imu_gravity_max_angular_velocity_rps = 0.0;

  EXPECT_FALSE(validateGLParams(params).empty());

  params.level0_raw_imu_gravity_filter_enabled = false;
  EXPECT_TRUE(validateGLParams(params).empty());
}

TEST(GlobalLocalizationValidation, AutomaticConfirmationRequiresApprovalContract) {
  GLParams params;
  params.auto_confirm_enabled = true;

  EXPECT_TRUE(validateGLParams(params).empty());
  EXPECT_FALSE(validateAutoConfirmContract(params).empty());

  params.flat_single_level_map = true;
  params.level0_base_ground_offset = 0.45;
  // Syntax-only fixture for this unit test; it is not a formal M2b approval.
  params.m2b_approval_id =
      "m2b:unit-fixture:map-a1b2c3d4:params-e5f6a7b8:manifest-01020304";
  EXPECT_TRUE(validateAutoConfirmContract(params).empty());
  EXPECT_TRUE(validateGLParams(params).empty());

  params.allow_degraded_fallback = true;
  EXPECT_FALSE(validateAutoConfirmContract(params).empty());
}

TEST(GlobalLocalizationValidation, RejectsPlaceholderApprovalIdentifiers) {
  constexpr const char* kPlaceholders[] = {
      "",
      "DEBUG NOT APPROVED",
      "m2b:todo:map:params:manifest",
      "m2b:tbd:map:params:manifest",
      "m2b:pending:map:params:manifest",
      "m2b:changeme:map:params:manifest",
      "m2b:placeholder:map:params:manifest",
      "m2b:example:map:params:manifest",
      "m2b:sample:map:params:manifest",
      "m2b:test:map:params:manifest",
      "m2b:dummy:map:params:manifest",
      "m2b:none:map:params:manifest",
      "m2b:null:map:params:manifest",
      "m2b:unknown:map:params:manifest",
      "m2b:unapproved:map:params:manifest",
      "m2b:debug_not_approved:map:params:manifest",
      "m2b:0:0:0:0",
      "m2b:0000:map:params:manifest",
      "m2b:0.0:map:params:manifest",
      "m2b:test-map:test-params:test-data",
      "m2b:abc:map:${params}:manifest",
      "m2b:abc:map:{{params}}:manifest",
      "m2b:abc:map:params:<manifest>",
  };
  for (const char* placeholder : kPlaceholders) {
    EXPECT_FALSE(isUsableM2bApprovalId(placeholder)) << placeholder;
  }
}

TEST(GlobalLocalizationValidation, AcceptsOnlySyntaxValidNonPlaceholderApprovalFixture) {
  // This fixture checks syntax only and does not represent formal M2b evidence.
  EXPECT_TRUE(isUsableM2bApprovalId(
      "  m2b:unit-fixture:map-a1b2c3d4:params-e5f6a7b8:manifest-01020304  "));
  EXPECT_FALSE(isUsableM2bApprovalId("m2b:a:b:c"));
  EXPECT_FALSE(isUsableM2bApprovalId("m2b:a:b:c:d with-space"));
  EXPECT_FALSE(isUsableM2bApprovalId("m2b:a:b:c:d<template>"));
}

TEST(GlobalLocalizationValidation, RuntimeResolutionFailsClosedOnExplicitConflict) {
  const GLEpisodeRuntimeResolution missing =
      resolveGLEpisodeRuntimeEnabled(std::nullopt, std::nullopt);
  EXPECT_FALSE(missing.enabled);
  EXPECT_FALSE(missing.legacy_parameter_used);
  EXPECT_FALSE(missing.conflict);

  EXPECT_TRUE(resolveGLEpisodeRuntimeEnabled(true, std::nullopt).enabled);
  EXPECT_FALSE(resolveGLEpisodeRuntimeEnabled(false, std::nullopt).enabled);

  const GLEpisodeRuntimeResolution legacy_true =
      resolveGLEpisodeRuntimeEnabled(std::nullopt, true);
  EXPECT_TRUE(legacy_true.enabled);
  EXPECT_TRUE(legacy_true.legacy_parameter_used);
  EXPECT_FALSE(legacy_true.conflict);
  EXPECT_FALSE(resolveGLEpisodeRuntimeEnabled(std::nullopt, false).enabled);

  EXPECT_TRUE(resolveGLEpisodeRuntimeEnabled(true, true).enabled);
  EXPECT_FALSE(resolveGLEpisodeRuntimeEnabled(false, false).enabled);

  const GLEpisodeRuntimeResolution true_false =
      resolveGLEpisodeRuntimeEnabled(true, false);
  EXPECT_FALSE(true_false.enabled);
  EXPECT_TRUE(true_false.legacy_parameter_used);
  EXPECT_TRUE(true_false.conflict);

  // The security-critical direction must remain disabled as well.
  const GLEpisodeRuntimeResolution false_true =
      resolveGLEpisodeRuntimeEnabled(false, true);
  EXPECT_FALSE(false_true.enabled);
  EXPECT_TRUE(false_true.legacy_parameter_used);
  EXPECT_TRUE(false_true.conflict);
}

TEST(GlobalLocalizationValidation, RejectsInvalidAutomaticVerificationTimeout) {
  GLParams params;
  params.auto_confirm_verify_timeout_s = 0.0;
  EXPECT_FALSE(validateGLParams(params).empty());
}

TEST(GlobalLocalizationValidation, AdvisoryModeRequiresCalibrationAndConfidence) {
  GLParams params;
  params.advisory_suggestion_enabled = true;

  EXPECT_FALSE(validateGLParams(params).empty());

  params.advisory_min_confidence = 0.8;
  params.advisory_calibration_id = "calibration:test";
  EXPECT_TRUE(validateGLParams(params).empty());
}

TEST(GlobalLocalizationValidation, StaticConfirmFramesMustBePositive) {
  GLParams params;
  params.gl_confirm_frames = 0;

  EXPECT_FALSE(validateGLParams(params).empty());
}

}  // namespace
}  // namespace localization
