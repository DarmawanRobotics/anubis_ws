#include "map_z_drift_correction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{

std::vector<anubis_mapping::MapZDriftObservation> smoothDrift()
{
    std::vector<anubis_mapping::MapZDriftObservation> observations;
    for (int index = 0; index <= 100; ++index)
    {
        const double ratio = static_cast<double>(index) / 100.0;
        observations.push_back({
            0.25 * index, -0.30 + 0.20 * ratio * ratio,
            0.2, 0.02, true, {}});
    }
    return observations;
}

}  // namespace

TEST(MapZDriftCorrectionTest, FlattensSmoothSyntheticDrift)
{
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        smoothDrift(), anubis_mapping::MapZDriftCorrectionConfig{});
    ASSERT_TRUE(result.success) << result.failure_reason;
    EXPECT_LT(result.corrected_floor_max - result.corrected_floor_min, 0.04);
    EXPECT_GT(result.max_abs_correction_m, 0.09);
    EXPECT_LE(result.max_abs_correction_m, 0.45);
}

TEST(MapZDriftCorrectionTest, InterpolatesBoundedMissingGroundIntervals)
{
    auto observations = smoothDrift();
    for (std::size_t index = 12U; index < observations.size(); index += 17U)
    {
        observations[index].valid = false;
        observations[index + 1U].valid = false;
    }
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, anubis_mapping::MapZDriftCorrectionConfig{});
    ASSERT_TRUE(result.success) << result.failure_reason;
    EXPECT_LT(result.valid_fraction, 1.0);
    EXPECT_LE(result.max_unobserved_gap_m, 2.0);
}

TEST(MapZDriftCorrectionTest, RejectsLongMissingGroundInterval)
{
    auto observations = smoothDrift();
    for (std::size_t index = 30U; index <= 45U; ++index)
    {
        observations[index].valid = false;
    }
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, anubis_mapping::MapZDriftCorrectionConfig{});
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.failure_reason.find("gap"), std::string::npos);
}

TEST(MapZDriftCorrectionTest, RejectsIsolatedOutlierWithoutMovingNeighbors)
{
    auto observations = smoothDrift();
    observations[50].floor_z += 0.35;
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, anubis_mapping::MapZDriftCorrectionConfig{});
    ASSERT_TRUE(result.success) << result.failure_reason;
    EXPECT_EQ(result.rejected_outlier_count, 1U);
    EXPECT_FALSE(result.used_observations[50]);
}

TEST(MapZDriftCorrectionTest, TreatsNonFiniteGroundAsMissing)
{
    auto observations = smoothDrift();
    observations[40].floor_z = std::numeric_limits<double>::quiet_NaN();
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, anubis_mapping::MapZDriftCorrectionConfig{});
    ASSERT_TRUE(result.success) << result.failure_reason;
    EXPECT_FALSE(result.used_observations[40]);
}

TEST(MapZDriftCorrectionTest, RejectsCorrectionBeyondAbsoluteLimit)
{
    auto observations = smoothDrift();
    for (std::size_t index = 0U; index < observations.size(); ++index)
    {
        observations[index].floor_z = -0.30 +
            1.20 * static_cast<double>(index) /
                static_cast<double>(observations.size() - 1U);
    }
    auto config = anubis_mapping::MapZDriftCorrectionConfig{};
    config.outlier_threshold_m = 1.0;
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, config);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.failure_reason.find("absolute"), std::string::npos);
}

TEST(MapZDriftCorrectionTest, StationaryFramesDoNotInflatePathCoverage)
{
    std::vector<anubis_mapping::MapZDriftObservation> observations;
    for (int index = 0; index < 100; ++index)
    {
        observations.push_back({0.0, -0.30, 0.1, 0.01, true, {}});
    }
    for (int index = 1; index <= 10; ++index)
    {
        observations.push_back({
            static_cast<double>(index), 0.0, 0.0, 0.0, false,
            "synthetic missing ground"});
    }

    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, anubis_mapping::MapZDriftCorrectionConfig{});
    EXPECT_FALSE(result.success);
    EXPECT_GT(result.valid_fraction, 0.80);
    EXPECT_LT(result.path_coverage_fraction, 0.10);
    EXPECT_NE(result.failure_reason.find("path coverage"), std::string::npos);
}

TEST(MapZDriftCorrectionTest, RecoversDriftWithIndependentSourceSupport)
{
    auto observations = smoothDrift();
    for (auto& observation : observations)
    {
        observation.independent_source_count = 3U;
        observation.sample_cells = 24U;
        observation.has_explicit_support = true;
        observation.support_s_min = std::max(0.0, observation.path_s - 0.15);
        observation.support_s_max = std::min(25.0, observation.path_s + 0.15);
    }
    auto config = anubis_mapping::MapZDriftCorrectionConfig{};
    config.min_independent_sources = 2U;
    config.min_sample_cells = 15U;
    config.max_local_tilt_deg = 3.0;
    config.max_local_residual_p95 = 0.08;
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, config);
    ASSERT_TRUE(result.success) << result.failure_reason;
    EXPECT_GT(result.path_coverage_fraction, 0.99);
    EXPECT_TRUE(result.invalid_index_ranges.empty());
}

TEST(MapZDriftCorrectionTest, RepeatedSourceWindowDoesNotInflateCoverage)
{
    auto observations = smoothDrift();
    for (auto& observation : observations)
    {
        observation.has_explicit_support = true;
        observation.support_s_min = 0.0;
        observation.support_s_max = 2.0;
    }
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, anubis_mapping::MapZDriftCorrectionConfig{});
    EXPECT_FALSE(result.success);
    EXPECT_LT(result.path_coverage_fraction, 0.10);
    EXPECT_GT(result.max_unobserved_gap_m, 20.0);
}

TEST(MapZDriftCorrectionTest, DisjointSourceCellsDoNotFillPathJump)
{
    auto observations = smoothDrift();
    for (auto& observation : observations)
    {
        observation.valid = false;
    }
    for (const std::size_t index : {0U, 1U, 2U, 98U, 99U, 100U})
    {
        auto& observation = observations[index];
        observation.valid = true;
        observation.has_explicit_support = true;
        observation.support_s_min = 0.0;
        observation.support_s_max = 25.0;
        if (index <= 2U)
        {
            observation.support_intervals.push_back({0.0, 0.25});
        }
        else
        {
            observation.support_intervals.push_back({24.75, 25.0});
        }
    }
    auto config = anubis_mapping::MapZDriftCorrectionConfig{};
    config.min_valid_fraction = 0.05;

    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, config);

    EXPECT_FALSE(result.success);
    EXPECT_LT(result.path_coverage_fraction, 0.10);
    EXPECT_GT(result.max_unobserved_gap_m, 20.0);
    EXPECT_NE(result.failure_reason.find("path coverage"), std::string::npos);
}

TEST(MapZDriftCorrectionTest, RejectsSingleSourceSupport)
{
    auto observations = smoothDrift();
    for (auto& observation : observations)
    {
        observation.independent_source_count = 1U;
        observation.sample_cells = 24U;
    }
    auto config = anubis_mapping::MapZDriftCorrectionConfig{};
    config.min_independent_sources = 2U;
    config.min_sample_cells = 15U;
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, config);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.invalid_index_ranges, "0-100");
}

TEST(MapZDriftCorrectionTest, RejectsPhysicalSlopeByLocalTiltGate)
{
    auto observations = smoothDrift();
    for (auto& observation : observations)
    {
        observation.local_tilt_deg = 5.0;
    }
    auto config = anubis_mapping::MapZDriftCorrectionConfig{};
    config.max_local_tilt_deg = 3.0;
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, config);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.invalid_index_ranges, "0-100");
}

TEST(MapZDriftCorrectionTest, RejectsWallContaminatedResiduals)
{
    auto observations = smoothDrift();
    for (auto& observation : observations)
    {
        observation.residual_p95 = 0.20;
    }
    auto config = anubis_mapping::MapZDriftCorrectionConfig{};
    config.max_local_residual_p95 = 0.08;
    const auto result = anubis_mapping::solveMapZDriftCorrection(
        observations, config);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.invalid_index_ranges, "0-100");
}
