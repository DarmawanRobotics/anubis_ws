#include "map_leveling_sidecar.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace
{

std::string validSidecar(bool enabled = true, bool fallback = false)
{
    std::ostringstream output;
    output
        << "schema: map_leveling\n"
        << "schema_version: 2\n"
        << "enabled: " << (enabled ? "true" : "false") << "\n"
        << "succeeded: true\n"
        << "correction_rpy_rad: [0, 0, 0]\n"
        << "correction_rpy_deg: [0, 0, 0]\n"
        << "correction_rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
        << "floor_z_map: -0.304\n"
        << "configured_sample_quantile: 0.05\n"
        << "selected_sample_quantile: " << (fallback ? "0.10" : "0.05") << "\n"
        << "fallback_sample_quantile: 0.10\n"
        << "ground_normal: [0, 0, 1]\n"
        << "inlier_ratio: " << (enabled ? "0.75" : "0") << "\n"
        << "final_tilt_deg: 0\n"
        << "iterations: " << (enabled ? "2" : "0") << "\n"
        << "q10_attempted: " << (fallback ? "true" : "false") << "\n"
        << "q10_accepted: " << (fallback ? "true" : "false") << "\n"
        << "seed_candidate_points: " << (enabled ? "300" : "0") << "\n"
        << "seed_sample_cells: " << (enabled ? "200" : "0") << "\n"
        << "seed_inlier_cells: " << (enabled ? "150" : "0") << "\n"
        << "seed_inlier_ratio: " << (enabled ? "0.75" : "0") << "\n"
        << "seed_residual_p95: " << (enabled ? "0.03" : "0") << "\n"
        << "seed_tilt_deg: " << (enabled ? "1.2" : "0") << "\n"
        << "refit_candidate_points: " << (enabled ? "300" : "0") << "\n"
        << "refit_sample_cells: " << (enabled ? "200" : "0") << "\n"
        << "refit_inlier_cells: " << (enabled ? "150" : "0") << "\n"
        << "refit_inlier_ratio: " << (enabled ? "0.75" : "0") << "\n"
        << "refit_residual_p95: " << (enabled ? "0.02" : "0") << "\n"
        << "refit_major_span: " << (enabled ? "12.0" : "0") << "\n"
        << "refit_minor_span: " << (enabled ? "4.0" : "0") << "\n"
        << "failure_reason: "
        << (enabled ? "" : "disabled by configuration") << "\n";
    return output.str();
}

bool parses(const std::string& sidecar, double* floor = nullptr)
{
    std::istringstream input(sidecar);
    double parsed_floor = 123.0;
    const bool accepted =
        anubis_mapping::parseValidatedMapLevelingFloor(input, parsed_floor);
    if (floor != nullptr)
    {
        *floor = parsed_floor;
    }
    return accepted;
}

std::string replaceOnce(
    std::string value, const std::string& from, const std::string& to)
{
    const std::size_t position = value.find(from);
    EXPECT_NE(position, std::string::npos);
    if (position != std::string::npos)
    {
        value.replace(position, from.size(), to);
    }
    return value;
}

TEST(MapLevelingSidecarTest, AcceptsPrimarySuccessWithNegativeFloor)
{
    double floor = 0.0;
    EXPECT_TRUE(parses(validSidecar(), &floor));
    EXPECT_DOUBLE_EQ(floor, -0.304);
}

TEST(MapLevelingSidecarTest, AcceptsFallbackSuccessWhenSelectedEqualsFallback)
{
    EXPECT_TRUE(parses(validSidecar(true, true)));
}

TEST(MapLevelingSidecarTest, AcceptsTinyResidualTiltAfterTextRounding)
{
    std::string sidecar = replaceOnce(
        validSidecar(), "ground_normal: [0, 0, 1]",
        "ground_normal: [0.0000001, 0, 1]");
    sidecar = replaceOnce(
        sidecar, "final_tilt_deg: 0", "final_tilt_deg: 0.00000572957795");
    EXPECT_TRUE(parses(sidecar));
}

TEST(MapLevelingSidecarTest, AcceptsRoundTripAdjacentQuantiles)
{
    const double configured = std::nextafter(0.2, 0.0);
    std::ostringstream configured_text;
    configured_text << std::setprecision(
        std::numeric_limits<double>::max_digits10) << configured;
    std::string sidecar = replaceOnce(
        validSidecar(), "configured_sample_quantile: 0.05",
        "configured_sample_quantile: " + configured_text.str());
    sidecar = replaceOnce(
        sidecar, "selected_sample_quantile: 0.05",
        "selected_sample_quantile: " + configured_text.str());
    sidecar = replaceOnce(
        sidecar, "fallback_sample_quantile: 0.10",
        "fallback_sample_quantile: 0.2");
    EXPECT_TRUE(parses(sidecar));
}

TEST(MapLevelingSidecarTest, AcceptsNonzeroWriterRotation)
{
    std::string sidecar = replaceOnce(
        validSidecar(), "correction_rpy_rad: [0, 0, 0]",
        "correction_rpy_rad: [-0.00078198328265, -0.0126382712381, 4.94152444099e-06]");
    sidecar = replaceOnce(
        sidecar, "correction_rpy_deg: [0, 0, 0]",
        "correction_rpy_deg: [-0.0448043417458, -0.724119602287, 0.00028312849483]");
    sidecar = replaceOnce(
        sidecar,
        "correction_rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]",
        "correction_rotation: [0.999920138101, 4.9411298015e-06, -0.0126379347976, 4.9411298015e-06, 0.999999694288, 0.000781920752299, 0.0126379347976, -0.000781920752299, 0.999919832389]");
    sidecar = replaceOnce(
        sidecar, "ground_normal: [0, 0, 1]",
        "ground_normal: [-1.38142252878e-05, 8.55490479695e-05, 0.999999996245]");
    sidecar = replaceOnce(
        sidecar, "final_tilt_deg: 0",
        "final_tilt_deg: 0.00496509243539");
    EXPECT_TRUE(parses(sidecar));
}

TEST(MapLevelingSidecarTest, AcceptsStrictDisabledSuccessState)
{
    EXPECT_TRUE(parses(validSidecar(false)));
}

TEST(MapLevelingSidecarTest, RejectsTransactionInvalidator)
{
    EXPECT_FALSE(parses(replaceOnce(
        validSidecar(), "succeeded: true", "succeeded: false")));
}

TEST(MapLevelingSidecarTest, RejectsMissingDegreesField)
{
    EXPECT_FALSE(parses(replaceOnce(
        validSidecar(), "correction_rpy_deg: [0, 0, 0]\n", "")));
}

TEST(MapLevelingSidecarTest, RejectsInconsistentQuantileState)
{
    EXPECT_FALSE(parses(replaceOnce(
        validSidecar(), "q10_attempted: false", "q10_attempted: true")));
}

TEST(MapLevelingSidecarTest, RejectsRatioOutsideUnitInterval)
{
    EXPECT_FALSE(parses(replaceOnce(
        validSidecar(), "inlier_ratio: 0.75", "inlier_ratio: 1.1")));
}

TEST(MapLevelingSidecarTest, RejectsNonUnitNormal)
{
    EXPECT_FALSE(parses(replaceOnce(
        validSidecar(), "ground_normal: [0, 0, 1]",
        "ground_normal: [0, 0, 0.99]")));
}

TEST(MapLevelingSidecarTest, RejectsRpyThatDoesNotMatchRotation)
{
    EXPECT_FALSE(parses(replaceOnce(
        validSidecar(), "correction_rpy_rad: [0, 0, 0]",
        "correction_rpy_rad: [0.1, 0, 0]")));
}

TEST(MapLevelingSidecarTest, RejectsInconsistentFitCounts)
{
    EXPECT_FALSE(parses(replaceOnce(
        validSidecar(), "refit_inlier_cells: 150",
        "refit_inlier_cells: 201")));
}

TEST(MapLevelingSidecarTest, RejectsNegativeDiagnosticMetric)
{
    EXPECT_FALSE(parses(replaceOnce(
        validSidecar(), "refit_residual_p95: 0.02",
        "refit_residual_p95: -0.02")));
}

TEST(MapLevelingSidecarTest, RejectsEnabledSuccessWithFailureReason)
{
    EXPECT_FALSE(parses(replaceOnce(
        validSidecar(), "failure_reason: \n", "failure_reason: stale error\n")));
}

}  // namespace
