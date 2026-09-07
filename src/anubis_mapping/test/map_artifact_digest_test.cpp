#include "map_leveling_sidecar.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

class ArtifactDirectoryTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        root_ = std::filesystem::temp_directory_path() /
            ("roamerx_map_artifacts_" + std::to_string(nonce));
        ASSERT_TRUE(std::filesystem::create_directories(root_));
        writeFile("map.pcd", "pcd-content\n");
        writeFile("map.pgm", "pgm-content\n");
        writeFile("map.yaml", "yaml-content\n");
        writeFile("map.txt", "0 0 0\n");
    }

    void TearDown() override
    {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    void writeFile(const std::filesystem::path& relative, const std::string& value)
    {
        const auto path = root_ / relative;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(output.is_open());
        output << value;
        output.flush();
        ASSERT_TRUE(output.good());
    }

    std::string sidecarFor(
        const std::vector<anubis_mapping::MapArtifactDigest>& artifacts,
        const std::string& generation = "g-test",
        bool leveling_applied = true,
        bool enabled = true) const
    {
        std::ostringstream output;
        output
            << "schema: map_leveling\n"
            << "schema_version: 3\n"
            << "generation: " << generation << "\n"
            << "artifact_count: " << artifacts.size() << "\n"
            << "enabled: " << (enabled ? "true" : "false") << "\n"
            << "succeeded: true\n"
            << "leveling_applied: "
            << (leveling_applied ? "true" : "false") << "\n"
            << "correction_rpy_rad: [0, 0, 0]\n"
            << "correction_rpy_deg: [0, 0, 0]\n"
            << "correction_rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
            << "floor_z_map: -0.304\n"
            << "configured_sample_quantile: 0.05\n"
            << "selected_sample_quantile: 0.05\n"
            << "fallback_sample_quantile: 0.10\n"
            << "ground_normal: [0, 0, 1]\n"
            << "inlier_ratio: " << (leveling_applied ? "0.75" : "0") << "\n"
            << "final_tilt_deg: 0\n"
            << "iterations: " << (leveling_applied ? "2" : "0") << "\n"
            << "q10_attempted: false\n"
            << "q10_accepted: false\n"
            << "seed_candidate_points: " << (leveling_applied ? "300" : "0") << "\n"
            << "seed_sample_cells: " << (leveling_applied ? "200" : "0") << "\n"
            << "seed_inlier_cells: " << (leveling_applied ? "150" : "0") << "\n"
            << "seed_inlier_ratio: " << (leveling_applied ? "0.75" : "0") << "\n"
            << "seed_residual_p95: " << (leveling_applied ? "0.03" : "0") << "\n"
            << "seed_tilt_deg: " << (leveling_applied ? "1.2" : "0") << "\n"
            << "refit_candidate_points: " << (leveling_applied ? "300" : "0") << "\n"
            << "refit_sample_cells: " << (leveling_applied ? "200" : "0") << "\n"
            << "refit_inlier_cells: " << (leveling_applied ? "150" : "0") << "\n"
            << "refit_inlier_ratio: " << (leveling_applied ? "0.75" : "0") << "\n"
            << "refit_residual_p95: " << (leveling_applied ? "0.02" : "0") << "\n"
            << "refit_major_span: " << (leveling_applied ? "12" : "0") << "\n"
            << "refit_minor_span: " << (leveling_applied ? "4" : "0") << "\n"
            << "failure_reason: "
            << (leveling_applied
                    ? ""
                    : (enabled ? "gate failed; saved unleveled"
                               : "disabled by configuration"))
            << "\n";
        for (std::size_t index = 0U; index < artifacts.size(); ++index)
        {
            const auto& artifact = artifacts[index];
            output << "artifact_" << index << "_path: "
                   << artifact.relative_path.generic_string() << "\n"
                   << "artifact_" << index << "_size: " << artifact.size << "\n"
                   << "artifact_" << index << "_sha256: " << artifact.sha256 << "\n";
        }
        return output.str();
    }

    bool parseSidecar(
        const std::string& sidecar, bool bind_to_root,
        double* floor = nullptr,
        anubis_mapping::MapLevelingMetadata* metadata = nullptr) const
    {
        std::istringstream input(sidecar);
        double parsed_floor = 0.0;
        const bool result = bind_to_root
            ? anubis_mapping::parseValidatedMapLevelingFloor(
                  input, parsed_floor, metadata, &root_)
            : anubis_mapping::parseValidatedMapLevelingFloor(
                  input, parsed_floor, metadata);
        if (floor != nullptr)
        {
            *floor = parsed_floor;
        }
        return result;
    }

    std::filesystem::path root_;
};

TEST_F(ArtifactDirectoryTest, V3ManifestBindsExactFilesAndIgnoresBackupDirectory)
{
    writeFile(".map_save_backup_old/map.pcd", "old-pcd\n");
    writeFile(".map_save_tmp_old/partial.bin", "partial\n");

    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    std::string error;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts, &error)) << error;
    ASSERT_EQ(artifacts.size(), 4U);

    anubis_mapping::MapLevelingMetadata metadata;
    double floor = 0.0;
    EXPECT_TRUE(parseSidecar(sidecarFor(artifacts), true, &floor, &metadata));
    EXPECT_DOUBLE_EQ(floor, -0.304);
    EXPECT_EQ(metadata.schema_version, 3U);
    EXPECT_EQ(metadata.generation, "g-test");
    EXPECT_EQ(metadata.artifacts.size(), artifacts.size());
}

TEST_F(ArtifactDirectoryTest, AcceptsV3SidecarSavedWithoutLeveling)
{
    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    std::string error;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts, &error)) << error;

    anubis_mapping::MapLevelingMetadata metadata;
    double floor = 0.0;
    EXPECT_TRUE(parseSidecar(
        sidecarFor(artifacts, "g-test", false), true, &floor, &metadata));
    EXPECT_DOUBLE_EQ(floor, -0.304);
    EXPECT_EQ(metadata.generation, "g-test");
    EXPECT_EQ(metadata.artifacts.size(), artifacts.size());
}

TEST_F(ArtifactDirectoryTest, AcceptsV3DisabledSidecarOnlyWhenUnapplied)
{
    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts));

    EXPECT_TRUE(parseSidecar(
        sidecarFor(artifacts, "g-disabled", false, false), true));
    EXPECT_FALSE(parseSidecar(
        sidecarFor(artifacts, "g-disabled-invalid", true, false), true));
}

TEST_F(ArtifactDirectoryTest, RejectsDisabledSidecarWithWrongFailureReason)
{
    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts));

    std::string sidecar = sidecarFor(
        artifacts, "g-disabled-reason", false, false);
    const std::string expected =
        "failure_reason: disabled by configuration";
    const std::size_t position = sidecar.find(expected);
    ASSERT_NE(position, std::string::npos);
    sidecar.replace(
        position, expected.size(), "failure_reason: disabled unexpectedly");
    EXPECT_FALSE(parseSidecar(sidecar, true));
}

TEST_F(ArtifactDirectoryTest, RejectsV3SidecarMissingLevelingApplied)
{
    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts));
    const std::string field = "leveling_applied: false\n";
    std::string sidecar = sidecarFor(artifacts, "g-missing", false);
    const std::size_t position = sidecar.find(field);
    ASSERT_NE(position, std::string::npos);
    sidecar.erase(position, field.size());
    EXPECT_FALSE(parseSidecar(sidecar, true));
}

TEST_F(ArtifactDirectoryTest, RejectsUnappliedSidecarWithAppliedRotation)
{
    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts));
    std::string sidecar = sidecarFor(artifacts, "g-test", false);
    const std::string identity =
        "correction_rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]";
    const std::size_t position = sidecar.find(identity);
    ASSERT_NE(position, std::string::npos);
    sidecar.replace(
        position, identity.size(),
        "correction_rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1]");
    const std::string rpy_rad = "correction_rpy_rad: [0, 0, 0]";
    const std::size_t rpy_position = sidecar.find(rpy_rad);
    ASSERT_NE(rpy_position, std::string::npos);
    sidecar.replace(
        rpy_position, rpy_rad.size(),
        "correction_rpy_rad: [0, 0, 1.5707963267948966]");
    const std::string rpy_deg = "correction_rpy_deg: [0, 0, 0]";
    const std::size_t deg_position = sidecar.find(rpy_deg);
    ASSERT_NE(deg_position, std::string::npos);
    sidecar.replace(deg_position, rpy_deg.size(), "correction_rpy_deg: [0, 0, 90]");
    EXPECT_FALSE(parseSidecar(sidecar, true));
}

TEST_F(ArtifactDirectoryTest, RejectsUnappliedSidecarWithEmptyFailureReason)
{
    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts));
    std::string sidecar = sidecarFor(artifacts, "g-test", false);
    const std::string reason = "failure_reason: gate failed; saved unleveled";
    const std::size_t position = sidecar.find(reason);
    ASSERT_NE(position, std::string::npos);
    sidecar.replace(position, reason.size(), "failure_reason: ");
    EXPECT_FALSE(parseSidecar(sidecar, true));
}

TEST_F(ArtifactDirectoryTest, RejectsTamperedArtifact)
{
    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts));
    const std::string sidecar = sidecarFor(artifacts);
    writeFile("map.pcd", "tampered\n");
    EXPECT_FALSE(parseSidecar(sidecar, true));
}

TEST_F(ArtifactDirectoryTest, RejectsExtraRegularArtifact)
{
    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts));
    const std::string sidecar = sidecarFor(artifacts);
    writeFile("unexpected.bin", "extra\n");
    EXPECT_FALSE(parseSidecar(sidecar, true));
}

TEST_F(ArtifactDirectoryTest, RejectsMissingArtifact)
{
    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts));
    const std::string sidecar = sidecarFor(artifacts);
    std::error_code error;
    std::filesystem::remove(root_ / "map.txt", error);
    ASSERT_FALSE(error);
    EXPECT_FALSE(parseSidecar(sidecar, true));
}

TEST_F(ArtifactDirectoryTest, LegacyV2IsParseCompatibleButNotProductionBound)
{
    const std::string legacy =
        "schema: map_leveling\n"
        "schema_version: 2\n"
        "enabled: true\n"
        "succeeded: true\n"
        "correction_rpy_rad: [0, 0, 0]\n"
        "correction_rpy_deg: [0, 0, 0]\n"
        "correction_rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
        "floor_z_map: -0.304\n"
        "configured_sample_quantile: 0.05\n"
        "selected_sample_quantile: 0.05\n"
        "fallback_sample_quantile: 0.10\n"
        "ground_normal: [0, 0, 1]\n"
        "inlier_ratio: 0.75\n"
        "final_tilt_deg: 0\n"
        "iterations: 2\n"
        "q10_attempted: false\n"
        "q10_accepted: false\n"
        "seed_candidate_points: 300\n"
        "seed_sample_cells: 200\n"
        "seed_inlier_cells: 150\n"
        "seed_inlier_ratio: 0.75\n"
        "seed_residual_p95: 0.03\n"
        "seed_tilt_deg: 1.2\n"
        "refit_candidate_points: 300\n"
        "refit_sample_cells: 200\n"
        "refit_inlier_cells: 150\n"
        "refit_inlier_ratio: 0.75\n"
        "refit_residual_p95: 0.02\n"
        "refit_major_span: 12\n"
        "refit_minor_span: 4\n"
        "failure_reason: \n";
    EXPECT_TRUE(parseSidecar(legacy, false));
    EXPECT_FALSE(parseSidecar(legacy, true));
}

TEST_F(ArtifactDirectoryTest, OrdinaryPrefixFileIsNotTransient)
{
    writeFile(".map_save_backup_file", "must-be-visible\n");
    std::vector<anubis_mapping::MapArtifactDigest> artifacts;
    ASSERT_TRUE(anubis_mapping::collectMapArtifactDigests(
        root_, "map_leveling.yaml", artifacts));
    EXPECT_TRUE(std::any_of(
        artifacts.begin(), artifacts.end(), [](const auto& artifact) {
            return artifact.relative_path == ".map_save_backup_file";
        }));
}

}  // namespace
