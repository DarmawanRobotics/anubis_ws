#include "map_artifact_transaction.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

namespace
{

using anubis_mapping::MapArtifactFileOperations;
using anubis_mapping::MapArtifactPromotionCallbacks;
using anubis_mapping::MapArtifactPromotionEntry;
using anubis_mapping::MapArtifactRollbackEntry;

class PromotionFixture
{
public:
    PromotionFixture()
    {
        const auto suffix = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        root_ = std::filesystem::temp_directory_path() /
            ("robot_slam_map_promotion_" + std::to_string(suffix));
        staging_ = root_ / ".staging";
        std::filesystem::create_directories(staging_);
    }

    ~PromotionFixture()
    {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    void write(const std::filesystem::path& path, const std::string& value) const
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(output.is_open());
        output << value;
        ASSERT_TRUE(output.good());
    }

    std::string read(const std::filesystem::path& path) const
    {
        std::ifstream input(path, std::ios::binary);
        return std::string(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    }

    std::filesystem::path root() const { return root_; }
    std::filesystem::path staging() const { return staging_; }

private:
    std::filesystem::path root_;
    std::filesystem::path staging_;
};

std::vector<MapArtifactPromotionEntry> promotionEntries(
    const PromotionFixture& fixture)
{
    return {
        {"map_leveling.yaml", fixture.staging() / "map_leveling.yaml",
            fixture.root() / "map_leveling.yaml", false},
        {"map.pcd", fixture.staging() / "map.pcd",
            fixture.root() / "map.pcd", false},
        {"map.pgm", fixture.staging() / "map.pgm",
            fixture.root() / "map.pgm", false},
        {"old_grid.yaml", {}, fixture.root() / "old_grid.yaml", true}};
}

MapArtifactPromotionCallbacks promotionCallbacks(
    const PromotionFixture& fixture, bool verification_result = true)
{
    MapArtifactPromotionCallbacks callbacks;
    callbacks.write_invalid_sidecar =
        [&fixture](const std::filesystem::path& path)
    {
        fixture.write(path, "succeeded: false\n");
        return true;
    };
    callbacks.verify_destination =
        [&fixture, verification_result](std::string* error)
    {
        if (!verification_result)
        {
            if (error != nullptr) *error = "injected verification failure";
            return false;
        }
        const bool complete =
            fixture.read(fixture.root() / "map.pcd") == "new-pcd\n" &&
            fixture.read(fixture.root() / "map.pgm") == "new-pgm\n" &&
            fixture.read(fixture.root() / "map_leveling.yaml") ==
                "new-sidecar\n" &&
            !std::filesystem::exists(fixture.root() / "old_grid.yaml");
        if (!complete && error != nullptr) *error = "injected completeness failure";
        return complete;
    };
    return callbacks;
}

std::vector<MapArtifactRollbackEntry> entries()
{
    // Sidecar is intentionally first to prove the helper imposes its own
    // validity ordering instead of relying on directory iteration order.
    return {
        {"map_leveling.yaml", "/map/map_leveling.yaml",
            "/backup/map_leveling.yaml", true},
        {"map.pcd", "/map/map.pcd", "/backup/map.pcd", true},
        {"map.pgm", "/map/map.pgm", "/backup/map.pgm", true}};
}

TEST(MapArtifactTransactionTest, RestoresSidecarAfterEveryDataArtifact)
{
    std::vector<std::string> operations;
    MapArtifactFileOperations file_ops;
    file_ops.copy_file = [&](const auto&, const auto& destination,
                             std::error_code& error)
    {
        error.clear();
        operations.push_back("copy:" + destination.generic_string());
        return true;
    };
    file_ops.remove_file = [&](const auto& path, std::error_code& error)
    {
        error.clear();
        operations.push_back("remove:" + path.generic_string());
    };
    const auto result = anubis_mapping::rollbackMapArtifactsFailClosed(
        entries(), "map_leveling.yaml",
        [&]() {
            operations.push_back("invalidate");
            return true;
        },
        file_ops);

    ASSERT_TRUE(result.restored);
    ASSERT_TRUE(result.safe_terminal_state);
    ASSERT_EQ(operations.size(), 4U);
    EXPECT_EQ(operations[0], "invalidate");
    EXPECT_EQ(operations[1], "copy:/map/map.pcd");
    EXPECT_EQ(operations[2], "copy:/map/map.pgm");
    EXPECT_EQ(operations[3], "copy:/map/map_leveling.yaml");
}

TEST(MapArtifactTransactionTest, DataRestoreFailureNeverRestoresValidSidecar)
{
    std::vector<std::string> operations;
    int invalidations = 0;
    MapArtifactFileOperations file_ops;
    file_ops.copy_file = [&](const auto&, const auto& destination,
                             std::error_code& error)
    {
        operations.push_back("copy:" + destination.generic_string());
        if (destination == "/map/map.pgm")
        {
            error = std::make_error_code(std::errc::io_error);
            return false;
        }
        error.clear();
        return true;
    };
    file_ops.remove_file = [](const auto&, std::error_code& error)
    {
        error.clear();
    };
    const auto result = anubis_mapping::rollbackMapArtifactsFailClosed(
        entries(), "map_leveling.yaml",
        [&]() {
            ++invalidations;
            operations.push_back("invalidate");
            return true;
        },
        file_ops);

    EXPECT_FALSE(result.restored);
    EXPECT_TRUE(result.safe_terminal_state);
    EXPECT_EQ(invalidations, 2);
    EXPECT_EQ(result.failed_path, std::filesystem::path("/map/map.pgm"));
    EXPECT_EQ(result.failed_operation, "restore artifact");
    for (const auto& operation : operations)
    {
        EXPECT_NE(operation, "copy:/map/map_leveling.yaml");
    }
}

TEST(MapArtifactTransactionTest, ReportsWhenFailureCannotBeFailClosed)
{
    int invalidations = 0;
    MapArtifactFileOperations file_ops;
    file_ops.copy_file = [](const auto&, const auto& destination,
                            std::error_code& error)
    {
        if (destination == "/map/map.pgm")
        {
            error = std::make_error_code(std::errc::io_error);
            return false;
        }
        error.clear();
        return true;
    };
    file_ops.remove_file = [](const auto&, std::error_code& error)
    {
        error.clear();
    };
    const auto result = anubis_mapping::rollbackMapArtifactsFailClosed(
        entries(), "map_leveling.yaml",
        [&]() {
            ++invalidations;
            return invalidations == 1;
        },
        file_ops);

    EXPECT_FALSE(result.restored);
    EXPECT_FALSE(result.safe_terminal_state);
    EXPECT_EQ(invalidations, 2);
}

TEST(MapArtifactTransactionTest, DoesNotRestoreDataWhenInitialInvalidationFails)
{
    int invalidations = 0;
    int copies = 0;
    MapArtifactFileOperations file_ops;
    file_ops.copy_file = [&](const auto&, const auto&, std::error_code& error)
    {
        ++copies;
        error.clear();
        return true;
    };
    file_ops.remove_file = [](const auto&, std::error_code& error)
    {
        error.clear();
    };
    const auto result = anubis_mapping::rollbackMapArtifactsFailClosed(
        entries(), "map_leveling.yaml",
        [&]() {
            ++invalidations;
            return false;
        },
        file_ops);

    EXPECT_FALSE(result.restored);
    EXPECT_FALSE(result.safe_terminal_state);
    EXPECT_EQ(invalidations, 1);
    EXPECT_EQ(copies, 0);
    EXPECT_EQ(result.failed_operation, "initial sidecar invalidation");
}

TEST(MapArtifactTransactionTest, PromotesRealFilesAndRemovesStaleArtifacts)
{
    PromotionFixture fixture;
    fixture.write(fixture.root() / "map.pcd", "old-pcd\n");
    fixture.write(fixture.root() / "map.pgm", "old-pgm\n");
    fixture.write(fixture.root() / "map_leveling.yaml", "old-sidecar\n");
    fixture.write(fixture.root() / "old_grid.yaml", "stale\n");
    fixture.write(fixture.staging() / "map.pcd", "new-pcd\n");
    fixture.write(fixture.staging() / "map.pgm", "new-pgm\n");
    fixture.write(fixture.staging() / "map_leveling.yaml", "new-sidecar\n");

    const auto result = anubis_mapping::promoteMapArtifactsTransactional(
        promotionEntries(fixture), "map_leveling.yaml", "g-success",
        promotionCallbacks(fixture));

    ASSERT_TRUE(result.promoted);
    ASSERT_TRUE(result.safe_terminal_state);
    ASSERT_FALSE(result.backup_directory.empty());
    EXPECT_EQ(fixture.read(fixture.root() / "map.pcd"), "new-pcd\n");
    EXPECT_EQ(fixture.read(fixture.root() / "map.pgm"), "new-pgm\n");
    EXPECT_EQ(fixture.read(fixture.root() / "map_leveling.yaml"),
              "new-sidecar\n");
    EXPECT_FALSE(std::filesystem::exists(fixture.root() / "old_grid.yaml"));
    EXPECT_EQ(fixture.read(result.backup_directory / "map.pcd"), "old-pcd\n");
    EXPECT_EQ(fixture.read(result.backup_directory / "map.pgm"), "old-pgm\n");
    EXPECT_EQ(fixture.read(result.backup_directory / "map_leveling.yaml"),
              "old-sidecar\n");
    EXPECT_EQ(fixture.read(result.backup_directory / "old_grid.yaml"),
              "stale\n");
}

TEST(MapArtifactTransactionTest, RestoresOldGenerationAfterInjectedPromotionFailure)
{
    PromotionFixture fixture;
    fixture.write(fixture.root() / "map.pcd", "old-pcd\n");
    fixture.write(fixture.root() / "map.pgm", "old-pgm\n");
    fixture.write(fixture.root() / "map_leveling.yaml", "old-sidecar\n");
    fixture.write(fixture.root() / "old_grid.yaml", "stale\n");
    fixture.write(fixture.staging() / "map.pcd", "new-pcd\n");
    fixture.write(fixture.staging() / "map.pgm", "new-pgm\n");
    fixture.write(fixture.staging() / "map_leveling.yaml", "new-sidecar\n");

    auto callbacks = promotionCallbacks(fixture);
    callbacks.before_mutation =
        [](const std::filesystem::path& path, const std::string& operation,
           std::string* error)
    {
        if (path == "map.pgm" && operation == "promote artifact")
        {
            if (error != nullptr) *error = "injected promotion failure";
            return false;
        }
        return true;
    };
    const auto result = anubis_mapping::promoteMapArtifactsTransactional(
        promotionEntries(fixture), "map_leveling.yaml", "g-failure",
        callbacks);

    EXPECT_FALSE(result.promoted);
    EXPECT_TRUE(result.rollback_attempted);
    EXPECT_TRUE(result.rollback_restored);
    EXPECT_TRUE(result.safe_terminal_state);
    EXPECT_EQ(fixture.read(fixture.root() / "map.pcd"), "old-pcd\n");
    EXPECT_EQ(fixture.read(fixture.root() / "map.pgm"), "old-pgm\n");
    EXPECT_EQ(fixture.read(fixture.root() / "map_leveling.yaml"),
              "old-sidecar\n");
    EXPECT_EQ(fixture.read(fixture.root() / "old_grid.yaml"), "stale\n");
}

TEST(MapArtifactTransactionTest, VerificationFailureRollsBackAndKeepsSidecarSafe)
{
    PromotionFixture fixture;
    fixture.write(fixture.root() / "map.pcd", "old-pcd\n");
    fixture.write(fixture.root() / "map.pgm", "old-pgm\n");
    fixture.write(fixture.root() / "map_leveling.yaml", "old-sidecar\n");
    fixture.write(fixture.staging() / "map.pcd", "new-pcd\n");
    fixture.write(fixture.staging() / "map.pgm", "new-pgm\n");
    fixture.write(fixture.staging() / "map_leveling.yaml", "new-sidecar\n");

    const auto result = anubis_mapping::promoteMapArtifactsTransactional(
        promotionEntries(fixture), "map_leveling.yaml", "g-verify",
        promotionCallbacks(fixture, false));

    EXPECT_FALSE(result.promoted);
    EXPECT_TRUE(result.rollback_attempted);
    EXPECT_TRUE(result.rollback_restored);
    EXPECT_TRUE(result.safe_terminal_state);
    EXPECT_EQ(fixture.read(fixture.root() / "map.pcd"), "old-pcd\n");
    EXPECT_EQ(fixture.read(fixture.root() / "map.pgm"), "old-pgm\n");
    EXPECT_EQ(fixture.read(fixture.root() / "map_leveling.yaml"),
              "old-sidecar\n");
}

}  // namespace
