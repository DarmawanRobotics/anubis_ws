#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "localization/map_artifact_manifest.hpp"

namespace localization {
namespace {

constexpr char kEmptySha256[] =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

struct ArtifactSpec {
  std::string path;
  std::uintmax_t size = 0U;
  std::string sha256 = kEmptySha256;
};

class ManifestFixture {
 public:
  ManifestFixture() {
    const auto suffix = std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count();
    root_ = std::filesystem::temp_directory_path() /
            ("localization_map_manifest_test_" + std::to_string(suffix));
    std::error_code directory_error;
    std::filesystem::create_directories(root_, directory_error);
    if (directory_error || !std::filesystem::is_directory(root_)) {
      throw std::runtime_error("cannot create manifest test directory");
    }
  }

  ~ManifestFixture() {
    std::error_code ignored;
    std::filesystem::remove_all(root_, ignored);
  }

  const std::filesystem::path& root() const { return root_; }

  void writeFile(const std::string& relative, const std::string& contents = {}) {
    const std::filesystem::path path = root_ / relative;
    std::error_code directory_error;
    std::filesystem::create_directories(path.parent_path(), directory_error);
    ASSERT_FALSE(directory_error) << directory_error.message();
    ASSERT_TRUE(std::filesystem::is_directory(path.parent_path()));
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output.is_open());
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    ASSERT_TRUE(output.good());
  }

  std::vector<ArtifactSpec> baseArtifacts() const {
    return {{"map.pcd", 0U, kEmptySha256},
            {"map.txt", 0U, kEmptySha256},
            {"map.pgm", 0U, kEmptySha256},
            {"map.yaml", 0U, kEmptySha256}};
  }

  void writeManifest(const std::vector<ArtifactSpec>& artifacts,
                     std::size_t schema_version = 3U,
                     bool succeeded = true,
                     const std::string& generation = "g-test") {
    std::ofstream output(root_ / "map_leveling.yaml",
                         std::ios::out | std::ios::trunc);
    ASSERT_TRUE(output.is_open());
    output << "schema: map_leveling\n"
           << "schema_version: " << schema_version << "\n"
           << "generation: " << generation << "\n"
           << "artifact_count: " << artifacts.size() << "\n"
           << "enabled: true\n"
           << "succeeded: " << (succeeded ? "true" : "false") << "\n";
    for (std::size_t index = 0U; index < artifacts.size(); ++index) {
      output << "artifact_" << index << "_path: " << artifacts[index].path
             << "\n"
             << "artifact_" << index << "_size: " << artifacts[index].size
             << "\n"
             << "artifact_" << index << "_sha256: "
             << artifacts[index].sha256 << "\n";
    }
    output.flush();
    ASSERT_TRUE(output.good());
  }

  void createBaseMap() {
    for (const auto& artifact : baseArtifacts()) {
      writeFile(artifact.path);
    }
    writeManifest(baseArtifacts());
  }

 private:
  std::filesystem::path root_;
};

TEST(MapArtifactManifest, ValidV3ManifestBindsExactGeneration) {
  ManifestFixture fixture;
  fixture.createBaseMap();

  MapArtifactManifest manifest;
  std::string error;
  ASSERT_TRUE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error))
      << error;
  EXPECT_EQ(manifest.generation, "g-test");
  EXPECT_EQ(manifest.artifact_root, fixture.root());
  EXPECT_TRUE(manifest.contains("map.pcd"));
  EXPECT_FALSE(manifest.contains("map_scd.bin"));
}

TEST(MapArtifactManifest, BackupDirectoryIsIgnoredButStagingIsRejected) {
  ManifestFixture fixture;
  fixture.createBaseMap();
  fixture.writeFile(".map_save_backup_old/old-map.pcd", "stale");

  MapArtifactManifest manifest;
  std::string error;
  EXPECT_TRUE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error))
      << error;

  fixture.writeFile(".map_save_tmp_g-test/partial", "partial");
  error.clear();
  EXPECT_FALSE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error));
  EXPECT_NE(error.find("unfinished map transaction"), std::string::npos);
}

TEST(MapArtifactManifest, MissingOrInvalidSidecarIsRejected) {
  ManifestFixture fixture;
  fixture.createBaseMap();
  std::error_code ignored;
  std::filesystem::remove(fixture.root() / "map_leveling.yaml", ignored);

  MapArtifactManifest manifest;
  std::string error;
  EXPECT_FALSE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error));

  fixture.writeManifest(fixture.baseArtifacts(), 2U, true);
  EXPECT_FALSE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error));

  fixture.writeManifest(fixture.baseArtifacts(), 3U, false);
  EXPECT_FALSE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error));
}

TEST(MapArtifactManifest, ContentSizeAndExtraArtifactChangesAreRejected) {
  ManifestFixture fixture;
  fixture.createBaseMap();
  fixture.writeFile("map.pcd", "x");

  MapArtifactManifest manifest;
  std::string error;
  EXPECT_FALSE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error));
  EXPECT_NE(error.find("digest mismatch"), std::string::npos);

  fixture.writeFile("map.pcd");
  fixture.writeFile("old_map_scd.bin");
  error.clear();
  EXPECT_FALSE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error));
  EXPECT_NE(error.find("artifact count"), std::string::npos);
}

TEST(MapArtifactManifest, UnsafeDuplicateAndUnboundEntriesAreRejected) {
  ManifestFixture fixture;
  fixture.createBaseMap();
  auto artifacts = fixture.baseArtifacts();
  artifacts[0].path = "../map.pcd";
  fixture.writeManifest(artifacts);

  MapArtifactManifest manifest;
  std::string error;
  EXPECT_FALSE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error));

  artifacts = fixture.baseArtifacts();
  artifacts.push_back(artifacts.front());
  fixture.writeManifest(artifacts);
  EXPECT_FALSE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error));

  fixture.writeManifest(fixture.baseArtifacts());
  fixture.writeFile("other.pcd");
  EXPECT_FALSE(validateMapArtifactManifest(
      fixture.root() / "other.pcd", manifest, &error));
}

TEST(MapArtifactManifest, SymbolicLinksAreNeverAccepted) {
  ManifestFixture fixture;
  fixture.createBaseMap();
  std::error_code symlink_error;
  std::filesystem::create_symlink(
      fixture.root() / "map.pcd", fixture.root() / "linked.pcd",
      symlink_error);
  if (symlink_error) {
    GTEST_SKIP() << "symbolic links are unavailable: " << symlink_error.message();
  }

  MapArtifactManifest manifest;
  std::string error;
  EXPECT_FALSE(validateMapArtifactManifest(
      fixture.root() / "map.pcd", manifest, &error));
  EXPECT_NE(error.find("symbolic link"), std::string::npos);
}

}  // namespace
}  // namespace localization
