#ifndef LOCALIZATION_MAP_ARTIFACT_MANIFEST_HPP_
#define LOCALIZATION_MAP_ARTIFACT_MANIFEST_HPP_

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace localization {

struct MapArtifactManifestEntry {
  std::filesystem::path relative_path;
  std::uintmax_t size = 0U;
  std::string sha256;
};

struct MapArtifactManifest {
  std::filesystem::path artifact_root;
  std::string generation;
  std::vector<MapArtifactManifestEntry> artifacts;

  const MapArtifactManifestEntry* find(
      const std::filesystem::path& relative_path) const noexcept;
  bool contains(const std::filesystem::path& relative_path) const noexcept;
};

// Validate the v3 save sidecar and bind the requested PCD to one exact
// artifact generation.  The validator checks the complete regular-file set
// and SHA-256 digests before a caller reads any map data.
bool validateMapArtifactManifest(
    const std::filesystem::path& map_path,
    MapArtifactManifest& manifest,
    std::string* error_message = nullptr);

}  // namespace localization

#endif  // LOCALIZATION_MAP_ARTIFACT_MANIFEST_HPP_
