#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace anubis_mapping
{

struct MapArtifactDigest
{
    std::filesystem::path relative_path;
    std::uintmax_t size = 0U;
    std::string sha256;
};

// Collect every regular file below root except excluded_relative.  The
// returned list is sorted by portable relative path and is suitable for a
// versioned map manifest.
bool collectMapArtifactDigests(
    const std::filesystem::path& root,
    const std::filesystem::path& excluded_relative,
    std::vector<MapArtifactDigest>& digests,
    std::string* error_message = nullptr);

// Verify both the exact file set and the content digest.  Extra or missing
// regular files are rejected so a valid sidecar cannot describe a mixed
// generation.
bool validateMapArtifactDigests(
    const std::filesystem::path& root,
    const std::vector<MapArtifactDigest>& expected,
    const std::filesystem::path& excluded_relative,
    std::string* error_message = nullptr);

bool isSafeMapArtifactRelativePath(const std::filesystem::path& path);

// Transaction workspaces are deliberately kept below the map directory so a
// crash can be recovered without exposing a mixed generation.  They are not
// map artifacts and must therefore be ignored by both manifest creation and
// validation once a promotion has completed.
bool isMapArtifactTransientPath(const std::filesystem::path& path);

}  // namespace anubis_mapping
