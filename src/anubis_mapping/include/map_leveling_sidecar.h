#pragma once

#include "map_artifact_digest.h"

#include <filesystem>
#include <istream>
#include <string>
#include <vector>

namespace anubis_mapping
{

struct MapLevelingMetadata
{
    std::size_t schema_version = 0U;
    std::string generation;
    std::vector<MapArtifactDigest> artifacts;
};

// Parses the versioned save-time leveling sidecar and returns the floor only
// when the complete file represents a self-consistent, consumable terminal
// state. A succeeded=false sidecar is an intentional transaction invalidator.
// succeeded=true with leveling_applied=false is a valid, loadable map that was
// saved without the gravity-leveling correction (either a gate-failed save or
// an explicitly disabled correction); the geometry fields must then describe
// the unmodified map frame.
// When artifact_root is supplied, the manifest is also checked against the
// exact regular-file set and SHA-256 content of that directory.
bool parseValidatedMapLevelingFloor(
    std::istream& input, double& floor_z_map,
    MapLevelingMetadata* metadata = nullptr,
    const std::filesystem::path* artifact_root = nullptr);

bool readValidatedMapLevelingFloor(
    const std::string& file_name, double& floor_z_map);

// Read a semantically valid sidecar and expose its generation/manifest
// metadata without validating the files on disk.  This is used only while
// preparing a replacement transaction so previously declared artifacts can be
// included in the rollback/cleanup set; production consumers should use the
// artifact-root overload below.
bool readValidatedMapLevelingFloor(
    const std::string& file_name, double& floor_z_map,
    MapLevelingMetadata* metadata);

bool readValidatedMapLevelingFloor(
    const std::string& file_name, double& floor_z_map,
    const std::filesystem::path& artifact_root,
    MapLevelingMetadata* metadata = nullptr);

}  // namespace anubis_mapping
