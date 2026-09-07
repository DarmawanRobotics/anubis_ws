#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <system_error>
#include <vector>

namespace anubis_mapping
{

struct MapArtifactRollbackEntry
{
    std::filesystem::path relative_path;
    std::filesystem::path destination_path;
    std::filesystem::path backup_path;
    bool had_destination = false;
};

struct MapArtifactFileOperations
{
    std::function<bool(
        const std::filesystem::path&, const std::filesystem::path&,
        std::error_code&)> copy_file;
    std::function<void(const std::filesystem::path&, std::error_code&)>
        remove_file;
};

struct MapArtifactRollbackResult
{
    bool restored = false;
    bool safe_terminal_state = false;
    std::filesystem::path failed_path;
    std::string failed_operation;
    std::error_code error;
};

struct MapArtifactPromotionEntry
{
    std::filesystem::path relative_path;
    std::filesystem::path staged_path;
    std::filesystem::path destination_path;
    bool remove_destination = false;
};

struct MapArtifactPromotionCallbacks
{
    // Writes a syntactically valid, succeeded=false sidecar at the requested
    // temporary path.  The transaction installs it before touching data.
    std::function<bool(const std::filesystem::path&)> write_invalid_sidecar;

    // Validates the complete promoted generation, including its sidecar and
    // content digests.  It is called only after the sidecar is promoted last.
    std::function<bool(std::string*)> verify_destination;

    // Optional deterministic fault hook for integration tests.  Returning
    // false aborts before the named mutation is attempted.
    std::function<bool(
        const std::filesystem::path&, const std::string&, std::string*)>
        before_mutation;
};

struct MapArtifactPromotionResult
{
    bool promoted = false;
    bool rollback_attempted = false;
    bool rollback_restored = false;
    bool safe_terminal_state = false;
    std::filesystem::path backup_directory;
    std::filesystem::path failed_path;
    std::string failed_operation;
    std::string detail;
};

// The validity sidecar is invalidated before restoring data and restored last.
// If any operation fails, force_sidecar_invalid is called again so a partial
// old/new artifact generation cannot be advertised as consumable.
MapArtifactRollbackResult rollbackMapArtifactsFailClosed(
    const std::vector<MapArtifactRollbackEntry>& entries,
    const std::filesystem::path& sidecar_relative_path,
    const std::function<bool()>& force_sidecar_invalid,
    const MapArtifactFileOperations& file_operations);

// Publishes a complete staged generation with the validity sidecar installed
// last. Existing destinations are backed up first, stale managed files can be
// removed transactionally, and any post-invalidation failure restores the old
// generation while keeping the sidecar fail-closed throughout rollback.
MapArtifactPromotionResult promoteMapArtifactsTransactional(
    const std::vector<MapArtifactPromotionEntry>& entries,
    const std::filesystem::path& sidecar_relative_path,
    const std::string& transaction_id,
    const MapArtifactPromotionCallbacks& callbacks);

}  // namespace anubis_mapping
