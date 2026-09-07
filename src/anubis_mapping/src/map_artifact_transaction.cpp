#include "map_artifact_transaction.h"

#include <algorithm>
#include <set>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace
{

void rememberFailure(
    anubis_mapping::MapArtifactPromotionResult& result,
    const std::filesystem::path& path, const std::string& operation,
    const std::string& detail = {})
{
    if (result.failed_operation.empty())
    {
        result.failed_path = path;
        result.failed_operation = operation;
        result.detail = detail;
    }
}

bool isSafeRelativePath(const std::filesystem::path& path)
{
    if (path.empty() || path.is_absolute() || path != path.lexically_normal())
    {
        return false;
    }
    for (const auto& component : path)
    {
        const std::string value = component.string();
        if (value.empty() || value == "." || value == "..")
        {
            return false;
        }
    }
    return true;
}

bool isSafeTransactionId(const std::string& value)
{
    return !value.empty() && value.size() <= 128U &&
        std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return (character >= 'a' && character <= 'z') ||
                (character >= 'A' && character <= 'Z') ||
                (character >= '0' && character <= '9') || character == '-' ||
                character == '_';
        });
}

bool regularFileOrMissing(
    const std::filesystem::path& path, bool& exists, std::error_code& error)
{
    error.clear();
    exists = std::filesystem::exists(path, error);
    if (error || !exists)
    {
        return !error;
    }
    const auto status = std::filesystem::symlink_status(path, error);
    return !error && !std::filesystem::is_symlink(status) &&
        std::filesystem::is_regular_file(status);
}

bool promoteFile(
    const std::filesystem::path& source,
    const std::filesystem::path& destination, std::error_code& error)
{
    bool source_exists = false;
    if (!regularFileOrMissing(source, source_exists, error) || !source_exists)
    {
        if (!error)
        {
            error = std::make_error_code(std::errc::no_such_file_or_directory);
        }
        return false;
    }

    std::filesystem::rename(source, destination, error);
    if (!error)
    {
        return true;
    }

#ifndef _WIN32
    // POSIX rename already replaces an existing regular file atomically.  Any
    // failure here is a real filesystem error; never delete the last valid
    // destination in an attempt to recover from permission, I/O, or device
    // errors.
    return false;
#else
    // The standard Windows rename does not replace an existing file.  Retry
    // only that documented conflict class and use the replace-existing OS
    // primitive, which leaves the destination intact when replacement fails.
    if (error != std::errc::file_exists &&
        error != std::errc::permission_denied)
    {
        return false;
    }
    bool destination_exists = false;
    std::error_code destination_error;
    if (!regularFileOrMissing(
            destination, destination_exists, destination_error) ||
        !destination_exists)
    {
        if (destination_error)
        {
            error = destination_error;
        }
        return false;
    }
    if (!MoveFileExW(
            source.c_str(), destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    {
        error = std::error_code(
            static_cast<int>(GetLastError()), std::system_category());
        return false;
    }
    error.clear();
    return true;
#endif
}

bool mutationAllowed(
    const anubis_mapping::MapArtifactPromotionCallbacks& callbacks,
    const std::filesystem::path& relative_path, const std::string& operation,
    anubis_mapping::MapArtifactPromotionResult& result)
{
    if (!callbacks.before_mutation)
    {
        return true;
    }
    std::string detail;
    if (callbacks.before_mutation(relative_path, operation, &detail))
    {
        return true;
    }
    rememberFailure(result, relative_path, operation,
                    detail.empty() ? "rejected by mutation hook" : detail);
    return false;
}

}  // namespace

namespace anubis_mapping
{

MapArtifactRollbackResult rollbackMapArtifactsFailClosed(
    const std::vector<MapArtifactRollbackEntry>& entries,
    const std::filesystem::path& sidecar_relative_path,
    const std::function<bool()>& force_sidecar_invalid,
    const MapArtifactFileOperations& file_operations)
{
    MapArtifactRollbackResult result;
    if (!force_sidecar_invalid || !file_operations.copy_file ||
        !file_operations.remove_file)
    {
        result.failed_operation = "invalid rollback callbacks";
        return result;
    }

    const MapArtifactRollbackEntry* sidecar = nullptr;
    for (const auto& entry : entries)
    {
        if (entry.relative_path == sidecar_relative_path)
        {
            if (sidecar != nullptr)
            {
                result.failed_path = entry.destination_path;
                result.failed_operation = "duplicate sidecar entry";
                result.safe_terminal_state = force_sidecar_invalid();
                return result;
            }
            sidecar = &entry;
        }
    }
    if (sidecar == nullptr)
    {
        result.failed_operation = "missing sidecar entry";
        result.safe_terminal_state = force_sidecar_invalid();
        return result;
    }

    bool rollback_ok = force_sidecar_invalid();
    if (!rollback_ok)
    {
        result.failed_path = sidecar->destination_path;
        result.failed_operation = "initial sidecar invalidation";
        // Never mutate data while a previously valid marker may still be
        // visible.  The caller must retry or leave the map unavailable.
        result.safe_terminal_state = false;
        return result;
    }

    const auto remember_failure = [&result](
        const std::filesystem::path& path, const char* operation,
        const std::error_code& error)
    {
        if (result.failed_operation.empty())
        {
            result.failed_path = path;
            result.failed_operation = operation;
            result.error = error;
        }
    };

    for (const auto& entry : entries)
    {
        if (entry.relative_path == sidecar_relative_path)
        {
            continue;
        }
        std::error_code operation_error;
        bool operation_ok = true;
        if (entry.had_destination)
        {
            operation_ok = file_operations.copy_file(
                entry.backup_path, entry.destination_path, operation_error);
            if (!operation_ok || operation_error)
            {
                remember_failure(
                    entry.destination_path, "restore artifact", operation_error);
                rollback_ok = false;
            }
        }
        else
        {
            file_operations.remove_file(entry.destination_path, operation_error);
            if (operation_error == std::make_error_code(
                    std::errc::no_such_file_or_directory))
            {
                operation_error.clear();
            }
            if (operation_error)
            {
                remember_failure(
                    entry.destination_path, "remove new artifact",
                    operation_error);
                rollback_ok = false;
            }
        }
    }

    if (rollback_ok)
    {
        std::error_code sidecar_error;
        if (sidecar->had_destination)
        {
            const bool copied = file_operations.copy_file(
                sidecar->backup_path, sidecar->destination_path, sidecar_error);
            if (!copied || sidecar_error)
            {
                remember_failure(
                    sidecar->destination_path, "restore sidecar", sidecar_error);
                rollback_ok = false;
            }
        }
        else
        {
            file_operations.remove_file(sidecar->destination_path, sidecar_error);
            if (sidecar_error == std::make_error_code(
                    std::errc::no_such_file_or_directory))
            {
                sidecar_error.clear();
            }
            if (sidecar_error)
            {
                remember_failure(
                    sidecar->destination_path, "remove new sidecar",
                    sidecar_error);
                rollback_ok = false;
            }
        }
    }

    result.restored = rollback_ok;
    if (rollback_ok)
    {
        result.safe_terminal_state = true;
        return result;
    }

    result.safe_terminal_state = force_sidecar_invalid();
    return result;
}

MapArtifactPromotionResult promoteMapArtifactsTransactional(
    const std::vector<MapArtifactPromotionEntry>& entries,
    const std::filesystem::path& sidecar_relative_path,
    const std::string& transaction_id,
    const MapArtifactPromotionCallbacks& callbacks)
{
    MapArtifactPromotionResult result;
    result.safe_terminal_state = true;
    if (entries.empty() || !isSafeRelativePath(sidecar_relative_path) ||
        !isSafeTransactionId(transaction_id) ||
        !callbacks.write_invalid_sidecar || !callbacks.verify_destination)
    {
        result.safe_terminal_state = false;
        result.failed_operation = "invalid promotion arguments";
        return result;
    }

    const MapArtifactPromotionEntry* sidecar = nullptr;
    std::set<std::filesystem::path> relative_paths;
    for (const auto& entry : entries)
    {
        if (!isSafeRelativePath(entry.relative_path) ||
            !relative_paths.insert(entry.relative_path).second)
        {
            rememberFailure(
                result, entry.relative_path, "invalid or duplicate relative path");
            return result;
        }
        if (entry.relative_path == sidecar_relative_path)
        {
            if (sidecar != nullptr || entry.remove_destination)
            {
                rememberFailure(
                    result, entry.relative_path, "invalid sidecar entry");
                return result;
            }
            sidecar = &entry;
        }
    }
    if (sidecar == nullptr)
    {
        result.failed_operation = "missing sidecar entry";
        return result;
    }

    const std::filesystem::path destination_root =
        sidecar->destination_path.parent_path().lexically_normal();
    const std::filesystem::path staging_root =
        sidecar->staged_path.parent_path().lexically_normal();
    std::error_code root_error;
    if (destination_root.empty() || staging_root.empty() ||
        !std::filesystem::is_directory(destination_root, root_error) || root_error)
    {
        result.failed_path = destination_root;
        result.failed_operation = "invalid destination root";
        result.detail = root_error.message();
        return result;
    }

    for (const auto& entry : entries)
    {
        if (entry.destination_path.lexically_normal() !=
            (destination_root / entry.relative_path).lexically_normal() ||
            (!entry.remove_destination &&
             entry.staged_path.lexically_normal() !=
                 (staging_root / entry.relative_path).lexically_normal()))
        {
            rememberFailure(result, entry.relative_path,
                            "artifact path escapes transaction root");
            return result;
        }
        bool staged_exists = false;
        std::error_code staged_error;
        if (!entry.remove_destination &&
            (!regularFileOrMissing(
                 entry.staged_path, staged_exists, staged_error) ||
             !staged_exists))
        {
            rememberFailure(
                result, entry.staged_path, "invalid staged artifact",
                staged_error ? staged_error.message() : "not a regular file");
            return result;
        }
    }

    for (int attempt = 0; attempt < 8 && result.backup_directory.empty(); ++attempt)
    {
        const std::filesystem::path candidate =
            destination_root /
            (".map_save_backup_" + transaction_id +
             (attempt == 0 ? std::string() : "_" + std::to_string(attempt)));
        std::error_code create_error;
        if (std::filesystem::create_directory(candidate, create_error))
        {
            result.backup_directory = candidate;
        }
        else if (create_error &&
                 create_error != std::make_error_code(std::errc::file_exists))
        {
            rememberFailure(result, candidate, "create backup directory",
                            create_error.message());
            return result;
        }
    }
    if (result.backup_directory.empty())
    {
        result.failed_operation = "backup directory name collision";
        return result;
    }

    std::vector<MapArtifactRollbackEntry> rollback_entries;
    rollback_entries.reserve(entries.size());
    for (const auto& entry : entries)
    {
        bool destination_exists = false;
        std::error_code destination_error;
        if (!regularFileOrMissing(
                entry.destination_path, destination_exists, destination_error))
        {
            rememberFailure(
                result, entry.destination_path, "invalid destination artifact",
                destination_error ? destination_error.message() :
                    "not a regular file");
            return result;
        }
        const std::filesystem::path backup_path =
            result.backup_directory / entry.relative_path;
        rollback_entries.push_back({
            entry.relative_path, entry.destination_path, backup_path,
            destination_exists});
        if (!destination_exists)
        {
            continue;
        }
        std::error_code parent_error;
        std::filesystem::create_directories(
            backup_path.parent_path(), parent_error);
        if (parent_error)
        {
            rememberFailure(result, backup_path, "create backup parent",
                            parent_error.message());
            return result;
        }
        std::error_code copy_error;
        if (!std::filesystem::copy_file(
                entry.destination_path, backup_path,
                std::filesystem::copy_options::overwrite_existing,
                copy_error) || copy_error)
        {
            rememberFailure(
                result, entry.destination_path, "backup artifact",
                copy_error ? copy_error.message() : "copy failed");
            return result;
        }
    }

    const std::filesystem::path invalidation_file =
        destination_root /
        (".map_leveling_invalid_" + transaction_id + ".yaml");
    const auto force_sidecar_invalid = [&]()
    {
        std::error_code cleanup_error;
        std::filesystem::remove(invalidation_file, cleanup_error);
        if (callbacks.write_invalid_sidecar(invalidation_file))
        {
            std::error_code promote_error;
            if (promoteFile(
                    invalidation_file, sidecar->destination_path,
                    promote_error))
            {
                return true;
            }
        }
        std::error_code remove_error;
        std::filesystem::remove(sidecar->destination_path, remove_error);
        return !remove_error || remove_error ==
            std::make_error_code(std::errc::no_such_file_or_directory);
    };

    if (!mutationAllowed(
            callbacks, sidecar_relative_path, "invalidate sidecar", result))
    {
        // The destination has not been touched, so the previously validated
        // generation remains safe and available.
        result.safe_terminal_state = true;
        return result;
    }
    if (!callbacks.write_invalid_sidecar(invalidation_file))
    {
        rememberFailure(result, invalidation_file, "write invalid sidecar");
        std::error_code cleanup_error;
        std::filesystem::remove(invalidation_file, cleanup_error);
        result.safe_terminal_state = true;
        return result;
    }
    std::error_code invalidation_error;
    if (!promoteFile(
            invalidation_file, sidecar->destination_path,
            invalidation_error))
    {
        rememberFailure(result, sidecar->destination_path,
                        "invalidate sidecar", invalidation_error.message());
        result.safe_terminal_state = force_sidecar_invalid();
        return result;
    }

    bool promotion_ok = true;
    for (const auto& entry : entries)
    {
        if (entry.relative_path == sidecar_relative_path)
        {
            continue;
        }
        const std::string operation = entry.remove_destination
            ? "remove stale artifact" : "promote artifact";
        if (!mutationAllowed(
                callbacks, entry.relative_path, operation, result))
        {
            promotion_ok = false;
            break;
        }
        std::error_code operation_error;
        if (entry.remove_destination)
        {
            std::filesystem::remove(entry.destination_path, operation_error);
            if (operation_error ==
                std::make_error_code(std::errc::no_such_file_or_directory))
            {
                operation_error.clear();
            }
        }
        else if (!promoteFile(
                     entry.staged_path, entry.destination_path,
                     operation_error))
        {
            // promoteFile reports the concrete filesystem error below.
        }
        if (operation_error)
        {
            rememberFailure(result, entry.destination_path, operation,
                            operation_error.message());
            promotion_ok = false;
            break;
        }
    }

    if (promotion_ok &&
        !mutationAllowed(
            callbacks, sidecar_relative_path, "promote sidecar", result))
    {
        promotion_ok = false;
    }
    if (promotion_ok)
    {
        std::error_code sidecar_error;
        if (!promoteFile(
                sidecar->staged_path, sidecar->destination_path,
                sidecar_error))
        {
            rememberFailure(result, sidecar->destination_path,
                            "promote sidecar", sidecar_error.message());
            promotion_ok = false;
        }
    }

    if (promotion_ok)
    {
        std::string verification_error;
        if (!callbacks.verify_destination(&verification_error))
        {
            rememberFailure(result, destination_root,
                            "verify promoted generation", verification_error);
            promotion_ok = false;
        }
    }
    if (promotion_ok)
    {
        std::error_code cleanup_error;
        std::filesystem::remove(invalidation_file, cleanup_error);
        result.promoted = true;
        result.safe_terminal_state = true;
        return result;
    }

    result.rollback_attempted = true;
    MapArtifactFileOperations file_operations;
    file_operations.copy_file = [](
        const std::filesystem::path& source,
        const std::filesystem::path& destination, std::error_code& error)
    {
        return std::filesystem::copy_file(
            source, destination,
            std::filesystem::copy_options::overwrite_existing, error);
    };
    file_operations.remove_file = [](
        const std::filesystem::path& path, std::error_code& error)
    {
        std::filesystem::remove(path, error);
    };
    const MapArtifactRollbackResult rollback =
        rollbackMapArtifactsFailClosed(
            rollback_entries, sidecar_relative_path,
            force_sidecar_invalid, file_operations);
    result.rollback_restored = rollback.restored;
    result.safe_terminal_state = rollback.safe_terminal_state;
    if (!rollback.restored && result.failed_operation.empty())
    {
        result.failed_path = rollback.failed_path;
        result.failed_operation = rollback.failed_operation;
        result.detail = rollback.error.message();
    }
    std::error_code cleanup_error;
    std::filesystem::remove(invalidation_file, cleanup_error);
    return result;
}

}  // namespace anubis_mapping
