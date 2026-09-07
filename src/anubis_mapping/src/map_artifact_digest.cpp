#include "map_artifact_digest.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <sstream>
#include <system_error>

#include <openssl/evp.h>

namespace
{

void setError(std::string* output, const std::string& message)
{
    if (output != nullptr)
    {
        *output = message;
    }
}

bool digestFile(
    const std::filesystem::path& path, std::uintmax_t& size,
    std::string& digest, std::string* error_message)
{
    std::error_code size_error;
    size = std::filesystem::file_size(path, size_error);
    if (size_error)
    {
        setError(error_message, "file_size failed for " + path.string() + ": " +
            size_error.message());
        return false;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open())
    {
        setError(error_message, "cannot open " + path.string());
        return false;
    }

    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr || EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1)
    {
        if (context != nullptr)
        {
            EVP_MD_CTX_free(context);
        }
        setError(error_message, "cannot initialize SHA-256");
        return false;
    }

    std::array<char, 64U * 1024U> buffer{};
    bool ok = true;
    while (input)
    {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0 && EVP_DigestUpdate(
                context, buffer.data(), static_cast<std::size_t>(count)) != 1)
        {
            ok = false;
            break;
        }
    }
    if (input.bad())
    {
        ok = false;
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> raw_digest{};
    unsigned int digest_size = 0U;
    if (ok && EVP_DigestFinal_ex(
                  context, raw_digest.data(), &digest_size) != 1)
    {
        ok = false;
    }
    EVP_MD_CTX_free(context);
    if (!ok || digest_size != 32U)
    {
        setError(error_message, "SHA-256 failed for " + path.string());
        return false;
    }

    std::ostringstream encoded;
    encoded << std::hex;
    for (unsigned int index = 0U; index < digest_size; ++index)
    {
        encoded.width(2);
        encoded.fill('0');
        encoded << static_cast<unsigned int>(raw_digest[index]);
    }
    digest = encoded.str();
    return true;
}

bool isHexDigest(const std::string& value)
{
    if (value.size() != 64U)
    {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isxdigit(character) != 0;
    });
}

bool hasPrefix(const std::string& value, const char* prefix)
{
    const std::string prefix_text(prefix);
    return value.size() >= prefix_text.size() &&
        value.compare(0U, prefix_text.size(), prefix_text) == 0;
}

}  // namespace

namespace anubis_mapping
{

bool isSafeMapArtifactRelativePath(const std::filesystem::path& path)
{
    if (path.empty() || path.is_absolute())
    {
        return false;
    }
    const std::filesystem::path normalized = path.lexically_normal();
    if (normalized.empty() || normalized == "." || normalized == ".." ||
        normalized != path)
    {
        return false;
    }
    for (const auto& component : normalized)
    {
        const std::string text = component.string();
        if (text.empty() || text == "." || text == "..")
        {
            return false;
        }
        for (const unsigned char character : text)
        {
            if (character < 0x20U || character == 0x7fU || character == '\0' ||
                character == ':' || character == '\\' || character == '/')
            {
                return false;
            }
        }
    }
    return normalized == path.lexically_normal();
}

bool isMapArtifactTransientPath(const std::filesystem::path& path)
{
    if (path.empty() || path.is_absolute())
    {
        return false;
    }
    const std::filesystem::path normalized = path.lexically_normal();
    if (normalized != path)
    {
        return false;
    }
    for (const auto& component : normalized)
    {
        const std::string name = component.string();
        if (hasPrefix(name, ".map_save_backup_") ||
            hasPrefix(name, ".map_save_tmp_"))
        {
            return true;
        }
    }
    // Invalidation markers are regular files at the map root.  They are not
    // ignored: their presence means a promotion may have been interrupted,
    // so exact-set validation must fail closed.  Only workspace directories
    // are transient here.
    return false;
}

bool collectMapArtifactDigests(
    const std::filesystem::path& root,
    const std::filesystem::path& excluded_relative,
    std::vector<MapArtifactDigest>& digests,
    std::string* error_message)
{
    digests.clear();
    if (!isSafeMapArtifactRelativePath(excluded_relative))
    {
        setError(error_message, "invalid excluded relative path");
        return false;
    }
    std::error_code root_error;
    if (!std::filesystem::is_directory(root, root_error) || root_error)
    {
        setError(error_message, "artifact root is not a directory: " + root.string());
        return false;
    }

    std::error_code iterate_error;
    for (std::filesystem::recursive_directory_iterator iterator(
             root, std::filesystem::directory_options::none, iterate_error),
         end;
         iterator != end; iterator.increment(iterate_error))
    {
        if (iterate_error)
        {
            setError(error_message, "artifact traversal failed: " +
                iterate_error.message());
            return false;
        }
        const auto& entry = *iterator;
        std::error_code status_error;
        const auto status = entry.symlink_status(status_error);
        if (status_error)
        {
            setError(error_message, "artifact status failed for " +
                entry.path().string() + ": " + status_error.message());
            return false;
        }
        const std::filesystem::path relative =
            std::filesystem::relative(entry.path(), root, status_error)
                .lexically_normal();
        if (status_error || !isSafeMapArtifactRelativePath(relative))
        {
            setError(error_message, "unsafe artifact path: " + entry.path().string());
            return false;
        }
        if (relative == excluded_relative)
        {
            continue;
        }
        if (std::filesystem::is_directory(status))
        {
            if (isMapArtifactTransientPath(relative))
            {
                iterator.disable_recursion_pending();
            }
            continue;
        }
        if (!std::filesystem::is_regular_file(status))
        {
            setError(error_message, "artifact is not a regular file: " +
                entry.path().string());
            return false;
        }

        MapArtifactDigest digest;
        digest.relative_path = relative;
        if (!digestFile(entry.path(), digest.size, digest.sha256, error_message))
        {
            return false;
        }
        digests.push_back(std::move(digest));
    }
    if (iterate_error)
    {
        setError(error_message, "artifact traversal failed: " +
            iterate_error.message());
        return false;
    }
    std::sort(digests.begin(), digests.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.relative_path.generic_string() < rhs.relative_path.generic_string();
    });
    for (std::size_t index = 1U; index < digests.size(); ++index)
    {
        if (digests[index - 1U].relative_path == digests[index].relative_path)
        {
            setError(error_message, "duplicate artifact path");
            return false;
        }
    }
    return !digests.empty();
}

bool validateMapArtifactDigests(
    const std::filesystem::path& root,
    const std::vector<MapArtifactDigest>& expected,
    const std::filesystem::path& excluded_relative,
    std::string* error_message)
{
    if (expected.empty())
    {
        setError(error_message, "artifact manifest is empty");
        return false;
    }
    std::vector<MapArtifactDigest> actual;
    if (!collectMapArtifactDigests(root, excluded_relative, actual,
                                   error_message))
    {
        return false;
    }
    std::vector<MapArtifactDigest> sorted_expected = expected;
    std::sort(sorted_expected.begin(), sorted_expected.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.relative_path.generic_string() <
                      rhs.relative_path.generic_string();
              });
    if (sorted_expected.size() != actual.size())
    {
        setError(error_message, "artifact count does not match manifest");
        return false;
    }
    for (std::size_t index = 0U; index < sorted_expected.size(); ++index)
    {
        const auto& expected_item = sorted_expected[index];
        const auto& actual_item = actual[index];
        if (!isSafeMapArtifactRelativePath(expected_item.relative_path) ||
            !isHexDigest(expected_item.sha256) ||
            expected_item.relative_path != actual_item.relative_path ||
            expected_item.size != actual_item.size ||
            expected_item.sha256 != actual_item.sha256)
        {
            setError(error_message, "artifact digest mismatch at " +
                expected_item.relative_path.string());
            return false;
        }
    }
    return true;
}

}  // namespace anubis_mapping
