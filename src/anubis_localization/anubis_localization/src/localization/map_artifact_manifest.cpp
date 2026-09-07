#include "localization/map_artifact_manifest.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <openssl/evp.h>

namespace localization {
namespace {

constexpr std::size_t kMaxManifestArtifacts = 64U;
constexpr char kSidecarName[] = "map_leveling.yaml";
constexpr char kBackupPrefix[] = ".map_save_backup_";
constexpr char kStagingPrefix[] = ".map_save_tmp_";
constexpr char kInvalidPrefix[] = ".map_leveling_invalid_";

void setError(std::string* output, const std::string& message) {
  if (output != nullptr) {
    *output = message;
  }
}

bool fail(std::string* output, const std::string& message) {
  setError(output, message);
  return false;
}

std::string trim(std::string value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1U);
}

bool parseUnsigned(const std::string& value, std::uintmax_t& output) {
  const std::string normalized = trim(value);
  if (normalized.empty() || normalized.front() == '-') {
    return false;
  }
  try {
    std::size_t consumed = 0U;
    const unsigned long long parsed = std::stoull(normalized, &consumed, 10);
    if (consumed != normalized.size() ||
        parsed > static_cast<unsigned long long>(
                     std::numeric_limits<std::uintmax_t>::max())) {
      return false;
    }
    output = static_cast<std::uintmax_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

bool parseBool(const std::string& value, bool& output) {
  const std::string normalized = trim(value);
  if (normalized == "true" || normalized == "yes" || normalized == "1") {
    output = true;
    return true;
  }
  if (normalized == "false" || normalized == "no" || normalized == "0") {
    output = false;
    return true;
  }
  return false;
}

bool validGeneration(const std::string& generation) {
  if (generation.empty() || generation.size() > 128U) {
    return false;
  }
  return std::all_of(generation.begin(), generation.end(),
                     [](unsigned char character) {
                       return std::isalnum(character) != 0 ||
                              character == '-' || character == '_' ||
                              character == '.';
                     });
}

bool validSha256(const std::string& digest) {
  if (digest.size() != 64U) {
    return false;
  }
  return std::all_of(digest.begin(), digest.end(),
                     [](unsigned char character) {
                       return (character >= '0' && character <= '9') ||
                              (character >= 'a' && character <= 'f');
                     });
}

bool safeRelativePath(const std::filesystem::path& path) {
  if (path.empty() || path.is_absolute()) {
    return false;
  }
  const std::filesystem::path normalized = path.lexically_normal();
  if (normalized.empty() || normalized == "." || normalized == ".." ||
      normalized != path) {
    return false;
  }
  for (const auto& component : normalized) {
    const std::string text = component.string();
    if (text.empty() || text == "." || text == "..") {
      return false;
    }
    for (const unsigned char character : text) {
      if (character < 0x20U || character == 0x7fU || character == '\0' ||
          character == ':' || character == '\\' || character == '/') {
        return false;
      }
    }
  }
  return true;
}

bool isPrefix(const std::string& value, const char* prefix) {
  const std::string prefix_text(prefix);
  return value.size() > prefix_text.size() &&
         value.compare(0U, prefix_text.size(), prefix_text) == 0;
}

bool digestFile(const std::filesystem::path& path,
                std::uintmax_t& size,
                std::string& digest,
                std::string* error_message) {
  std::error_code size_error;
  size = std::filesystem::file_size(path, size_error);
  if (size_error) {
    return fail(error_message, "file_size failed for " + path.string() +
                                  ": " + size_error.message());
  }

  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    return fail(error_message, "cannot open artifact " + path.string());
  }

  EVP_MD_CTX* context = EVP_MD_CTX_new();
  if (context == nullptr || EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1) {
    if (context != nullptr) {
      EVP_MD_CTX_free(context);
    }
    return fail(error_message, "cannot initialize SHA-256");
  }

  std::array<char, 64U * 1024U> buffer{};
  bool ok = true;
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 && EVP_DigestUpdate(
                         context, buffer.data(),
                         static_cast<std::size_t>(count)) != 1) {
      ok = false;
      break;
    }
  }
  if (input.bad()) {
    ok = false;
  }

  std::array<unsigned char, EVP_MAX_MD_SIZE> raw_digest{};
  unsigned int digest_size = 0U;
  if (ok && EVP_DigestFinal_ex(
                context, raw_digest.data(), &digest_size) != 1) {
    ok = false;
  }
  EVP_MD_CTX_free(context);
  if (!ok || digest_size != 32U) {
    return fail(error_message, "SHA-256 failed for " + path.string());
  }

  std::ostringstream encoded;
  encoded << std::hex;
  for (unsigned int index = 0U; index < digest_size; ++index) {
    encoded.width(2);
    encoded.fill('0');
    encoded << static_cast<unsigned int>(raw_digest[index]);
  }
  digest = encoded.str();
  return true;
}

bool parseSidecar(const std::filesystem::path& sidecar_path,
                  MapArtifactManifest& manifest,
                  std::string* error_message) {
  std::ifstream input(sidecar_path, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    return fail(error_message, "cannot open map sidecar " +
                                  sidecar_path.string());
  }

  std::map<std::string, std::string> fields;
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t comment = line.find('#');
    if (comment != std::string::npos) {
      line.erase(comment);
    }
    line = trim(std::move(line));
    if (line.empty()) {
      continue;
    }
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
      return fail(error_message, "malformed map sidecar line");
    }
    const std::string key = trim(line.substr(0U, colon));
    const std::string value = trim(line.substr(colon + 1U));
    if (key.empty() || fields.find(key) != fields.end()) {
      return fail(error_message, "duplicate or empty sidecar key");
    }
    fields.emplace(key, value);
  }
  if (input.bad()) {
    return fail(error_message, "cannot read map sidecar " +
                                  sidecar_path.string());
  }

  const auto get = [&fields](const std::string& key) -> const std::string* {
    const auto iterator = fields.find(key);
    return iterator == fields.end() ? nullptr : &iterator->second;
  };
  const std::string* schema = get("schema");
  const std::string* schema_version = get("schema_version");
  const std::string* generation = get("generation");
  const std::string* artifact_count = get("artifact_count");
  const std::string* succeeded = get("succeeded");
  if (schema == nullptr || schema_version == nullptr || generation == nullptr ||
      artifact_count == nullptr || succeeded == nullptr ||
      trim(*schema) != "map_leveling") {
    return fail(error_message, "map sidecar is missing v3 identity fields");
  }

  std::uintmax_t parsed_version = 0U;
  std::uintmax_t parsed_count = 0U;
  bool parsed_succeeded = false;
  if (!parseUnsigned(*schema_version, parsed_version) || parsed_version != 3U ||
      !parseUnsigned(*artifact_count, parsed_count) || parsed_count == 0U ||
      parsed_count > kMaxManifestArtifacts ||
      !parseBool(*succeeded, parsed_succeeded) || !parsed_succeeded ||
      !validGeneration(trim(*generation))) {
    return fail(error_message, "map sidecar has invalid v3 identity fields");
  }

  manifest.generation = trim(*generation);
  manifest.artifacts.clear();
  manifest.artifacts.reserve(static_cast<std::size_t>(parsed_count));
  for (std::uintmax_t index = 0U; index < parsed_count; ++index) {
    const std::string index_text = std::to_string(index);
    const std::string path_key = "artifact_" + index_text + "_path";
    const std::string size_key = "artifact_" + index_text + "_size";
    const std::string hash_key = "artifact_" + index_text + "_sha256";
    const std::string* path_value = get(path_key);
    const std::string* size_value = get(size_key);
    const std::string* hash_value = get(hash_key);
    std::uintmax_t parsed_size = 0U;
    if (path_value == nullptr || size_value == nullptr || hash_value == nullptr ||
        !parseUnsigned(*size_value, parsed_size)) {
      return fail(error_message, "map sidecar has an incomplete artifact entry");
    }
    MapArtifactManifestEntry entry;
    entry.relative_path = std::filesystem::path(trim(*path_value));
    entry.size = parsed_size;
    entry.sha256 = trim(*hash_value);
    if (!safeRelativePath(entry.relative_path) ||
        entry.relative_path == kSidecarName || !validSha256(entry.sha256)) {
      return fail(error_message, "map sidecar has an unsafe artifact entry");
    }
    manifest.artifacts.push_back(std::move(entry));
  }

  // Reject unbound artifact_N_* keys and duplicate paths.  This prevents a
  // producer typo from silently creating an artifact outside artifact_count.
  std::vector<std::filesystem::path> paths;
  paths.reserve(manifest.artifacts.size());
  for (const auto& field : fields) {
    if (field.first == "artifact_count" ||
        field.first.compare(0U, 9U, "artifact_") != 0) {
      continue;
    }
    const std::size_t separator = field.first.find('_', 9U);
    if (separator == std::string::npos || separator == 9U) {
      return fail(error_message, "malformed artifact key in map sidecar");
    }
    const std::string index_text = field.first.substr(9U, separator - 9U);
    const std::string suffix = field.first.substr(separator + 1U);
    if (index_text.empty() ||
        !std::all_of(index_text.begin(), index_text.end(), [](unsigned char c) {
          return std::isdigit(c) != 0;
        })) {
      return fail(error_message, "malformed artifact index in map sidecar");
    }
    std::uintmax_t index = 0U;
    if (!parseUnsigned(index_text, index) || index >= parsed_count ||
        (suffix != "path" && suffix != "size" && suffix != "sha256")) {
      return fail(error_message, "unbound artifact key in map sidecar");
    }
  }
  for (const auto& artifact : manifest.artifacts) {
    paths.push_back(artifact.relative_path);
  }
  std::sort(paths.begin(), paths.end());
  if (std::adjacent_find(paths.begin(), paths.end()) != paths.end()) {
    return fail(error_message, "duplicate artifact path in map sidecar");
  }

  bool has_pcd = false;
  bool has_trajectory = false;
  bool has_pgm = false;
  bool has_grid_yaml = false;
  for (const auto& artifact : manifest.artifacts) {
    const std::string name = artifact.relative_path.generic_string();
    has_pcd = has_pcd || name == "map.pcd";
    has_trajectory = has_trajectory || name == "map.txt";
    has_pgm = has_pgm ||
              (name.size() > 4U && name.compare(name.size() - 4U, 4U, ".pgm") == 0);
    has_grid_yaml = has_grid_yaml ||
                    (name.size() > 5U &&
                     name.compare(name.size() - 5U, 5U, ".yaml") == 0);
  }
  if (!has_pcd || !has_trajectory || !has_pgm || !has_grid_yaml) {
    return fail(error_message, "map sidecar is missing a canonical map artifact");
  }
  return true;
}

bool validateArtifactSet(const std::filesystem::path& root,
                         const std::vector<MapArtifactManifestEntry>& expected,
                         std::string* error_message) {
  std::vector<MapArtifactManifestEntry> actual;
  std::error_code iterate_error;
  for (std::filesystem::recursive_directory_iterator iterator(
           root, std::filesystem::directory_options::none, iterate_error),
       end;
       iterator != end; iterator.increment(iterate_error)) {
    if (iterate_error) {
      return fail(error_message, "artifact traversal failed: " +
                                    iterate_error.message());
    }
    const auto& entry = *iterator;
    std::error_code status_error;
    const auto status = entry.symlink_status(status_error);
    if (status_error) {
      return fail(error_message, "artifact status failed for " +
                                    entry.path().string() + ": " +
                                    status_error.message());
    }
    const std::string name = entry.path().filename().string();
    if (std::filesystem::is_symlink(status)) {
      return fail(error_message, "map artifact is a symbolic link: " +
                                    entry.path().string());
    }
    if (std::filesystem::is_directory(status)) {
      if (isPrefix(name, kStagingPrefix) || isPrefix(name, kInvalidPrefix)) {
        return fail(error_message, "unfinished map transaction remains: " +
                                      entry.path().string());
      }
      if (isPrefix(name, kBackupPrefix)) {
        iterator.disable_recursion_pending();
      }
      continue;
    }
    if (!std::filesystem::is_regular_file(status)) {
      return fail(error_message, "map artifact is not a regular file: " +
                                    entry.path().string());
    }
    if (isPrefix(name, kStagingPrefix) || isPrefix(name, kInvalidPrefix)) {
      return fail(error_message, "unfinished map transaction remains: " +
                                    entry.path().string());
    }
    const std::filesystem::path relative =
        entry.path().lexically_relative(root).lexically_normal();
    if (!safeRelativePath(relative)) {
      return fail(error_message, "unsafe artifact path: " +
                                    entry.path().string());
    }
    if (relative == std::filesystem::path(kSidecarName)) {
      continue;
    }
    MapArtifactManifestEntry digest;
    digest.relative_path = relative;
    if (!digestFile(entry.path(), digest.size, digest.sha256, error_message)) {
      return false;
    }
    actual.push_back(std::move(digest));
  }
  if (iterate_error) {
    return fail(error_message, "artifact traversal failed: " +
                                  iterate_error.message());
  }

  auto sort_entries = [](std::vector<MapArtifactManifestEntry>& entries) {
    std::sort(entries.begin(), entries.end(),
              [](const auto& lhs, const auto& rhs) {
                return lhs.relative_path.generic_string() <
                       rhs.relative_path.generic_string();
              });
  };
  std::vector<MapArtifactManifestEntry> sorted_expected = expected;
  sort_entries(sorted_expected);
  sort_entries(actual);
  if (sorted_expected.size() != actual.size()) {
    return fail(error_message, "artifact count does not match map sidecar");
  }
  for (std::size_t index = 0U; index < sorted_expected.size(); ++index) {
    const auto& wanted = sorted_expected[index];
    const auto& observed = actual[index];
    if (wanted.relative_path != observed.relative_path ||
        wanted.size != observed.size || wanted.sha256 != observed.sha256) {
      return fail(error_message, "artifact digest mismatch at " +
                                    wanted.relative_path.string());
    }
  }
  return true;
}

}  // namespace

const MapArtifactManifestEntry* MapArtifactManifest::find(
    const std::filesystem::path& relative_path) const noexcept {
  for (const auto& artifact : artifacts) {
    if (artifact.relative_path == relative_path) {
      return &artifact;
    }
  }
  return nullptr;
}

bool MapArtifactManifest::contains(
    const std::filesystem::path& relative_path) const noexcept {
  return find(relative_path) != nullptr;
}

bool validateMapArtifactManifest(const std::filesystem::path& map_path,
                                 MapArtifactManifest& manifest,
                                 std::string* error_message) {
  manifest = MapArtifactManifest{};
  if (map_path.empty()) {
    return fail(error_message, "PCD map path is empty");
  }

  std::error_code path_error;
  const auto input_status = std::filesystem::symlink_status(map_path, path_error);
  if (path_error || std::filesystem::is_symlink(input_status) ||
      !std::filesystem::is_regular_file(input_status)) {
    return fail(error_message, "PCD map path is not a regular non-symlink file: " +
                                  map_path.string());
  }
  const std::filesystem::path absolute_map =
      std::filesystem::absolute(map_path, path_error).lexically_normal();
  if (path_error || absolute_map.empty()) {
    return fail(error_message, "cannot resolve PCD map path: " + map_path.string());
  }
  const std::filesystem::path root = absolute_map.parent_path();
  const auto root_status = std::filesystem::symlink_status(root, path_error);
  if (path_error || std::filesystem::is_symlink(root_status) ||
      !std::filesystem::is_directory(root_status)) {
    return fail(error_message, "map artifact root is not a regular directory: " +
                                  root.string());
  }

  const std::filesystem::path sidecar = root / kSidecarName;
  const auto sidecar_status = std::filesystem::symlink_status(sidecar, path_error);
  if (path_error || std::filesystem::is_symlink(sidecar_status) ||
      !std::filesystem::is_regular_file(sidecar_status)) {
    return fail(error_message, "map sidecar is missing or is not a regular file: " +
                                  sidecar.string());
  }

  MapArtifactManifest parsed;
  if (!parseSidecar(sidecar, parsed, error_message)) {
    return false;
  }
  const std::filesystem::path requested_relative = absolute_map.filename();
  if (!safeRelativePath(requested_relative) || requested_relative != "map.pcd" ||
      !parsed.contains(requested_relative)) {
    return fail(error_message, "requested PCD is not the manifest's map.pcd artifact");
  }
  if (!validateArtifactSet(root, parsed.artifacts, error_message)) {
    return false;
  }

  parsed.artifact_root = root;
  manifest = std::move(parsed);
  return true;
}

}  // namespace localization
