#include "map_leveling_sidecar.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

namespace
{

constexpr double kComparisonTolerance = 1e-8;
constexpr double kRotationTolerance = 1e-4;
constexpr double kNormalTolerance = 1e-4;
constexpr double kMaxSampleQuantile = 0.20;
constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
constexpr std::size_t kMaxManifestArtifacts = 64U;

std::string trimYamlScalar(std::string value)
{
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return {};
    }
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1U);
}

bool parseYamlBoolScalar(const std::string& value, bool& output)
{
    std::string normalized = trimYamlScalar(value);
    std::transform(
        normalized.begin(), normalized.end(), normalized.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    if (normalized == "true" || normalized == "yes" || normalized == "1")
    {
        output = true;
        return true;
    }
    if (normalized == "false" || normalized == "no" || normalized == "0")
    {
        output = false;
        return true;
    }
    return false;
}

bool parseYamlDoubleScalar(const std::string& value, double& output)
{
    const std::string normalized = trimYamlScalar(value);
    if (normalized.empty())
    {
        return false;
    }
    try
    {
        std::size_t consumed = 0U;
        const double parsed = std::stod(normalized, &consumed);
        if (consumed != normalized.size() || !std::isfinite(parsed))
        {
            return false;
        }
        output = parsed;
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool parseYamlUnsignedScalar(const std::string& value, std::size_t& output)
{
    const std::string normalized = trimYamlScalar(value);
    if (normalized.empty() || normalized.front() == '-')
    {
        return false;
    }
    try
    {
        std::size_t consumed = 0U;
        const unsigned long long parsed = std::stoull(normalized, &consumed);
        if (consumed != normalized.size() ||
            parsed > static_cast<unsigned long long>(
                std::numeric_limits<std::size_t>::max()))
        {
            return false;
        }
        output = static_cast<std::size_t>(parsed);
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool parseYamlNumberList(
    const std::string& value, std::size_t expected_count,
    std::vector<double>& output)
{
    std::string normalized = trimYamlScalar(value);
    if (normalized.size() < 2U || normalized.front() != '[' ||
        normalized.back() != ']')
    {
        return false;
    }
    normalized = normalized.substr(1U, normalized.size() - 2U);
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream stream(normalized);
    output.clear();
    std::string token;
    while (stream >> token)
    {
        double parsed = 0.0;
        if (!parseYamlDoubleScalar(token, parsed))
        {
            return false;
        }
        output.push_back(parsed);
    }
    return output.size() == expected_count;
}

bool nearlyEqual(double lhs, double rhs, double tolerance = kComparisonTolerance)
{
    return std::abs(lhs - rhs) <=
        tolerance * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

double determinant3x3(const Eigen::Matrix3d& matrix)
{
    // Keep this validation path independent of Eigen's optional LU
    // out-of-line instantiation, which is not linked consistently across the
    // ROS images used by the board and the host.
    return matrix(0, 0) *
            (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) -
        matrix(0, 1) *
            (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
        matrix(0, 2) *
            (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
}

Eigen::Vector3d canonicalRpy(const Eigen::Matrix3d& rotation)
{
    const double pitch = std::atan2(
        -rotation(2, 0), std::hypot(rotation(0, 0), rotation(1, 0)));
    const double roll = std::atan2(rotation(2, 1), rotation(2, 2));
    const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    return Eigen::Vector3d(roll, pitch, yaw);
}

bool validRotation(const Eigen::Matrix3d& rotation)
{
    return rotation.allFinite() &&
        (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                .cwiseAbs()
                .maxCoeff() <= kRotationTolerance &&
        std::abs(determinant3x3(rotation) - 1.0) <= kRotationTolerance;
}

struct FitMetrics
{
    std::size_t candidate_points = 0U;
    std::size_t sample_cells = 0U;
    std::size_t inlier_cells = 0U;
    double inlier_ratio = 0.0;
    double residual_p95 = 0.0;
    double tilt_deg = 0.0;
    double major_span = 0.0;
    double minor_span = 0.0;
};

bool validFiniteNonnegative(double value)
{
    return std::isfinite(value) && value >= 0.0;
}

bool validEnabledFitMetrics(const FitMetrics& metrics, bool require_spans)
{
    if (metrics.inlier_cells == 0U ||
        metrics.inlier_cells > metrics.sample_cells ||
        metrics.sample_cells > metrics.candidate_points ||
        metrics.inlier_ratio <= 0.0 || metrics.inlier_ratio > 1.0 ||
        !validFiniteNonnegative(metrics.residual_p95) ||
        !validFiniteNonnegative(metrics.tilt_deg) ||
        !validFiniteNonnegative(metrics.major_span) ||
        !validFiniteNonnegative(metrics.minor_span))
    {
        return false;
    }
    if (require_spans &&
        (metrics.major_span <= 0.0 || metrics.minor_span <= 0.0))
    {
        return false;
    }
    const double count_ratio = static_cast<double>(metrics.inlier_cells) /
        static_cast<double>(metrics.sample_cells);
    return nearlyEqual(metrics.inlier_ratio, count_ratio);
}

bool zeroFitMetrics(const FitMetrics& metrics)
{
    return metrics.candidate_points == 0U && metrics.sample_cells == 0U &&
        metrics.inlier_cells == 0U && nearlyEqual(metrics.inlier_ratio, 0.0) &&
        nearlyEqual(metrics.residual_p95, 0.0) &&
        nearlyEqual(metrics.tilt_deg, 0.0) &&
        nearlyEqual(metrics.major_span, 0.0) &&
        nearlyEqual(metrics.minor_span, 0.0);
}

bool validGeneration(const std::string& generation)
{
    if (generation.empty() || generation.size() > 128U)
    {
        return false;
    }
    return std::all_of(generation.begin(), generation.end(),
                       [](unsigned char character) {
                           return std::isalnum(character) != 0 ||
                               character == '-' || character == '_' ||
                               character == '.';
                       });
}

bool validManifestShape(const std::vector<anubis_mapping::MapArtifactDigest>& artifacts)
{
    if (artifacts.empty() || artifacts.size() > kMaxManifestArtifacts)
    {
        return false;
    }
    bool has_pcd = false;
    bool has_txt = false;
    bool has_pgm = false;
    bool has_yaml = false;
    for (const auto& artifact : artifacts)
    {
        if (!anubis_mapping::isSafeMapArtifactRelativePath(artifact.relative_path) ||
            artifact.relative_path == "map_leveling.yaml" ||
            artifact.sha256.size() != 64U)
        {
            return false;
        }
        for (const unsigned char character : artifact.sha256)
        {
            if (std::isxdigit(character) == 0)
            {
                return false;
            }
        }
        const std::string name = artifact.relative_path.generic_string();
        has_pcd = has_pcd || name == "map.pcd";
        has_txt = has_txt || name == "map.txt";
        has_pgm = has_pgm ||
            (name.size() > 4U && name.substr(name.size() - 4U) == ".pgm");
        has_yaml = has_yaml ||
            (name.size() > 5U && name.substr(name.size() - 5U) == ".yaml");
    }
    if (!has_pcd || !has_txt || !has_pgm || !has_yaml)
    {
        return false;
    }
    std::vector<std::filesystem::path> paths;
    paths.reserve(artifacts.size());
    for (const auto& artifact : artifacts)
    {
        paths.push_back(artifact.relative_path);
    }
    std::sort(paths.begin(), paths.end());
    return std::adjacent_find(paths.begin(), paths.end()) == paths.end();
}

}  // namespace

namespace anubis_mapping
{

bool parseValidatedMapLevelingFloor(
    std::istream& input, double& floor_z_map,
    MapLevelingMetadata* metadata,
    const std::filesystem::path* artifact_root)
{
    if (metadata != nullptr)
    {
        *metadata = MapLevelingMetadata{};
    }
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(input, line))
    {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos)
        {
            line.erase(comment);
        }
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos)
        {
            continue;
        }
        const std::string key = trimYamlScalar(line.substr(0, colon));
        const std::string value = trimYamlScalar(line.substr(colon + 1U));
        if (key.empty() || fields.find(key) != fields.end())
        {
            return false;
        }
        fields.emplace(key, value);
    }
    if (input.bad())
    {
        return false;
    }

    const auto get = [&](const char* key) -> const std::string*
    {
        const auto iterator = fields.find(key);
        return iterator == fields.end() ? nullptr : &iterator->second;
    };
    const std::array<const char*, 28> required_fields = {
        "schema", "schema_version", "enabled", "succeeded",
        "correction_rpy_rad", "correction_rpy_deg", "correction_rotation",
        "floor_z_map", "configured_sample_quantile",
        "selected_sample_quantile", "fallback_sample_quantile",
        "ground_normal", "inlier_ratio", "final_tilt_deg", "iterations",
        "q10_attempted", "q10_accepted", "seed_candidate_points",
        "seed_sample_cells", "seed_inlier_cells", "seed_inlier_ratio",
        "seed_residual_p95", "seed_tilt_deg", "refit_candidate_points",
        "refit_sample_cells", "refit_inlier_cells", "refit_inlier_ratio",
        "refit_residual_p95"};
    for (const char* key : required_fields)
    {
        if (get(key) == nullptr)
        {
            return false;
        }
    }
    for (const char* key : {"refit_major_span", "refit_minor_span",
                            "failure_reason"})
    {
        if (get(key) == nullptr)
        {
            return false;
        }
    }
    if (trimYamlScalar(*get("schema")) != "map_leveling")
    {
        return false;
    }

    std::size_t schema_version = 0U;
    bool enabled = false;
    bool succeeded = false;
    // Legacy v2 sidecars did not carry this field and retain the historical
    // "succeeded implies applied" interpretation.  Schema v3 makes the field
    // mandatory so a transaction cannot silently claim that identity geometry
    // was corrected.
    bool leveling_applied = true;
    bool q10_attempted = false;
    bool q10_accepted = false;
    double parsed_floor = 0.0;
    double configured_quantile = 0.0;
    double selected_quantile = 0.0;
    double fallback_quantile = 0.0;
    double inlier_ratio = 0.0;
    double final_tilt_deg = 0.0;
    std::size_t iterations = 0U;
    std::vector<double> rotation_values;
    std::vector<double> rpy_rad_values;
    std::vector<double> rpy_deg_values;
    std::vector<double> normal_values;
    std::string generation;
    std::size_t artifact_count = 0U;
    std::vector<MapArtifactDigest> artifacts;
    FitMetrics seed;
    FitMetrics refit;
    if (!parseYamlUnsignedScalar(*get("schema_version"), schema_version) ||
        (schema_version != 2U && schema_version != 3U) ||
        !parseYamlBoolScalar(*get("enabled"), enabled) ||
        !parseYamlBoolScalar(*get("succeeded"), succeeded) || !succeeded ||
        (schema_version >= 3U && get("leveling_applied") == nullptr) ||
        (get("leveling_applied") != nullptr &&
            !parseYamlBoolScalar(*get("leveling_applied"), leveling_applied)) ||
        !parseYamlBoolScalar(*get("q10_attempted"), q10_attempted) ||
        !parseYamlBoolScalar(*get("q10_accepted"), q10_accepted) ||
        !parseYamlDoubleScalar(*get("floor_z_map"), parsed_floor) ||
        !parseYamlDoubleScalar(
            *get("configured_sample_quantile"), configured_quantile) ||
        !parseYamlDoubleScalar(
            *get("selected_sample_quantile"), selected_quantile) ||
        !parseYamlDoubleScalar(
            *get("fallback_sample_quantile"), fallback_quantile) ||
        !parseYamlDoubleScalar(*get("inlier_ratio"), inlier_ratio) ||
        !parseYamlDoubleScalar(*get("final_tilt_deg"), final_tilt_deg) ||
        !parseYamlUnsignedScalar(*get("iterations"), iterations) ||
        !parseYamlNumberList(
            *get("correction_rotation"), 9U, rotation_values) ||
        !parseYamlNumberList(
            *get("correction_rpy_rad"), 3U, rpy_rad_values) ||
        !parseYamlNumberList(
            *get("correction_rpy_deg"), 3U, rpy_deg_values) ||
        !parseYamlNumberList(*get("ground_normal"), 3U, normal_values) ||
        !parseYamlUnsignedScalar(
            *get("seed_candidate_points"), seed.candidate_points) ||
        !parseYamlUnsignedScalar(
            *get("seed_sample_cells"), seed.sample_cells) ||
        !parseYamlUnsignedScalar(
            *get("seed_inlier_cells"), seed.inlier_cells) ||
        !parseYamlDoubleScalar(
            *get("seed_inlier_ratio"), seed.inlier_ratio) ||
        !parseYamlDoubleScalar(
            *get("seed_residual_p95"), seed.residual_p95) ||
        !parseYamlDoubleScalar(*get("seed_tilt_deg"), seed.tilt_deg) ||
        !parseYamlUnsignedScalar(
            *get("refit_candidate_points"), refit.candidate_points) ||
        !parseYamlUnsignedScalar(
            *get("refit_sample_cells"), refit.sample_cells) ||
        !parseYamlUnsignedScalar(
            *get("refit_inlier_cells"), refit.inlier_cells) ||
        !parseYamlDoubleScalar(
            *get("refit_inlier_ratio"), refit.inlier_ratio) ||
        !parseYamlDoubleScalar(
            *get("refit_residual_p95"), refit.residual_p95) ||
        !parseYamlDoubleScalar(
            *get("refit_major_span"), refit.major_span) ||
        !parseYamlDoubleScalar(
            *get("refit_minor_span"), refit.minor_span))
    {
        return false;
    }

    // A disabled v3 correction must never be represented as applied.  Keep
    // v2 permissive for backwards compatibility with already-exported maps.
    if (schema_version >= 3U && !enabled && leveling_applied)
    {
        return false;
    }

    if (schema_version >= 3U)
    {
        const std::string* generation_value = get("generation");
        const std::string* artifact_count_value = get("artifact_count");
        if (generation_value == nullptr || artifact_count_value == nullptr ||
            !validGeneration(generation = trimYamlScalar(*generation_value)) ||
            !parseYamlUnsignedScalar(*artifact_count_value, artifact_count) ||
            artifact_count == 0U || artifact_count > kMaxManifestArtifacts)
        {
            return false;
        }
        artifacts.reserve(artifact_count);
        for (std::size_t index = 0U; index < artifact_count; ++index)
        {
            const std::string index_text = std::to_string(index);
            const std::string path_key = "artifact_" + index_text + "_path";
            const std::string size_key = "artifact_" + index_text + "_size";
            const std::string hash_key = "artifact_" + index_text + "_sha256";
            const std::string* path_value = get(path_key.c_str());
            const std::string* size_value = get(size_key.c_str());
            const std::string* hash_value = get(hash_key.c_str());
            std::size_t parsed_size = 0U;
            if (path_value == nullptr || size_value == nullptr ||
                hash_value == nullptr ||
                !parseYamlUnsignedScalar(*size_value, parsed_size))
            {
                return false;
            }
            MapArtifactDigest artifact;
            artifact.relative_path = trimYamlScalar(*path_value);
            artifact.size = static_cast<std::uintmax_t>(parsed_size);
            artifact.sha256 = trimYamlScalar(*hash_value);
            artifacts.push_back(std::move(artifact));
        }
        if (!validManifestShape(artifacts))
        {
            return false;
        }
    }
    else if (artifact_root != nullptr)
    {
        // Legacy sidecars have no way to bind their floor to a concrete
        // artifact generation.  They may still be inspected for diagnostics,
        // but never authorize a production load with an artifact root.
        return false;
    }

    const std::string failure_reason =
        trimYamlScalar(*get("failure_reason"));
    for (const unsigned char character : failure_reason)
    {
        if (character < 0x20U || character == 0x7fU)
        {
            return false;
        }
    }
    if (!std::isfinite(parsed_floor) || configured_quantile <= 0.0 ||
        configured_quantile >= fallback_quantile ||
        fallback_quantile > kMaxSampleQuantile ||
        selected_quantile <= 0.0 ||
        !validFiniteNonnegative(inlier_ratio) || inlier_ratio > 1.0 ||
        !validFiniteNonnegative(final_tilt_deg) ||
        !validFiniteNonnegative(seed.inlier_ratio) || seed.inlier_ratio > 1.0 ||
        !validFiniteNonnegative(seed.residual_p95) ||
        !validFiniteNonnegative(seed.tilt_deg) ||
        !validFiniteNonnegative(refit.inlier_ratio) || refit.inlier_ratio > 1.0 ||
        !validFiniteNonnegative(refit.residual_p95) ||
        !validFiniteNonnegative(refit.major_span) ||
        !validFiniteNonnegative(refit.minor_span))
    {
        return false;
    }

    Eigen::Matrix3d rotation;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            rotation(row, column) = rotation_values[
                static_cast<std::size_t>(row * 3 + column)];
        }
    }
    const Eigen::Vector3d rpy_rad(
        rpy_rad_values[0], rpy_rad_values[1], rpy_rad_values[2]);
    const Eigen::Vector3d rpy_deg(
        rpy_deg_values[0], rpy_deg_values[1], rpy_deg_values[2]);
    const Eigen::Vector3d normal(
        normal_values[0], normal_values[1], normal_values[2]);
    if (!validRotation(rotation) || !rpy_rad.allFinite() ||
        !rpy_deg.allFinite() || !normal.allFinite() || normal.z() <= 0.0 ||
        std::abs(normal.norm() - 1.0) > kNormalTolerance)
    {
        return false;
    }
    const Eigen::Vector3d expected_rpy = canonicalRpy(rotation);
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!nearlyEqual(rpy_rad(axis), expected_rpy(axis)) ||
            !nearlyEqual(
                rpy_deg(axis), rpy_rad(axis) * kRadiansToDegrees, 1e-6))
        {
            return false;
        }
    }
    const double expected_tilt = std::atan2(
        std::hypot(normal.x(), normal.y()), normal.z()) * kRadiansToDegrees;
    if (!nearlyEqual(final_tilt_deg, expected_tilt, 1e-4))
    {
        return false;
    }

    const auto commit_metadata = [&]()
    {
        if (metadata != nullptr)
        {
            metadata->schema_version = schema_version;
            metadata->generation = generation;
            metadata->artifacts = artifacts;
        }
    };

    if (schema_version >= 3U && artifact_root != nullptr)
    {
        std::string manifest_error;
        if (!validateMapArtifactDigests(
                *artifact_root, artifacts, "map_leveling.yaml", &manifest_error))
        {
            return false;
        }
    }

    if (!enabled)
    {
        // A disabled correction is a deliberate identity-frame save.  Keep
        // this branch before the generic "unapplied" branch so that a v3
        // sidecar cannot hide a disabled configuration behind an arbitrary
        // non-empty failure reason.
        if (iterations != 0U || q10_attempted || q10_accepted ||
            !nearlyEqual(selected_quantile, configured_quantile) ||
            !rotation.isApprox(Eigen::Matrix3d::Identity(), kComparisonTolerance) ||
            !rpy_rad.isZero(kComparisonTolerance) ||
            !rpy_deg.isZero(kComparisonTolerance) ||
            !normal.isApprox(Eigen::Vector3d::UnitZ(), kComparisonTolerance) ||
            !nearlyEqual(inlier_ratio, 0.0) ||
            !nearlyEqual(final_tilt_deg, 0.0) || !zeroFitMetrics(seed) ||
            !zeroFitMetrics(refit) ||
            failure_reason != "disabled by configuration")
        {
            return false;
        }
        floor_z_map = parsed_floor;
        commit_metadata();
        return true;
    }

    if (!leveling_applied)
    {
        // An enabled map may still be saved without the correction when
        // enforce_quality_gates=false.  It must describe the original
        // geometry and retain a reason, while quantile/fit diagnostics are
        // reset to the identity/unapplied state emitted by the writer.
        if (iterations != 0U || q10_attempted || q10_accepted ||
            !nearlyEqual(selected_quantile, configured_quantile) ||
            !nearlyEqual(inlier_ratio, 0.0) ||
            !nearlyEqual(final_tilt_deg, 0.0) ||
            !rotation.isApprox(Eigen::Matrix3d::Identity(), kRotationTolerance) ||
            !rpy_rad.isZero(kComparisonTolerance) ||
            !rpy_deg.isZero(kComparisonTolerance) ||
            !normal.isApprox(Eigen::Vector3d::UnitZ(), kNormalTolerance) ||
            !zeroFitMetrics(seed) || !zeroFitMetrics(refit) ||
            failure_reason.empty())
        {
            return false;
        }
        floor_z_map = parsed_floor;
        commit_metadata();
        return true;
    }

    const bool primary_state = !q10_attempted && !q10_accepted &&
        nearlyEqual(selected_quantile, configured_quantile);
    const bool fallback_state = q10_attempted && q10_accepted &&
        nearlyEqual(selected_quantile, fallback_quantile);
    if ((!primary_state && !fallback_state) || iterations == 0U ||
        inlier_ratio <= 0.0 || seed.inlier_ratio <= 0.0 ||
        refit.inlier_ratio <= 0.0 || !validEnabledFitMetrics(seed, false) ||
        !validEnabledFitMetrics(refit, true) ||
        !nearlyEqual(inlier_ratio, refit.inlier_ratio) ||
        !failure_reason.empty())
    {
        return false;
    }

    floor_z_map = parsed_floor;
    commit_metadata();
    return true;
}

bool readValidatedMapLevelingFloor(
    const std::string& file_name, double& floor_z_map)
{
    std::ifstream input(file_name);
    return input.is_open() && parseValidatedMapLevelingFloor(input, floor_z_map);
}

bool readValidatedMapLevelingFloor(
    const std::string& file_name, double& floor_z_map,
    MapLevelingMetadata* metadata)
{
    std::ifstream input(file_name);
    return input.is_open() && parseValidatedMapLevelingFloor(
        input, floor_z_map, metadata);
}

bool readValidatedMapLevelingFloor(
    const std::string& file_name, double& floor_z_map,
    const std::filesystem::path& artifact_root, MapLevelingMetadata* metadata)
{
    std::ifstream input(file_name);
    return input.is_open() && parseValidatedMapLevelingFloor(
        input, floor_z_map, metadata, &artifact_root);
}

}  // namespace anubis_mapping
