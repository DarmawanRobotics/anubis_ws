/**
 * @file mapping_alg.cpp
 * @brief
 * @author Liuzhao Li (liliuzhao@jushenzhiren.com)
 * @version 1.0
 * @date 2025-07-31
 * @copyright Copyright (C) 2025 具身智人(北京)科技有限公司
 */

#include "mapping_alg.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <unordered_map>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include "map_artifact_digest.h"
#include "map_artifact_transaction.h"
#include "map_leveling_sidecar.h"
#include "scan_descriptor/point_adapter.hpp"
#include "pcd2grid_logic.h"
#include "lio_time_guard.h"

namespace
{
template <typename Function>
class ScopeGuard
{
public:
    explicit ScopeGuard(Function function) : function_(std::move(function)) {}
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ~ScopeGuard() noexcept
    {
        if (active_)
        {
            function_();
        }
    }

    void dismiss() noexcept { active_ = false; }

private:
    Function function_;
    bool     active_ = true;
};

template <typename Function>
ScopeGuard(Function) -> ScopeGuard<Function>;

struct CloudZStats
{
    std::size_t count        = 0;
    std::size_t sample_count = 0;
    double      min          = 0.0;
    double      p05          = 0.0;
    double      p50          = 0.0;
    double      p95          = 0.0;
    double      max          = 0.0;
};

struct ImuBatchStats
{
    std::size_t    count                = 0;
    std::size_t    nonpositive_dt_count = 0;
    Eigen::Vector3d mean_acc            = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_gyr            = Eigen::Vector3d::Zero();
    double          mean_acc_norm       = 0.0;
    double          std_acc_norm        = 0.0;
    double          mean_gyr_norm       = 0.0;
    double          max_gyr_norm        = 0.0;
    double          dt_mean             = 0.0;
    double          dt_min              = 0.0;
    double          dt_max              = 0.0;
};

template <typename PointT>
CloudZStats summarizeCloudZ(const pcl::PointCloud<PointT>& cloud)
{
    CloudZStats result;
    if (cloud.empty())
    {
        return result;
    }

    const std::size_t stride = std::max<std::size_t>(1U, cloud.size() / 4096U);
    std::vector<double> samples;
    samples.reserve(cloud.size() / stride + 1U);
    bool have_range = false;
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const double z = cloud.points[i].z;
        if (!std::isfinite(z))
        {
            continue;
        }
        if (!have_range)
        {
            result.min = z;
            result.max = z;
            have_range = true;
        }
        else
        {
            result.min = std::min(result.min, z);
            result.max = std::max(result.max, z);
        }
        ++result.count;
        if ((i % stride) == 0U)
        {
            samples.push_back(z);
        }
    }
    if (samples.empty())
    {
        return result;
    }

    std::sort(samples.begin(), samples.end());
    result.sample_count = samples.size();
    const auto quantile = [&samples](double q)
    {
        const std::size_t index = static_cast<std::size_t>(
            std::floor(q * static_cast<double>(samples.size() - 1U)));
        return samples[index];
    };
    result.p05 = quantile(0.05);
    result.p50 = quantile(0.50);
    result.p95 = quantile(0.95);
    return result;
}

struct PointOffsetStats
{
    std::size_t count                = 0;
    std::size_t nonfinite_count      = 0;
    std::size_t regression_count     = 0;
    double      first_ms             = 0.0;
    double      last_ms              = 0.0;
    double      min_ms               = 0.0;
    double      max_ms               = 0.0;
};

PointOffsetStats summarizePointOffsets(const anubis_mapping::PointCloudType& cloud)
{
    PointOffsetStats result;
    bool have_value = false;
    double previous = 0.0;
    for (const auto& point : cloud.points)
    {
        const double offset_ms = point.curvature;
        if (!std::isfinite(offset_ms))
        {
            ++result.nonfinite_count;
            continue;
        }
        if (!have_value)
        {
            result.first_ms = offset_ms;
            result.min_ms = offset_ms;
            result.max_ms = offset_ms;
            have_value = true;
        }
        else
        {
            if (offset_ms < previous)
            {
                ++result.regression_count;
            }
            result.min_ms = std::min(result.min_ms, offset_ms);
            result.max_ms = std::max(result.max_ms, offset_ms);
        }
        previous = offset_ms;
        result.last_ms = offset_ms;
        ++result.count;
    }
    return result;
}

ImuBatchStats summarizeImuBatch(const std::deque<ImuMessagePtr>& imu)
{
    ImuBatchStats result;
    result.count = imu.size();
    if (imu.empty())
    {
        return result;
    }

    double acc_norm_sum = 0.0;
    double acc_norm_sq_sum = 0.0;
    double gyr_norm_sum = 0.0;
    double dt_sum = 0.0;
    std::size_t dt_count = 0;
    for (std::size_t i = 0; i < imu.size(); ++i)
    {
        result.mean_acc += imu[i]->acc;
        result.mean_gyr += imu[i]->gyr;
        const double acc_norm = imu[i]->acc.norm();
        const double gyr_norm = imu[i]->gyr.norm();
        acc_norm_sum += acc_norm;
        acc_norm_sq_sum += acc_norm * acc_norm;
        gyr_norm_sum += gyr_norm;
        result.max_gyr_norm = std::max(result.max_gyr_norm, gyr_norm);
        if (i == 0U)
        {
            continue;
        }
        const double dt = imu[i]->timestamp - imu[i - 1U]->timestamp;
        if (dt_count == 0U)
        {
            result.dt_min = dt;
            result.dt_max = dt;
        }
        else
        {
            result.dt_min = std::min(result.dt_min, dt);
            result.dt_max = std::max(result.dt_max, dt);
        }
        dt_sum += dt;
        ++dt_count;
        if (dt <= 0.0)
        {
            ++result.nonpositive_dt_count;
        }
    }

    const double count = static_cast<double>(imu.size());
    result.mean_acc /= count;
    result.mean_gyr /= count;
    result.mean_acc_norm = acc_norm_sum / count;
    result.mean_gyr_norm = gyr_norm_sum / count;
    result.std_acc_norm = std::sqrt(std::max(
        0.0, acc_norm_sq_sum / count - result.mean_acc_norm * result.mean_acc_norm));
    if (dt_count > 0U)
    {
        result.dt_mean = dt_sum / static_cast<double>(dt_count);
    }
    return result;
}

double covarianceSigma(double variance)
{
    return std::sqrt(std::max(0.0, variance));
}

bool pointCloudHasFloat64Timestamp(const sensor_msgs::msg::PointCloud2& msg)
{
    for (const auto& field : msg.fields)
    {
        if (field.name == "timestamp" &&
            field.datatype == sensor_msgs::msg::PointField::FLOAT64 &&
            field.count >= 1 &&
            msg.point_step >= field.offset + sizeof(double))
        {
            return true;
        }
    }
    return false;
}

bool finitePointXyz(const anubis_mapping::PointType& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) &&
        std::isfinite(point.z);
}

std::uint64_t cloudFingerprint(const anubis_mapping::PointCloudType& cloud)
{
    constexpr std::uint64_t offset = 1469598103934665603ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = offset;
    const auto mix_u32 = [&](std::uint32_t value, std::uint64_t& target)
    {
        for (int byte = 0; byte < 4; ++byte)
        {
            target ^= static_cast<std::uint8_t>(value & 0xffU);
            target *= prime;
            value >>= 8U;
        }
    };
    for (const auto& point : cloud.points)
    {
        const float values[] = {
            point.x, point.y, point.z, point.intensity,
            point.normal_x, point.normal_y, point.normal_z, point.curvature};
        for (const float value : values)
        {
            std::uint32_t bits = 0U;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(bits));
            mix_u32(bits, hash);
        }
    }
    const std::uint64_t count = static_cast<std::uint64_t>(cloud.size());
    mix_u32(static_cast<std::uint32_t>(count & 0xffffffffULL), hash);
    mix_u32(static_cast<std::uint32_t>(count >> 32U), hash);
    return hash;
}

bool finiteRigidPose(const Eigen::Matrix4d& pose)
{
    if (!pose.allFinite() ||
        !pose.row(3).isApprox(
            Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-6))
    {
        return false;
    }
    const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
    return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
               .cwiseAbs()
               .maxCoeff() <= 1e-4 &&
        std::abs(rotation.determinant() - 1.0) <= 1e-4;
}

// Use the canonical ZYX branch for diagnostics. Eigen's generic XYZ
// decomposition can represent a small leveling rotation with an equivalent
// 180-degree roll/pitch pair, which is operationally misleading in logs and
// map_leveling.yaml even though the rotation matrix itself is correct.
Eigen::Vector3d canonicalRpy(const Eigen::Matrix3d& rotation)
{
    const double pitch = std::atan2(
        -rotation(2, 0),
        std::hypot(rotation(0, 0), rotation(1, 0)));
    const double roll = std::atan2(rotation(2, 1), rotation(2, 2));
    const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    return Eigen::Vector3d(roll, pitch, yaw);
}
}  // namespace

namespace anubis_mapping
{
    MappingAlg::MappingAlg(const rclcpp::NodeOptions& options)
        : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.world_points_en", true);
        this->declare_parameter<bool>("publish.body_points_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("mapping.imu_init_duration_s", 3.0);
        this->declare_parameter<double>("mapping.imu_init_max_acc_std_g", 0.05);
        this->declare_parameter<double>("mapping.imu_init_max_gyr_std_rad_s", 0.05);
        this->declare_parameter<double>("mapping.imu_init_max_mean_gyr_rad_s", 0.05);
        this->declare_parameter<double>("mapping.imu_init_max_acc_norm_error_g", 0.05);
        this->declare_parameter<vector<double>>(
            "mapping.init_ba", vector<double>{0.0, 0.0, 0.0});
        this->declare_parameter<vector<double>>(
            "mapping.init_bg", vector<double>{0.0, 0.0, 0.0});
        this->declare_parameter<vector<double>>(
            "mapping.init_gravity_correction_rpy", vector<double>{0.0, 0.0, 0.0});
        this->declare_parameter<bool>("diagnostics.lio_trace_enabled", false);
        this->declare_parameter<int>("diagnostics.lio_trace_every_n_scans", 1);
        this->declare_parameter<bool>("lio.time_guard.enabled", true);
        this->declare_parameter<double>("lio.time_guard.max_scan_gap", 0.25);
        this->declare_parameter<double>("lio.time_guard.max_imu_dt", 0.10);
        this->declare_parameter<double>("lio.time_guard.max_scan_overlap", 0.002);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        // Edge-aware residuals. Only meaningful when feature_extract_enable is
        // true, since the FeatureLabel in normal_x is produced there.
        this->declare_parameter<bool>("edge_constraint_enable", false);
        this->declare_parameter<double>("filter_size_edge", 0.1);
        this->declare_parameter<double>("edge_residual_weight", 0.5);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<vector<double>>(
            "mapping.extrinsic_T", vector<double>{-0.011, -0.02329, 0.04412});
        this->declare_parameter<vector<double>>(
            "mapping.extrinsic_R", vector<double>{
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

        this->declare_parameter<string>("pcd2pgm.file_name", "map");
        this->declare_parameter<double>("pcd2pgm.thre_z_min", 0.10);
        this->declare_parameter<double>("pcd2pgm.thre_z_max", 3.0);
        this->declare_parameter<int>("pcd2pgm.flag_pass_through", 0);
        this->declare_parameter<double>("pcd2pgm.save_voxel", 0.1);
        this->declare_parameter<double>("pcd2pgm.map_resolution", 0.05);
        this->declare_parameter<double>("pcd2pgm.thre_radius", 0.15);
        this->declare_parameter<int>("pcd2pgm.thres_point_count", 2);
        this->declare_parameter<int>("pcd2pgm.min_points_occupied", 2);
        this->declare_parameter<int>("pcd2pgm.min_obstacle_frames", 2);
        this->declare_parameter<int>("pcd2pgm.min_free_frames", 2);
        this->declare_parameter<double>(
            "pcd2pgm.free_to_obstacle_frame_ratio", 2.0);
        this->declare_parameter<double>("pcd2pgm.overhead_clearance_z", 0.55);
        this->declare_parameter<bool>("pcd2pgm.low_point_gate_enabled", true);
        this->declare_parameter<int>("pcd2pgm.min_low_points_occupied", 2);
        this->declare_parameter<double>("pcd2pgm.overhead_min_low_ratio", 0.25);
        this->declare_parameter<int>(
            "pcd2pgm.overhead_low_frame_persistence", 10);
        this->declare_parameter<int>(
            "pcd2pgm.overhead_min_ray_crossing_frames", 2);
        this->declare_parameter<double>(
            "pcd2pgm.overhead_min_ray_crossing_ratio", 0.10);
        this->declare_parameter<double>("pcd2pgm.grid_crop_margin", 3.0);
        this->declare_parameter<bool>("pcd2pgm.ground_free_enabled", true);
        this->declare_parameter<double>("pcd2pgm.ground_free_min_height", -0.15);
        this->declare_parameter<bool>("pcd2pgm.raytrace_free_enabled", true);
        this->declare_parameter<double>("pcd2pgm.raytrace_max_range", 10.0);
        this->declare_parameter<double>("pcd2pgm.low_obstacle_diagnostic_height", 0.25);
        this->declare_parameter<bool>(
            "pcd2pgm.cell_evidence_diagnostics_enabled", true);
        this->declare_parameter<bool>("pcd2pgm.ground_plane_enabled", true);
        // Independent from map_leveling.enforce_quality_gates: this controls
        // whether PGM generation may fall back to the fixed floor when its
        // adaptive ground model cannot be fitted.
        this->declare_parameter<bool>("pcd2pgm.ground_plane_required", true);
        this->declare_parameter<double>("pcd2pgm.ground_plane_candidate_min_height", -0.15);
        this->declare_parameter<double>("pcd2pgm.ground_plane_candidate_max_height", 0.30);
        this->declare_parameter<int>("pcd2pgm.ground_plane_min_sample_cells", 100);
        this->declare_parameter<double>("pcd2pgm.ground_plane_min_inlier_ratio", 0.60);
        this->declare_parameter<double>("pcd2pgm.ground_plane_max_tilt_deg", 3.0);
        this->declare_parameter<double>("pcd2pgm.ground_plane_max_residual_p95", 0.08);
        this->declare_parameter<double>("pcd2pgm.ground_plane_max_floor_offset", 0.25);
        this->declare_parameter<bool>("pcd2pgm.ground_plane_quadratic_enabled", true);
        this->declare_parameter<double>(
            "pcd2pgm.ground_plane_quadratic_max_local_tilt_deg", 3.5);
        this->declare_parameter<double>(
            "pcd2pgm.ground_plane_quadratic_max_curvature", 0.02);
        this->declare_parameter<bool>("map_leveling.enabled", true);
        this->declare_parameter<double>("map_leveling.max_correction_deg", 8.0);
        this->declare_parameter<double>("map_leveling.min_inlier_ratio", 0.60);
        this->declare_parameter<double>("map_leveling.sample_quantile", 0.05);
        this->declare_parameter<double>(
            "map_leveling.fallback_sample_quantile", 0.10);
        this->declare_parameter<double>("map_leveling.converged_tilt_deg", 0.05);
        this->declare_parameter<int>("map_leveling.max_iterations", 8);
        this->declare_parameter<bool>("map_leveling.z_correction.enabled", true);
        this->declare_parameter<double>(
            "map_leveling.z_correction.local_candidate_min_height", -0.65);
        this->declare_parameter<double>(
            "map_leveling.z_correction.local_candidate_max_height", 0.65);
        this->declare_parameter<int>(
            "map_leveling.z_correction.local_min_sample_cells", 15);
        this->declare_parameter<double>(
            "map_leveling.z_correction.local_max_tilt_deg", 3.0);
        this->declare_parameter<double>(
            "map_leveling.z_correction.local_max_residual_p95", 0.08);
        this->declare_parameter<double>(
            "map_leveling.z_correction.local_min_range", 0.5);
        this->declare_parameter<double>(
            "map_leveling.z_correction.local_max_range", 8.0);
        this->declare_parameter<double>(
            "map_leveling.z_correction.local_max_source_distance", 2.5);
        this->declare_parameter<int>(
            "map_leveling.z_correction.local_min_independent_sources", 2);
        this->declare_parameter<int>(
            "map_leveling.z_correction.local_min_source_sample_cells", 3);
        this->declare_parameter<int>(
            "map_leveling.z_correction.local_keyframe_window", 6);
        this->declare_parameter<double>(
            "map_leveling.z_correction.min_valid_fraction", 0.80);
        this->declare_parameter<double>(
            "map_leveling.z_correction.max_unobserved_gap_m", 2.0);
        this->declare_parameter<double>(
            "map_leveling.z_correction.outlier_threshold_m", 0.12);
        this->declare_parameter<int>(
            "map_leveling.z_correction.outlier_window", 3);
        this->declare_parameter<double>(
            "map_leveling.z_correction.smoothness", 0.03);
        this->declare_parameter<double>(
            "map_leveling.z_correction.max_abs_correction_m", 0.45);
        this->declare_parameter<double>(
            "map_leveling.z_correction.max_correction_slope", 0.25);
        this->declare_parameter<double>(
            "map_leveling.z_correction.max_corrected_floor_span_m", 0.04);
        this->declare_parameter<double>(
            "map_leveling.max_final_floor_span_m", 0.05);
        this->declare_parameter<string>("map_output_dir", "");
        this->declare_parameter<bool>("keyframe.enable", false);
        this->declare_parameter<double>("keyframe.trans_threshold", 0.5);
        this->declare_parameter<double>("keyframe.yaw_threshold_deg", 10.0);
        this->declare_parameter<double>("keyframe.store_leaf", 0.1);
        this->declare_parameter<double>("keyframe.ground_store_leaf", 0.025);
        this->declare_parameter<double>("keyframe.floor_z_map", -0.304);
        this->declare_parameter<vector<double>>("keyframe.lidar_to_base_T", vector<double>());
        this->declare_parameter<int>("keyframe.max_count", 5000);
        this->declare_parameter<int>("keyframe.descriptor.ring_count", 20);
        this->declare_parameter<int>("keyframe.descriptor.sector_count", 60);
        this->declare_parameter<double>("keyframe.descriptor.max_range", 10.0);
        this->declare_parameter<double>("keyframe.descriptor.min_range", 0.3);
        this->declare_parameter<double>("keyframe.descriptor.z_min", -0.2);
        this->declare_parameter<double>("keyframe.descriptor.z_max", 2.0);
        this->declare_parameter<bool>("loop.enable", false);
        this->declare_parameter<int>("loop.min_keyframe_gap", 60);
        this->declare_parameter<int>("loop.submap_half_window", 12);
        this->declare_parameter<int>("loop.descriptor_candidate_count", 5);
        this->declare_parameter<double>("loop.descriptor_max_distance", 0.30);
        this->declare_parameter<double>("loop.icp_fitness_threshold", 0.15);
        this->declare_parameter<double>("loop.icp_max_correspondence_distance", 2.0);
        this->declare_parameter<int>("loop.icp_max_iterations", 80);
        this->declare_parameter<double>("loop.max_odom_xy_distance", 2.0);
        this->declare_parameter<int>("loop.consistency_min_hits", 3);
        this->declare_parameter<int>("loop.consistency_max_keyframe_gap", 3);
        this->declare_parameter<double>(
            "loop.consistency_candidate_xy_tolerance", 1.0);
        this->declare_parameter<double>(
            "loop.max_constraint_xy_correction", 0.50);
        this->declare_parameter<double>(
            "loop.max_constraint_z_correction", 0.45);
        this->declare_parameter<double>(
            "loop.max_constraint_yaw_correction_deg", 5.0);
        this->declare_parameter<double>(
            "loop.max_optimized_xy_deviation", 0.50);
        this->declare_parameter<double>(
            "loop.max_optimized_z_deviation", 0.45);
        this->declare_parameter<double>(
            "loop.max_optimized_roll_pitch_deviation_deg", 7.0);
        this->declare_parameter<double>(
            "loop.max_optimized_yaw_deviation_deg", 5.0);
        this->declare_parameter<double>(
            "loop.sanity_path_reference_length_m", 10.0);
        this->declare_parameter<double>("loop.sanity_path_scale_max", 4.0);
        this->declare_parameter<double>("loop.odom_roll_pitch_sigma", 0.01);
        this->declare_parameter<double>("loop.odom_z_sigma", 0.05);
        this->declare_parameter<double>("loop.loop_roll_pitch_sigma", 0.05);
        this->declare_parameter<double>("loop.loop_z_sigma", 0.10);
        this->declare_parameter<double>("loop.odom_rotation_sigma", 0.02);
        this->declare_parameter<double>("loop.odom_translation_sigma", 0.05);
        this->declare_parameter<double>("loop.loop_rotation_sigma", 0.05);
        this->declare_parameter<double>("loop.loop_translation_sigma", 0.10);
        this->declare_parameter<double>("loop.huber_k", 1.345);

        this->get_parameter_or<string>("pcd2pgm.file_name", pcd2pgm_options_.file_name, "map");
        if (!is_safe_pcd2grid_basename(pcd2pgm_options_.file_name))
        {
            throw std::invalid_argument(
                "pcd2pgm.file_name must be a single relative basename without path separators");
        }
        this->get_parameter_or<double>("pcd2pgm.thre_z_min", pcd2pgm_options_.thre_z_min, 0.10);
        this->get_parameter_or<double>("pcd2pgm.thre_z_max", pcd2pgm_options_.thre_z_max, 3.0);
        this->get_parameter_or<int>("pcd2pgm.flag_pass_through", pcd2pgm_options_.flag_pass_through, 0);
        this->get_parameter_or<double>(
            "pcd2pgm.save_voxel", pcd2pgm_options_.save_voxel, 0.1);
        this->get_parameter_or<double>("pcd2pgm.map_resolution", pcd2pgm_options_.map_resolution, 0.05);
        this->get_parameter_or<double>("pcd2pgm.thre_radius", pcd2pgm_options_.thre_radius, 0.15);
        this->get_parameter_or<int>("pcd2pgm.thres_point_count", pcd2pgm_options_.thres_point_count, 2);
        this->get_parameter_or<int>(
            "pcd2pgm.min_points_occupied",
            pcd2pgm_options_.min_points_occupied, 2);
        this->get_parameter_or<int>(
            "pcd2pgm.min_obstacle_frames",
            pcd2pgm_options_.min_obstacle_frames, 2);
        this->get_parameter_or<int>(
            "pcd2pgm.min_free_frames", pcd2pgm_options_.min_free_frames, 2);
        this->get_parameter_or<double>(
            "pcd2pgm.free_to_obstacle_frame_ratio",
            pcd2pgm_options_.free_to_obstacle_frame_ratio, 2.0);
        this->get_parameter_or<double>("pcd2pgm.overhead_clearance_z", pcd2pgm_options_.overhead_clearance_z, 0.55);
        this->get_parameter_or<bool>(
            "pcd2pgm.low_point_gate_enabled",
            pcd2pgm_options_.low_point_gate_enabled, true);
        this->get_parameter_or<int>(
            "pcd2pgm.min_low_points_occupied",
            pcd2pgm_options_.min_low_points_occupied, 2);
        this->get_parameter_or<double>(
            "pcd2pgm.overhead_min_low_ratio",
            pcd2pgm_options_.overhead_min_low_ratio, 0.25);
        this->get_parameter_or<int>(
            "pcd2pgm.overhead_low_frame_persistence",
            pcd2pgm_options_.overhead_low_frame_persistence, 10);
        this->get_parameter_or<int>(
            "pcd2pgm.overhead_min_ray_crossing_frames",
            pcd2pgm_options_.overhead_min_ray_crossing_frames, 2);
        this->get_parameter_or<double>(
            "pcd2pgm.overhead_min_ray_crossing_ratio",
            pcd2pgm_options_.overhead_min_ray_crossing_ratio, 0.10);
        this->get_parameter_or<double>("pcd2pgm.grid_crop_margin", grid_crop_margin_, 3.0);
        this->get_parameter_or<bool>(
            "pcd2pgm.ground_free_enabled", pcd2pgm_options_.ground_free_enabled, true);
        this->get_parameter_or<double>(
            "pcd2pgm.ground_free_min_height",
            pcd2pgm_options_.ground_free_min_height, -0.15);
        this->get_parameter_or<bool>(
            "pcd2pgm.raytrace_free_enabled",
            pcd2pgm_options_.raytrace_free_enabled, true);
        this->get_parameter_or<double>(
            "pcd2pgm.raytrace_max_range",
            pcd2pgm_options_.raytrace_max_range, 10.0);
        this->get_parameter_or<double>(
            "pcd2pgm.low_obstacle_diagnostic_height",
            pcd2pgm_options_.low_obstacle_diagnostic_height, 0.25);
        this->get_parameter_or<bool>(
            "pcd2pgm.cell_evidence_diagnostics_enabled",
            pcd2pgm_options_.cell_evidence_diagnostics_enabled, true);
        this->get_parameter_or<bool>(
            "pcd2pgm.ground_plane_enabled",
            pcd2pgm_options_.ground_plane_enabled, true);
        this->get_parameter_or<bool>(
            "pcd2pgm.ground_plane_required",
            pcd2pgm_options_.ground_plane_required, true);
        this->get_parameter_or<double>(
            "pcd2pgm.ground_plane_candidate_min_height",
            pcd2pgm_options_.ground_plane_candidate_min_height, -0.15);
        this->get_parameter_or<double>(
            "pcd2pgm.ground_plane_candidate_max_height",
            pcd2pgm_options_.ground_plane_candidate_max_height, 0.30);
        this->get_parameter_or<int>(
            "pcd2pgm.ground_plane_min_sample_cells",
            pcd2pgm_options_.ground_plane_min_sample_cells, 100);
        this->get_parameter_or<double>(
            "pcd2pgm.ground_plane_min_inlier_ratio",
            pcd2pgm_options_.ground_plane_min_inlier_ratio, 0.60);
        this->get_parameter_or<double>(
            "pcd2pgm.ground_plane_max_tilt_deg",
            pcd2pgm_options_.ground_plane_max_tilt_deg, 3.0);
        this->get_parameter_or<double>(
            "pcd2pgm.ground_plane_max_residual_p95",
            pcd2pgm_options_.ground_plane_max_residual_p95, 0.08);
        this->get_parameter_or<double>(
            "pcd2pgm.ground_plane_max_floor_offset",
            pcd2pgm_options_.ground_plane_max_floor_offset, 0.25);
        this->get_parameter_or<bool>(
            "pcd2pgm.ground_plane_quadratic_enabled",
            pcd2pgm_options_.ground_plane_quadratic_enabled, true);
        this->get_parameter_or<double>(
            "pcd2pgm.ground_plane_quadratic_max_local_tilt_deg",
            pcd2pgm_options_.ground_plane_quadratic_max_local_tilt_deg, 3.5);
        this->get_parameter_or<double>(
            "pcd2pgm.ground_plane_quadratic_max_curvature",
            pcd2pgm_options_.ground_plane_quadratic_max_curvature, 0.02);
        this->get_parameter_or<bool>(
            "map_leveling.enabled", map_leveling_enabled_, true);
        this->get_parameter_or<bool>(
            "map_leveling.enforce_quality_gates",
            map_leveling_enforce_gates_, false);
        RCLCPP_INFO(
            get_logger(),
            "Map save gate contract: map_leveling.enforce_quality_gates=%s "
            "pcd2pgm.ground_plane_required=%s "
            "pcd2pgm.ground_plane_enabled=%s",
            map_leveling_enforce_gates_ ? "true" : "false",
            pcd2pgm_options_.ground_plane_required ? "true" : "false",
            pcd2pgm_options_.ground_plane_enabled ? "true" : "false");
        this->get_parameter_or<double>(
            "map_leveling.max_correction_deg",
            map_leveling_max_correction_deg_, 8.0);
        this->get_parameter_or<double>(
            "map_leveling.min_inlier_ratio",
            map_leveling_min_inlier_ratio_, 0.60);
        this->get_parameter_or<double>(
            "map_leveling.sample_quantile",
            map_leveling_sample_quantile_, 0.05);
        this->get_parameter_or<double>(
            "map_leveling.fallback_sample_quantile",
            map_leveling_fallback_sample_quantile_, 0.10);
        this->get_parameter_or<double>(
            "map_leveling.converged_tilt_deg",
            map_leveling_converged_tilt_deg_, 0.05);
        this->get_parameter_or<int>(
            "map_leveling.max_iterations", map_leveling_max_iterations_, 8);
        this->get_parameter_or<bool>(
            "map_leveling.z_correction.enabled",
            map_z_correction_enabled_, true);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.local_candidate_min_height",
            map_z_local_candidate_min_height_, -0.65);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.local_candidate_max_height",
            map_z_local_candidate_max_height_, 0.65);
        this->get_parameter_or<int>(
            "map_leveling.z_correction.local_min_sample_cells",
            map_z_local_min_sample_cells_, 15);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.local_max_tilt_deg",
            map_z_local_max_tilt_deg_, 3.0);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.local_max_residual_p95",
            map_z_local_max_residual_p95_, 0.08);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.local_min_range",
            map_z_local_min_range_, 0.5);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.local_max_range",
            map_z_local_max_range_, 8.0);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.local_max_source_distance",
            map_z_local_max_source_distance_, 2.5);
        this->get_parameter_or<int>(
            "map_leveling.z_correction.local_min_independent_sources",
            map_z_local_min_independent_sources_, 2);
        this->get_parameter_or<int>(
            "map_leveling.z_correction.local_min_source_sample_cells",
            map_z_local_min_source_sample_cells_, 3);
        this->get_parameter_or<int>(
            "map_leveling.z_correction.local_keyframe_window",
            map_z_local_keyframe_window_, 6);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.min_valid_fraction",
            map_z_correction_config_.min_valid_fraction, 0.80);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.max_unobserved_gap_m",
            map_z_correction_config_.max_unobserved_gap_m, 2.0);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.outlier_threshold_m",
            map_z_correction_config_.outlier_threshold_m, 0.12);
        int map_z_outlier_window = 3;
        this->get_parameter_or<int>(
            "map_leveling.z_correction.outlier_window",
            map_z_outlier_window, 3);
        map_z_correction_config_.outlier_window = map_z_outlier_window > 0
            ? static_cast<std::size_t>(map_z_outlier_window) : 0U;
        map_z_correction_config_.min_independent_sources =
            map_z_local_min_independent_sources_ > 0
                ? static_cast<std::size_t>(map_z_local_min_independent_sources_)
                : 0U;
        map_z_correction_config_.min_sample_cells =
            map_z_local_min_sample_cells_ > 0
                ? static_cast<std::size_t>(map_z_local_min_sample_cells_)
                : 0U;
        map_z_correction_config_.max_local_tilt_deg =
            map_z_local_max_tilt_deg_;
        map_z_correction_config_.max_local_residual_p95 =
            map_z_local_max_residual_p95_;
        this->get_parameter_or<double>(
            "map_leveling.z_correction.smoothness",
            map_z_correction_config_.smoothness, 0.03);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.max_abs_correction_m",
            map_z_correction_config_.max_abs_correction_m, 0.45);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.max_correction_slope",
            map_z_correction_config_.max_correction_slope, 0.25);
        this->get_parameter_or<double>(
            "map_leveling.z_correction.max_corrected_floor_span_m",
            map_z_correction_config_.max_corrected_floor_span_m, 0.04);
        this->get_parameter_or<double>(
            "map_leveling.max_final_floor_span_m",
            map_leveling_max_final_floor_span_m_, 0.05);
        std::string configured_map_output_dir;
        this->get_parameter_or<string>("map_output_dir", configured_map_output_dir, "");
        this->get_parameter_or<bool>("keyframe.enable", keyframe_enable_, false);
        this->get_parameter_or<double>("keyframe.trans_threshold", keyframe_trans_threshold_, 0.5);
        this->get_parameter_or<double>("keyframe.yaw_threshold_deg", keyframe_yaw_threshold_deg_, 10.0);
        this->get_parameter_or<double>("keyframe.store_leaf", keyframe_store_leaf_, 0.1);
        this->get_parameter_or<double>(
            "keyframe.ground_store_leaf", keyframe_ground_store_leaf_, 0.025);
        this->get_parameter_or<double>("keyframe.floor_z_map", keyframe_floor_z_map_, -0.304);
        vector<double> lidar_to_base_values;
        this->get_parameter_or<vector<double>>(
            "keyframe.lidar_to_base_T", lidar_to_base_values, vector<double>());
        bool lidar_to_base_valid = lidar_to_base_values.size() == 16;
        if (lidar_to_base_valid)
        {
            for (int row = 0; row < 4; ++row)
            {
                for (int col = 0; col < 4; ++col)
                {
                    lidar_to_base_extrinsic_(row, col) =
                        lidar_to_base_values[static_cast<size_t>(row * 4 + col)];
                }
            }
            const Eigen::Matrix3d rotation =
                lidar_to_base_extrinsic_.block<3, 3>(0, 0);
            lidar_to_base_valid = lidar_to_base_extrinsic_.allFinite() &&
                (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                    .cwiseAbs().maxCoeff() <= 1e-5 &&
                std::abs(rotation.determinant() - 1.0) <= 1e-5 &&
                lidar_to_base_extrinsic_.row(3).isApprox(
                    Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-9);
        }
        if (keyframe_enable_ && !lidar_to_base_valid)
        {
            throw std::invalid_argument(
                "keyframe.lidar_to_base_T is required when keyframe.enable=true "
                "and must be a finite 4x4 rigid transform");
        }
        if (lidar_to_base_valid)
        {
            const Eigen::Vector3d translation =
                lidar_to_base_extrinsic_.block<3, 1>(0, 3);
            const Eigen::Matrix3d rotation =
                lidar_to_base_extrinsic_.block<3, 3>(0, 0);
            const Eigen::Vector3d rpy(
                std::atan2(rotation(2, 1), rotation(2, 2)),
                std::asin(std::clamp(-rotation(2, 0), -1.0, 1.0)),
                std::atan2(rotation(1, 0), rotation(0, 0)));
            RCLCPP_INFO(
                get_logger(),
                "[EXTRINSIC CONFIG] source=keyframe.lidar_to_base_T "
                "direction=lidar_to_base xyz=[%.6f,%.6f,%.6f] "
                "rpy_zyx_deg=[%.3f,%.3f,%.3f]",
                translation.x(), translation.y(), translation.z(),
                rpy.x() * 180.0 / PI_M, rpy.y() * 180.0 / PI_M,
                rpy.z() * 180.0 / PI_M);
        }
        int max_count = 5000;
        this->get_parameter_or<int>("keyframe.max_count", max_count, 5000);
        keyframe_max_count_ = max_count > 0 ? static_cast<size_t>(max_count) : 5000U;
        this->get_parameter_or<int>("keyframe.descriptor.ring_count", descriptor_config_.ring_count, 20);
        this->get_parameter_or<int>("keyframe.descriptor.sector_count", descriptor_config_.sector_count, 60);
        this->get_parameter_or<double>("keyframe.descriptor.max_range", descriptor_config_.max_range, 10.0);
        this->get_parameter_or<double>("keyframe.descriptor.min_range", descriptor_config_.min_range, 0.3);
        this->get_parameter_or<double>("keyframe.descriptor.z_min", descriptor_config_.z_min, -0.2);
        this->get_parameter_or<double>("keyframe.descriptor.z_max", descriptor_config_.z_max, 2.0);
        this->get_parameter_or<bool>("loop.enable", loop_config_.enable, false);
        this->get_parameter_or<int>("loop.min_keyframe_gap", loop_config_.min_keyframe_gap, 60);
        this->get_parameter_or<int>("loop.submap_half_window", loop_config_.submap_half_window, 12);
        this->get_parameter_or<int>(
            "loop.descriptor_candidate_count",
            loop_config_.descriptor_candidate_count, 5);
        this->get_parameter_or<double>("loop.descriptor_max_distance", loop_config_.descriptor_max_distance, 0.30);
        this->get_parameter_or<double>("loop.icp_fitness_threshold", loop_config_.icp_fitness_threshold, 0.15);
        this->get_parameter_or<double>("loop.icp_max_correspondence_distance", loop_config_.icp_max_correspondence_distance, 2.0);
        this->get_parameter_or<int>("loop.icp_max_iterations", loop_config_.icp_max_iterations, 80);
        this->get_parameter_or<double>(
            "loop.max_odom_xy_distance", loop_config_.max_odom_xy_distance, 2.0);
        this->get_parameter_or<int>(
            "loop.consistency_min_hits", loop_config_.consistency_min_hits, 3);
        this->get_parameter_or<int>(
            "loop.consistency_max_keyframe_gap",
            loop_config_.consistency_max_keyframe_gap, 3);
        this->get_parameter_or<double>(
            "loop.consistency_candidate_xy_tolerance",
            loop_config_.consistency_candidate_xy_tolerance, 1.0);
        this->get_parameter_or<double>(
            "loop.max_constraint_xy_correction",
            loop_config_.max_constraint_xy_correction, 0.50);
        this->get_parameter_or<double>(
            "loop.max_constraint_z_correction",
            loop_config_.max_constraint_z_correction, 0.45);
        this->get_parameter_or<double>(
            "loop.max_constraint_yaw_correction_deg",
            loop_config_.max_constraint_yaw_correction_deg, 5.0);
        this->get_parameter_or<double>(
            "loop.max_optimized_xy_deviation",
            loop_config_.max_optimized_xy_deviation, 0.50);
        this->get_parameter_or<double>(
            "loop.max_optimized_z_deviation",
            loop_config_.max_optimized_z_deviation, 0.45);
        this->get_parameter_or<double>(
            "loop.max_optimized_roll_pitch_deviation_deg",
            loop_config_.max_optimized_roll_pitch_deviation_deg, 7.0);
        this->get_parameter_or<double>(
            "loop.max_optimized_yaw_deviation_deg",
            loop_config_.max_optimized_yaw_deviation_deg, 5.0);
        this->get_parameter_or<double>(
            "loop.sanity_path_reference_length_m",
            loop_config_.sanity_path_reference_length_m, 10.0);
        this->get_parameter_or<double>(
            "loop.sanity_path_scale_max", loop_config_.sanity_path_scale_max, 4.0);
        this->get_parameter_or<double>(
            "loop.odom_roll_pitch_sigma", loop_config_.odom_roll_pitch_sigma, 0.01);
        this->get_parameter_or<double>(
            "loop.odom_z_sigma", loop_config_.odom_z_sigma, 0.05);
        this->get_parameter_or<double>(
            "loop.loop_roll_pitch_sigma", loop_config_.loop_roll_pitch_sigma, 0.05);
        this->get_parameter_or<double>(
            "loop.loop_z_sigma", loop_config_.loop_z_sigma, 0.10);
        this->get_parameter_or<double>(
            "loop.odom_rotation_sigma", loop_config_.odom_rotation_sigma, 0.02);
        this->get_parameter_or<double>(
            "loop.odom_translation_sigma", loop_config_.odom_translation_sigma, 0.05);
        this->get_parameter_or<double>(
            "loop.loop_rotation_sigma", loop_config_.loop_rotation_sigma, 0.05);
        this->get_parameter_or<double>(
            "loop.loop_translation_sigma", loop_config_.loop_translation_sigma, 0.10);
        this->get_parameter_or<double>(
            "loop.huber_k", loop_config_.huber_k, 1.345);
        if (!scan_descriptor::validConfig(descriptor_config_))
        {
            RCLCPP_ERROR(get_logger(), "Invalid keyframe descriptor configuration; keyframe collection disabled");
            keyframe_enable_ = false;
        }
        if (!std::isfinite(keyframe_floor_z_map_))
        {
            throw std::invalid_argument(
                "keyframe.floor_z_map must be finite because GL descriptors "
                "and PCD-to-PGM height filtering share it");
        }
        // GL descriptors use this fixed map-frame floor. PCD-to-PGM uses it as
        // the fit seed/fallback, then applies its accepted local ground plane.
        pcd2pgm_options_.floor_z_map = keyframe_floor_z_map_;
        if (!std::isfinite(pcd2pgm_options_.save_voxel) ||
            pcd2pgm_options_.save_voxel <= 0.0 ||
            pcd2pgm_options_.save_voxel >
                static_cast<double>(std::numeric_limits<float>::max()))
        {
            throw std::invalid_argument(
                "pcd2pgm.save_voxel must be finite, positive, and representable as float");
        }
        if (keyframe_enable_ &&
            (!std::isfinite(keyframe_store_leaf_) ||
             keyframe_store_leaf_ <= 0.0 ||
             !std::isfinite(keyframe_ground_store_leaf_) ||
             keyframe_ground_store_leaf_ <= 0.0 ||
             !std::isfinite(pcd2pgm_options_.map_resolution) ||
             pcd2pgm_options_.map_resolution <= 0.0 ||
             keyframe_ground_store_leaf_ >
                 0.5 * pcd2pgm_options_.map_resolution + 1e-9))
        {
            throw std::invalid_argument(
                "keyframe.store_leaf must be positive and "
                "keyframe.ground_store_leaf must be positive and no coarser "
                "than half pcd2pgm.map_resolution");
        }
        if (pcd2pgm_options_.min_points_occupied < 1 ||
            pcd2pgm_options_.min_obstacle_frames < 1 ||
            pcd2pgm_options_.min_free_frames < 1 ||
            !std::isfinite(pcd2pgm_options_.free_to_obstacle_frame_ratio) ||
            pcd2pgm_options_.free_to_obstacle_frame_ratio < 0.0)
        {
            throw std::invalid_argument(
                "pcd2pgm occupancy evidence thresholds are invalid; "
                "min_points_occupied must be >= 1");
        }
        if (pcd2pgm_options_.overhead_min_ray_crossing_frames < 1 ||
            !std::isfinite(pcd2pgm_options_.overhead_min_ray_crossing_ratio) ||
            pcd2pgm_options_.overhead_min_ray_crossing_ratio < 0.0 ||
            pcd2pgm_options_.overhead_min_ray_crossing_ratio > 1.0)
        {
            throw std::invalid_argument(
                "pcd2pgm overhead visibility thresholds are invalid");
        }
        if (!std::isfinite(map_leveling_max_correction_deg_) ||
            map_leveling_max_correction_deg_ <= 0.0 ||
            !std::isfinite(map_leveling_min_inlier_ratio_) ||
            map_leveling_min_inlier_ratio_ <= 0.0 ||
            map_leveling_min_inlier_ratio_ > 1.0 ||
            !std::isfinite(map_leveling_sample_quantile_) ||
            map_leveling_sample_quantile_ <= 0.0 ||
            map_leveling_sample_quantile_ > 0.20 ||
            !std::isfinite(map_leveling_fallback_sample_quantile_) ||
            map_leveling_fallback_sample_quantile_ <=
                map_leveling_sample_quantile_ ||
            map_leveling_fallback_sample_quantile_ > 0.20 ||
            !std::isfinite(map_leveling_converged_tilt_deg_) ||
            map_leveling_converged_tilt_deg_ <= 0.0 ||
            map_leveling_converged_tilt_deg_ >=
                map_leveling_max_correction_deg_ ||
            map_leveling_max_iterations_ < 1)
        {
            throw std::invalid_argument("map_leveling parameters are invalid");
        }
        if (!std::isfinite(map_leveling_max_final_floor_span_m_) ||
            map_leveling_max_final_floor_span_m_ <= 0.0 ||
            !std::isfinite(map_z_local_candidate_min_height_) ||
            !std::isfinite(map_z_local_candidate_max_height_) ||
            map_z_local_candidate_min_height_ >= map_z_local_candidate_max_height_ ||
            map_z_local_min_sample_cells_ < 3 ||
            !std::isfinite(map_z_local_max_tilt_deg_) ||
            map_z_local_max_tilt_deg_ <= 0.0 ||
            !std::isfinite(map_z_local_max_residual_p95_) ||
            map_z_local_max_residual_p95_ <= 0.0 ||
            !std::isfinite(map_z_local_min_range_) ||
             !std::isfinite(map_z_local_max_range_) ||
             map_z_local_min_range_ < 0.0 ||
             map_z_local_max_range_ <= map_z_local_min_range_ ||
             !std::isfinite(map_z_local_max_source_distance_) ||
             map_z_local_max_source_distance_ <= 0.0 ||
             map_z_local_min_independent_sources_ < 2 ||
             map_z_local_min_source_sample_cells_ < 1 ||
             map_z_local_keyframe_window_ < 0 ||
            map_z_local_keyframe_window_ > 30 ||
            (map_z_correction_enabled_ &&
             (map_z_correction_config_.outlier_window == 0U ||
              !std::isfinite(map_z_correction_config_.min_valid_fraction) ||
              map_z_correction_config_.min_valid_fraction <= 0.0 ||
              map_z_correction_config_.min_valid_fraction > 1.0 ||
              !std::isfinite(map_z_correction_config_.max_unobserved_gap_m) ||
              map_z_correction_config_.max_unobserved_gap_m <= 0.0 ||
              !std::isfinite(map_z_correction_config_.outlier_threshold_m) ||
              map_z_correction_config_.outlier_threshold_m <= 0.0 ||
              !std::isfinite(map_z_correction_config_.smoothness) ||
              map_z_correction_config_.smoothness < 0.0 ||
              !std::isfinite(map_z_correction_config_.max_abs_correction_m) ||
              map_z_correction_config_.max_abs_correction_m <= 0.0 ||
              !std::isfinite(map_z_correction_config_.max_correction_slope) ||
              map_z_correction_config_.max_correction_slope <= 0.0 ||
              !std::isfinite(map_z_correction_config_.max_corrected_floor_span_m) ||
              map_z_correction_config_.max_corrected_floor_span_m <= 0.0)))
        {
            throw std::invalid_argument("map_leveling z-correction parameters are invalid");
        }
        if (map_z_correction_enabled_ && !keyframe_enable_)
        {
            throw std::invalid_argument(
                "map_leveling.z_correction.enabled requires keyframe.enable=true");
        }
        if (loop_config_.enable && !keyframe_enable_)
        {
            RCLCPP_ERROR(get_logger(), "loop.enable requires keyframe.enable=true; loop closure disabled");
            loop_config_.enable = false;
        }
        if (loop_config_.enable)
        {
            const auto require_positive = [](double value, const char* name)
            {
                if (!std::isfinite(value) || value <= 0.0)
                {
                    throw std::invalid_argument(
                        std::string(name) + " must be finite and > 0");
                }
            };
            require_positive(
                loop_config_.descriptor_max_distance,
                "loop.descriptor_max_distance");
            require_positive(
                loop_config_.icp_fitness_threshold,
                "loop.icp_fitness_threshold");
            require_positive(
                loop_config_.icp_max_correspondence_distance,
                "loop.icp_max_correspondence_distance");
            require_positive(
                loop_config_.max_odom_xy_distance,
                "loop.max_odom_xy_distance");
            require_positive(
                loop_config_.consistency_candidate_xy_tolerance,
                "loop.consistency_candidate_xy_tolerance");
            require_positive(
                loop_config_.max_constraint_xy_correction,
                "loop.max_constraint_xy_correction");
            require_positive(
                loop_config_.max_constraint_z_correction,
                "loop.max_constraint_z_correction");
            require_positive(
                loop_config_.max_constraint_yaw_correction_deg,
                "loop.max_constraint_yaw_correction_deg");
            require_positive(
                loop_config_.max_optimized_xy_deviation,
                "loop.max_optimized_xy_deviation");
            require_positive(
                loop_config_.max_optimized_z_deviation,
                "loop.max_optimized_z_deviation");
            require_positive(
                loop_config_.max_optimized_roll_pitch_deviation_deg,
                "loop.max_optimized_roll_pitch_deviation_deg");
            require_positive(
                loop_config_.max_optimized_yaw_deviation_deg,
                "loop.max_optimized_yaw_deviation_deg");
            require_positive(
                loop_config_.sanity_path_reference_length_m,
                "loop.sanity_path_reference_length_m");
            require_positive(
                loop_config_.sanity_path_scale_max,
                "loop.sanity_path_scale_max");
            require_positive(
                loop_config_.odom_roll_pitch_sigma,
                "loop.odom_roll_pitch_sigma");
            require_positive(
                loop_config_.odom_z_sigma, "loop.odom_z_sigma");
            require_positive(
                loop_config_.loop_roll_pitch_sigma,
                "loop.loop_roll_pitch_sigma");
            require_positive(loop_config_.loop_z_sigma, "loop.loop_z_sigma");
            require_positive(
                loop_config_.odom_rotation_sigma, "loop.odom_rotation_sigma");
            require_positive(
                loop_config_.odom_translation_sigma,
                "loop.odom_translation_sigma");
            require_positive(
                loop_config_.loop_rotation_sigma, "loop.loop_rotation_sigma");
            require_positive(
                loop_config_.loop_translation_sigma,
                "loop.loop_translation_sigma");
            require_positive(loop_config_.huber_k, "loop.huber_k");
            if (loop_config_.min_keyframe_gap <= 0 ||
                loop_config_.submap_half_window < 0 ||
                loop_config_.descriptor_candidate_count <= 0 ||
                loop_config_.descriptor_candidate_count > 20 ||
                loop_config_.icp_max_iterations <= 0 ||
                loop_config_.consistency_min_hits <= 0 ||
                loop_config_.consistency_max_keyframe_gap <= 0)
            {
                throw std::invalid_argument(
                    "loop integer parameters must be positive, "
                    "loop.descriptor_candidate_count must be in [1,20], and "
                    "loop.submap_half_window may be zero");
            }
            if (loop_config_.max_constraint_yaw_correction_deg > 180.0 ||
                loop_config_.max_optimized_roll_pitch_deviation_deg > 90.0 ||
                loop_config_.max_optimized_yaw_deviation_deg > 180.0)
            {
                throw std::invalid_argument(
                    "loop angular safety limits exceed their physical range");
            }
        }

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.world_points_en", pub_world_points_flag_, true);
        this->get_parameter_or<bool>("publish.body_points_en", pub_body_points_flag_, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic, "/livox/imu");
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_corner", filter_size_corner_min, 0.5);
        this->get_parameter_or<double>("filter_size_surf", filter_size_surf_min, 0.5);
        this->get_parameter_or<double>("filter_size_map", filter_size_map_min, 0.5);
        this->get_parameter_or<double>("cube_side_length", cube_len, 200.f);
        this->get_parameter_or<float>("mapping.det_range", DET_RANGE, 300.f);
        this->get_parameter_or<double>("mapping.fov_degree", fov_deg, 180.f);
        this->get_parameter_or<double>("mapping.gyr_cov", gyr_cov, 0.1);
        this->get_parameter_or<double>("mapping.acc_cov", acc_cov, 0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov", b_gyr_cov, 0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov", b_acc_cov, 0.0001);
        double imu_init_duration_s = 3.0;
        double imu_init_max_acc_std_g = 0.05;
        double imu_init_max_gyr_std_rad_s = 0.05;
        double imu_init_max_mean_gyr_rad_s = 0.05;
        double imu_init_max_acc_norm_error_g = 0.05;
        vector<double> init_ba_values{0.0, 0.0, 0.0};
        vector<double> init_bg_values{0.0, 0.0, 0.0};
        vector<double> init_gravity_correction_values{0.0, 0.0, 0.0};
        this->get_parameter_or<double>(
            "mapping.imu_init_duration_s", imu_init_duration_s, 3.0);
        this->get_parameter_or<double>(
            "mapping.imu_init_max_acc_std_g", imu_init_max_acc_std_g, 0.05);
        this->get_parameter_or<double>(
            "mapping.imu_init_max_gyr_std_rad_s",
            imu_init_max_gyr_std_rad_s, 0.05);
        this->get_parameter_or<double>(
            "mapping.imu_init_max_mean_gyr_rad_s",
            imu_init_max_mean_gyr_rad_s, 0.05);
        this->get_parameter_or<double>(
            "mapping.imu_init_max_acc_norm_error_g",
            imu_init_max_acc_norm_error_g, 0.05);
        this->get_parameter_or<vector<double>>(
            "mapping.init_ba", init_ba_values,
            vector<double>{0.0, 0.0, 0.0});
        this->get_parameter_or<vector<double>>(
            "mapping.init_bg", init_bg_values,
            vector<double>{0.0, 0.0, 0.0});
        this->get_parameter_or<vector<double>>(
            "mapping.init_gravity_correction_rpy",
            init_gravity_correction_values,
            vector<double>{0.0, 0.0, 0.0});
        this->get_parameter_or<bool>(
            "diagnostics.lio_trace_enabled", lio_trace_enabled_, false);
        this->get_parameter_or<int>(
            "diagnostics.lio_trace_every_n_scans", lio_trace_every_n_scans_, 1);
        lio_trace_every_n_scans_ = std::max(1, lio_trace_every_n_scans_);
        this->get_parameter_or<bool>(
            "lio.time_guard.enabled", lio_time_guard_enabled_, true);
        this->get_parameter_or<double>(
            "lio.time_guard.max_scan_gap", lio_time_guard_max_scan_gap_, 0.25);
        this->get_parameter_or<double>(
            "lio.time_guard.max_imu_dt", lio_time_guard_max_imu_dt_, 0.10);
        this->get_parameter_or<double>(
            "lio.time_guard.max_scan_overlap",
            lio_time_guard_max_scan_overlap_, 0.002);
        if (lio_time_guard_enabled_ &&
            (!std::isfinite(lio_time_guard_max_scan_gap_) ||
             lio_time_guard_max_scan_gap_ <= 0.0 ||
             !std::isfinite(lio_time_guard_max_imu_dt_) ||
             lio_time_guard_max_imu_dt_ <= 0.0 ||
             !std::isfinite(lio_time_guard_max_scan_overlap_) ||
             lio_time_guard_max_scan_overlap_ < 0.0))
        {
            throw std::invalid_argument(
                "lio.time_guard thresholds must be finite; scan/imu gaps must be > 0 "
                "and scan overlap must be >= 0 when enabled");
        }
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("edge_constraint_enable", edge_constraint_enable, false);
        this->get_parameter_or<double>("filter_size_edge", filter_size_edge_min, 0.1);
        this->get_parameter_or<double>("edge_residual_weight", edge_residual_weight, 0.5);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<vector<double>>(
            "mapping.extrinsic_T", extrinT,
            vector<double>{-0.011, -0.02329, 0.04412});
        this->get_parameter_or<vector<double>>(
            "mapping.extrinsic_R", extrinR,
            vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        if (extrinT.size() != 3U || extrinR.size() != 9U ||
            !std::all_of(extrinT.begin(), extrinT.end(),
                [](double value) { return std::isfinite(value); }) ||
            !std::all_of(extrinR.begin(), extrinR.end(),
                [](double value) { return std::isfinite(value); }))
        {
            throw std::invalid_argument(
                "mapping.extrinsic_T/R must contain finite 3/9 values");
        }
        Eigen::Matrix3d configured_extrinsic_R;
        configured_extrinsic_R << MAT_FROM_ARRAY(extrinR);
        if ((configured_extrinsic_R.transpose() * configured_extrinsic_R -
             Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1e-5 ||
            std::abs(configured_extrinsic_R.determinant() - 1.0) > 1e-5)
        {
            throw std::invalid_argument(
                "mapping.extrinsic_R must be a proper rotation matrix");
        }

        if (!configured_map_output_dir.empty())
        {
            data_path_ = configured_map_output_dir;
        }
        else
        {
#ifdef ROOT_DIR
            data_path_ = std::string(ROOT_DIR) + "/map";
#else
            RCLCPP_INFO(this->get_logger(), "There is no macro definition of ROOT_DIR");
            data_path_ = "/home/user_name/.jszr/map";
#endif
        }
        if (!checkDirExist(data_path_))
        {
            RCLCPP_INFO(this->get_logger(), "Create map directory failed!!!!!!.");
            return;
        }
        // A successfully leveled map carries the floor used by its PCD/PGM
        // artifacts. Prefer that sidecar over a stale compile-time default,
        // but never consume a partial or explicitly failed leveling result.
        const std::string leveling_file = data_path_ + "/map_leveling.yaml";
        double leveled_floor_z = 0.0;
        anubis_mapping::MapLevelingMetadata leveling_metadata;
        if (anubis_mapping::readValidatedMapLevelingFloor(
                leveling_file, leveled_floor_z,
                std::filesystem::path(data_path_), &leveling_metadata))
        {
            RCLCPP_INFO(
                this->get_logger(),
                "Using floor_z_map=%.6f from validated %s (generation=%s, artifacts=%zu, config value was %.6f)",
                leveled_floor_z, leveling_file.c_str(),
                leveling_metadata.generation.c_str(),
                leveling_metadata.artifacts.size(), keyframe_floor_z_map_);
            keyframe_floor_z_map_ = leveled_floor_z;
        }
        else if (std::filesystem::exists(leveling_file))
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Ignoring %s because its v3 manifest is missing, stale, or inconsistent",
                leveling_file.c_str());
        }
        pcd2pgm_options_.floor_z_map = keyframe_floor_z_map_;
        RCLCPP_INFO(this->get_logger(), "Map output directory: %s", data_path_.c_str());

        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

        path.header.stamp    = this->get_clock()->now();
        path.header.frame_id = "map";

        FOV_DEG      = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudType());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        // Edges are sparse and geometrically informative: keep them at a finer
        // leaf size than planar points so they are not thinned out.
        downSizeFilterEdge.setLeafSize(filter_size_edge_min, filter_size_edge_min, filter_size_edge_min);
        if (edge_constraint_enable && !p_pre->feature_enabled)
        {
            RCLCPP_WARN(this->get_logger(),
                        "edge_constraint_enable=true but feature_extract_enable=false; "
                        "no FeatureLabel is produced, edge constraints stay inactive.");
        }
        RCLCPP_INFO(this->get_logger(),
                    "Feature extraction: %s | edge constraints: %s (leaf=%.3f weight=%.2f)",
                    p_pre->feature_enabled ? "on" : "off",
                    edge_constraint_enable ? "on" : "off",
                    filter_size_edge_min, edge_residual_weight);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(Vec3d(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(Vec3d(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(Vec3d(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(Vec3d(b_acc_cov, b_acc_cov, b_acc_cov));
        p_imu->set_init_duration(imu_init_duration_s);
        p_imu->set_init_stationary_thresholds(
            imu_init_max_acc_std_g,
            imu_init_max_gyr_std_rad_s,
            imu_init_max_mean_gyr_rad_s);
        if (init_ba_values.size() != 3U ||
            init_bg_values.size() != 3U ||
            init_gravity_correction_values.size() != 3U ||
            !std::all_of(init_ba_values.begin(), init_ba_values.end(),
                [](double value) { return std::isfinite(value); }) ||
            !std::all_of(init_bg_values.begin(), init_bg_values.end(),
                [](double value) { return std::isfinite(value); }) ||
            !std::all_of(init_gravity_correction_values.begin(),
                init_gravity_correction_values.end(),
                [](double value) { return std::isfinite(value); }))
        {
            throw std::invalid_argument(
                "mapping.init_ba, mapping.init_bg and "
                "init_gravity_correction_rpy must each contain three finite "
                "values");
        }
        p_imu->set_initial_biases(
            Vec3d(init_ba_values[0], init_ba_values[1], init_ba_values[2]),
            Vec3d(init_bg_values[0], init_bg_values[1], init_bg_values[2]));
        p_imu->set_initial_gravity_correction_rpy(Vec3d(
            init_gravity_correction_values[0],
            init_gravity_correction_values[1],
            init_gravity_correction_values[2]));
        p_imu->set_init_acc_norm_error(imu_init_max_acc_norm_error_g);
        RCLCPP_INFO(this->get_logger(),
            "[LIO CONFIG] trace=%s every_n=%d imu_init_s=%.3f "
            "imu_init_gate=[acc_std_g<=%.3f gyr_std_rad_s<=%.3f "
            "mean_gyr_rad_s<=%.3f acc_norm_error_g<=%.3f] "
            "acc_cov=%.9f gyr_cov=%.9f b_acc_cov=%.9f b_gyr_cov=%.9f "
            "laser_point_cov=%.9f time_offset_lidar_to_imu=%.9f "
            "extrinsic_est=%s extrinsic_T=[%.9f %.9f %.9f]",
            lio_trace_enabled_ ? "on" : "off", lio_trace_every_n_scans_,
            imu_init_duration_s,
            imu_init_max_acc_std_g, imu_init_max_gyr_std_rad_s,
            imu_init_max_mean_gyr_rad_s, imu_init_max_acc_norm_error_g,
            acc_cov, gyr_cov, b_acc_cov, b_gyr_cov,
            LASER_POINT_COV, time_diff_lidar_to_imu,
            extrinsic_est_en ? "on" : "off",
            Lidar_T_wrt_IMU.x(), Lidar_T_wrt_IMU.y(), Lidar_T_wrt_IMU.z());
        RCLCPP_INFO(this->get_logger(),
            "[LIO CONFIG BIAS] init_ba_g=[%.9f %.9f %.9f] "
            "init_bg_rad_s=[%.9f %.9f %.9f]",
            init_ba_values[0], init_ba_values[1], init_ba_values[2],
            init_bg_values[0], init_bg_values[1], init_bg_values[2]);
        RCLCPP_INFO(this->get_logger(),
            "[LIO CONFIG GEOM] max_iteration=%d surf_leaf=%.3f map_leaf=%.3f "
            "edge_enable=%s edge_leaf=%.3f edge_weight=%.3f blind=%.3f "
            "lidar_type=%d scan_line=%d point_filter_num=%d "
            "scan_rate=%d timestamp_unit=%d "
            "extrinsic_R=[%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f]",
            NUM_MAX_ITERATIONS, filter_size_surf_min, filter_size_map_min,
            edge_constraint_enable ? "on" : "off", filter_size_edge_min,
            edge_residual_weight, p_pre->blind, p_pre->lidar_type,
            p_pre->N_SCANS, p_pre->point_filter_num, p_pre->SCAN_RATE,
            p_pre->time_unit,
            Lidar_R_wrt_IMU(0, 0), Lidar_R_wrt_IMU(0, 1), Lidar_R_wrt_IMU(0, 2),
            Lidar_R_wrt_IMU(1, 0), Lidar_R_wrt_IMU(1, 1), Lidar_R_wrt_IMU(1, 2),
            Lidar_R_wrt_IMU(2, 0), Lidar_R_wrt_IMU(2, 1), Lidar_R_wrt_IMU(2, 2));

        init();
        RCLCPP_INFO(
            get_logger(),
            "[TIME GUARD CONFIG] enabled=%d max_scan_gap=%.3fs "
            "max_scan_overlap=%.3fs max_imu_dt=%.3fs",
            lio_time_guard_enabled_ ? 1 : 0,
            lio_time_guard_max_scan_gap_,
            lio_time_guard_max_scan_overlap_,
            lio_time_guard_max_imu_dt_);
        pcd2grid_ptr_ = std::make_shared<Pcd2Grid>(pcd2pgm_options_);
        if (keyframe_enable_)
        {
            loop_closure_ = std::make_unique<LoopClosure>(
                keyframe_store_, descriptor_config_, keyframe_floor_z_map_, loop_config_,
                [this](const std::string& message) { RCLCPP_INFO(get_logger(), "%s", message.c_str()); },
                [this](const std::string& message) { RCLCPP_WARN(get_logger(), "%s", message.c_str()); });
            RCLCPP_INFO(get_logger(),
                        "Keyframe collection enabled: trans=%.2fm yaw=%.1fdeg "
                         "max=%zu loop=%s voxel_frame=base_link stored_frame=livox_frame "
                         "descriptor_leaf=%.3fm dense_leaf=%.3fm floor_z_map=%.3fm",
                         keyframe_trans_threshold_, keyframe_yaw_threshold_deg_, keyframe_max_count_,
                         loop_config_.enable ? "on" : "off", keyframe_store_leaf_,
                         keyframe_ground_store_leaf_, keyframe_floor_z_map_);
            RCLCPP_INFO(
                get_logger(),
                "[MAP Z CONFIG] enabled=%d window=%d source_distance=%.2fm "
                "independent_sources=%d source_cells=%d total_cells=%d "
                "valid_fraction=%.2f max_gap=%.2fm",
                map_z_correction_enabled_ ? 1 : 0,
                map_z_local_keyframe_window_,
                map_z_local_max_source_distance_,
                map_z_local_min_independent_sources_,
                map_z_local_min_source_sample_cells_,
                map_z_local_min_sample_cells_,
                map_z_correction_config_.min_valid_fraction,
                map_z_correction_config_.max_unobserved_gap_m);
            if (loop_config_.enable)
            {
                RCLCPP_INFO(
                    get_logger(),
                    "[LOOP CONFIG] candidates=%d desc_max=%.3f icp_fitness_max=%.3f "
                    "icp_corr_max=%.2fm icp_iter=%d odom_xy_max=%.2fm "
                    "consistency=[hits=%d current_gap=%d candidate_xy=%.2fm]",
                    loop_config_.descriptor_candidate_count,
                    loop_config_.descriptor_max_distance,
                    loop_config_.icp_fitness_threshold,
                    loop_config_.icp_max_correspondence_distance,
                    loop_config_.icp_max_iterations,
                    loop_config_.max_odom_xy_distance,
                    loop_config_.consistency_min_hits,
                    loop_config_.consistency_max_keyframe_gap,
                    loop_config_.consistency_candidate_xy_tolerance);
                RCLCPP_INFO(
                    get_logger(),
                    "[LOOP SAFETY] constraint_max=[xy=%.2fm z=%.2fm yaw=%.1fdeg] "
                    "optimized_max=[xy=%.2fm z=%.2fm rp=%.1fdeg yaw=%.1fdeg]",
                    loop_config_.max_constraint_xy_correction,
                    loop_config_.max_constraint_z_correction,
                    loop_config_.max_constraint_yaw_correction_deg,
                    loop_config_.max_optimized_xy_deviation,
                    loop_config_.max_optimized_z_deviation,
                     loop_config_.max_optimized_roll_pitch_deviation_deg,
                     loop_config_.max_optimized_yaw_deviation_deg);
                RCLCPP_INFO(
                    get_logger(),
                    "[LOOP NOISE] odom=[rp=%.4frad yaw=%.4frad xy=%.4fm z=%.4fm] "
                    "loop=[rp=%.4frad yaw=%.4frad xy=%.4fm z=%.4fm] "
                    "sanity_path=[reference=%.2fm scale_max=%.2f]",
                    loop_config_.odom_roll_pitch_sigma,
                    loop_config_.odom_rotation_sigma,
                    loop_config_.odom_translation_sigma,
                    loop_config_.odom_z_sigma,
                    loop_config_.loop_roll_pitch_sigma,
                    loop_config_.loop_rotation_sigma,
                    loop_config_.loop_translation_sigma,
                    loop_config_.loop_z_sigma,
                    loop_config_.sanity_path_reference_length_m,
                    loop_config_.sanity_path_scale_max);
            }
        }

        sub_lidar_ptr_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lid_topic, rclcpp::QoS(10).reliable(), std::bind(&MappingAlg::lidarCallBack, this, std::placeholders::_1));

        sub_imu_ptr_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, rclcpp::QoS(200).reliable(), std::bind(&MappingAlg::imuCallBack, this, std::placeholders::_1));
        pubLaserCloudFull_      = this->create_publisher<sensor_msgs::msg::PointCloud2>("/world_points", rclcpp::QoS(20).reliable());
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/body_points", 20);
        pubLaserCloudMap_       = this->create_publisher<sensor_msgs::msg::PointCloud2>("/map_points", 20);
        pubOdomAftMapped_       = this->create_publisher<nav_msgs::msg::Odometry>("/slam_odom", 20);
        pubPath_                = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        tf_broadcaster_         = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        state_service_ = this->create_service<anubis_interfaces::srv::MapState>(
            "/slam_state_service", std::bind(&MappingAlg::stateCallBack, this, std::placeholders::_1, std::placeholders::_2));

        // --- AprilTag anchor integration ---
        // Detection (image capture, AprilTag decoding) runs in a separate
        // node (anubis_perception, wrapping apriltag_ros). apriltag_ros
        // does not carry a 3D pose in its detection message at all (only
        // pixel corners + a homography) -- it publishes the tag pose on
        // /tf instead, child_frame_id "tag<family>:<id>". This node reads
        // that via a TF lookup rather than any field on msg -- see
        // aprilTagDetectionCallBack(). Because anubis_description already
        // publishes the full base_link->livox_frame AND
        // base_link->camera_link->...->camera_color_optical_frame chain,
        // there is no separate lidar-to-camera extrinsic parameter here
        // at all (unlike the old aruco.lidar_to_camera_T) -- tf2 composes
        // the whole chain for us.
        this->declare_parameter<bool>("apriltag.enable", false);
        this->declare_parameter<std::string>("apriltag.detection_topic", "/perception/apriltag/detections");
        this->declare_parameter<double>("apriltag.max_keyframe_time_gap_s", 1.0);
        this->declare_parameter<int>("apriltag.min_observations", 3);
        this->declare_parameter<double>("apriltag.max_position_deviation_m", 0.05);
        // hamming = corrected bit count from the detector (0 is best);
        // decision_margin = detector confidence (higher is better). These
        // replace the old reprojection_error_px/depth_validated gate,
        // which depended on ArUco's own PnP pose estimate -- apriltag_ros
        // does not expose an equivalent per-detection residual.
        this->declare_parameter<int>("apriltag.max_hamming", 0);
        this->declare_parameter<double>("apriltag.min_decision_margin", 50.0);
        this->declare_parameter<std::string>("apriltag.tag_family", "36h11");
        this->declare_parameter<std::string>("apriltag.lidar_frame_id", "livox_frame");
        this->declare_parameter<bool>("apriltag.publish_debug_tf", true);

        apriltag_enable_                  = this->get_parameter("apriltag.enable").as_bool();
        apriltag_detection_topic_         = this->get_parameter("apriltag.detection_topic").as_string();
        apriltag_max_keyframe_time_gap_s_ = this->get_parameter("apriltag.max_keyframe_time_gap_s").as_double();
        apriltag_min_observations_        = this->get_parameter("apriltag.min_observations").as_int();
        apriltag_max_position_deviation_m_ = this->get_parameter("apriltag.max_position_deviation_m").as_double();
        apriltag_max_hamming_             = this->get_parameter("apriltag.max_hamming").as_int();
        apriltag_min_decision_margin_     = this->get_parameter("apriltag.min_decision_margin").as_double();
        apriltag_family_                  = this->get_parameter("apriltag.tag_family").as_string();
        apriltag_lidar_frame_id_          = this->get_parameter("apriltag.lidar_frame_id").as_string();
        apriltag_publish_debug_tf_        = this->get_parameter("apriltag.publish_debug_tf").as_bool();

        if (apriltag_enable_)
        {
            // Buffer duration (10s) only needs to cover the time between an
            // image being captured and this callback running -- not the
            // whole mapping session.
            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

            sub_apriltag_ptr_ = this->create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>(
                apriltag_detection_topic_, rclcpp::QoS(50),
                std::bind(&MappingAlg::aprilTagDetectionCallBack, this, std::placeholders::_1));
            RCLCPP_INFO(
                this->get_logger(), "AprilTag anchor capture enabled on topic '%s' (family=%s, lidar_frame=%s)",
                apriltag_detection_topic_.c_str(), apriltag_family_.c_str(), apriltag_lidar_frame_id_.c_str());
        }

        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, std::bind(&MappingAlg::map_publish_callback, this));

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    MappingAlg ::~MappingAlg() {}

    void MappingAlg::init()
    {
        std::vector<double> epsi(23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, std::bind(&MappingAlg::h_share_model, this, std::placeholders::_1, std::placeholders::_2),
            NUM_MAX_ITERATIONS, epsi.data());
    }

    void MappingAlg::stateCallBack(
        anubis_interfaces::srv::MapState::Request::SharedPtr request, anubis_interfaces::srv::MapState::Response::SharedPtr response)
    {
        uint8_t receive_message = request->data;
        switch (receive_message)
        {
            case 0:
                state_.store(SlamState::STABLE);
                response->success = true;
                response->message = "Set STABLE State!!!!!!";
                break;
            case 1:
                state_.store(SlamState::PASSIVE);
                response->success = true;
                response->message = "Set PASSIVE State!!!!!!";
                break;
            case 2:
                state_.store(SlamState::READY);
                response->success = true;
                response->message = "Set READY State!!!!!!";
                break;
            case 3:
                if (mapping_session_started_)
                {
                    response->success = false;
                    response->message =
                        "A mapping session has already run in this process; restart robot_slam before starting a new map";
                    RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
                    break;
                }
                mapping_session_started_ = true;
                state_.store(SlamState::ACTIVE);
                reset();
                response->success = true;
                response->message = "Set ACTIVE State!!!!!!";
                break;
            case 4:
                state_.store(SlamState::ERROR);
                response->success = true;
                response->message = "Set ERROR State!!!!!!";
                break;
            case 5:
                state_.store(SlamState::SAVE);
                response->success = finish();
                state_.store(
                    (!response->success && lio_time_guard_session_.failed)
                        ? SlamState::ERROR : SlamState::READY);
                response->message = response->success
                    ? "Map saved successfully to " + data_path_
                    : "Map save failed; check robot_slam logs";
                break;
            default:
                response->success = false;
                response->message = "SLAM not this State Fail !!!!!!";
                break;
        }
    }

    void MappingAlg::mark_lio_time_guard_failed(const char* reason)
    {
        // Message-level discontinuities (non-finite/out-of-order stamps in a
        // single callback) drop that message and are counted for the save
        // diagnostics.  They no longer abort the session or flip the node to
        // ERROR: one bad message must not discard a whole mapping run.
        ++lio_time_guard_session_.rejected_frames;
        lio_time_guard_session_.failure_reason =
            reason != nullptr ? reason : "unknown";
        RCLCPP_WARN(
            get_logger(),
            "[TIME GAP GUARD] dropped sensor message: reason=%s "
            "total_rejected=%lu",
            lio_time_guard_session_.failure_reason.c_str(),
            static_cast<unsigned long>(
                lio_time_guard_session_.rejected_frames));
    }

    void MappingAlg::reset()
    {
        time_buffer.clear();
        lidar_buffer.clear();
        imu_buffer.clear();
        is_first_lidar      = true;
        flg_first_scan      = true;
        flg_EKF_inited      = false;
        lidar_pushed        = false;
        scan_num            = 0;
        // Timestamp cursors belong to the mapping session, not to the ROS
        // subscription lifetime.  Keeping an old cursor can make the first
        // frame after a service-triggered reset look like a clock rollback.
        last_timestamp_lidar = 0.0;
        last_timestamp_imu = -1.0;
        has_received_imu_timestamp_ = false;
        hessian_diag_counter_ = 0;
        lio_frame_counter_ = 0;
        current_lio_frame_id_ = 0;
        current_lio_frame_time_ = 0.0;
        lio_time_guard_session_.reset();
        trace_current_lio_frame_ = false;
        previous_synced_lidar_end_time_ = -1.0;
        lio_sync_diag_ = LioSyncDiagnostics{};
        lio_match_diag_ = LioMatchDiagnostics{};
        lidar_mean_scantime = 0.0;
        memset(point_selected_surf, true, sizeof(point_selected_surf));

        // These outputs are independent of the ikd-tree and must start empty
        // even for the first service-triggered session.
        pcl_wait_pub->clear();
        pcl_wait_save->clear();
        path.poses.clear();
        path.header.stamp = this->get_clock()->now();

        p_imu->reset();

        state_ikfom state_updated;
        state_updated.pos = Zero3d;
        state_updated.rot = Quatd(1.0, 0.0, 0.0, 0.0);
        state_point       = state_updated;  // 对state_point进行更新，state_point可视化用到
        kf.change_x(state_updated);
        Localmap_Initialized = false;
        keyframe_store_->clear();
    }


    double MappingAlg::get_time_sec(const builtin_interfaces::msg::Time& time)
    {
        return rclcpp::Time(time).seconds();
    }

    rclcpp::Time MappingAlg::get_ros_time(double timestamp)
    {
        int32_t  sec       = std::floor(timestamp);
        auto     nanosec_d = (timestamp - std::floor(timestamp)) * 1e9;
        uint32_t nanosec   = nanosec_d;
        return rclcpp::Time(sec, nanosec);
    }

    void MappingAlg::pointsBody2World(PointType const* const pi, PointType* const po)
    {
        Vec3d p_body(pi->x, pi->y, pi->z);
        Vec3d p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

        po->x         = p_global(0);
        po->y         = p_global(1);
        po->z         = p_global(2);
        po->intensity = pi->intensity;
    }

    void MappingAlg::pointsBody2Imu(PointType const* const pi, PointType* const po)
    {
        Vec3d p_body_lidar(pi->x, pi->y, pi->z);
        Vec3d p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

        po->x         = p_body_imu(0);
        po->y         = p_body_imu(1);
        po->z         = p_body_imu(2);
        po->intensity = pi->intensity;
    }

    void MappingAlg::points_cache_collect()
    {
        PointVector points_history;
        ikdtree.acquire_removed_points(points_history);
    }

    void MappingAlg::lasermap_fov_segment()
    {
        cub_needrm.clear();
        int   kdtree_delete_counter = 0;
        Vec3d pos_LiD               = pos_lid;
        if (!Localmap_Initialized)
        {
            for (int i = 0; i < 3; i++)
            {
                LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
                LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
            }
            Localmap_Initialized = true;
            return;
        }
        float dist_to_map_edge[3][2];
        bool  need_move = false;
        for (int i = 0; i < 3; i++)
        {
            dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
            dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
                need_move = true;
        }
        if (!need_move)
            return;
        BoxPointType New_LocalMap_Points, tmp_boxpoints;
        New_LocalMap_Points = LocalMap_Points;
        float mov_dist      = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
        for (int i = 0; i < 3; i++)
        {
            tmp_boxpoints = LocalMap_Points;
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
            {
                New_LocalMap_Points.vertex_max[i] -= mov_dist;
                New_LocalMap_Points.vertex_min[i] -= mov_dist;
                tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
                cub_needrm.push_back(tmp_boxpoints);
            }
            else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            {
                New_LocalMap_Points.vertex_max[i] += mov_dist;
                New_LocalMap_Points.vertex_min[i] += mov_dist;
                tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
                cub_needrm.push_back(tmp_boxpoints);
            }
        }
        LocalMap_Points = New_LocalMap_Points;

        points_cache_collect();
        double delete_begin = omp_get_wtime();
        if (cub_needrm.size() > 0)
            kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    }

    void MappingAlg::lidarCallBack(const sensor_msgs::msg::PointCloud2::UniquePtr msg)
    {
        mtx_buffer.lock();
        double cur_time              = get_time_sec(msg->header.stamp);
        double preprocess_start_time = omp_get_wtime();
        scan_count++;
        const int input_scan_sequence = scan_count;
        const bool mapping_active = state_.load() == SlamState::ACTIVE;
        if (!std::isfinite(cur_time))
        {
            if (mapping_active)
            {
                mark_lio_time_guard_failed("non_finite_lidar_timestamp");
            }
            RCLCPP_ERROR(
                get_logger(),
                "[TIME GAP GUARD] dropping LiDAR message with non-finite timestamp");
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            return;
        }
        if (!is_first_lidar && !std::isfinite(last_timestamp_lidar))
        {
            if (mapping_active)
            {
                mark_lio_time_guard_failed("non_finite_previous_lidar_timestamp");
            }
            RCLCPP_ERROR(
                get_logger(),
                "[TIME GAP GUARD] previous LiDAR timestamp is non-finite; "
                "dropping current message");
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            return;
        }
        if (!is_first_lidar && cur_time <= last_timestamp_lidar)
        {
            const char* reason = cur_time == last_timestamp_lidar
                ? "zero_lidar_dt" : "lidar_timestamp_out_of_order";
            if (mapping_active)
            {
                mark_lio_time_guard_failed(reason);
            }
            RCLCPP_ERROR(
                get_logger(),
                "[TIME GAP GUARD] dropping LiDAR timestamp regression: "
                "current=%.9f previous=%.9f reason=%s",
                cur_time, last_timestamp_lidar, reason);
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            return;
        }
        if (is_first_lidar)
        {
            is_first_lidar = false;
        }
        last_timestamp_lidar = cur_time;

        PointCloudType::Ptr ptr(new PointCloudType());

        pcl::PointCloud<livox_pcl::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);

        const bool trace_input_scan = lio_trace_enabled_ &&
            state_.load() == SlamState::ACTIVE &&
            ((input_scan_sequence % lio_trace_every_n_scans_) == 0);
        const CloudZStats input_z_stats = trace_input_scan
            ? summarizeCloudZ(pl_orig)
            : CloudZStats{};
        const double arrival_ros_time = trace_input_scan
            ? this->get_clock()->now().seconds()
            : 0.0;
        bool        has_timestamp = false;
        bool        use_hardware_timestamp = false;
        const char* timestamp_fallback_reason = "not_evaluated";
        double      timestamp_min = 0.0;
        double      timestamp_max = 0.0;
        size_t      timestamp_regressions = 0;

        // The current Mid-360 driver publishes a FLOAT64 timestamp field.  Its
        // value is an absolute nanosecond timestamp (the first point equals the
        // PointCloud2 header stamp), not a relative offset.  The old code
        // unconditionally replaced it with index-based interpolation, losing
        // packet timing and making motion compensation depend on point order.
        // Normalize a valid hardware timestamp to the relative nanoseconds
        // expected by lidar_process.cpp; use interpolation only for older or
        // malformed PointCloud2 publishers.
        if (!pl_orig.empty() && p_pre->SCAN_RATE > 0)
        {
            const double frame_period_ns =
                1e9 / static_cast<double>(p_pre->SCAN_RATE);
            const double header_ns =
                static_cast<double>(rclcpp::Time(msg->header.stamp).nanoseconds());
            has_timestamp = pointCloudHasFloat64Timestamp(*msg);
            use_hardware_timestamp = has_timestamp && std::isfinite(header_ns);
            timestamp_fallback_reason = has_timestamp
                ? (std::isfinite(header_ns) ? "candidate" : "invalid_header")
                : "missing_field";
            timestamp_min = std::numeric_limits<double>::infinity();
            timestamp_max = -std::numeric_limits<double>::infinity();
            double previous_timestamp = 0.0;
            bool timestamp_values_valid = use_hardware_timestamp;

            if (use_hardware_timestamp)
            {
                for (size_t i = 0; i < pl_orig.size(); ++i)
                {
                    const double raw_timestamp = pl_orig.points[i].timestamp;
                    double relative_timestamp = raw_timestamp;
                    // Driver source uses pkt.time_stamp + offset_time, so the
                    // field is absolute when it is close to the frame stamp.
                    if (std::abs(raw_timestamp) > 1e12)
                    {
                        relative_timestamp = raw_timestamp - header_ns;
                    }
                    if (!std::isfinite(relative_timestamp) ||
                        relative_timestamp < -1e6 ||
                        relative_timestamp > frame_period_ns * 2.0)
                    {
                        use_hardware_timestamp = false;
                        timestamp_values_valid = false;
                        timestamp_fallback_reason = "invalid_value_or_range";
                        break;
                    }
                    if (i > 0 && relative_timestamp < previous_timestamp)
                    {
                        ++timestamp_regressions;
                    }
                    previous_timestamp = relative_timestamp;
                    timestamp_min = std::min(timestamp_min, relative_timestamp);
                    timestamp_max = std::max(timestamp_max, relative_timestamp);
                    pl_orig.points[i].timestamp = relative_timestamp;
                }
                // A valid scan should span a substantial part of the configured
                // period.  This rejects a present-but-empty/uninitialized field.
                const bool timestamp_span_valid = pl_orig.size() == 1 ||
                    (timestamp_min <= frame_period_ns * 0.1 &&
                     timestamp_max >= frame_period_ns * 0.5);
                use_hardware_timestamp = use_hardware_timestamp && timestamp_span_valid;
                if (use_hardware_timestamp)
                {
                    timestamp_fallback_reason = "none";
                }
                else if (timestamp_values_valid && !timestamp_span_valid)
                {
                    timestamp_fallback_reason = "insufficient_span";
                }
            }

            if (!use_hardware_timestamp)
            {
                timestamp_min = 0.0;
                timestamp_max = pl_orig.size() > 1 ? frame_period_ns : 0.0;
                const double inv_n_minus_1 =
                    pl_orig.size() > 1
                    ? 1.0 / static_cast<double>(pl_orig.size() - 1)
                    : 0.0;
                for (size_t i = 0; i < pl_orig.size(); ++i)
                {
                    pl_orig.points[i].timestamp =
                        static_cast<double>(i) * inv_n_minus_1 * frame_period_ns;
                }
            }

            if ((scan_count % 100) == 1)
            {
                fprintf(stderr,
                    "[LIDAR TIME] mode=%s points=%zu offset_ms=[%.3f,%.3f] regressions=%zu header_ns=%.0f\n",
                    use_hardware_timestamp ? "hardware" : "interpolated",
                    pl_orig.size(), timestamp_min / 1e6, timestamp_max / 1e6,
                    timestamp_regressions, header_ns);
            }
        }
        else
        {
            timestamp_fallback_reason = pl_orig.empty()
                ? "empty_cloud"
                : "invalid_scan_rate";
        }

        p_pre->process(pl_orig, ptr);
        const CloudZStats processed_z_stats = trace_input_scan
            ? summarizeCloudZ(*ptr)
            : CloudZStats{};
        const PointOffsetStats point_offset_stats = trace_input_scan
            ? summarizePointOffsets(*ptr)
            : PointOffsetStats{};
        lidar_buffer.push_back(ptr);
        time_buffer.push_back(last_timestamp_lidar);
        const std::size_t lidar_queue_size = lidar_buffer.size();
        const std::size_t imu_queue_size = imu_buffer.size();

        mtx_buffer.unlock();
        if (trace_input_scan)
        {
            fprintf(stderr,
                "[LIO INPUT] scan=%d t=%.9f arrival_ros=%.9f stamp_age=%.6f "
                "timestamp_field=%d time_mode=%s fallback=%s "
                "offset_ms=[%.6f %.6f] raw_time_regressions=%zu "
                "raw_n=%zu raw_sample=%zu raw_z=[%.6f %.6f %.6f %.6f %.6f] "
                "processed_n=%zu processed_sample=%zu processed_z=[%.6f %.6f %.6f %.6f %.6f] "
                "point_offset_ms=[first=%.6f last=%.6f min=%.6f max=%.6f] "
                "point_time_n=%zu nonfinite=%zu regressions=%zu queues=[lidar=%zu imu=%zu]\n",
                input_scan_sequence, cur_time, arrival_ros_time,
                arrival_ros_time - cur_time,
                has_timestamp ? 1 : 0,
                use_hardware_timestamp ? "hardware" : "interpolated",
                timestamp_fallback_reason, timestamp_min / 1e6,
                timestamp_max / 1e6,
                timestamp_regressions,
                input_z_stats.count, input_z_stats.sample_count,
                input_z_stats.min, input_z_stats.p05, input_z_stats.p50,
                input_z_stats.p95, input_z_stats.max,
                processed_z_stats.count, processed_z_stats.sample_count,
                processed_z_stats.min, processed_z_stats.p05,
                processed_z_stats.p50, processed_z_stats.p95,
                processed_z_stats.max,
                point_offset_stats.first_ms, point_offset_stats.last_ms,
                point_offset_stats.min_ms, point_offset_stats.max_ms,
                point_offset_stats.count, point_offset_stats.nonfinite_count,
                point_offset_stats.regression_count,
                lidar_queue_size, imu_queue_size);
        }
        sig_buffer.notify_all();
    }

    void MappingAlg::imuCallBack(const sensor_msgs::msg::Imu::UniquePtr msg_in)
    {
        sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

        const double raw_timestamp = get_time_sec(msg_in->header.stamp);
        const double adjusted_timestamp = raw_timestamp - time_diff_lidar_to_imu;
        double timestamp = adjusted_timestamp;

        mtx_buffer.lock();
        const bool mapping_active = state_.load() == SlamState::ACTIVE;
        const bool finite_measurement =
            std::isfinite(msg->angular_velocity.x) &&
            std::isfinite(msg->angular_velocity.y) &&
            std::isfinite(msg->angular_velocity.z) &&
            std::isfinite(msg->linear_acceleration.x) &&
            std::isfinite(msg->linear_acceleration.y) &&
            std::isfinite(msg->linear_acceleration.z);
        if (!std::isfinite(raw_timestamp) ||
            !std::isfinite(adjusted_timestamp) || !finite_measurement)
        {
            if (mapping_active)
            {
                mark_lio_time_guard_failed(
                    finite_measurement
                        ? "non_finite_imu_timestamp"
                        : "non_finite_imu_measurement");
            }
            RCLCPP_ERROR(
                get_logger(),
                "[TIME GAP GUARD] dropping IMU message with non-finite "
                "timestamp or measurement");
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            return;
        }
        msg->header.stamp = get_ros_time(adjusted_timestamp);

        if (has_received_imu_timestamp_)
        {
            const double dt = timestamp - last_timestamp_imu;
            if (!std::isfinite(last_timestamp_imu) || !std::isfinite(dt) ||
                dt <= 0.0)
            {
                const char* reason = dt == 0.0
                    ? "zero_imu_dt" : "imu_timestamp_out_of_order";
                if (mapping_active)
                {
                    mark_lio_time_guard_failed(reason);
                }
                RCLCPP_ERROR(
                    get_logger(),
                    "[TIME GAP GUARD] dropping IMU timestamp regression: "
                    "current=%.9f previous=%.9f dt=%.9f reason=%s",
                    timestamp, last_timestamp_imu, dt, reason);
                imu_buffer.clear();
                mtx_buffer.unlock();
                sig_buffer.notify_all();
                return;
            }
            if (mapping_active && lio_time_guard_enabled_ &&
                dt > lio_time_guard_max_imu_dt_)
            {
                // [2026-08-31 修复] 拒绝跨缺口消息的同时必须前移参考
                // 时间戳(last_timestamp_imu 在函数尾部才会更新):否则
                // 一次瞬态投递断流后,后续每条消息与冻结参考的差值只会
                // 增大,IMU 流被永久锁死,SLAM 冻结(实测 0831 建图
                // 431s 只处理了前 164s)。丢弃动作与诊断保留。
                last_timestamp_imu = timestamp;
                mark_lio_time_guard_failed("imu_dt");
                RCLCPP_ERROR(
                    get_logger(),
                    "[TIME GAP GUARD] IMU callback gap exceeds limit: "
                    "current=%.9f previous=%.9f gap=%.6f limit=%.6f "
                    "(reference advanced; stream will re-sync)",
                    timestamp, last_timestamp_imu, dt,
                    lio_time_guard_max_imu_dt_);
                imu_buffer.clear();
                mtx_buffer.unlock();
                sig_buffer.notify_all();
                return;
            }
        }

        last_timestamp_imu = timestamp;
        has_received_imu_timestamp_ = true;

        ImuMessagePtr imu_msg_ptr =
            std::make_shared<ImuMessage>(timestamp, Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z));

        imu_buffer.push_back(imu_msg_ptr);
        mtx_buffer.unlock();
        sig_buffer.notify_all();
    }

    bool MappingAlg::syncData(MeasureGroup& meas)
    {
        if (lidar_buffer.empty() || imu_buffer.empty())
        {
            return false;
        }

        if (!lidar_pushed)
        {
            lio_sync_diag_ = LioSyncDiagnostics{};
            lio_sync_diag_.lidar_queue_before = lidar_buffer.size();
            lio_sync_diag_.previous_lidar_end = previous_synced_lidar_end_time_;
            meas.lidar          = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();
            const PointOffsetStats point_offsets =
                summarizePointOffsets(*meas.lidar);
            const double fallback_scan_time = lidar_mean_scantime > 0.0
                ? lidar_mean_scantime
                : (p_pre->SCAN_RATE > 0
                    ? 1.0 / static_cast<double>(p_pre->SCAN_RATE)
                    : 0.1);
            const double max_point_scan_time = point_offsets.count > 0U &&
                    point_offsets.max_ms >= 0.0
                ? point_offsets.max_ms / 1000.0
                : -1.0;
            lio_sync_diag_.point_offset_last_ms = point_offsets.last_ms;
            lio_sync_diag_.point_offset_max_ms = point_offsets.max_ms;
            if (meas.lidar->points.size() <= 1)  // time too little
            {
                lidar_end_time = meas.lidar_beg_time + fallback_scan_time;
                fprintf(stderr, "SYNC: Too few points! size=%zu curvature=%.6f fallback_scan=%.6f\n",
                        meas.lidar->points.size(),
                        meas.lidar->points.empty() ? -1.0 : meas.lidar->points.back().curvature,
                        fallback_scan_time);
            }
            else if (max_point_scan_time < 0.5 * fallback_scan_time)
            {
                lidar_end_time = meas.lidar_beg_time + fallback_scan_time;
                static int curvature_low_cnt = 0;
                if (++curvature_low_cnt % 100 == 1)
                    fprintf(stderr,
                        "SYNC: low max_point_time=%.6fms last=%.6fms "
                        "fallback_scan=%.6fs lidar_beg=%.6f\n",
                        point_offsets.max_ms, point_offsets.last_ms,
                        fallback_scan_time, meas.lidar_beg_time);
            }
            else
            {
                scan_num++;
                lidar_end_time = meas.lidar_beg_time + max_point_scan_time;
                lidar_mean_scantime +=
                    (max_point_scan_time - lidar_mean_scantime) / scan_num;
                lio_sync_diag_.used_max_point_time = true;
            }

            meas.lidar_end_time = lidar_end_time;
            if (!std::isfinite(meas.lidar_beg_time) ||
                !std::isfinite(meas.lidar_end_time))
            {
                mark_lio_time_guard_failed("non_finite_lidar_timestamp");
                lidar_buffer.pop_front();
                time_buffer.pop_front();
                lidar_pushed = false;
                return false;
            }
            if (meas.lidar_end_time <= meas.lidar_beg_time)
            {
                mark_lio_time_guard_failed(
                    meas.lidar_end_time == meas.lidar_beg_time
                        ? "zero_lidar_duration"
                        : "lidar_timestamp_reversed");
                lidar_buffer.pop_front();
                time_buffer.pop_front();
                lidar_pushed = false;
                return false;
            }
            lio_sync_diag_.scan_duration_s =
                meas.lidar_end_time - meas.lidar_beg_time;

            lidar_pushed = true;
        }

        if (last_timestamp_imu < lidar_end_time)
        {
            static int imu_wait_cnt = 0;
            if (++imu_wait_cnt % 5000 == 1)
                fprintf(stderr, "SYNC: waiting for IMU, last_imu=%.6f lidar_end=%.6f diff=%.6f\n",
                        last_timestamp_imu, lidar_end_time, lidar_end_time - last_timestamp_imu);
            return false;
        }

        if (!std::isfinite(last_timestamp_imu) ||
            !std::isfinite(imu_buffer.front()->timestamp) ||
            !std::isfinite(imu_buffer.back()->timestamp))
        {
            mark_lio_time_guard_failed("non_finite_imu_timestamp");
            imu_buffer.clear();
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            lidar_pushed = false;
            return false;
        }
        double imu_time = imu_buffer.front()->timestamp;
        lio_sync_diag_.imu_queue_before = imu_buffer.size();
        lio_sync_diag_.imu_queue_front_before = imu_buffer.front()->timestamp;
        lio_sync_diag_.imu_queue_back_before = imu_buffer.back()->timestamp;
        meas.imu.clear();
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = imu_buffer.front()->timestamp;
            if (imu_time > lidar_end_time)
                break;
            meas.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }

        lio_sync_diag_.imu_selected = meas.imu.size();
        for (const auto& imu : meas.imu)
        {
            if (imu->timestamp < meas.lidar_beg_time)
            {
                ++lio_sync_diag_.imu_before_lidar_begin;
            }
            else if (imu->timestamp <= meas.lidar_end_time)
            {
                ++lio_sync_diag_.imu_inside_lidar_scan;
            }
        }
        if (!meas.imu.empty())
        {
            lio_sync_diag_.imu_first_selected = meas.imu.front()->timestamp;
            lio_sync_diag_.imu_last_selected = meas.imu.back()->timestamp;
        }
        else
        {
            fprintf(stderr,
                "[LIO SYNC DROP] lidar=[%.9f %.9f] reason=no_imu_in_scan "
                "imu_queue=[front=%.9f back=%.9f] queues=[lidar=%zu imu=%zu]\n",
                meas.lidar_beg_time, meas.lidar_end_time,
                lio_sync_diag_.imu_queue_front_before,
                lio_sync_diag_.imu_queue_back_before,
                lidar_buffer.size(), imu_buffer.size());
            mark_lio_time_guard_failed("missing_imu_samples");
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            lidar_pushed = false;
            return false;
        }
        lio_sync_diag_.imu_queue_after = imu_buffer.size();
        if (!imu_buffer.empty())
        {
            lio_sync_diag_.imu_queue_front_after = imu_buffer.front()->timestamp;
        }
        lio_sync_diag_.valid = true;
        previous_synced_lidar_end_time_ = meas.lidar_end_time;

        lidar_buffer.pop_front();
        time_buffer.pop_front();
        lidar_pushed = false;
        return true;
    }

    void MappingAlg::map_incremental()
    {
        PointVector PointToAdd;
        PointVector PointNoNeedDownsample;
        PointToAdd.reserve(feats_down_size);
        PointNoNeedDownsample.reserve(feats_down_size);
        for (int i = 0; i < feats_down_size; i++)
        {
            pointsBody2World(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            if (!Nearest_Points[i].empty() && flg_EKF_inited)
            {
                const PointVector& points_near = Nearest_Points[i];
                bool               need_add    = true;
                BoxPointType       Box_of_Point;
                PointType          downsample_result, mid_point;
                mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
                mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
                mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
                float dist  = calc_dist(feats_down_world->points[i], mid_point);
                if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min
                    && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min
                    && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
                {
                    PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                    continue;
                }
                for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
                {
                    if (points_near.size() < NUM_MATCH_POINTS)
                        break;
                    if (calc_dist(points_near[readd_i], mid_point) < dist)
                    {
                        need_add = false;
                        break;
                    }
                }
                if (need_add)
                    PointToAdd.push_back(feats_down_world->points[i]);
            }
            else
            {
                PointToAdd.push_back(feats_down_world->points[i]);
            }
        }

        double st_time        = omp_get_wtime();
        int    add_point_size = ikdtree.Add_Points(PointToAdd, true);
        ikdtree.Add_Points(PointNoNeedDownsample, false);
        add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    }

    void MappingAlg::pubWorldPoints(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
    {

        PointCloudType::Ptr laserCloudFullRes(feats_undistort);
        int                 size = laserCloudFullRes->points.size();
        PointCloudType::Ptr laserCloudWorld(new PointCloudType(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointsBody2World(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }
        if (!loop_config_.enable)
        {
            *pcl_wait_pub += *laserCloudWorld;
        }
        else if (!loop_log_once_)
        {
            loop_log_once_ = true;
            RCLCPP_INFO(get_logger(), "loop.enable=true: full-resolution map accumulation disabled; map is rebuilt from keyframes at save");
        }
        if (pub_world_points_flag_)
        {
            sensor_msgs::msg::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
            laserCloudmsg.header.stamp    = get_ros_time(lidar_end_time);
            laserCloudmsg.header.frame_id = "map";
            pubLaserCloudFull->publish(laserCloudmsg);
        }
    }

    void MappingAlg::pubBodyPoints(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
    {
        int                 size = feats_undistort->points.size();
        PointCloudType::Ptr laserCloudIMUBody(new PointCloudType(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointsBody2Imu(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
        laserCloudmsg.header.stamp    = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "body";
        pubLaserCloudFull_body->publish(laserCloudmsg);
    }

    void MappingAlg::pubMapPoints(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
    {
        // PointCloudType::Ptr laserCloudFullRes(feats_down_body);
        // int                 size = laserCloudFullRes->points.size();
        // PointCloudType::Ptr laserCloudWorld(new PointCloudType(size, 1));

        // for (int i = 0; i < size; i++)
        // {
        //     pointsBody2World(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        // }
        // *pcl_wait_pub += *laserCloudWorld;

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
        laserCloudmsg.header.stamp    = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "map";
        pubLaserCloudMap->publish(laserCloudmsg);
    }

    void MappingAlg::publish_odometry(
        const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster>& tf_br)
    {
        odomAftMapped.header.frame_id = "map";
        odomAftMapped.child_frame_id  = "body";
        odomAftMapped.header.stamp    = get_ros_time(lidar_end_time);
        set_posestamp(odomAftMapped.pose);
        pubOdomAftMapped->publish(odomAftMapped);
        auto P = kf.get_P();
        for (int i = 0; i < 6; i++)
        {
            int k                                    = i < 3 ? i + 3 : i - 3;
            odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
            odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
            odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
            odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
            odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
            odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
        }

        geometry_msgs::msg::TransformStamped trans;
        trans.header.frame_id         = "map";
        trans.child_frame_id          = "body";
        trans.header.stamp            = get_ros_time(lidar_end_time);
        trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
        trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
        trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
        trans.transform.rotation.w    = odomAftMapped.pose.pose.orientation.w;
        trans.transform.rotation.x    = odomAftMapped.pose.pose.orientation.x;
        trans.transform.rotation.y    = odomAftMapped.pose.pose.orientation.y;
        trans.transform.rotation.z    = odomAftMapped.pose.pose.orientation.z;
        tf_br->sendTransform(trans);
    }

    void MappingAlg::publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
    {
        set_posestamp(msg_body_pose);
        msg_body_pose.header.stamp    = get_ros_time(lidar_end_time);  // ros::Time().fromSec(lidar_end_time);
        msg_body_pose.header.frame_id = "map";

        static int jjj = 0;
        jjj++;
        if (jjj % 10 == 0)
        {
            path.poses.push_back(msg_body_pose);
            pubPath->publish(path);
        }
    }

    void MappingAlg::h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data)
    {
        laserCloudOri->clear();
        corr_normvect->clear();

#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
        for (int i = 0; i < feats_down_size; i++)
        {
            PointType& point_body = feats_down_body->points[i];
            PointType  point_world;

            Vec3d p_body(point_body.x, point_body.y, point_body.z);
            Vec3d p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
            point_world.x         = p_global(0);
            point_world.y         = p_global(1);
            point_world.z         = p_global(2);
            point_world.intensity = point_body.intensity;

            vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

            auto& points_near = Nearest_Points[i];

            if (ekfom_data.converge)
            {
                /** Find the closest surfaces in the map **/
                ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
                point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS        ? false
                                         : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                      : true;
            }

            if (!point_selected_surf[i])
                continue;

            // 添加额外的安全检查
            if (points_near.size() < NUM_MATCH_POINTS)
            {
                point_selected_surf[i] = false;
                continue;
            }

            VF(4) pabcd;
            point_selected_surf[i] = false;
            if (esti_plane<float>(pabcd, points_near, 0.1f))
            {
                float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                float s   = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

                // A point on an edge has no well-defined local plane, so the
                // fitted normal is unstable and its point-to-plane residual is
                // noisier than a genuine surface hit. Rather than dropping the
                // geometry (edges carry the strongest positional information in
                // corridors and doorways), require a tighter fit and down-weight
                // what survives.
                const bool is_edge =
                    edge_constraint_enable
                    && static_cast<int>(feats_down_body->points[i].normal_x) == static_cast<int>(FeatureLabel::Edge);
                const float accept_thr = is_edge ? 0.95f : 0.9f;

                if (s > accept_thr)
                {
                    const float w                = is_edge ? static_cast<float>(edge_residual_weight) : 1.0f;
                    point_selected_surf[i]       = true;
                    normvec->points[i].x         = pabcd(0) * w;
                    normvec->points[i].y         = pabcd(1) * w;
                    normvec->points[i].z         = pabcd(2) * w;
                    normvec->points[i].intensity = pd2 * w;
                    res_last[i]                  = abs(pd2);
                }
            }
        }

        effct_feat_num = 0;
        int effct_edge_num = 0;

        for (int i = 0; i < feats_down_size; i++)
        {
            if (point_selected_surf[i])
            {
                laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
                corr_normvect->points[effct_feat_num] = normvec->points[i];
                if (edge_constraint_enable
                    && static_cast<int>(feats_down_body->points[i].normal_x) == static_cast<int>(FeatureLabel::Edge))
                {
                    ++effct_edge_num;
                }
                effct_feat_num++;
            }
        }

        if (edge_constraint_enable && edge_point_count > 0)
        {
            RCLCPP_INFO_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "[edge] down=%d edge_in=%d effective=%d edge_effective=%d (%.1f%% of edges kept)",
                feats_down_size, edge_point_count, effct_feat_num, effct_edge_num,
                edge_point_count > 0 ? 100.0 * effct_edge_num / edge_point_count : 0.0);
        }

        if (effct_feat_num < 1)
        {
            ekfom_data.valid = false;
            std::cerr << "No Effective Points!" << std::endl;
            // ROS_WARN("No Effective Points! \n");
            return;
        }

        /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
        ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, 12);  // 23
        ekfom_data.h.resize(effct_feat_num);

        for (int i = 0; i < effct_feat_num; i++)
        {
            const PointType& laser_p = laserCloudOri->points[i];
            Vec3d            point_this_be(laser_p.x, laser_p.y, laser_p.z);
            Mat3d            point_be_crossmat;
            point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
            Vec3d point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
            Mat3d point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_this);

            const PointType& norm_p = corr_normvect->points[i];
            Vec3d            norm_vec(norm_p.x, norm_p.y, norm_p.z);

            Vec3d C(s.rot.conjugate() * norm_vec);
            Vec3d A(point_crossmat * C);
            if (extrinsic_est_en)
            {
                Vec3d B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
                ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            }
            else
            {
                ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            }

            ekfom_data.h(i) = -norm_p.intensity;
        }

        if (trace_current_lio_frame_)
        {
            std::vector<double> residual_abs;
            residual_abs.reserve(static_cast<std::size_t>(effct_feat_num));
            double residual_sum = 0.0;
            double residual_abs_sum = 0.0;
            double floor_residual_sum = 0.0;
            double floor_residual_abs_sum = 0.0;
            double wall_residual_sum = 0.0;
            double wall_residual_abs_sum = 0.0;
            double z_information = 0.0;
            double z_rhs = 0.0;
            int floor_normals = 0;
            int wall_normals = 0;
            int mixed_normals = 0;
            for (int i = 0; i < effct_feat_num; ++i)
            {
                const double residual = ekfom_data.h(i);
                const double abs_residual = std::abs(residual);
                residual_sum += residual;
                residual_abs_sum += abs_residual;
                residual_abs.push_back(abs_residual);
                const double z_coefficient = ekfom_data.h_x(i, 2);
                z_information += z_coefficient * z_coefficient;
                z_rhs += z_coefficient * residual;

                const PointType& normal = corr_normvect->points[i];
                const double normal_length = std::sqrt(
                    normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
                const double abs_nz = normal_length > 1e-12
                    ? std::abs(static_cast<double>(normal.z)) / normal_length
                    : 0.0;
                if (abs_nz >= 0.7)
                {
                    ++floor_normals;
                    floor_residual_sum += residual;
                    floor_residual_abs_sum += abs_residual;
                }
                else if (abs_nz <= 0.3)
                {
                    ++wall_normals;
                    wall_residual_sum += residual;
                    wall_residual_abs_sum += abs_residual;
                }
                else
                {
                    ++mixed_normals;
                }
            }

            const std::size_t p95_index = static_cast<std::size_t>(
                std::floor(0.95 * static_cast<double>(residual_abs.size() - 1U)));
            std::nth_element(
                residual_abs.begin(),
                residual_abs.begin() + static_cast<std::ptrdiff_t>(p95_index),
                residual_abs.end());
            const double residual_mean = residual_sum / static_cast<double>(effct_feat_num);
            const double residual_abs_mean = residual_abs_sum / static_cast<double>(effct_feat_num);
            const double residual_abs_p95 = residual_abs[p95_index];
            const double residual_abs_max = *std::max_element(
                residual_abs.begin(), residual_abs.end());
            const Vec3d model_rpy_deg = SO3ToEuler(s.rot);

            ++lio_match_diag_.model_calls;
            lio_match_diag_.valid = true;
            if (lio_match_diag_.model_calls == 1)
            {
                lio_match_diag_.initial_effective = effct_feat_num;
                lio_match_diag_.initial_residual_mean = residual_mean;
                lio_match_diag_.initial_residual_abs_mean = residual_abs_mean;
                lio_match_diag_.initial_residual_abs_p95 = residual_abs_p95;
                lio_match_diag_.initial_state_z = s.pos(2);
                lio_match_diag_.initial_state_pitch_deg = model_rpy_deg(1);
            }
            lio_match_diag_.final_effective = effct_feat_num;
            lio_match_diag_.final_edge_effective = effct_edge_num;
            lio_match_diag_.final_floor_normals = floor_normals;
            lio_match_diag_.final_wall_normals = wall_normals;
            lio_match_diag_.final_mixed_normals = mixed_normals;
            lio_match_diag_.final_residual_mean = residual_mean;
            lio_match_diag_.final_residual_abs_mean = residual_abs_mean;
            lio_match_diag_.final_residual_abs_p95 = residual_abs_p95;
            lio_match_diag_.final_residual_abs_max = residual_abs_max;
            lio_match_diag_.final_floor_residual_mean = floor_normals > 0
                ? floor_residual_sum / static_cast<double>(floor_normals)
                : 0.0;
            lio_match_diag_.final_floor_residual_abs_mean = floor_normals > 0
                ? floor_residual_abs_sum / static_cast<double>(floor_normals)
                : 0.0;
            lio_match_diag_.final_wall_residual_mean = wall_normals > 0
                ? wall_residual_sum / static_cast<double>(wall_normals)
                : 0.0;
            lio_match_diag_.final_wall_residual_abs_mean = wall_normals > 0
                ? wall_residual_abs_sum / static_cast<double>(wall_normals)
                : 0.0;
            lio_match_diag_.final_z_information = z_information;
            lio_match_diag_.final_z_rhs = z_rhs;
            lio_match_diag_.final_z_only_step = z_information > 1e-12
                ? z_rhs / z_information
                : 0.0;
            lio_match_diag_.final_model_state_z = s.pos(2);
            lio_match_diag_.final_model_pitch_deg = model_rpy_deg(1);
        }

        // Diagnostic only: inspect the six pose columns of the LiDAR
        // point-to-plane Jacobian.  Position (m) and rotation (rad) have
        // different natural scales, so report both the raw normal matrix and
        // a column-normalized spectrum.  This does not alter the EKF update;
        // it makes corridor/open-room observability visible in the mapping log.
        ++hessian_diag_counter_;
        const bool periodic_hessian_log = (hessian_diag_counter_ % 50) == 1;
        if ((periodic_hessian_log || trace_current_lio_frame_) && effct_feat_num >= 6)
        {
            const Eigen::MatrixXd h_pose = ekfom_data.h_x.leftCols(6);
            const Eigen::Matrix<double, 6, 6> hessian =
                h_pose.transpose() * h_pose;
            Eigen::Matrix<double, 6, 6> scale =
                Eigen::Matrix<double, 6, 6>::Zero();
            for (int column = 0; column < 6; ++column)
            {
                const double norm = h_pose.col(column).norm();
                scale(column, column) = 1.0 / std::max(norm, 1e-12);
            }
            const Eigen::Matrix<double, 6, 6> normalized_hessian =
                scale * hessian * scale;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(
                normalized_hessian);
            if (solver.info() == Eigen::Success)
            {
                const Eigen::Matrix<double, 6, 1> eigenvalues =
                    solver.eigenvalues();
                Eigen::Matrix<double, 6, 1> weak =
                    scale * solver.eigenvectors().col(0);
                const double weak_norm = weak.norm();
                if (weak_norm > 1e-12)
                {
                    weak /= weak_norm;
                }
                const double condition = eigenvalues(5) /
                    std::max(eigenvalues(0), 1e-12);
                if (trace_current_lio_frame_)
                {
                    if (lio_match_diag_.model_calls == 1)
                    {
                        lio_match_diag_.initial_condition = condition;
                        lio_match_diag_.initial_weak = weak;
                    }
                    lio_match_diag_.final_condition = condition;
                    lio_match_diag_.final_weak = weak;
                    lio_match_diag_.final_hessian_diag = hessian.diagonal();
                    lio_match_diag_.final_normalized_eigenvalues = eigenvalues;
                }
                if (periodic_hessian_log)
                {
                    fprintf(stderr,
                        "[LIO H] eff=%d edge=%d frame=%llu t=%.9f "
                        "eig_norm=[%.3e %.3e %.3e %.3e %.3e %.3e] "
                        "cond=%.3e weak_pos_rpy=[%.3f %.3f %.3f %.3f %.3f %.3f] "
                        "diag_raw=[%.3e %.3e %.3e %.3e %.3e %.3e]\n",
                        effct_feat_num, effct_edge_num,
                        static_cast<unsigned long long>(current_lio_frame_id_),
                        current_lio_frame_time_,
                        eigenvalues(0), eigenvalues(1), eigenvalues(2),
                        eigenvalues(3), eigenvalues(4), eigenvalues(5),
                        condition,
                        weak(0), weak(1), weak(2), weak(3), weak(4), weak(5),
                        hessian(0, 0), hessian(1, 1), hessian(2, 2),
                        hessian(3, 3), hessian(4, 4), hessian(5, 5));
                }
            }
        }
    }

    void MappingAlg::run()
    {
        if (state_.load() == SlamState::ACTIVE)
        {

            if (syncData(Measures))
            {
                // Sensor-timeline continuity guard.  A lidar frame whose gap
                // from the last accepted frame is oversized, or whose IMU
                // batch contains a hole, would force the state prediction to
                // integrate across a data blackout (observed on the board as
                // multi-second IMU dropouts followed by run-level trajectory
                // divergence).  Such frames are dropped and counted; the
                // cursor re-anchors so the stream resumes afterwards.
                if (lio_time_guard_enabled_)
                {
                    std::vector<double> imu_timestamps;
                    imu_timestamps.reserve(Measures.imu.size());
                    for (const auto& imu : Measures.imu)
                    {
                        imu_timestamps.push_back(
                            imu != nullptr
                                ? imu->timestamp
                                : std::numeric_limits<double>::quiet_NaN());
                    }
                    const LioTimeGuardConfig guard_config{
                        true, lio_time_guard_max_scan_gap_,
                        lio_time_guard_max_imu_dt_,
                        lio_time_guard_max_scan_overlap_};
                    const LioTimeGuardDecision decision =
                        evaluate_lio_time_guard(
                            Measures.lidar_beg_time, Measures.lidar_end_time,
                            lio_time_guard_session_.previous_lidar_end,
                            imu_timestamps, guard_config,
                            lio_time_guard_session_.previous_imu_end);
                    if (!decision.accepted)
                    {
                        // Drop the frame for LIO but re-anchor the guard
                        // cursor to it so the next continuous frame is
                        // evaluated against the live timeline.
                        commit_lio_time_guard_frame(
                            lio_time_guard_session_, decision,
                            Measures.lidar_end_time,
                            Measures.imu.empty()
                                ? std::numeric_limits<double>::quiet_NaN()
                                : Measures.imu.back()->timestamp);
                        RCLCPP_WARN(
                            get_logger(),
                            "[TIME GAP GUARD] frame dropped #%llu; continuing "
                            "reason=%s "
                            "lidar_beg=%.9f lidar_end=%.9f scan_gap=%.6fs "
                            "(max_gap %.2fs overlap %.3fs) imu_n=%zu imu_dt_max=%.3fs "
                            "(limit %.2f) imu_cross_gap=%.3fs "
                            "total_skipped=%llu",
                            static_cast<unsigned long long>(
                                lio_frame_counter_ + 1U),
                            decision.rejection_reason != nullptr
                                ? decision.rejection_reason : "unknown",
                            Measures.lidar_beg_time, Measures.lidar_end_time,
                            decision.scan_gap, lio_time_guard_max_scan_gap_,
                            lio_time_guard_max_scan_overlap_, Measures.imu.size(),
                            decision.imu_dt_max,
                            lio_time_guard_max_imu_dt_,
                            decision.imu_cross_gap,
                            static_cast<unsigned long long>(
                                lio_time_guard_session_.rejected_frames));
                        return;
                    }
                    if (!commit_lio_time_guard_frame(
                            lio_time_guard_session_, decision,
                            Measures.lidar_end_time,
                            Measures.imu.back()->timestamp))
                    {
                        RCLCPP_ERROR(
                            get_logger(),
                            "[TIME GAP GUARD] failed to commit accepted "
                            "frame timestamps; mapping session aborted");
                        return;
                    }
                }
                current_lio_frame_id_ = ++lio_frame_counter_;
                current_lio_frame_time_ = Measures.lidar_end_time;
                trace_current_lio_frame_ = lio_trace_enabled_ &&
                    ((current_lio_frame_id_ % static_cast<std::uint64_t>(
                        lio_trace_every_n_scans_)) == 0U);
                lio_match_diag_ = LioMatchDiagnostics{};

                const CloudZStats raw_z_stats = trace_current_lio_frame_
                    ? summarizeCloudZ(*Measures.lidar)
                    : CloudZStats{};
                const ImuBatchStats imu_batch_stats = trace_current_lio_frame_
                    ? summarizeImuBatch(Measures.imu)
                    : ImuBatchStats{};
                if (trace_current_lio_frame_)
                {
                    const double no_value = std::numeric_limits<double>::quiet_NaN();
                    const double previous_lidar_end = lio_sync_diag_.previous_lidar_end;
                    const double scan_gap = previous_lidar_end >= 0.0
                        ? Measures.lidar_beg_time - previous_lidar_end
                        : no_value;
                    const double selected_first = Measures.imu.empty()
                        ? no_value
                        : lio_sync_diag_.imu_first_selected;
                    const double selected_last = Measures.imu.empty()
                        ? no_value
                        : lio_sync_diag_.imu_last_selected;
                    const double queue_next = lio_sync_diag_.imu_queue_after == 0U
                        ? no_value
                        : lio_sync_diag_.imu_queue_front_after;
                    fprintf(stderr,
                        "[LIO SYNC] frame=%llu t=%.9f valid=%d "
                        "queues_before=[lidar=%zu imu=%zu] imu_queue_after=%zu "
                        "previous_lidar_end=%.9f scan_gap=%.9f "
                        "imu_queue=[front=%.9f back=%.9f next=%.9f] "
                        "selected=%zu before_scan=%zu inside_scan=%zu "
                        "selected_time=[%.9f %.9f] point_offset_ms=[last=%.6f max=%.6f] "
                        "scan_dt=%.9f end_source=%s\n",
                        static_cast<unsigned long long>(current_lio_frame_id_),
                        current_lio_frame_time_, lio_sync_diag_.valid ? 1 : 0,
                        lio_sync_diag_.lidar_queue_before,
                        lio_sync_diag_.imu_queue_before,
                        lio_sync_diag_.imu_queue_after,
                        previous_lidar_end, scan_gap,
                        lio_sync_diag_.imu_queue_front_before,
                        lio_sync_diag_.imu_queue_back_before, queue_next,
                        lio_sync_diag_.imu_selected,
                        lio_sync_diag_.imu_before_lidar_begin,
                        lio_sync_diag_.imu_inside_lidar_scan,
                        selected_first, selected_last,
                        lio_sync_diag_.point_offset_last_ms,
                        lio_sync_diag_.point_offset_max_ms,
                        lio_sync_diag_.scan_duration_s,
                        lio_sync_diag_.used_max_point_time ? "max_point" : "fallback");

                    const double imu_beg = Measures.imu.empty()
                        ? std::numeric_limits<double>::quiet_NaN()
                        : Measures.imu.front()->timestamp;
                    const double imu_end = Measures.imu.empty()
                        ? std::numeric_limits<double>::quiet_NaN()
                        : Measures.imu.back()->timestamp;
                    fprintf(stderr,
                        "[LIO IMU] frame=%llu lidar=[%.9f %.9f] scan_dt=%.6f "
                        "imu_n=%zu imu=[%.9f %.9f] cover_head=%.6f cover_tail=%.6f "
                        "imu_dt_mean=%.9f min=%.9f max=%.9f nonpos=%zu "
                        "acc_mean=[%.9f %.9f %.9f] acc_norm_mean=%.9f std=%.9f "
                        "gyr_mean=[%.9f %.9f %.9f] gyr_norm_mean=%.9f max=%.9f\n",
                        static_cast<unsigned long long>(current_lio_frame_id_),
                        Measures.lidar_beg_time, Measures.lidar_end_time,
                        Measures.lidar_end_time - Measures.lidar_beg_time,
                        imu_batch_stats.count, imu_beg, imu_end,
                        imu_beg - Measures.lidar_beg_time,
                        imu_end - Measures.lidar_end_time,
                        imu_batch_stats.dt_mean, imu_batch_stats.dt_min,
                        imu_batch_stats.dt_max,
                        imu_batch_stats.nonpositive_dt_count,
                        imu_batch_stats.mean_acc.x(), imu_batch_stats.mean_acc.y(),
                        imu_batch_stats.mean_acc.z(), imu_batch_stats.mean_acc_norm,
                        imu_batch_stats.std_acc_norm,
                        imu_batch_stats.mean_gyr.x(), imu_batch_stats.mean_gyr.y(),
                        imu_batch_stats.mean_gyr.z(), imu_batch_stats.mean_gyr_norm,
                        imu_batch_stats.max_gyr_norm);
                }

                if (flg_first_scan)
                {
                    first_lidar_time        = Measures.lidar_beg_time;
                    p_imu->first_lidar_time = first_lidar_time;
                    flg_first_scan          = false;
                    if (trace_current_lio_frame_)
                    {
                        fprintf(stderr,
                            "[LIO SKIP] frame=%llu t=%.9f reason=first_lidar_frame\n",
                            static_cast<unsigned long long>(current_lio_frame_id_),
                            current_lio_frame_time_);
                    }
                    return;
                }

                const state_ikfom state_before_imu = kf.get_x();
                const auto covariance_before_imu = kf.get_P();
                p_imu->Process(Measures, kf, feats_undistort);
                state_point = kf.get_x();
                const state_ikfom state_after_imu = state_point;
                const auto covariance_after_imu = kf.get_P();
                pos_lid     = state_point.pos + state_point.rot * state_point.offset_T_L_I;

                if (trace_current_lio_frame_)
                {
                    if (Measures.imu.empty())
                    {
                        fprintf(stderr,
                            "[LIO ALERT] frame=%llu t=%.9f reason=empty_imu_batch "
                            "process_returned_early=1 undist_cloud_after_return=%zu "
                            "stale_cloud_reuse_risk=%d\n",
                            static_cast<unsigned long long>(current_lio_frame_id_),
                            current_lio_frame_time_, feats_undistort->size(),
                            feats_undistort->empty() ? 0 : 1);
                    }
                    const Vec3d rpy_before_imu = SO3ToEuler(state_before_imu.rot);
                    const Vec3d rpy_after_imu = SO3ToEuler(state_after_imu.rot);
                    fprintf(stderr,
                        "[LIO PRED] frame=%llu t=%.9f "
                        "pre_pos=[%.6f %.6f %.6f] pred_pos=[%.6f %.6f %.6f] "
                        "dpos=[%.6f %.6f %.6f] pre_v=[%.6f %.6f %.6f] "
                        "pred_v=[%.6f %.6f %.6f] pre_rpy_deg=[%.5f %.5f %.5f] "
                        "pred_rpy_deg=[%.5f %.5f %.5f] drpy_deg=[%.5f %.5f %.5f] "
                        "ba=[%.8f %.8f %.8f] bg=[%.8f %.8f %.8f] "
                        "grav=[%.6f %.6f %.6f] "
                        "sigma_pre_z_pitch_vz_baz=[%.6e %.6e %.6e %.6e] "
                        "sigma_pred_z_pitch_vz_baz=[%.6e %.6e %.6e %.6e]\n",
                        static_cast<unsigned long long>(current_lio_frame_id_),
                        current_lio_frame_time_,
                        state_before_imu.pos(0), state_before_imu.pos(1),
                        state_before_imu.pos(2), state_after_imu.pos(0),
                        state_after_imu.pos(1), state_after_imu.pos(2),
                        state_after_imu.pos(0) - state_before_imu.pos(0),
                        state_after_imu.pos(1) - state_before_imu.pos(1),
                        state_after_imu.pos(2) - state_before_imu.pos(2),
                        state_before_imu.vel(0), state_before_imu.vel(1),
                        state_before_imu.vel(2), state_after_imu.vel(0),
                        state_after_imu.vel(1), state_after_imu.vel(2),
                        rpy_before_imu(0), rpy_before_imu(1), rpy_before_imu(2),
                        rpy_after_imu(0), rpy_after_imu(1), rpy_after_imu(2),
                        rpy_after_imu(0) - rpy_before_imu(0),
                        rpy_after_imu(1) - rpy_before_imu(1),
                        rpy_after_imu(2) - rpy_before_imu(2),
                        state_after_imu.ba[0], state_after_imu.ba[1],
                        state_after_imu.ba[2], state_after_imu.bg[0],
                        state_after_imu.bg[1], state_after_imu.bg[2],
                        state_after_imu.grav[0], state_after_imu.grav[1],
                        state_after_imu.grav[2],
                        covarianceSigma(covariance_before_imu(2, 2)),
                        covarianceSigma(covariance_before_imu(4, 4)),
                        covarianceSigma(covariance_before_imu(14, 14)),
                        covarianceSigma(covariance_before_imu(20, 20)),
                        covarianceSigma(covariance_after_imu(2, 2)),
                        covarianceSigma(covariance_after_imu(4, 4)),
                        covarianceSigma(covariance_after_imu(14, 14)),
                        covarianceSigma(covariance_after_imu(20, 20)));

                    const CloudZStats undistorted_z_stats =
                        summarizeCloudZ(*feats_undistort);
                    const ImuProcess::Diagnostics& imu_diagnostics =
                        p_imu->diagnostics();
                    fprintf(stderr,
                        "[LIO DESKEW] frame=%llu t=%.9f valid=%d "
                        "raw_n=%zu raw_sample=%zu raw_z=[%.6f %.6f %.6f %.6f %.6f] "
                        "undist_n=%zu undist_sample=%zu undist_z=[%.6f %.6f %.6f %.6f %.6f] "
                        "predict_steps=%zu predict_dt_sum=%.9f min=%.9f max=%.9f "
                        "nonpos=%zu end_extrap_dt=%.9f last_lidar_end=%.9f "
                        "skipped_pairs=%zu acc_scale=%.9f "
                        "world_acc_z=[n=%zu mean=%.6f min=%.6f max=%.6f dt_int=%.9f dt=%.9f] "
                        "point_time=[n=%zu nonfinite=%zu outside=%zu min=%.9f max=%.9f] "
                        "deskew_n=%zu "
                        "deskew_dz_mean=%.6f rms=%.6f abs_p95=%.6f abs_max=%.6f "
                        "deskew_rot_dz=[mean=%.6f rms=%.6f abs_p95=%.6f abs_max=%.6f] "
                        "deskew_trans_dz=[mean=%.6f rms=%.6f abs_p95=%.6f abs_max=%.6f] "
                        "deskew_peak=[point_t=%.9f input_z=%.6f output_z=%.6f] "
                        "last_acc_input=[%.6f %.6f %.6f] "
                        "last_gyr_input=[%.6f %.6f %.6f] "
                        "last_world_acc=[%.6f %.6f %.6f]\n",
                        static_cast<unsigned long long>(current_lio_frame_id_),
                        current_lio_frame_time_, imu_diagnostics.valid ? 1 : 0,
                        raw_z_stats.count, raw_z_stats.sample_count,
                        raw_z_stats.min, raw_z_stats.p05, raw_z_stats.p50,
                        raw_z_stats.p95, raw_z_stats.max,
                        undistorted_z_stats.count,
                        undistorted_z_stats.sample_count,
                        undistorted_z_stats.min, undistorted_z_stats.p05,
                        undistorted_z_stats.p50, undistorted_z_stats.p95,
                        undistorted_z_stats.max,
                        imu_diagnostics.predict_steps,
                        imu_diagnostics.predict_dt_sum,
                        imu_diagnostics.predict_dt_min,
                        imu_diagnostics.predict_dt_max,
                        imu_diagnostics.nonpositive_dt_count,
                        imu_diagnostics.end_extrapolation_dt,
                        imu_diagnostics.last_lidar_end_before,
                        imu_diagnostics.skipped_imu_pair_count,
                        imu_diagnostics.acc_scale_factor,
                        imu_diagnostics.world_acc_sample_count,
                        imu_diagnostics.world_acc_z_mean,
                        imu_diagnostics.world_acc_z_min,
                        imu_diagnostics.world_acc_z_max,
                        imu_diagnostics.world_acc_z_dt_integral,
                        imu_diagnostics.positive_predict_dt_sum,
                        imu_diagnostics.point_time_valid_count,
                        imu_diagnostics.point_time_nonfinite_count,
                        imu_diagnostics.point_time_outside_scan_count,
                        imu_diagnostics.point_time_min,
                        imu_diagnostics.point_time_max,
                        imu_diagnostics.deskew_point_count,
                        imu_diagnostics.deskew_dz_mean,
                        imu_diagnostics.deskew_dz_rms,
                        imu_diagnostics.deskew_dz_abs_p95,
                        imu_diagnostics.deskew_dz_abs_max,
                        imu_diagnostics.deskew_rotation_dz_mean,
                        imu_diagnostics.deskew_rotation_dz_rms,
                        imu_diagnostics.deskew_rotation_dz_abs_p95,
                        imu_diagnostics.deskew_rotation_dz_abs_max,
                        imu_diagnostics.deskew_translation_dz_mean,
                        imu_diagnostics.deskew_translation_dz_rms,
                        imu_diagnostics.deskew_translation_dz_abs_p95,
                        imu_diagnostics.deskew_translation_dz_abs_max,
                        imu_diagnostics.deskew_peak_point_time,
                        imu_diagnostics.deskew_peak_input_z,
                        imu_diagnostics.deskew_peak_output_z,
                        imu_diagnostics.last_acc_input.x(),
                        imu_diagnostics.last_acc_input.y(),
                        imu_diagnostics.last_acc_input.z(),
                        imu_diagnostics.last_gyr_input.x(),
                        imu_diagnostics.last_gyr_input.y(),
                        imu_diagnostics.last_gyr_input.z(),
                        imu_diagnostics.last_world_acc.x(),
                        imu_diagnostics.last_world_acc.y(),
                        imu_diagnostics.last_world_acc.z());
                }

                if (feats_undistort->empty())
                {
                    if (trace_current_lio_frame_)
                    {
                        fprintf(stderr,
                            "[LIO SKIP] frame=%llu t=%.9f reason=imu_initializing_or_empty_cloud\n",
                            static_cast<unsigned long long>(current_lio_frame_id_),
                            current_lio_frame_time_);
                    }
                    RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                    return;
                }

                flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
                lasermap_fov_segment();

                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                edge_point_count = 0;

                // VoxelGrid averages every field within a voxel, so running it
                // over a mixed cloud would blend the FeatureLabel in normal_x
                // into a fraction that matches no class. Split by label, filter
                // each subset on its own, then re-tag and merge.
                if (edge_constraint_enable && p_pre->feature_enabled)
                {
                    feats_surf_raw->clear();
                    feats_edge_raw->clear();
                    feats_surf_raw->reserve(feats_undistort->size());
                    feats_edge_raw->reserve(feats_undistort->size() / 8 + 1);
                    for (const auto& pt : feats_undistort->points)
                    {
                        const int lbl = static_cast<int>(pt.normal_x);
                        if (lbl == static_cast<int>(FeatureLabel::Edge) || lbl == static_cast<int>(FeatureLabel::Wire))
                        {
                            feats_edge_raw->push_back(pt);
                        }
                        else
                        {
                            feats_surf_raw->push_back(pt);
                        }
                    }

                    if (!feats_edge_raw->empty())
                    {
                        downSizeFilterSurf.setInputCloud(feats_surf_raw);
                        downSizeFilterSurf.filter(*feats_surf_ds);
                        downSizeFilterEdge.setInputCloud(feats_edge_raw);
                        downSizeFilterEdge.filter(*feats_edge_ds);

                        // Restore the exact label values that voxel averaging blurred.
                        for (auto& pt : feats_surf_ds->points)
                            pt.normal_x = static_cast<float>(FeatureLabel::Surface);
                        for (auto& pt : feats_edge_ds->points)
                            pt.normal_x = static_cast<float>(FeatureLabel::Edge);

                        feats_down_body->clear();
                        feats_down_body->reserve(feats_surf_ds->size() + feats_edge_ds->size());
                        *feats_down_body += *feats_surf_ds;
                        *feats_down_body += *feats_edge_ds;
                        edge_point_count = static_cast<int>(feats_edge_ds->size());
                    }
                }
                feats_down_size = feats_down_body->points.size();
                if (ikdtree.Root_Node == nullptr)
                {
                    RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                    if (feats_down_size > 5)
                    {
                        ikdtree.set_downsample_param(filter_size_map_min);
                        feats_down_world->resize(feats_down_size);
                        for (int i = 0; i < feats_down_size; i++)
                        {
                            pointsBody2World(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                        }
                        ikdtree.Build(feats_down_world->points);
                    }
                    if (trace_current_lio_frame_)
                    {
                        fprintf(stderr,
                            "[LIO SKIP] frame=%llu t=%.9f reason=map_kdtree_init "
                            "undist_n=%zu down_n=%d edge_down=%d\n",
                            static_cast<unsigned long long>(current_lio_frame_id_),
                            current_lio_frame_time_, feats_undistort->size(),
                            feats_down_size, edge_point_count);
                    }
                    return;
                }
                if (feats_down_size < 5)
                {
                    if (trace_current_lio_frame_)
                    {
                        fprintf(stderr,
                            "[LIO SKIP] frame=%llu t=%.9f reason=too_few_downsampled_points "
                            "undist_n=%zu down_n=%d edge_down=%d\n",
                            static_cast<unsigned long long>(current_lio_frame_id_),
                            current_lio_frame_time_, feats_undistort->size(),
                            feats_down_size, edge_point_count);
                    }
                    RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                    return;
                }

                normvec->resize(feats_down_size);
                feats_down_world->resize(feats_down_size);

                Vec3d ext_euler = SO3ToEuler(state_point.offset_R_L_I);

                if (0)  // If you need to see map point, change to "if(1)"
                {
                    PointVector().swap(ikdtree.PCL_Storage);
                    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                    featsFromMap->clear();
                    featsFromMap->points = ikdtree.PCL_Storage;
                }

                pointSearchInd_surf.resize(feats_down_size);
                Nearest_Points.resize(feats_down_size);
                int  rematch_num       = 0;
                bool nearest_search_en = true;  //

                double solve_H_time = 0;
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
                state_point = kf.get_x();
                const state_ikfom state_after_lidar = state_point;
                const auto covariance_after_lidar = kf.get_P();
                euler_cur   = SO3ToEuler(state_point.rot);
                pos_lid     = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                geoQuat.x   = state_point.rot.coeffs()[0];
                geoQuat.y   = state_point.rot.coeffs()[1];
                geoQuat.z   = state_point.rot.coeffs()[2];
                geoQuat.w   = state_point.rot.coeffs()[3];

                if (trace_current_lio_frame_)
                {
                    const Vec3d rpy_after_imu = SO3ToEuler(state_after_imu.rot);
                    const Vec3d rpy_after_lidar = SO3ToEuler(state_after_lidar.rot);
                    const Vec3d ext_rpy_after_imu =
                        SO3ToEuler(state_after_imu.offset_R_L_I);
                    const Vec3d ext_rpy_after_lidar =
                        SO3ToEuler(state_after_lidar.offset_R_L_I);
                    const Vec3d pred_gravity(
                        state_after_imu.grav[0], state_after_imu.grav[1],
                        state_after_imu.grav[2]);
                    const Vec3d post_gravity(
                        state_after_lidar.grav[0], state_after_lidar.grav[1],
                        state_after_lidar.grav[2]);
                    const double gravity_norm_product =
                        pred_gravity.norm() * post_gravity.norm();
                    const double gravity_update_angle_deg = gravity_norm_product > 0.0
                        ? std::acos(std::clamp(
                            pred_gravity.dot(post_gravity) / gravity_norm_product,
                            -1.0, 1.0)) * 180.0 / PI_M
                        : 0.0;
                    fprintf(stderr,
                        "[LIO UPDATE] frame=%llu t=%.9f solve_h_s=%.9f "
                        "pred_pos=[%.6f %.6f %.6f] post_pos=[%.6f %.6f %.6f] "
                        "lidar_dpos=[%.6f %.6f %.6f] pred_v=[%.6f %.6f %.6f] "
                        "post_v=[%.6f %.6f %.6f] pred_rpy_deg=[%.5f %.5f %.5f] "
                        "post_rpy_deg=[%.5f %.5f %.5f] lidar_drpy_deg=[%.5f %.5f %.5f] "
                        "sigma_pred_z_pitch_vz_baz=[%.6e %.6e %.6e %.6e] "
                        "sigma_post_z_pitch_vz_baz=[%.6e %.6e %.6e %.6e] "
                        "post_ba=[%.8f %.8f %.8f] post_bg=[%.8f %.8f %.8f] "
                        "pred_grav=[%.6f %.6f %.6f] post_grav=[%.6f %.6f %.6f] "
                        "gravity_lidar_update_deg=%.9f "
                        "gravity_cov=[%.9e %.9e] "
                        "ext_T=[%.8f %.8f %.8f] ext_dT=[%.8f %.8f %.8f] "
                        "ext_rpy_deg=[%.6f %.6f %.6f] ext_drpy_deg=[%.6f %.6f %.6f]\n",
                        static_cast<unsigned long long>(current_lio_frame_id_),
                        current_lio_frame_time_, solve_H_time,
                        state_after_imu.pos(0), state_after_imu.pos(1),
                        state_after_imu.pos(2), state_after_lidar.pos(0),
                        state_after_lidar.pos(1), state_after_lidar.pos(2),
                        state_after_lidar.pos(0) - state_after_imu.pos(0),
                        state_after_lidar.pos(1) - state_after_imu.pos(1),
                        state_after_lidar.pos(2) - state_after_imu.pos(2),
                        state_after_imu.vel(0), state_after_imu.vel(1),
                        state_after_imu.vel(2), state_after_lidar.vel(0),
                        state_after_lidar.vel(1), state_after_lidar.vel(2),
                        rpy_after_imu(0), rpy_after_imu(1), rpy_after_imu(2),
                        rpy_after_lidar(0), rpy_after_lidar(1), rpy_after_lidar(2),
                        rpy_after_lidar(0) - rpy_after_imu(0),
                        rpy_after_lidar(1) - rpy_after_imu(1),
                        rpy_after_lidar(2) - rpy_after_imu(2),
                        covarianceSigma(covariance_after_imu(2, 2)),
                        covarianceSigma(covariance_after_imu(4, 4)),
                        covarianceSigma(covariance_after_imu(14, 14)),
                        covarianceSigma(covariance_after_imu(20, 20)),
                        covarianceSigma(covariance_after_lidar(2, 2)),
                        covarianceSigma(covariance_after_lidar(4, 4)),
                        covarianceSigma(covariance_after_lidar(14, 14)),
                        covarianceSigma(covariance_after_lidar(20, 20)),
                        state_after_lidar.ba[0], state_after_lidar.ba[1],
                        state_after_lidar.ba[2], state_after_lidar.bg[0],
                        state_after_lidar.bg[1], state_after_lidar.bg[2],
                        state_after_imu.grav[0], state_after_imu.grav[1],
                        state_after_imu.grav[2], state_after_lidar.grav[0],
                        state_after_lidar.grav[1], state_after_lidar.grav[2],
                        gravity_update_angle_deg,
                        covariance_after_lidar(21, 21),
                        covariance_after_lidar(22, 22),
                        state_after_lidar.offset_T_L_I(0),
                        state_after_lidar.offset_T_L_I(1),
                        state_after_lidar.offset_T_L_I(2),
                        state_after_lidar.offset_T_L_I(0) -
                            state_after_imu.offset_T_L_I(0),
                        state_after_lidar.offset_T_L_I(1) -
                            state_after_imu.offset_T_L_I(1),
                        state_after_lidar.offset_T_L_I(2) -
                            state_after_imu.offset_T_L_I(2),
                        ext_rpy_after_lidar(0), ext_rpy_after_lidar(1),
                        ext_rpy_after_lidar(2),
                        ext_rpy_after_lidar(0) - ext_rpy_after_imu(0),
                        ext_rpy_after_lidar(1) - ext_rpy_after_imu(1),
                        ext_rpy_after_lidar(2) - ext_rpy_after_imu(2));

                    fprintf(stderr,
                        "[LIO MATCH] frame=%llu t=%.9f calls=%d valid=%d "
                        "down=%d edge_down=%d eff_initial=%d eff_final=%d edge_eff=%d "
                        "res_initial=[mean=%.6f abs_mean=%.6f abs_p95=%.6f] "
                        "res_final=[mean=%.6f abs_mean=%.6f abs_p95=%.6f abs_max=%.6f] "
                        "normals_final=[floor=%d wall=%d mixed=%d] "
                        "group_residual=[floor_mean=%.6f floor_abs_mean=%.6f "
                        "wall_mean=%.6f wall_abs_mean=%.6f] "
                        "z_constraint=[information=%.6e rhs=%.6e only_step=%.6f] "
                        "model_z_pitch=[%.6f %.5f -> %.6f %.5f] "
                        "cond=[%.6e -> %.6e] weak_initial=[%.4f %.4f %.4f %.4f %.4f %.4f] "
                        "weak_final=[%.4f %.4f %.4f %.4f %.4f %.4f] "
                        "hdiag_pos_rpy=[%.6e %.6e %.6e %.6e %.6e %.6e] "
                        "eig_norm=[%.6e %.6e %.6e %.6e %.6e %.6e]\n",
                        static_cast<unsigned long long>(current_lio_frame_id_),
                        current_lio_frame_time_, lio_match_diag_.model_calls,
                        lio_match_diag_.valid ? 1 : 0, feats_down_size,
                        edge_point_count, lio_match_diag_.initial_effective,
                        lio_match_diag_.final_effective,
                        lio_match_diag_.final_edge_effective,
                        lio_match_diag_.initial_residual_mean,
                        lio_match_diag_.initial_residual_abs_mean,
                        lio_match_diag_.initial_residual_abs_p95,
                        lio_match_diag_.final_residual_mean,
                        lio_match_diag_.final_residual_abs_mean,
                        lio_match_diag_.final_residual_abs_p95,
                        lio_match_diag_.final_residual_abs_max,
                        lio_match_diag_.final_floor_normals,
                        lio_match_diag_.final_wall_normals,
                        lio_match_diag_.final_mixed_normals,
                        lio_match_diag_.final_floor_residual_mean,
                        lio_match_diag_.final_floor_residual_abs_mean,
                        lio_match_diag_.final_wall_residual_mean,
                        lio_match_diag_.final_wall_residual_abs_mean,
                        lio_match_diag_.final_z_information,
                        lio_match_diag_.final_z_rhs,
                        lio_match_diag_.final_z_only_step,
                        lio_match_diag_.initial_state_z,
                        lio_match_diag_.initial_state_pitch_deg,
                        lio_match_diag_.final_model_state_z,
                        lio_match_diag_.final_model_pitch_deg,
                        lio_match_diag_.initial_condition,
                        lio_match_diag_.final_condition,
                        lio_match_diag_.initial_weak(0),
                        lio_match_diag_.initial_weak(1),
                        lio_match_diag_.initial_weak(2),
                        lio_match_diag_.initial_weak(3),
                        lio_match_diag_.initial_weak(4),
                        lio_match_diag_.initial_weak(5),
                        lio_match_diag_.final_weak(0),
                        lio_match_diag_.final_weak(1),
                        lio_match_diag_.final_weak(2),
                        lio_match_diag_.final_weak(3),
                        lio_match_diag_.final_weak(4),
                        lio_match_diag_.final_weak(5),
                        lio_match_diag_.final_hessian_diag(0),
                        lio_match_diag_.final_hessian_diag(1),
                        lio_match_diag_.final_hessian_diag(2),
                        lio_match_diag_.final_hessian_diag(3),
                        lio_match_diag_.final_hessian_diag(4),
                        lio_match_diag_.final_hessian_diag(5),
                        lio_match_diag_.final_normalized_eigenvalues(0),
                        lio_match_diag_.final_normalized_eigenvalues(1),
                        lio_match_diag_.final_normalized_eigenvalues(2),
                        lio_match_diag_.final_normalized_eigenvalues(3),
                        lio_match_diag_.final_normalized_eigenvalues(4),
                        lio_match_diag_.final_normalized_eigenvalues(5));
                }
                map_incremental();
                if (trace_current_lio_frame_)
                {
                    const CloudZStats body_down_z_stats =
                        summarizeCloudZ(*feats_down_body);
                    const CloudZStats world_down_z_stats =
                        summarizeCloudZ(*feats_down_world);
                    fprintf(stderr,
                        "[LIO WORLD] frame=%llu t=%.9f state_z=%.6f lidar_z=%.6f "
                        "body_down_n=%zu body_z=[%.6f %.6f %.6f %.6f %.6f] "
                        "world_down_n=%zu world_z=[%.6f %.6f %.6f %.6f %.6f]\n",
                        static_cast<unsigned long long>(current_lio_frame_id_),
                        current_lio_frame_time_, state_point.pos(2), pos_lid(2),
                        body_down_z_stats.count, body_down_z_stats.min,
                        body_down_z_stats.p05, body_down_z_stats.p50,
                        body_down_z_stats.p95, body_down_z_stats.max,
                        world_down_z_stats.count, world_down_z_stats.min,
                        world_down_z_stats.p05, world_down_z_stats.p50,
                        world_down_z_stats.p95, world_down_z_stats.max);
                }
                maybeAddKeyframe();

                publish_odometry(pubOdomAftMapped_, tf_broadcaster_);
                pubWorldPoints(pubLaserCloudFull_);

                if (path_en)
                    publish_path(pubPath_);
                if (pub_body_points_flag_)
                    pubBodyPoints(pubLaserCloudFull_body_);
            }
        }
        else if (state_.load() == SlamState::SAVE)
        {
            const bool save_ok = finish();
            state_.store(
                (!save_ok && lio_time_guard_session_.failed)
                    ? SlamState::ERROR : SlamState::READY);
        }
        else
        {
            return;
        }
    }

    void MappingAlg::map_publish_callback()
    {
        if (map_pub_en)
            pubMapPoints(pubLaserCloudMap_);
    }

    void MappingAlg::aprilTagDetectionCallBack(const apriltag_msgs::msg::AprilTagDetectionArray::UniquePtr msg)
    {
        if (msg->detections.empty())
        {
            return;
        }

        // Nearest-keyframe association by timestamp, once per incoming image
        // frame (not per marker) -- keyframe_store_ is small enough (default
        // max_count 5000) that a linear scan here is negligible next to
        // camera-rate frames; snapshot() is only taken when markers are
        // actually visible, not per LIO frame.
        const double t = get_time_sec(msg->header.stamp);
        const KeyframeStore::Snapshot snapshot = keyframe_store_->snapshot();
        if (snapshot.empty())
        {
            return;
        }
        size_t best_idx = 0U;
        double best_dt  = std::abs(snapshot[0].stamp - t);
        for (size_t i = 1U; i < snapshot.size(); ++i)
        {
            const double dt = std::abs(snapshot[i].stamp - t);
            if (dt < best_dt)
            {
                best_dt  = dt;
                best_idx = i;
            }
        }
        if (best_dt > apriltag_max_keyframe_time_gap_s_)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "apriltag frame at t=%.3f: nearest keyframe is %.3fs away (> %.3fs), discarding %zu detection(s)",
                t, best_dt, apriltag_max_keyframe_time_gap_s_, msg->detections.size());
            return;
        }
        const uint64_t keyframe_id = snapshot[best_idx].id;

        for (const auto& det : msg->detections)
        {
            if (det.family != apriltag_family_)
            {
                // A different tag family in view (or a stray decode) --
                // not this deployment's anchor set.
                continue;
            }
            if (det.hamming > apriltag_max_hamming_)
            {
                RCLCPP_DEBUG(
                    this->get_logger(),
                    "apriltag id=%d: hamming=%d exceeds max %d, dropping",
                    det.id, det.hamming, apriltag_max_hamming_);
                continue;
            }
            if (det.decision_margin < apriltag_min_decision_margin_)
            {
                RCLCPP_DEBUG(
                    this->get_logger(),
                    "apriltag id=%d: decision_margin=%.1f below min %.1f, dropping",
                    det.id, det.decision_margin, apriltag_min_decision_margin_);
                continue;
            }

            // [PATCH -- AprilTag migration] apriltag_ros publishes the tag
            // pose on /tf (child_frame_id "tag<family>:<id>"), not as a
            // field on this message -- so T_lidar_marker comes from a TF
            // lookup, composed by tf2 across
            // livox_frame -> ... -> base_link -> ... -> camera_color_optical_frame
            // -> tag<family>:<id>, using anubis_description's static
            // chain plus whatever the camera driver publishes at runtime
            // for its own optical-frame sub-tree. No hand-maintained
            // extrinsic matrix is involved.
            const std::string tag_frame = "tag" + apriltag_family_ + ":" + std::to_string(det.id);
            geometry_msgs::msg::TransformStamped transform;
            try
            {
                transform = tf_buffer_->lookupTransform(
                    apriltag_lidar_frame_id_, tag_frame, msg->header.stamp,
                    rclcpp::Duration::from_seconds(0.1));
            }
            catch (const tf2::TransformException& ex)
            {
                RCLCPP_DEBUG(
                    this->get_logger(), "apriltag id=%d: TF lookup %s -> %s failed: %s",
                    det.id, apriltag_lidar_frame_id_.c_str(), tag_frame.c_str(), ex.what());
                continue;
            }

            Eigen::Matrix4d T_lidar_marker = Eigen::Matrix4d::Identity();
            T_lidar_marker(0, 3) = transform.transform.translation.x;
            T_lidar_marker(1, 3) = transform.transform.translation.y;
            T_lidar_marker(2, 3) = transform.transform.translation.z;
            Eigen::Quaterniond q(
                transform.transform.rotation.w, transform.transform.rotation.x,
                transform.transform.rotation.y, transform.transform.rotation.z);
            if (!q.coeffs().allFinite() || q.norm() < 1e-6)
            {
                RCLCPP_WARN(this->get_logger(), "apriltag id=%d: non-finite orientation from TF, discarding", det.id);
                continue;
            }
            T_lidar_marker.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();

            AprilTagObservation obs;
            obs.marker_id      = det.id;
            obs.keyframe_id    = keyframe_id;
            obs.T_lidar_marker = T_lidar_marker;

            {
                std::lock_guard<std::mutex> lock(apriltag_mtx_);
                apriltag_observations_.push_back(obs);
            }

            // Optional live debug TF, so a validated-looking detection is
            // visible in RViz while mapping is still in progress -- this is
            // NOT the final map-frame anchor pose (that only exists after
            // save-time resolution against the loop-closure-corrected
            // keyframe pose in saveAprilTagMap()); it is the raw per-sighting
            // pose relative to whatever the online LIO pose happens to be
            // right now, purely for visual sanity-checking during capture.
            if (apriltag_publish_debug_tf_ && tf_broadcaster_)
            {
                const Eigen::Matrix4d T_odom_lidar = currentLidarPose();
                const Eigen::Matrix4d T_odom_marker = T_odom_lidar * T_lidar_marker;
                Eigen::Quaterniond debug_q(Eigen::Matrix3d(T_odom_marker.block<3, 3>(0, 0)));
                debug_q.normalize();

                geometry_msgs::msg::TransformStamped debug_tf;
                debug_tf.header.stamp = msg->header.stamp;
                debug_tf.header.frame_id = "odom";
                debug_tf.child_frame_id = "apriltag_debug_" + std::to_string(det.id);
                debug_tf.transform.translation.x = T_odom_marker(0, 3);
                debug_tf.transform.translation.y = T_odom_marker(1, 3);
                debug_tf.transform.translation.z = T_odom_marker(2, 3);
                debug_tf.transform.rotation.x = debug_q.x();
                debug_tf.transform.rotation.y = debug_q.y();
                debug_tf.transform.rotation.z = debug_q.z();
                debug_tf.transform.rotation.w = debug_q.w();
                tf_broadcaster_->sendTransform(debug_tf);
            }
        }
    }

    bool MappingAlg::saveAprilTagMap(
        const KeyframeStore::Snapshot& snapshot,
        const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& optimized_poses,
        const std::string& output_dir)
    {
        std::vector<AprilTagObservation> observations;
        {
            std::lock_guard<std::mutex> lock(apriltag_mtx_);
            observations = apriltag_observations_;
        }
        if (observations.empty())
        {
            // Nothing detected during this session is not a save failure.
            return true;
        }
        if (optimized_poses.size() != snapshot.size())
        {
            RCLCPP_ERROR(
                this->get_logger(),
                "saveAprilTagMap: optimized_poses/snapshot size mismatch (%zu vs %zu), skipping apriltag_anchors.yaml",
                optimized_poses.size(), snapshot.size());
            return false;
        }

        std::unordered_map<uint64_t, size_t> id_to_index;
        id_to_index.reserve(snapshot.size());
        for (size_t i = 0U; i < snapshot.size(); ++i)
        {
            id_to_index.emplace(snapshot[i].id, i);
        }

        struct Agg
        {
            std::vector<Eigen::Vector3d> positions;
            std::vector<Eigen::Vector4d> quats_xyzw;
        };
        std::map<int, Agg> per_marker;

        size_t dropped_unmatched = 0U;
        for (const auto& obs : observations)
        {
            const auto it = id_to_index.find(obs.keyframe_id);
            if (it == id_to_index.end())
            {
                // Keyframe was never part of the final snapshot (e.g. session
                // reset between detection and save); skip rather than guess.
                ++dropped_unmatched;
                continue;
            }
            const Eigen::Matrix4d T_map_marker = optimized_poses[it->second] * obs.T_lidar_marker;
            if (!finiteRigidPose(T_map_marker))
            {
                continue;
            }
            Eigen::Quaterniond q(Eigen::Matrix3d(T_map_marker.block<3, 3>(0, 0)));
            q.normalize();
            auto& agg = per_marker[obs.marker_id];
            agg.positions.emplace_back(T_map_marker.block<3, 1>(0, 3));
            agg.quats_xyzw.emplace_back(q.x(), q.y(), q.z(), q.w());
        }
        if (dropped_unmatched > 0U)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "saveAprilTagMap: %zu observation(s) referenced a keyframe outside the final "
                "snapshot and were dropped", dropped_unmatched);
        }
        if (per_marker.empty())
        {
            RCLCPP_WARN(this->get_logger(), "saveAprilTagMap: no observation resolved to a valid pose, skipping apriltag_anchors.yaml");
            return true;
        }

        // [PATCH -- AprilTag migration] Filename changed from aruco.yaml to
        // apriltag_anchors.yaml; schema (frame_id/markers list with
        // id/position/orientation/observation_count/max_position_deviation_m/
        // validated) is unchanged from darmawan_ws's aruco.yaml on purpose,
        // so anubis_localization's consumer is a straightforward port of
        // the equivalent loader, not a new format to design.
        const std::string apriltag_path = output_dir + "/apriltag_anchors.yaml";
        std::ofstream    ofs(apriltag_path);
        if (!ofs.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "saveAprilTagMap: cannot open %s for writing", apriltag_path.c_str());
            return false;
        }
        ofs << "frame_id: \"map\"\n";
        ofs << "tag_family: \"" << apriltag_family_ << "\"\n";
        ofs << "markers:\n";
        for (auto& [id, agg] : per_marker)
        {
            Eigen::Vector3d mean_p = Eigen::Vector3d::Zero();
            for (const auto& p : agg.positions)
                mean_p += p;
            mean_p /= static_cast<double>(agg.positions.size());

            double max_dev = 0.0;
            for (const auto& p : agg.positions)
                max_dev = std::max(max_dev, (p - mean_p).norm());

            // Component-wise quaternion average (hemisphere-aligned then
            // renormalized). Adequate for the small angular spread expected
            // across viewing angles of one static, wall-mounted marker; this
            // is a diagnostic anchor pose, not a metrology result.
            Eigen::Vector4d qacc = Eigen::Vector4d::Zero();
            for (auto qv : agg.quats_xyzw)
            {
                if (qacc.dot(qv) < 0.0)
                    qv = -qv;
                qacc += qv;
            }
            qacc.normalize();

            const bool validated =
                static_cast<int>(agg.positions.size()) >= apriltag_min_observations_ &&
                max_dev <= apriltag_max_position_deviation_m_;
            if (!validated)
            {
                RCLCPP_WARN(
                    this->get_logger(),
                    "apriltag id=%d: %zu observation(s), max_dev=%.4fm -> validated=false "
                    "(need >=%d obs and <=%.3fm)",
                    id, agg.positions.size(), max_dev, apriltag_min_observations_,
                    apriltag_max_position_deviation_m_);
            }

            ofs << "  - id: " << id << "\n";
            ofs << "    position: {x: " << mean_p.x() << ", y: " << mean_p.y()
                << ", z: " << mean_p.z() << "}\n";
            ofs << "    orientation: {x: " << qacc.x() << ", y: " << qacc.y()
                << ", z: " << qacc.z() << ", w: " << qacc.w() << "}\n";
            ofs << "    observation_count: " << agg.positions.size() << "\n";
            ofs << "    max_position_deviation_m: " << max_dev << "\n";
            ofs << "    validated: " << (validated ? "true" : "false") << "\n";
        }
        ofs.flush();
        if (!ofs.good())
        {
            RCLCPP_ERROR(this->get_logger(), "saveAprilTagMap: write error on %s", apriltag_path.c_str());
            return false;
        }
        RCLCPP_INFO(
            this->get_logger(), "apriltag_anchors.yaml written: %s (%zu marker id(s))",
            apriltag_path.c_str(), per_marker.size());
        return true;
    }

    bool MappingAlg::correctKeyframeZForSave(
        const KeyframeStore::Snapshot& snapshot,
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& optimized_poses,
        PointCloudType& map_source)
    {
        map_z_observations_.clear();
        map_z_correction_result_ = MapZDriftCorrectionResult{};
        map_z_final_floor_span_ = 0.0;
        if (!map_z_correction_enabled_)
        {
            return true;
        }
        if (snapshot.size() < 3U || optimized_poses.size() != snapshot.size())
        {
            RCLCPP_ERROR(
                get_logger(),
                "Map Z correction failed: keyframe clouds and final poses are unavailable");
            return false;
        }

        map_z_observations_.reserve(snapshot.size());
        std::vector<double> path_coordinates(snapshot.size(), 0.0);
        for (std::size_t index = 1U; index < snapshot.size(); ++index)
        {
            const double step =
                (optimized_poses[index].block<2, 1>(0, 3) -
                 optimized_poses[index - 1U].block<2, 1>(0, 3)).norm();
            if (!std::isfinite(step))
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map Z correction failed: non-finite path step at index %zu",
                    index);
                return false;
            }
            path_coordinates[index] = path_coordinates[index - 1U] + step;
        }
        const double min_range_sq =
            map_z_local_min_range_ * map_z_local_min_range_;
        const double max_range_sq =
            map_z_local_max_range_ * map_z_local_max_range_;
        const double max_source_distance_sq =
            map_z_local_max_source_distance_ *
            map_z_local_max_source_distance_;
        GroundPlaneFitOptions local_options;
        local_options.fixed_floor_z = keyframe_floor_z_map_;
        local_options.candidate_min_height = map_z_local_candidate_min_height_;
        local_options.candidate_max_height = map_z_local_candidate_max_height_;
        local_options.sample_cell_size = 0.25;
        local_options.min_points_per_cell = 1;
        local_options.sample_quantile = map_leveling_sample_quantile_;
        local_options.min_sample_cells = static_cast<std::size_t>(
            map_z_local_min_sample_cells_);
        local_options.min_xy_span = 1.0;
        local_options.min_inlier_ratio = 0.45;
        local_options.max_tilt_deg = map_z_local_max_tilt_deg_;
        local_options.max_residual_p95 = map_z_local_max_residual_p95_;
        local_options.max_floor_offset = std::max(
            std::abs(map_z_local_candidate_min_height_),
            std::abs(map_z_local_candidate_max_height_));
        local_options.min_inlier_span_fraction = 0.25;
        local_options.min_inlier_minor_span = 0.10;
        local_options.quadratic_enabled = false;

        struct LocalGroundCandidate
        {
            GroundPlaneModel model;
            bool valid = false;
            int window = 0;
            std::size_t source_count = 0U;
            std::size_t independent_source_count = 0U;
            double support_s_min = 0.0;
            double support_s_max = 0.0;
            std::vector<MapZSupportInterval> support_intervals;
            std::string source_ids;
            std::string failure_reason;
            double score = std::numeric_limits<double>::infinity();
        };
        struct SourceGroundEvidence
        {
            std::size_t source = 0U;
            bool considered = false;
            bool independently_supported = false;
            std::vector<GroundPlanePoint> points;
        };

        // Each retained point keeps its source keyframe until that source has
        // independently supplied enough candidate cells.  This prevents one
        // dense scan, or a temporally near but spatially remote revisit, from
        // being counted repeatedly as path support.
        const auto collect_source_evidence = [&](
            std::size_t center, std::size_t source)
        {
            SourceGroundEvidence evidence;
            evidence.source = source;
            const double sensor_x = optimized_poses[center](0, 3);
            const double sensor_y = optimized_poses[center](1, 3);
            using Cell = std::pair<std::int64_t, std::int64_t>;
            if (!finiteRigidPose(optimized_poses[source]) ||
                !snapshot[source].dense_cloud_lidar ||
                snapshot[source].dense_cloud_lidar->empty())
            {
                return evidence;
            }
            const double source_dx = optimized_poses[source](0, 3) - sensor_x;
            const double source_dy = optimized_poses[source](1, 3) - sensor_y;
            if (source_dx * source_dx + source_dy * source_dy >
                max_source_distance_sq)
            {
                return evidence;
            }
            evidence.considered = true;
            PointCloudType transformed;
            pcl::transformPointCloud(
                *snapshot[source].dense_cloud_lidar, transformed,
                optimized_poses[source].cast<float>());
            evidence.points.reserve(transformed.size());
            std::set<Cell> candidate_cells;
            for (const auto& point : transformed.points)
            {
                if (!finitePointXyz(point))
                {
                    continue;
                }
                const double dx = static_cast<double>(point.x) - sensor_x;
                const double dy = static_cast<double>(point.y) - sensor_y;
                const double range_sq = dx * dx + dy * dy;
                if (range_sq < min_range_sq || range_sq > max_range_sq)
                {
                    continue;
                }
                const double height =
                    static_cast<double>(point.z) - keyframe_floor_z_map_;
                if (height >= map_z_local_candidate_min_height_ &&
                    height <= map_z_local_candidate_max_height_)
                {
                    // Keep the local fit source-pure: points outside the
                    // configured candidate height band are never ground
                    // evidence.  In particular, low returns from nearby
                    // walls/obstacles used to be appended before this check
                    // and could bias a sparse source's plane estimate.
                    evidence.points.push_back({point.x, point.y, point.z});
                    candidate_cells.emplace(
                        static_cast<std::int64_t>(std::floor(
                            static_cast<double>(point.x) /
                            local_options.sample_cell_size)),
                        static_cast<std::int64_t>(std::floor(
                            static_cast<double>(point.y) /
                            local_options.sample_cell_size)));
                }
            }
            evidence.independently_supported =
                candidate_cells.size() >= static_cast<std::size_t>(
                    map_z_local_min_source_sample_cells_);
            return evidence;
        };
        const auto fit_local_ground = [&](
            std::size_t center, int window,
            const std::vector<SourceGroundEvidence>& source_evidence)
        {
            LocalGroundCandidate candidate;
            candidate.window = window;
            const double sensor_x = optimized_poses[center](0, 3);
            const double sensor_y = optimized_poses[center](1, 3);
            std::vector<GroundPlanePoint> local_points;
            const std::size_t frame_span = static_cast<std::size_t>(
                std::max(0, window) * 2 + 1);
            local_points.reserve(frame_span * 2048U);
            candidate.support_intervals.reserve(source_evidence.size());
            bool have_support = false;
            bool first_source_id = true;
            std::ostringstream source_ids;
            for (const auto& evidence : source_evidence)
            {
                const std::size_t distance = evidence.source > center
                    ? evidence.source - center : center - evidence.source;
                if (distance > static_cast<std::size_t>(window) ||
                    !evidence.considered)
                {
                    continue;
                }
                ++candidate.source_count;
                if (!evidence.independently_supported)
                {
                    continue;
                }
                ++candidate.independent_source_count;
                local_points.insert(
                    local_points.end(), evidence.points.begin(),
                    evidence.points.end());
                if (!first_source_id)
                {
                    source_ids << ';';
                }
                first_source_id = false;
                source_ids << snapshot[evidence.source].id;
                // Give each accepted source its path-adjacent Voronoi cell.
                // A large pose-to-pose step is capped so endpoint cells do
                // not touch at the midpoint and falsely claim unsupported
                // path as observed.
                const std::size_t source_index = evidence.source;
                const double source_s = path_coordinates[source_index];
                const double max_support_half_width =
                    0.5 * map_z_correction_config_.max_unobserved_gap_m;
                const double previous_step = source_index == 0U
                    ? 0.0
                    : std::max(0.0, source_s -
                        path_coordinates[source_index - 1U]);
                const double next_step = source_index + 1U >=
                        path_coordinates.size()
                    ? 0.0
                    : std::max(0.0,
                        path_coordinates[source_index + 1U] - source_s);
                const double left_width = source_index == 0U
                    ? source_s - path_coordinates.front()
                    : std::min(0.5 * previous_step,
                        max_support_half_width);
                const double right_width = source_index + 1U >=
                        path_coordinates.size()
                    ? path_coordinates.back() - source_s
                    : std::min(0.5 * next_step,
                        max_support_half_width);
                const double support_begin = source_s - left_width;
                const double support_end = source_s + right_width;
                if (std::isfinite(support_begin) &&
                    std::isfinite(support_end) &&
                    support_end >= support_begin)
                {
                    candidate.support_intervals.push_back({
                        std::max(path_coordinates.front(), support_begin),
                        std::min(path_coordinates.back(), support_end)});
                }
                candidate.support_s_min = have_support
                    ? std::min(candidate.support_s_min,
                        path_coordinates[evidence.source])
                    : path_coordinates[evidence.source];
                candidate.support_s_max = have_support
                    ? std::max(candidate.support_s_max,
                        path_coordinates[evidence.source])
                    : path_coordinates[evidence.source];
                have_support = true;
            }
            candidate.source_ids = source_ids.str();
            if (candidate.independent_source_count < static_cast<std::size_t>(
                    map_z_local_min_independent_sources_))
            {
                candidate.failure_reason = "independent source support below minimum";
                return candidate;
            }
            candidate.model = fit_ground_plane(local_points, local_options);
            if (!candidate.model.adaptive)
            {
                candidate.failure_reason = candidate.model.fallback_reason.empty()
                    ? "local ground quality gate failed"
                    : candidate.model.fallback_reason;
                return candidate;
            }
            const double floor_z = candidate.model.floor_z(sensor_x, sensor_y);
            candidate.valid = std::isfinite(floor_z) &&
                std::isfinite(candidate.model.tilt_deg) &&
                candidate.model.tilt_deg <= map_z_local_max_tilt_deg_ &&
                std::isfinite(candidate.model.residual_p95) &&
                candidate.model.residual_p95 <= map_z_local_max_residual_p95_;
            if (!candidate.valid)
            {
                candidate.failure_reason = "local ground metrics are invalid";
                return candidate;
            }
            candidate.score =
                candidate.model.residual_p95 / map_z_local_max_residual_p95_ +
                candidate.model.tilt_deg / map_z_local_max_tilt_deg_ +
                0.015 * static_cast<double>(window) -
                0.01 * static_cast<double>(std::min<std::size_t>(
                    candidate.independent_source_count, 5U));
            return candidate;
        };
        for (std::size_t index = 0U; index < snapshot.size(); ++index)
        {
            MapZDriftObservation observation;
            observation.path_s = path_coordinates[index];
            if (!finiteRigidPose(optimized_poses[index]) ||
                !snapshot[index].dense_cloud_lidar ||
                snapshot[index].dense_cloud_lidar->empty())
            {
                observation.failure_reason =
                    "invalid pose or empty dense keyframe cloud";
                map_z_observations_.push_back(std::move(observation));
                continue;
            }
            LocalGroundCandidate selected;
            bool have_selected = false;
            std::ostringstream diagnostics;
            const std::size_t max_window = static_cast<std::size_t>(
                map_z_local_keyframe_window_);
            const std::size_t source_begin = index > max_window
                ? index - max_window : 0U;
            const std::size_t source_end = std::min(
                snapshot.size(), index + max_window + 1U);
            std::vector<SourceGroundEvidence> source_evidence;
            source_evidence.reserve(source_end - source_begin);
            for (std::size_t source = source_begin; source < source_end; ++source)
            {
                source_evidence.push_back(
                    collect_source_evidence(index, source));
            }
            for (int window = 0; window <= map_z_local_keyframe_window_; ++window)
            {
                LocalGroundCandidate candidate = fit_local_ground(
                    index, window, source_evidence);
                if (window > 0)
                {
                    diagnostics << '|';
                }
                diagnostics << "w=" << window
                    << ":src=" << candidate.independent_source_count
                    << '/' << candidate.source_count
                    << ":cells=" << candidate.model.sample_cells
                    << ":inliers=" << candidate.model.inlier_cells
                    << ":tilt=" << candidate.model.tilt_deg
                    << ":p95=" << candidate.model.residual_p95
                    << ":reason=" << (candidate.valid
                        ? "accepted" : candidate.failure_reason);
                if (candidate.valid &&
                    (!have_selected || candidate.score < selected.score))
                {
                    selected = std::move(candidate);
                    have_selected = true;
                }
            }
            observation.window_diagnostics = diagnostics.str();
            if (!have_selected)
            {
                observation.failure_reason =
                    "no source-aware local window passed all quality gates";
            }
            else
            {
                const double sensor_x = optimized_poses[index](0, 3);
                const double sensor_y = optimized_poses[index](1, 3);
                observation.floor_z = selected.model.floor_z(sensor_x, sensor_y);
                observation.local_tilt_deg = selected.model.tilt_deg;
                observation.residual_p95 = selected.model.residual_p95;
                observation.selected_window = static_cast<std::size_t>(
                    selected.window);
                observation.candidate_points = selected.model.candidate_points;
                observation.sample_cells = selected.model.sample_cells;
                observation.inlier_cells = selected.model.inlier_cells;
                observation.source_count = selected.source_count;
                observation.independent_source_count =
                    selected.independent_source_count;
                observation.support_s_min = selected.support_s_min;
                observation.support_s_max = selected.support_s_max;
                observation.support_intervals = selected.support_intervals;
                observation.has_explicit_support =
                    !observation.support_intervals.empty();
                observation.source_ids = selected.source_ids;
                observation.valid = true;
                RCLCPP_DEBUG(
                    get_logger(),
                    "Map Z local ground selected: center=%zu window=%d "
                    "sources=%zu/%zu cells=%zu inliers=%zu tilt=%.3fdeg "
                    "residual_p95=%.4fm support=[%.3f,%.3f]m",
                    index, selected.window,
                    selected.independent_source_count, selected.source_count,
                    selected.model.sample_cells, selected.model.inlier_cells,
                    observation.local_tilt_deg, observation.residual_p95,
                    observation.support_s_min, observation.support_s_max);
            }
            map_z_observations_.push_back(std::move(observation));
        }

        map_z_correction_result_ = solveMapZDriftCorrection(
            map_z_observations_, map_z_correction_config_);
        if (!map_z_correction_result_.success ||
            map_z_correction_result_.corrections.size() != optimized_poses.size())
        {
            std::map<std::string, std::size_t> failure_counts;
            for (std::size_t index = 0U;
                 index < map_z_observations_.size(); ++index)
            {
                const auto& observation = map_z_observations_[index];
                if (!observation.valid)
                {
                    ++failure_counts[observation.failure_reason.empty()
                        ? std::string("unspecified local ground failure")
                        : observation.failure_reason];
                }
                    const bool used =
                        index < map_z_correction_result_.used_observations.size() &&
                        map_z_correction_result_.used_observations[index];
                if (!used)
                {
                    RCLCPP_WARN(
                        get_logger(),
                        "[MAP Z OBS] index=%zu path_s=%.3f valid=%d used=0 "
                        "window=%zu sources=%zu/%zu cells=%zu inliers=%zu "
                        "support=[%.3f,%.3f] source_ids=%s reason=%s windows=%s",
                        index, observation.path_s,
                        observation.valid ? 1 : 0,
                        observation.selected_window,
                        observation.independent_source_count,
                        observation.source_count, observation.sample_cells,
                        observation.inlier_cells, observation.support_s_min,
                        observation.support_s_max,
                        observation.source_ids.c_str(),
                        observation.failure_reason.c_str(),
                        observation.window_diagnostics.c_str());
                }
            }
            std::string failure_summary;
            for (const auto& [reason, count] : failure_counts)
            {
                if (!failure_summary.empty())
                {
                    failure_summary += "; ";
                }
                failure_summary += reason + "=" + std::to_string(count);
            }
            RCLCPP_ERROR(
                get_logger(),
                "Map Z correction failed: %s "
                "(frame_valid=%.3f path_coverage=%.3f max_gap=%.3fm "
                "invalid_ranges=%s local_failures={%s})",
                map_z_correction_result_.failure_reason.c_str(),
                map_z_correction_result_.valid_fraction,
                map_z_correction_result_.path_coverage_fraction,
                map_z_correction_result_.max_unobserved_gap_m,
                map_z_correction_result_.invalid_index_ranges.c_str(),
                failure_summary.c_str());
            return false;
        }
        for (std::size_t index = 0U; index < optimized_poses.size(); ++index)
        {
            optimized_poses[index](2, 3) +=
                map_z_correction_result_.corrections[index];
            if (!finiteRigidPose(optimized_poses[index]))
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map Z correction produced an invalid pose at index %zu", index);
                return false;
            }
        }

        PointCloudType corrected_map;
        corrected_map.reserve(map_source.size());
        for (std::size_t index = 0U; index < snapshot.size(); ++index)
        {
            PointCloudType transformed;
            pcl::transformPointCloud(
                *snapshot[index].dense_cloud_lidar, transformed,
                optimized_poses[index].cast<float>());
            corrected_map += transformed;
        }
        if (corrected_map.empty())
        {
            RCLCPP_ERROR(get_logger(), "Map Z correction rebuilt an empty map");
            return false;
        }
        map_source.swap(corrected_map);
        RCLCPP_INFO(
            get_logger(),
            "Map Z correction accepted: observations=%zu/%zu outliers=%zu "
            "frame_valid=%.3f path_coverage=%.3f max_gap=%.3fm target_floor=%.4fm "
            "max_abs=%.4fm max_slope=%.4f corrected_floor_range=[%.4f,%.4f]m "
            "invalid_ranges=%s",
            map_z_correction_result_.used_valid_count,
            map_z_observations_.size(),
            map_z_correction_result_.rejected_outlier_count,
            map_z_correction_result_.valid_fraction,
            map_z_correction_result_.path_coverage_fraction,
            map_z_correction_result_.max_unobserved_gap_m,
            map_z_correction_result_.target_floor_z,
            map_z_correction_result_.max_abs_correction_m,
            map_z_correction_result_.max_correction_slope,
            map_z_correction_result_.corrected_floor_min,
            map_z_correction_result_.corrected_floor_max,
            map_z_correction_result_.invalid_index_ranges.c_str());
        return true;
    }

    bool MappingAlg::levelMapForSave(
        PointCloudType& map_source,
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& optimized_poses)
    {
        const double original_keyframe_floor_z_map = keyframe_floor_z_map_;
        const double original_pcd2pgm_floor_z_map = pcd2pgm_options_.floor_z_map;
        const Eigen::Matrix3d original_map_leveling_rotation =
            map_leveling_rotation_;
        const Eigen::Vector3d original_map_leveling_normal =
            map_leveling_normal_;
        const double original_map_leveling_floor_z = map_leveling_floor_z_;
        const double original_map_leveling_tilt_deg = map_leveling_tilt_deg_;
        const double original_map_leveling_inlier_ratio =
            map_leveling_inlier_ratio_;
        const int original_map_leveling_iterations = map_leveling_iterations_;
        const bool original_map_leveling_succeeded = map_leveling_succeeded_;
        const double original_map_leveling_selected_quantile =
            map_leveling_selected_quantile_;
        const bool original_map_leveling_q10_attempted =
            map_leveling_q10_attempted_;
        const bool original_map_leveling_q10_accepted =
            map_leveling_q10_accepted_;
        const std::size_t original_map_leveling_seed_candidate_points =
            map_leveling_seed_candidate_points_;
        const std::size_t original_map_leveling_seed_sample_cells =
            map_leveling_seed_sample_cells_;
        const std::size_t original_map_leveling_seed_inlier_cells =
            map_leveling_seed_inlier_cells_;
        const double original_map_leveling_seed_inlier_ratio =
            map_leveling_seed_inlier_ratio_;
        const double original_map_leveling_seed_residual_p95 =
            map_leveling_seed_residual_p95_;
        const double original_map_leveling_seed_tilt_deg =
            map_leveling_seed_tilt_deg_;
        const std::size_t original_map_leveling_refit_candidate_points =
            map_leveling_refit_candidate_points_;
        const std::size_t original_map_leveling_refit_sample_cells =
            map_leveling_refit_sample_cells_;
        const std::size_t original_map_leveling_refit_inlier_cells =
            map_leveling_refit_inlier_cells_;
        const double original_map_leveling_refit_inlier_ratio =
            map_leveling_refit_inlier_ratio_;
        const double original_map_leveling_refit_residual_p95 =
            map_leveling_refit_residual_p95_;
        const double original_map_leveling_refit_major_span =
            map_leveling_refit_major_span_;
        const double original_map_leveling_refit_minor_span =
            map_leveling_refit_minor_span_;
        const std::string original_map_leveling_failure_reason =
            map_leveling_failure_reason_;
        Eigen::Matrix3d accumulated = Eigen::Matrix3d::Identity();
        bool leveling_committed = false;
        const auto restore_state = [&]() noexcept
        {
            if (leveling_committed)
            {
                return;
            }
            if (!accumulated.isApprox(Eigen::Matrix3d::Identity(), 1e-12))
            {
                const Eigen::Matrix3f inverse_rotation =
                    accumulated.transpose().cast<float>();
                for (auto& point : map_source.points)
                {
                    const float x = point.x;
                    const float y = point.y;
                    const float z = point.z;
                    point.x = inverse_rotation(0, 0) * x +
                        inverse_rotation(0, 1) * y +
                        inverse_rotation(0, 2) * z;
                    point.y = inverse_rotation(1, 0) * x +
                        inverse_rotation(1, 1) * y +
                        inverse_rotation(1, 2) * z;
                    point.z = inverse_rotation(2, 0) * x +
                        inverse_rotation(2, 1) * y +
                        inverse_rotation(2, 2) * z;
                }
                for (auto& pose : optimized_poses)
                {
                    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                    transform.block<3, 3>(0, 0) = accumulated.transpose();
                    pose = transform * pose;
                }
            }
            keyframe_floor_z_map_ = original_keyframe_floor_z_map;
            pcd2pgm_options_.floor_z_map = original_pcd2pgm_floor_z_map;
            map_leveling_rotation_ = original_map_leveling_rotation;
            map_leveling_normal_ = original_map_leveling_normal;
            map_leveling_floor_z_ = original_map_leveling_floor_z;
            map_leveling_tilt_deg_ = original_map_leveling_tilt_deg;
            map_leveling_inlier_ratio_ = original_map_leveling_inlier_ratio;
            map_leveling_iterations_ = original_map_leveling_iterations;
            map_leveling_succeeded_ = original_map_leveling_succeeded;
            map_leveling_selected_quantile_ = original_map_leveling_selected_quantile;
            map_leveling_q10_attempted_ = original_map_leveling_q10_attempted;
            map_leveling_q10_accepted_ = original_map_leveling_q10_accepted;
            map_leveling_seed_candidate_points_ = original_map_leveling_seed_candidate_points;
            map_leveling_seed_sample_cells_ = original_map_leveling_seed_sample_cells;
            map_leveling_seed_inlier_cells_ = original_map_leveling_seed_inlier_cells;
            map_leveling_seed_inlier_ratio_ = original_map_leveling_seed_inlier_ratio;
            map_leveling_seed_residual_p95_ = original_map_leveling_seed_residual_p95;
            map_leveling_seed_tilt_deg_ = original_map_leveling_seed_tilt_deg;
            map_leveling_refit_candidate_points_ = original_map_leveling_refit_candidate_points;
            map_leveling_refit_sample_cells_ = original_map_leveling_refit_sample_cells;
            map_leveling_refit_inlier_cells_ = original_map_leveling_refit_inlier_cells;
            map_leveling_refit_inlier_ratio_ = original_map_leveling_refit_inlier_ratio;
            map_leveling_refit_residual_p95_ = original_map_leveling_refit_residual_p95;
            map_leveling_refit_major_span_ = original_map_leveling_refit_major_span;
            map_leveling_refit_minor_span_ = original_map_leveling_refit_minor_span;
            map_leveling_failure_reason_ = original_map_leveling_failure_reason;
        };
        ScopeGuard rollback_guard(restore_state);

        map_leveling_rotation_.setIdentity();
        map_leveling_normal_ = Eigen::Vector3d::UnitZ();
        map_leveling_floor_z_ = keyframe_floor_z_map_;
        map_leveling_tilt_deg_ = 0.0;
        map_leveling_inlier_ratio_ = 0.0;
        map_leveling_iterations_ = 0;
        map_leveling_succeeded_ = false;
        map_leveling_selected_quantile_ = map_leveling_sample_quantile_;
        map_leveling_q10_attempted_ = false;
        map_leveling_q10_accepted_ = false;
        map_leveling_seed_candidate_points_ = 0U;
        map_leveling_seed_sample_cells_ = 0U;
        map_leveling_seed_inlier_cells_ = 0U;
        map_leveling_seed_inlier_ratio_ = 0.0;
        map_leveling_seed_residual_p95_ = 0.0;
        map_leveling_seed_tilt_deg_ = 0.0;
        map_leveling_refit_candidate_points_ = 0U;
        map_leveling_refit_sample_cells_ = 0U;
        map_leveling_refit_inlier_cells_ = 0U;
        map_leveling_refit_inlier_ratio_ = 0.0;
        map_leveling_refit_residual_p95_ = 0.0;
        map_leveling_refit_major_span_ = 0.0;
        map_leveling_refit_minor_span_ = 0.0;
        map_leveling_failure_reason_.clear();
        map_leveling_last_gate_reason_.clear();
        const auto fail = [&](const std::string& reason)
        {
            map_leveling_failure_reason_ = reason;
            map_leveling_last_gate_reason_ = reason;
            RCLCPP_ERROR(get_logger(), "Map leveling failed: %s", reason.c_str());
            return false;
        };
        if (!map_leveling_enabled_)
        {
            // Returning true here means that disabling the optional
            // correction is not an execution error.  It does not mean a
            // correction was applied; the sidecar must report an identity
            // transform and leveling_applied=false.
            map_leveling_succeeded_ = false;
            map_leveling_failure_reason_ = "disabled by configuration";
            map_leveling_last_gate_reason_ = map_leveling_failure_reason_;
            leveling_committed = true;
            rollback_guard.dismiss();
            return true;
        }
        if (map_source.empty())
        {
            return fail("map source is empty");
        }
        if (!std::isfinite(map_leveling_max_correction_deg_) ||
            map_leveling_max_correction_deg_ <= 0.0 ||
            !std::isfinite(map_leveling_min_inlier_ratio_) ||
            map_leveling_min_inlier_ratio_ <= 0.0 ||
            map_leveling_min_inlier_ratio_ > 1.0 ||
            !std::isfinite(map_leveling_sample_quantile_) ||
            map_leveling_sample_quantile_ <= 0.0 ||
            map_leveling_sample_quantile_ > 0.20 ||
            !std::isfinite(map_leveling_fallback_sample_quantile_) ||
            map_leveling_fallback_sample_quantile_ <=
                map_leveling_sample_quantile_ ||
            map_leveling_fallback_sample_quantile_ > 0.20 ||
            !std::isfinite(map_leveling_converged_tilt_deg_) ||
            map_leveling_converged_tilt_deg_ <= 0.0 ||
            map_leveling_converged_tilt_deg_ >=
                map_leveling_max_correction_deg_ ||
            map_leveling_max_iterations_ < 1)
        {
            return fail("invalid leveling configuration");
        }
        const double configured_sample_quantile =
            map_leveling_sample_quantile_;
        const double fallback_sample_quantile =
            map_leveling_fallback_sample_quantile_;
        double active_sample_quantile = configured_sample_quantile;
        bool fallback_quantile_attempted = false;
        const auto retry_with_fallback_quantile = [&](const char* reason)
        {
            if (fallback_quantile_attempted)
            {
                return false;
            }
            fallback_quantile_attempted = true;
            map_leveling_q10_attempted_ = true;
            active_sample_quantile = fallback_sample_quantile;
            RCLCPP_WARN(
                get_logger(),
                "Map leveling primary sample quantile %.3f failed (%s); retrying with fallback %.3f",
                configured_sample_quantile, reason, fallback_sample_quantile);
            return true;
        };

        double crop_x_min = std::numeric_limits<double>::max();
        double crop_x_max = std::numeric_limits<double>::lowest();
        double crop_y_min = std::numeric_limits<double>::max();
        double crop_y_max = std::numeric_limits<double>::lowest();
        bool have_pose_bounds = false;
        for (const auto& pose : optimized_poses)
        {
            if (!pose.allFinite() ||
                !pose.row(3).isApprox(
                    Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-6))
            {
                return fail("trajectory contains a non-finite pose");
            }
            crop_x_min = std::min(crop_x_min, pose(0, 3));
            crop_x_max = std::max(crop_x_max, pose(0, 3));
            crop_y_min = std::min(crop_y_min, pose(1, 3));
            crop_y_max = std::max(crop_y_max, pose(1, 3));
            have_pose_bounds = true;
        }
        if (!have_pose_bounds)
        {
            for (const auto& point : map_source.points)
            {
                if (!std::isfinite(point.x) || !std::isfinite(point.y))
                {
                    continue;
                }
                crop_x_min = std::min(crop_x_min, static_cast<double>(point.x));
                crop_x_max = std::max(crop_x_max, static_cast<double>(point.x));
                crop_y_min = std::min(crop_y_min, static_cast<double>(point.y));
                crop_y_max = std::max(crop_y_max, static_cast<double>(point.y));
            }
        }
        if (!std::isfinite(crop_x_min) || !std::isfinite(crop_x_max) ||
            !std::isfinite(crop_y_min) || !std::isfinite(crop_y_max) ||
            crop_x_min > crop_x_max || crop_y_min > crop_y_max)
        {
            return fail("no finite XY bounds for leveling");
        }
        const double crop_margin = std::max(1.0, grid_crop_margin_);
        crop_x_min -= crop_margin;
        crop_x_max += crop_margin;
        crop_y_min -= crop_margin;
        crop_y_max += crop_margin;
        const auto refresh_crop_bounds = [&]()
        {
            double next_x_min = std::numeric_limits<double>::max();
            double next_x_max = std::numeric_limits<double>::lowest();
            double next_y_min = std::numeric_limits<double>::max();
            double next_y_max = std::numeric_limits<double>::lowest();
            bool have_bounds = false;
            for (const auto& pose : optimized_poses)
            {
                if (!finiteRigidPose(pose))
                {
                    return false;
                }
                next_x_min = std::min(next_x_min, pose(0, 3));
                next_x_max = std::max(next_x_max, pose(0, 3));
                next_y_min = std::min(next_y_min, pose(1, 3));
                next_y_max = std::max(next_y_max, pose(1, 3));
                have_bounds = true;
            }
            if (!have_bounds)
            {
                for (const auto& point : map_source.points)
                {
                    if (!std::isfinite(point.x) || !std::isfinite(point.y))
                    {
                        continue;
                    }
                    next_x_min = std::min(next_x_min, static_cast<double>(point.x));
                    next_x_max = std::max(next_x_max, static_cast<double>(point.x));
                    next_y_min = std::min(next_y_min, static_cast<double>(point.y));
                    next_y_max = std::max(next_y_max, static_cast<double>(point.y));
                    have_bounds = true;
                }
            }
            if (!have_bounds || !std::isfinite(next_x_min) ||
                !std::isfinite(next_x_max) || !std::isfinite(next_y_min) ||
                !std::isfinite(next_y_max) || next_x_min > next_x_max ||
                next_y_min > next_y_max)
            {
                return false;
            }
            crop_x_min = next_x_min - crop_margin;
            crop_x_max = next_x_max + crop_margin;
            crop_y_min = next_y_min - crop_margin;
            crop_y_max = next_y_max + crop_margin;
            return std::isfinite(crop_x_min) && std::isfinite(crop_x_max) &&
                std::isfinite(crop_y_min) && std::isfinite(crop_y_max);
        };

        using CellKey = std::pair<std::int64_t, std::int64_t>;
        const double sample_cell_size = 0.20;
        const auto build_low_envelope = [&]()
        {
            std::map<CellKey, std::vector<double>> cell_heights;
            for (const auto& point : map_source.points)
            {
                if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                    !std::isfinite(point.z) || point.x < crop_x_min ||
                    point.x > crop_x_max || point.y < crop_y_min ||
                    point.y > crop_y_max)
                {
                    continue;
                }
                const double cell_x = std::floor(
                    static_cast<double>(point.x) / sample_cell_size);
                const double cell_y = std::floor(
                    static_cast<double>(point.y) / sample_cell_size);
                if (!std::isfinite(cell_x) || !std::isfinite(cell_y) ||
                    cell_x < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
                    cell_x > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
                    cell_y < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
                    cell_y > static_cast<double>(std::numeric_limits<std::int64_t>::max()))
                {
                    continue;
                }
                cell_heights[{static_cast<std::int64_t>(cell_x),
                              static_cast<std::int64_t>(cell_y)}]
                    .push_back(static_cast<double>(point.z));
            }
            std::vector<GroundPlanePoint> samples;
            samples.reserve(cell_heights.size());
            for (auto& entry : cell_heights)
            {
                auto& heights = entry.second;
                if (heights.empty())
                {
                    continue;
                }
                std::sort(heights.begin(), heights.end());
                const double position = active_sample_quantile *
                    static_cast<double>(heights.size() - 1U);
                const std::size_t lower = static_cast<std::size_t>(
                    std::floor(position));
                const std::size_t upper = static_cast<std::size_t>(
                    std::ceil(position));
                const double weight = position - static_cast<double>(lower);
                const double z = heights[lower] * (1.0 - weight) +
                    heights[upper] * weight;
                samples.push_back({
                    (static_cast<double>(entry.first.first) + 0.5) *
                        sample_cell_size,
                    (static_cast<double>(entry.first.second) + 0.5) *
                        sample_cell_size,
                    z});
            }
            return samples;
        };
        const auto collect_raw_points = [&]()
        {
            std::vector<GroundPlanePoint> points;
            points.reserve(map_source.size());
            for (const auto& point : map_source.points)
            {
                if (std::isfinite(point.x) && std::isfinite(point.y) &&
                    std::isfinite(point.z) && point.x >= crop_x_min &&
                    point.x <= crop_x_max && point.y >= crop_y_min &&
                    point.y <= crop_y_max)
                {
                    points.push_back({point.x, point.y, point.z});
                }
            }
            return points;
        };
        const auto median_value = [](std::vector<double> values)
        {
            if (values.empty())
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            std::sort(values.begin(), values.end());
            const double position = 0.5 *
                static_cast<double>(values.size() - 1U);
            const std::size_t lower = static_cast<std::size_t>(
                std::floor(position));
            const std::size_t upper = static_cast<std::size_t>(
                std::ceil(position));
            const double weight = position - static_cast<double>(lower);
            return values[lower] * (1.0 - weight) + values[upper] * weight;
        };
        const auto model_passes = [](const GroundPlaneModel& model,
                                     double min_ratio, double max_p95,
                                     double max_tilt, double max_floor_offset)
        {
            const Eigen::Vector3d normal = model.normalAt(
                model.center_x, model.center_y);
            const double floor_deviation = std::max(
                std::abs(model.fitted_floor_min - model.fixed_floor_z),
                std::abs(model.fitted_floor_max - model.fixed_floor_z));
            const double required_major = model.sample_major_span * 0.35;
            return model.adaptive && model.candidate_points > 0U &&
                model.sample_cells >= 3U && model.inlier_cells >= 3U &&
                std::isfinite(model.inlier_ratio) &&
                model.inlier_ratio >= min_ratio &&
                std::isfinite(model.residual_p95) &&
                model.residual_p95 <= max_p95 &&
                std::isfinite(model.tilt_deg) && model.tilt_deg <= max_tilt &&
                std::isfinite(model.fitted_floor_min) &&
                std::isfinite(model.fitted_floor_max) &&
                std::isfinite(floor_deviation) &&
                floor_deviation <= max_floor_offset && normal.allFinite() &&
                std::isfinite(normal.norm()) && normal.norm() > 0.99 &&
                normal.z() > 0.0 &&
                std::isfinite(model.sample_major_span) &&
                std::isfinite(model.sample_minor_span) &&
                std::isfinite(model.inlier_major_span) &&
                std::isfinite(model.inlier_minor_span) &&
                model.sample_major_span >= 1.0 &&
                model.sample_minor_span >= 0.05 &&
                model.inlier_major_span >= std::max(1.0, required_major) &&
                model.inlier_minor_span >= 0.05;
        };

        GroundPlaneModel final_model;
        bool converged = false;
        for (int iteration = 0; iteration < map_leveling_max_iterations_; ++iteration)
        {
            if (!refresh_crop_bounds())
            {
                return fail("non-finite crop bounds during leveling");
            }
            const std::vector<GroundPlanePoint> seed_points = build_low_envelope();
            if (seed_points.empty())
            {
                return fail("no finite XY low-envelope samples");
            }
            GroundPlaneFitOptions seed_options;
            seed_options.fixed_floor_z = keyframe_floor_z_map_;
            seed_options.candidate_min_height = -2.0;
            seed_options.candidate_max_height = 2.0;
            seed_options.sample_cell_size = 0.25;
            seed_options.min_points_per_cell = 1;
            seed_options.sample_quantile = active_sample_quantile;
            seed_options.min_sample_cells = 30;
            seed_options.min_xy_span = 1.0;
            seed_options.min_inlier_ratio = 0.35;
            seed_options.max_tilt_deg = 45.0;
            seed_options.max_residual_p95 = 0.40;
            seed_options.max_floor_offset = 2.0;
            seed_options.min_inlier_span_fraction = 0.25;
            seed_options.min_inlier_minor_span = 0.05;
            seed_options.quadratic_enabled = false;
            seed_options.max_iterations = 8;
            const GroundPlaneModel seed_model =
                fit_ground_plane(seed_points, seed_options);
            map_leveling_seed_candidate_points_ = seed_model.candidate_points;
            map_leveling_seed_sample_cells_ = seed_model.sample_cells;
            map_leveling_seed_inlier_cells_ = seed_model.inlier_cells;
            map_leveling_seed_inlier_ratio_ = seed_model.inlier_ratio;
            map_leveling_seed_residual_p95_ = seed_model.residual_p95;
            map_leveling_seed_tilt_deg_ = seed_model.tilt_deg;
            if (!model_passes(seed_model, 0.35, 0.40, 45.0, 2.0))
            {
                if (retry_with_fallback_quantile("seed quality gate"))
                {
                    --iteration;
                    continue;
                }
                return fail(
                    "robust seed quality gate failed: " +
                    (seed_model.fallback_reason.empty()
                         ? std::string("incomplete seed metrics")
                         : seed_model.fallback_reason));
            }

            std::vector<double> near_zero_residuals;
            const double seed_near_band = std::clamp(
                std::max(0.08, 2.0 * seed_model.residual_p95 + 0.02),
                0.08, 0.35);
            near_zero_residuals.reserve(seed_points.size());
            for (const auto& point : seed_points)
            {
                const double residual = point.z -
                    (seed_model.a * point.x + seed_model.b * point.y +
                     seed_model.c);
                if (std::isfinite(residual) &&
                    std::abs(residual) <= seed_near_band)
                {
                    near_zero_residuals.push_back(residual);
                }
            }
            if (near_zero_residuals.size() < 15U)
            {
                if (retry_with_fallback_quantile(
                        "insufficient seed near-zero residual candidates"))
                {
                    --iteration;
                    continue;
                }
                return fail("insufficient seed near-zero residual candidates");
            }
            std::vector<double> absolute_residuals;
            absolute_residuals.reserve(near_zero_residuals.size());
            for (const double residual : near_zero_residuals)
            {
                absolute_residuals.push_back(std::abs(residual));
            }
            const double residual_mad = median_value(std::move(absolute_residuals));
            if (!std::isfinite(residual_mad))
            {
                if (retry_with_fallback_quantile(
                        "non-finite seed residual statistics"))
                {
                    --iteration;
                    continue;
                }
                return fail("seed near-zero residual statistics are non-finite");
            }
            const double residual_gate = std::clamp(
                std::max(0.05, 4.0 * 1.4826 * residual_mad), 0.05, 0.30);
            const std::vector<GroundPlanePoint> raw_points = collect_raw_points();
            std::vector<GroundPlanePoint> refined_points;
            refined_points.reserve(raw_points.size());
            double predicted_floor_min = std::numeric_limits<double>::max();
            double predicted_floor_max = std::numeric_limits<double>::lowest();
            for (const auto& point : raw_points)
            {
                const double predicted = seed_model.a * point.x +
                    seed_model.b * point.y + seed_model.c;
                const double residual = point.z - predicted;
                if (!std::isfinite(predicted) || !std::isfinite(residual))
                {
                    continue;
                }
                if (std::abs(residual) <= residual_gate)
                {
                    refined_points.push_back(point);
                    predicted_floor_min = std::min(predicted_floor_min, predicted);
                    predicted_floor_max = std::max(predicted_floor_max, predicted);
                }
            }
            if (refined_points.size() < seed_options.min_sample_cells ||
                !std::isfinite(predicted_floor_min) ||
                !std::isfinite(predicted_floor_max))
            {
                if (retry_with_fallback_quantile(
                        "insufficient raw points in seed band"))
                {
                    --iteration;
                    continue;
                }
                return fail("seed zero-residual band has insufficient raw points");
            }

            GroundPlaneFitOptions refit_options = seed_options;
            refit_options.sample_cell_size = 0.20;
            refit_options.min_sample_cells = 30;
            refit_options.min_inlier_ratio = map_leveling_min_inlier_ratio_;
            refit_options.max_residual_p95 = 0.20;
            refit_options.max_tilt_deg = 45.0;
            refit_options.max_floor_offset = 2.0;
            const double candidate_padding = std::max(0.10, residual_gate);
            refit_options.candidate_min_height =
                predicted_floor_min - candidate_padding - keyframe_floor_z_map_;
            refit_options.candidate_max_height =
                predicted_floor_max + candidate_padding - keyframe_floor_z_map_;
            if (!std::isfinite(refit_options.candidate_min_height) ||
                !std::isfinite(refit_options.candidate_max_height) ||
                refit_options.candidate_min_height < -2.0 ||
                refit_options.candidate_max_height > 2.0 ||
                refit_options.candidate_max_height <=
                    refit_options.candidate_min_height + 0.02)
            {
                if (retry_with_fallback_quantile("invalid refit height band"))
                {
                    --iteration;
                    continue;
                }
                return fail("refit candidate height band is invalid");
            }
            refit_options.sample_quantile = active_sample_quantile;
            GroundPlaneModel refit_model = fit_ground_plane(
                refined_points, refit_options);
            if (!model_passes(refit_model, map_leveling_min_inlier_ratio_,
                              0.20, 45.0, 2.0))
            {
                if (retry_with_fallback_quantile("ground refit quality gate"))
                {
                    --iteration;
                    continue;
                }
            }
            final_model = refit_model;
            map_leveling_refit_candidate_points_ = final_model.candidate_points;
            map_leveling_refit_sample_cells_ = final_model.sample_cells;
            map_leveling_refit_inlier_cells_ = final_model.inlier_cells;
            map_leveling_refit_inlier_ratio_ = final_model.inlier_ratio;
            map_leveling_refit_residual_p95_ = final_model.residual_p95;
            map_leveling_refit_major_span_ = final_model.inlier_major_span;
            map_leveling_refit_minor_span_ = final_model.inlier_minor_span;
            if (!model_passes(final_model, map_leveling_min_inlier_ratio_,
                              0.20, 45.0, 2.0))
            {
                return fail(
                    "ground refit quality gate failed: " +
                    (final_model.fallback_reason.empty()
                         ? std::string("incomplete final metrics")
                         : final_model.fallback_reason));
            }
            const Eigen::Vector3d normal = final_model.normalAt(
                final_model.center_x, final_model.center_y);
            if (!normal.allFinite() || normal.norm() <= 0.99 || normal.z() <= 0.0)
            {
                return fail("ground refit normal is invalid");
            }
            map_leveling_normal_ = normal.normalized();
            const double tilt_rad = std::acos(std::clamp(
                map_leveling_normal_.dot(Eigen::Vector3d::UnitZ()), -1.0, 1.0));
            const double tilt_deg = tilt_rad * 180.0 / PI_M;
            if (!std::isfinite(tilt_deg))
            {
                return fail("ground refit tilt is non-finite");
            }
            map_leveling_tilt_deg_ = tilt_deg;
            map_leveling_inlier_ratio_ = final_model.inlier_ratio;
            map_leveling_iterations_ = iteration + 1;
            if (tilt_deg <= map_leveling_converged_tilt_deg_)
            {
                converged = true;
                break;
            }
            const Eigen::Matrix3d correction =
                Eigen::Quaterniond::FromTwoVectors(
                    map_leveling_normal_, Eigen::Vector3d::UnitZ())
                    .toRotationMatrix();
            const Eigen::Matrix3d next_accumulated = correction * accumulated;
            const Eigen::AngleAxisd total_rotation(next_accumulated);
            const double total_correction_deg =
                std::abs(total_rotation.angle()) * 180.0 / PI_M;
            if (!std::isfinite(total_correction_deg) ||
                total_correction_deg > map_leveling_max_correction_deg_)
            {
                return fail("rigid correction exceeds configured limit");
            }
            Eigen::Matrix4f correction_transform = Eigen::Matrix4f::Identity();
            correction_transform.block<3, 3>(0, 0) = correction.cast<float>();
            PointCloudType corrected;
            pcl::transformPointCloud(map_source, corrected, correction_transform);
            map_source.swap(corrected);
            accumulated = next_accumulated;
            for (auto& pose : optimized_poses)
            {
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = correction;
                pose = transform * pose;
            }
        }
        if (!converged)
        {
            return fail("did not converge below configured tilt threshold");
        }

        const double fitted_floor =
            keyframe_floor_z_map_ + final_model.median_floor_offset;
        if (!std::isfinite(fitted_floor))
        {
            return fail("fitted floor is non-finite");
        }
        map_leveling_floor_z_ = fitted_floor;
        keyframe_floor_z_map_ = fitted_floor;
        pcd2pgm_options_.floor_z_map = fitted_floor;
        map_leveling_rotation_ = accumulated;
        map_leveling_selected_quantile_ = active_sample_quantile;
        map_leveling_q10_accepted_ = fallback_quantile_attempted;
        map_leveling_succeeded_ = true;
        map_leveling_failure_reason_.clear();
        const Eigen::Vector3d correction_rpy = canonicalRpy(accumulated);
        RCLCPP_INFO(
            get_logger(),
            "Map leveling accepted: correction_rpy_deg=[%.4f,%.4f,%.4f] "
            "tilt=%.4fdeg floor_z_map=%.6f inlier_ratio=%.3f iterations=%d "
            "quantile=%.3f q10_attempted=%d q10_accepted=%d coverage=[%.3f,%.3f]",
            correction_rpy.x() * 180.0 / PI_M,
            correction_rpy.y() * 180.0 / PI_M,
            correction_rpy.z() * 180.0 / PI_M,
            map_leveling_tilt_deg_, map_leveling_floor_z_,
            map_leveling_inlier_ratio_, map_leveling_iterations_,
            map_leveling_selected_quantile_, map_leveling_q10_attempted_ ? 1 : 0,
            map_leveling_q10_accepted_ ? 1 : 0,
            map_leveling_refit_major_span_, map_leveling_refit_minor_span_);
        leveling_committed = true;
        rollback_guard.dismiss();
        return true;
    }

    bool MappingAlg::finish()
    {
        if (lio_time_guard_session_.rejected_frames > 0U)
        {
            // Data hiccups are recorded and counted but must not block
            // saving: the guard already dropped the offending frames, and
            // refusing here would discard minutes of valid mapping over a
            // transient sensor-timeline hiccup.
            RCLCPP_WARN(
                get_logger(),
                "Mapping session had %lu rejected sensor frame(s) "
                "(last_reason=%s); saving map with that diagnostic recorded",
                static_cast<unsigned long>(lio_time_guard_session_.rejected_frames),
                lio_time_guard_session_.failure_reason.empty()
                    ? "none" : lio_time_guard_session_.failure_reason.c_str());
        }
        // Freeze one immutable generation before waiting for loop closure.
        // Every artifact below consumes this exact snapshot; no later store
        // lookup is allowed to silently mix keyframe generations.
        const KeyframeStore::Snapshot snapshot = keyframe_store_->snapshot();
        const size_t keyframe_count = snapshot.size();
        if (!is_safe_pcd2grid_basename(pcd2pgm_options_.file_name))
        {
            RCLCPP_ERROR(
                get_logger(),
                "Map save failed: pcd2pgm.file_name must be a single relative basename");
            return false;
        }
        if (loop_config_.enable && keyframe_count == 0)
        {
            RCLCPP_ERROR(get_logger(), "Map save failed: loop mode has no keyframes");
            return false;
        }
        if (!keyframe_enable_ && !loop_config_.enable && pcl_wait_pub->empty())
        {
            RCLCPP_ERROR(get_logger(), "Map save failed: no accumulated map points");
            return false;
        }
        if (keyframe_enable_)
        {
            if (snapshot.empty())
            {
                RCLCPP_ERROR(get_logger(), "Map save failed: keyframe snapshot is empty");
                return false;
            }
            double previous_stamp = -std::numeric_limits<double>::infinity();
            for (std::size_t index = 0U; index < snapshot.size(); ++index)
            {
                const auto& keyframe = snapshot[index];
                if (keyframe.id != static_cast<std::uint64_t>(index) ||
                    !std::isfinite(keyframe.stamp) ||
                    keyframe.stamp < previous_stamp ||
                    !finiteRigidPose(keyframe.T_map_imu) ||
                    !finiteRigidPose(keyframe.T_map_lidar) ||
                    !keyframe.cloud_lidar || keyframe.cloud_lidar->empty() ||
                    !keyframe.dense_cloud_lidar ||
                    keyframe.dense_cloud_lidar->empty())
                {
                    RCLCPP_ERROR(
                        get_logger(),
                        "Map save failed: frozen keyframe snapshot is invalid at "
                        "index %zu (id=%llu stamp=%.9f coarse=%zu dense=%zu)",
                        index,
                        static_cast<unsigned long long>(keyframe.id),
                        keyframe.stamp,
                        keyframe.cloud_lidar ? keyframe.cloud_lidar->size() : 0U,
                        keyframe.dense_cloud_lidar
                            ? keyframe.dense_cloud_lidar->size() : 0U);
                    return false;
                }
                previous_stamp = keyframe.stamp;
            }
        }

        // Resolve the final pose set before constructing any output.  Every
        // artifact below (PCD, PGM, trajectory and SCDB) derives from this
        // same set, so a leveling correction cannot leave mixed frames.
        PointCloudType::Ptr map_source(new PointCloudType());
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> optimized_poses;
        bool loop_sanity_accepted = true;
        std::string loop_sanity_reason = "disabled";
        if (loop_config_.enable)
        {
            if (loop_closure_)
            {
                loop_closure_->notifyNewKeyframe();
                // [2026-08-14] 60s 不够：333 关键帧时回环线程积压未清完，
                // 保存失败（"worker timeout while processing 333 keyframes"）。
                // 放宽到 180s 兜底；根治靠 icp_max_iterations/submap 提速。
                if (!loop_closure_->waitUntilProcessed(keyframe_count, 180.0))
                {
                    RCLCPP_ERROR(get_logger(), "Loop closure worker timeout while processing %zu keyframes", keyframe_count);
                    return false;
                }
                optimized_poses = loop_closure_->recomputeOptimizedPoses(
                    snapshot, &loop_sanity_accepted, &loop_sanity_reason);
            }
            else
            {
                loop_sanity_accepted = false;
                loop_sanity_reason = "loop_closure_unavailable";
            }
            // optimizePoseGraph deliberately returns the original trajectory
            // when its all-or-nothing sanity gate rejects an optimization.
            // That fallback is useful to the public API, but it must never be
            // mistaken for a validated save generation.  Reject the save
            // explicitly and require a fresh run or operator review; the
            // map-leveling opt-out is only for single-floor geometry gates.
            if (!loop_sanity_accepted)
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed: loop pose-graph sanity rejected the "
                    "optimized trajectory (reason=%s); manual confirmation "
                    "required",
                    loop_sanity_reason.empty()
                        ? "pose_graph_sanity" : loop_sanity_reason.c_str());
                return false;
            }
            if (optimized_poses.size() != snapshot.size())
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed: optimized pose count %zu does not match "
                    "frozen snapshot %zu",
                    optimized_poses.size(), snapshot.size());
                return false;
            }
            double max_loop_dxy = 0.0;
            double max_loop_dz = 0.0;
            double max_loop_drp_deg = 0.0;
            double max_loop_dyaw_deg = 0.0;
            for (size_t index = 0; index < snapshot.size(); ++index)
            {
                const Eigen::Matrix4d& raw_pose = snapshot[index].T_map_lidar;
                const Eigen::Matrix4d& optimized_pose = optimized_poses[index];
                max_loop_dxy = std::max(
                    max_loop_dxy,
                    (optimized_pose.block<2, 1>(0, 3) -
                     raw_pose.block<2, 1>(0, 3)).norm());
                max_loop_dz = std::max(
                    max_loop_dz,
                    std::abs(optimized_pose(2, 3) - raw_pose(2, 3)));
                max_loop_drp_deg = std::max(
                    max_loop_drp_deg,
                    detail::rollPitchDeviationRadians(
                        raw_pose.block<3, 3>(0, 0),
                        optimized_pose.block<3, 3>(0, 0)) * 180.0 / PI_M);
                max_loop_dyaw_deg = std::max(
                    max_loop_dyaw_deg,
                    detail::yawDeviationRadians(
                        raw_pose.block<3, 3>(0, 0),
                        optimized_pose.block<3, 3>(0, 0)) * 180.0 / PI_M);
            }
            for (size_t index = 0; index < snapshot.size(); ++index)
            {
                PointCloudType transformed;
                pcl::transformPointCloud(
                    *snapshot[index].dense_cloud_lidar, transformed,
                    optimized_poses[index].cast<float>());
                *map_source += transformed;
            }
            RCLCPP_INFO(
                get_logger(),
                "[LOOP SUMMARY] keyframes=%zu accepted_constraints=%zu "
                "sanity=accepted reason=%s "
                "optimized_max=[xy=%.4fm z=%.4fm rp=%.3fdeg yaw=%.3fdeg]",
                snapshot.size(),
                loop_closure_ ? loop_closure_->acceptedLoopCount() : 0U,
                loop_sanity_reason.empty() ? "accepted" : loop_sanity_reason.c_str(),
                max_loop_dxy, max_loop_dz,
                max_loop_drp_deg, max_loop_dyaw_deg);
        }
        else
        {
            if (keyframe_enable_)
            {
                optimized_poses.reserve(snapshot.size());
                for (const auto& keyframe : snapshot)
                {
                    optimized_poses.push_back(keyframe.T_map_lidar);
                }
                for (std::size_t index = 0U; index < snapshot.size(); ++index)
                {
                    PointCloudType transformed;
                    pcl::transformPointCloud(
                        *snapshot[index].dense_cloud_lidar, transformed,
                        optimized_poses[index].cast<float>());
                    *map_source += transformed;
                }
            }
            else
            {
                *map_source = *pcl_wait_pub;
            }
            RCLCPP_INFO(
                get_logger(),
                "[LOOP SUMMARY] disabled sanity=disabled reason=%s",
                loop_sanity_reason.c_str());
        }
        // Non-keyframe mapping still publishes a sampled trajectory.  Bring
        // those poses into the same final-pose set so save-time leveling can
        // rotate map.txt together with the PCD/PGM artifacts.  A malformed
        // trajectory is rejected rather than leaving mixed coordinate frames.
        if (optimized_poses.empty() && !path.poses.empty())
        {
            optimized_poses.reserve(path.poses.size());
            for (const auto& stamped_pose : path.poses)
            {
                const auto& pose = stamped_pose.pose;
                const Eigen::Quaterniond orientation(
                    pose.orientation.w, pose.orientation.x,
                    pose.orientation.y, pose.orientation.z);
                if (!std::isfinite(pose.position.x) ||
                    !std::isfinite(pose.position.y) ||
                    !std::isfinite(pose.position.z) ||
                    !orientation.coeffs().allFinite() ||
                    orientation.norm() <= std::numeric_limits<double>::epsilon())
                {
                    RCLCPP_ERROR(
                        get_logger(),
                        "Map save failed: path contains a non-finite or zero-norm pose");
                    return false;
                }
                Eigen::Matrix4d trajectory_pose = Eigen::Matrix4d::Identity();
                trajectory_pose.block<3, 3>(0, 0) =
                    orientation.normalized().toRotationMatrix();
                trajectory_pose(0, 3) = pose.position.x;
                trajectory_pose(1, 3) = pose.position.y;
                trajectory_pose(2, 3) = pose.position.z;
                optimized_poses.push_back(trajectory_pose);
            }
        }
        if (map_source->empty())
        {
            RCLCPP_ERROR(get_logger(), "Map save failed: map source is empty");
            return false;
        }
        for (const auto& point : map_source->points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed: map source contains a non-finite XYZ point");
                return false;
            }
        }

        // Leveling updates the shared floor and the Pcd2Grid instance before
        // any filesystem work.  Keep those values provisional until the full
        // artifact promotion succeeds; otherwise a failed save would leave
        // the node describing a map different from the validated files on
        // disk.
        const double previous_keyframe_floor_z_map = keyframe_floor_z_map_;
        const double previous_pcd2pgm_floor_z_map = pcd2pgm_options_.floor_z_map;
        const Eigen::Matrix3d previous_map_leveling_rotation =
            map_leveling_rotation_;
        const Eigen::Vector3d previous_map_leveling_normal =
            map_leveling_normal_;
        const double previous_map_leveling_floor_z = map_leveling_floor_z_;
        const double previous_map_leveling_tilt_deg = map_leveling_tilt_deg_;
        const double previous_map_leveling_inlier_ratio =
            map_leveling_inlier_ratio_;
        const int previous_map_leveling_iterations = map_leveling_iterations_;
        const bool previous_map_leveling_succeeded = map_leveling_succeeded_;
        const double previous_map_leveling_selected_quantile =
            map_leveling_selected_quantile_;
        const bool previous_map_leveling_q10_attempted =
            map_leveling_q10_attempted_;
        const bool previous_map_leveling_q10_accepted =
            map_leveling_q10_accepted_;
        const std::size_t previous_map_leveling_seed_candidate_points =
            map_leveling_seed_candidate_points_;
        const std::size_t previous_map_leveling_seed_sample_cells =
            map_leveling_seed_sample_cells_;
        const std::size_t previous_map_leveling_seed_inlier_cells =
            map_leveling_seed_inlier_cells_;
        const double previous_map_leveling_seed_inlier_ratio =
            map_leveling_seed_inlier_ratio_;
        const double previous_map_leveling_seed_residual_p95 =
            map_leveling_seed_residual_p95_;
        const double previous_map_leveling_seed_tilt_deg =
            map_leveling_seed_tilt_deg_;
        const std::size_t previous_map_leveling_refit_candidate_points =
            map_leveling_refit_candidate_points_;
        const std::size_t previous_map_leveling_refit_sample_cells =
            map_leveling_refit_sample_cells_;
        const std::size_t previous_map_leveling_refit_inlier_cells =
            map_leveling_refit_inlier_cells_;
        const double previous_map_leveling_refit_inlier_ratio =
            map_leveling_refit_inlier_ratio_;
        const double previous_map_leveling_refit_residual_p95 =
            map_leveling_refit_residual_p95_;
        const double previous_map_leveling_refit_major_span =
            map_leveling_refit_major_span_;
        const double previous_map_leveling_refit_minor_span =
            map_leveling_refit_minor_span_;
        const std::string previous_map_leveling_failure_reason =
            map_leveling_failure_reason_;
        const auto previous_map_z_observations = map_z_observations_;
        const auto previous_map_z_correction_result = map_z_correction_result_;
        const double previous_map_z_final_floor_span = map_z_final_floor_span_;
        const auto previous_pcd2grid = pcd2grid_ptr_;
        const auto restore_provisional_leveling = [&]() noexcept
        {
            keyframe_floor_z_map_ = previous_keyframe_floor_z_map;
            pcd2pgm_options_.floor_z_map = previous_pcd2pgm_floor_z_map;
            map_leveling_rotation_ = previous_map_leveling_rotation;
            map_leveling_normal_ = previous_map_leveling_normal;
            map_leveling_floor_z_ = previous_map_leveling_floor_z;
            map_leveling_tilt_deg_ = previous_map_leveling_tilt_deg;
            map_leveling_inlier_ratio_ = previous_map_leveling_inlier_ratio;
            map_leveling_iterations_ = previous_map_leveling_iterations;
            map_leveling_succeeded_ = previous_map_leveling_succeeded;
            map_leveling_selected_quantile_ = previous_map_leveling_selected_quantile;
            map_leveling_q10_attempted_ = previous_map_leveling_q10_attempted;
            map_leveling_q10_accepted_ = previous_map_leveling_q10_accepted;
            map_leveling_seed_candidate_points_ = previous_map_leveling_seed_candidate_points;
            map_leveling_seed_sample_cells_ = previous_map_leveling_seed_sample_cells;
            map_leveling_seed_inlier_cells_ = previous_map_leveling_seed_inlier_cells;
            map_leveling_seed_inlier_ratio_ = previous_map_leveling_seed_inlier_ratio;
            map_leveling_seed_residual_p95_ = previous_map_leveling_seed_residual_p95;
            map_leveling_seed_tilt_deg_ = previous_map_leveling_seed_tilt_deg;
            map_leveling_refit_candidate_points_ = previous_map_leveling_refit_candidate_points;
            map_leveling_refit_sample_cells_ = previous_map_leveling_refit_sample_cells;
            map_leveling_refit_inlier_cells_ = previous_map_leveling_refit_inlier_cells;
            map_leveling_refit_inlier_ratio_ = previous_map_leveling_refit_inlier_ratio;
            map_leveling_refit_residual_p95_ = previous_map_leveling_refit_residual_p95;
            map_leveling_refit_major_span_ = previous_map_leveling_refit_major_span;
            map_leveling_refit_minor_span_ = previous_map_leveling_refit_minor_span;
            map_leveling_failure_reason_ = previous_map_leveling_failure_reason;
            map_z_observations_ = previous_map_z_observations;
            map_z_correction_result_ = previous_map_z_correction_result;
            map_z_final_floor_span_ = previous_map_z_final_floor_span;
            pcd2grid_ptr_ = previous_pcd2grid;
        };
        ScopeGuard leveling_state_guard(restore_provisional_leveling);
        // Apply the rigid gravity correction before extracting per-keyframe
        // ground observations.  The local plane tilt is a safety diagnostic
        // for residual roll/pitch; measuring it in the pre-leveling frame
        // would classify the known global installation tilt as a local
        // failure and make the path-coverage gate reject otherwise usable
        // observations.  Leveling only rotates the common map frame; the
        // subsequent Z correction still changes translation z only.
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
            raw_keyframe_poses;
        if (keyframe_enable_)
        {
            raw_keyframe_poses.reserve(snapshot.size());
            for (const auto& keyframe : snapshot)
            {
                raw_keyframe_poses.push_back(keyframe.T_map_lidar);
            }
        }
        if (!levelMapForSave(*map_source, optimized_poses))
        {
            if (map_leveling_enforce_gates_)
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed: gravity leveling did not pass its quality gates");
                return false;
            }
            // levelMapForSave already rolled the point cloud, poses and
            // metrics back to the pre-leveling state; map_leveling_failure_reason_
            // records why the correction was skipped.  Save the un-corrected
            // geometry instead of refusing, e.g. for cross-floor maps where a
            // single-floor ground model can never pass.
            RCLCPP_WARN(
                get_logger(),
                "Map leveling gate failed (%s); saving map without "
                "gravity-leveling correction (map_leveling.enforce_quality_gates=false)",
                map_leveling_last_gate_reason_.c_str());
            map_leveling_succeeded_ = false;
        }
        const bool leveling_applied_for_save =
            map_leveling_enabled_ && map_leveling_succeeded_;
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
            leveled_raw_poses = raw_keyframe_poses;
        if (keyframe_enable_)
        {
            Eigen::Matrix4d leveling_transform = Eigen::Matrix4d::Identity();
            leveling_transform.block<3, 3>(0, 0) = map_leveling_rotation_;
            for (auto& pose : leveled_raw_poses)
            {
                pose = leveling_transform * pose;
            }
        }
        const auto pre_local_correction_poses = optimized_poses;
        if (!correctKeyframeZForSave(snapshot, optimized_poses, *map_source))
        {
            // A late failure can leave partial per-keyframe z corrections
            // behind; the pre-correction snapshot is the only pose set that
            // matches the un-corrected map_source.  Zero-fill the correction
            // vector so the audit CSV stays size-consistent.
            optimized_poses = pre_local_correction_poses;
            map_z_correction_result_.corrections.assign(snapshot.size(), 0.0);
            map_z_correction_result_.used_observations.assign(
                snapshot.size(), false);
            if (map_z_observations_.size() != snapshot.size())
            {
                // Early-exit failures never populated the per-keyframe
                // observations; pad so the audit CSV stays writable.
                map_z_observations_.resize(snapshot.size());
            }
            if (map_leveling_enforce_gates_)
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed: per-keyframe vertical drift correction did not pass");
                return false;
            }
            RCLCPP_WARN(
                get_logger(),
                "Map Z correction gate failed (%s); saving map without "
                "per-keyframe vertical correction (map_leveling.enforce_quality_gates=false)",
                map_z_correction_result_.failure_reason.c_str());
        }
        if (keyframe_enable_)
        {
            const auto loop_report = detail::evaluatePoseGraphSanity(
                leveled_raw_poses, pre_local_correction_poses, loop_config_);
            const auto local_report = detail::evaluatePoseGraphSanity(
                pre_local_correction_poses, optimized_poses, loop_config_);
            const auto final_report = detail::evaluatePoseGraphSanity(
                leveled_raw_poses, optimized_poses, loop_config_);
            RCLCPP_INFO(
                get_logger(),
                "[FINAL POSE GATE] loop=[xy=%.4fm z=%.4fm rp=%.3fdeg "
                "yaw=%.3fdeg] local=[xy=%.4fm z=%.4fm rp=%.3fdeg "
                "yaw=%.3fdeg] composite=[xy=%.4fm z=%.4fm rp=%.3fdeg "
                "yaw=%.3fdeg] accepted=%d",
                loop_report.max_xy_deviation, loop_report.max_z_deviation,
                loop_report.max_roll_pitch_deviation_rad * 180.0 / PI_M,
                loop_report.max_yaw_deviation_rad * 180.0 / PI_M,
                local_report.max_xy_deviation, local_report.max_z_deviation,
                local_report.max_roll_pitch_deviation_rad * 180.0 / PI_M,
                local_report.max_yaw_deviation_rad * 180.0 / PI_M,
                final_report.max_xy_deviation, final_report.max_z_deviation,
                final_report.max_roll_pitch_deviation_rad * 180.0 / PI_M,
                final_report.max_yaw_deviation_rad * 180.0 / PI_M,
                final_report.accepted ? 1 : 0);
            if (!detail::acceptsFinalPoseSafetyGate(
                    loop_report, local_report, final_report))
            {
                // The cross-floor opt-out applies only to geometry/leveling
                // quality.  A rejected or non-finite composed pose is always
                // fail-closed, otherwise a map could be published with a
                // corrupted trajectory merely because leveling was disabled.
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed: final composed pose safety gate rejected "
                    "(loop=%s local=%s composite=%s index=%zu)",
                    loop_report.rejection_reason.c_str(),
                    local_report.rejection_reason.c_str(),
                    final_report.rejection_reason.c_str(),
                    final_report.offending_index);
                return false;
            }
        }
        // Pcd2Grid owns a value copy of its options.  Recreate it after
        // leveling so the newly fitted floor_z_map is consumed by height
        // filtering, evidence classification, and the emitted YAML/PGM.
        pcd2grid_ptr_ = std::make_shared<Pcd2Grid>(pcd2pgm_options_);
        // One generation identifier binds the sidecar to the exact staged
        // artifact set.  It is generated before any output is written so the
        // same value can be checked again after promotion.
        const std::string save_generation =
            "g" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count());
        const auto write_leveling_sidecar =
            [&](const std::string& output_file, bool transaction_valid)
        {
            const auto finite_or = [](double value, double fallback)
            {
                return std::isfinite(value) ? value : fallback;
            };
            // transaction_valid=false is the intentional poison write used
            // when a later save step already failed.  leveling_applied records
            // whether the gravity-leveling correction is actually present in
            // the saved geometry; a gate-failed save is valid but unleveled.
            const bool leveling_applied =
                transaction_valid && leveling_applied_for_save;
            if (leveling_applied && map_leveling_enabled_)
            {
                if (!map_leveling_rotation_.allFinite() ||
                    !map_leveling_normal_.allFinite() ||
                    !std::isfinite(map_leveling_floor_z_) ||
                    !std::isfinite(map_leveling_selected_quantile_) ||
                    !std::isfinite(map_leveling_inlier_ratio_) ||
                    !std::isfinite(map_leveling_tilt_deg_) ||
                    map_leveling_iterations_ < 1 ||
                    !std::isfinite(map_leveling_seed_inlier_ratio_) ||
                    !std::isfinite(map_leveling_seed_residual_p95_) ||
                    !std::isfinite(map_leveling_seed_tilt_deg_) ||
                    !std::isfinite(map_leveling_refit_inlier_ratio_) ||
                    !std::isfinite(map_leveling_refit_residual_p95_) ||
                    !std::isfinite(map_leveling_refit_major_span_) ||
                    !std::isfinite(map_leveling_refit_minor_span_))
                {
                    RCLCPP_ERROR(
                        get_logger(),
                        "Map save failed: leveling sidecar metrics are non-finite");
                    return false;
                }
                const Eigen::Matrix3d rotation = map_leveling_rotation_;
                if ((rotation.transpose() * rotation -
                     Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1e-4 ||
                    std::abs(rotation.determinant() - 1.0) > 1e-4 ||
                    map_leveling_normal_.norm() < 0.99 ||
                    map_leveling_normal_.z() <= 0.0)
                {
                    RCLCPP_ERROR(
                        get_logger(),
                        "Map save failed: leveling sidecar rotation/normal is invalid");
                    return false;
                }
            }
            std::vector<anubis_mapping::MapArtifactDigest> artifacts;
            if (transaction_valid)
            {
                std::string digest_error;
                const std::filesystem::path artifact_root =
                    std::filesystem::path(output_file).parent_path();
                if (!anubis_mapping::collectMapArtifactDigests(
                        artifact_root, "map_leveling.yaml", artifacts,
                        &digest_error))
                {
                    RCLCPP_ERROR(
                        get_logger(),
                        "Map save failed: cannot construct v3 artifact manifest for %s: %s",
                        artifact_root.string().c_str(), digest_error.c_str());
                    return false;
                }
            }
            std::ofstream leveling_output(
                output_file, std::ios::out | std::ios::trunc);
            if (!leveling_output.is_open())
            {
                RCLCPP_ERROR(
                    get_logger(), "Map save failed: cannot open %s",
                    output_file.c_str());
                return false;
            }
            const Eigen::Matrix3d sidecar_rotation = leveling_applied
                ? map_leveling_rotation_ : Eigen::Matrix3d::Identity();
            const Eigen::Vector3d sidecar_normal = leveling_applied
                ? map_leveling_normal_ : Eigen::Vector3d::UnitZ();
            const Eigen::Vector3d leveling_rpy = canonicalRpy(sidecar_rotation);
            std::string failure_reason = map_leveling_failure_reason_;
            for (char& character : failure_reason)
            {
                if (static_cast<unsigned char>(character) < 0x20U)
                {
                    character = ' ';
                }
            }
            if (failure_reason.empty() && !transaction_valid)
            {
                failure_reason = "map save transaction invalidated";
            }
            if (failure_reason.empty() && transaction_valid && !leveling_applied)
            {
                failure_reason = !map_leveling_last_gate_reason_.empty()
                    ? map_leveling_last_gate_reason_
                    : "leveling gate failed without a recorded reason";
            }
            leveling_output << std::setprecision(
                std::numeric_limits<double>::max_digits10)
                << "schema: map_leveling\n"
                << "schema_version: 3\n"
                << "generation: "
                << (transaction_valid ? save_generation :
                    std::string("invalid-") + save_generation)
                << "\n"
                << "artifact_count: " << artifacts.size() << "\n"
                << "enabled: " << (map_leveling_enabled_ ? "true" : "false") << "\n"
                << "succeeded: " << (transaction_valid ? "true" : "false") << "\n"
                << "leveling_applied: "
                << (leveling_applied ? "true" : "false") << "\n"
                << "correction_rpy_rad: [" << leveling_rpy.x() << ", "
                << leveling_rpy.y() << ", " << leveling_rpy.z() << "]\n"
                << "correction_rpy_deg: [" << leveling_rpy.x() * 180.0 / PI_M << ", "
                << leveling_rpy.y() * 180.0 / PI_M << ", "
                << leveling_rpy.z() * 180.0 / PI_M << "]\n"
                << "correction_rotation: ["
                << sidecar_rotation(0, 0) << ", "
                << sidecar_rotation(0, 1) << ", "
                << sidecar_rotation(0, 2) << ", "
                << sidecar_rotation(1, 0) << ", "
                << sidecar_rotation(1, 1) << ", "
                << sidecar_rotation(1, 2) << ", "
                << sidecar_rotation(2, 0) << ", "
                << sidecar_rotation(2, 1) << ", "
                << sidecar_rotation(2, 2) << "]\n"
                << "floor_z_map: " << (leveling_applied
                    ? finite_or(map_leveling_floor_z_, keyframe_floor_z_map_)
                    : finite_or(keyframe_floor_z_map_, 0.0)) << "\n"
                << "configured_sample_quantile: " << finite_or(
                    map_leveling_sample_quantile_, 0.05) << "\n"
                << "selected_sample_quantile: " << finite_or(
                    leveling_applied ? map_leveling_selected_quantile_ :
                        map_leveling_sample_quantile_, 0.05) << "\n"
                << "fallback_sample_quantile: " << finite_or(
                    map_leveling_fallback_sample_quantile_, 0.10) << "\n"
                << "ground_normal: [" << sidecar_normal.x() << ", "
                << sidecar_normal.y() << ", " << sidecar_normal.z() << "]\n"
                << "inlier_ratio: " << (leveling_applied
                    ? finite_or(map_leveling_inlier_ratio_, 0.0) : 0.0) << "\n"
                << "final_tilt_deg: " << (leveling_applied
                    ? finite_or(map_leveling_tilt_deg_, 0.0) : 0.0) << "\n"
                << "iterations: "
                << (leveling_applied ? map_leveling_iterations_ : 0) << "\n"
                << "q10_attempted: "
                << ((leveling_applied && map_leveling_q10_attempted_)
                        ? "true" : "false") << "\n"
                << "q10_accepted: "
                << ((leveling_applied && map_leveling_q10_accepted_)
                        ? "true" : "false") << "\n"
                << "seed_candidate_points: "
                << (leveling_applied ? map_leveling_seed_candidate_points_ : 0U)
                << "\n"
                << "seed_sample_cells: "
                << (leveling_applied ? map_leveling_seed_sample_cells_ : 0U)
                << "\n"
                << "seed_inlier_cells: "
                << (leveling_applied ? map_leveling_seed_inlier_cells_ : 0U)
                << "\n"
                << "seed_inlier_ratio: " << (leveling_applied
                    ? finite_or(map_leveling_seed_inlier_ratio_, 0.0) : 0.0)
                << "\n"
                << "seed_residual_p95: " << (leveling_applied
                    ? finite_or(map_leveling_seed_residual_p95_, 0.0) : 0.0)
                << "\n"
                << "seed_tilt_deg: " << (leveling_applied
                    ? finite_or(map_leveling_seed_tilt_deg_, 0.0) : 0.0)
                << "\n"
                << "refit_candidate_points: "
                << (leveling_applied ? map_leveling_refit_candidate_points_ : 0U)
                << "\n"
                << "refit_sample_cells: "
                << (leveling_applied ? map_leveling_refit_sample_cells_ : 0U)
                << "\n"
                << "refit_inlier_cells: "
                << (leveling_applied ? map_leveling_refit_inlier_cells_ : 0U)
                << "\n"
                << "refit_inlier_ratio: " << (leveling_applied
                    ? finite_or(map_leveling_refit_inlier_ratio_, 0.0) : 0.0)
                << "\n"
                << "refit_residual_p95: " << (leveling_applied
                    ? finite_or(map_leveling_refit_residual_p95_, 0.0) : 0.0)
                << "\n"
                << "refit_major_span: " << (leveling_applied
                    ? finite_or(map_leveling_refit_major_span_, 0.0) : 0.0)
                << "\n"
                << "refit_minor_span: " << (leveling_applied
                    ? finite_or(map_leveling_refit_minor_span_, 0.0) : 0.0)
                << "\n"
                << "snapshot_keyframes: " << snapshot.size() << "\n"
                << "dense_ground_leaf: " << keyframe_ground_store_leaf_ << "\n"
                << "pcd_save_voxel: " << finite_or(
                    pcd2pgm_options_.save_voxel, 0.1) << "\n"
                << "pgm_map_resolution: " << finite_or(
                    pcd2pgm_options_.map_resolution, 0.05) << "\n"
                << "pgm_min_points_occupied: "
                << std::max(0, pcd2pgm_options_.min_points_occupied) << "\n"
                << "pgm_thre_radius: " << finite_or(
                    pcd2pgm_options_.thre_radius, 0.0) << "\n"
                << "pgm_thres_point_count: "
                << std::max(0, pcd2pgm_options_.thres_point_count) << "\n"
                << "pgm_overhead_min_ray_crossing_frames: "
                << std::max(0,
                    pcd2pgm_options_.overhead_min_ray_crossing_frames) << "\n"
                << "pgm_overhead_min_ray_crossing_ratio: "
                << finite_or(
                    pcd2pgm_options_.overhead_min_ray_crossing_ratio, 0.10)
                << "\n"
                << "grid_source: dense_map_source\n"
                << "z_correction_enabled: "
                << (map_z_correction_enabled_ ? "true" : "false") << "\n"
                << "z_correction_succeeded: "
                << (map_z_correction_enabled_
                        ? (map_z_correction_result_.success ? "true" : "false")
                        : "true") << "\n"
                << "z_correction_frame_valid_fraction: "
                << finite_or(map_z_correction_result_.valid_fraction, 0.0) << "\n"
                << "z_correction_path_coverage_fraction: "
                << finite_or(map_z_correction_result_.path_coverage_fraction, 0.0) << "\n"
                << "z_correction_max_unobserved_gap_m: "
                << finite_or(map_z_correction_result_.max_unobserved_gap_m, 0.0) << "\n"
                << "z_correction_invalid_index_ranges: "
                << map_z_correction_result_.invalid_index_ranges << "\n"
                << "time_guard_skipped_frames: "
                << lio_time_guard_session_.rejected_frames << "\n"
                << "time_guard_failed: "
                << (lio_time_guard_session_.failed ? "true" : "false") << "\n"
                << "time_guard_failure_reason: "
                << lio_time_guard_session_.failure_reason << "\n"
                << "final_fixed_floor_span_m: "
                << finite_or(map_z_final_floor_span_, 0.0) << "\n"
                << "failure_reason: " << failure_reason << "\n";
            for (std::size_t index = 0U; index < artifacts.size(); ++index)
            {
                const auto& artifact = artifacts[index];
                leveling_output
                    << "artifact_" << index << "_path: "
                    << artifact.relative_path.generic_string() << "\n"
                    << "artifact_" << index << "_size: "
                    << artifact.size << "\n"
                    << "artifact_" << index << "_sha256: "
                    << artifact.sha256 << "\n";
            }
            leveling_output.flush();
            if (!leveling_output.good())
            {
                RCLCPP_ERROR(
                    get_logger(), "Map save failed while writing %s",
                    output_file.c_str());
                return false;
            }
            return true;
        };

        const std::string staging_suffix = save_generation;
        std::filesystem::path staging_dir;
        for (int attempt = 0; attempt < 8 && staging_dir.empty(); ++attempt)
        {
            const std::filesystem::path candidate =
                std::filesystem::path(data_path_) /
                (".map_save_tmp_" + staging_suffix +
                 (attempt == 0 ? std::string() : "_" + std::to_string(attempt)));
            std::error_code create_error;
            if (std::filesystem::create_directory(candidate, create_error))
            {
                staging_dir = candidate;
            }
            else if (create_error &&
                     create_error != std::make_error_code(std::errc::file_exists))
            {
                RCLCPP_ERROR(
                    get_logger(), "Map save failed: cannot create staging directory %s: %s",
                    candidate.string().c_str(), create_error.message().c_str());
                return false;
            }
        }
        if (staging_dir.empty())
        {
            RCLCPP_ERROR(
                get_logger(), "Map save failed: staging directory name collision");
            return false;
        }
        const auto cleanup_staging = [&]()
        {
            std::error_code cleanup_error;
            std::filesystem::remove_all(staging_dir, cleanup_error);
            if (cleanup_error)
            {
                RCLCPP_WARN(
                    get_logger(), "Could not remove map staging directory %s: %s",
                    staging_dir.string().c_str(), cleanup_error.message().c_str());
            }
        };

        // Keep the sidecar in staging until every artifact is ready.  The
        // destination sidecar is invalidated immediately before promotion.
        const std::string staged_leveling_file =
            (staging_dir / "map_leveling.yaml").string();
        // VoxelGrid averages every field, so the FeatureLabel in normal_x does
        // not survive this step; the saved map is unlabelled by design here.
        pcl::PointCloud<PointType>::Ptr map_downsampled(new pcl::PointCloud<PointType>());
        pcl::VoxelGrid<PointType> save_voxel;
        save_voxel.setInputCloud(map_source);
        const float save_voxel_leaf = static_cast<float>(
            pcd2pgm_options_.save_voxel);
        if (!std::isfinite(save_voxel_leaf) || save_voxel_leaf <= 0.0F)
        {
            RCLCPP_ERROR(
                get_logger(),
                "Map save failed: pcd2pgm.save_voxel is not representable as a positive float");
            cleanup_staging();
            return false;
        }
        save_voxel.setLeafSize(
            save_voxel_leaf, save_voxel_leaf, save_voxel_leaf);
        save_voxel.filter(*map_downsampled);
        if (map_downsampled->empty())
        {
            RCLCPP_ERROR(get_logger(), "Map save failed: voxel filtering removed all map points");
            cleanup_staging();
            return false;
        }
        for (const auto& point : map_downsampled->points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed: downsampled PCD contains a non-finite XYZ point");
                cleanup_staging();
                return false;
            }
        }

        bool save_success = true;
        const std::string file_name = (staging_dir / "map.pcd").string();
        pcl::PCDWriter pcd_writer;
        RCLCPP_INFO(get_logger(), "Saving PCD %s (%zu -> %zu points)",
            file_name.c_str(), map_source->size(), map_downsampled->size());
        if (pcd_writer.writeBinary(file_name, *map_downsampled) != 0)
        {
            RCLCPP_ERROR(get_logger(), "Failed to save PCD file: %s", file_name.c_str());
            save_success = false;
        }
        if (save_success && keyframe_enable_)
        {
            const std::string snapshot_file =
                (staging_dir / "map_snapshot.csv").string();
            std::ofstream snapshot_output(
                snapshot_file, std::ios::out | std::ios::trunc);
            if (!snapshot_output.is_open() ||
                optimized_poses.size() != snapshot.size())
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Failed to create frozen map snapshot artifact: %s",
                    snapshot_file.c_str());
                save_success = false;
            }
            else
            {
                snapshot_output << std::setprecision(
                    std::numeric_limits<double>::max_digits10)
                    << "index,keyframe_id,stamp,coarse_points,dense_points,"
                       "coarse_fingerprint_fnv64,dense_fingerprint_fnv64,"
                       "raw_x,raw_y,raw_z,final_x,final_y,final_z\n";
                for (std::size_t index = 0U; index < snapshot.size(); ++index)
                {
                    const auto& raw = snapshot[index].T_map_lidar;
                    const auto& final = optimized_poses[index];
                    snapshot_output
                        << index << ',' << snapshot[index].id << ','
                        << snapshot[index].stamp << ','
                        << snapshot[index].cloud_lidar->size() << ','
                        << snapshot[index].dense_cloud_lidar->size() << ','
                        << std::hex << std::setw(16) << std::setfill('0')
                        << cloudFingerprint(*snapshot[index].cloud_lidar) << ','
                        << std::setw(16)
                        << cloudFingerprint(*snapshot[index].dense_cloud_lidar)
                        << std::dec << std::setfill(' ') << ','
                        << raw(0, 3) << ',' << raw(1, 3) << ',' << raw(2, 3)
                        << ',' << final(0, 3) << ',' << final(1, 3) << ','
                        << final(2, 3) << '\n';
                }
                snapshot_output.flush();
                if (!snapshot_output.good())
                {
                    RCLCPP_ERROR(
                        get_logger(),
                        "Failed while writing frozen map snapshot artifact: %s",
                        snapshot_file.c_str());
                    save_success = false;
                }
            }
        }

        const std::string pcd2grid_dir =
            (staging_dir / pcd2pgm_options_.file_name).string();
        // Keep the dense source for the 5 cm PGM.  The 10 cm voxelized cloud
        // remains the PCD artifact only; using it for the grid creates a
        // one-point-per-cell salt-and-pepper pattern at half the voxel size.
        PointCloudType::Ptr grid_source = map_source;
        // [2026-08-14] 按关键帧轨迹 ±grid_crop_margin 裁剪：透过门窗看到的
        // 远处杂点会把栅格图撑大（实测轨迹只有 14m，栅格图却拉到 53.7m、
        // 91% 是未知区），既拖慢导航栈加载，也让 map.yaml 的 origin 偏移到
        // 无关区域。裁剪只影响 2D 栅格图，PCD 点云保持完整（定位要用）。
        if (keyframe_enable_ && grid_crop_margin_ > 0.0)
        {
            if (!snapshot.empty())
            {
                double x_min = 1e9, x_max = -1e9, y_min = 1e9, y_max = -1e9;
                for (size_t index = 0; index < snapshot.size(); ++index)
                {
                    const Eigen::Matrix4d& pose =
                        optimized_poses[index];
                    x_min = std::min(x_min, pose(0, 3));
                    x_max = std::max(x_max, pose(0, 3));
                    y_min = std::min(y_min, pose(1, 3));
                    y_max = std::max(y_max, pose(1, 3));
                }
                PointCloudType::Ptr cropped(new PointCloudType());
                cropped->reserve(grid_source->size());
                for (const auto& point : grid_source->points)
                {
                    if (point.x > x_min - grid_crop_margin_ &&
                        point.x < x_max + grid_crop_margin_ &&
                        point.y > y_min - grid_crop_margin_ &&
                        point.y < y_max + grid_crop_margin_)
                    {
                        cropped->push_back(point);
                    }
                }
                if (!cropped->empty())
                {
                    RCLCPP_INFO(get_logger(),
                        "Grid crop (traj +/-%.1fm): %zu -> %zu points",
                        grid_crop_margin_, grid_source->size(), cropped->size());
                    grid_source = cropped;
                }
            }
        }
        if (!grid_source || grid_source->empty())
        {
            RCLCPP_ERROR(get_logger(),
                         "Map save failed: grid source is empty after cropping");
            cleanup_staging();
            return false;
        }
        for (const auto& point : grid_source->points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed: grid source contains a non-finite XYZ point");
                cleanup_staging();
                return false;
            }
        }
        std::vector<GroundPlanePoint> final_ground_points;
        final_ground_points.reserve(grid_source->size());
        for (const auto& point : grid_source->points)
        {
            final_ground_points.push_back({point.x, point.y, point.z});
        }
        GroundPlaneFitOptions final_ground_options;
        final_ground_options.fixed_floor_z = keyframe_floor_z_map_;
        final_ground_options.candidate_min_height =
            pcd2pgm_options_.ground_plane_candidate_min_height;
        final_ground_options.candidate_max_height =
            pcd2pgm_options_.ground_plane_candidate_max_height;
        // Match the save-time leveling envelope instead of relying on the
        // generic GroundPlaneFitOptions defaults.  The final geometry gate
        // must measure the same dense, low-envelope evidence that drove the
        // rigid/non-rigid corrections above.
        final_ground_options.sample_cell_size = 0.20;
        final_ground_options.min_points_per_cell = 1;
        final_ground_options.sample_quantile =
            std::clamp(
                std::isfinite(map_leveling_selected_quantile_)
                    ? map_leveling_selected_quantile_
                    : map_leveling_sample_quantile_,
                0.01, 0.20);
        final_ground_options.min_sample_cells = static_cast<std::size_t>(
            std::max(0, pcd2pgm_options_.ground_plane_min_sample_cells));
        final_ground_options.min_inlier_ratio =
            pcd2pgm_options_.ground_plane_min_inlier_ratio;
        final_ground_options.max_tilt_deg =
            pcd2pgm_options_.ground_plane_max_tilt_deg;
        final_ground_options.max_residual_p95 =
            pcd2pgm_options_.ground_plane_max_residual_p95;
        final_ground_options.max_floor_offset =
            pcd2pgm_options_.ground_plane_max_floor_offset;
        // This is the fail-closed geometry gate, not the adaptive PGM model.
        // A quadratic fit can hide the exact non-rigid floor curvature that
        // the per-keyframe correction is required to remove.
        final_ground_options.quadratic_enabled = false;
        final_ground_options.quadratic_max_local_tilt_deg =
            pcd2pgm_options_.ground_plane_quadratic_max_local_tilt_deg;
        final_ground_options.quadratic_max_curvature =
            pcd2pgm_options_.ground_plane_quadratic_max_curvature;
        const GroundPlaneModel final_ground_model = fit_ground_plane(
            final_ground_points, final_ground_options);
        map_z_final_floor_span_ = final_ground_model.fitted_floor_max -
            final_ground_model.fitted_floor_min;
        const bool final_span_finite =
            std::isfinite(map_z_final_floor_span_);
        const bool final_span_within_limit = final_span_finite &&
            map_z_final_floor_span_ <= map_leveling_max_final_floor_span_m_;
        // The disabled-ground-plane run is an explicit A/B control. It must
        // still produce the fixed-floor PGM, so do not require the optional
        // adaptive fit to pass in that mode. The adaptive-fit requirement is
        // independent from the save-time leveling/pose gate below.
        const bool final_ground_model_required_failure =
            !ground_plane_fallback_allowed(
                pcd2pgm_options_.ground_plane_enabled,
                pcd2pgm_options_.ground_plane_required,
                final_ground_model.adaptive);
        if (final_ground_model_required_failure)
        {
            RCLCPP_ERROR(
                get_logger(),
                "Map save failed: required PGM ground model did not pass "
                "(model=%s reason=%s; set pcd2pgm.ground_plane_required=false "
                "only for an explicit fixed-floor/cross-floor run)",
                final_ground_model.model_name(),
                final_ground_model.fallback_reason.c_str());
            cleanup_staging();
            return false;
        }
        if (!final_span_finite || !final_span_within_limit)
        {
            if (map_leveling_enforce_gates_)
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed: final fixed-ground geometry gate did not pass "
                    "(model=%s span=%.6fm limit=%.6fm reason=%s)",
                    final_ground_model.model_name(), map_z_final_floor_span_,
                    map_leveling_max_final_floor_span_m_,
                    final_ground_model.fallback_reason.c_str());
                cleanup_staging();
                return false;
            }
            RCLCPP_WARN(
                get_logger(),
                "Final fixed-ground geometry gate did not pass; saving map "
                "anyway (map_leveling.enforce_quality_gates=false): "
                "model=%s span=%.6fm limit=%.6fm reason=%s",
                final_ground_model.model_name(), map_z_final_floor_span_,
                map_leveling_max_final_floor_span_m_,
                final_ground_model.fallback_reason.c_str());
        }
        if (pcd2pgm_options_.ground_plane_enabled)
        {
            RCLCPP_INFO(
                get_logger(),
                "Final fixed-ground geometry accepted: model=%s "
                "fitted_floor_range=[%.6f,%.6f]m span=%.6fm limit=%.6fm "
                "residual_p95=%.4fm inlier_ratio=%.3f",
                final_ground_model.model_name(),
                final_ground_model.fitted_floor_min,
                final_ground_model.fitted_floor_max,
                map_z_final_floor_span_, map_leveling_max_final_floor_span_m_,
                final_ground_model.residual_p95,
                final_ground_model.inlier_ratio);
        }
        else
        {
            RCLCPP_INFO(
                get_logger(),
                "Final fixed-ground geometry A/B control: adaptive_fit=%s "
                "fitted_floor_range=[%.6f,%.6f]m span=%.6fm "
                "limit=%.6fm; adaptive PGM filtering disabled",
                final_ground_model.adaptive ? "available" : "unavailable",
                final_ground_model.fitted_floor_min,
                final_ground_model.fitted_floor_max,
                map_z_final_floor_span_, map_leveling_max_final_floor_span_m_);
        }
        Pcd2GridScans grid_scans;
        if (keyframe_enable_)
        {
            grid_scans.reserve(snapshot.size());
            for (size_t index = 0; index < snapshot.size(); ++index)
            {
                Pcd2GridScan scan;
                scan.T_map_lidar = optimized_poses[index];
                scan.cloud_lidar = snapshot[index].dense_cloud_lidar;
                grid_scans.push_back(std::move(scan));
            }
        }
        if (!pcd2grid_ptr_->run(grid_source, pcd2grid_dir, grid_scans))
        {
            RCLCPP_ERROR(get_logger(), "Failed to save PGM/YAML map: %s", pcd2grid_dir.c_str());
            save_success = false;
        }

        const std::string path_file = (staging_dir / "map.txt").string();
        std::ofstream ofs(path_file, std::ios::out | std::ios::trunc);
        if (!ofs.is_open())
        {
            RCLCPP_ERROR(get_logger(), "Failed to open trajectory file: %s", path_file.c_str());
            save_success = false;
        }
        else
        {
            ofs << "# path" << std::endl;
            if (!optimized_poses.empty())
            {
                for (const auto& pose : optimized_poses)
                {
                    const double theta = std::atan2(pose(1, 0), pose(0, 0));
                    ofs << std::fixed << std::setprecision(2)
                        << pose(0, 3) << " " << pose(1, 3) << " " << theta << std::endl;
                }
            }
            else for (const auto& p : path.poses)
            {
                const double theta = QuaternionToYaw(
                    p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w);
                ofs << std::fixed << std::setprecision(2)
                    << p.pose.position.x << " " << p.pose.position.y << " " << theta << std::endl;
            }
            ofs.close();
            if (!ofs)
            {
                RCLCPP_ERROR(get_logger(), "Failed while writing trajectory file: %s", path_file.c_str());
                save_success = false;
            }
        }

        if (save_success && map_z_correction_enabled_)
        {
            const std::string corrections_file =
                (staging_dir / "map_pose_corrections.csv").string();
            std::ofstream corrections_output(
                corrections_file, std::ios::out | std::ios::trunc);
            if (!corrections_output.is_open() ||
                map_z_observations_.size() != snapshot.size() ||
                map_z_correction_result_.corrections.size() != snapshot.size() ||
                optimized_poses.size() != snapshot.size())
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Failed to create auditable map pose correction artifact: %s",
                    corrections_file.c_str());
                save_success = false;
            }
            else
            {
                corrections_output << std::setprecision(
                    std::numeric_limits<double>::max_digits10)
                    << "index,keyframe_id,stamp,path_s,observation_valid,observation_used,"
                       "selected_window,candidate_points,sample_cells,inlier_cells,"
                       "source_count,independent_source_count,support_s_min,support_s_max,"
                       "source_ids,observed_floor_z,local_tilt_deg,residual_p95,z_correction,"
                       "final_x,final_y,final_z,final_roll,final_pitch,final_yaw,failure_reason\n";
                for (std::size_t index = 0U; index < snapshot.size(); ++index)
                {
                    const auto& observation = map_z_observations_[index];
                    const auto& pose = optimized_poses[index];
                    const Eigen::Vector3d rpy = canonicalRpy(
                        pose.block<3, 3>(0, 0));
                    std::string reason = observation.failure_reason;
                    std::replace(reason.begin(), reason.end(), ',', ';');
                    std::replace(reason.begin(), reason.end(), '\n', ' ');
                    std::replace(reason.begin(), reason.end(), '\r', ' ');
                    std::string source_ids = observation.source_ids;
                    std::replace(source_ids.begin(), source_ids.end(), ',', ';');
                    std::string diagnostics = observation.window_diagnostics;
                    std::replace(diagnostics.begin(), diagnostics.end(), ',', ';');
                    std::replace(diagnostics.begin(), diagnostics.end(), '\n', ' ');
                    std::replace(diagnostics.begin(), diagnostics.end(), '\r', ' ');
                    if (!diagnostics.empty())
                    {
                        if (!reason.empty()) reason += "; ";
                        reason += "windows=" + diagnostics;
                    }
                    corrections_output
                        << index << ',' << snapshot[index].id << ','
                        << snapshot[index].stamp << ',' << observation.path_s << ','
                        << (observation.valid ? 1 : 0) << ','
                        << (index < map_z_correction_result_.used_observations.size() &&
                                map_z_correction_result_.used_observations[index]
                            ? 1 : 0) << ','
                        << observation.selected_window << ','
                        << observation.candidate_points << ','
                        << observation.sample_cells << ','
                        << observation.inlier_cells << ','
                        << observation.source_count << ','
                        << observation.independent_source_count << ','
                        << observation.support_s_min << ','
                        << observation.support_s_max << ',' << source_ids << ','
                        << observation.floor_z << ',' << observation.local_tilt_deg << ','
                        << observation.residual_p95 << ','
                        << map_z_correction_result_.corrections[index] << ','
                        << pose(0, 3) << ',' << pose(1, 3) << ',' << pose(2, 3) << ','
                        << rpy.x() << ',' << rpy.y() << ',' << rpy.z() << ','
                        << reason << '\n';
                }
                corrections_output.flush();
                if (!corrections_output.good())
                {
                    RCLCPP_ERROR(
                        get_logger(),
                        "Failed while writing map pose correction artifact: %s",
                        corrections_file.c_str());
                    save_success = false;
                }
            }
        }

        if (save_success && !saveDescriptorDatabase(
                snapshot, optimized_poses, staging_dir.string()))
        {
            save_success = false;
        }
        if (save_success && apriltag_enable_ && !saveAprilTagMap(
                snapshot, optimized_poses, staging_dir.string()))
        {
            save_success = false;
        }
        if (save_success)
        {
            save_success = write_leveling_sidecar(staged_leveling_file, true);
            double staged_floor_z = 0.0;
            anubis_mapping::MapLevelingMetadata staged_metadata;
            if (save_success &&
                !anubis_mapping::readValidatedMapLevelingFloor(
                    staged_leveling_file, staged_floor_z, staging_dir,
                    &staged_metadata))
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Save Map Failed: staged leveling sidecar violates v3 manifest");
                save_success = false;
            }
            if (save_success && staged_metadata.generation != save_generation)
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Save Map Failed: staged manifest generation mismatch (%s != %s)",
                    staged_metadata.generation.c_str(), save_generation.c_str());
                save_success = false;
            }
        }
        if (!save_success)
        {
            RCLCPP_ERROR(get_logger(), "Save Map Failed: one or more output files could not be written");
            cleanup_staging();
            return false;
        }

        std::vector<anubis_mapping::MapArtifactPromotionEntry> promotion_entries;
        std::set<std::filesystem::path> staged_paths;
        std::error_code iterate_error;
        for (std::filesystem::recursive_directory_iterator iterator(
                 staging_dir, iterate_error), end;
             iterator != end; iterator.increment(iterate_error))
        {
            if (iterate_error)
            {
                RCLCPP_ERROR(
                    get_logger(), "Map save failed while scanning staging directory %s: %s",
                    staging_dir.string().c_str(), iterate_error.message().c_str());
                cleanup_staging();
                return false;
            }
            const auto& entry = *iterator;
            std::error_code entry_error;
            if (!entry.is_regular_file(entry_error) || entry_error)
            {
                RCLCPP_ERROR(
                    get_logger(), "Map save failed: staging contains a non-regular entry %s",
                    entry.path().string().c_str());
                cleanup_staging();
                return false;
            }
            const std::filesystem::path relative =
                std::filesystem::relative(entry.path(), staging_dir, entry_error)
                    .lexically_normal();
            if (entry_error ||
                !anubis_mapping::isSafeMapArtifactRelativePath(relative) ||
                !staged_paths.insert(relative).second)
            {
                RCLCPP_ERROR(
                    get_logger(), "Map save failed: invalid staged relative path %s",
                    entry.path().string().c_str());
                cleanup_staging();
                return false;
            }
            promotion_entries.push_back({
                relative, entry.path(),
                std::filesystem::path(data_path_) / relative, false});
        }
        if (iterate_error || promotion_entries.empty())
        {
            RCLCPP_ERROR(
                get_logger(), "Map save failed: staging directory contains no artifacts");
            cleanup_staging();
            return false;
        }
        const auto has_staged = [&](const std::filesystem::path& relative)
        {
            return std::any_of(
                promotion_entries.begin(), promotion_entries.end(),
                [&](const anubis_mapping::MapArtifactPromotionEntry& entry) {
                    return entry.relative_path == relative;
                });
        };
        const std::filesystem::path grid_prefix(
            pcd2pgm_options_.file_name);
        const std::filesystem::path required_grid_pgm =
            std::filesystem::path(grid_prefix.string() + ".pgm");
        const std::filesystem::path required_grid_yaml =
            std::filesystem::path(grid_prefix.string() + ".yaml");
        std::vector<std::filesystem::path> required_files = {
            std::filesystem::path("map.pcd"),
            required_grid_pgm,
            required_grid_yaml,
            std::filesystem::path("map.txt"),
            std::filesystem::path("map_leveling.yaml")};
        if (pcd2pgm_options_.cell_evidence_diagnostics_enabled)
        {
            required_files.emplace_back(
                std::filesystem::path(
                    grid_prefix.string() + "_cell_evidence.csv"));
        }
        if (keyframe_enable_)
        {
            required_files.emplace_back("map_snapshot.csv");
            required_files.emplace_back("map_scd.bin");
        }
        if (map_z_correction_enabled_)
        {
            required_files.emplace_back("map_pose_corrections.csv");
        }
        for (const auto& required : required_files)
        {
            if (!has_staged(required))
            {
                RCLCPP_ERROR(
                    get_logger(), "Map save failed: required staged artifact is missing: %s",
                    required.string().c_str());
                cleanup_staging();
                return false;
            }
        }
        // The staging directory describes the new generation, but a previous
        // save may have produced optional artifacts under a different
        // configuration (for example an old grid prefix or diagnostics CSV).
        // Include only canonical outputs and paths explicitly declared by a
        // semantically valid previous v3 sidecar.  Unknown historical files
        // are intentionally left untouched; the exact-set manifest check will
        // fail closed if they are still mixed into this active output dir.
        std::set<std::filesystem::path> managed_paths = {
            std::filesystem::path("map.pcd"),
            std::filesystem::path("map.txt"),
            std::filesystem::path("map_scd.bin"),
            std::filesystem::path("map_snapshot.csv"),
            std::filesystem::path("map_pose_corrections.csv"),
            std::filesystem::path(grid_prefix.string() + ".pgm"),
            std::filesystem::path(grid_prefix.string() + ".yaml"),
            std::filesystem::path(grid_prefix.string() + "_cell_evidence.csv")};
        const std::filesystem::path previous_sidecar_path =
            std::filesystem::path(data_path_) / "map_leveling.yaml";
        std::error_code previous_sidecar_error;
        if (std::filesystem::is_regular_file(
                previous_sidecar_path, previous_sidecar_error) &&
            !previous_sidecar_error)
        {
            double previous_floor = 0.0;
            anubis_mapping::MapLevelingMetadata previous_metadata;
            if (anubis_mapping::readValidatedMapLevelingFloor(
                    previous_sidecar_path.string(), previous_floor,
                    std::filesystem::path(data_path_), &previous_metadata) &&
                previous_metadata.schema_version >= 3U)
            {
                for (const auto& artifact : previous_metadata.artifacts)
                {
                    if (anubis_mapping::isSafeMapArtifactRelativePath(
                            artifact.relative_path))
                    {
                        managed_paths.insert(artifact.relative_path);
                    }
                }
            }
        }
        for (const auto& managed : managed_paths)
        {
            if (!anubis_mapping::isSafeMapArtifactRelativePath(managed) ||
                staged_paths.find(managed) != staged_paths.end())
            {
                continue;
            }
            const std::filesystem::path destination =
                std::filesystem::path(data_path_) / managed;
            std::error_code destination_error;
            const bool destination_exists = std::filesystem::exists(
                destination, destination_error);
            if (destination_error)
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed while checking stale artifact %s: %s",
                    destination.string().c_str(), destination_error.message().c_str());
                cleanup_staging();
                return false;
            }
            if (!destination_exists)
            {
                continue;
            }
            if (std::filesystem::is_directory(destination, destination_error) ||
                destination_error)
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Map save failed: managed stale artifact is a directory: %s",
                    destination.string().c_str());
                cleanup_staging();
                return false;
            }
            promotion_entries.push_back({managed, {}, destination, true});
        }

        anubis_mapping::MapArtifactPromotionCallbacks promotion_callbacks;
        promotion_callbacks.write_invalid_sidecar =
            [&](const std::filesystem::path& output_path)
        {
            return write_leveling_sidecar(output_path.string(), false);
        };
        promotion_callbacks.verify_destination =
            [&](std::string* error_message)
        {
            double verified_floor_z = 0.0;
            anubis_mapping::MapLevelingMetadata verified_metadata;
            if (!anubis_mapping::readValidatedMapLevelingFloor(
                    (std::filesystem::path(data_path_) /
                     "map_leveling.yaml").string(), verified_floor_z,
                    std::filesystem::path(data_path_), &verified_metadata))
            {
                if (error_message != nullptr)
                {
                    *error_message =
                        "final v3 sidecar/artifact set is not validated";
                }
                return false;
            }
            if (verified_metadata.generation != save_generation)
            {
                if (error_message != nullptr)
                {
                    *error_message =
                        "generation mismatch (" + verified_metadata.generation +
                        " != " + save_generation + ")";
                }
                return false;
            }
            return true;
        };

        const anubis_mapping::MapArtifactPromotionResult promotion_result =
            anubis_mapping::promoteMapArtifactsTransactional(
                promotion_entries, "map_leveling.yaml", save_generation,
                promotion_callbacks);
        if (!promotion_result.promoted)
        {
            const std::string failure_detail =
                promotion_result.detail.empty()
                    ? std::string("operation failed")
                    : promotion_result.detail;
            RCLCPP_ERROR(
                get_logger(),
                "Map promotion failed at %s (%s): %s",
                promotion_result.failed_path.string().c_str(),
                promotion_result.failed_operation.c_str(),
                failure_detail.c_str());
            if (promotion_result.rollback_attempted &&
                promotion_result.rollback_restored)
            {
                RCLCPP_ERROR(
                    get_logger(),
                    "Validated old map restored. Backup: %s",
                    promotion_result.backup_directory.string().c_str());
            }
            else if (!promotion_result.safe_terminal_state)
            {
                RCLCPP_FATAL(
                    get_logger(),
                    "Map transaction could not keep sidecar fail-closed; do not consume %s",
                    data_path_.c_str());
            }
            cleanup_staging();
            return false;
        }
        const std::filesystem::path& backup_dir =
            promotion_result.backup_directory;

        cleanup_staging();
        leveling_state_guard.dismiss();
        RCLCPP_INFO(
            get_logger(), "Save Map Success: %s (previous artifacts preserved in %s)",
            data_path_.c_str(), backup_dir.string().c_str());
        return true;
    }

    Eigen::Matrix4d MappingAlg::currentLidarPose() const
    {
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) =
            (state_point.rot * state_point.offset_R_L_I).toRotationMatrix();
        pose.block<3, 1>(0, 3) = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        return pose;
    }

    // Loop closure remains entirely in the LiDAR frame: its ICP source,
    // submaps and pose graph all use T_map_lidar.
    scan_descriptor::Descriptor MappingAlg::makeLoopDescriptor(
        const PointCloudType& cloud_lidar, const Eigen::Matrix4d& T_map_lidar) const
    {
        const double yaw = std::atan2(T_map_lidar(1, 0), T_map_lidar(0, 0));
        const Eigen::Matrix3d R_level =
            Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
            T_map_lidar.block<3, 3>(0, 0);
        return scan_descriptor::makeDescriptor(
            scan_descriptor::adaptPoints(
                cloud_lidar, R_level,
                keyframe_floor_z_map_ - T_map_lidar(2, 3),
                descriptor_config_),
            descriptor_config_);
    }

    // The exported GL database is queried with localization's base-frame scan.
    // Transform the keyframe cloud with the same T_base_lidar used by
    // localization, then normalize heading and height from T_map_base.
    scan_descriptor::Descriptor MappingAlg::makeGlobalLocalizationDescriptor(
        const PointCloudType& cloud_lidar, const Eigen::Matrix4d& T_map_base) const
    {
        struct XyzPoint { double x, y, z; };
        std::vector<XyzPoint> adapted;
        adapted.reserve(cloud_lidar.size());
        for (const auto& point : cloud_lidar.points)
        {
            const Eigen::Vector4d p_base = lidar_to_base_extrinsic_ *
                Eigen::Vector4d(point.x, point.y, point.z, 1.0);
            adapted.push_back({p_base.x(), p_base.y(), p_base.z()});
        }
        const double yaw = std::atan2(T_map_base(1, 0), T_map_base(0, 0));
        const Eigen::Matrix3d R_level =
            Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
            T_map_base.block<3, 3>(0, 0);
        return scan_descriptor::makeDescriptor(
            scan_descriptor::adaptPoints(
                adapted, R_level, keyframe_floor_z_map_ - T_map_base(2, 3),
                descriptor_config_),
            descriptor_config_);
    }

    void MappingAlg::maybeAddKeyframe()
    {
        if (!keyframe_enable_ || !feats_undistort || feats_undistort->empty()) return;
        const Eigen::Matrix4d lidar_pose = currentLidarPose();
        const size_t count = keyframe_store_->size();
        if (count > 0)
        {
            const auto previous = keyframe_store_->snapshot().back();
            if (!shouldCreateKeyframe(previous.T_map_lidar, lidar_pose,
                                      keyframe_trans_threshold_,
                                      keyframe_yaw_threshold_deg_ * PI_M / 180.0)) return;
        }
        if (count >= keyframe_max_count_)
        {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                                  "keyframe.max_count=%zu reached; stop appending keyframes", keyframe_max_count_);
            return;
        }
        // Localization transforms each raw scan into base_link before applying
        // its VoxelGrid. Use the same voxel lattice for stored keyframes so the
        // exported GL descriptors are built from the same sampling convention.
        // Transform the result back because loop descriptors, ICP and the pose
        // graph intentionally keep their existing LiDAR-frame contract.
        PointCloudType::Ptr cloud_base(new PointCloudType());
        pcl::transformPointCloud(
            *feats_undistort, *cloud_base,
            lidar_to_base_extrinsic_.cast<float>());
        pcl::VoxelGrid<PointType> voxel;
        voxel.setInputCloud(cloud_base);
        const float leaf = static_cast<float>(keyframe_store_leaf_);
        voxel.setLeafSize(leaf, leaf, leaf);
        PointCloudType::Ptr downsampled_base(new PointCloudType());
        voxel.filter(*downsampled_base);
        if (downsampled_base->empty()) return;
        PointCloudType::Ptr downsampled_lidar(new PointCloudType());
        pcl::transformPointCloud(
            *downsampled_base, *downsampled_lidar,
            lidar_to_base_extrinsic_.inverse().cast<float>());
        pcl::VoxelGrid<PointType> dense_voxel;
        dense_voxel.setInputCloud(cloud_base);
        const float dense_leaf = static_cast<float>(keyframe_ground_store_leaf_);
        dense_voxel.setLeafSize(dense_leaf, dense_leaf, dense_leaf);
        PointCloudType::Ptr dense_base(new PointCloudType());
        dense_voxel.filter(*dense_base);
        if (dense_base->empty()) return;
        PointCloudType::Ptr dense_lidar(new PointCloudType());
        pcl::transformPointCloud(
            *dense_base, *dense_lidar,
            lidar_to_base_extrinsic_.inverse().cast<float>());
        Keyframe keyframe;
        keyframe.id = static_cast<uint64_t>(count);
        keyframe.stamp = lidar_end_time;
        keyframe.T_map_imu = Eigen::Matrix4d::Identity();
        keyframe.T_map_imu.block<3, 3>(0, 0) = state_point.rot.toRotationMatrix();
        keyframe.T_map_imu.block<3, 1>(0, 3) = state_point.pos;
        keyframe.T_map_lidar = lidar_pose;
        keyframe.cloud_lidar = downsampled_lidar;
        keyframe.dense_cloud_lidar = dense_lidar;
        keyframe.desc = makeLoopDescriptor(
            *downsampled_lidar, keyframe.T_map_lidar);
        keyframe.desc_ready = scan_descriptor::validDescriptor(
            keyframe.desc, descriptor_config_);
        if (keyframe_store_->append(std::move(keyframe), keyframe_max_count_))
        {
            if (loop_closure_) loop_closure_->notifyNewKeyframe();
            if (count % 100 == 0)
            {
                const double mb = static_cast<double>(count + 1) *
                    static_cast<double>(
                        (downsampled_lidar->size() + dense_lidar->size()) *
                        sizeof(PointType)) / (1024.0 * 1024.0);
                RCLCPP_INFO(
                    get_logger(),
                    "keyframe=%zu store≈%.1fMB coarse=%zu dense=%zu",
                    count + 1, mb, downsampled_lidar->size(),
                    dense_lidar->size());
            }
        }
    }

    bool MappingAlg::saveDescriptorDatabase(
        const KeyframeStore::Snapshot& snapshot,
        const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& optimized_poses,
        const std::string& output_dir)
    {
        if (!keyframe_enable_) return true;
        if (output_dir.empty())
        {
            RCLCPP_ERROR(get_logger(), "Failed to save descriptor DB: output directory is empty");
            return false;
        }
        if (snapshot.empty() || optimized_poses.size() != snapshot.size())
        {
            RCLCPP_ERROR(
                get_logger(),
                "Failed to save descriptor DB: frozen snapshot/pose count mismatch");
            return false;
        }
        scan_descriptor::DescriptorDatabase database(descriptor_config_);
        const Eigen::Matrix4d lidar_from_base =
            lidar_to_base_extrinsic_.inverse();
        for (size_t index = 0; index < snapshot.size(); ++index)
        {
            const Keyframe& keyframe = snapshot[index];
            const Eigen::Matrix4d& T_map_lidar = optimized_poses[index];
            const Eigen::Matrix4d T_map_base = T_map_lidar * lidar_from_base;
            scan_descriptor::KeyframeRecord record;
            record.id = keyframe.id;
            record.stamp = keyframe.stamp;
            // SCDB v1 stores T_map_lidar. Localization explicitly converts it
            // to T_map_base with this same installation extrinsic.
            record.T_map_lidar = T_map_lidar;
            record.desc = makeGlobalLocalizationDescriptor(
                *keyframe.cloud_lidar, T_map_base);
            if (!database.add(record))
            {
                RCLCPP_ERROR(get_logger(), "Failed to add keyframe %zu to descriptor DB", index);
                return false;
            }
        }
        std::string error;
        const std::string path =
            (std::filesystem::path(output_dir) / "map_scd.bin").string();
        if (!database.save(path, &error))
        {
            RCLCPP_ERROR(get_logger(), "Failed to save descriptor DB %s: %s", path.c_str(), error.c_str());
            return false;
        }
        RCLCPP_INFO(get_logger(),
            "Descriptor DB saved: %s (%zu keyframes, descriptor_frame=base_link, "
            "pose_frame=livox_frame, floor_z_map=%.3f)",
            path.c_str(), database.size(), keyframe_floor_z_map_);
        return true;
    }
}  // namespace anubis_mapping