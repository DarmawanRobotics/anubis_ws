#include "loop_closure.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>

#include "scan_descriptor/point_adapter.hpp"

namespace anubis_mapping {
namespace {

gtsam::Pose3 toPose3(const Eigen::Matrix4d& pose)
{
    const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
    return gtsam::Pose3(
        gtsam::Rot3(rotation),
        gtsam::Point3(pose(0, 3), pose(1, 3), pose(2, 3)));
}

Eigen::Matrix4d fromPose3(const gtsam::Pose3& pose)
{
    return pose.matrix();
}

bool isFiniteRigidPose(const Eigen::Matrix4d& pose)
{
    if (!pose.allFinite() ||
        !pose.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-8))
    {
        return false;
    }
    const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
    return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
               .cwiseAbs()
               .maxCoeff() <= 1e-5 &&
        std::abs(rotation.determinant() - 1.0) <= 1e-5;
}

bool hasValidNoiseConfiguration(const LoopClosureConfig& config)
{
    const double values[] = {
        config.odom_rotation_sigma, config.odom_translation_sigma,
        config.odom_roll_pitch_sigma, config.odom_z_sigma,
        config.loop_rotation_sigma, config.loop_translation_sigma,
        config.loop_roll_pitch_sigma, config.loop_z_sigma, config.huber_k};
    for (const double value : values)
    {
        if (!std::isfinite(value) || value <= 0.0)
        {
            return false;
        }
    }
    return true;
}

bool hasValidSanityConfiguration(const LoopClosureConfig& config)
{
    const double values[] = {
        config.max_optimized_xy_deviation,
        config.max_optimized_z_deviation,
        config.max_optimized_roll_pitch_deviation_deg,
        config.max_optimized_yaw_deviation_deg,
        config.sanity_path_reference_length_m,
        config.sanity_path_scale_max};
    for (const double value : values)
    {
        if (!std::isfinite(value) || value <= 0.0)
        {
            return false;
        }
    }
    return config.sanity_path_scale_max >= 1.0 &&
        config.max_optimized_roll_pitch_deviation_deg <= 180.0 &&
        config.max_optimized_yaw_deviation_deg <= 360.0;
}

gtsam::SharedNoiseModel diagonalNoise(double rotation_sigma,
                                      double translation_sigma)
{
    gtsam::Vector6 sigmas;
    sigmas << rotation_sigma, rotation_sigma, rotation_sigma,
              translation_sigma, translation_sigma, translation_sigma;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

// [2026-08-14 实测] 各向同性噪声是"地面被抹平"的根因：室内平面场景里
// ICP 回环约束对 z/roll/pitch 几乎不可观测（平行墙几何退化），错误回环
// 的 z/俯仰误差以与 xy 相同的权重进入优化，把 333 帧的位姿在 z/姿态上
// 洗牌 —— 实测优化后 z 直方图在 [-0.6, 1.3] 均匀分布（地面消失）、22 帧
// 高度超 1m、栅格图 82% 被标为墙。修正：各向异性噪声 ——
//   回环因子：z/roll/pitch 用大 sigma（优化器忽略这些不可信分量）
//   里程计因子：z/roll/pitch 用小 sigma（地面平面锚定在短程精确的里程计上）
gtsam::SharedNoiseModel planarNoise(double rp_sigma, double yaw_sigma,
                                    double xy_sigma, double z_sigma)
{
    gtsam::Vector6 sigmas;
    sigmas << rp_sigma, rp_sigma, yaw_sigma,
              xy_sigma, xy_sigma, z_sigma;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

Eigen::Matrix3d leveledRotation(const Eigen::Matrix4d& pose)
{
    const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
    const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    return Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * rotation;
}

}  // namespace

namespace detail {

bool projectNearRigidPose(
    const Eigen::Matrix4d& input,
    Eigen::Matrix4d* output,
    double max_distortion)
{
    if (output == nullptr || !std::isfinite(max_distortion) ||
        max_distortion <= 0.0 || !input.allFinite() ||
        !input.row(3).isApprox(
            Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), max_distortion))
    {
        return false;
    }

    const Eigen::Matrix3d rotation = input.block<3, 3>(0, 0);
    const double orthogonality_error =
        (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
            .cwiseAbs()
            .maxCoeff();
    const double determinant_error = std::abs(rotation.determinant() - 1.0);
    if (!std::isfinite(orthogonality_error) ||
        !std::isfinite(determinant_error) ||
        orthogonality_error > max_distortion ||
        determinant_error > max_distortion)
    {
        return false;
    }

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d projected = svd.matrixU() * svd.matrixV().transpose();
    if (!projected.allFinite() || projected.determinant() <= 0.0)
    {
        return false;
    }

    Eigen::Matrix4d result = input;
    result.block<3, 3>(0, 0) = projected;
    result.row(3) = Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0);
    if (!isFiniteRigidPose(result))
    {
        return false;
    }
    *output = result;
    return true;
}

double yawDeviationRadians(
    const Eigen::Matrix3d& reference,
    const Eigen::Matrix3d& candidate)
{
    const double reference_yaw = std::atan2(reference(1, 0), reference(0, 0));
    const double candidate_yaw = std::atan2(candidate(1, 0), candidate(0, 0));
    return std::abs(std::remainder(
        candidate_yaw - reference_yaw, 2.0 * M_PI));
}

double rollPitchDeviationRadians(
    const Eigen::Matrix3d& reference,
    const Eigen::Matrix3d& candidate)
{
    const auto roll_pitch = [](const Eigen::Matrix3d& rotation)
    {
        const double horizontal = std::hypot(rotation(0, 0), rotation(1, 0));
        return Eigen::Vector2d(
            std::atan2(rotation(2, 1), rotation(2, 2)),
            std::atan2(-rotation(2, 0), horizontal));
    };
    const auto angle_difference = [](double first, double second)
    {
        return std::abs(std::remainder(first - second, 2.0 * M_PI));
    };

    const Eigen::Vector2d reference_rp = roll_pitch(reference);
    const Eigen::Vector2d candidate_rp = roll_pitch(candidate);
    return std::max(
        angle_difference(candidate_rp.x(), reference_rp.x()),
        angle_difference(candidate_rp.y(), reference_rp.y()));
}

LoopInnovation measureLoopInnovation(
    const Eigen::Matrix4d& odometry_relative,
    const Eigen::Matrix4d& loop_relative)
{
    LoopInnovation output;
    if (!odometry_relative.allFinite() || !loop_relative.allFinite())
    {
        const double invalid = std::numeric_limits<double>::infinity();
        output.odom_xy_distance = invalid;
        output.correction_xy = invalid;
        output.correction_z = invalid;
        output.correction_yaw_rad = invalid;
        return output;
    }
    const Eigen::Vector3d odometry_translation =
        odometry_relative.block<3, 1>(0, 3);
    const Eigen::Vector3d loop_translation =
        loop_relative.block<3, 1>(0, 3);
    output.odom_xy_distance = odometry_translation.head<2>().norm();
    output.correction_xy =
        (loop_translation - odometry_translation).head<2>().norm();
    output.correction_z =
        std::abs(loop_translation.z() - odometry_translation.z());
    output.correction_yaw_rad = yawDeviationRadians(
        odometry_relative.block<3, 3>(0, 0),
        loop_relative.block<3, 3>(0, 0));
    return output;
}

const char* loopInnovationRejectionReason(
    const LoopInnovation& innovation,
    const LoopClosureConfig& config)
{
    if (!hasValidSanityConfiguration(config) ||
        !std::isfinite(config.max_odom_xy_distance) ||
        config.max_odom_xy_distance <= 0.0 ||
        !std::isfinite(config.max_constraint_xy_correction) ||
        config.max_constraint_xy_correction <= 0.0 ||
        !std::isfinite(config.max_constraint_z_correction) ||
        config.max_constraint_z_correction <= 0.0 ||
        !std::isfinite(config.max_constraint_yaw_correction_deg) ||
        config.max_constraint_yaw_correction_deg <= 0.0)
    {
        return "invalid_constraint_configuration";
    }
    if (!std::isfinite(innovation.odom_xy_distance) ||
        !std::isfinite(innovation.correction_xy) ||
        !std::isfinite(innovation.correction_z) ||
        !std::isfinite(innovation.correction_yaw_rad))
    {
        return "constraint_non_finite";
    }
    if (innovation.odom_xy_distance > config.max_odom_xy_distance)
    {
        return "odom_xy_distance";
    }
    if (innovation.correction_xy > config.max_constraint_xy_correction)
    {
        return "constraint_xy_correction";
    }
    if (innovation.correction_z > config.max_constraint_z_correction)
    {
        return "constraint_z_correction";
    }
    if (innovation.correction_yaw_rad * 180.0 / M_PI >
        config.max_constraint_yaw_correction_deg)
    {
        return "constraint_yaw_correction";
    }
    return nullptr;
}

}  // namespace detail

namespace detail {

PoseGraphSanityReport evaluatePoseGraphSanity(
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& odometry_poses,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& optimized_poses,
    const LoopClosureConfig& config)
{
    PoseGraphSanityReport report;
    if (odometry_poses.size() != optimized_poses.size())
    {
        report.accepted = false;
        report.rejection_reason = "pose_count_mismatch";
        report.offending_index = 0U;
        return report;
    }
    if (odometry_poses.empty())
    {
        return report;
    }

    if (!hasValidSanityConfiguration(config))
    {
        report.accepted = false;
        report.offending_index = 0U;
        report.rejection_reason = "invalid_sanity_configuration";
        return report;
    }

    const double reference_length =
        std::isfinite(config.sanity_path_reference_length_m) &&
                config.sanity_path_reference_length_m > 0.0
            ? config.sanity_path_reference_length_m
            : 1.0;
    const double scale_cap =
        std::isfinite(config.sanity_path_scale_max) &&
                config.sanity_path_scale_max >= 1.0
            ? config.sanity_path_scale_max
            : 1.0;
    double path_length = 0.0;
    for (std::size_t index = 0; index < odometry_poses.size(); ++index)
    {
        if (!isFiniteRigidPose(odometry_poses[index]) ||
            !isFiniteRigidPose(optimized_poses[index]))
        {
            report.accepted = false;
            report.offending_index = index;
            report.rejection_reason = "non_finite_pose";
            return report;
        }
        if (index > 0U)
        {
            // Only horizontal travel should relax the XY/yaw sanity limits.
            // A vertical FAST-LIO drift must remain covered by the dedicated
            // z/roll-pitch limits instead of inflating the path scale.
            path_length += (odometry_poses[index].block<2, 1>(0, 3) -
                            odometry_poses[index - 1U].block<2, 1>(0, 3))
                               .norm();
            if (!std::isfinite(path_length))
            {
                report.accepted = false;
                report.offending_index = index;
                report.rejection_reason = "path_length_non_finite";
                return report;
            }
        }
        const double path_scale = std::min(
            scale_cap, std::max(1.0, path_length / reference_length));
        const double xy_limit = config.max_optimized_xy_deviation * path_scale;
        const double yaw_limit = config.max_optimized_yaw_deviation_deg *
            path_scale * M_PI / 180.0;
        const Eigen::Matrix4d& odom = odometry_poses[index];
        const Eigen::Matrix4d& optimized = optimized_poses[index];
        const double dxy =
            (optimized.block<2, 1>(0, 3) - odom.block<2, 1>(0, 3)).norm();
        const double dz = std::abs(optimized(2, 3) - odom(2, 3));
        const double drp = detail::rollPitchDeviationRadians(
            odom.block<3, 3>(0, 0), optimized.block<3, 3>(0, 0));
        const double dyaw = detail::yawDeviationRadians(
            odom.block<3, 3>(0, 0), optimized.block<3, 3>(0, 0));
        report.path_length_m = path_length;
        report.max_xy_deviation = std::max(report.max_xy_deviation, dxy);
        report.max_z_deviation = std::max(report.max_z_deviation, dz);
        report.max_roll_pitch_deviation_rad = std::max(
            report.max_roll_pitch_deviation_rad, drp);
        report.max_yaw_deviation_rad = std::max(
            report.max_yaw_deviation_rad, dyaw);
        report.effective_xy_limit = xy_limit;
        report.effective_yaw_limit_rad = yaw_limit;

        const double max_drp = config.max_optimized_roll_pitch_deviation_deg *
            M_PI / 180.0;
        if (dxy > xy_limit)
        {
            report.accepted = false;
            report.offending_index = index;
            report.rejection_reason = "xy_deviation";
            break;
        }
        if (dz > config.max_optimized_z_deviation)
        {
            report.accepted = false;
            report.offending_index = index;
            report.rejection_reason = "z_deviation";
            break;
        }
        if (drp > max_drp)
        {
            report.accepted = false;
            report.offending_index = index;
            report.rejection_reason = "roll_pitch_deviation";
            break;
        }
        if (dyaw > yaw_limit)
        {
            report.accepted = false;
            report.offending_index = index;
            report.rejection_reason = "yaw_deviation";
            break;
        }
    }
    return report;
}

bool acceptsFinalPoseSafetyGate(
    const PoseGraphSanityReport& loop_report,
    const PoseGraphSanityReport& local_report,
    const PoseGraphSanityReport& final_report)
{
    // This is deliberately independent of map_leveling.enforce_quality_gates:
    // that option describes whether a single-floor geometry correction is
    // required, not whether an invalid/non-finite pose may enter a map.
    return loop_report.accepted && local_report.accepted &&
        final_report.accepted;
}

}  // namespace detail

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
optimizePoseGraph(
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& odometry_poses,
    const std::vector<LoopConstraint>& loops,
    const LoopClosureConfig& config,
    bool* sanity_accepted,
    std::string* sanity_reason)
{
    using gtsam::symbol_shorthand::X;
    if (sanity_accepted != nullptr) *sanity_accepted = true;
    if (sanity_reason != nullptr) sanity_reason->clear();
    const auto mark_failure = [&](const std::string& reason)
    {
        if (sanity_accepted != nullptr) *sanity_accepted = false;
        if (sanity_reason != nullptr) *sanity_reason = reason;
    };
    if (odometry_poses.empty())
    {
        mark_failure("empty_trajectory");
        return {};
    }
    // Validate every transform before handing it to GTSAM.  Rot3 accepts
    // finite but non-orthonormal matrices in some builds and can then fail
    // deep inside optimization, where falling back to the original trajectory
    // is no longer deterministic.
    for (std::size_t index = 0; index < odometry_poses.size(); ++index)
    {
        if (!isFiniteRigidPose(odometry_poses[index]))
        {
            mark_failure("invalid_odometry_pose");
            std::cerr << "[loop_closure] MANUAL CONFIRMATION REQUIRED: invalid "
                      << "odometry pose at index " << index << std::endl;
            return odometry_poses;
        }
    }
    if (!hasValidNoiseConfiguration(config))
    {
        mark_failure("invalid_noise_configuration");
        std::cerr << "[loop_closure] MANUAL CONFIRMATION REQUIRED: invalid "
                  << "pose-graph noise configuration" << std::endl;
        return odometry_poses;
    }
    for (const LoopConstraint& loop : loops)
    {
        if (loop.first >= odometry_poses.size() ||
            loop.second >= odometry_poses.size() ||
            loop.first == loop.second ||
            !isFiniteRigidPose(loop.T_first_second))
        {
            mark_failure("invalid_loop_constraint");
            std::cerr << "[loop_closure] MANUAL CONFIRMATION REQUIRED: invalid "
                      << "loop constraint (first=" << loop.first
                      << ", second=" << loop.second << ")" << std::endl;
            return odometry_poses;
        }
    }
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;
    // Keep the locally observable planar motion tighter than the weak z and
    // roll/pitch dimensions.  All four values are configurable so a replay
    // can account for the measured FAST-LIO vertical drift.
    const auto odom_noise = planarNoise(
        config.odom_roll_pitch_sigma, config.odom_rotation_sigma,
        config.odom_translation_sigma, config.odom_z_sigma);
    const auto loop_base_noise = planarNoise(
        config.loop_roll_pitch_sigma, config.loop_rotation_sigma,
        config.loop_translation_sigma, config.loop_z_sigma);
    const auto loop_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(config.huber_k),
        loop_base_noise);
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(
        X(0), toPose3(odometry_poses.front()),
        diagonalNoise(1e-4, 1e-4)));
    for (size_t index = 0; index < odometry_poses.size(); ++index)
    {
        initial.insert(X(index), toPose3(odometry_poses[index]));
        if (index > 0)
        {
            const Eigen::Matrix4d relative =
                odometry_poses[index - 1].inverse() * odometry_poses[index];
            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                X(index - 1), X(index), toPose3(relative), odom_noise));
        }
    }
    for (const LoopConstraint& loop : loops)
    {
        if (loop.first >= odometry_poses.size() ||
            loop.second >= odometry_poses.size() ||
            loop.first == loop.second || !loop.T_first_second.allFinite())
        {
            continue;
        }
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            X(loop.first), X(loop.second), toPose3(loop.T_first_second),
            loop_noise));
    }
    gtsam::Values result;
    try
    {
        gtsam::LevenbergMarquardtParams parameters;
        result = gtsam::LevenbergMarquardtOptimizer(
            graph, initial, parameters).optimize();
    }
    catch (const std::exception& exception)
    {
        mark_failure("optimizer_exception");
        std::cerr << "[loop_closure] MANUAL CONFIRMATION REQUIRED: optimizer "
                  << "failed: " << exception.what() << std::endl;
        return odometry_poses;
    }
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> output;
    output.reserve(odometry_poses.size());
    try
    {
        for (size_t index = 0; index < odometry_poses.size(); ++index)
        {
            const Eigen::Matrix4d pose =
                fromPose3(result.at<gtsam::Pose3>(X(index)));
            if (!isFiniteRigidPose(pose))
            {
                throw std::runtime_error(
                    "optimizer returned a non-finite or non-rigid pose");
            }
            output.push_back(pose);
        }
    }
    catch (const std::exception& exception)
    {
        mark_failure("result_extraction_failed");
        std::cerr << "[loop_closure] MANUAL CONFIRMATION REQUIRED: optimizer "
                  << "result extraction failed: " << exception.what()
                  << std::endl;
        return odometry_poses;
    }
    const detail::PoseGraphSanityReport report = detail::evaluatePoseGraphSanity(
        odometry_poses, output, config);
    if (!report.accepted)
    {
        mark_failure(report.rejection_reason.empty()
                ? std::string("pose_graph_sanity")
                : report.rejection_reason);
        std::cerr << "[loop_closure] MANUAL CONFIRMATION REQUIRED: optimized "
                  << "pose graph rejected at index " << report.offending_index
                  << " reason=" << report.rejection_reason
                  << " path_length=" << report.path_length_m
                  << "m max=[xy=" << report.max_xy_deviation
                  << "m z=" << report.max_z_deviation
                  << "m rp=" << report.max_roll_pitch_deviation_rad
                  << "rad yaw=" << report.max_yaw_deviation_rad
                  << "rad] effective_limits=[xy="
                  << report.effective_xy_limit << "m yaw="
                  << report.effective_yaw_limit_rad << "rad]; "
                  << "falling back to odometry poses" << std::endl;
        return odometry_poses;
    }
    if (sanity_reason != nullptr) *sanity_reason = "accepted";
    return output;
}

LoopClosure::LoopClosure(
    std::shared_ptr<KeyframeStore> store,
    const scan_descriptor::DescriptorConfig& descriptor_config,
    double floor_z,
    const LoopClosureConfig& config,
    LogFunction info_log,
    LogFunction warning_log)
    : store_(std::move(store)),
      descriptor_config_(descriptor_config),
      floor_z_(floor_z),
      config_(config),
      info_log_(std::move(info_log)),
      warning_log_(std::move(warning_log)),
      database_(descriptor_config_)
{
    worker_ = std::thread(&LoopClosure::workerMain, this);
}

LoopClosure::~LoopClosure()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_condition_.notify_all();
    if (worker_.joinable())
    {
        worker_.join();
    }
}

void LoopClosure::notifyNewKeyframe()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        requested_count_ = std::max(requested_count_, store_->size());
    }
    wake_condition_.notify_one();
}

bool LoopClosure::waitUntilProcessed(size_t keyframe_count, double timeout_seconds)
{
    notifyNewKeyframe();
    std::unique_lock<std::mutex> lock(mutex_);
    return processed_condition_.wait_for(
        lock, std::chrono::duration<double>(timeout_seconds),
        [&]() { return processed_count_ >= keyframe_count || stop_; });
}

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
LoopClosure::optimizedPoses() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return optimized_poses_;
}

size_t LoopClosure::acceptedLoopCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return loop_constraints_.size();
}

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
LoopClosure::recomputeOptimizedPoses(
    const KeyframeStore::Snapshot& snapshot,
    bool* sanity_accepted,
    std::string* sanity_reason) const
{
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> odometry;
    odometry.reserve(snapshot.size());
    for (const auto& keyframe : snapshot) odometry.push_back(keyframe.T_map_lidar);
    std::vector<LoopConstraint> loops;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // The keyframe store snapshot is an immutable save generation, while
        // the worker may have accepted a constraint for a newer keyframe
        // between snapshot() and this call.  Never hand an out-of-generation
        // edge to optimizePoseGraph: its index would either fail the whole
        // recomputation or mix a later trajectory into the frozen artifacts.
        loops.reserve(loop_constraints_.size());
        for (const auto& loop : loop_constraints_)
        {
            if (loop.first < snapshot.size() &&
                loop.second < snapshot.size())
            {
                loops.push_back(loop);
            }
        }
    }
    return optimizePoseGraph(
        odometry, loops, config_, sanity_accepted, sanity_reason);
}

void LoopClosure::expireCandidateConsistency(size_t current_index)
{
    if (pending_candidate_hits_ > 0 && current_index > pending_current_index_ &&
        current_index - pending_current_index_ >
            static_cast<size_t>(config_.consistency_max_keyframe_gap))
    {
        pending_candidate_hits_ = 0;
    }
}

bool LoopClosure::candidateConsistencyReady(
    size_t current_index,
    size_t candidate_index,
    const KeyframeStore::Snapshot& snapshot,
    double* candidate_shift,
    int* hit_count)
{
    double shift = std::numeric_limits<double>::infinity();
    bool continues_track = pending_candidate_hits_ > 0 &&
        current_index > pending_current_index_ &&
        current_index - pending_current_index_ <=
            static_cast<size_t>(config_.consistency_max_keyframe_gap) &&
        pending_candidate_index_ < snapshot.size() &&
        candidate_index < snapshot.size();
    if (continues_track)
    {
        shift = (snapshot[pending_candidate_index_]
                     .T_map_lidar.block<2, 1>(0, 3) -
                 snapshot[candidate_index].T_map_lidar.block<2, 1>(0, 3))
                    .norm();
        continues_track =
            shift <= config_.consistency_candidate_xy_tolerance;
    }
    pending_candidate_hits_ = continues_track
        ? pending_candidate_hits_ + 1
        : 1;
    pending_current_index_ = current_index;
    pending_candidate_index_ = candidate_index;
    if (candidate_shift != nullptr)
    {
        *candidate_shift = shift;
    }
    if (hit_count != nullptr)
    {
        *hit_count = pending_candidate_hits_;
    }
    return pending_candidate_hits_ >= config_.consistency_min_hits;
}

void LoopClosure::workerMain()
{
    while (true)
    {
        size_t target = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_condition_.wait(lock, [&]() {
                return stop_ || requested_count_ > processed_count_;
            });
            if (stop_)
            {
                return;
            }
            target = requested_count_;
        }
        while (true)
        {
            size_t index = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (processed_count_ >= target || stop_)
                {
                    break;
                }
                index = processed_count_;
            }
            const KeyframeStore::Snapshot snapshot = store_->snapshot();
            if (index >= snapshot.size())
            {
                break;
            }
            processKeyframe(index, snapshot);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                processed_count_ = index + 1;
            }
            processed_condition_.notify_all();
        }
    }
}

void LoopClosure::processKeyframe(
    size_t index, const KeyframeStore::Snapshot& snapshot)
{
    expireCandidateConsistency(index);
    const Keyframe& current = snapshot[index];
    const scan_descriptor::Descriptor descriptor = current.desc_ready
        ? current.desc
        : scan_descriptor::makeDescriptor(
            scan_descriptor::adaptPoints(
                *current.cloud_lidar, leveledRotation(current.T_map_lidar),
                floor_z_ - current.T_map_lidar(2, 3), descriptor_config_),
            descriptor_config_);
    if (!current.desc_ready) store_->setDescriptor(index, descriptor);

    std::vector<scan_descriptor::DatabaseMatch> matches;
    if (config_.enable && current.id > static_cast<uint64_t>(config_.min_keyframe_gap))
    {
        matches = database_.queryTopK(
            descriptor,
            static_cast<std::size_t>(config_.descriptor_candidate_count),
            config_.descriptor_max_distance,
            current.id - static_cast<uint64_t>(config_.min_keyframe_gap), 0.0);
    }
    if (!matches.empty())
    {
        std::size_t selected_match_index = matches.size();
        std::size_t selected_candidate_index = snapshot.size();
        Eigen::Matrix4d selected_odom_relative = Eigen::Matrix4d::Identity();
        detail::LoopInnovation selected_odom_metrics;
        for (std::size_t rank = 0; rank < matches.size(); ++rank)
        {
            const auto& match = matches[rank];
            const uint64_t candidate_id = database_.at(match.record_index).id;
            std::size_t candidate_index = snapshot.size();
            for (std::size_t scan_index = 0; scan_index < snapshot.size(); ++scan_index)
            {
                if (snapshot[scan_index].id == candidate_id)
                {
                    candidate_index = scan_index;
                    break;
                }
            }
            if (candidate_index >= index)
            {
                continue;
            }
            const Eigen::Matrix4d odom_relative =
                snapshot[candidate_index].T_map_lidar.inverse() *
                current.T_map_lidar;
            const detail::LoopInnovation odom_metrics =
                detail::measureLoopInnovation(odom_relative, odom_relative);
            const char* odom_reason =
                detail::loopInnovationRejectionReason(odom_metrics, config_);
            if (odom_reason == nullptr)
            {
                selected_match_index = rank;
                selected_candidate_index = candidate_index;
                selected_odom_relative = odom_relative;
                selected_odom_metrics = odom_metrics;
                break;
            }
            if (warning_log_)
            {
                std::ostringstream message;
                message << "LOOP CANDIDATE SKIPPED i=" << current.id
                        << " c=" << candidate_id
                        << " rank=" << (rank + 1U) << "/" << matches.size()
                        << " desc=" << match.distance
                        << " odom_xy=" << odom_metrics.odom_xy_distance
                        << " reason=" << odom_reason;
                warning_log_(message.str());
            }
        }
        if (selected_match_index >= matches.size())
        {
            pending_candidate_hits_ = 0;
        }
        const auto process_match = [&]()
        {
            if (selected_match_index >= matches.size())
            {
                return;
            }
            const auto& match = matches[selected_match_index];
            const uint64_t candidate_id = database_.at(match.record_index).id;
            const size_t candidate_index = selected_candidate_index;
            const Eigen::Matrix4d& odom_relative = selected_odom_relative;
            const detail::LoopInnovation& odom_metrics = selected_odom_metrics;
            double candidate_shift = 0.0;
            int consistency_hits = 0;
            if (!candidateConsistencyReady(
                    index, candidate_index, snapshot,
                    &candidate_shift, &consistency_hits))
            {
                if (info_log_)
                {
                    std::ostringstream message;
                    message << "LOOP PENDING i=" << current.id
                            << " c=" << candidate_id
                            << " desc=" << match.distance
                            << " odom_xy=" << odom_metrics.odom_xy_distance
                            << " consistency=" << consistency_hits << "/"
                            << config_.consistency_min_hits
                            << " candidate_shift=" << candidate_shift;
                    info_log_(message.str());
                }
                return;
            }
            PointCloudType::Ptr submap(new PointCloudType());
            const size_t begin = candidate_index > static_cast<size_t>(config_.submap_half_window)
                ? candidate_index - static_cast<size_t>(config_.submap_half_window) : 0;
            const size_t end = std::min(
                index, candidate_index + static_cast<size_t>(config_.submap_half_window) + 1);
            for (size_t neighbor = begin; neighbor < end; ++neighbor)
            {
                PointCloudType transformed;
                const Eigen::Matrix4d T_candidate_neighbor =
                    snapshot[candidate_index].T_map_lidar.inverse() *
                    snapshot[neighbor].T_map_lidar;
                pcl::transformPointCloud(
                    *snapshot[neighbor].cloud_lidar, transformed,
                    T_candidate_neighbor.cast<float>());
                *submap += transformed;
            }
            // Descriptor matching is defined as
            // T_map_query ~= T_map_candidate * Rz(yaw_rad).  The matched yaw
            // is therefore the absolute candidate->query yaw, not an extra
            // delta to multiply onto the odometry yaw.  Keep odometry's
            // translation and roll/pitch, but replace its yaw with the
            // descriptor result before using it as the ICP source->submap
            // initial transform.
            Eigen::Matrix4d initial_double = odom_relative;
            const double odometry_yaw = std::atan2(
                initial_double(1, 0), initial_double(0, 0));
            const Eigen::Matrix3d roll_pitch =
                Eigen::AngleAxisd(-odometry_yaw, Eigen::Vector3d::UnitZ())
                    .toRotationMatrix() * initial_double.block<3, 3>(0, 0);
            initial_double.block<3, 3>(0, 0) =
                Eigen::AngleAxisd(match.yaw_rad, Eigen::Vector3d::UnitZ())
                    .toRotationMatrix() * roll_pitch;
            const Eigen::Matrix4f initial = initial_double.cast<float>();
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setInputSource(current.cloud_lidar);
            icp.setInputTarget(submap);
            icp.setMaximumIterations(config_.icp_max_iterations);
            icp.setMaxCorrespondenceDistance(
                config_.icp_max_correspondence_distance);
            PointCloudType aligned;
            icp.align(aligned, initial);
            const double fitness = icp.getFitnessScore();
            if (!icp.hasConverged() || !std::isfinite(fitness) ||
                fitness >= config_.icp_fitness_threshold)
            {
                if (warning_log_)
                {
                    std::ostringstream message;
                    message << "LOOP REJECTED i=" << current.id
                            << " c=" << candidate_id
                            << " desc=" << match.distance
                            << " fitness=" << fitness
                            << " odom_xy=" << odom_metrics.odom_xy_distance
                            << " consistency=" << consistency_hits
                            << " reason="
                            << (icp.hasConverged()
                                    ? "icp_fitness" : "icp_not_converged");
                    warning_log_(message.str());
                }
            }
            else
            {
                LoopConstraint constraint;
                constraint.first = candidate_index;
                constraint.second = index;
                const Eigen::Matrix4d raw_icp_transform =
                    icp.getFinalTransformation().cast<double>();
                if (!detail::projectNearRigidPose(
                        raw_icp_transform, &constraint.T_first_second))
                {
                    if (warning_log_)
                    {
                        std::ostringstream message;
                        message << "LOOP REJECTED i=" << current.id
                                << " c=" << candidate_id
                                << " desc=" << match.distance
                                << " fitness=" << fitness
                                << " odom_xy=" << odom_metrics.odom_xy_distance
                                << " consistency=" << consistency_hits
                                << " reason=icp_non_rigid";
                        warning_log_(message.str());
                    }
                    return;
                }
                const detail::LoopInnovation innovation =
                    detail::measureLoopInnovation(
                        odom_relative, constraint.T_first_second);
                const double correction_yaw_deg =
                    innovation.correction_yaw_rad * 180.0 / M_PI;
                const char* innovation_reason =
                    detail::loopInnovationRejectionReason(innovation, config_);
                if (innovation_reason != nullptr)
                {
                    if (warning_log_)
                    {
                        std::ostringstream message;
                        message << "LOOP REJECTED i=" << current.id
                                << " c=" << candidate_id
                                << " desc=" << match.distance
                                << " fitness=" << fitness
                                << " odom_xy=" << innovation.odom_xy_distance
                                << " correction_xy=" << innovation.correction_xy
                                << " correction_z=" << innovation.correction_z
                                << " correction_yaw_deg=" << correction_yaw_deg
                                << " consistency=" << consistency_hits
                                << " reason=" << innovation_reason;
                        warning_log_(message.str());
                    }
                    return;
                }
                std::vector<LoopConstraint> loops;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    loops = loop_constraints_;
                }
                loops.push_back(constraint);
                std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> odometry;
                odometry.reserve(index + 1);
                for (size_t pose_index = 0; pose_index <= index; ++pose_index)
                {
                    odometry.push_back(snapshot[pose_index].T_map_lidar);
                }
                bool sanity_accepted = false;
                std::string sanity_reason;
                const auto optimized = optimizePoseGraph(
                    odometry, loops, config_, &sanity_accepted,
                    &sanity_reason);
                if (sanity_accepted)
                {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        loop_constraints_.push_back(constraint);
                        optimized_poses_ = optimized;
                    }
                    if (info_log_)
                    {
                        std::ostringstream message;
                        message << "LOOP ACCEPTED i=" << current.id
                                << " c=" << candidate_id
                                << " desc=" << match.distance
                                << " fitness=" << fitness
                                << " odom_xy=" << innovation.odom_xy_distance
                                << " correction_xy=" << innovation.correction_xy
                                << " correction_z=" << innovation.correction_z
                                << " correction_yaw_deg=" << correction_yaw_deg
                                << " consistency=" << consistency_hits;
                        info_log_(message.str());
                    }
                }
                else if (warning_log_)
                {
                    std::ostringstream message;
                    message << "LOOP REJECTED i=" << current.id
                            << " c=" << candidate_id
                            << " desc=" << match.distance
                            << " fitness=" << fitness
                            << " odom_xy=" << innovation.odom_xy_distance
                            << " correction_xy=" << innovation.correction_xy
                            << " correction_z=" << innovation.correction_z
                            << " correction_yaw_deg=" << correction_yaw_deg
                                << " consistency=" << consistency_hits
                                << " reason=" << (sanity_reason.empty()
                                    ? "pose_graph_sanity" : sanity_reason);
                    warning_log_(message.str());
                }
            }
        };
        process_match();
    }

    scan_descriptor::KeyframeRecord record;
    record.id = current.id;
    record.stamp = current.stamp;
    record.T_map_lidar = current.T_map_lidar;
    record.desc = descriptor;
    if (!database_.add(record) && warning_log_)
    {
        warning_log_("Failed to add keyframe descriptor to loop database");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (optimized_poses_.size() < index + 1)
        {
            optimized_poses_.reserve(index + 1);
            while (optimized_poses_.size() < index + 1)
            {
                optimized_poses_.push_back(
                    snapshot[optimized_poses_.size()].T_map_lidar);
            }
        }
    }
}

}  // namespace anubis_mapping
