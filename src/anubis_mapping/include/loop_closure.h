#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>

#include "keyframe_store.h"
#include "scan_descriptor/database.hpp"

namespace anubis_mapping {

namespace detail {

double yawDeviationRadians(
    const Eigen::Matrix3d& reference,
    const Eigen::Matrix3d& candidate);

double rollPitchDeviationRadians(
    const Eigen::Matrix3d& reference,
    const Eigen::Matrix3d& candidate);

struct LoopInnovation {
    double odom_xy_distance = 0.0;
    double correction_xy = 0.0;
    double correction_z = 0.0;
    double correction_yaw_rad = 0.0;
};

LoopInnovation measureLoopInnovation(
    const Eigen::Matrix4d& odometry_relative,
    const Eigen::Matrix4d& loop_relative);

bool projectNearRigidPose(
    const Eigen::Matrix4d& input,
    Eigen::Matrix4d* output,
    double max_distortion = 1e-3);

}  // namespace detail

struct LoopClosureConfig {
    bool enable = false;
    int min_keyframe_gap = 60;
    int submap_half_window = 12;
    int descriptor_candidate_count = 5;
    double descriptor_max_distance = 0.30;
    double icp_fitness_threshold = 0.15;
    double icp_max_correspondence_distance = 2.0;
    int icp_max_iterations = 80;
    double max_odom_xy_distance = 3.5;
    int consistency_min_hits = 3;
    int consistency_max_keyframe_gap = 3;
    double consistency_candidate_xy_tolerance = 1.0;
    double max_constraint_xy_correction = 0.50;
    double max_constraint_z_correction = 0.45;
    double max_constraint_yaw_correction_deg = 5.0;
    double max_optimized_xy_deviation = 0.50;
    double max_optimized_z_deviation = 0.45;
    double max_optimized_roll_pitch_deviation_deg = 7.0;
    double max_optimized_yaw_deviation_deg = 5.0;
    // The FAST-LIO trajectory is locally reliable in z/roll/pitch but can
    // accumulate xy/yaw drift over a long route.  The sanity gate therefore
    // scales these two limits with the original trajectory length.
    double sanity_path_reference_length_m = 10.0;
    double sanity_path_scale_max = 4.0;
    double odom_rotation_sigma = 0.02;
    double odom_translation_sigma = 0.05;
    double odom_roll_pitch_sigma = 0.01;
    double odom_z_sigma = 0.05;
    double loop_rotation_sigma = 0.05;
    double loop_translation_sigma = 0.10;
    double loop_roll_pitch_sigma = 0.05;
    double loop_z_sigma = 0.10;
    double huber_k = 1.345;
};

namespace detail {

const char* loopInnovationRejectionReason(
    const LoopInnovation& innovation,
    const LoopClosureConfig& config);

struct PoseGraphSanityReport {
    bool accepted = true;
    std::size_t offending_index = std::numeric_limits<std::size_t>::max();
    double path_length_m = 0.0;
    double max_xy_deviation = 0.0;
    double max_z_deviation = 0.0;
    double max_roll_pitch_deviation_rad = 0.0;
    double max_yaw_deviation_rad = 0.0;
    double effective_xy_limit = 0.0;
    double effective_yaw_limit_rad = 0.0;
    std::string rejection_reason;
};

PoseGraphSanityReport evaluatePoseGraphSanity(
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& odometry_poses,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& optimized_poses,
    const LoopClosureConfig& config);

// A save generation may opt out of single-floor leveling/geometry quality
// gates (for example, while mapping across floors), but it may never opt out
// of the final pose-safety contract.  Keep this decision in a pure helper so
// the save path and its regression test cannot accidentally re-bind it to a
// geometry policy switch.
bool acceptsFinalPoseSafetyGate(
    const PoseGraphSanityReport& loop_report,
    const PoseGraphSanityReport& local_report,
    const PoseGraphSanityReport& final_report);

}  // namespace detail

struct LoopConstraint {
    size_t first = 0;
    size_t second = 0;
    Eigen::Matrix4d T_first_second = Eigen::Matrix4d::Identity();
};

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
optimizePoseGraph(
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& odometry_poses,
    const std::vector<LoopConstraint>& loops,
    const LoopClosureConfig& config,
    bool* sanity_accepted = nullptr,
    std::string* sanity_reason = nullptr);

class LoopClosure {
public:
    using LogFunction = std::function<void(const std::string&)>;

    LoopClosure(std::shared_ptr<KeyframeStore> store,
                const scan_descriptor::DescriptorConfig& descriptor_config,
                double floor_z,
                const LoopClosureConfig& config,
                LogFunction info_log,
                LogFunction warning_log);
    ~LoopClosure();

    void notifyNewKeyframe();
    bool waitUntilProcessed(size_t keyframe_count, double timeout_seconds);
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
    optimizedPoses() const;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
    recomputeOptimizedPoses(
        const KeyframeStore::Snapshot& snapshot,
        bool* sanity_accepted = nullptr,
        std::string* sanity_reason = nullptr) const;
    size_t acceptedLoopCount() const;

private:
    void workerMain();
    void processKeyframe(size_t index, const KeyframeStore::Snapshot& snapshot);
    bool candidateConsistencyReady(
        size_t current_index,
        size_t candidate_index,
        const KeyframeStore::Snapshot& snapshot,
        double* candidate_shift,
        int* hit_count);
    void expireCandidateConsistency(size_t current_index);

    std::shared_ptr<KeyframeStore> store_;
    scan_descriptor::DescriptorConfig descriptor_config_;
    double floor_z_ = 0.0;
    LoopClosureConfig config_;
    LogFunction info_log_;
    LogFunction warning_log_;

    mutable std::mutex mutex_;
    std::condition_variable wake_condition_;
    mutable std::condition_variable processed_condition_;
    bool stop_ = false;
    size_t requested_count_ = 0;
    size_t processed_count_ = 0;
    std::thread worker_;
    scan_descriptor::DescriptorDatabase database_;
    std::vector<LoopConstraint> loop_constraints_;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> optimized_poses_;
    size_t pending_current_index_ = 0;
    size_t pending_candidate_index_ = 0;
    int pending_candidate_hits_ = 0;
};

}  // namespace anubis_mapping
