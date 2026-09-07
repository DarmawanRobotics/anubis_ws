#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include <Eigen/Core>

#include "common/types.h"
#include "scan_descriptor/descriptor.hpp"

namespace anubis_mapping {

struct Keyframe {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    uint64_t id = 0;
    double stamp = 0.0;
    Eigen::Matrix4d T_map_imu = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_map_lidar = Eigen::Matrix4d::Identity();
    // Coarse cloud used by descriptors and loop-closure ICP.
    PointCloudType::ConstPtr cloud_lidar;
    // Dense cloud retained for map export, ground fitting and PGM evidence.
    // Keeping the two products separate avoids feeding a 5 cm grid from the
    // 10 cm descriptor lattice.
    PointCloudType::ConstPtr dense_cloud_lidar;
    scan_descriptor::Descriptor desc;
    bool desc_ready = false;
};

class KeyframeStore {
public:
    using Snapshot = std::vector<Keyframe, Eigen::aligned_allocator<Keyframe>>;

    bool append(Keyframe keyframe, size_t max_count);
    bool setDescriptor(size_t index, const scan_descriptor::Descriptor& descriptor);
    void clear();
    size_t size() const;
    Snapshot snapshot() const;

private:
    mutable std::mutex mutex_;
    Snapshot keyframes_;
};

bool shouldCreateKeyframe(const Eigen::Matrix4d& last_pose,
                          const Eigen::Matrix4d& current_pose,
                          double translation_threshold,
                          double yaw_threshold_rad);

}  // namespace anubis_mapping
