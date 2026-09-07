#include "keyframe_store.h"

#include <cmath>
#include <utility>

namespace anubis_mapping {

bool KeyframeStore::append(Keyframe keyframe, size_t max_count)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (keyframes_.size() >= max_count || !keyframe.cloud_lidar ||
        keyframe.cloud_lidar->empty() || !keyframe.dense_cloud_lidar ||
        keyframe.dense_cloud_lidar->empty() || !keyframe.T_map_imu.allFinite() ||
        !keyframe.T_map_lidar.allFinite())
    {
        return false;
    }
    keyframes_.push_back(std::move(keyframe));
    return true;
}

bool KeyframeStore::setDescriptor(
    size_t index, const scan_descriptor::Descriptor& descriptor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= keyframes_.size())
    {
        return false;
    }
    keyframes_[index].desc = descriptor;
    keyframes_[index].desc_ready = true;
    return true;
}

void KeyframeStore::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_.clear();
}

size_t KeyframeStore::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_.size();
}

KeyframeStore::Snapshot KeyframeStore::snapshot() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_;
}

bool shouldCreateKeyframe(const Eigen::Matrix4d& last_pose,
                          const Eigen::Matrix4d& current_pose,
                          double translation_threshold,
                          double yaw_threshold_rad)
{
    if (!last_pose.allFinite() || !current_pose.allFinite() ||
        !std::isfinite(translation_threshold) || translation_threshold <= 0.0 ||
        !std::isfinite(yaw_threshold_rad) || yaw_threshold_rad <= 0.0)
    {
        return false;
    }
    const double translation =
        (last_pose.block<3, 1>(0, 3) - current_pose.block<3, 1>(0, 3)).norm();
    const Eigen::Matrix3d relative_rotation =
        last_pose.block<3, 3>(0, 0).transpose() * current_pose.block<3, 3>(0, 0);
    const double yaw = std::abs(std::atan2(relative_rotation(1, 0),
                                           relative_rotation(0, 0)));
    return translation >= translation_threshold || yaw >= yaw_threshold_rad;
}

}  // namespace anubis_mapping
