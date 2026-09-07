#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Ported from darmawan_ws's src/slam/src/include/lio_time_guard.h
// (namespace anubis_mapping -> anubis_mapping). Logic unchanged -- this
// module's own gtest suite (test/lio_time_guard_test.cpp) is the
// verification that the port preserved behavior exactly, not a rewrite.
namespace anubis_mapping
{

// Sensor-timeline continuity policy used before a FAST-LIO prediction.
// Message/IMU timestamps must be finite and ordered; adjacent LiDAR
// point-time windows may have only the explicitly bounded overlap below.
// Oversized gaps or overlaps are rejected instead of being silently
// integrated.
struct LioTimeGuardConfig
{
    bool enabled = true;
    double max_scan_gap = 0.25;
    double max_imu_dt = 0.10;
    // Mid-360 point-time windows can overlap by a small amount even when
    // message header timestamps are strictly increasing. Keep this
    // tolerance bounded; larger negative gaps remain a hard failure.
    double max_scan_overlap = 0.002;
};

struct LioTimeGuardDecision
{
    bool accepted = true;
    double scan_gap = std::numeric_limits<double>::quiet_NaN();
    // Gap between the last IMU sample committed with the previous frame
    // and the first sample selected for this frame. This is distinct from
    // the largest interval within the current batch because syncData()
    // removes samples as it advances the queue.
    double imu_cross_gap = std::numeric_limits<double>::quiet_NaN();
    double imu_dt_max = 0.0;
    const char* rejection_reason = nullptr;
};

inline bool valid_lio_time_guard_config(const LioTimeGuardConfig& config)
{
    return std::isfinite(config.max_scan_gap) &&
        config.max_scan_gap > 0.0 &&
        std::isfinite(config.max_imu_dt) && config.max_imu_dt > 0.0 &&
        std::isfinite(config.max_scan_overlap) &&
        config.max_scan_overlap >= 0.0;
}

// Cross-frame IMU samples may be prepended only when the previous sample
// belongs to the same sensor epoch. This tiny predicate is kept pure so
// the reset contract can be regression-tested without ROS or an EKF
// fixture.
inline bool can_reuse_imu_time_anchor(
    bool anchor_valid, bool anchor_pointer_present)
{
    return anchor_valid && anchor_pointer_present;
}

inline LioTimeGuardDecision evaluate_lio_time_guard(
    double lidar_begin, double lidar_end, double previous_lidar_end,
    const std::vector<double>& imu_timestamps,
    const LioTimeGuardConfig& config,
    // -1 is the explicit "no previous accepted IMU sample" sentinel.
    double previous_imu_end = -1.0)
{
    LioTimeGuardDecision decision;
    if (!config.enabled)
    {
        return decision;
    }
    if (!valid_lio_time_guard_config(config))
    {
        decision.accepted = false;
        decision.rejection_reason = "invalid_configuration";
        return decision;
    }
    if (!std::isfinite(lidar_begin) || !std::isfinite(lidar_end))
    {
        decision.accepted = false;
        decision.rejection_reason = "non_finite_lidar_timestamp";
        return decision;
    }
    if (lidar_end < lidar_begin)
    {
        decision.accepted = false;
        decision.rejection_reason = "lidar_timestamp_reversed";
        return decision;
    }
    if (lidar_end == lidar_begin)
    {
        decision.accepted = false;
        decision.rejection_reason = "zero_lidar_duration";
        return decision;
    }
    // -1 is the explicit "no previous accepted frame" sentinel.
    if (previous_lidar_end != -1.0 &&
        !std::isfinite(previous_lidar_end))
    {
        decision.accepted = false;
        decision.rejection_reason = "non_finite_previous_lidar_timestamp";
        return decision;
    }
    if (previous_lidar_end < 0.0 && previous_lidar_end != -1.0)
    {
        decision.accepted = false;
        decision.rejection_reason = "invalid_previous_lidar_timestamp";
        return decision;
    }
    if (previous_imu_end != -1.0 &&
        !std::isfinite(previous_imu_end))
    {
        decision.accepted = false;
        decision.rejection_reason = "non_finite_previous_imu_timestamp";
        return decision;
    }
    if (previous_imu_end < 0.0 && previous_imu_end != -1.0)
    {
        decision.accepted = false;
        decision.rejection_reason = "invalid_previous_imu_timestamp";
        return decision;
    }
    if (imu_timestamps.empty())
    {
        decision.accepted = false;
        decision.rejection_reason = "missing_imu_samples";
        return decision;
    }
    for (std::size_t index = 0; index < imu_timestamps.size(); ++index)
    {
        const double timestamp = imu_timestamps[index];
        if (!std::isfinite(timestamp))
        {
            decision.accepted = false;
            decision.rejection_reason = "non_finite_imu_timestamp";
            return decision;
        }
        if (index == 0U)
        {
            continue;
        }
        const double dt = timestamp - imu_timestamps[index - 1U];
        if (!std::isfinite(dt) || dt <= 0.0)
        {
            decision.accepted = false;
            decision.rejection_reason =
                dt == 0.0 ? "zero_imu_dt" : "imu_timestamp_out_of_order";
            return decision;
        }
        decision.imu_dt_max = std::max(decision.imu_dt_max, dt);
        if (dt > config.max_imu_dt)
        {
            decision.accepted = false;
            decision.rejection_reason = "imu_dt";
            return decision;
        }
    }
    if (previous_imu_end >= 0.0)
    {
        decision.imu_cross_gap = imu_timestamps.front() - previous_imu_end;
        if (!std::isfinite(decision.imu_cross_gap))
        {
            decision.accepted = false;
            decision.rejection_reason = "non_finite_imu_gap";
            return decision;
        }
        if (decision.imu_cross_gap < 0.0)
        {
            decision.accepted = false;
            decision.rejection_reason = "imu_timestamp_out_of_order";
            return decision;
        }
        if (decision.imu_cross_gap == 0.0)
        {
            decision.accepted = false;
            decision.rejection_reason = "zero_imu_dt";
            return decision;
        }
        decision.imu_dt_max = std::max(
            decision.imu_dt_max, decision.imu_cross_gap);
        if (decision.imu_cross_gap > config.max_imu_dt)
        {
            decision.accepted = false;
            decision.rejection_reason = "imu_cross_frame_gap";
            return decision;
        }
    }
    // Check the LiDAR epoch after validating the IMU batch so a malformed
    // IMU sample cannot be hidden behind a simultaneous large scan gap.
    if (previous_lidar_end >= 0.0)
    {
        decision.scan_gap = lidar_begin - previous_lidar_end;
        if (!std::isfinite(decision.scan_gap))
        {
            decision.accepted = false;
            decision.rejection_reason = "non_finite_scan_gap";
            return decision;
        }
        // A tiny negative gap is the expected overlap between adjacent
        // Mid-360 point-time windows. It must not be confused with a
        // genuinely reordered scan; the configured bound keeps the
        // fail-closed discontinuity check intact.
        if (decision.scan_gap < -config.max_scan_overlap)
        {
            decision.accepted = false;
            decision.rejection_reason = "lidar_out_of_order";
            return decision;
        }
        if (decision.scan_gap > config.max_scan_gap)
        {
            decision.accepted = false;
            decision.rejection_reason = "scan_gap";
            return decision;
        }
    }
    return decision;
}

// Mutable session cursor used by the mapping node and by the ROS-free
// regression tests. A rejected frame is terminal for this state; only
// reset() may establish a new sensor epoch.
struct LioTimeGuardSessionState
{
    bool failed = false;
    std::uint64_t rejected_frames = 0U;
    double previous_lidar_end = -1.0;
    double previous_imu_end = -1.0;
    std::string failure_reason;

    void reset()
    {
        failed = false;
        rejected_frames = 0U;
        previous_lidar_end = -1.0;
        previous_imu_end = -1.0;
        failure_reason.clear();
    }
};

inline void fail_lio_time_guard_session(
    LioTimeGuardSessionState& session, const char* reason)
{
    if (!session.failed)
    {
        session.failed = true;
        session.failure_reason =
            reason != nullptr ? reason : "unknown";
    }
}

inline bool commit_lio_time_guard_frame(
    LioTimeGuardSessionState& session,
    const LioTimeGuardDecision& decision,
    double lidar_end,
    double imu_end)
{
    if (session.failed)
    {
        return false;
    }
    if (!decision.accepted)
    {
        // A rejected frame is dropped from LIO integration but must not
        // terminate the session: transient reordering or a single gap
        // would otherwise poison minutes of mapping and block saving.
        // Count it, remember the reason for diagnostics, and re-anchor
        // the cursor so the stream can resume.
        ++session.rejected_frames;
        session.failure_reason =
            decision.rejection_reason != nullptr
                ? decision.rejection_reason : "unknown";
        if (std::isfinite(lidar_end))
        {
            session.previous_lidar_end = lidar_end;
        }
        if (std::isfinite(imu_end))
        {
            session.previous_imu_end = imu_end;
        }
        return false;
    }
    if (!std::isfinite(lidar_end) || !std::isfinite(imu_end))
    {
        ++session.rejected_frames;
        return false;
    }
    session.previous_lidar_end = lidar_end;
    session.previous_imu_end = imu_end;
    return true;
}

}  // namespace anubis_mapping
