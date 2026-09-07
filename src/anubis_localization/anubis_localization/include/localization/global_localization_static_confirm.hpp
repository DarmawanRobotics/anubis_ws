#ifndef GLOBAL_LOCALIZATION_STATIC_CONFIRM_HPP
#define GLOBAL_LOCALIZATION_STATIC_CONFIRM_HPP

#include <cstdint>

#include <Eigen/Core>
#include <rclcpp/time.hpp>

#include "localization/global_localization_types.hpp"

namespace localization {

// Static (motion-free) confirmation for descriptor-retrieval recall.
//
// The retrieval recall probe is fast enough (descriptor lookup + a few dozen
// seeded ICP runs, ~1 s) to be repeated on consecutive scans, which replaces
// the old motion-driven temporal vote: instead of asking the robot to walk or
// rotate so the recall can be re-run from a NEW viewpoint, the same viewpoint
// is re-evaluated N times independently.
//
// A pose is confirmed only when three gates all hold:
//   1. geometric verification  - overlap/inlier/residual/fitness thresholds,
//      already enforced by filterRefinedCandidates + selectBestCluster.
//   2. discriminative check    - the winning cluster must beat the best
//      competitor located somewhere ELSE by ambiguous_score_ratio, already
//      enforced by selectBestCluster (otherwise the result is Ambiguous).
//      This is what disambiguates self-similar rooms; it replaces the
//      multi-viewpoint vote with a comparison against the alternatives the
//      retrieval already returned.
//   3. temporal repeatability  - THIS module: confirm_frames consecutive
//      independent probes must agree within confirm_xy_tol / confirm_yaw_tol.
//      A transient mismatch (someone walking past, one noisy scan) cannot
//      produce the SAME wrong pose several times in a row.
//
// Moving the robot still helps when a scene is genuinely ambiguous, but it is
// no longer machinery: a new location simply produces a fresh, independent
// attempt. That removes the odometry anchor, drift compensation, rotation
// latches and re-anchor budget that the motion-vote path required.
struct StaticConfirmParams {
  // Consecutive agreeing probes required before a pose is confirmed.
  int confirm_frames = 3;
  // Agreement tolerance between consecutive probe poses.
  double confirm_xy_tol = 0.30;
  double confirm_yaw_tol_deg = 5.0;
  // Minimum spacing between probes; scans arrive at ~10 Hz but a probe costs
  // far more than 100 ms, so probing every scan would saturate the callback.
  double probe_period_s = 0.5;
  // Give up (and request an external prior) after this long without a confirm.
  double attempt_timeout_s = 30.0;
};

struct StaticConfirmState {
  StaticConfirmState()
      : first_probe_stamp(int64_t{0}, RCL_ROS_TIME),
        last_probe_stamp(int64_t{0}, RCL_ROS_TIME) {}

  int agree_count = 0;
  int probe_count = 0;
  bool has_pose = false;
  Eigen::Matrix4d last_pose = Eigen::Matrix4d::Identity();
  rclcpp::Time first_probe_stamp;
  rclcpp::Time last_probe_stamp;
};

enum class StaticConfirmOutcome : uint8_t {
  Skip,       // throttled; no probe should run on this scan
  Pending,    // probe ran, not enough agreement yet
  Confirmed,  // confirm_frames consecutive probes agreed
  Timeout,    // attempt_timeout_s elapsed without a confirmation
};

bool validStaticConfirmParams(const StaticConfirmParams& params);

// True when the accumulated poses agree within tolerance.
bool staticConfirmPoseMatches(const Eigen::Matrix4d& first,
                              const Eigen::Matrix4d& second,
                              const StaticConfirmParams& params);

// Throttle: probe only when the previous probe is at least probe_period_s old
// (or when no probe has run yet).
bool shouldRunStaticProbe(const StaticConfirmState& state,
                          const rclcpp::Time& now,
                          const StaticConfirmParams& params);

// Feed one completed probe result. `result.status == GLStatus::PendingVote`
// with a finite `final_pose` means gates 1+2 passed; anything else (Ambiguous,
// LowConsensus, NoCandidate) resets the agreement chain to zero.
StaticConfirmOutcome ingestStaticProbeResult(StaticConfirmState& state,
                                             const GlobalLocalizationResult& result,
                                             const rclcpp::Time& now,
                                             const StaticConfirmParams& params,
                                             GLSummaryCounts& summary);

void resetStaticConfirm(StaticConfirmState& state);

}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_STATIC_CONFIRM_HPP
