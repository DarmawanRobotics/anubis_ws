#include "localization/global_localization_static_confirm.hpp"

#include <cmath>

namespace localization {
namespace {

constexpr double kPi = 3.14159265358979323846;

bool finitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

bool validTimePair(const rclcpp::Time& first, const rclcpp::Time& second) {
  return first.get_clock_type() == second.get_clock_type();
}

bool stampSet(const rclcpp::Time& stamp) {
  return stamp.nanoseconds() > 0;
}

double yawDeg(const Eigen::Matrix4d& pose) {
  return std::atan2(pose(1, 0), pose(0, 0)) * 180.0 / kPi;
}

}  // namespace

bool validStaticConfirmParams(const StaticConfirmParams& params) {
  return params.confirm_frames > 0 && finitePositive(params.confirm_xy_tol) &&
      finitePositive(params.confirm_yaw_tol_deg) &&
      finitePositive(params.probe_period_s) &&
      finitePositive(params.attempt_timeout_s);
}

bool staticConfirmPoseMatches(const Eigen::Matrix4d& first,
                              const Eigen::Matrix4d& second,
                              const StaticConfirmParams& params) {
  if (!first.allFinite() || !second.allFinite() ||
      !finitePositive(params.confirm_xy_tol) ||
      !finitePositive(params.confirm_yaw_tol_deg)) {
    return false;
  }
  const double dx = first(0, 3) - second(0, 3);
  const double dy = first(1, 3) - second(1, 3);
  if (std::hypot(dx, dy) > params.confirm_xy_tol) {
    return false;
  }
  // std::remainder keeps the comparison correct across the +/-180 deg wrap.
  const double yaw_delta =
      std::abs(std::remainder(yawDeg(first) - yawDeg(second), 360.0));
  return yaw_delta <= params.confirm_yaw_tol_deg;
}

bool shouldRunStaticProbe(const StaticConfirmState& state,
                          const rclcpp::Time& now,
                          const StaticConfirmParams& params) {
  if (!validStaticConfirmParams(params) || !stampSet(now)) {
    return false;
  }
  if (!stampSet(state.last_probe_stamp) ||
      !validTimePair(now, state.last_probe_stamp)) {
    return true;
  }
  const double elapsed = (now - state.last_probe_stamp).seconds();
  // A backwards clock step must not wedge the probe forever.
  return elapsed < 0.0 || elapsed >= params.probe_period_s;
}

StaticConfirmOutcome ingestStaticProbeResult(StaticConfirmState& state,
                                             const GlobalLocalizationResult& result,
                                             const rclcpp::Time& now,
                                             const StaticConfirmParams& params,
                                             GLSummaryCounts& summary) {
  if (!validStaticConfirmParams(params)) {
    return StaticConfirmOutcome::Skip;
  }
  ++state.probe_count;
  if (!stampSet(state.first_probe_stamp) ||
      !validTimePair(now, state.first_probe_stamp)) {
    state.first_probe_stamp = now;
  }
  state.last_probe_stamp = now;

  // Gates 1+2 live in the pipeline: PendingVote means the winning cluster
  // passed the quality threshold, carried enough independent support and was
  // not shadowed by a competing cluster somewhere else.
  const bool gates_passed = result.status == GLStatus::PendingVote &&
      result.final_pose.allFinite();
  if (!gates_passed) {
    state.agree_count = 0;
    state.has_pose = false;
  } else if (state.has_pose &&
             staticConfirmPoseMatches(state.last_pose, result.final_pose, params)) {
    ++state.agree_count;
    state.last_pose = result.final_pose;
  } else {
    // First usable pose, or the probe moved somewhere else: restart the chain
    // from this pose rather than discarding it.
    state.agree_count = 1;
    state.has_pose = true;
    state.last_pose = result.final_pose;
  }

  summary.static_confirm_probe_count = state.probe_count;
  summary.static_confirm_agree_count = state.agree_count;

  if (state.agree_count >= params.confirm_frames) {
    return StaticConfirmOutcome::Confirmed;
  }
  // Timeout is only meaningful once an attempt has actually started, and is
  // checked AFTER confirmation so a last-moment agreement still wins.
  if (validTimePair(now, state.first_probe_stamp) &&
      (now - state.first_probe_stamp).seconds() > params.attempt_timeout_s) {
    return StaticConfirmOutcome::Timeout;
  }
  return StaticConfirmOutcome::Pending;
}

void resetStaticConfirm(StaticConfirmState& state) {
  state = StaticConfirmState{};
}

}  // namespace localization
