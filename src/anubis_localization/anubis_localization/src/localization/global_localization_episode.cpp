#include "localization/global_localization_episode.hpp"

#include <cmath>

namespace localization {
namespace {

bool finitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

bool validTimePair(const rclcpp::Time& first, const rclcpp::Time& second) {
  return first.get_clock_type() == second.get_clock_type();
}

}  // namespace

void beginGLEpisode(GLEpisodeState& state,
                    uint64_t episode_id,
                    const rclcpp::Time& now,
                    const GLParams& params) {
  (void)params;
  state = GLEpisodeState{};
  state.active = true;
  state.episode_id = episode_id;
  state.start_time = now;
  state.phase = GLEpisodePhase::RecallAnchor;
  state.status = GLStatus::NoCandidate;
}

void endGLEpisode(GLEpisodeState& state, GLStatus final_status) {
  state.active = false;
  state.status = final_status;
  state.phase = final_status == GLStatus::Confirmed ? GLEpisodePhase::Idle
                                                    : GLEpisodePhase::AwaitingPrior;
}

bool checkGLEpisodeTimeout(GLEpisodeState& state,
                           const rclcpp::Time& now,
                           const GLParams& params) {
  if (!state.active || !validTimePair(now, state.start_time) ||
      !finitePositive(params.episode_timeout_s)) {
    return false;
  }
  const int64_t elapsed_ns = now.nanoseconds() - state.start_time.nanoseconds();
  if (elapsed_ns < 0 ||
      elapsed_ns < static_cast<int64_t>(params.episode_timeout_s * 1e9)) {
    return false;
  }
  state.active = false;
  state.phase = GLEpisodePhase::AwaitingPrior;
  state.status = GLStatus::NoCandidate;
  state.budget_exceeded = true;
  return true;
}

}  // namespace localization
