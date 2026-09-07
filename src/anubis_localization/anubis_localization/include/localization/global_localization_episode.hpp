#ifndef GLOBAL_LOCALIZATION_EPISODE_HPP
#define GLOBAL_LOCALIZATION_EPISODE_HPP

#include <cstdint>

#include <rclcpp/time.hpp>

#include "localization/global_localization_types.hpp"

namespace localization {

// Lightweight lifecycle wrapper around one global-localization attempt.
//
// [2026-08-14] The motion-driven temporal vote (anchor odom baseline, low
// dynamic dwell, candidate bank, rotation-evidence latches and the re-anchor
// budget) was removed together with the grid-recall era it was built for.
// Descriptor retrieval makes a recall probe cheap enough to repeat in place,
// so confirmation now comes from global_localization_static_confirm.hpp:
// N consecutive independent probes must agree. What remains here is only the
// episode bookkeeping the diagnostics and the output gate still need.
struct GLEpisodeState {
  GLEpisodeState() : start_time(int64_t{0}, RCL_ROS_TIME) {}

  bool active = false;
  uint64_t episode_id = 0;
  GLEpisodePhase phase = GLEpisodePhase::Idle;
  GLStatus status = GLStatus::NoCandidate;
  rclcpp::Time start_time;
  bool budget_exceeded = false;
  bool anchor_invalid = false;
};

void beginGLEpisode(GLEpisodeState& state,
                    uint64_t episode_id,
                    const rclcpp::Time& now,
                    const GLParams& params);

void endGLEpisode(GLEpisodeState& state, GLStatus final_status);

// True (and moves the episode to AwaitingPrior) once episode_timeout_s has
// elapsed since the episode started.
bool checkGLEpisodeTimeout(GLEpisodeState& state,
                           const rclcpp::Time& now,
                           const GLParams& params);

}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_EPISODE_HPP
