#ifndef LOCALIZATION__TRACKING_HEALTH_MONITOR_HPP_
#define LOCALIZATION__TRACKING_HEALTH_MONITOR_HPP_

#include <algorithm>

namespace localization {

class TrackingHealthMonitor {
public:
  enum class State {
    Healthy,
    Grace,
    TimedOut,
  };

  struct Update {
    State state = State::Healthy;
    bool state_changed = false;
    bool localization_valid = true;
    bool freeze_tf = false;
    bool enter_relocalization = false;
  };

  TrackingHealthMonitor(double rejection_timeout_s, int recovery_valid_frames)
  : rejection_timeout_s_(std::max(0.0, rejection_timeout_s)),
    recovery_valid_frames_(std::max(1, recovery_valid_frames)) {}

  void configure(double rejection_timeout_s, int recovery_valid_frames) {
    rejection_timeout_s_ = std::max(0.0, rejection_timeout_s);
    recovery_valid_frames_ = std::max(1, recovery_valid_frames);
    resetHealthy();
  }

  Update observe(bool accepted, double stamp_s) {
    const State previous = state_;

    if (state_ == State::Healthy && !accepted) {
      state_ = State::Grace;
      rejection_start_s_ = stamp_s;
      recovery_count_ = 0;
    } else if (state_ == State::Grace) {
      // A frame that matches again is recovery evidence even when it arrives at
      // the grace deadline.  Timing out before checking `accepted` made a good
      // frame enter TimedOut and produced the contradictory runtime message
      // "after 0 rejected frames".  Only a currently rejected frame may time
      // out the episode; accepted frames keep accumulating toward recovery.
      if (accepted) {
        ++recovery_count_;
        if (recovery_count_ >= recovery_valid_frames_) {
          state_ = State::Healthy;
          recovery_count_ = 0;
        }
      } else if (stamp_s - rejection_start_s_ >= rejection_timeout_s_) {
        state_ = State::TimedOut;
        recovery_count_ = 0;
      } else {
        recovery_count_ = 0;
      }
    }

    return makeUpdate(previous);
  }

  void resetHealthy() {
    state_ = State::Healthy;
    rejection_start_s_ = 0.0;
    recovery_count_ = 0;
  }

  State state() const { return state_; }
  // [2026-08-16] Grace 期间位姿仍可用（只是稍旧，TF 已冻结），不应宣告
  // 定位无效：否则单帧拒帧就触发 bridge 零速锁停（实测 71 次/轮，狗走
  // 两步停一下）。只有 TimedOut（grace 超时仍未恢复）才真正无效。
  bool localizationValid() const { return state_ != State::TimedOut; }
  bool freezeTf() const { return state_ == State::Grace; }
  int recoveryCount() const { return recovery_count_; }

private:
  Update makeUpdate(State previous) const {
    Update update;
    update.state = state_;
    update.state_changed = state_ != previous;
    update.localization_valid = state_ != State::TimedOut;
    update.freeze_tf = state_ == State::Grace;
    update.enter_relocalization =
      state_ == State::TimedOut && previous != State::TimedOut;
    return update;
  }

  double rejection_timeout_s_ = 2.0;
  int recovery_valid_frames_ = 3;
  State state_ = State::Healthy;
  double rejection_start_s_ = 0.0;
  int recovery_count_ = 0;
};

}  // namespace localization

#endif  // LOCALIZATION__TRACKING_HEALTH_MONITOR_HPP_
