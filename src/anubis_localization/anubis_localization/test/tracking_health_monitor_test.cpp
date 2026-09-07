#include <gtest/gtest.h>

#include <localization/tracking_health_monitor.hpp>

namespace localization {

TEST(TrackingHealthMonitor, FreezesImmediatelyAndRecoversAfterConsecutiveFrames) {
  TrackingHealthMonitor monitor(2.0, 3);

  auto update = monitor.observe(false, 10.0);
  // [2026-08-16] Grace 期间定位仍视为有效（TF 冻结、位姿稍旧但可用），
  // 只有 TimedOut 才置 invalid —— 单帧抖动不再触发 bridge 零速锁停。
  EXPECT_TRUE(update.localization_valid);
  EXPECT_TRUE(update.freeze_tf);
  EXPECT_FALSE(update.enter_relocalization);

  EXPECT_TRUE(monitor.observe(true, 10.1).freeze_tf);
  EXPECT_TRUE(monitor.observe(true, 10.2).freeze_tf);
  update = monitor.observe(true, 10.3);
  EXPECT_TRUE(update.localization_valid);
  EXPECT_FALSE(update.freeze_tf);
}

TEST(TrackingHealthMonitor, RejectionBreaksRecoverySequence) {
  TrackingHealthMonitor monitor(2.0, 2);
  monitor.observe(false, 5.0);
  monitor.observe(true, 5.1);
  monitor.observe(false, 5.2);

  EXPECT_TRUE(monitor.observe(true, 5.3).freeze_tf);
  EXPECT_TRUE(monitor.observe(true, 5.4).localization_valid);
}

TEST(TrackingHealthMonitor, TimesOutIntoRelocalization) {
  TrackingHealthMonitor monitor(2.0, 3);
  monitor.observe(false, 20.0);
  const auto update = monitor.observe(false, 22.0);

  EXPECT_FALSE(update.localization_valid);
  EXPECT_FALSE(update.freeze_tf);
  EXPECT_TRUE(update.enter_relocalization);
}

TEST(TrackingHealthMonitor, AcceptedFrameAtDeadlineContinuesRecovery) {
  TrackingHealthMonitor monitor(2.0, 3);
  monitor.observe(false, 10.0);

  auto update = monitor.observe(true, 12.0);
  EXPECT_TRUE(update.localization_valid);
  EXPECT_TRUE(update.freeze_tf);
  EXPECT_FALSE(update.enter_relocalization);

  EXPECT_TRUE(monitor.observe(true, 12.1).freeze_tf);
  update = monitor.observe(true, 12.2);
  EXPECT_TRUE(update.localization_valid);
  EXPECT_FALSE(update.freeze_tf);
  EXPECT_EQ(monitor.state(), TrackingHealthMonitor::State::Healthy);
}

TEST(TrackingHealthMonitor, RejectionAfterLatePartialRecoveryTimesOut) {
  TrackingHealthMonitor monitor(2.0, 3);
  monitor.observe(false, 10.0);
  monitor.observe(true, 12.0);

  const auto update = monitor.observe(false, 12.1);
  EXPECT_FALSE(update.localization_valid);
  EXPECT_FALSE(update.freeze_tf);
  EXPECT_TRUE(update.enter_relocalization);
}

}  // namespace localization
