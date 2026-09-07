#ifndef LOCALIZATION__POSE_PREDICTION_HPP_
#define LOCALIZATION__POSE_PREDICTION_HPP_

#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace localization {

inline Eigen::Matrix4f applyOdomDelta(
    const Eigen::Matrix4f& last_accepted_observation,
    const Eigen::Matrix4f& odom_delta) {
  return last_accepted_observation * odom_delta;
}

inline Eigen::Vector3f rotationToRpy(const Eigen::Matrix3f& rotation) {
  const float roll = std::atan2(rotation(2, 1), rotation(2, 2));
  const float pitch = std::atan2(
    -rotation(2, 0),
    std::hypot(rotation(2, 1), rotation(2, 2)));
  const float yaw = std::atan2(rotation(1, 0), rotation(0, 0));
  return {roll, pitch, yaw};
}

inline Eigen::Quaternionf orientationFromGravityPreservingYaw(
    const Eigen::Vector3f& up_body, float yaw) {
  const float norm = up_body.norm();
  if (!std::isfinite(yaw)) {
    return Eigen::Quaternionf::Identity();
  }
  if (!up_body.allFinite() || norm < 1e-6f) {
    return Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
  }

  const Eigen::Vector3f up = up_body / norm;
  const float roll = std::atan2(up.y(), up.z());
  const float pitch = std::atan2(-up.x(), std::hypot(up.y(), up.z()));
  Eigen::Quaternionf orientation(
    Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
    Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
    Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));
  orientation.normalize();
  return orientation;
}

/**
 * @brief 判断这一帧 IMU 采样能否当作纯重力观测
 *
 * 门控挡不住与重力垂直的水平加速度（1 m/s² 只让比力模长变 0.05 m/s²，却带来
 * 5.8° 的倾角误差）。真正压住它的是"小增益 + 步态水平加速度零均值"，本门控
 * 只负责剔除明显的高动态样本（急停、落足冲击、快速转身）。
 */
inline bool isLowDynamicGravitySample(
    const Eigen::Vector3f& acc_body, const Eigen::Vector3f& gyro_body,
    float gravity_magnitude, float acc_tolerance, float gyro_tolerance) {
  if (!acc_body.allFinite() || !gyro_body.allFinite()) {
    return false;
  }
  const float acc_norm = acc_body.norm();
  if (acc_norm < 1e-3f) {
    return false;
  }
  return std::abs(acc_norm - gravity_magnitude) < acc_tolerance &&
    gyro_body.norm() < gyro_tolerance;
}

/**
 * @brief 用重力的 roll/pitch 重建姿态，**精确保持 yaw**
 *
 * [2026-08-16] 为什么不能直接用"对齐两向量的最小旋转"：该旋转的转轴虽然水平、
 * 不含 yaw 分量，但当 roll 与 pitch **同时非零**时，左乘它仍会改变 ZYX 分解出
 * 的 yaw（姿态已倾斜时绕水平轴转动与 yaw 存在几何耦合）。实测 roll=3.4°、
 * pitch=10.5° 时注入 0.44° 的 yaw 偏移——航向应完全由 NDT 决定，这是污染。
 *
 * 本函数改为显式重建：把姿态转到"重力对齐系"取其 yaw，用重力解出的 roll/pitch
 * 加该 yaw 重新组装，再转回 map 系。yaw 按构造精确保留。
 */
inline Eigen::Quaternionf gravityLeveledOrientation(
    const Eigen::Quaternionf& current,
    const Eigen::Vector3f& gravity_body,
    const Eigen::Vector3f& gravity_map_dir) {
  const float gb_norm = gravity_body.norm();
  const float gm_norm = gravity_map_dir.norm();
  if (gb_norm < 1e-6f || gm_norm < 1e-6f ||
      !gravity_body.allFinite() || !gravity_map_dir.allFinite()) {
    return current;
  }
  const Eigen::Quaternionf q = current.normalized();
  // 重力对齐系 → map 系（地图 z 轴与真实重力的固定偏差，实测 1.23°）
  const Eigen::Quaternionf r_g = Eigen::Quaternionf::FromTwoVectors(
    Eigen::Vector3f::UnitZ(), gravity_map_dir / gm_norm);
  // 在重力对齐系里取 yaw —— 这才是物理意义上的航向
  const Eigen::Quaternionf q_level = (r_g.conjugate() * q).normalized();
  const float yaw = rotationToRpy(q_level.toRotationMatrix()).z();
  const Eigen::Quaternionf leveled =
    orientationFromGravityPreservingYaw(gravity_body / gb_norm, yaw);
  return (r_g * leveled).normalized();
}

/**
 * @brief 把姿态朝重力基准方向拉回 alpha 比例，**不改变 yaw**
 *
 * alpha=1 时等价于 gravityLeveledOrientation（完全替换 roll/pitch，用于改写
 * NDT 观测）；小 alpha 用于 predict 侧的互补滤波。
 */
inline Eigen::Quaternionf gravityAnchoredOrientation(
    const Eigen::Quaternionf& current,
    const Eigen::Vector3f& gravity_body,
    const Eigen::Vector3f& gravity_map_dir,
    float alpha) {
  if (!(alpha > 0.0f)) {
    return current;
  }
  const Eigen::Quaternionf q = current.normalized();
  const Eigen::Quaternionf target =
    gravityLeveledOrientation(q, gravity_body, gravity_map_dir);
  if (alpha >= 1.0f) {
    return target;
  }
  return q.slerp(alpha, target).normalized();
}

/** @brief 当前姿态下，重力观测方向与 map 系重力基准的夹角（度） */
inline float gravityTiltErrorDeg(
    const Eigen::Quaternionf& current,
    const Eigen::Vector3f& gravity_body,
    const Eigen::Vector3f& gravity_map_dir) {
  const float gb_norm = gravity_body.norm();
  const float gm_norm = gravity_map_dir.norm();
  if (gb_norm < 1e-6f || gm_norm < 1e-6f) {
    return 0.0f;
  }
  const Eigen::Vector3f measured_map =
    current.normalized() * (gravity_body / gb_norm);
  const float cos_angle = std::clamp(
    measured_map.dot(gravity_map_dir / gm_norm), -1.0f, 1.0f);
  return std::acos(cos_angle) * 180.0f / static_cast<float>(M_PI);
}

/**
 * @brief 一阶低通的单步系数（按 dt 折算，行为不随采样率变化）
 *
 * [2026-08-16] 用低通滤波取代"瞬时低动态门控"来估计重力方向。
 *
 * 原门控要求 |‖acc‖-g|<0.6 且 ‖gyro‖<0.4rad/s，**四足行走时几乎永远不成立**
 * （落足冲击、机身俯仰角速度都远超），实测导致重力锚定"一走就下线"：静止时
 * pitch 跨度 7.9°，行走时 22.6°；z 在运动中塌陷 0.3~0.6m，停下才恢复。
 *
 * 低通方案的依据：四足步态的水平加速度在**一个步态周期内是零均值的**（身体没
 * 有净加速度），所以用比步态周期长的时间常数滤波，剩下的就是重力——行走时同样
 * 有效。残余偏差只来自真实的持续加速：0.5 m/s² 持续加速对应 2.9° 倾角误差。
 */
inline float lowPassAlpha(float dt, float tau) {
  if (!(tau > 0.0f)) {
    return 1.0f;
  }
  if (!(dt > 0.0f)) {
    return 0.0f;
  }
  return std::min(1.0f, dt / (tau + dt));
}

/** @brief 两个位姿之间的 yaw 差（度），已归一化到 [-180,180] */
inline float yawDeltaDeg(const Eigen::Matrix3f& from, const Eigen::Matrix3f& to) {
  float d = rotationToRpy(to).z() - rotationToRpy(from).z();
  while (d > static_cast<float>(M_PI)) d -= 2.0f * static_cast<float>(M_PI);
  while (d < -static_cast<float>(M_PI)) d += 2.0f * static_cast<float>(M_PI);
  return d * 180.0f / static_cast<float>(M_PI);
}

inline bool initialPoseRequiresReset(
    bool has_set_initial_pose, bool initialization_succeeded,
    bool initialpose_requested, float position_change,
    float quaternion_change, float position_threshold,
    float quaternion_threshold) {
  return !has_set_initial_pose || !initialization_succeeded ||
    initialpose_requested || position_change > position_threshold ||
    quaternion_change > quaternion_threshold;
}

/**
 * @brief 融合 IMU 预测与 SDK 里程计预测，作为 NDT 的初值
 *
 * 分工：
 *  - x/y、yaw 取 odom —— odom_prediction 是"上一次已接受的 NDT 位姿 × odom
 *    增量"，误差只累积一帧；UKF 积分的水平位置在拒帧宽限期会自由发散。
 *  - z、roll/pitch 取 IMU/UKF —— 见下方说明。
 *
 * [2026-08-16 上午] 曾把 z 也改成取 odom，理由是"UKF 的 z 会发散"。
 * [2026-08-16 下午 已回退] 实测数据推翻了这个理由：`innov dz`（NDT 相对预测的
 * z 修正）每帧只有 ±0.03 且正负交替，而 z 却掉了 0.64m —— 说明**下沉是预测带着
 * 走的**，而预测的 z 恰恰来自 odom。腿式里程计的 z 靠腿部运动学估机身高度，
 * 本就不可靠。
 * 而 UKF 侧 `enable_vertical_velocity_prediction: false`（vz 冻结）意味着两次
 * NDT 之间 z 基本不变 —— 对一个"没有可靠垂直观测"的系统，**保持不变**是比
 * "跟随不可靠的里程计"更安全的预测。
 */
inline Eigen::Matrix4f composePlanarOdomPrediction(
    const Eigen::Matrix4f& imu_prediction,
    const Eigen::Matrix4f& odom_prediction) {
  Eigen::Matrix4f prediction = imu_prediction;
  prediction(0, 3) = odom_prediction(0, 3);
  prediction(1, 3) = odom_prediction(1, 3);
  // z 保持 imu_prediction（见上方注释，不要再改成 odom）

  const Eigen::Vector3f imu_rpy = rotationToRpy(
    imu_prediction.block<3, 3>(0, 0));
  const float odom_yaw = rotationToRpy(
    odom_prediction.block<3, 3>(0, 0)).z();
  prediction.block<3, 3>(0, 0) =
    (Eigen::AngleAxisf(odom_yaw, Eigen::Vector3f::UnitZ()) *
     Eigen::AngleAxisf(imu_rpy.y(), Eigen::Vector3f::UnitY()) *
     Eigen::AngleAxisf(imu_rpy.x(), Eigen::Vector3f::UnitX()))
      .toRotationMatrix();
  return prediction;
}

}  // namespace localization

#endif  // LOCALIZATION__POSE_PREDICTION_HPP_
