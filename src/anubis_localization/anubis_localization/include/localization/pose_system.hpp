#ifndef POSE_SYSTEM_HPP
#define POSE_SYSTEM_HPP

#include <kkl/alg/unscented_kalman_filter.hpp>

namespace localization {

/**
 * @brief Definition of system to be estimated by ukf
 * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
 */
class PoseSystem {
public:
  typedef float T;
  typedef Eigen::Matrix<T, 3, 1> Vector3t;
  typedef Eigen::Matrix<T, 4, 4> Matrix4t;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Quaternion<T> Quaterniont;
public:
  static constexpr T kGravityMagnitude = 9.80665f;

  explicit PoseSystem(bool freeze_vertical_velocity = true)
  : freeze_vertical_velocity_(freeze_vertical_velocity),
    g_map_(0.0f, 0.0f, kGravityMagnitude) {
    dt = 0.01;
  }

  /**
   * @brief 设置 map 系下的重力向量
   *
   * [2026-08-16] 原实现硬编码 g=(0,0,9.80665)，等价于"假设地图 z 轴严格对齐
   * 重力"。实际地图由未标定外参的雷达建出，地面平面拟合倾角实测 3.18°；姿态
   * 与该假设每失配 δ，本方程每帧就泄漏 g·sin(δ) 的虚假水平加速度（δ=3.18° →
   * 0.54 m/s²，拒帧宽限 2s 自由积分即 1.09 m/s 幽灵速度）。
   *
   * 改为注入后，由 PoseEstimator 用自学习的 map 系重力方向设置，使本方程与
   * 姿态锚定用的是同一个重力基准，泄漏项归零。默认值与改动前完全一致。
   */
  void set_gravity(const Vector3t& g_map) { g_map_ = g_map; }
  const Vector3t& gravity() const { return g_map_; }

  // ---- 流形修正钩子（2026-08-16）-------------------------------------------
  // UKF 把 16 维状态当欧氏向量处理，但第 6~9 维是四元数，活在 S³ 上。两个后果
  // 实测都发生了：
  //   1) `mean = mean_pred`（或 correct 里的 mean + K·innovation）之后模长不再
  //      为 1，而 h() 又把观测侧归一化掉，模长成了**不可观测且无阻尼**的自由度。
  //      实测均值四元数模长塌到 0.1424（正常应为 1.0）。
  //   2) q 与 −q 表示同一旋转。sigma 点扰动实测达 0.28~0.73（对应姿态角 31°~72°），
  //      足以让部分 sigma 点落到对映半球，算术平均时相互抵消。
  // 由 System 提供钩子而不是让 UKF 硬编码索引，是为了保持 UKF 的泛型性。
  static constexpr int kQuatIndex    = 6;   ///< 状态向量里四元数起始行
  static constexpr int kObsQuatIndex = 3;   ///< 观测向量里四元数起始行

  /** 把状态里的四元数拉回单位球 */
  void normalizeState(VectorXt& state) const {
    if (state.size() < kQuatIndex + 4) {
      return;
    }
    const T n = state.middleRows(kQuatIndex, 4).norm();
    if (n > T(1e-6)) {
      state.middleRows(kQuatIndex, 4) /= n;
    }
  }

  /** 半球对齐：与参考四元数内积为负则整体取反（同一旋转，避免平均时相消） */
  void alignStateToReference(VectorXt& state, const VectorXt& reference) const {
    if (state.size() < kQuatIndex + 4 || reference.size() < kQuatIndex + 4) {
      return;
    }
    if (state.middleRows(kQuatIndex, 4).dot(reference.middleRows(kQuatIndex, 4)) < T(0)) {
      state.middleRows(kQuatIndex, 4) *= T(-1);
    }
  }

  /** 观测向量的半球对齐（观测是 [pos(3), quat(4)]） */
  void alignObservationToReference(VectorXt& obs, const VectorXt& reference) const {
    if (obs.size() < kObsQuatIndex + 4 || reference.size() < kObsQuatIndex + 4) {
      return;
    }
    if (obs.middleRows(kObsQuatIndex, 4).dot(reference.middleRows(kObsQuatIndex, 4)) < T(0)) {
      obs.middleRows(kObsQuatIndex, 4) *= T(-1);
    }
  }

  // system equation (without input)
  VectorXt f(const VectorXt& state) const {
    VectorXt next_state(16);

    Vector3t pt = state.middleRows(0, 3);
    Vector3t vt = state.middleRows(3, 3);
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    qt.normalize();

    Vector3t acc_bias = state.middleRows(10, 3);
    Vector3t gyro_bias = state.middleRows(13, 3);

    // position
    next_state.middleRows(0, 3) = pt + vt * dt;  //

    // velocity
    next_state.middleRows(3, 3) = vt;

    // orientation
    Quaterniont qt_ = qt;

    next_state.middleRows(6, 4) << qt_.w(), qt_.x(), qt_.y(), qt_.z();
    next_state.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
    next_state.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity

    return next_state;
  }

  // system equation
  VectorXt f(const VectorXt& state, const VectorXt& control) const {
    VectorXt next_state(16);

    Vector3t pt = state.middleRows(0, 3);
    Vector3t vt = state.middleRows(3, 3);
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    qt.normalize();

    Vector3t acc_bias = state.middleRows(10, 3);
    Vector3t gyro_bias = state.middleRows(13, 3);

    Vector3t raw_acc = control.middleRows(0, 3);
    Vector3t raw_gyro = control.middleRows(3, 3);

    // position
    next_state.middleRows(0, 3) = pt + vt * dt;  //

    // velocity
    // Integrate acceleration in map frame. Vertical velocity remains frozen until
    // the G1 static gate explicitly enables full 3D prediction.
    Vector3t acc_ = raw_acc - acc_bias;
    Vector3t acc = qt * acc_;
    next_state.middleRows(3, 3) = vt + (acc - g_map_) * dt;
    if (freeze_vertical_velocity_) {
      next_state[5] = state[5];
    }

    // orientation
    Vector3t gyro = raw_gyro - gyro_bias;
    Quaterniont dq(1, gyro[0] * dt / 2, gyro[1] * dt / 2, gyro[2] * dt / 2);
    dq.normalize();
    Quaterniont qt_ = (qt * dq).normalized();
    next_state.middleRows(6, 4) << qt_.w(), qt_.x(), qt_.y(), qt_.z();

    next_state.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
    next_state.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity

    return next_state;
  }

  // observation equation
  VectorXt h(const VectorXt& state) const {
    VectorXt observation(7);
    observation.middleRows(0, 3) = state.middleRows(0, 3);
    observation.middleRows(3, 4) = state.middleRows(6, 4).normalized();

    return observation;
  }

  double dt;

private:
  bool freeze_vertical_velocity_;
  Vector3t g_map_;   ///< map 系重力向量，见 set_gravity()
};

}

#endif // POSE_SYSTEM_HPP
