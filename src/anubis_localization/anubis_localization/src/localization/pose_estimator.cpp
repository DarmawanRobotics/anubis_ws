#include <localization/pose_estimator.hpp>

#include <algorithm>
#include <limits>
#include <numeric>

#include <pcl/filters/voxel_grid.h>
#include <localization/pose_system.hpp>
#include <localization/pose_prediction.hpp>
#include <kkl/alg/unscented_kalman_filter.hpp>

namespace localization {
/**
 * @brief constructor
 * @param registration        registration method
 * @param stamp               timestamp
 * @param pos                 initial position
 * @param quat                initial orientation
 * @param cool_time_duration  during "cool time", prediction is not performed
 * @param bias_acc            initial acceleration bias
 * @param bias_gyro           initial gyro bias
 */
PoseEstimator::PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const rclcpp::Time& stamp,
    const Eigen::Vector3f& pos, const Eigen::Quaternionf& quat, double cool_time_duration, double ndt_score_threshold,
    bool freeze_vertical_velocity, Eigen::Vector3d bias_acc, Eigen::Vector3d bias_gyro)
    : init_stamp(stamp), registration(registration), cool_time_duration(cool_time_duration),
      ndt_score_threshold_(ndt_score_threshold),
      match_result_{false, std::numeric_limits<float>::max()} {

  prev_stamp = rclcpp::Time((int64_t)0, init_stamp.get_clock_type());
  last_correction_stamp = rclcpp::Time((int64_t)0, init_stamp.get_clock_type());
  gravity_diag_window_start_ = rclcpp::Time((int64_t)0, init_stamp.get_clock_type());
  gravity_diag_norms_.reserve(256);
  last_observation = Eigen::Matrix4f::Identity();
  last_observation.block<3, 3>(0, 0) = quat.toRotationMatrix();
  last_observation.block<3, 1>(0, 3) = pos;

  process_noise = Eigen::MatrixXf::Identity(16, 16);
  process_noise.middleRows(0, 3) *= 0.5;     // 1.0
  process_noise.middleRows(3, 3) *= 1.0;
  process_noise.middleRows(6, 4) *= 0.5;
  process_noise.middleRows(10, 3) *= 1e-6;
  process_noise.middleRows(13, 3) *= 1e-6;

  // Round 10 base R (still): do not over-trust NDT jitter → velocity corruption.
  // R13 overrides R online when NDT displacement indicates real motion.
  measurement_noise_ = Eigen::MatrixXf::Identity(7, 7);
  measurement_noise_.middleRows(0, 3) *= 0.1;    // position std ≈ 0.32 m
  measurement_noise_.middleRows(3, 4) *= 0.005;  // quat component std ≈ 0.07

  last_ndt_stamp_ = rclcpp::Time((int64_t)0, init_stamp.get_clock_type());
  motion_window_stamp_ = last_ndt_stamp_;

  Eigen::VectorXf mean(16);
  mean.middleRows(0, 3) = pos;
  mean.middleRows(3, 3).setZero();
  mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());
  mean.middleRows(10, 3).setZero();
  mean.middleRows(13, 3) = bias_gyro.cast<float>();
  // mean.middleRows(13, 3).setZero();

  Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;

  PoseSystem system(freeze_vertical_velocity);
  ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>(
    system, 16, 6, 7, process_noise, measurement_noise_, mean, cov));
}

Eigen::MatrixXf PoseEstimator::scaledProcessNoise() const {
  Eigen::MatrixXf Q = process_noise;
  // Moving: inflate velocity (and mildly position) process noise so IMU foot-strike
  // residuals are trusted less and NDT can correct harder on the next update.
  float q_vel = 1.0f + 4.0f * motion_scale_;   // still=1x … moving=5x
  float q_pos = 1.0f + 1.0f * motion_scale_;   // still=1x … moving=2x
  // Ghost velocity while NDT says still: also inflate Q_vel so correct() can squash it.
  if (ukf) {
    const float vnorm = ukf->mean.middleRows<3>(3).norm();
    // Tier 1: moderate ghost — NDT says near-still but UKF velocity is building
    if (motion_scale_ < 0.30f && vnorm > 0.5f) {
      q_vel = std::max(q_vel, 4.0f);
    }
    // Tier 2: hard ghost — NDT very still but UKF |v| is large
    if (motion_scale_ < 0.25f && vnorm > 0.8f) {
      q_vel = std::max(q_vel, 6.0f);
    }
  }
  Q.middleRows(0, 3) *= q_pos;
  Q.middleRows(3, 3) *= q_vel;
  return Q;
}

PoseEstimator::~PoseEstimator() {}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const rclcpp::Time& stamp) {
  if ((stamp - init_stamp).seconds() < cool_time_duration || prev_stamp == rclcpp::Time((int64_t)0, prev_stamp.get_clock_type()) || prev_stamp == stamp) {
    prev_stamp = stamp;
    return;
  }

  double dt = (stamp - prev_stamp).seconds();
  prev_stamp = stamp;

  ukf->setProcessNoiseCov(scaledProcessNoise() * dt);
  ukf->system.dt = dt;

  ukf->predict();
}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const rclcpp::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
  if (/*(stamp - init_stamp).seconds() < cool_time_duration || */prev_stamp == rclcpp::Time((int64_t)0, prev_stamp.get_clock_type()) || prev_stamp == stamp) {
    prev_stamp = stamp;
    RCLCPP_INFO(rclcpp::get_logger("PoseEstimator"), "Some ploblems with prev_stamp, not predict!");
    return;
  }

  double dt = (stamp - prev_stamp).seconds();
  if (dt > 0.1){
    dt = 0.1;
  } else if (dt < 0.0) {
    RCLCPP_INFO(rclcpp::get_logger("PoseEstimator"), "dt < 0.0, not predict!");
    return;
  }
  prev_stamp = stamp;

  ukf->setProcessNoiseCov(scaledProcessNoise() * dt);
  ukf->system.dt = dt;

  Eigen::VectorXf control(6);
  control.head<3>() = acc;
  control.tail<3>() = gyro;

  // DIAG: log first few IMU predicts after (re)init to trace ghost-velocity root cause.
  // acc_bias / gyro_bias from UKF state [10:12] / [13:15]; control is body-frame acc/gyro.
  predict_diag_count_++;
  if (predict_diag_count_ <= 8) {
    const Eigen::Vector3f bias_acc  = ukf->mean.middleRows<3>(10);
    const Eigen::Vector3f bias_gyro = ukf->mean.middleRows<3>(13);
    const Eigen::Vector3f vel_before = ukf->mean.middleRows<3>(3);
    ukf->predict(control);
    const Eigen::Vector3f vel_after = ukf->mean.middleRows<3>(3);
    const Eigen::Vector3f vel_delta = vel_after - vel_before;
    RCLCPP_INFO(rclcpp::get_logger("PoseEstimator"),
      "[IMU DIAG #%d] dt=%.4fs acc_ctrl=[%.3f,%.3f,%.3f] gyro_ctrl=[%.4f,%.4f,%.4f] "
      "bias_acc=[%.4f,%.4f,%.4f] bias_gyro=[%.4f,%.4f,%.4f] "
      "v_before=[%.3f,%.3f,%.3f] v_after=[%.3f,%.3f,%.3f] v_delta=[%.3f,%.3f,%.3f]",
      predict_diag_count_, dt,
      acc.x(), acc.y(), acc.z(), gyro.x(), gyro.y(), gyro.z(),
      bias_acc.x(), bias_acc.y(), bias_acc.z(),
      bias_gyro.x(), bias_gyro.y(), bias_gyro.z(),
      vel_before.x(), vel_before.y(), vel_before.z(),
      vel_after.x(), vel_after.y(), vel_after.z(),
      vel_delta.x(), vel_delta.y(), vel_delta.z());
  } else {
    ukf->predict(control);
  }

  // 预测后立刻用重力观测锚定 roll/pitch，保证下一帧 NDT 拿到的初值是"正"的。
  // 放在 predict 之后而非之前：修正的是刚积分出来的新姿态。
  updateGravityAnchor(stamp, acc, gyro);
}

void PoseEstimator::predict_with_dt(const rclcpp::Time& stamp, const Eigen::Vector3f& acc,
                                    const Eigen::Vector3f& gyro, double dt) {
  if (!ukf) {
    return;
  }
  double capped = dt;
  if (prev_stamp.nanoseconds() != 0) {
    const double actual = (stamp - prev_stamp).seconds();
    if (actual > 0.0 && actual < capped) {
      capped = actual;
    }
  }
  // R11: hard-cap catch-up step so a backlog cannot integrate a full stall duration
  if (capped > 0.15) {
    capped = 0.15;
  }
  if (capped <= 0.0) {
    prev_stamp = stamp;
    return;
  }
  prev_stamp = stamp;

  ukf->setProcessNoiseCov(scaledProcessNoise() * static_cast<float>(capped));
  ukf->system.dt = static_cast<float>(capped);

  Eigen::VectorXf control(6);
  control.head<3>() = acc;
  control.tail<3>() = gyro;
  ukf->predict(control);

  updateGravityAnchor(stamp, acc, gyro);
}

void PoseEstimator::predict_without_imu(const rclcpp::Time& stamp, double dt) {
  if (!ukf) {
    return;
  }

  // The LiDAR period is the only trustworthy time interval while the IMU clock is
  // stale. Bound it so a delayed callback cannot inject a large covariance jump.
  if (!std::isfinite(dt)) {
    dt = 0.1;
  }
  dt = std::clamp(dt, 0.01, 0.15);

  // Keep the sensor-time cursor monotonic. A delayed IMU sample may arrive after
  // this fallback and must be rejected by the normal negative-dt guard until it
  // catches up, rather than moving the cursor backwards.
  if (prev_stamp.nanoseconds() == 0 || stamp > prev_stamp) {
    prev_stamp = stamp;
  }

  ukf->setProcessNoiseCov(scaledProcessNoise() * static_cast<float>(dt));
  ukf->system.dt = static_cast<float>(dt);
  // PoseSystem::f(state) is a constant-velocity propagation: no synthetic
  // acceleration or angular velocity is introduced.
  ukf->predict();
}

void PoseEstimator::set_prediction_diagnostic(
    bool imu_available, double imu_age_ms, bool fallback_active, double fallback_dt_s) {
  tracking_diag_.imu_prediction_available = imu_available;
  tracking_diag_.imu_fallback_active = fallback_active;
  tracking_diag_.imu_age_ms = std::isfinite(imu_age_ms)
    ? static_cast<float>(imu_age_ms) : std::numeric_limits<float>::quiet_NaN();
  tracking_diag_.imu_fallback_dt_ms = fallback_active
    ? static_cast<float>(std::max(0.0, fallback_dt_s) * 1000.0) : 0.0f;
}

long PoseEstimator::predict_count() const {
  return ukf ? ukf->predict_count : 0;
}

void PoseEstimator::reset_velocity() {
  if (!ukf) {
    return;
  }
  ukf->mean.middleRows<3>(3).setZero();
  // R13: after recovery, treat as still until NDT displacement rebuilds motion_scale_
  motion_scale_ = 0.0f;
  has_last_ndt_ = false;
  has_motion_window_ = false;
}

void PoseEstimator::set_initial_biases(const Eigen::Vector3f& acc_bias, const Eigen::Vector3f& gyro_bias){
  if(!ukf){
    return;
  }
  // state layout: [px,py,pz, vx,vy,vz, qw,qx,qy,qz, bax,bay,baz, bgx,bgy,bgz]
  ukf->mean.middleRows(10, 3) = acc_bias;
  ukf->mean.middleRows(13, 3) = gyro_bias;
}

void PoseEstimator::set_orientation(const Eigen::Quaternionf& quat){
  if(!ukf){
    return;
  }
  Eigen::Quaternionf q = quat.normalized();
  // state layout: [px,py,pz, vx,vy,vz, qw,qx,qy,qz, bax,bay,baz, bgx,bgy,bgz]
  ukf->mean[6] = q.w();
  ukf->mean[7] = q.x();
  ukf->mean[8] = q.y();
  ukf->mean[9] = q.z();
}

/**
 * @brief 用已收敛的重力方向播种低通滤波器（详见头文件注释）
 */
void PoseEstimator::seed_gravity_estimate(
    const Eigen::Vector3f& up_body, const rclcpp::Time& stamp) {
  if (!up_body.allFinite() || up_body.norm() < 1e-6f) {
    RCLCPP_WARN(logger_,
      "[GRAV SEED] 拒绝非法的 up_body [%.4f,%.4f,%.4f]，锚定将走常规预热",
      up_body.x(), up_body.y(), up_body.z());
    return;
  }

  const Eigen::Vector3f up = up_body.normalized();
  // gravity_lpf_ 存的是**比力**（静止时指向上、模长为 g），与 updateGravityAnchor
  // 里 acc_corrected 的口径一致；gravity_body_ 是它的单位向量。
  gravity_lpf_ = up * kGravityMagnitude;
  gravity_mag_lpf_ = kGravityMagnitude;
  gravity_body_ = up;
  has_gravity_lpf_ = true;
  has_gravity_body_ = true;
  // 标记为已收敛，跳过 20 帧预热——播种值本身就来自静态初始化的收敛结果
  gravity_lpf_count_ = kGravityLpfMinSamples;
  last_gravity_stamp_ = stamp;

  const float roll_deg = std::atan2(up.y(), up.z()) * 180.0f / static_cast<float>(M_PI);
  const float pitch_deg =
    std::atan2(-up.x(), std::hypot(up.y(), up.z())) * 180.0f / static_cast<float>(M_PI);
  RCLCPP_INFO(logger_,
    "[GRAV SEED] 重力锚定已播种 up_body=[%.4f,%.4f,%.4f] "
    "(roll=%.2f° pitch=%.2f°)；落位首帧即可用，不再有 %d 帧预热真空期",
    up.x(), up.y(), up.z(), roll_deg, pitch_deg, kGravityLpfMinSamples);
}

/**
 * @brief 设置 map 系重力方向（详见头文件注释）
 */
void PoseEstimator::set_map_gravity_direction(const Eigen::Vector3f& dir) {  if (!dir.allFinite() || dir.norm() < 1e-6f) {
    RCLCPP_WARN(logger_,
      "map_gravity_direction 非法 [%.4f,%.4f,%.4f]，保持 (0,0,1)",
      dir.x(), dir.y(), dir.z());
    return;
  }
  g_map_dir_ = dir.normalized();
  const float tilt_deg =
    std::acos(std::clamp(g_map_dir_.z(), -1.0f, 1.0f)) * 180.0f / static_cast<float>(M_PI);

  // 关键：机械编排里减去的重力必须与姿态锚定用的是同一基准。两者失配 δ 时，
  // PoseSystem::f() 每帧泄漏 g·sin(δ) 的虚假水平加速度。
  if (ukf) {
    ukf->system.set_gravity(g_map_dir_ * kGravityMagnitude);
  }

  RCLCPP_INFO(logger_,
    "Map gravity direction set from config: [%.4f,%.4f,%.4f] "
    "(tilt vs map z-axis = %.2f°)",
    g_map_dir_.x(), g_map_dir_.y(), g_map_dir_.z(), tilt_deg);
}

bool PoseEstimator::gravityObservationFresh(const rclcpp::Time& stamp) const {
  if (!has_gravity_body_ || last_gravity_stamp_.nanoseconds() == 0) {
    return false;
  }
  if (stamp.get_clock_type() != last_gravity_stamp_.get_clock_type()) {
    return false;
  }
  const double age = (stamp - last_gravity_stamp_).seconds();
  return age >= -kGravityObservationMaxAgeS && age <= kGravityObservationMaxAgeS;
}

void PoseEstimator::recordGravityDiagnostic(
    const rclcpp::Time& stamp, float acceleration_norm, bool usable) {
  if (!std::isfinite(acceleration_norm)) {
    return;
  }

  if (gravity_diag_window_start_.nanoseconds() == 0 ||
      gravity_diag_window_start_.get_clock_type() != stamp.get_clock_type() ||
      stamp < gravity_diag_window_start_) {
    gravity_diag_window_start_ = stamp;
    gravity_diag_norms_.clear();
    gravity_diag_total_ = 0;
    gravity_diag_usable_ = 0;
  }

  gravity_diag_norms_.push_back(acceleration_norm);
  ++gravity_diag_total_;
  if (usable) {
    ++gravity_diag_usable_;
  }

  if ((stamp - gravity_diag_window_start_).seconds() < 1.0 ||
      gravity_diag_norms_.empty()) {
    return;
  }

  std::vector<float> sorted = gravity_diag_norms_;
  std::sort(sorted.begin(), sorted.end());
  const float minimum = sorted.front();
  const float median = sorted[sorted.size() / 2];
  const float maximum = sorted.back();
  const float mean = std::accumulate(sorted.begin(), sorted.end(), 0.0f) /
    static_cast<float>(sorted.size());
  const float pass_rate = gravity_diag_total_ > 0
    ? static_cast<float>(gravity_diag_usable_) /
      static_cast<float>(gravity_diag_total_)
    : 0.0f;

  RCLCPP_INFO(
    logger_,
    "[GRAV] acc_norm_mps2[min=%.3f median=%.3f max=%.3f mean=%.3f] "
    "usable=%zu/%zu pass_rate=%.3f mag_lpf=%.3f mag_err=%.3f tol=%.2f",
    minimum, median, maximum, mean,
    gravity_diag_usable_, gravity_diag_total_, pass_rate,
    gravity_mag_lpf_, tracking_diag_.gravity_mag_err, kGravityLpfMagTol);

  gravity_diag_window_start_ = stamp;
  gravity_diag_norms_.clear();
  gravity_diag_total_ = 0;
  gravity_diag_usable_ = 0;
}

/**
 * @brief 用加速度计的重力观测锚定 UKF 的 roll/pitch（详见头文件注释）
 */
void PoseEstimator::updateGravityAnchor(const rclcpp::Time& stamp,
                                        const Eigen::Vector3f& acc,
                                        const Eigen::Vector3f& gyro) {
  (void)gyro;  // 低通方案不再依赖角速度门控（见头文件：门控在行走时永远不成立）
  if (!ukf) {
    return;
  }

  // 扣除已估计的加速度计零偏后再滤波；bias 与 acc 同在 body 系。
  const Eigen::Vector3f acc_bias = ukf->mean.middleRows<3>(10);
  const Eigen::Vector3f acc_corrected = acc - acc_bias;
  if (!acc_corrected.allFinite()) {
    return;
  }
  gravity_total_++;

  // 单帧离群剔除：落足冲击/自由落体会污染滤波器，但不该让已有估计失效。
  const float acc_norm = acc_corrected.norm();
  if (std::abs(acc_norm - kGravityMagnitude) > kGravitySampleOutlier) {
    tracking_diag_.gravity_accept = static_cast<float>(gravity_usable_) /
      static_cast<float>(gravity_total_);
    recordGravityDiagnostic(stamp, acc_norm, false);
    return;
  }

  // 一阶低通。时间常数 1s > 四足步态周期(0.3~0.5s)，步态的水平加速度是零均值
  // 的，滤完只剩重力。
  // **向量低通给方向，标量低通给有效性** —— 两者必须分开：向量平均会因姿态摆动
  // 造成的方向发散而缩短模长（±20° 摆动 → 缩水 6% ≈ 0.59 m/s²），用它判有效性
  // 会在运动中误杀重力（实测 gmag 0.33~0.43 撞上 0.35 阈值 → grav=0）。
  const float dt = static_cast<float>(ukf->system.dt);
  if (!has_gravity_lpf_) {
    gravity_lpf_ = acc_corrected;
    gravity_mag_lpf_ = acc_norm;
    has_gravity_lpf_ = true;
    gravity_lpf_count_ = 1;
  } else {
    const float a = lowPassAlpha(dt, kGravityLpfTauS);
    gravity_lpf_ = (1.0f - a) * gravity_lpf_ + a * acc_corrected;
    gravity_mag_lpf_ = (1.0f - a) * gravity_mag_lpf_ + a * acc_norm;
    if (gravity_lpf_count_ < kGravityLpfMinSamples) {
      gravity_lpf_count_++;
    }
  }

  const float lpf_norm = gravity_lpf_.norm();
  // 有效性用标量平均：只有真实的持续加速才会让它偏离 g
  tracking_diag_.gravity_mag_err = std::abs(gravity_mag_lpf_ - kGravityMagnitude);

  if (gravity_lpf_count_ < kGravityLpfMinSamples ||
      lpf_norm < 1e-3f ||
      tracking_diag_.gravity_mag_err > kGravityLpfMagTol) {
    tracking_diag_.gravity_accept = static_cast<float>(gravity_usable_) /
      static_cast<float>(gravity_total_);
    recordGravityDiagnostic(stamp, acc_norm, false);
    return;
  }

  gravity_body_ = gravity_lpf_ / lpf_norm;
  has_gravity_body_ = true;
  last_gravity_stamp_ = stamp;
  gravity_usable_++;

  const Eigen::Quaternionf q = quat();
  const float err_deg = gravityTiltErrorDeg(q, gravity_body_, g_map_dir_);

  const float alpha = std::min(
    kGravityAnchorMaxAlpha,
    kGravityAnchorGainPerSec * dt);

  set_orientation(gravityAnchoredOrientation(q, gravity_body_, g_map_dir_, alpha));

  // 重力样本可用率：直接暴露"锚定在运动中到底有没有工作"。
  if (gravity_total_ > 0) {
    tracking_diag_.gravity_accept =
      static_cast<float>(gravity_usable_) / static_cast<float>(gravity_total_);
  }
  recordGravityDiagnostic(stamp, acc_norm, true);

  if (err_deg > kGravityAnchorWarnDeg) {
    if ((gravity_warn_throttle_++ % 200) == 0) {
      RCLCPP_WARN(logger_,
        "Gravity anchor: attitude is %.1f° off the map gravity reference "
        "(pulling back at %.4f/frame). 若持续不收敛，检查 config 的 "
        "map_gravity_direction 是否与地图地面法向一致。",
        err_deg, alpha);
    }
  }
  // 注意：不要在 else 分支重置 throttle —— 误差在阈值附近抖动时会让计数器反复
  // 归零，导致每次越界都打印（实测刷屏 2-4 条/秒）。让它单调累加即可。
}

/**
 * @brief update the state of the odomety-based pose estimation
 */
void PoseEstimator::predict_odom(const Eigen::Matrix4f& odom_delta) {
  // odom_delta is inv(T_odom_base_last_ndt) * T_odom_base_now. Always rebuild
  // from the last accepted NDT pose so a rejected frame cannot apply a
  // cumulative odometry delta more than once.
  odom_prediction_ = applyOdomDelta(last_observation, odom_delta);
}

void PoseEstimator::set_trust_anchor(
    const Eigen::Matrix4f& map_pose, const Eigen::Matrix4f& odom_pose) {
  clear_trust_anchor();
  if (!map_pose.allFinite() || !odom_pose.allFinite()) {
    RCLCPP_WARN(logger_, "[TRUST] anchor rejected because map/odom pose is non-finite");
    return;
  }

  trust_anchor_map_pose_ = map_pose;
  trust_anchor_odom_pose_ = odom_pose;
  trust_current_odom_pose_ = odom_pose;
  trust_last_odom_pose_ = odom_pose;
  trust_anchor_valid_ = true;
  trust_current_odom_valid_ = true;
  trust_last_odom_valid_ = true;
  RCLCPP_INFO(
    logger_,
    "[TRUST] anchor set map_xy=[%.3f,%.3f] odom_xy=[%.3f,%.3f] "
    "base_allowance=%.2fm growth=%.2fm/m min_xy_gate=%.2fm",
    map_pose(0, 3), map_pose(1, 3), odom_pose(0, 3), odom_pose(1, 3),
    kTrustBaseAllowanceM, kTrustAllowancePerTravelM, kTrustMinCorrectionXY);
}

void PoseEstimator::clear_trust_anchor() {
  trust_anchor_valid_ = false;
  trust_current_odom_valid_ = false;
  trust_last_odom_valid_ = false;
  trust_travel_m_ = 0.0f;
  trust_exceed_count_ = 0;
}

void PoseEstimator::update_trust_odom(const Eigen::Matrix4f& odom_pose) {
  if (!trust_anchor_valid_ || !odom_pose.allFinite()) {
    return;
  }

  if (trust_last_odom_valid_) {
    const float step =
      (odom_pose.block<2, 1>(0, 3) - trust_last_odom_pose_.block<2, 1>(0, 3)).norm();
    if (!std::isfinite(step) || step > kTrustMaxOdomStepM) {
      RCLCPP_WARN(
        logger_,
        "[TRUST] anchor cleared after discontinuous odom step %.3fm (>%.2fm)",
        step, kTrustMaxOdomStepM);
      clear_trust_anchor();
      return;
    }
    trust_travel_m_ += step;
  }

  trust_current_odom_pose_ = odom_pose;
  trust_last_odom_pose_ = odom_pose;
  trust_current_odom_valid_ = true;
  trust_last_odom_valid_ = true;
}

/**
 * @brief correct
 * @param cloud   input cloud
 * @return cloud aligned to the globalmap
 */
pcl::PointCloud<PoseEstimator::PointT>::Ptr PoseEstimator::correct(const rclcpp::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
  tracking_diag_.attitude_chain_valid = false;
  const Eigen::Matrix4f imu_guess = matrix();
  const Eigen::Matrix4f odom_guess = odom_prediction_ ? *odom_prediction_ : imu_guess;
  const Eigen::Matrix4f init_guess = odom_prediction_
    ? composePlanarOdomPrediction(imu_guess, odom_guess)
    : imu_guess;

  tracking_diag_.position_chain_valid = true;
  tracking_diag_.ukf_correct_applied = false;
  tracking_diag_.ukf_mahalanobis_sq = -1.0f;
  tracking_diag_.pred_imu_x = imu_guess(0, 3);
  tracking_diag_.pred_imu_y = imu_guess(1, 3);
  tracking_diag_.pred_imu_z_full = imu_guess(2, 3);
  tracking_diag_.pred_odom_x = odom_guess(0, 3);
  tracking_diag_.pred_odom_y = odom_guess(1, 3);
  tracking_diag_.pred_odom_z_full = odom_guess(2, 3);
  tracking_diag_.pred_used_x = init_guess(0, 3);
  tracking_diag_.pred_used_y = init_guess(1, 3);
  tracking_diag_.pred_used_z_full = init_guess(2, 3);
  tracking_diag_.ukf_pre_x = ukf->mean[0];
  tracking_diag_.ukf_pre_y = ukf->mean[1];
  tracking_diag_.ukf_pre_z = ukf->mean[2];
  tracking_diag_.ukf_post_x = ukf->mean[0];
  tracking_diag_.ukf_post_y = ukf->mean[1];
  tracking_diag_.ukf_post_z = ukf->mean[2];
  tracking_diag_.ukf_pre_vx = ukf->mean[3];
  tracking_diag_.ukf_pre_vy = ukf->mean[4];
  tracking_diag_.ukf_pre_vz = ukf->mean[5];
  tracking_diag_.ukf_post_vx = ukf->mean[3];
  tracking_diag_.ukf_post_vy = ukf->mean[4];
  tracking_diag_.ukf_post_vz = ukf->mean[5];
  tracking_diag_.ukf_pos_cov_x = ukf->getCov()(0, 0);
  tracking_diag_.ukf_pos_cov_y = ukf->getCov()(1, 1);
  tracking_diag_.ukf_pos_cov_z = ukf->getCov()(2, 2);

  pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
  // [2026-08-17] setInputSource 与 align 分开计时：若前者不可忽略，说明每帧在
  // 重建源点云的搜索结构（PCL 的 initComputeReciprocal 会建源 KD 树），那是
  // 纯浪费——NDT 只需要目标侧的体素结构。
  const auto t_setsrc = std::chrono::steady_clock::now();
  registration->setInputSource(cloud);
  const auto t_align = std::chrono::steady_clock::now();
  registration->align(*aligned, init_guess);
  const auto t_align_end = std::chrono::steady_clock::now();
  // [2026-08-17] getFitnessScore 单独计时。它是 PCL 的模板，在本包内实例化（本包
  // 当前是 -O0），且内部对**每个源点**做一次目标 KD 树最近邻搜索 —— 4000 点 ×
  // 15404 点的树。它与 align() 是完全不同的开销，必须分开看。
  double ndt_score = registration->getFitnessScore();
  tracking_diag_.ndt_setsrc_ms =
    std::chrono::duration<double, std::milli>(t_align - t_setsrc).count();
  tracking_diag_.ndt_align_ms =
    std::chrono::duration<double, std::milli>(t_align_end - t_align).count();
  tracking_diag_.ndt_score_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t_align_end).count();

  if (ndt_score > 0.5) {
    RCLCPP_INFO(rclcpp::get_logger("PoseEstimator"), "ndt_score > 0.5, ndt_score: %f", ndt_score);
  }

  Eigen::Matrix4f trans = registration->getFinalTransformation();
  tracking_diag_.ndt_x = trans(0, 3);
  tracking_diag_.ndt_y = trans(1, 3);
  tracking_diag_.ndt_z = trans(2, 3);

  match_result_.is_converged_ = registration->hasConverged();
  match_result_.fitness_score_ = ndt_score;

  // NDT result sanity check: reject NaN/Inf, unreasonably large single-frame
  // displacement (>2m, a robot dog at 0.5 m/s cannot move >0.05 m between 100 ms
  // LiDAR frames), or bad fitness score.
  // NOTE: this checks the jump from the fused prediction, NOT the absolute distance
  // from map origin (that would reject all frames on any map larger than 5 m radius).
  Eigen::Vector3f p_ndt = trans.block<3, 1>(0, 3);
  float frame_jump = (p_ndt - init_guess.block<3, 1>(0, 3)).norm();
  if (!trans.allFinite() || frame_jump > 2.0f || ndt_score > ndt_score_threshold_) {
    const bool bad_pos = !trans.allFinite() || frame_jump > 2.0f;
    const bool bad_score = ndt_score > ndt_score_threshold_;
    RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
      "NDT result rejected: allFinite=%d, frame_jump=%.2f%s, ndt_score=%.2f%s",
      trans.allFinite(), frame_jump,
      bad_pos ? " (>2m)" : "",
      ndt_score,
      bad_score ? " (>score threshold)" : "");
    // Abnormal velocity protection: if UKF velocity is already corrupted during
    // consecutive rejections, reset it to break predict→drift→rejection loops.
    // This used to only fire when pos_norm > 50 m (too late to matter).
    const float vnorm = ukf->mean.middleRows<3>(3).norm();
    const float vznorm = std::abs(ukf->mean[5]);
    const bool pos_corrupt = ukf->mean.head<3>().norm() > 50.0f;
    if (pos_corrupt || vnorm > 1.5f || vznorm > 0.4f) {
      reset_velocity();
      RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
        "Velocity reset on base rejection: |v|=%.2f vz=%.2f%s",
        vnorm, vznorm, pos_corrupt ? " (pos also abnormal)" : "");
    }
    // Mark as rejected so upper-layer state machine sees this frame was dropped
    match_result_.is_converged_ = false;
    // Skip UKF correction for this frame
    return aligned;
  }

  Eigen::Vector3f p = trans.block<3, 1>(0, 3);
  Eigen::Quaternionf q(trans.block<3, 3>(0, 0));
  q.normalize();

  if(quat().coeffs().dot(q.coeffs()) < 0.0f) {
    q.coeffs() *= -1.0f;
  }

  auto roll_pitch = [](const Eigen::Quaternionf& quat) -> Eigen::Vector2f {
    float roll  = std::atan2(2.0f * (quat.w() * quat.x() + quat.y() * quat.z()),
                             1.0f - 2.0f * (quat.x() * quat.x() + quat.y() * quat.y()));
    float pitch = std::asin(std::clamp(
      2.0f * (quat.w() * quat.y() - quat.z() * quat.x()), -1.0f, 1.0f));
    return {roll, pitch};
  };
  const Eigen::Vector2f obs_rp = roll_pitch(q);

  // Innovation gate: reject NDT observations whose correction jump from the
  // UKF prediction violates ground-robot physics. Low-score false matches in
  // large maps can inject large single-frame corrections that pollute velocity.
  //
  // Bootstrap warmup: first N frames after Init/Recovery use relaxed thresholds
  // so the UKF can converge from the GL initial guess before hard gates engage.
  bool in_warmup = (post_init_correction_count_ > 0);
  float trust_gate_xy_lim = max_ndt_correction_xy_;
  tracking_diag_.trust_valid = trust_anchor_valid_ && trust_current_odom_valid_;
  tracking_diag_.trust_deviation_xy = 0.0f;
  tracking_diag_.trust_allowed_xy = 0.0f;
  tracking_diag_.trust_gate_xy = trust_gate_xy_lim;
  tracking_diag_.trust_travel_m = trust_travel_m_;
  tracking_diag_.trust_exceed_count = trust_exceed_count_;
  if (tracking_diag_.trust_valid) {
    const Eigen::Matrix4f odom_delta_from_anchor =
      trust_anchor_odom_pose_.inverse() * trust_current_odom_pose_;
    const Eigen::Matrix4f expected_map_pose =
      trust_anchor_map_pose_ * odom_delta_from_anchor;
    const float deviation =
      (p.head<2>() - expected_map_pose.block<2, 1>(0, 3)).norm();
    const float allowed =
      kTrustBaseAllowanceM + kTrustAllowancePerTravelM * trust_travel_m_;
    tracking_diag_.trust_deviation_xy = deviation;
    tracking_diag_.trust_allowed_xy = allowed;

    if (deviation > allowed) {
      ++trust_exceed_count_;
      const float excess = deviation - allowed;
      trust_gate_xy_lim = std::max(
        kTrustMinCorrectionXY,
        max_ndt_correction_xy_ / (1.0f + excess / kTrustTighteningScaleM));
      if (trust_exceed_count_ == 1 || trust_exceed_count_ % 10 == 0) {
        RCLCPP_WARN(
          logger_,
          "[TRUST] candidate outside GL/odom trust region: deviation=%.3fm "
          "allowed=%.3fm travel=%.3fm xy_gate=%.3fm consecutive=%d",
          deviation, allowed, trust_travel_m_, trust_gate_xy_lim,
          trust_exceed_count_);
      }
    } else {
      trust_exceed_count_ = 0;
    }
    tracking_diag_.trust_gate_xy = trust_gate_xy_lim;
    tracking_diag_.trust_exceed_count = trust_exceed_count_;
  }
  {
    Eigen::Vector3f pred_p = init_guess.block<3, 1>(0, 3);
    Eigen::Quaternionf pred_q(init_guess.block<3, 3>(0, 0));
    pred_q.normalize();

    float dz  = std::abs(p.z() - pred_p.z());
    float dxy = (p.head<2>() - pred_p.head<2>()).norm();

    Eigen::Vector2f pred_rp = roll_pitch(pred_q);
    float droll  = std::abs(obs_rp.x() - pred_rp.x()) * 180.0f / M_PI;
    float dpitch = std::abs(obs_rp.y() - pred_rp.y()) * 180.0f / M_PI;

    // Warmup only relaxes Z/attitude convergence. XY remains tied to robot
    // motion, and may be tightened further by the GL/odom trust region.
    float gate_z_lim     = in_warmup ? kWarmupCorrectionZ     : max_ndt_correction_z_;
    float gate_xy_lim    = std::min(max_ndt_correction_xy_, trust_gate_xy_lim);
    float gate_angle_lim = in_warmup ? kWarmupCorrectionAngle : max_ndt_correction_angle_;

    bool gate_z     = dz     > gate_z_lim;
    bool gate_xy    = dxy    > gate_xy_lim;
    bool gate_roll  = droll  > gate_angle_lim;
    bool gate_pitch = dpitch > gate_angle_lim;

    // [2026-08-17] 只观察不改：warmup 期角度门放宽到 30°，是落位后姿态被 NDT
    // 一帧拉走 6° 的允许条件之一。播种重力锚定后如果姿态仍被拉走，这条日志会
    // 显示"NDT 想改多少 / 门限允许多少"，据此再决定要不要收紧 30° 这个值。
    // 先不动它，避免与播种改动同时生效导致无法归因。
    if (in_warmup) {
      RCLCPP_INFO(logger_,
        "[WARMUP GATE] remaining=%d droll=%.1f° dpitch=%.1f° limit=%.0f° "
        "dxy=%.3f/%.3f dz=%.3f/%.3f | grav_applied=%d",
        post_init_correction_count_, droll, dpitch, gate_angle_lim,
        dxy, gate_xy_lim, dz, gate_z_lim,
        tracking_diag_.gravity_applied ? 1 : 0);
    }

    if (gate_z || gate_xy || gate_roll || gate_pitch) {
      RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
        "NDT result rejected by innovation gate: "
        "dz=%.3fm%s dxy=%.3fm%s droll=%.1f°%s dpitch=%.1f°%s "
        "obs=[%.2f,%.2f,%.2f | %.2f,%.2f,%.2f,%.2f] "
        "pred=[%.2f,%.2f,%.2f | %.2f,%.2f,%.2f,%.2f] ndt_score=%.4f",
        dz,     gate_z     ? " (>Z)"     : "",
        dxy,    gate_xy    ? " (>XY)"    : "",
        droll,  gate_roll  ? " (>roll)"  : "",
        dpitch, gate_pitch ? " (>pitch)" : "",
        p.x(), p.y(), p.z(), q.w(), q.x(), q.y(), q.z(),
        pred_p.x(), pred_p.y(), pred_p.z(),
        pred_q.w(), pred_q.x(), pred_q.y(), pred_q.z(),
        ndt_score);

      // Mark as rejected so upper-layer state machine sees this frame was dropped
      match_result_.is_converged_ = false;

      // If velocity is already corrupted, reset it to break the predict→drift→gate loop
      const float vnorm = ukf->mean.middleRows<3>(3).norm();
      const float vznorm = std::abs(ukf->mean[5]);
      if (vnorm > 1.5f || vznorm > 0.4f) {
        reset_velocity();
        RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
          "Velocity reset on innovation gate: |v|=%.2f vz=%.2f", vnorm, vznorm);
      }

      return aligned;
    }
  }

  // Absolute geometry check, observe-only for the first calibration run. Query
  // the NDT target KD tree so this always uses the same PCD as scan matching.
  // Centering the vertical band on the captured body/reference height excludes
  // floor and ceiling returns. Without a reference, the frame is explicitly
  // allowed and reported as not ready.
  tracking_diag_.occupancy_ready = false;
  tracking_diag_.occupancy_blocked = false;
  tracking_diag_.occupancy_points = 0;
  if (has_reference_z_) {
    const auto target = registration->getInputTarget();
    const auto search = registration->getSearchMethodTarget();
    if (target && !target->empty() && search) {
      PointT query;
      query.x = p.x();
      query.y = p.y();
      query.z = reference_z_;
      query.intensity = 0.0f;

      std::vector<int> indices;
      std::vector<float> squared_distances;
      const float search_radius = std::hypot(kOccupancyRadiusXY, kOccupancyHalfHeight);
      search->radiusSearch(query, search_radius, indices, squared_distances);

      int points_in_body_band = 0;
      for (const int index : indices) {
        if (index < 0 || static_cast<size_t>(index) >= target->size()) {
          continue;
        }
        const PointT& map_point = target->at(static_cast<size_t>(index));
        const float dx = map_point.x - p.x();
        const float dy = map_point.y - p.y();
        if (std::hypot(dx, dy) <= kOccupancyRadiusXY &&
            std::abs(map_point.z - reference_z_) <= kOccupancyHalfHeight) {
          ++points_in_body_band;
        }
      }

      tracking_diag_.occupancy_ready = true;
      tracking_diag_.occupancy_points = points_in_body_band;
      tracking_diag_.occupancy_blocked =
        points_in_body_band >= kOccupancyPointThreshold;
      RCLCPP_INFO(
        logger_,
        "[OCCUPANCY] mode=observe ready=1 candidate_xy=[%.3f,%.3f] ref_z=%.3f "
        "radius_xy=%.2f half_height=%.2f points=%d threshold=%d would_reject=%d",
        p.x(), p.y(), reference_z_, kOccupancyRadiusXY, kOccupancyHalfHeight,
        points_in_body_band, kOccupancyPointThreshold,
        tracking_diag_.occupancy_blocked ? 1 : 0);

      if (kOccupancyRejectEnabled && tracking_diag_.occupancy_blocked) {
        match_result_.is_converged_ = false;
        return aligned;
      }
    }
  }

  // Cumulative drift constraint: after warmup, reject slow coherent drift that
  // remains small relative to the UKF prediction but violates the accepted
  // ground-height and attitude reference.
  //
  // [2026-08-16 第一版] 四足步态行走时真实俯仰/横滚摆动可达 20-25°（实测起步/
  // 转弯 pitch 25.6° 被误拒），因此运动时跳过本约束。
  // [2026-08-16 修正] 上一版把 **Z 检查也一起跳过了**，这是个回归：四足步态的
  // 垂直起伏只有 ±5cm，0.5m 的 Z 门限有 10 倍余量，本就不需要为运动放行。结果
  // 是运动中 Z 完全失去约束——实测 z 漂到 1.05m、pitch 发散到 -47°，而 NDT
  // 在室内平面环境对 z/pitch 不敏感，ndt_score 仍是 0.21 的"好分"，无人拦得住。
  // 现在拆开：Z 始终检查，roll/pitch 仅在近似静止时检查。
  // 0.03m/帧 ≈ 0.3m/s：静止帧间位移 <0.01m，行走 0.3-0.8m/s 对应
  // 0.03-0.08m/帧，阈值两侧余量充足。
  constexpr float kMotionFrameJumpForDriftSkip = 0.03f;
  if (!in_warmup && has_reference_z_) {
    const bool nearly_still = frame_jump < kMotionFrameJumpForDriftSkip;

    const float z_drift = std::abs(p.z() - reference_z_);
    const float roll_drift = std::abs(std::atan2(
      std::sin(obs_rp.x() - reference_roll_),
      std::cos(obs_rp.x() - reference_roll_))) * 180.0f / M_PI;
    const float pitch_drift = std::abs(std::atan2(
      std::sin(obs_rp.y() - reference_pitch_),
      std::cos(obs_rp.y() - reference_pitch_))) * 180.0f / M_PI;

    // Z：静止与运动都检查（步态起伏远小于门限）
    const bool bad_z = z_drift > kMaxZDriftFromReference;
    // 姿态：只在近似静止时检查，避免把步态摆动当漂移
    const bool bad_roll  = nearly_still && roll_drift  > kMaxRollDriftDeg;
    const bool bad_pitch = nearly_still && pitch_drift > kMaxPitchDriftDeg;

    if (bad_z || bad_roll || bad_pitch) {
      RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
        "NDT result rejected by drift constraint: "
        "z_drift=%.3fm%s roll_drift=%.1f°%s pitch_drift=%.1f°%s "
        "ref=[z=%.3f roll=%.1f° pitch=%.1f°] still=%d ndt_score=%.4f",
        z_drift, bad_z ? " (>Z)" : "",
        roll_drift, bad_roll ? " (>roll)" : "",
        pitch_drift, bad_pitch ? " (>pitch)" : "",
        reference_z_, reference_roll_ * 180.0f / M_PI,
        reference_pitch_ * 180.0f / M_PI, nearly_still ? 1 : 0, ndt_score);
      match_result_.is_converged_ = false;
      const float vnorm_drift = ukf->mean.middleRows<3>(3).norm();
      const float vznorm_drift = std::abs(ukf->mean[5]);
      if (vnorm_drift > 1.5f || vznorm_drift > 0.4f) {
        reset_velocity();
        RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
          "Velocity reset on drift constraint: |v|=%.2f vz=%.2f",
          vnorm_drift, vznorm_drift);
      }
      return aligned;
    }
  }

  // 观测的 roll/pitch 改由重力决定，只保留 NDT 的位置与 yaw。
  //
  // [2026-08-16] 依据：室内平面地图里 NDT 对 roll/pitch **没有信息量**——倾斜
  // 40° 墙面照样对得上。让它参与这两个自由度的表决本身就是错的。实测落位后
  // 不到 1 秒，NDT 就把姿态从"重力推导的正确初值"拖走 10.5°、把 z 拉低 0.23m，
  // 而 ndt_score 全程正常、所有门控零拒帧。
  //
  // 之前只在 predict 侧做小增益锚定，等于"重力小增益拉、NDT 大权重推"的拔河，
  // NDT 赢（实测误差稳在 10.2° 不收敛）。这里直接改写观测，从结构上让 NDT
  // 不可能再拖动姿态：对齐两向量的最小旋转不含 yaw 分量，航向仍完全由 NDT 定。
  //
  // 只在重力样本新鲜时替换；样本过期（高动态持续、IMU 断流）则退回纯 NDT 姿态，
  // 保持改动前行为。
  Eigen::Quaternionf q_obs = q;
  const bool use_gravity_attitude = gravityObservationFresh(stamp);

  // 归因诊断：NDT 结果相对**预测**改了多少。这是区分故障源的唯一手段——
  //   innov_dz 持续为负 → NDT 在把 z 往下拽（平面环境配准退化）
  //   innov_dz ≈ 0 但 z 仍下沉 → 预测自己在漂（IMU/odom 侧）
  // yaw 同理。只看最终位姿无法归因，必须看这个分解。
  tracking_diag_.innov_dxy =
    (p.head<2>() - init_guess.block<3, 1>(0, 3).head<2>()).norm();
  tracking_diag_.innov_dz = p.z() - init_guess(2, 3);
  tracking_diag_.innov_dyaw_deg =
    yawDeltaDeg(init_guess.block<3, 3>(0, 0), trans.block<3, 3>(0, 0));
  tracking_diag_.gravity_applied = use_gravity_attitude;
  tracking_diag_.gravity_err_deg =
    use_gravity_attitude ? gravityTiltErrorDeg(q, gravity_body_, g_map_dir_) : 0.0f;

  // 预测来源分解：三个 z 放一起，一眼看出是谁在往下走
  tracking_diag_.odom_pred_valid  = static_cast<bool>(odom_prediction_);
  tracking_diag_.pred_imu_z       = imu_guess(2, 3);
  tracking_diag_.pred_odom_z      = odom_guess(2, 3);
  tracking_diag_.pred_used_z      = init_guess(2, 3);
  tracking_diag_.pred_odom_dz     = odom_guess(2, 3) - last_observation(2, 3);
  tracking_diag_.pred_imu_yaw_deg =
    rotationToRpy(imu_guess.block<3, 3>(0, 0)).z() * 180.0f / static_cast<float>(M_PI);
  tracking_diag_.pred_odom_yaw_deg =
    rotationToRpy(odom_guess.block<3, 3>(0, 0)).z() * 180.0f / static_cast<float>(M_PI);

  // UKF 健康度：用实测替代推算，验证流形近似到底有多严重
  {
    const Eigen::Vector4f qm = ukf->mean.middleRows<4>(6);
    tracking_diag_.ukf_quat_norm  = qm.norm();
    tracking_diag_.ukf_quat_cov_tr = ukf->cov.block<4, 4>(6, 6).trace();
    tracking_diag_.ukf_llt_failures = ukf->llt_failure_count;
    tracking_diag_.ukf_predicts = ukf->predict_count;
    // [2026-08-16] 原来测的是"sigma 点四元数偏离单位模的最大值"，但 f() 已经把
    // 每个 sigma 点归一化过，该值按构造恒为 0，测不到任何东西。改测真正有意义的
    // 量：sigma 点与均值的**最大夹角**——它直接反映无迹变换是否已超出局部有效域，
    // 以及是否有点跨到对映半球。
    float max_ang_deg = 0.0f;
    if (ukf->predict_count > 0 && qm.norm() > 1e-6f) {
      const Eigen::Vector4f qn = qm.normalized();
      for (int i = 0; i < ukf->sigma_points.rows(); ++i) {
        const Eigen::Vector4f qs = ukf->sigma_points.row(i).middleCols<4>(6).transpose();
        const float n = qs.norm();
        if (n < 1e-6f) {
          continue;
        }
        const float c = std::clamp(std::abs(qs.dot(qn) / n), -1.0f, 1.0f);
        max_ang_deg = std::max(
          max_ang_deg, 2.0f * std::acos(c) * 180.0f / static_cast<float>(M_PI));
      }
    }
    tracking_diag_.ukf_sigma_quat_dev = max_ang_deg;
  }

  if (use_gravity_attitude) {
    q_obs = gravityLeveledOrientation(q, gravity_body_, g_map_dir_);
    if (tracking_diag_.gravity_err_deg > kGravityAnchorWarnDeg &&
        (gravity_substitution_log_++ % 100) == 0) {
      RCLCPP_WARN(logger_,
        "NDT attitude overridden by gravity: NDT was %.1f° off the map gravity "
        "reference (position/yaw kept). 若该值持续偏大，多半是 NDT 在平面环境"
        "退化，或 map_gravity_direction 配置与地图不符。",
        tracking_diag_.gravity_err_deg);
    }
  }

  // Observe the complete attitude hand-off in one frame.  This is diagnostic
  // only: pred is the registration initial guess, raw_ndt is the optimizer
  // result, and gravity_obs is the quaternion actually sent to the UKF.
  constexpr float kRadToDeg = 180.0f / static_cast<float>(M_PI);
  const Eigen::Vector3f pred_rpy_diag =
    rotationToRpy(init_guess.block<3, 3>(0, 0)) * kRadToDeg;
  const Eigen::Vector3f ndt_rpy_diag =
    rotationToRpy(q.toRotationMatrix()) * kRadToDeg;
  const Eigen::Vector3f gravity_obs_rpy_diag =
    rotationToRpy(q_obs.toRotationMatrix()) * kRadToDeg;
  tracking_diag_.pred_roll_deg = pred_rpy_diag.x();
  tracking_diag_.pred_pitch_deg = pred_rpy_diag.y();
  tracking_diag_.pred_yaw_deg = pred_rpy_diag.z();
  tracking_diag_.ndt_roll_deg = ndt_rpy_diag.x();
  tracking_diag_.ndt_pitch_deg = ndt_rpy_diag.y();
  tracking_diag_.ndt_yaw_deg = ndt_rpy_diag.z();
  tracking_diag_.gravity_obs_roll_deg = gravity_obs_rpy_diag.x();
  tracking_diag_.gravity_obs_pitch_deg = gravity_obs_rpy_diag.y();
  tracking_diag_.gravity_obs_yaw_deg = gravity_obs_rpy_diag.z();
  tracking_diag_.attitude_chain_valid = true;

  Eigen::VectorXf observation(7);
  observation.middleRows(0, 3) = p;
  observation.middleRows(3, 4) =
    Eigen::Vector4f(q_obs.w(), q_obs.x(), q_obs.y(), q_obs.z());

  const float motion_scale_before = motion_scale_;
  const bool has_motion_window_before = has_motion_window_;
  const Eigen::Vector3f motion_window_origin_before = motion_window_origin_;
  const rclcpp::Time motion_window_stamp_before = motion_window_stamp_;
  const Eigen::MatrixXf measurement_noise_before = ukf->getMeasurementNoiseCov();

  // R13/R13.1: motion from NDT map-frame displacement (not UKF |v|).
  // 72809: per-frame NDT jitter 2–5cm @10Hz → 0.2–0.5 m/s false "walk".
  // Deadzone + 1s net-displacement window so still actually enters still mode.
  // R13.3: XY-only displacement so Z drift is NOT misinterpreted as robot motion
  // (a ground robot cannot generate sustained vertical speed).
  float ndt_speed = 0.0f;
  if (has_last_ndt_) {
    const double dt_ndt = (stamp - last_ndt_stamp_).seconds();
    if (dt_ndt > 0.02 && dt_ndt < 1.0) {
      ndt_speed = (p.head<2>() - last_ndt_pos_.head<2>()).norm() / static_cast<float>(dt_ndt);
    }
  }

  // Per-frame speed deadzone: ignore jitter below 0.20 m/s; full scale at 0.40 m/s
  constexpr float kSpeedDeadzone = 0.20f;
  constexpr float kSpeedFull = 0.40f;
  float instant_motion = 0.0f;
  if (ndt_speed > kSpeedDeadzone) {
    instant_motion = std::min(
      1.0f, (ndt_speed - kSpeedDeadzone) / (kSpeedFull - kSpeedDeadzone));
  }

  // ~1s window net displacement (still if <8cm/s-equivalent)
  constexpr float kWindowDeadzoneM = 0.08f;  // net meters over the window → still
  constexpr float kWindowFullM = 0.30f;      // ~0.3 m/s sustained
  if (!has_motion_window_) {
    motion_window_origin_ = p;
    motion_window_stamp_ = stamp;
    has_motion_window_ = true;
  } else {
    const double wdt = (stamp - motion_window_stamp_).seconds();
    const float net = (p.head<2>() - motion_window_origin_.head<2>()).norm();
    if (wdt >= 0.50 && net < 0.05f) {
      // Early evidence of still inside the window — kill frame-speed false positives
      instant_motion = 0.0f;
    }
    if (wdt >= 1.0) {
      float window_motion = 0.0f;
      if (net > kWindowDeadzoneM) {
        window_motion = std::min(
          1.0f, (net - kWindowDeadzoneM) / (kWindowFullM - kWindowDeadzoneM));
      }
      // Window dominates: sustained displacement is ground truth for walk vs still
      motion_scale_ = 0.4f * motion_scale_ + 0.6f * window_motion;
      motion_window_origin_ = p;
      motion_window_stamp_ = stamp;
    }
  }

  // Frame update with asymmetric EMA: decay to still faster than rise to move
  if (instant_motion < 0.05f) {
    motion_scale_ = 0.55f * motion_scale_;
  } else {
    motion_scale_ = 0.7f * motion_scale_ + 0.3f * instant_motion;
  }
  if (motion_scale_ < 0.02f) {
    motion_scale_ = 0.0f;
  }

  Eigen::MatrixXf R = Eigen::MatrixXf::Identity(7, 7);
  // Blend position/quat R: still keeps 0.1/0.005; moving → 0.02/0.001
  const float r_pos = 0.1f * (1.0f - motion_scale_) + 0.02f * motion_scale_;
  const float r_quat = 0.005f * (1.0f - motion_scale_) + 0.001f * motion_scale_;
  R.middleRows(0, 3) *= r_pos;
  R.middleRows(3, 4) *= r_quat;
  ukf->setMeasurementNoiseCov(R);
  tracking_diag_.ukf_measurement_pos_variance = r_pos;
  tracking_diag_.ukf_motion_scale = motion_scale_;

  // Round 9: Mahalanobis distance gate — reject NDT results that are statistically
  // inconsistent with the current UKF belief, even if they pass basic sanity checks.
  // This catches cases where NDT converges to a wrong local minimum that happens to
  // have acceptable score and position (a false-positive match).
  //
  // We use a 3-DOF position-only Mahalanobis distance to avoid quaternion
  // normalization ambiguity. Chi-squared 95% threshold for 3 DOF ≈ 7.81.
  // Use 20.0 as a conservative threshold to only reject obvious outliers.
  {
    Eigen::Vector3f pos_pred = ukf->mean.head<3>();
    Eigen::Vector3f pos_obs  = p;
    Eigen::Vector3f innovation_3d = pos_obs - pos_pred;

    Eigen::Matrix3f P_pos = ukf->getCov().block<3, 3>(0, 0);
    Eigen::Matrix3f R_pos = ukf->getMeasurementNoiseCov().block<3, 3>(0, 0);
    Eigen::Matrix3f S_pos = P_pos + R_pos;

    float mahalanobis_sq = innovation_3d.transpose() * S_pos.ldlt().solve(innovation_3d);
    tracking_diag_.ukf_mahalanobis_sq = mahalanobis_sq;

    // Skip Mahalanobis gate during bootstrap warmup — same obs-pred consistency
    // check that should be relaxed while UKF converges from GL initial guess.
    if (!in_warmup && mahalanobis_sq > 20.0f) {
      RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
        "NDT result rejected by Mahalanobis gate: distance²=%.1f (>20 threshold), "
        "obs=[%.2f,%.2f,%.2f] pred=[%.2f,%.2f,%.2f] ndt_score=%.4f",
        mahalanobis_sq, p.x(), p.y(), p.z(),
        ukf->mean[0], ukf->mean[1], ukf->mean[2], ndt_score);
      match_result_.is_converged_ = false;
      motion_scale_ = motion_scale_before;
      has_motion_window_ = has_motion_window_before;
      motion_window_origin_ = motion_window_origin_before;
      motion_window_stamp_ = motion_window_stamp_before;
      ukf->setMeasurementNoiseCov(measurement_noise_before);
      const float vnorm_m = ukf->mean.middleRows<3>(3).norm();
      const float vznorm_m = std::abs(ukf->mean[5]);
      if (vnorm_m > 1.5f || vznorm_m > 0.4f) {
        reset_velocity();
        RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
          "Velocity reset on Mahalanobis gate: |v|=%.2f vz=%.2f", vnorm_m, vznorm_m);
      }
      return aligned;
    }
  }

  // wo_pred_error = no_guess.inverse() * registration->getFinalTransformation();

  // Save the full UKF belief for rollback in case the NDT observation
  // produces an unreasonably high velocity (observed: |v|=4.8 m/s on an
  // accept=1 frame that passed all pre-correction gates).
  Eigen::VectorXf mean_before_correct = ukf->mean;
  Eigen::MatrixXf cov_before_correct = ukf->getCov();

  ukf->correct(observation);
  tracking_diag_.ukf_correct_applied = true;
  tracking_diag_.ukf_post_x = ukf->mean[0];
  tracking_diag_.ukf_post_y = ukf->mean[1];
  tracking_diag_.ukf_post_z = ukf->mean[2];
  tracking_diag_.ukf_post_vx = ukf->mean[3];
  tracking_diag_.ukf_post_vy = ukf->mean[4];
  tracking_diag_.ukf_post_vz = ukf->mean[5];
  // imu_pred_error = imu_guess.inverse() * registration->getFinalTransformation();

  // R12/R13/R13.2: adaptive velocity damping.
  // still ≈0.85 (kill phantom |v|); moving ≈0.98 (preserve walk speed).
  float damp = 0.85f * (1.0f - motion_scale_) + 0.98f * motion_scale_;
  const float vnorm = ukf->mean.middleRows<3>(3).norm();
  // Tier 1: moderate ghost — catch velocity buildup early before it reaches 0.8
  if (motion_scale_ < 0.30f && vnorm > 0.5f) {
    damp = std::min(damp, 0.65f);
  }
  // Tier 2: hard ghost — NDT says still but UKF |v| is already large
  if (motion_scale_ < 0.25f && vnorm > 0.8f) {
    damp = 0.55f;  // aggressive kill (was 0.70)
  }
  ukf->mean.middleRows<3>(3) *= damp;
  tracking_diag_.ukf_post_vx = ukf->mean[3];
  tracking_diag_.ukf_post_vy = ukf->mean[4];
  tracking_diag_.ukf_post_vz = ukf->mean[5];

  // [2026-08-31 走廊漂移修复] SDK 速度一致性门。走廊内 NDT 沿轴向不可
  // 观测,UKF 会把滑移学成速度自我强化(实测静止 0.35m/s 滑移、行走超速
  // 60%)。SDK twist 提供该自由度的真值:
  //   SDK 静止(|v|<0.10) → UKF 速度直接归零(位姿钉死);
  //   SDK 运动 → UKF 机体速度 XY 钳制到 SDK±0.35(钳掉走廊滑移增量)。
  // SDK 速度跳变(>3 m/s,超出狗体物理)时丢弃该样本不使用。
  if (robot_odom_velocity_valid_) {
    const float sdk_vnorm = robot_odom_velocity_body_.head<2>().norm();
    if (sdk_vnorm < 3.0f) {  // SDK twist 物理合理性(狗最大 ~1.5m/s)
      if (sdk_vnorm < 0.10f) {
        if (ukf->mean.middleRows<3>(3).norm() > 0.05f) {
          ++sdk_gate_still_count_;
          if (sdk_gate_still_count_ == 1 || sdk_gate_still_count_ % 50 == 0) {
            RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
              "[SDK GATE] still: UKF velocity zeroed (was %.2f m/s, "
              "consecutive=%u, age=%.3fs)",
              ukf->mean.middleRows<3>(3).norm(), sdk_gate_still_count_,
              robot_odom_velocity_age_);
          }
        } else {
          sdk_gate_still_count_ = 0;
        }
        ukf->mean.middleRows<3>(3).setZero();
      } else {
        sdk_gate_still_count_ = 0;
        // UKF 速度转到机体坐标系与 SDK 对比(只用 XY,Z 由地面约束)。
        const Eigen::Matrix3f R_wb(
          Eigen::Quaternionf(
            ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9])
            .normalized());
        const Eigen::Vector3f v_map = ukf->mean.middleRows<3>(3);
        const Eigen::Vector3f v_body = R_wb.transpose() * v_map;
        const Eigen::Vector2f sdk_v = robot_odom_velocity_body_.head<2>();
        Eigen::Vector2f delta = v_body.head<2>() - sdk_v;
        const float delta_norm = delta.norm();
        if (delta_norm > 0.35f) {
          ++sdk_gate_clamp_count_;
          delta *= 0.35f / delta_norm;
          const Eigen::Vector2f clamped_body = sdk_v + delta;
          const Eigen::Vector3f clamped_map =
            R_wb * Eigen::Vector3f(
              clamped_body.x(), clamped_body.y(), v_body.z());
          ukf->mean.middleRows<3>(3) = clamped_map;
          if (sdk_gate_clamp_count_ == 1 || sdk_gate_clamp_count_ % 50 == 0) {
            RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
              "[SDK GATE] clamp: UKF %.2f m/s vs SDK %.2f m/s, "
              "clamped to SDK+0.35 (count=%u)",
              v_body.head<2>().norm(), sdk_vnorm, sdk_gate_clamp_count_);
          }
        }
      }
      tracking_diag_.ukf_post_vx = ukf->mean[3];
      tracking_diag_.ukf_post_vy = ukf->mean[4];
      tracking_diag_.ukf_post_vz = ukf->mean[5];
    } else if (sdk_vnorm < 10.0f) {
      // 一次性的跳变样本(如 SDK odom 3m 级位置跳变伴随的速度尖峰):
      // 不使用,不累加,仅提示。跳变本身罕见,无需节流。
      RCLCPP_WARN(
        rclcpp::get_logger("PoseEstimator"),
        "[SDK GATE] SDK twist jump %.2f m/s ignored", sdk_vnorm);
    }
  }

  // Post-correction velocity health check: a correct NDT observation
  // cannot produce |v| > 1.5 m/s or |vz| > 0.4 m/s on a ground robot.
  // If it does, the observation was likely a false match — rollback.
  {
    const float vnorm_after = ukf->mean.middleRows<3>(3).norm();
    const float vznorm_after = std::abs(ukf->mean[5]);
    if (vnorm_after > 1.5f || vznorm_after > 0.4f) {
      ukf->mean = mean_before_correct;
      ukf->setCov(cov_before_correct);
      ukf->setMeasurementNoiseCov(measurement_noise_before);
      reset_velocity();
      tracking_diag_.ukf_correct_applied = false;
      tracking_diag_.ukf_post_x = ukf->mean[0];
      tracking_diag_.ukf_post_y = ukf->mean[1];
      tracking_diag_.ukf_post_z = ukf->mean[2];
      tracking_diag_.ukf_post_vx = ukf->mean[3];
      tracking_diag_.ukf_post_vy = ukf->mean[4];
      tracking_diag_.ukf_post_vz = ukf->mean[5];
      RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
        "Post-correction velocity unhealthy: |v|=%.2f vz=%.2f — correction rolled back, velocity reset",
        vnorm_after, vznorm_after);
      match_result_.is_converged_ = false;
      return aligned;
    }
  }

  // 采集漂移参考的候选帧。
  //
  // [2026-08-16 v1] 原环形移位保留的是 warmup 的**最后 5 帧**——被 NDT 拖得最歪
  //   的那几帧。改为只保留最早的 5 帧。
  // [2026-08-16 v2] 仍然不够：落位后若立刻开始导航，warmup 全程狗在动，姿态中位
  //   roll 达 12.1° 超过按**静止**标定的 10° 上限 → 参考被拒 → 整轮 Z 保护禁用
  //   （实测该轮 dz_ref 全程 n/a、z 掉 0.64m 无人拦截）。
  //   两处修正：
  //     a) 只在**近似静止**时采样 —— 动着的姿态不能当静止基准；
  //     b) 采不到就**继续等**，不再永久禁用 —— 狗迟早会停，那时再建立参考。
  {
    const bool still_enough = (frame_jump < kRefCaptureMaxFrameJump) &&
                              (motion_scale_ < kRefCaptureMaxMotionScale);
    if (!has_reference_z_ && still_enough &&
        ref_candidate_count_ < kRefCandidateBufSize) {
      const int idx = ref_candidate_count_;
      ref_z_buf_[idx]     = p.z();
      ref_roll_buf_[idx]  = obs_rp.x();
      ref_pitch_buf_[idx] = obs_rp.y();
      ref_candidate_count_ = idx + 1;
    }
  }

  // Only a correction that passed every pre- and post-correction check may
  // consume warmup or establish the cumulative-drift reference.
  if (post_init_correction_count_ > 0) {
    post_init_correction_count_--;
    ++warmup_segment_consumed_;
    ++warmup_total_consumed_;
    if (post_init_correction_count_ == 0) {
      RCLCPP_INFO(
        logger_,
        "[WARMUP] completed segment_frames=%d total_frames=%d restarts=%d",
        warmup_segment_consumed_, warmup_total_consumed_, warmup_restart_count_);
    }
  }

  // 攒够静止候选就建立参考（不再绑定 warmup 结束这一个时刻）
  if (!has_reference_z_ &&
      ref_candidate_count_ >= kRefCandidateBufSize) {
    const int n = ref_candidate_count_;
    {
      // Median over the collected candidates (robust to single outliers)
      auto median = [](float* buf, int count) -> float {
        std::sort(buf, buf + count);
        return buf[count / 2];
      };
      const float med_z     = median(ref_z_buf_, n);
      const float med_roll  = median(ref_roll_buf_, n);
      const float med_pitch = median(ref_pitch_buf_, n);

      const float abs_roll_deg  = std::abs(med_roll)  * 180.0f / M_PI;
      const float abs_pitch_deg = std::abs(med_pitch) * 180.0f / M_PI;

      if (abs_roll_deg <= kMaxRefAbsRollDeg && abs_pitch_deg <= kMaxRefAbsPitchDeg) {
        reference_z_     = med_z;
        reference_roll_  = med_roll;
        reference_pitch_ = med_pitch;
        has_reference_z_ = true;
        RCLCPP_INFO(logger_,
          "Ground reference captured (median of %d still frames): "
          "Z=%.3f roll=%.1f° pitch=%.1f°",
          n, reference_z_, abs_roll_deg, abs_pitch_deg);
      } else {
        // 姿态超界：丢弃这批候选重新采，而**不是**永久禁用漂移门控。
        // 上一版在这里置 drift_gate_disabled_for_session_，导致落位后立刻导航时
        // Z 保护整轮失效（实测 z 掉 0.64m 无人拦截）。现在只是重来。
        ref_candidate_count_ = 0;
        RCLCPP_WARN(logger_,
          "Ground reference candidates discarded: attitude out of bounds "
          "(roll=%.1f° > %.0f° | pitch=%.1f° > %.0f°). 重新采集，门控暂不启用。",
          abs_roll_deg, kMaxRefAbsRollDeg,
          abs_pitch_deg, kMaxRefAbsPitchDeg);
      }
    }
  }

  last_ndt_pos_ = p;
  last_ndt_stamp_ = stamp;
  has_last_ndt_ = true;
  last_observation = trans;
  last_correction_stamp = stamp;

  if (odom_prediction_) {
    odom_pred_error = odom_guess.inverse() * registration->getFinalTransformation();
    odom_prediction_ = trans;
  }

  return aligned;
}

/* getters */
rclcpp::Time PoseEstimator::last_correction_time() const {
  return last_correction_stamp;
}

Eigen::Vector3f PoseEstimator::pos() const {
  return Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
}

Eigen::Vector3f PoseEstimator::vel() const {
  return Eigen::Vector3f(ukf->mean[3], ukf->mean[4], ukf->mean[5]);
}

Eigen::Quaternionf PoseEstimator::quat() const {
  return Eigen::Quaternionf(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]).normalized();
}

Eigen::Matrix4f PoseEstimator::matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = pos();
  return m;
}

Eigen::Vector3f PoseEstimator::odom_pos() const {
  const Eigen::Matrix4f& prediction = odom_prediction_ ? *odom_prediction_ : last_observation;
  return prediction.block<3, 1>(0, 3);
}

Eigen::Quaternionf PoseEstimator::odom_quat() const {
  const Eigen::Matrix4f& prediction = odom_prediction_ ? *odom_prediction_ : last_observation;
  return Eigen::Quaternionf(prediction.block<3, 3>(0, 0)).normalized();
}

Eigen::Matrix4f PoseEstimator::odom_matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = odom_quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = odom_pos();
  return m;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::wo_prediction_error() const {
  return wo_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::imu_prediction_error() const {
  return imu_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::odom_prediction_error() const {
  return odom_pred_error;
}

PoseEstimator::MatchResult PoseEstimator::GetMatchState() const {
    return match_result_;
}

Eigen::VectorXf PoseEstimator::GetCurrentUkfState() {
    Eigen::Matrix4f state_matrix = matrix();
    Eigen::Vector3f t_state = state_matrix.block<3, 1>(0, 3);
    Eigen::Matrix3f rot = state_matrix.block<3, 3>(0, 0);
    Eigen::Quaternionf q_state(rot);
    q_state.normalize();
    Eigen::VectorXf state(7);
    state.head<3>() = t_state;
    state.tail<4>() << q_state.w(), q_state.x(), q_state.y(), q_state.z();
    return state;
}

}
