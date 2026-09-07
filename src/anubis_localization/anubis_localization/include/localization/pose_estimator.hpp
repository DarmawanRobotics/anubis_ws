#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <memory>
#include <vector>
#include <boost/optional.hpp>

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

namespace kkl {
  namespace alg {
template<typename T, class System> class UnscentedKalmanFilterX;
  }
}

namespace localization {

class PoseSystem;

/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator {
public:
  using PointT = pcl::PointXYZI;

  struct MatchResult{
    bool  is_converged_;   ///< Indicates whether the matching operation converged.
    float fitness_score_;  ///< The fitness score of the matching operation.
  };

  /**
   * @brief constructor
   * @param registration        registration method
   * @param stamp               timestamp
   * @param pos                 initial position
   * @param quat                initial orientation
   * @param cool_time_duration  during "cool time", prediction is not performed
   */
  PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const rclcpp::Time& stamp, const Eigen::Vector3f& pos,
    const Eigen::Quaternionf& quat, double cool_time_duration, double ndt_score_threshold = 0.5,
    bool freeze_vertical_velocity = true,
    Eigen::Vector3d bias_acc = Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d bias_gyro = Eigen::Vector3d(0.0, 0.0, 0.0));
  ~PoseEstimator();

  /**
   * @brief predict
   * @param stamp    timestamp
   */
  void predict(const rclcpp::Time& stamp);

  /**
   * @brief predict
   * @param stamp    timestamp
   * @param acc      acceleration
   * @param gyro     angular velocity
   */
  void predict(const rclcpp::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro);

  /**
   * @brief predict with an explicit dt (R11 catch-up: one-step backlog integration)
   */
  void predict_with_dt(const rclcpp::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro, double dt);

  /**
   * @brief Propagate the constant-velocity model when no fresh IMU sample is available.
   *
   * This is deliberately different from treating a zero IMU measurement as real
   * acceleration.  It only grows the UKF process covariance and keeps the current
   * velocity/orientation model, so a quality-gated NDT observation can still pull the
   * state when the IMU publisher is delayed or temporarily absent.
   */
  void predict_without_imu(const rclcpp::Time& stamp, double dt);

  /** Record the sensor-prediction path used for the next correction diagnostics. */
  void set_prediction_diagnostic(bool imu_available, double imu_age_ms,
                                 bool fallback_active, double fallback_dt_s);

  /** Cumulative UKF prediction count, used to verify that a frame really predicted. */
  long predict_count() const;

  /**
   * @brief zero UKF linear velocity (R11 recovery)
   */
  void reset_velocity();

  /**
   * @brief update the state of the odomety-based pose estimation
   */
  void predict_odom(const Eigen::Matrix4f& odom_delta);
  void clear_odom_prediction() { odom_prediction_ = boost::none; }

  /** Bind a confirmed GL pose to the robot odometry pose at the same instant. */
  void set_trust_anchor(
    const Eigen::Matrix4f& map_pose, const Eigen::Matrix4f& odom_pose);
  void clear_trust_anchor();
  bool has_trust_anchor() const { return trust_anchor_valid_; }
  /** Update absolute odom->base pose used by the GL trust-region check. */
  void update_trust_odom(const Eigen::Matrix4f& odom_pose);
  void suspend_trust_odom() { trust_current_odom_valid_ = false; }

  /**
   * @brief correct
   * @param cloud   input cloud
   * @return cloud aligned to the globalmap
   */
  pcl::PointCloud<PointT>::Ptr correct(const rclcpp::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud);

  /* getters */
  rclcpp::Time last_correction_time() const;

  Eigen::Vector3f pos() const;
  Eigen::Vector3f vel() const;
  Eigen::Quaternionf quat() const;
  Eigen::Matrix4f matrix() const;

  Eigen::Vector3f odom_pos() const;
  Eigen::Quaternionf odom_quat() const;
  Eigen::Matrix4f odom_matrix() const;

  const boost::optional<Eigen::Matrix4f>& wo_prediction_error() const;
  const boost::optional<Eigen::Matrix4f>& imu_prediction_error() const;
  const boost::optional<Eigen::Matrix4f>& odom_prediction_error() const;

  MatchResult GetMatchState() const;
  Eigen::VectorXf GetCurrentUkfState();

  // IMU bias setters (used after static IMU initialization)
  void set_initial_biases(const Eigen::Vector3f& acc_bias, const Eigen::Vector3f& gyro_bias);

  // Round 9: update UKF orientation (used after gravity alignment from static IMU init)
  void set_orientation(const Eigen::Quaternionf& quat);

  /** R13: 0=still … 1=moving, from NDT displacement EMA (not UKF |v|) */
  float motion_scale() const { return motion_scale_; }

  /** [2026-08-31 走廊漂移修复] 喂入 SDK 里程计机体速度(体坐标系,线性 m/s)。
   *  correct() 内做 UKF 速度一致性门:SDK 报静止时强制 UKF 速度归零;
   *  SDK 报运动时把 UKF 速度钳制到 SDK±余量。速度新鲜度由调用方保证。 */
  void set_robot_odom_velocity(const Eigen::Vector3f& v_body, double age_s) {
    robot_odom_velocity_body_ = v_body;
    robot_odom_velocity_age_ = age_s;
    robot_odom_velocity_valid_ = v_body.allFinite() && std::isfinite(age_s);
  }
  void clear_robot_odom_velocity() { robot_odom_velocity_valid_ = false; }

  /** Bootstrap warmup for innovation gate: call after Init success to give UKF
   *  time to converge from the GL initial guess before hard physics gates engage.
   *  Also resets the cumulative-drift reference so a fresh reference is captured
   *  from the multi-frame median at the end of the next warmup period. */
  void start_tracking_gate_warmup() {
    post_init_correction_count_ = kWarmupFrames;
    warmup_segment_consumed_ = 0;
    ++warmup_restart_count_;
    reference_z_ = 0.0f;
    reference_roll_ = 0.0f;
    reference_pitch_ = 0.0f;
    has_reference_z_ = false;
    ref_candidate_count_ = 0;
    drift_gate_disabled_for_session_ = false;
  }

  int tracking_gate_warmup_remaining() const { return post_init_correction_count_; }
  int tracking_gate_warmup_consumed() const { return warmup_total_consumed_; }
  int tracking_gate_warmup_restarts() const { return warmup_restart_count_; }

  /**
   * @brief 设置 map 系重力方向（单位向量），由 config 的 map_gravity_direction 提供
   *
   * [2026-08-16] 原实现在线从"已接受的 NDT 姿态"学习本基准，实测被污染：落位后
   * 不到 1 秒 NDT 就把姿态拖了 10.5°，学到的基准与地图地面法向差 11.67°
   * （实测地面拟合倾角仅 1.23°，法向 [-0.0215, 0.0016, 0.9998]）。
   * 地图的重力方向是**离线可确定的固定属性**，不应依赖运行时 NDT。
   */
  void set_map_gravity_direction(const Eigen::Vector3f& dir);
  const Eigen::Vector3f& map_gravity_direction() const { return g_map_dir_; }

  /**
   * @brief 用已收敛的重力方向播种低通滤波器，使姿态锚定在第一帧即可用
   *
   * [2026-08-17] 实测的确定性故障：每次 GL 落位都会新建 PoseEstimator，重力 LPF
   * 计数归零，需要 kGravityLpfMinSamples(20) 个样本（约 0.5s @40Hz）才可用。而
   * 落位时 imu_data.clear() 又清空了缓冲，于是**落位后第一次 correct() 时锚定必然
   * 不可用**，`gravityObservationFresh()` 返回 false，原始 NDT 姿态直接进 UKF；
   * 同时 warmup 把角度门放宽到 30°，NDT 获得最大改写权限。
   *
   * 四次落位的实测序列完全一致：
   *     落位帧 grav=0 pitch≈+0.6°  →  +1s grav=1 pitch≈-6°  gerr=10~14°
   * 即 applyImuBiasIfReady 设好的重力姿态(pitch≈-5.7°)被第一帧 NDT 拉走 6°，
   * 锚定上线后才发现偏离 10~14°，但污染已进 UKF → score 0.023→0.08 → 拒帧 →
   * grace → 重新 GL → 又一次真空期（本轮 11 次落位 / 78 次 grace 的成因）。
   *
   * static_imu_init_ 已持有一个收敛的重力估计，直接拿来播种即可消除真空期。
   *
   * @param up_body  body 系下的"上"方向（= -重力方向），无需预先归一化
   * @param stamp    播种时刻，用于满足观测新鲜度检查
   */
  void seed_gravity_estimate(const Eigen::Vector3f& up_body, const rclcpp::Time& stamp);

  /** 落位后捕获的地面 z 参考；未捕获时返回 false。用于诊断 z 漂移。 */
  bool ground_reference_z(float& z_out) const {
    if (!has_reference_z_) {
      return false;
    }
    z_out = reference_z_;
    return true;
  }

  /**
   * @brief 跟踪层逐帧诊断量
   *
   * 设计目的：回答"运动中到底是哪个量在漂、为什么漂"。关键是 innov_* ——
   * 它是 NDT 结果相对**预测**的修正量，能区分两种完全不同的故障：
   *   innov_dz 持续为负  → NDT 在把 z 往下拽（配准退化）
   *   innov_dz ≈ 0 但 z 仍下沉 → 预测自己在往下漂（IMU/odom 侧问题）
   * 姿态同理。没有这个分解，只看最终位姿无法归因。
   */
  struct TrackingDiag {
    bool  gravity_applied   = false;  ///< 本帧观测的 roll/pitch 是否被重力改写
    float gravity_err_deg   = 0.0f;   ///< 改写前 NDT 姿态偏离重力基准的角度
    float gravity_accept    = 0.0f;   ///< 重力样本可用率 [0,1]
    float gravity_mag_err   = 0.0f;   ///< 低通后加速度模长与 g 的偏差 (m/s²)
    float innov_dxy         = 0.0f;   ///< |NDT_xy - 预测_xy| (m)
    float innov_dz          = 0.0f;   ///< NDT_z - 预测_z (m，带符号)
    float innov_dyaw_deg    = 0.0f;   ///< NDT_yaw - 预测_yaw (deg，带符号)
    // ---- 姿态链路：同帧区分 IMU/预测、原始 NDT、重力替换和 UKF 融合 ----
    bool  attitude_chain_valid = false;
    float pred_roll_deg        = 0.0f;
    float pred_pitch_deg       = 0.0f;
    float pred_yaw_deg         = 0.0f;
    float ndt_roll_deg         = 0.0f;
    float ndt_pitch_deg        = 0.0f;
    float ndt_yaw_deg          = 0.0f;
    float gravity_obs_roll_deg = 0.0f;
    float gravity_obs_pitch_deg = 0.0f;
    float gravity_obs_yaw_deg  = 0.0f;
    // ---- 平移链路：UKF/IMU、SDK odom、NDT 初值/观测、UKF 校正后 ----
    bool  position_chain_valid = false;
    float pred_imu_x = 0.0f;
    float pred_imu_y = 0.0f;
    float pred_imu_z_full = 0.0f;
    float pred_odom_x = 0.0f;
    float pred_odom_y = 0.0f;
    float pred_odom_z_full = 0.0f;
    float pred_used_x = 0.0f;
    float pred_used_y = 0.0f;
    float pred_used_z_full = 0.0f;
    float ndt_x = 0.0f;
    float ndt_y = 0.0f;
    float ndt_z = 0.0f;
    float ukf_pre_x = 0.0f;
    float ukf_pre_y = 0.0f;
    float ukf_pre_z = 0.0f;
    float ukf_post_x = 0.0f;
    float ukf_post_y = 0.0f;
    float ukf_post_z = 0.0f;
    float ukf_pre_vx = 0.0f;
    float ukf_pre_vy = 0.0f;
    float ukf_pre_vz = 0.0f;
    float ukf_post_vx = 0.0f;
    float ukf_post_vy = 0.0f;
    float ukf_post_vz = 0.0f;
    float ukf_pos_cov_x = 0.0f;
    float ukf_pos_cov_y = 0.0f;
    float ukf_pos_cov_z = 0.0f;
    float ukf_measurement_pos_variance = 0.0f;
    float ukf_mahalanobis_sq = -1.0f;
    float ukf_motion_scale = 0.0f;
    bool  ukf_correct_applied = false;
    // ---- 本帧预测供给：区分 IMU 正常、IMU 延迟和无 IMU 兜底 ----
    bool  imu_prediction_available = false;
    bool  imu_fallback_active = false;
    float imu_age_ms = 0.0f;
    float imu_fallback_dt_ms = 0.0f;
    // ---- 预测来源分解：定位"预测为什么漂"的唯一手段 ----
    bool  odom_pred_valid   = false;  ///< 本帧是否有可用的 odom 预测
    float pred_imu_z        = 0.0f;   ///< UKF 预测的 z
    float pred_odom_z       = 0.0f;   ///< odom 预测的 z（= 上次 NDT z + odom Δz）
    float pred_used_z       = 0.0f;   ///< 实际送进 NDT 的初值 z
    float pred_odom_dz      = 0.0f;   ///< odom 相对上次已接受 NDT 位姿的 Δz
    float pred_imu_yaw_deg  = 0.0f;
    float pred_odom_yaw_deg = 0.0f;
    // ---- UKF 健康度：验证流形近似的实际严重程度 ----
    float ukf_quat_norm     = 1.0f;   ///< 均值四元数的模（流形修正后应恒为 1）
    float ukf_quat_cov_tr   = 0.0f;   ///< 四元数块协方差的迹
    float ukf_sigma_quat_dev = 0.0f;  ///< sigma 点四元数与均值的最大夹角(度)
    long  ukf_llt_failures  = 0;      ///< LLT 分解失败累计次数（正常应为 0）
    long  ukf_predicts      = 0;      ///< predict() 累计调用次数（为 0 = 滤波器只在被 correct，必然塌缩）
    // ---- NDT 分段：区分"重建源结构"与"真正的配准迭代" ----
    double ndt_setsrc_ms    = 0.0;    ///< setInputSource 耗时（应≈0，否则每帧在建源 KD 树）
    double ndt_align_ms     = 0.0;    ///< align 耗时（走 ndt_omp 的 Release 编译版）
    double ndt_score_ms     = 0.0;    ///< getFitnessScore 耗时（PCL 模板，在本包内实例化，每点一次 KD 搜索）
    bool  occupancy_ready   = false;  ///< 已有 reference_z 且 target KD 树可查询
    bool  occupancy_blocked = false;  ///< 标定阈值判定；observe-only 时不会拒帧
    int   occupancy_points  = 0;      ///< XY 邻域和机身高度带内的地图点数
    bool  trust_valid       = false;  ///< GL/odom 锚点与当前 odom 均可用
    float trust_deviation_xy = 0.0f;  ///< 候选 XY 与 odom 可解释期望位置的偏离
    float trust_allowed_xy   = 0.0f;  ///< 基准 + 里程增长后的允许偏离
    float trust_gate_xy      = 0.40f; ///< 本帧实际使用的 XY innovation 门限
    float trust_travel_m     = 0.0f;  ///< 自锚点以来累计 odom 平面里程
    int   trust_exceed_count = 0;     ///< 连续超出信任域的候选帧数
  };
  const TrackingDiag& tracking_diag() const { return tracking_diag_; }

private:
  /** Scale Q for the upcoming predict using last motion_scale_ / ghost-velocity state */
  Eigen::MatrixXf scaledProcessNoise() const;

  /**
   * @brief 用加速度计的重力观测锚定 UKF 的 roll/pitch
   *
   * [2026-08-16] UKF 的 roll/pitch 此前只由陀螺积分与 NDT 观测决定，**没有任何
   * 绝对参考**：陀螺积分会漂，而室内平面环境下 NDT 对 pitch/z 不敏感（倾斜
   * 40° 时墙面仍能对上，fitness_score 照样是 0.21 的"好分"）。两者耦合成正
   * 反馈——预测倾斜 → NDT 从倾斜初值收敛到倾斜局部极小 → 观测"确认"倾斜 →
   * 更倾斜。实测 pitch 发散到 -47°、z 漂到 1.05m。
   *
   * 加速度计在低动态时直接测量重力方向，是 roll/pitch 唯一的绝对参考。本函数
   * 做标准 AHRS 互补滤波：仅在低动态窗口内，以小增益把姿态朝"重力对齐"方向
   * 拉回。高动态时不修正，短时纯陀螺积分足够可信。四足行走每步支撑相都有低
   * 动态瞬间，采样充足。
   *
   * 修正量是对齐两个向量的最小旋转，天然不含 yaw 分量，因此不会干扰由 NDT
   * 独占约束的航向。
   */
  void updateGravityAnchor(const rclcpp::Time& stamp, const Eigen::Vector3f& acc,
                           const Eigen::Vector3f& gyro);

  /** 低动态重力样本是否足够新鲜，可用于改写观测的 roll/pitch */
  bool gravityObservationFresh(const rclcpp::Time& stamp) const;

  /** Accumulate and emit the one-second gravity-validity calibration window. */
  void recordGravityDiagnostic(
    const rclcpp::Time& stamp, float acceleration_norm, bool usable);

  rclcpp::Time init_stamp;             // when the estimator was initialized
  rclcpp::Time prev_stamp;             // when the estimator was updated last time
  rclcpp::Time last_correction_stamp;  // when the estimator performed the correction step
  double cool_time_duration;

  Eigen::MatrixXf process_noise;       // base Q (still)
  Eigen::MatrixXf measurement_noise_;  // base R (still / R10 defaults)
  MatchResult     match_result_;
  double          ndt_score_threshold_;  ///< NDT fitness score threshold for rejecting bad matches
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>> ukf;
  boost::optional<Eigen::Matrix4f> odom_prediction_;

  Eigen::Matrix4f last_observation;
  boost::optional<Eigen::Matrix4f> wo_pred_error;
  boost::optional<Eigen::Matrix4f> imu_pred_error;
  boost::optional<Eigen::Matrix4f> odom_pred_error;

  // R13 dynamic noise: motion from NDT map-frame speed, not inflated UKF velocity
  Eigen::Vector3f last_ndt_pos_ = Eigen::Vector3f::Zero();
  rclcpp::Time    last_ndt_stamp_;
  bool            has_last_ndt_ = false;
  float           motion_scale_ = 0.0f;  // EMA in [0,1]

  // R13.1: ~1s displacement window (kills cm-level NDT jitter false motion)
  Eigen::Vector3f motion_window_origin_ = Eigen::Vector3f::Zero();
  rclcpp::Time    motion_window_stamp_;
  bool            has_motion_window_ = false;

  // [2026-08-31 走廊漂移修复] SDK 里程计机体速度(与雷达帧对齐后由
  // nodelet 喂入)。走廊内 NDT 沿轴向不可观测,UKF 会把滑移学成速度并
  // 自我强化(实测:狗静止定位 0.35m/s 滑移、行走超速 60%)。SDK twist
  // 提供该缺失自由度的真值。
  Eigen::Vector3f robot_odom_velocity_body_ = Eigen::Vector3f::Zero();
  double          robot_odom_velocity_age_ = 0.0;
  bool            robot_odom_velocity_valid_ = false;
  uint32_t        sdk_gate_still_count_ = 0;   // 连续静止帧数(诊断)
  uint32_t        sdk_gate_clamp_count_ = 0;   // 钳制触发帧数(诊断)

  pcl::Registration<PointT, PointT>::Ptr registration;

  // DIAG: counter for first-N IMU predict logging (resets per PoseEstimator instance)
  int predict_diag_count_ = 0;

  // Innovation gate: reject NDT observations whose position/orientation jump from
  // the UKF prediction exceeds physics-based limits for a ground robot.
  float max_ndt_correction_xy_    = 0.40f;   ///< max allowed |obs_xy - pred_xy| per frame (m)
  float max_ndt_correction_z_     = 0.30f;   ///< max allowed |obs_z  - pred_z|  per frame (m)
  float max_ndt_correction_angle_ = 15.0f;   ///< max allowed |obs_rpy - pred_rpy| per frame (deg)

  // Bootstrap warmup: relaxed thresholds for first N frames after init, giving UKF
  // time to converge from the GL initial guess before hard physics gates engage.
  static constexpr int    kWarmupFrames                  = 10;
  static constexpr float  kWarmupCorrectionZ             = 0.80f;
  static constexpr float  kWarmupCorrectionAngle         = 30.0f;
  int post_init_correction_count_                        = kWarmupFrames;
  int warmup_segment_consumed_                           = 0;
  int warmup_total_consumed_                             = 0;
  int warmup_restart_count_                              = 0;

  // GL trust region: compare map tracking motion with motion explainable by
  // odometry, then progressively reduce NDT's per-frame XY authority.
  static constexpr float kTrustBaseAllowanceM             = 0.30f;
  static constexpr float kTrustAllowancePerTravelM        = 0.10f;
  static constexpr float kTrustTighteningScaleM           = 0.20f;
  static constexpr float kTrustMinCorrectionXY            = 0.05f;
  static constexpr float kTrustMaxOdomStepM               = 1.00f;
  Eigen::Matrix4f trust_anchor_map_pose_ = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f trust_anchor_odom_pose_ = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f trust_current_odom_pose_ = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f trust_last_odom_pose_ = Eigen::Matrix4f::Identity();
  bool trust_anchor_valid_ = false;
  bool trust_current_odom_valid_ = false;
  bool trust_last_odom_valid_ = false;
  float trust_travel_m_ = 0.0f;
  int trust_exceed_count_ = 0;

  // Cumulative drift constraint: after warmup expires, the ground robot's
  // Z and attitude must stay within physical limits.  Single-frame innovation
  // gates cannot detect slow consistent drift (e.g. NDT and UKF drifting
  // together at ~0.15 m/frame over 10+ seconds).
  static constexpr float  kMaxZDriftFromReference        = 0.50f;   ///< max |Z - ref_Z| after warmup (m)
  static constexpr float  kMaxRollDriftDeg               = 25.0f;   ///< max |roll - ref_roll| after warmup (deg)
  static constexpr float  kMaxPitchDriftDeg              = 25.0f;   ///< max |pitch - ref_pitch| after warmup (deg)

  // Reference-capture guards: the reference is only accepted when the
  // multi-frame median attitude is within these absolute limits.  A single
  // bad warmup-end frame (observed: roll=-62°) must not become the drift
  // baseline.  When the reference cannot pass these limits the drift gate
  // stays disabled for the entire tracking session.
  static constexpr int    kRefCandidateBufSize            = 5;       ///< median over last N warmup frames
  static constexpr int    kMinRefCandidateCount           = 3;       ///< minimum accepted warmup frames required to build a reference
  static constexpr float  kMaxRefAbsRollDeg               = 10.0f;   ///< refuse reference if |median_roll| exceeds this
  static constexpr float  kMaxRefAbsPitchDeg              = 15.0f;   ///< refuse reference if |median_pitch| exceeds this
  /// 参考只能在近似静止时采集：运动中的姿态不是静止基准，按静止阈值判会被拒，
  /// 进而让 Z 漂移门控整轮失效（实测 roll 中位 12.1° > 10° → 门控禁用）。
  static constexpr float  kRefCaptureMaxFrameJump         = 0.03f;   ///< 帧间位移上限 (m)
  static constexpr float  kRefCaptureMaxMotionScale       = 0.15f;   ///< motion_scale_ 上限

  // Candidate occupancy calibration. Keep rejection disabled until a field run
  // establishes normal/doorway/wall-adjacent point-count distributions.
  static constexpr bool   kOccupancyRejectEnabled          = false;
  static constexpr float  kOccupancyRadiusXY               = 0.30f;
  static constexpr float  kOccupancyHalfHeight             = 0.12f;
  static constexpr int    kOccupancyPointThreshold         = 4;

  float  reference_z_     = 0.0f;
  float  reference_roll_  = 0.0f;
  float  reference_pitch_ = 0.0f;
  bool   has_reference_z_ = false;

  // Multi-frame buffer for robust reference capture (ring buffer, oldest first)
  float ref_z_buf_[kRefCandidateBufSize]     = {};
  float ref_roll_buf_[kRefCandidateBufSize]  = {};
  float ref_pitch_buf_[kRefCandidateBufSize] = {};
  int   ref_candidate_count_                 = 0;
  bool  drift_gate_disabled_for_session_     = false;

  // ---- 重力锚定姿态（2026-08-16） -------------------------------------------
  // [第一版] 用"瞬时低动态门控"筛重力样本：|‖acc‖-g|<0.6 且 ‖gyro‖<0.4rad/s。
  // 实测**四足行走时几乎永远不成立**，锚定一走就下线 —— 静止 pitch 跨度 7.9°，
  // 行走 22.6°；z 运动中塌陷 0.3~0.6m，停下才恢复；yaw 一次移动永久偏 25°。
  // [第二版] 改为低通滤波：步态水平加速度在一个步态周期内零均值，用比步态周期
  // 长的时间常数滤掉即得重力，**行走时同样有效**。仅剔除极端离群帧（落足冲击）。
  static constexpr float kGravityMagnitude       = 9.80665f;
  /// 低通时间常数，必须 > 步态周期（四足约 0.3~0.5s）
  static constexpr float kGravityLpfTauS         = 1.0f;
  /// 低通后模长与 g 的偏差上限；超出说明存在真实的持续加速，此时重力不可信
  /// [2026-08-16 修正] 判据必须用「模长的标量平均」，不能用「向量平均的模长」：
  /// 运动中机身姿态摆动 ±20°+，向量平均会因方向发散而缩短模长约
  /// 1-cos(20°)=6% → 9.81×0.06≈0.59 m/s²，正好撞上本阈值，导致重力在运动中被
  /// 误判为不可用（实测 gmag 0.33~0.43，grav=0）。标量平均不受方向发散影响。
  static constexpr float kGravityLpfMagTol       = 0.80f;
  /// 单帧离群剔除（落足冲击/自由落体），不进滤波器但不影响已有估计
  static constexpr float kGravitySampleOutlier   = 5.0f;
  /// 滤波器收敛所需的最少样本（约 0.5s @40Hz），未达到前不启用锚定
  static constexpr int   kGravityLpfMinSamples   = 20;
  /// 每秒增益：与 dt 相乘得到单帧 slerp 比例，使行为不随 IMU 频率变化。
  static constexpr float kGravityAnchorGainPerSec = 0.40f;
  static constexpr float kGravityAnchorMaxAlpha   = 0.02f;  ///< 单帧修正比例上限
  static constexpr float kGravityAnchorWarnDeg    = 10.0f;  ///< 倾角误差超此值打日志
  /// 重力样本新鲜度上限：超过此年龄就不改写观测的 roll/pitch，退回纯 NDT 姿态。
  static constexpr double kGravityObservationMaxAgeS = 0.30;

  Eigen::Vector3f g_map_dir_        = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  Eigen::Vector3f gravity_body_     = Eigen::Vector3f(0.0f, 0.0f, 1.0f);  ///< 低通后的重力方向（body 系单位向量）
  Eigen::Vector3f gravity_lpf_      = Eigen::Vector3f::Zero();            ///< 加速度低通累加器（向量：给方向）
  float           gravity_mag_lpf_  = 0.0f;                              ///< ‖acc‖ 的标量低通（给有效性判据，不受方向发散影响）
  bool            has_gravity_lpf_  = false;
  int             gravity_lpf_count_ = 0;
  bool            has_gravity_body_ = false;
  rclcpp::Time    last_gravity_stamp_;
  int             gravity_warn_throttle_ = 0;
  int             gravity_substitution_log_ = 0;
  // 重力样本统计（用于诊断门控是否过严）
  long            gravity_total_    = 0;
  long            gravity_usable_   = 0;
  rclcpp::Time    gravity_diag_window_start_;
  std::vector<float> gravity_diag_norms_;
  size_t          gravity_diag_total_  = 0;
  size_t          gravity_diag_usable_ = 0;

  TrackingDiag    tracking_diag_;

  rclcpp::Logger logger_ = rclcpp::get_logger("pose_estimator");
};

}  // namespace localization

#endif  // POSE_ESTIMATOR_HPP
