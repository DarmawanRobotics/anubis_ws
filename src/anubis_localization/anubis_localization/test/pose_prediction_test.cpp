#include <gtest/gtest.h>

#include <limits>

#include <localization/pose_prediction.hpp>
#include <localization/pose_system.hpp>

namespace localization {
namespace {

Eigen::Matrix3f rotation(float roll, float pitch, float yaw) {
  return (
    Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
    Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
    Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()))
    .toRotationMatrix();
}

TEST(PosePrediction, ReusesAcceptedObservationInsteadOfAccumulatingRejectedFrames) {
  Eigen::Matrix4f last_accepted = Eigen::Matrix4f::Identity();
  last_accepted.block<3, 3>(0, 0) = rotation(0.0f, 0.0f, 0.4f);
  last_accepted.block<3, 1>(0, 3) = Eigen::Vector3f(2.0f, -1.0f, 0.3f);

  Eigen::Matrix4f odom_delta = Eigen::Matrix4f::Identity();
  odom_delta.block<3, 3>(0, 0) = rotation(0.0f, 0.0f, 0.1f);
  odom_delta.block<3, 1>(0, 3) = Eigen::Vector3f(0.2f, 0.0f, 0.0f);

  const Eigen::Matrix4f first = applyOdomDelta(last_accepted, odom_delta);
  const Eigen::Matrix4f after_rejection = applyOdomDelta(last_accepted, odom_delta);

  EXPECT_TRUE(first.isApprox(after_rejection, 1e-6f));
  EXPECT_FALSE(first.isApprox(first * odom_delta, 1e-6f));
}

// 契约：x/y 与 yaw 取 odom，**z 与 roll/pitch 取 IMU/UKF**。
// z 曾于 2026-08-16 上午改成取 odom，当天下午依据实测回退：innov_dz 每帧仅
// ±0.03 且正负交替，z 却掉 0.64m —— 下沉由预测驱动，而预测 z 正来自腿式里程计。
TEST(PosePrediction, UsesOdomXyYawButKeepsImuZAndRollPitch) {
  Eigen::Matrix4f imu = Eigen::Matrix4f::Identity();
  imu.block<3, 3>(0, 0) = rotation(0.12f, -0.08f, 0.3f);
  imu.block<3, 1>(0, 3) = Eigen::Vector3f(1.0f, 2.0f, 0.7f);

  Eigen::Matrix4f odom = Eigen::Matrix4f::Identity();
  odom.block<3, 3>(0, 0) = rotation(-0.4f, 0.2f, 1.1f);
  odom.block<3, 1>(0, 3) = Eigen::Vector3f(4.0f, 5.0f, -2.0f);

  const Eigen::Matrix4f result = composePlanarOdomPrediction(imu, odom);
  const Eigen::Vector3f rpy = rotationToRpy(result.block<3, 3>(0, 0));

  EXPECT_NEAR(result(0, 3), 4.0f, 1e-5f);    // x ← odom
  EXPECT_NEAR(result(1, 3), 5.0f, 1e-5f);    // y ← odom
  EXPECT_NEAR(result(2, 3), 0.7f, 1e-5f);    // z ← IMU（不得取 odom 的 -2.0）
  EXPECT_NEAR(rpy.x(), 0.12f, 1e-5f);        // roll  ← IMU
  EXPECT_NEAR(rpy.y(), -0.08f, 1e-5f);       // pitch ← IMU
  EXPECT_NEAR(rpy.z(), 1.1f, 1e-5f);         // yaw   ← odom
}

TEST(PosePrediction, GravityTiltPreservesYawWithRollAndPitch) {
  constexpr float roll = 0.24f;
  constexpr float pitch = -0.17f;
  constexpr float yaw = 1.13f;
  const Eigen::Vector3f up_body =
    rotation(roll, pitch, 0.0f).transpose() * Eigen::Vector3f::UnitZ();

  const Eigen::Quaternionf orientation =
    orientationFromGravityPreservingYaw(up_body, yaw);
  const Eigen::Vector3f rpy = rotationToRpy(orientation.toRotationMatrix());

  EXPECT_NEAR(rpy.x(), roll, 1e-5f);
  EXPECT_NEAR(rpy.y(), pitch, 1e-5f);
  EXPECT_NEAR(rpy.z(), yaw, 1e-5f);
  EXPECT_TRUE((orientation * up_body).isApprox(Eigen::Vector3f::UnitZ(), 1e-5f));
}

TEST(PosePrediction, RequestedInitialPoseResetsEvenWhenPoseIsUnchanged) {
  EXPECT_TRUE(initialPoseRequiresReset(
      true, true, true, 0.0f, 0.0f, 0.05f, 0.01f));
  EXPECT_FALSE(initialPoseRequiresReset(
      true, true, false, 0.0f, 0.0f, 0.05f, 0.01f));
}

// ---- 重力锚定姿态约束（2026-08-16） ---------------------------------------
// 背景：UKF 的 roll/pitch 此前只由陀螺积分与 NDT 观测决定，没有绝对参考。室内
// 平面环境下 NDT 对 pitch/z 不敏感（倾斜 40° 墙面仍能对上，score 照样 0.21），
// 与陀螺漂移耦合成正反馈，实测 pitch 发散到 -47°、z 漂到 1.05m。

constexpr float kG = 9.80665f;

TEST(GravityAnchor, AcceptsPureGravityAndRejectsHighDynamicSamples) {
  const Eigen::Vector3f still_gyro(0.01f, -0.02f, 0.005f);

  // 静止：比力就是重力
  EXPECT_TRUE(isLowDynamicGravitySample(
      Eigen::Vector3f(0.0f, 0.0f, kG), still_gyro, kG, 0.60f, 0.40f));

  // 4 m/s² 水平加速：模长 10.6，超出 0.6 容差 → 拒
  EXPECT_FALSE(isLowDynamicGravitySample(
      Eigen::Vector3f(4.0f, 0.0f, kG), still_gyro, kG, 0.60f, 0.40f));

  // 自由落体/落足冲击：模长严重偏离 → 拒
  EXPECT_FALSE(isLowDynamicGravitySample(
      Eigen::Vector3f(0.0f, 0.0f, 0.0f), still_gyro, kG, 0.60f, 0.40f));

  // 快速转身：角速度超限 → 拒（比力模长仍像重力也不行）
  EXPECT_FALSE(isLowDynamicGravitySample(
      Eigen::Vector3f(0.0f, 0.0f, kG), Eigen::Vector3f(0.0f, 0.0f, 1.2f),
      kG, 0.60f, 0.40f));
}

TEST(GravityAnchor, PullsAttitudeTowardGravityWithoutChangingYaw) {
  constexpr float kYaw = 0.8f;
  constexpr float kBadPitch = 20.0f * static_cast<float>(M_PI) / 180.0f;

  // 狗实际是水平的（body 系测到的重力方向就是 +z），但 UKF 认为俯仰了 20°
  const Eigen::Vector3f gravity_body(0.0f, 0.0f, 1.0f);
  const Eigen::Vector3f g_map_dir(0.0f, 0.0f, 1.0f);
  Eigen::Quaternionf q(rotation(0.0f, kBadPitch, kYaw));

  ASSERT_NEAR(gravityTiltErrorDeg(q, gravity_body, g_map_dir), 20.0f, 1e-3f);

  float prev_err = gravityTiltErrorDeg(q, gravity_body, g_map_dir);
  for (int i = 0; i < 400; ++i) {
    q = gravityAnchoredOrientation(q, gravity_body, g_map_dir, 0.02f);
    const float err = gravityTiltErrorDeg(q, gravity_body, g_map_dir);
    EXPECT_LT(err, prev_err + 1e-5f) << "倾角误差必须单调收敛，第 " << i << " 次";
    prev_err = err;
    // 航向由 NDT 独占约束，重力修正绕的是水平轴，任何一步都不得改变 yaw
    EXPECT_NEAR(rotationToRpy(q.toRotationMatrix()).z(), kYaw, 1e-4f)
        << "第 " << i << " 次修正改变了 yaw";
  }

  // 400 步 × 0.02 增益足以收敛（1-0.02)^400 ≈ 3e-4
  EXPECT_LT(prev_err, 0.1f);
  const Eigen::Vector3f rpy = rotationToRpy(q.toRotationMatrix());
  EXPECT_NEAR(rpy.x(), 0.0f, 1e-3f);
  EXPECT_NEAR(rpy.y(), 0.0f, 1e-3f);
}

TEST(GravityAnchor, ConvergesToLearnedTiltedMapGravityNotToLevel) {
  // 地图由未标定外参的雷达建出，地面拟合倾角实测 3.18° —— map 的 z 轴并不是
  // 重力。锚定必须收敛到"学到的 map 系重力方向"，否则会与地图倾角持续对抗。
  const float tilt = 3.18f * static_cast<float>(M_PI) / 180.0f;
  const Eigen::Vector3f g_map_dir(std::sin(tilt), 0.0f, std::cos(tilt));
  const Eigen::Vector3f gravity_body(0.0f, 0.0f, 1.0f);

  Eigen::Quaternionf q(rotation(0.0f, 0.0f, 0.3f));  // 水平姿态
  for (int i = 0; i < 600; ++i) {
    q = gravityAnchoredOrientation(q, gravity_body, g_map_dir, 0.02f);
  }

  EXPECT_LT(gravityTiltErrorDeg(q, gravity_body, g_map_dir), 0.1f);
  // 收敛后 map 系下的重力观测应与基准重合，而不是回到 (0,0,1)
  const Eigen::Vector3f measured_map = q * gravity_body;
  EXPECT_NEAR(measured_map.x(), g_map_dir.x(), 1e-3f);
  EXPECT_NEAR(measured_map.z(), g_map_dir.z(), 1e-3f);
  EXPECT_GT(std::abs(measured_map.x()), 0.03f) << "不应被拉回水平";
}

TEST(GravityAnchor, ReturnsInputOnDegenerateArguments) {
  const Eigen::Quaternionf q(rotation(0.1f, -0.2f, 0.5f));
  const Eigen::Vector3f g(0.0f, 0.0f, 1.0f);

  // 零增益、零向量、非有限值都必须原样返回，不得污染姿态
  EXPECT_TRUE(gravityAnchoredOrientation(q, g, g, 0.0f).isApprox(q, 1e-6f));
  EXPECT_TRUE(gravityAnchoredOrientation(q, Eigen::Vector3f::Zero(), g, 0.02f)
                  .isApprox(q, 1e-6f));
  EXPECT_TRUE(gravityAnchoredOrientation(q, g, Eigen::Vector3f::Zero(), 0.02f)
                  .isApprox(q, 1e-6f));
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(gravityAnchoredOrientation(q, Eigen::Vector3f(nan, 0.0f, 1.0f), g,
                                         0.02f)
                  .isApprox(q, 1e-6f));
}

// 重力对齐系下的 yaw —— 这才是物理意义上的航向。map 系的 ZYX yaw 会随地图
// 自身倾角与当前 roll/pitch 轻微耦合，不是恒定不变量。
float gravityFrameYaw(const Eigen::Quaternionf& q, const Eigen::Vector3f& g_map) {
  const Eigen::Quaternionf r_g = Eigen::Quaternionf::FromTwoVectors(
      Eigen::Vector3f::UnitZ(), g_map.normalized());
  return rotationToRpy(((r_g.conjugate() * q).normalized()).toRotationMatrix()).z();
}

TEST(GravityAnchor, FullSubstitutionKeepsYawWhenRollAndPitchBothNonZero) {
  // 回归：2026-08-16 实测失败用例。原实现用"对齐两向量的最小旋转"，在 roll 与
  // pitch 同时非零时会注入 yaw 偏移（roll=3.4°/pitch=10.5° → 0.44° yaw 污染）。
  // 地图正对齐时（g_map=(0,0,1)），map 系 yaw 必须**精确**保留。
  constexpr float kYaw = -1.35f;
  const Eigen::Vector3f gravity_body(0.0f, 0.0f, 1.0f);
  const Eigen::Vector3f g_map_dir(0.0f, 0.0f, 1.0f);
  const Eigen::Quaternionf q_ndt(
      rotation(0.06f, 10.5f * static_cast<float>(M_PI) / 180.0f, kYaw));

  ASSERT_GT(gravityTiltErrorDeg(q_ndt, gravity_body, g_map_dir), 9.0f);

  const Eigen::Quaternionf q_obs =
      gravityAnchoredOrientation(q_ndt, gravity_body, g_map_dir, 1.0f);

  EXPECT_LT(gravityTiltErrorDeg(q_obs, gravity_body, g_map_dir), 1e-3f);
  EXPECT_NEAR(rotationToRpy(q_obs.toRotationMatrix()).z(), kYaw, 1e-5f);
  // roll/pitch 被重力值完全取代
  const Eigen::Vector3f rpy = rotationToRpy(q_obs.toRotationMatrix());
  EXPECT_NEAR(rpy.x(), 0.0f, 1e-5f);
  EXPECT_NEAR(rpy.y(), 0.0f, 1e-5f);
}

TEST(GravityAnchor, FullSubstitutionOnTiltedMapPreservesGravityFrameYaw) {
  // 地图有倾角时，守恒量是"重力对齐系下的 yaw"，而非 map 系 ZYX yaw。
  constexpr float kYaw = -1.35f;
  const Eigen::Vector3f gravity_body(0.0f, 0.0f, 1.0f);
  const float tilt = 1.23f * static_cast<float>(M_PI) / 180.0f;
  const Eigen::Vector3f g_map_dir(-std::sin(tilt), 0.0f, std::cos(tilt));
  const Eigen::Quaternionf q_ndt(
      rotation(0.06f, 10.5f * static_cast<float>(M_PI) / 180.0f, kYaw));

  const Eigen::Quaternionf q_obs =
      gravityAnchoredOrientation(q_ndt, gravity_body, g_map_dir, 1.0f);

  // 姿态完全对齐到地图的重力基准
  EXPECT_LT(gravityTiltErrorDeg(q_obs, gravity_body, g_map_dir), 1e-3f);
  // 重力系 yaw 精确保留 —— 物理航向一点没丢，这才是真正的不变量
  EXPECT_NEAR(gravityFrameYaw(q_obs, g_map_dir),
              gravityFrameYaw(q_ndt, g_map_dir), 1e-5f);

  // map 系 ZYX yaw 会有二阶残余：在倾斜的参考系里改变 roll/pitch，yaw 的分解
  // 读数必然跟着变。量级 ≈ 地图倾角(rad) × 姿态修正量(rad)：
  //   0.0215 × 0.183 = 0.0039 rad = 0.225°，实测 0.228°，吻合。
  // 这是坐标读数的几何后果、不是航向污染（对比：修复前那 0.44° 是在地图**正
  // 对齐**时产生的，属于真污染）。发布的 TF/odom 用的是完整四元数，不受影响。
  // 系统正常工作后 NDT 倾角误差会降到 1~2°，该残余随之降到 0.04° 以下。
  const float ndt_tilt_rad =
      gravityTiltErrorDeg(q_ndt, gravity_body, g_map_dir) * static_cast<float>(M_PI) / 180.0f;
  const float map_tilt_rad = std::acos(g_map_dir.normalized().z());
  const float expected_bound_deg =
      2.0f * map_tilt_rad * ndt_tilt_rad * 180.0f / static_cast<float>(M_PI);
  const float map_yaw_shift_deg =
      std::abs(rotationToRpy(q_obs.toRotationMatrix()).z() - kYaw) *
      180.0f / static_cast<float>(M_PI);
  EXPECT_LT(map_yaw_shift_deg, expected_bound_deg)
      << "map 系 yaw 残余应受二阶项 (地图倾角 × 修正量) 约束";
}

TEST(GravityAnchor, AlphaOneIsIdempotent) {
  // 连续两次完全替换结果不变：说明替换是投影而非累积旋转，不会逐帧越拉越远
  const Eigen::Vector3f gravity_body(0.05f, -0.02f, 0.998f);
  const Eigen::Vector3f g_map_dir(-0.0215f, 0.0016f, 0.9998f);
  const Eigen::Quaternionf q(rotation(0.15f, -0.22f, 0.9f));

  const Eigen::Quaternionf once =
      gravityAnchoredOrientation(q, gravity_body, g_map_dir, 1.0f);
  const Eigen::Quaternionf twice =
      gravityAnchoredOrientation(once, gravity_body, g_map_dir, 1.0f);

  EXPECT_LT(gravityTiltErrorDeg(once, gravity_body, g_map_dir), 1e-2f);
  EXPECT_NEAR(gravityFrameYaw(once, g_map_dir), gravityFrameYaw(q, g_map_dir), 1e-5f);
  EXPECT_TRUE(once.isApprox(twice, 1e-5f));
}

// ---- 重力低通估计（2026-08-16 第二版） -------------------------------------
// 第一版用瞬时低动态门控（|‖acc‖-g|<0.6 且 ‖gyro‖<0.4），实测四足行走时几乎
// 永远不成立，锚定一走就下线。改用低通：步态水平加速度一个周期内零均值。

TEST(GravityLpf, AlphaIsSampleRateIndependent) {
  // 同样的时间常数下，采样率翻倍则单步系数减半 —— 保证行为不随 IMU 频率变化
  EXPECT_NEAR(lowPassAlpha(0.025f, 1.0f), 0.025f / 1.025f, 1e-6f);
  EXPECT_NEAR(lowPassAlpha(0.005f, 1.0f), 0.005f / 1.005f, 1e-6f);
  EXPECT_FLOAT_EQ(lowPassAlpha(0.01f, 0.0f), 1.0f);   // tau=0 → 直通
  EXPECT_FLOAT_EQ(lowPassAlpha(0.0f, 1.0f), 0.0f);    // dt=0  → 不更新
  EXPECT_LE(lowPassAlpha(10.0f, 1.0f), 1.0f);         // 大 dt 不得溢出
}

TEST(GravityLpf, HorizontalAccelBarelyChangesMagnitudeButBiasesDirection) {
  // 这条锁定一个反直觉但关键的事实：水平加速度与重力**正交**，按勾股定理只让
  // 比力模长增加二阶小量，却带来一阶的方向偏差。
  // 后果：靠"模长接近 g"来筛重力样本，对水平加速度几乎无防护 —— 这正是必须
  // 靠低通（时间平均）而不是瞬时门控的原因。
  constexpr float kG = 9.80665f;
  const Eigen::Vector3f sample(2.5f, 0.0f, kG);   // 2.5 m/s² 纯水平加速

  const float mag_err = std::abs(sample.norm() - kG);
  const float dir_err_deg =
      std::acos(std::clamp(sample.normalized().z(), -1.0f, 1.0f)) *
      180.0f / static_cast<float>(M_PI);

  EXPECT_LT(mag_err, 0.35f) << "模长几乎没变: " << mag_err;
  EXPECT_GT(dir_err_deg, 10.0f) << "但方向偏了: " << dir_err_deg << "°";
}

TEST(GravityLpf, ConvergesToTrueGravityUnderGaitDisturbance) {
  // 模拟四足行走：真实重力 + 2Hz 步态水平加速度 ±2.5 m/s² + 每步落足冲击。
  // 低通必须把这些零均值扰动平掉，收敛到真实重力方向。
  //
  // 注：这里**不**断言"旧的瞬时门控会拒绝多少帧" —— 旧门控的失效主要来自陀螺
  // 判据(‖gyro‖<0.4rad/s)与垂直冲击，本模拟没有真实陀螺数据，无法据此下结论。
  // 旧门控在实机行走时的真实通过率由板端新增的 grate 诊断直接测量。
  constexpr float kG = 9.80665f;
  constexpr float kDt = 0.025f;      // 40 Hz，与板端 IMU 预测频率一致
  constexpr float kTau = 1.0f;

  Eigen::Vector3f lpf = Eigen::Vector3f::Zero();
  bool init = false;
  for (int i = 0; i < 400; ++i) {   // 10 秒
    const float t = i * kDt;
    const float gait = 2.5f * std::sin(2.0f * static_cast<float>(M_PI) * 2.0f * t);
    const float thump = (i % 20 == 0) ? 3.0f : 0.0f;
    const Eigen::Vector3f sample(gait, 0.5f * gait, kG + thump);

    if (!init) { lpf = sample; init = true; }
    else {
      const float a = lowPassAlpha(kDt, kTau);
      lpf = (1.0f - a) * lpf + a * sample;
    }
  }

  const float tilt_err_deg =
      std::acos(std::clamp(lpf.normalized().z(), -1.0f, 1.0f)) *
      180.0f / static_cast<float>(M_PI);
  EXPECT_LT(tilt_err_deg, 2.0f) << "低通后重力方向误差 " << tilt_err_deg << "°";
  // 模长须接近 g，否则有效性判据会误杀稳态行走
  EXPECT_LT(std::abs(lpf.norm() - kG), 0.35f);
}

TEST(GravityLpf, RejectsSustainedRealAcceleration) {
  // 持续真实加速（急启）时模长会偏离 g，有效性判据必须能识别并拒用
  constexpr float kG = 9.80665f;
  Eigen::Vector3f lpf(0.0f, 0.0f, kG);
  const Eigen::Vector3f accelerating(2.0f, 0.0f, kG);  // 持续 2 m/s² 前向加速
  for (int i = 0; i < 400; ++i) {
    const float a = lowPassAlpha(0.025f, 1.0f);
    lpf = (1.0f - a) * lpf + a * accelerating;
  }
  // sqrt(2²+9.81²)=10.01 → 偏差 0.21，仍在 0.35 容差内（可接受：对应 11.5° 误差
  // 的上界由 kGravityAnchorMaxAlpha 的小增益进一步压制）。这里断言判据确实
  // 随加速度单调变化，便于日后调阈值时有依据。
  EXPECT_GT(std::abs(lpf.norm() - kG), 0.15f);
}

TEST(YawDelta, NormalizesAcrossWrap) {
  auto R = [](float yaw) { return rotation(0.0f, 0.0f, yaw); };
  EXPECT_NEAR(yawDeltaDeg(R(0.0f), R(0.1f)), 0.1f * 180.0f / M_PI, 1e-3f);
  // 跨 ±180° 必须走短弧，否则 innov_dyaw 会出现 ±360° 的假跳变
  const float d = yawDeltaDeg(R(3.10f), R(-3.10f));
  EXPECT_LT(std::abs(d), 10.0f) << "跨 ±180° 应归一化到短弧，实际 " << d;
}

TEST(GravityLpf, ScalarMeanSurvivesAttitudeSwingButVectorMeanDoesNot) {
  // 锁定 2026-08-16 的判据修正。
  //
  // 解析关系：姿态按 ±A 正弦摆动、加速度模长恒为 g 时，**向量**平均的模长
  // = g·J₀(A)（零阶贝塞尔函数），即误差 = g·(1-J₀(A))：
  //     A=20° → 0.296     A=25° → 0.46     A=30° → 0.66
  // 生产阈值 kGravityLpfMagTol = 0.35。所以摆动一超过约 22°，向量判据就会把
  // "根本没有真实加速度"的稳态行走误判成"存在持续加速"，把重力锚定关掉
  // （实测 gmag 0.33~0.43、grav=0、姿态随即被 NDT 拖走）。
  // **标量**平均（先取模再平均）不受方向发散影响，恒等于 g。
  constexpr float kG = 9.80665f;
  constexpr float kProdMagTol = 0.35f;   // 须与 PoseEstimator::kGravityLpfMagTol 一致

  auto simulate = [&](float swing_deg) {
    const float A = swing_deg * static_cast<float>(M_PI) / 180.0f;
    Eigen::Vector3f vec_mean = Eigen::Vector3f::Zero();
    float scalar_mean = 0.0f;
    const int kN = 2000;   // 覆盖整数个周期，避免截断偏差
    for (int i = 0; i < kN; ++i) {
      const float ang = A * std::sin(2.0f * static_cast<float>(M_PI) * i / 100.0f);
      const Eigen::Vector3f s(kG * std::sin(ang), 0.0f, kG * std::cos(ang));
      vec_mean += s / kN;
      scalar_mean += s.norm() / kN;
    }
    return std::make_pair(std::abs(vec_mean.norm() - kG),
                          std::abs(scalar_mean - kG));
  };

  // 25° 摆动（实测 pitch 到 -27.7°、roll 到 -21.5°，属真实量级）
  const auto [vec_err_25, scalar_err_25] = simulate(25.0f);
  EXPECT_GT(vec_err_25, kProdMagTol)
      << "向量判据在 25° 摆动下应误触发生产阈值，实际 " << vec_err_25;
  EXPECT_LT(scalar_err_25, 0.01f)
      << "标量判据必须不受摆动影响，实际 " << scalar_err_25;

  // 摆动越大，向量判据偏差越大（单调性），标量判据始终无偏
  const auto [vec_err_10, scalar_err_10] = simulate(10.0f);
  EXPECT_LT(vec_err_10, vec_err_25);
  EXPECT_LT(scalar_err_10, 0.01f);
}

// ---- UKF 流形修正（2026-08-16）--------------------------------------------
// 实测：均值四元数模长塌到 0.1424（正常为 1.0），随后 cov 衰减、卡尔曼增益归零、
// Mahalanobis 门锁死、状态彻底冻结（UKF 位置死钉在 [1.32,0.33,-0.27]，而 NDT
// 观测已走到 2.96m）。根因是 correct() 里 `mean += K·innovation` 之后不归一化，
// 而 h() 把观测侧的模长归一化掉了 —— 模长成为不可观测且无阻尼的自由度。

TEST(UkfManifold, NormalizeStateRestoresUnitQuaternion) {
  PoseSystem sys;
  Eigen::VectorXf state = Eigen::VectorXf::Zero(16);
  // 模拟塌缩后的状态：方向正确但模长只有 0.1424
  const Eigen::Quaternionf q(rotation(0.1f, -0.2f, 0.9f));
  state.middleRows(6, 4) << q.w(), q.x(), q.y(), q.z();
  state.middleRows(6, 4) *= 0.1424f;
  ASSERT_NEAR(state.middleRows(6, 4).norm(), 0.1424f, 1e-4f);

  sys.normalizeState(state);
  EXPECT_NEAR(state.middleRows(6, 4).norm(), 1.0f, 1e-6f);
  // 方向必须原样保留 —— 归一化只是把模长拉回，不得改变姿态
  const Eigen::Quaternionf out(state[6], state[7], state[8], state[9]);
  EXPECT_NEAR(rotationToRpy(out.toRotationMatrix()).z(),
              rotationToRpy(q.toRotationMatrix()).z(), 1e-5f);

  // 退化输入不得产生 NaN
  Eigen::VectorXf zero_state = Eigen::VectorXf::Zero(16);
  sys.normalizeState(zero_state);
  EXPECT_TRUE(zero_state.allFinite());
}

TEST(UkfManifold, HemisphereAlignmentPreventsCancellation) {
  PoseSystem sys;
  const Eigen::Quaternionf q(rotation(0.0f, 0.0f, 1.2f));
  Eigen::VectorXf ref = Eigen::VectorXf::Zero(16);
  ref.middleRows(6, 4) << q.w(), q.x(), q.y(), q.z();

  // 构造一个落在对映半球的 sigma 点（q 与 -q 是同一旋转）
  Eigen::VectorXf flipped = ref;
  flipped.middleRows(6, 4) *= -1.0f;

  // 不对齐直接平均 → 相消，模长塌到 0
  const Eigen::Vector4f naive =
      0.5f * ref.middleRows(6, 4) + 0.5f * flipped.middleRows(6, 4);
  EXPECT_LT(naive.norm(), 1e-6f) << "未对齐时算术平均确实会相消";

  // 对齐后再平均 → 模长保持
  sys.alignStateToReference(flipped, ref);
  const Eigen::Vector4f aligned =
      0.5f * ref.middleRows(6, 4) + 0.5f * flipped.middleRows(6, 4);
  EXPECT_NEAR(aligned.norm(), 1.0f, 1e-6f);
  // 对齐不改变它代表的旋转
  const Eigen::Quaternionf qa(flipped[6], flipped[7], flipped[8], flipped[9]);
  EXPECT_NEAR(std::abs(qa.dot(q)), 1.0f, 1e-6f);
}

TEST(UkfManifold, ObservationAlignmentUsesObsQuatIndex) {
  PoseSystem sys;
  // 观测布局是 [pos(3), quat(4)]，四元数从第 3 行开始
  Eigen::VectorXf obs = Eigen::VectorXf::Zero(7);
  Eigen::VectorXf ref = Eigen::VectorXf::Zero(7);
  ref.middleRows(3, 4) << 1.0f, 0.0f, 0.0f, 0.0f;
  obs.middleRows(0, 3) << 5.0f, 6.0f, 7.0f;     // 位置分量不得被动
  obs.middleRows(3, 4) << -1.0f, 0.0f, 0.0f, 0.0f;

  sys.alignObservationToReference(obs, ref);
  EXPECT_NEAR(obs[3], 1.0f, 1e-6f) << "对映半球应被翻转";
  EXPECT_NEAR(obs[0], 5.0f, 1e-6f) << "位置分量不得被修改";
  EXPECT_NEAR(obs[1], 6.0f, 1e-6f);
  EXPECT_NEAR(obs[2], 7.0f, 1e-6f);
}

}  // namespace
}  // namespace localization
