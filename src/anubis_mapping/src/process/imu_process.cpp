#include "process/imu_process.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "process/imu_init_gate.h"
#include "lio_time_guard.h"

const bool time_list(anubis_mapping::PointType& x, anubis_mapping::PointType& y)
{
    return (x.curvature < y.curvature);
}

ImuProcess::ImuProcess()
    : b_first_frame_(true),
      imu_need_init_(true),
      time_anchor_valid_(false),
      start_timestamp_(-1),
      last_lidar_end_time_(-1.0),
      init_duration_s_(3.0)
{
    init_iter_num   = 1;
    Q               = process_noise_cov();
    cov_acc         = anubis_mapping::Vec3d(0.1, 0.1, 0.1);
    cov_gyr         = anubis_mapping::Vec3d(0.1, 0.1, 0.1);
    cov_bias_gyr    = anubis_mapping::Vec3d(0.0001, 0.0001, 0.0001);
    cov_bias_acc    = anubis_mapping::Vec3d(0.0001, 0.0001, 0.0001);
    mean_acc        = anubis_mapping::Vec3d(0, 0, -1.0);
    mean_gyr        = anubis_mapping::Vec3d(0, 0, 0);
    corrected_mean_acc_g_ = anubis_mapping::Zero3d;
    corrected_mean_acc_norm_g_ = 0.0;
    acc_scale_factor_m_s2_per_g_ = 0.0;
    angvel_last     = anubis_mapping::Zero3d;
    acc_s_last      = anubis_mapping::Zero3d;
    Lidar_T_wrt_IMU = anubis_mapping::Zero3d;
    Lidar_R_wrt_IMU = anubis_mapping::Eye3d;
    last_imu_.reset(new anubis_mapping::ImuMessage());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()
{
    // ROS_WARN("Reset ImuProcess");
    mean_acc         = anubis_mapping::Vec3d(0, 0, -1.0);
    mean_gyr         = anubis_mapping::Vec3d(0, 0, 0);
    corrected_mean_acc_g_ = anubis_mapping::Zero3d;
    corrected_mean_acc_norm_g_ = 0.0;
    acc_scale_factor_m_s2_per_g_ = 0.0;
    angvel_last      = anubis_mapping::Zero3d;
    acc_s_last       = anubis_mapping::Zero3d;
    imu_need_init_   = true;
    time_anchor_valid_ = false;
    start_timestamp_ = -1;
    last_lidar_end_time_ = -1.0;
    init_iter_num    = 1;
    v_imu_.clear();
    IMUpose.clear();
    last_imu_.reset(new anubis_mapping::ImuMessage());
    cur_pcl_un_.reset(new anubis_mapping::PointCloudType());
    diagnostics_ = Diagnostics{};
}

void ImuProcess::set_extrinsic(const MD(4, 4) & T)
{
    Lidar_T_wrt_IMU = T.block<3, 1>(0, 3);
    Lidar_R_wrt_IMU = T.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const anubis_mapping::Vec3d& transl)
{
    Lidar_T_wrt_IMU = transl;
    Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const anubis_mapping::Vec3d& transl, const anubis_mapping::Mat3d& rot)
{
    Lidar_T_wrt_IMU = transl;
    Lidar_R_wrt_IMU = rot;
}

void ImuProcess::reset()
{
    mean_acc         = anubis_mapping::Vec3d(0, 0, -1.0);
    mean_gyr         = anubis_mapping::Vec3d(0, 0, 0);
    corrected_mean_acc_g_ = anubis_mapping::Zero3d;
    corrected_mean_acc_norm_g_ = 0.0;
    acc_scale_factor_m_s2_per_g_ = 0.0;
    angvel_last      = anubis_mapping::Zero3d;
    acc_s_last       = anubis_mapping::Zero3d;
    b_first_frame_   = true;
    imu_need_init_   = true;
    time_anchor_valid_ = false;
    start_timestamp_ = -1;
    last_lidar_end_time_ = -1.0;
    init_iter_num    = 1;
    init_attempt_count_ = 0;
    v_imu_.clear();
    IMUpose.clear();
    last_imu_.reset(new anubis_mapping::ImuMessage());
    cur_pcl_un_.reset(new anubis_mapping::PointCloudType());
    Q            = process_noise_cov();
    cov_acc      = anubis_mapping::Vec3d(0.1, 0.1, 0.1);
    cov_gyr      = anubis_mapping::Vec3d(0.1, 0.1, 0.1);
    cov_bias_gyr = anubis_mapping::Vec3d(0.0001, 0.0001, 0.0001);
    cov_bias_acc = anubis_mapping::Vec3d(0.0001, 0.0001, 0.0001);
    diagnostics_ = Diagnostics{};
}

void ImuProcess::set_gyr_cov(const anubis_mapping::Vec3d& scaler)
{
    cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const anubis_mapping::Vec3d& scaler)
{
    cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const anubis_mapping::Vec3d& b_g)
{
    cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const anubis_mapping::Vec3d& b_a)
{
    cov_bias_acc = b_a;
}

void ImuProcess::set_init_duration(double seconds)
{
    if (std::isfinite(seconds) && seconds > 0.0)
    {
        init_duration_s_ = seconds;
    }
}

void ImuProcess::set_init_stationary_thresholds(
    double max_acc_std_g,
    double max_gyr_std_rad_s,
    double max_mean_gyr_rad_s)
{
    if (std::isfinite(max_acc_std_g) && max_acc_std_g > 0.0)
    {
        init_max_acc_std_g_ = max_acc_std_g;
    }
    if (std::isfinite(max_gyr_std_rad_s) && max_gyr_std_rad_s > 0.0)
    {
        init_max_gyr_std_rad_s_ = max_gyr_std_rad_s;
    }
    if (std::isfinite(max_mean_gyr_rad_s) && max_mean_gyr_rad_s > 0.0)
    {
        init_max_mean_gyr_rad_s_ = max_mean_gyr_rad_s;
    }
}

void ImuProcess::set_initial_biases(
    const anubis_mapping::Vec3d& accel_bias,
    const anubis_mapping::Vec3d& gyro_bias)
{
    if (accel_bias.allFinite())
    {
        initial_ba_ = accel_bias;
    }
    if (gyro_bias.allFinite())
    {
        initial_bg_ = gyro_bias;
    }
}

void ImuProcess::set_initial_gravity_correction_rpy(
    const anubis_mapping::Vec3d& rpy_rad)
{
    if (rpy_rad.allFinite())
    {
        gravity_correction_rpy_rad_ = rpy_rad;
    }
}

void ImuProcess::set_init_acc_norm_error(double max_error_g)
{
    if (std::isfinite(max_error_g) && max_error_g > 0.0)
    {
        init_max_acc_norm_error_g_ = max_error_g;
    }
}

void ImuProcess::reset_time_anchor()
{
    // Keep the current filter state and bias/gravity estimate, but discard
    // only history that carries timestamps across a sensor blackout.
    last_imu_.reset();
    v_imu_.clear();
    IMUpose.clear();
    v_rot_pcl_.clear();
    last_lidar_end_time_ = -1.0;
    start_timestamp_ = -1.0;
    angvel_last = anubis_mapping::Zero3d;
    acc_s_last = anubis_mapping::Zero3d;
    time_anchor_valid_ = false;
    diagnostics_ = Diagnostics{};
}

const ImuProcess::Diagnostics& ImuProcess::diagnostics() const
{
    return diagnostics_;
}

void ImuProcess::IMU_init(const anubis_mapping::MeasureGroup& meas, esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state, int& N)
{
    anubis_mapping::Vec3d cur_acc, cur_gyr;

    if (b_first_frame_)
    {
        Reset();
        N                   = 1;
        b_first_frame_      = false;
        const auto& imu_acc = meas.imu.front()->acc;
        const auto& gyr_acc = meas.imu.front()->gyr;
        mean_acc << imu_acc.x(), imu_acc.y(), imu_acc.z();
        mean_gyr << gyr_acc.x(), gyr_acc.y(), gyr_acc.z();
        first_lidar_time = meas.lidar_beg_time;
    }

    for (const auto& imu : meas.imu)
    {
        const auto& imu_acc = imu->acc;
        const auto& gyr_acc = imu->gyr;
        cur_acc << imu_acc.x(), imu_acc.y(), imu_acc.z();
        cur_gyr << gyr_acc.x(), gyr_acc.y(), gyr_acc.z();

        mean_acc += (cur_acc - mean_acc) / N;
        mean_gyr += (cur_gyr - mean_gyr) / N;

        cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
        cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

        N++;
    }
    state_ikfom init_state = kf_state.get_x();

    Eigen::Vector3d state_ba_m_s2;
    double corrected_mean_norm_g = 0.0;
    double acc_scale_m_s2_per_g = 0.0;
    const bool accel_terms_valid =
        anubis_mapping::derive_imu_accel_initialization_terms(
            mean_acc, initial_ba_, anubis_mapping::G_m_s2,
            corrected_mean_acc_g_, state_ba_m_s2,
            corrected_mean_norm_g, acc_scale_m_s2_per_g);
    Eigen::Vector3d state_bg_rad_s;
    const bool gyro_terms_valid = anubis_mapping::compose_imu_gyro_bias_rad_s(
        mean_gyr, initial_bg_, state_bg_rad_s);
    if (!accel_terms_valid || !gyro_terms_valid)
    {
        corrected_mean_acc_g_ = anubis_mapping::Zero3d;
        corrected_mean_acc_norm_g_ = 0.0;
        acc_scale_factor_m_s2_per_g_ = 0.0;
        fprintf(stderr,
            "[IMU INIT INVALID] mean_acc_g=[%.9f %.9f %.9f] "
            "init_ba_g=[%.9f %.9f %.9f] mean_gyr_rad_s=[%.9f %.9f %.9f] "
            "init_bg_rad_s=[%.9f %.9f %.9f]\n",
            mean_acc.x(), mean_acc.y(), mean_acc.z(),
            initial_ba_.x(), initial_ba_.y(), initial_ba_.z(),
            mean_gyr.x(), mean_gyr.y(), mean_gyr.z(),
            initial_bg_.x(), initial_bg_.y(), initial_bg_.z());
        return;
    }
    corrected_mean_acc_norm_g_ = corrected_mean_norm_g;
    acc_scale_factor_m_s2_per_g_ = acc_scale_m_s2_per_g;

    Eigen::Vector3d gravity_target;
    if (!anubis_mapping::make_imu_gravity_alignment_target_g(
            corrected_mean_acc_g_, gravity_target))
    {
        corrected_mean_acc_g_ = anubis_mapping::Zero3d;
        corrected_mean_acc_norm_g_ = 0.0;
        acc_scale_factor_m_s2_per_g_ = 0.0;
        fprintf(stderr,
            "[IMU INIT INVALID] corrected gravity vector is degenerate\n");
        return;
    }
    const Eigen::Matrix3d gravity_alignment =
        Eigen::Quaterniond::FromTwoVectors(
            corrected_mean_acc_g_, gravity_target).toRotationMatrix();
    const Eigen::Matrix3d gravity_correction =
        (Eigen::AngleAxisd(gravity_correction_rpy_rad_.z(), Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(gravity_correction_rpy_rad_.y(), Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(gravity_correction_rpy_rad_.x(), Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    init_state.rot = gravity_correction * gravity_alignment;
    Eigen::Vector3d g(0.0, 0.0, -anubis_mapping::G_m_s2);
    init_state.grav = S2(g);

    init_state.bg           = state_bg_rad_s;
    init_state.ba           = state_ba_m_s2;
    init_state.offset_T_L_I = Lidar_T_wrt_IMU;
    init_state.offset_R_L_I = Lidar_R_wrt_IMU;
    kf_state.change_x(init_state);
    time_anchor_valid_ = true;

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
    init_P.setIdentity();
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = 0.00001;
    kf_state.change_P(init_P);
    last_imu_ = meas.imu.back();
}

void ImuProcess::UndistortPcl(
    const anubis_mapping::MeasureGroup& meas, esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state, anubis_mapping::PointCloudType& pcl_out)
{
    diagnostics_                = Diagnostics{};
    diagnostics_.valid         = true;
    diagnostics_.lidar_beg_time = meas.lidar_beg_time;
    diagnostics_.lidar_end_time = meas.lidar_end_time;
    diagnostics_.imu_count      = meas.imu.size();
    diagnostics_.last_lidar_end_before = last_lidar_end_time_;
    if (!meas.imu.empty())
    {
        diagnostics_.imu_beg_time = meas.imu.front()->timestamp;
        diagnostics_.imu_end_time = meas.imu.back()->timestamp;
    }

    const auto record_predict_dt = [this](double predict_dt)
    {
        if (diagnostics_.predict_steps == 0)
        {
            diagnostics_.predict_dt_min = predict_dt;
            diagnostics_.predict_dt_max = predict_dt;
        }
        else
        {
            diagnostics_.predict_dt_min = std::min(diagnostics_.predict_dt_min, predict_dt);
            diagnostics_.predict_dt_max = std::max(diagnostics_.predict_dt_max, predict_dt);
        }
        diagnostics_.predict_dt_sum += predict_dt;
        ++diagnostics_.predict_steps;
        if (predict_dt <= 0.0)
        {
            ++diagnostics_.nonpositive_dt_count;
        }
    };

    /*** add the imu of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu;
    if (anubis_mapping::can_reuse_imu_time_anchor(
            time_anchor_valid_, static_cast<bool>(last_imu_)))
    {
        v_imu.push_front(last_imu_);
    }
    const double& imu_beg_time = v_imu.front()->timestamp;
    const double& imu_end_time = v_imu.back()->timestamp;
    const double& pcl_beg_time = meas.lidar_beg_time;
    const double& pcl_end_time = meas.lidar_end_time;
    const double scan_duration = pcl_end_time - pcl_beg_time;

    /*** sort point clouds by offset time ***/
    pcl_out = *(meas.lidar);
    bool have_point_time = false;
    for (const auto& point : pcl_out.points)
    {
        const double point_time = point.curvature / 1000.0;
        if (!std::isfinite(point_time))
        {
            ++diagnostics_.point_time_nonfinite_count;
            continue;
        }
        if (!have_point_time)
        {
            diagnostics_.point_time_min = point_time;
            diagnostics_.point_time_max = point_time;
            have_point_time = true;
        }
        else
        {
            diagnostics_.point_time_min = std::min(
                diagnostics_.point_time_min, point_time);
            diagnostics_.point_time_max = std::max(
                diagnostics_.point_time_max, point_time);
        }
        if (point_time < -1e-6 || point_time > scan_duration + 1e-6)
        {
            ++diagnostics_.point_time_outside_scan_count;
        }
        ++diagnostics_.point_time_valid_count;
    }
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
    // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

    /*** Initialize IMU pose ***/
    state_ikfom imu_state = kf_state.get_x();
    IMUpose.clear();
    IMUpose.push_back(
        anubis_mapping::set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

    /*** forward propagation at each imu point ***/
    anubis_mapping::Vec3d angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    anubis_mapping::Mat3d R_imu;

    double dt = 0;
    double world_acc_z_sum = 0.0;

    input_ikfom in;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
    {
        auto&& head = *(it_imu);
        auto&& tail = *(it_imu + 1);

        double tail_stamp = tail->timestamp;
        double head_stamp = head->timestamp;

        if (tail_stamp < last_lidar_end_time_)
        {
            ++diagnostics_.skipped_imu_pair_count;
            continue;
        }

        angvel_avr << 0.5 * (head->gyr.x() + tail->gyr.x()), 0.5 * (head->gyr.y() + tail->gyr.y()), 0.5 * (head->gyr.z() + tail->gyr.z());
        acc_avr << 0.5 * (head->acc.x() + tail->acc.x()), 0.5 * (head->acc.y() + tail->acc.y()), 0.5 * (head->acc.z() + tail->acc.z());

        diagnostics_.acc_scale_factor = acc_scale_factor_m_s2_per_g_;
        acc_avr = acc_avr * diagnostics_.acc_scale_factor;  // - state_inout.ba;

        if (head_stamp < last_lidar_end_time_)
        {
            dt = tail_stamp - last_lidar_end_time_;
        }
        else
        {
            dt = tail_stamp - head_stamp;
        }

        in.acc                         = acc_avr;
        in.gyro                        = angvel_avr;
        Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
        Q.block<3, 3>(3, 3).diagonal() = cov_acc;
        Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
        Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
        kf_state.predict(dt, Q, in);
        record_predict_dt(dt);

        /* save the poses at each IMU measurements */
        imu_state   = kf_state.get_x();
        angvel_last = angvel_avr - imu_state.bg;
        acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
        for (int i = 0; i < 3; i++)
        {
            acc_s_last[i] += imu_state.grav[i];
        }
        diagnostics_.last_acc_input = acc_avr;
        diagnostics_.last_gyr_input = angvel_avr;
        diagnostics_.last_world_acc = acc_s_last;
        if (diagnostics_.world_acc_sample_count == 0U)
        {
            diagnostics_.world_acc_z_min = acc_s_last.z();
            diagnostics_.world_acc_z_max = acc_s_last.z();
        }
        else
        {
            diagnostics_.world_acc_z_min = std::min(
                diagnostics_.world_acc_z_min, acc_s_last.z());
            diagnostics_.world_acc_z_max = std::max(
                diagnostics_.world_acc_z_max, acc_s_last.z());
        }
        world_acc_z_sum += acc_s_last.z();
        ++diagnostics_.world_acc_sample_count;
        if (dt > 0.0)
        {
            diagnostics_.world_acc_z_dt_integral += acc_s_last.z() * dt;
            diagnostics_.positive_predict_dt_sum += dt;
        }
        double&& offs_t = tail_stamp - pcl_beg_time;
        IMUpose.push_back(
            anubis_mapping::set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt          = note * (pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q, in);
    diagnostics_.end_extrapolation_dt = dt;
    record_predict_dt(dt);
    if (diagnostics_.world_acc_sample_count > 0U)
    {
        diagnostics_.world_acc_z_mean = world_acc_z_sum /
            static_cast<double>(diagnostics_.world_acc_sample_count);
    }

    imu_state            = kf_state.get_x();
    last_imu_            = meas.imu.back();
    time_anchor_valid_   = true;
    last_lidar_end_time_ = pcl_end_time;

    /*** undistort each lidar point (backward propagation) ***/
    if (pcl_out.points.begin() == pcl_out.points.end())
        return;
    double              deskew_dz_sum    = 0.0;
    double              deskew_dz_sq_sum = 0.0;
    double              deskew_rotation_dz_sum = 0.0;
    double              deskew_rotation_dz_sq_sum = 0.0;
    double              deskew_translation_dz_sum = 0.0;
    double              deskew_translation_dz_sq_sum = 0.0;
    std::vector<double> deskew_abs_dz_samples;
    std::vector<double> deskew_rotation_abs_dz_samples;
    std::vector<double> deskew_translation_abs_dz_samples;
    deskew_abs_dz_samples.reserve(pcl_out.points.size() / 16U + 1U);
    deskew_rotation_abs_dz_samples.reserve(pcl_out.points.size() / 16U + 1U);
    deskew_translation_abs_dz_samples.reserve(pcl_out.points.size() / 16U + 1U);
    auto it_pcl = pcl_out.points.end() - 1;
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
        auto head = it_kp - 1;
        auto tail = it_kp;
        R_imu << MAT_FROM_ARRAY(head->rot);
        vel_imu << VEC_FROM_ARRAY(head->vel);
        pos_imu << VEC_FROM_ARRAY(head->pos);
        acc_imu << VEC_FROM_ARRAY(tail->acc);
        angvel_avr << VEC_FROM_ARRAY(tail->gyr);

        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
        {
            dt = it_pcl->curvature / double(1000) - head->offset_time;

            anubis_mapping::Mat3d R_i(R_imu * Exp(angvel_avr, dt));

            anubis_mapping::Vec3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            anubis_mapping::Vec3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
            anubis_mapping::Vec3d P_compensate =
                imu_state.offset_R_L_I.conjugate()
                * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei)
                    - imu_state.offset_T_L_I);
            const anubis_mapping::Vec3d P_rotation_only =
                imu_state.offset_R_L_I.conjugate()
                * (imu_state.rot.conjugate()
                    * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I))
                    - imu_state.offset_T_L_I);

            const double deskew_dz = P_compensate(2) - P_i(2);
            const double deskew_rotation_dz = P_rotation_only(2) - P_i(2);
            const double deskew_translation_dz =
                P_compensate(2) - P_rotation_only(2);
            deskew_dz_sum += deskew_dz;
            deskew_dz_sq_sum += deskew_dz * deskew_dz;
            deskew_rotation_dz_sum += deskew_rotation_dz;
            deskew_rotation_dz_sq_sum +=
                deskew_rotation_dz * deskew_rotation_dz;
            deskew_translation_dz_sum += deskew_translation_dz;
            deskew_translation_dz_sq_sum +=
                deskew_translation_dz * deskew_translation_dz;
            if (std::abs(deskew_dz) >= diagnostics_.deskew_dz_abs_max)
            {
                diagnostics_.deskew_dz_abs_max = std::abs(deskew_dz);
                diagnostics_.deskew_peak_point_time =
                    it_pcl->curvature / double(1000);
                diagnostics_.deskew_peak_input_z = P_i(2);
                diagnostics_.deskew_peak_output_z = P_compensate(2);
            }
            diagnostics_.deskew_rotation_dz_abs_max = std::max(
                diagnostics_.deskew_rotation_dz_abs_max,
                std::abs(deskew_rotation_dz));
            diagnostics_.deskew_translation_dz_abs_max = std::max(
                diagnostics_.deskew_translation_dz_abs_max,
                std::abs(deskew_translation_dz));
            if ((diagnostics_.deskew_point_count % 16U) == 0U)
            {
                deskew_abs_dz_samples.push_back(std::abs(deskew_dz));
                deskew_rotation_abs_dz_samples.push_back(
                    std::abs(deskew_rotation_dz));
                deskew_translation_abs_dz_samples.push_back(
                    std::abs(deskew_translation_dz));
            }
            ++diagnostics_.deskew_point_count;

            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == pcl_out.points.begin())
                break;
        }
    }

    if (diagnostics_.deskew_point_count > 0U)
    {
        const double count = static_cast<double>(diagnostics_.deskew_point_count);
        diagnostics_.deskew_dz_mean = deskew_dz_sum / count;
        diagnostics_.deskew_dz_rms = std::sqrt(deskew_dz_sq_sum / count);
        diagnostics_.deskew_rotation_dz_mean =
            deskew_rotation_dz_sum / count;
        diagnostics_.deskew_rotation_dz_rms = std::sqrt(
            deskew_rotation_dz_sq_sum / count);
        diagnostics_.deskew_translation_dz_mean =
            deskew_translation_dz_sum / count;
        diagnostics_.deskew_translation_dz_rms = std::sqrt(
            deskew_translation_dz_sq_sum / count);
    }
    const auto absolute_p95 = [](std::vector<double>& samples)
    {
        if (samples.empty())
        {
            return 0.0;
        }
        const std::size_t p95_index = static_cast<std::size_t>(
            std::floor(0.95 * static_cast<double>(samples.size() - 1U)));
        std::nth_element(
            samples.begin(),
            samples.begin() + static_cast<std::ptrdiff_t>(p95_index),
            samples.end());
        return samples[p95_index];
    };
    diagnostics_.deskew_dz_abs_p95 = absolute_p95(deskew_abs_dz_samples);
    diagnostics_.deskew_rotation_dz_abs_p95 =
        absolute_p95(deskew_rotation_abs_dz_samples);
    diagnostics_.deskew_translation_dz_abs_p95 =
        absolute_p95(deskew_translation_abs_dz_samples);
}

void ImuProcess::Process(
    const anubis_mapping::MeasureGroup& meas, esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state, anubis_mapping::CloudPtr cur_pcl_un_)
{
    double t1, t2, t3;
    t1 = omp_get_wtime();
    diagnostics_ = Diagnostics{};
    cur_pcl_un_->clear();

    if (meas.imu.empty())
    {
        return;
    };
    assert(meas.lidar != nullptr);

    if (imu_need_init_)
    {
        IMU_init(meas, kf_state, init_iter_num);

        imu_need_init_ = true;

        last_imu_ = meas.imu.back();
        time_anchor_valid_ = true;

        state_ikfom imu_state = kf_state.get_x();
        const double init_elapsed_s = meas.imu.empty()
            ? 0.0
            : std::max(0.0, meas.imu.back()->timestamp - first_lidar_time);
        if (init_elapsed_s >= init_duration_s_ ||
            init_iter_num > anubis_mapping::MAX_INI_COUNT)
        {
            const Eigen::Vector3d acc_std =
                cov_acc.cwiseMax(0.0).cwiseSqrt();
            const Eigen::Vector3d gyr_std =
                cov_gyr.cwiseMax(0.0).cwiseSqrt();
            const Eigen::Vector3d init_rpy_deg = SO3ToEuler(imu_state.rot);
            Eigen::Vector3d corrected_mean_acc_g;
            Eigen::Vector3d state_ba_m_s2;
            double corrected_mean_norm_g = 0.0;
            double acc_scale_m_s2_per_g = 0.0;
            const bool accel_terms_valid =
                anubis_mapping::derive_imu_accel_initialization_terms(
                    mean_acc, initial_ba_, anubis_mapping::G_m_s2,
                    corrected_mean_acc_g, state_ba_m_s2,
                    corrected_mean_norm_g, acc_scale_m_s2_per_g);
            ++init_attempt_count_;
            const bool stationary = anubis_mapping::is_imu_init_window_stationary(
                acc_std, gyr_std, mean_gyr,
                init_max_acc_std_g_, init_max_gyr_std_rad_s_,
                init_max_mean_gyr_rad_s_) &&
                accel_terms_valid &&
                anubis_mapping::is_imu_acc_norm_valid(
                    corrected_mean_norm_g, init_max_acc_norm_error_g_);
            if (!stationary)
            {
                fprintf(stderr,
                    "[IMU INIT RETRY] attempt=%zu samples=%d duration_s=%.6f "
                    "reason=motion_detected acc_std_max_g=%.9f limit=%.9f "
                    "gyr_std_max_rad_s=%.9f limit=%.9f "
                    "mean_gyr_norm_rad_s=%.9f limit=%.9f "
                    "acc_norm_g=%.9f acc_norm_raw_g=%.9f "
                    "acc_norm_corrected_g=%.9f "
                    "norm_error_corrected_g=%.9f norm_limit_g=%.9f "
                    "init_ba_g=[%.9f %.9f %.9f]\n",
                    init_attempt_count_, init_iter_num, init_elapsed_s,
                    acc_std.maxCoeff(), init_max_acc_std_g_,
                    gyr_std.maxCoeff(), init_max_gyr_std_rad_s_,
                    mean_gyr.norm(), init_max_mean_gyr_rad_s_, mean_acc.norm(),
                    mean_acc.norm(), corrected_mean_norm_g,
                    std::abs(corrected_mean_norm_g - 1.0),
                    init_max_acc_norm_error_g_,
                    initial_ba_.x(), initial_ba_.y(), initial_ba_.z());
                fflush(stderr);
                b_first_frame_ = true;
                return;
            }
            fprintf(stderr,
                "[IMU INIT] attempts=%zu static_gate=pass samples=%d "
                "duration_s=%.6f target_duration_s=%.3f "
                "acc_norm_raw=%.9f acc_norm_raw_g=%.9f "
                "acc_norm_corrected_g=%.9f "
                "mean_acc_raw=[%.9f %.9f %.9f] "
                "mean_acc_raw_g=[%.9f %.9f %.9f] "
                "mean_acc_corrected_g=[%.9f %.9f %.9f] "
                "acc_std_raw=[%.9f %.9f %.9f] mean_gyr=[%.9f %.9f %.9f] "
                "gyr_std=[%.9f %.9f %.9f] gravity_align_rpy_deg=[%.6f %.6f %.6f] "
                "init_bg=[%.9f %.9f %.9f] "
                "init_bg_state_rad_s=[%.9f %.9f %.9f] "
                "init_ba=[%.9f %.9f %.9f] "
                "init_ba_config_g=[%.9f %.9f %.9f] "
                "state_ba_m_s2=[%.9f %.9f %.9f] "
                "acc_scale_m_s2_per_g=%.9f\n",
                init_attempt_count_, init_iter_num, init_elapsed_s,
                init_duration_s_, mean_acc.norm(), mean_acc.norm(),
                corrected_mean_norm_g,
                mean_acc.x(), mean_acc.y(), mean_acc.z(),
                mean_acc.x(), mean_acc.y(), mean_acc.z(),
                corrected_mean_acc_g.x(), corrected_mean_acc_g.y(),
                corrected_mean_acc_g.z(),
                acc_std.x(), acc_std.y(), acc_std.z(),
                mean_gyr.x(), mean_gyr.y(), mean_gyr.z(),
                gyr_std.x(), gyr_std.y(), gyr_std.z(),
                init_rpy_deg.x(), init_rpy_deg.y(), init_rpy_deg.z(),
                imu_state.bg[0], imu_state.bg[1], imu_state.bg[2],
                imu_state.bg[0], imu_state.bg[1], imu_state.bg[2],
                state_ba_m_s2.x(), state_ba_m_s2.y(), state_ba_m_s2.z(),
                initial_ba_.x(), initial_ba_.y(), initial_ba_.z(),
                state_ba_m_s2.x(), state_ba_m_s2.y(), state_ba_m_s2.z(),
                acc_scale_m_s2_per_g);
            fprintf(stderr,
                "[IMU INIT PARAM] acc_cov=[%.9f %.9f %.9f] "
                "gyr_cov=[%.9f %.9f %.9f] b_acc_cov=[%.9f %.9f %.9f] "
                "b_gyr_cov=[%.9f %.9f %.9f]\n",
                cov_acc_scale.x(), cov_acc_scale.y(), cov_acc_scale.z(),
                cov_gyr_scale.x(), cov_gyr_scale.y(), cov_gyr_scale.z(),
                cov_bias_acc.x(), cov_bias_acc.y(), cov_bias_acc.z(),
                cov_bias_gyr.x(), cov_bias_gyr.y(), cov_bias_gyr.z());
            fflush(stderr);
            cov_acc *= pow(acc_scale_m_s2_per_g, 2);
            imu_need_init_ = false;

            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;
            fprintf(stderr, "IMU Initial Done\n");
        }

        return;
    }

    UndistortPcl(meas, kf_state, *cur_pcl_un_);

    t2 = omp_get_wtime();
    t3 = omp_get_wtime();
}
