/**
 * @file imu_process.h
 * @brief
 * @author Liuzhao Li (liliuzhao@jushenzhiren.com)
 * @version 1.0
 * @date 2025-07-31
 * @copyright Copyright (C) 2025 具身智人(北京)科技有限公司
 */

#pragma once
#include <cstddef>
#include <deque>
#include <mutex>
#include "so3_math.h"
#include "common.h"
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "use_ikfom.h"

const bool time_list(anubis_mapping::PointType& x, anubis_mapping::PointType& y);

class ImuProcess
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Diagnostics
    {
        bool        valid                 = false;
        double      lidar_beg_time        = 0.0;
        double      lidar_end_time        = 0.0;
        double      imu_beg_time          = 0.0;
        double      imu_end_time          = 0.0;
        std::size_t imu_count             = 0;
        std::size_t predict_steps         = 0;
        std::size_t nonpositive_dt_count  = 0;
        double      predict_dt_sum        = 0.0;
        double      predict_dt_min        = 0.0;
        double      predict_dt_max        = 0.0;
        double      end_extrapolation_dt  = 0.0;
        double      last_lidar_end_before = -1.0;
        double      acc_scale_factor      = 0.0;
        std::size_t skipped_imu_pair_count = 0;
        std::size_t world_acc_sample_count = 0;
        double      world_acc_z_mean      = 0.0;
        double      world_acc_z_min       = 0.0;
        double      world_acc_z_max       = 0.0;
        double      world_acc_z_dt_integral = 0.0;
        double      positive_predict_dt_sum = 0.0;
        std::size_t point_time_valid_count = 0;
        std::size_t point_time_nonfinite_count = 0;
        std::size_t point_time_outside_scan_count = 0;
        double      point_time_min        = 0.0;
        double      point_time_max        = 0.0;
        std::size_t deskew_point_count    = 0;
        double      deskew_dz_mean        = 0.0;
        double      deskew_dz_rms         = 0.0;
        double      deskew_dz_abs_p95     = 0.0;
        double      deskew_dz_abs_max     = 0.0;
        double      deskew_rotation_dz_mean = 0.0;
        double      deskew_rotation_dz_rms = 0.0;
        double      deskew_rotation_dz_abs_p95 = 0.0;
        double      deskew_rotation_dz_abs_max = 0.0;
        double      deskew_translation_dz_mean = 0.0;
        double      deskew_translation_dz_rms = 0.0;
        double      deskew_translation_dz_abs_p95 = 0.0;
        double      deskew_translation_dz_abs_max = 0.0;
        double      deskew_peak_point_time = 0.0;
        double      deskew_peak_input_z   = 0.0;
        double      deskew_peak_output_z  = 0.0;
        anubis_mapping::Vec3d last_acc_input = anubis_mapping::Zero3d;  // m/s^2
        anubis_mapping::Vec3d last_gyr_input = anubis_mapping::Zero3d;  // rad/s
        anubis_mapping::Vec3d last_world_acc = anubis_mapping::Zero3d;  // m/s^2
    };

    ImuProcess();

    void reset();

    ~ImuProcess();

    void                          Reset();
    void                          set_extrinsic(const anubis_mapping::Vec3d& transl, const anubis_mapping::Mat3d& rot);
    void                          set_extrinsic(const anubis_mapping::Vec3d& transl);
    void                          set_extrinsic(const MD(4, 4) & T);
    void                          set_gyr_cov(const anubis_mapping::Vec3d& scaler);
    void                          set_acc_cov(const anubis_mapping::Vec3d& scaler);
    void                          set_gyr_bias_cov(const anubis_mapping::Vec3d& b_g);
    void                          set_acc_bias_cov(const anubis_mapping::Vec3d& b_a);
    void                          set_init_duration(double seconds);
    void                          set_init_stationary_thresholds(
                                 double max_acc_std_g,
                                 double max_gyr_std_rad_s,
                                 double max_mean_gyr_rad_s);
    // `accel_bias` is expressed in raw Livox accelerometer units (g).  It is
    // converted to m/s^2 when stored in the filter state.  `gyro_bias` and its
    // correction are expressed in rad/s and are never gravity-scaled.
    void                          set_initial_biases(
                                 const anubis_mapping::Vec3d& accel_bias,
                                 const anubis_mapping::Vec3d& gyro_bias);
    // RPY is expressed in radians and maps the measured initial gravity frame
    // into the configured map frame (left-multiplied correction).
    void                          set_initial_gravity_correction_rpy(
                                 const anubis_mapping::Vec3d& rpy_rad);
    void                          set_init_acc_norm_error(double max_error_g);
    // Drop the cross-frame IMU time anchor after a sensor-timeline gap.  The
    // next frame starts from its own first sample and therefore cannot
    // integrate a stale multi-second interval.
    void                          reset_time_anchor();
    bool                          has_time_anchor() const
    {
        return time_anchor_valid_ && static_cast<bool>(last_imu_);
    }
    bool                          initialization_complete() const { return !imu_need_init_; }
    const Diagnostics&            diagnostics() const;
    Eigen::Matrix<double, 12, 12> Q;
    void                          Process(
                                 const anubis_mapping::MeasureGroup& meas, esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state, anubis_mapping::CloudPtr pcl_un_);

    anubis_mapping::Vec3d cov_acc;
    anubis_mapping::Vec3d cov_gyr;
    anubis_mapping::Vec3d cov_acc_scale;
    anubis_mapping::Vec3d cov_gyr_scale;
    anubis_mapping::Vec3d cov_bias_gyr;
    anubis_mapping::Vec3d cov_bias_acc;
    double             first_lidar_time;

private:
    void IMU_init(const anubis_mapping::MeasureGroup& meas, esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state, int& N);
    void UndistortPcl(const anubis_mapping::MeasureGroup& meas, esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state,
        anubis_mapping::PointCloudType& pcl_in_out);

    anubis_mapping::CloudPtr            cur_pcl_un_;
    ImuMessagePtr                    last_imu_;
    std::deque<ImuMessagePtr>        v_imu_;
    std::vector<anubis_mapping::Pose6D> IMUpose;
    std::vector<anubis_mapping::Mat3d>  v_rot_pcl_;
    anubis_mapping::Mat3d               Lidar_R_wrt_IMU;
    anubis_mapping::Vec3d               Lidar_T_wrt_IMU;
    // Initialization means are kept in the sensor's native units: g and
    // rad/s respectively.  The propagated filter input is converted to SI
    // units in UndistortPcl().
    anubis_mapping::Vec3d               mean_acc;
    anubis_mapping::Vec3d               mean_gyr;
    anubis_mapping::Vec3d               corrected_mean_acc_g_ = anubis_mapping::Zero3d;
    double                           corrected_mean_acc_norm_g_ = 0.0;
    double                           acc_scale_factor_m_s2_per_g_ = 0.0;
    anubis_mapping::Vec3d               angvel_last;
    anubis_mapping::Vec3d               acc_s_last;
    double                           start_timestamp_;
    // Explicitly initialized because the first undistortion pass compares
    // against this value before a previous LiDAR frame exists.
    double                           last_lidar_end_time_ = -1.0;
    double                           init_duration_s_ = 3.0;
    double                           init_max_acc_std_g_ = 0.05;
    double                           init_max_gyr_std_rad_s_ = 0.05;
    double                           init_max_mean_gyr_rad_s_ = 0.05;
    double                           init_max_acc_norm_error_g_ = 0.05;
    anubis_mapping::Vec3d               initial_ba_ = anubis_mapping::Zero3d;
    anubis_mapping::Vec3d               initial_bg_ = anubis_mapping::Zero3d;
    anubis_mapping::Vec3d               gravity_correction_rpy_rad_ = anubis_mapping::Zero3d;
    std::size_t                      init_attempt_count_ = 0;
    int                              init_iter_num  = 1;
    bool                             b_first_frame_ = true;
    bool                             imu_need_init_ = true;
    bool                             time_anchor_valid_ = false;
    Diagnostics                      diagnostics_;
};
