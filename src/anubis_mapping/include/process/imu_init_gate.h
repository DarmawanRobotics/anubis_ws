#pragma once

#include <cmath>

#include <Eigen/Core>

namespace anubis_mapping
{
    // The Livox initialization stream is expressed in multiples of standard
    // gravity (g).  The filter state and propagation input are expressed in
    // SI units (m/s^2).  Keep the conversion explicit at this boundary so a
    // configured accelerometer bias cannot be silently applied in the wrong
    // unit system.
    constexpr double kMinImuAccelNormG = 1.0e-9;

    inline bool compute_corrected_imu_accel_mean_g(
        const Eigen::Vector3d& mean_acc_g,
        const Eigen::Vector3d& initial_ba_g,
        Eigen::Vector3d& corrected_mean_acc_g,
        double& corrected_mean_norm_g)
    {
        corrected_mean_acc_g.setZero();
        corrected_mean_norm_g = 0.0;
        if (!mean_acc_g.allFinite() || !initial_ba_g.allFinite())
        {
            return false;
        }
        corrected_mean_acc_g = mean_acc_g - initial_ba_g;
        if (!corrected_mean_acc_g.allFinite())
        {
            return false;
        }
        corrected_mean_norm_g = corrected_mean_acc_g.norm();
        return std::isfinite(corrected_mean_norm_g) &&
            corrected_mean_norm_g > kMinImuAccelNormG;
    }

    inline bool derive_imu_accel_initialization_terms(
        const Eigen::Vector3d& mean_acc_g,
        const Eigen::Vector3d& initial_ba_g,
        double gravity_m_s2,
        Eigen::Vector3d& corrected_mean_acc_g,
        Eigen::Vector3d& state_ba_m_s2,
        double& corrected_mean_norm_g,
        double& acc_scale_m_s2_per_g)
    {
        corrected_mean_acc_g.setZero();
        state_ba_m_s2.setZero();
        corrected_mean_norm_g = 0.0;
        acc_scale_m_s2_per_g = 0.0;

        if (!std::isfinite(gravity_m_s2) || gravity_m_s2 <= 0.0 ||
            !compute_corrected_imu_accel_mean_g(
                mean_acc_g, initial_ba_g, corrected_mean_acc_g,
                corrected_mean_norm_g))
        {
            return false;
        }

        acc_scale_m_s2_per_g = gravity_m_s2 / corrected_mean_norm_g;
        // Prediction scales each raw accelerometer sample by this measured
        // factor.  The configured raw-g bias must use the identical factor so
        // `in.acc - state.ba` remains a coherent SI-unit subtraction.
        state_ba_m_s2 = initial_ba_g * acc_scale_m_s2_per_g;
        return state_ba_m_s2.allFinite() &&
            std::isfinite(acc_scale_m_s2_per_g) &&
            acc_scale_m_s2_per_g > 0.0;
    }

    inline bool compose_imu_gyro_bias_rad_s(
        const Eigen::Vector3d& mean_gyr_rad_s,
        const Eigen::Vector3d& initial_bg_rad_s,
        Eigen::Vector3d& state_bg_rad_s)
    {
        state_bg_rad_s.setZero();
        if (!mean_gyr_rad_s.allFinite() || !initial_bg_rad_s.allFinite())
        {
            return false;
        }
        // `initial_bg_rad_s` is an additive correction in the same units as
        // the measured gyro stream; it is never scaled by gravity.
        state_bg_rad_s = mean_gyr_rad_s + initial_bg_rad_s;
        return state_bg_rad_s.allFinite();
    }

    inline bool make_imu_gravity_alignment_target_g(
        const Eigen::Vector3d& corrected_mean_acc_g,
        Eigen::Vector3d& gravity_target_g)
    {
        gravity_target_g.setZero();
        if (!corrected_mean_acc_g.allFinite())
        {
            return false;
        }
        const double norm_g = corrected_mean_acc_g.norm();
        if (!std::isfinite(norm_g) || norm_g <= kMinImuAccelNormG)
        {
            return false;
        }
        // FAST-LIO's initialization convention maps measured specific force
        // to +Z; the state gravity vector then remains [0, 0, -G].
        gravity_target_g.z() = norm_g;
        return true;
    }

    inline bool is_imu_acc_norm_valid(
        double mean_acc_norm_g, double max_norm_error_g)
    {
        return std::isfinite(mean_acc_norm_g) &&
            std::isfinite(max_norm_error_g) && max_norm_error_g > 0.0 &&
            std::abs(mean_acc_norm_g - 1.0) <= max_norm_error_g;
    }

    inline bool is_imu_acc_norm_valid(
        const Eigen::Vector3d& mean_acc_g,
        const Eigen::Vector3d& initial_ba_g,
        double max_norm_error_g)
    {
        Eigen::Vector3d corrected_mean_acc_g;
        double corrected_mean_norm_g = 0.0;
        if (!compute_corrected_imu_accel_mean_g(
                mean_acc_g, initial_ba_g, corrected_mean_acc_g,
                corrected_mean_norm_g))
        {
            return false;
        }
        return is_imu_acc_norm_valid(
            corrected_mean_norm_g, max_norm_error_g);
    }

    inline bool is_imu_init_window_stationary(
        const Eigen::Vector3d& acc_std_g,
        const Eigen::Vector3d& gyr_std_rad_s,
        const Eigen::Vector3d& mean_gyr_rad_s,
        double max_acc_std_g,
        double max_gyr_std_rad_s,
        double max_mean_gyr_rad_s)
    {
        return acc_std_g.allFinite() && gyr_std_rad_s.allFinite() &&
               mean_gyr_rad_s.allFinite() && std::isfinite(max_acc_std_g) &&
               std::isfinite(max_gyr_std_rad_s) &&
               std::isfinite(max_mean_gyr_rad_s) && max_acc_std_g > 0.0 &&
               max_gyr_std_rad_s > 0.0 && max_mean_gyr_rad_s > 0.0 &&
               acc_std_g.cwiseAbs().maxCoeff() <= max_acc_std_g &&
               gyr_std_rad_s.cwiseAbs().maxCoeff() <= max_gyr_std_rad_s &&
               mean_gyr_rad_s.norm() <= max_mean_gyr_rad_s;
    }
}  // namespace anubis_mapping
