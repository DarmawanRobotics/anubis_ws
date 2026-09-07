
#include "localization/static_imu_init.hpp"

namespace localization
{

    void StaticIMUInit::SetParam(float time_long, int max_queue_size, float max_gyro_var, float max_acce_var)
    {
        init_time_seconds_   = time_long;
        init_queue_max_size_ = max_queue_size;
        max_static_gyro_var_ = max_gyro_var;
        max_static_acce_var_ = max_acce_var;
    }

    void StaticIMUInit::Reset()
    {
        mean_gyro_ = Eigen::Vector3d(0.0, 0.0, 0.0);
        mean_acce_ = Eigen::Vector3d(0.0, 0.0, -1.0);
    }


    bool StaticIMUInit::AddIMUData(const sensor_msgs::msg::Imu::ConstSharedPtr imu)
    {
        if (init_success_)
        {
            return true;
        }

        if (init_imu_buffer_.empty())
        {
            init_start_time_ = rclcpp::Time(imu->header.stamp).seconds();
        }

        init_imu_buffer_.push_back(imu);

        double init_time = rclcpp::Time(imu->header.stamp).seconds() - init_start_time_;
        if (init_time > init_time_seconds_)
        {
            TryInit();
        }

        while (init_imu_buffer_.size() > init_queue_max_size_)
        {
            init_imu_buffer_.pop_front();
        }
        return false;
    }

    bool StaticIMUInit::TryInit()
    {
        if (init_imu_buffer_.size() < 10)
        {
            return false;
        }
        ++init_attempt_count_;
        last_attempt_sample_count_ = init_imu_buffer_.size();
        last_attempt_duration_seconds_ =
            rclcpp::Time(init_imu_buffer_.back()->header.stamp).seconds() -
            rclcpp::Time(init_imu_buffer_.front()->header.stamp).seconds();
        Eigen::Vector3d cur_acc, cur_gyr;
        ComputeMeanAndCovDiag(init_imu_buffer_, mean_gyro_, gyro_cov_,
            [](const sensor_msgs::msg::Imu::ConstSharedPtr& imu)
            {
                return Eigen::Vector3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
            });
        ComputeMeanAndCovDiag(init_imu_buffer_, mean_acce_, acce_cov_,
            [](const sensor_msgs::msg::Imu::ConstSharedPtr& imu)
            {
                return Eigen::Vector3d(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
            });
        // Input samples have already been transformed by the fixed IMU-to-base
        // extrinsic, so this is the base-frame gravity direction (pointing down).
        // It initializes UKF roll/pitch and the base-frame accelerometer bias.
        gravity_ = -mean_acce_.normalized();

        if (gyro_cov_.norm() > max_static_gyro_var_)
        {
            if (init_attempt_count_ == 1 || init_attempt_count_ % 200 == 0)
            {
                std::cout << "[IMU INIT] result=retry attempts=" << init_attempt_count_
                          << " samples=" << last_attempt_sample_count_
                          << " duration=" << last_attempt_duration_seconds_
                          << "s gyro_mean=" << mean_gyro_.transpose()
                          << " gyro_var=" << gyro_cov_.transpose()
                          << " gyro_var_norm=" << gyro_cov_.norm()
                          << " gyro_limit=" << max_static_gyro_var_
                          << " gyro_gate=enforced acc_mean=" << mean_acce_.transpose()
                          << " acc_var=" << acce_cov_.transpose()
                          << " acc_var_norm=" << acce_cov_.norm()
                          << " acc_limit=" << max_static_acce_var_
                          << " acc_gate=diagnostic_only" << std::endl;
            }
            return false;
        }

        // if ( acce_cov_.norm () >  max_static_acce_var_) {
        //         std::cout << "加速度计测量噪声太大  --> " <<  acce_cov_.norm () << " > " <<  max_static_acce_var_ << std::endl;
        //         return false;
        // }

        init_bias_gyro_ = mean_gyro_;
        init_bias_acce_ = mean_acce_;
        init_success_   = true;
        std::cout << "IMU Init Sucessful !!!!!!" << std::endl;
        return true;
    }

}  // namespace localization
