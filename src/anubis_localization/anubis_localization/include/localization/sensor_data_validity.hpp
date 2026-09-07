#ifndef LOCALIZATION__SENSOR_DATA_VALIDITY_HPP_
#define LOCALIZATION__SENSOR_DATA_VALIDITY_HPP_

namespace localization {

inline bool sensorDataValid(bool use_imu, bool lidar_valid, bool imu_valid) noexcept
{
  return lidar_valid && (!use_imu || imu_valid);
}

}  // namespace localization

#endif  // LOCALIZATION__SENSOR_DATA_VALIDITY_HPP_
