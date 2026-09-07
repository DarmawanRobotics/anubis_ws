#ifndef GLOBAL_LOCALIZATION_MATH_HPP
#define GLOBAL_LOCALIZATION_MATH_HPP

#include <cmath>

#include <Eigen/Core>

namespace localization {
namespace global_localization_math {

inline Eigen::Matrix4d composeAbsolutePose(
    const Eigen::Matrix4d& correction_second,
    const Eigen::Matrix4d& correction_first,
    const Eigen::Matrix4d& seed_pose) {
  return correction_second * correction_first * seed_pose;
}

inline double wrapAngleDeg(double angle_deg) {
  double wrapped = std::fmod(angle_deg + 180.0, 360.0);
  if (wrapped < 0.0) {
    wrapped += 360.0;
  }
  return wrapped - 180.0;
}

}  // namespace global_localization_math
}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_MATH_HPP
