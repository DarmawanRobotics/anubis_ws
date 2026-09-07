#ifndef LOCALIZATION__NDT_OPTIMIZER_DIAGNOSTICS_HPP_
#define LOCALIZATION__NDT_OPTIMIZER_DIAGNOSTICS_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

namespace localization {

struct NdtOptimizerDiagnostics {
  bool hessian_valid = false;
  std::array<double, 6> hessian_eigenvalues{};
  double hessian_abs_condition = std::numeric_limits<double>::infinity();
  int hessian_positive = 0;
  int hessian_negative = 0;
  int hessian_near_zero = 0;
  Eigen::Matrix<double, 6, 1> weakest_direction =
    Eigen::Matrix<double, 6, 1>::Zero();

  bool trajectory_valid = false;
  double iteration_translation_path_m = 0.0;
  double iteration_rotation_path_deg = 0.0;
  double net_translation_m = 0.0;
  double net_rotation_deg = 0.0;
  double translation_path_ratio = 0.0;
  double final_step_translation_m = 0.0;
  double final_step_rotation_deg = 0.0;
  size_t transformation_count = 0;
  std::array<double, 3> initial_xyz{};
  std::array<double, 3> final_xyz{};
  std::array<double, 3> correction_xyz_local{};
  std::array<double, 3> correction_rpy_deg{};
};

inline double rotationAngleDeg(const Eigen::Matrix3f& rotation) {
  Eigen::Quaternionf quaternion(rotation);
  if (!quaternion.coeffs().allFinite() || quaternion.norm() < 1e-6f) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  quaternion.normalize();
  const double w = std::clamp(std::abs(static_cast<double>(quaternion.w())), 0.0, 1.0);
  return 2.0 * std::acos(w) * 180.0 / 3.14159265358979323846;
}

inline std::array<double, 3> rotationToRpyDeg(const Eigen::Matrix3f& rotation) {
  constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
  const double pitch = std::asin(std::clamp(
    -static_cast<double>(rotation(2, 0)), -1.0, 1.0));
  const double cos_pitch = std::cos(pitch);

  double roll = 0.0;
  double yaw = 0.0;
  if (std::abs(cos_pitch) > 1e-8) {
    roll = std::atan2(
      static_cast<double>(rotation(2, 1)),
      static_cast<double>(rotation(2, 2)));
    yaw = std::atan2(
      static_cast<double>(rotation(1, 0)),
      static_cast<double>(rotation(0, 0)));
  } else {
    roll = std::atan2(
      -static_cast<double>(rotation(1, 2)),
      static_cast<double>(rotation(1, 1)));
  }
  return {roll * kRadToDeg, pitch * kRadToDeg, yaw * kRadToDeg};
}

inline NdtOptimizerDiagnostics analyzeNdtOptimizer(
    const Eigen::Matrix<double, 6, 6>& hessian,
    const std::vector<Eigen::Matrix4f,
      Eigen::aligned_allocator<Eigen::Matrix4f>>& transformations) {
  NdtOptimizerDiagnostics diagnostics;
  diagnostics.transformation_count = transformations.size();

  if (hessian.allFinite()) {
    const Eigen::Matrix<double, 6, 6> symmetric = 0.5 * (hessian + hessian.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(symmetric);
    if (solver.info() == Eigen::Success && solver.eigenvalues().allFinite() &&
        solver.eigenvectors().allFinite()) {
      diagnostics.hessian_valid = true;
      double max_abs = 0.0;
      double min_abs = std::numeric_limits<double>::infinity();
      int weakest_index = 0;
      for (int index = 0; index < 6; ++index) {
        const double value = solver.eigenvalues()[index];
        diagnostics.hessian_eigenvalues[static_cast<size_t>(index)] = value;
        const double abs_value = std::abs(value);
        max_abs = std::max(max_abs, abs_value);
        if (abs_value < min_abs) {
          min_abs = abs_value;
          weakest_index = index;
        }
      }

      const double sign_tolerance = std::max(1e-9, max_abs * 1e-6);
      for (const double value : diagnostics.hessian_eigenvalues) {
        if (value > sign_tolerance) {
          ++diagnostics.hessian_positive;
        } else if (value < -sign_tolerance) {
          ++diagnostics.hessian_negative;
        } else {
          ++diagnostics.hessian_near_zero;
        }
      }
      diagnostics.hessian_abs_condition = min_abs <= sign_tolerance
        ? std::numeric_limits<double>::infinity()
        : max_abs / min_abs;
      diagnostics.weakest_direction = solver.eigenvectors().col(weakest_index);
    }
  }

  for (const Eigen::Matrix4f& transformation : transformations) {
    if (!transformation.allFinite()) {
      return diagnostics;
    }
  }
  if (!transformations.empty()) {
    for (int axis = 0; axis < 3; ++axis) {
      diagnostics.initial_xyz[static_cast<size_t>(axis)] =
        transformations.front()(axis, 3);
      diagnostics.final_xyz[static_cast<size_t>(axis)] =
        transformations.back()(axis, 3);
    }
  }
  if (transformations.size() < 2) {
    return diagnostics;
  }

  diagnostics.trajectory_valid = true;
  for (size_t index = 1; index < transformations.size(); ++index) {
    const Eigen::Matrix4f delta = transformations[index - 1].inverse() * transformations[index];
    const double translation = delta.block<3, 1>(0, 3).norm();
    const double rotation = rotationAngleDeg(delta.block<3, 3>(0, 0));
    if (!std::isfinite(translation) || !std::isfinite(rotation)) {
      diagnostics.trajectory_valid = false;
      return diagnostics;
    }
    diagnostics.iteration_translation_path_m += translation;
    diagnostics.iteration_rotation_path_deg += rotation;
    diagnostics.final_step_translation_m = translation;
    diagnostics.final_step_rotation_deg = rotation;
  }

  const Eigen::Matrix4f net = transformations.front().inverse() * transformations.back();
  for (int axis = 0; axis < 3; ++axis) {
    diagnostics.correction_xyz_local[static_cast<size_t>(axis)] = net(axis, 3);
  }
  diagnostics.correction_rpy_deg = rotationToRpyDeg(net.block<3, 3>(0, 0));
  diagnostics.net_translation_m = net.block<3, 1>(0, 3).norm();
  diagnostics.net_rotation_deg = rotationAngleDeg(net.block<3, 3>(0, 0));
  diagnostics.translation_path_ratio = diagnostics.net_translation_m > 1e-6
    ? diagnostics.iteration_translation_path_m / diagnostics.net_translation_m
    : (diagnostics.iteration_translation_path_m > 1e-6
        ? std::numeric_limits<double>::infinity() : 1.0);
  return diagnostics;
}

}  // namespace localization

#endif  // LOCALIZATION__NDT_OPTIMIZER_DIAGNOSTICS_HPP_
