#include "localization/global_localization_gravity.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>

#include <Eigen/Geometry>

namespace localization {
namespace {

constexpr double kRotationTolerance = 1e-6;

bool isRotationMatrix(const Eigen::Matrix3d& rotation) {
  if (!rotation.allFinite()) {
    return false;
  }
  const Eigen::Matrix3d orthogonality =
      rotation.transpose() * rotation - Eigen::Matrix3d::Identity();
  return orthogonality.cwiseAbs().maxCoeff() <= kRotationTolerance &&
      std::abs(rotation.determinant() - 1.0) <= kRotationTolerance;
}

bool finiteNonNegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

double conservativeMaximum(double first, double second) {
  if (!finiteNonNegative(first) || !finiteNonNegative(second)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::max(first, second);
}

}  // namespace

bool makeLevelRotationFromUp(const Eigen::Vector3d& up_base,
                             Eigen::Matrix3d& R_level_base) {
  R_level_base = Eigen::Matrix3d::Identity();
  if (!up_base.allFinite() || up_base.norm() <= 1e-12) {
    return false;
  }

  const Eigen::Quaterniond level_from_base = Eigen::Quaterniond::FromTwoVectors(
      up_base.normalized(), Eigen::Vector3d::UnitZ());
  if (!level_from_base.coeffs().allFinite()) {
    return false;
  }
  R_level_base = level_from_base.normalized().toRotationMatrix();
  return isRotationMatrix(R_level_base);
}

void RawImuGravityFilter::configure(
    double window_s, int min_samples, double max_angular_velocity_rps,
    double max_accel_residual_mps2) {
  window_s_ = window_s;
  min_samples_ = min_samples;
  max_angular_velocity_rps_ = max_angular_velocity_rps;
  max_accel_residual_mps2_ = max_accel_residual_mps2;
  reset();
}

void RawImuGravityFilter::reset() {
  entries_.clear();
  up_sum_.setZero();
}

std::size_t RawImuGravityFilter::sampleCount() const {
  return entries_.size();
}

void RawImuGravityFilter::popFront() {
  if (entries_.empty()) {
    return;
  }
  up_sum_ -= entries_.front().up_base;
  entries_.pop_front();
}

bool RawImuGravityFilter::addSample(
    const rclcpp::Time& stamp,
    const Eigen::Vector3d& acceleration_base_mps2,
    const Eigen::Vector3d& angular_velocity_base_rps,
    GravityAlignmentSample& sample) {
  GravityAlignmentSample empty_sample;
  sample = empty_sample;
  if (stamp.nanoseconds() <= 0 || !acceleration_base_mps2.allFinite() ||
      !angular_velocity_base_rps.allFinite() ||
      !std::isfinite(window_s_) || window_s_ <= 0.0 || min_samples_ < 2 ||
      !std::isfinite(max_angular_velocity_rps_) ||
      max_angular_velocity_rps_ <= 0.0 ||
      !std::isfinite(max_accel_residual_mps2_) ||
      max_accel_residual_mps2_ <= 0.0) {
    reset();
    return false;
  }

  if (!entries_.empty() &&
      (entries_.back().stamp.get_clock_type() != stamp.get_clock_type() ||
       stamp <= entries_.back().stamp)) {
    reset();
    return false;
  }

  constexpr double kGravityMps2 = 9.80665;
  const double acceleration_norm = acceleration_base_mps2.norm();
  const double angular_velocity_norm = angular_velocity_base_rps.norm();
  const double acceleration_residual =
      std::abs(acceleration_norm - kGravityMps2);
  if (!std::isfinite(acceleration_norm) || acceleration_norm <= 1e-6 ||
      !std::isfinite(angular_velocity_norm) ||
      angular_velocity_norm > max_angular_velocity_rps_ ||
      !std::isfinite(acceleration_residual) ||
      acceleration_residual > max_accel_residual_mps2_) {
    reset();
    return false;
  }

  Entry entry;
  entry.stamp = stamp;
  entry.up_base = acceleration_base_mps2 / acceleration_norm;
  entry.angular_velocity_rps = angular_velocity_norm;
  entry.accel_norm_residual_mps2 = acceleration_residual;
  entries_.push_back(entry);
  up_sum_ += entry.up_base;

  // Retain the shortest suffix that still spans the requested dwell window.
  while (entries_.size() > 1U &&
         (stamp - entries_[1].stamp).seconds() >= window_s_) {
    popFront();
  }

  if (entries_.size() < static_cast<std::size_t>(min_samples_) ||
      (stamp - entries_.front().stamp).seconds() < window_s_ ||
      up_sum_.norm() <= 1e-12) {
    return false;
  }

  const Eigen::Vector3d mean_up_base = up_sum_.normalized();
  if (!makeLevelRotationFromUp(mean_up_base, sample.R_level_base)) {
    reset();
    return false;
  }

  double max_angular_velocity = 0.0;
  double max_acceleration_residual = 0.0;
  for (const Entry& buffered : entries_) {
    max_angular_velocity = std::max(
        max_angular_velocity, buffered.angular_velocity_rps);
    max_acceleration_residual = std::max(
        max_acceleration_residual, buffered.accel_norm_residual_mps2);
  }

  constexpr double kRadiansToDegrees =
      180.0 / 3.14159265358979323846;
  const double resultant_length = std::clamp(
      up_sum_.norm() / static_cast<double>(entries_.size()), 0.0, 1.0);
  const double direction_rms_rad = std::sqrt(
      std::max(0.0, 2.0 * (1.0 - resultant_length)));

  sample.stamp = stamp;
  sample.roll_deg = std::atan2(
      sample.R_level_base(2, 1), sample.R_level_base(2, 2)) *
      kRadiansToDegrees;
  sample.pitch_deg = std::asin(std::clamp(
      -sample.R_level_base(2, 0), -1.0, 1.0)) * kRadiansToDegrees;
  sample.uncertainty_deg = direction_rms_rad * kRadiansToDegrees;
  sample.angular_velocity_rps = max_angular_velocity;
  sample.accel_norm_residual_mps2 = max_acceleration_residual;
  sample.source = "raw_imu_gravity_filter";
  sample.valid = std::isfinite(sample.roll_deg) &&
      std::isfinite(sample.pitch_deg) &&
      std::isfinite(sample.uncertainty_deg) &&
      std::isfinite(sample.angular_velocity_rps) &&
      std::isfinite(sample.accel_norm_residual_mps2);
  return sample.valid;
}

bool makeGravitySampleFromOrientation(
    const rclcpp::Time& stamp,
    const Eigen::Quaterniond& world_from_sensor,
    const Eigen::Vector3d& orientation_variance,
    const Eigen::Matrix3d& body_from_sensor,
    const Eigen::Vector3d& acceleration_body,
    const Eigen::Vector3d& angular_velocity_body,
    const std::string& source,
    GravityAlignmentSample& sample) {
  GravityAlignmentSample empty_sample;
  sample = empty_sample;
  if (stamp.nanoseconds() <= 0 || !world_from_sensor.coeffs().allFinite() ||
      world_from_sensor.norm() <= 1e-12 ||
      !orientation_variance.allFinite() ||
      (orientation_variance.array() < 0.0).any() ||
      orientation_variance.sum() <= 0.0 ||
      !isRotationMatrix(body_from_sensor) ||
      !acceleration_body.allFinite() ||
      !angular_velocity_body.allFinite()) {
    return false;
  }

  const Eigen::Matrix3d world_from_body =
      world_from_sensor.normalized().toRotationMatrix() *
      body_from_sensor.transpose();
  const Eigen::Vector3d up_body =
      world_from_body.transpose() * Eigen::Vector3d::UnitZ();
  if (!makeLevelRotationFromUp(up_body, sample.R_level_base)) {
    return false;
  }

  constexpr double kRadiansToDegrees =
      180.0 / 3.14159265358979323846;
  sample.stamp = stamp;
  sample.roll_deg = std::atan2(
      world_from_body(2, 1), world_from_body(2, 2)) * kRadiansToDegrees;
  sample.pitch_deg = std::asin(std::clamp(
      -world_from_body(2, 0), -1.0, 1.0)) * kRadiansToDegrees;
  sample.uncertainty_deg = std::sqrt(std::max(
      orientation_variance.x(), orientation_variance.y())) *
      kRadiansToDegrees;
  sample.angular_velocity_rps = angular_velocity_body.norm();
  sample.accel_norm_residual_mps2 = std::abs(
      acceleration_body.norm() - 9.80665);
  sample.source = source;
  sample.valid = std::isfinite(sample.roll_deg) &&
      std::isfinite(sample.pitch_deg) &&
      std::isfinite(sample.uncertainty_deg) &&
      std::isfinite(sample.angular_velocity_rps) &&
      std::isfinite(sample.accel_norm_residual_mps2);
  return sample.valid;
}

bool slerpShortestRotation(const Eigen::Quaterniond& first,
                           const Eigen::Quaterniond& second,
                           double fraction,
                           Eigen::Matrix3d& interpolated_rotation) {
  interpolated_rotation = Eigen::Matrix3d::Identity();
  if (!first.coeffs().allFinite() || !second.coeffs().allFinite() ||
      first.norm() <= 1e-12 || second.norm() <= 1e-12 ||
      !std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0) {
    return false;
  }

  Eigen::Quaterniond first_normalized = first.normalized();
  Eigen::Quaterniond second_normalized = second.normalized();
  if (first_normalized.dot(second_normalized) < 0.0) {
    second_normalized.coeffs() *= -1.0;
  }
  const Eigen::Quaterniond interpolated =
      first_normalized.slerp(fraction, second_normalized).normalized();
  if (!interpolated.coeffs().allFinite()) {
    return false;
  }
  interpolated_rotation = interpolated.toRotationMatrix();
  return isRotationMatrix(interpolated_rotation);
}

bool selectGravitySampleForAnchor(
    const std::vector<GravityAlignmentSample>& history,
    const rclcpp::Time& anchor_stamp,
    const GLParams& params,
    GravityAlignmentSample& selected_sample) {
  GravityAlignmentSample empty_sample;
  selected_sample = empty_sample;
  if (history.empty() || !std::isfinite(params.level0_gravity_time_offset_ms) ||
      !std::isfinite(params.level0_gravity_max_age_ms) ||
      params.level0_gravity_max_age_ms <= 0.0) {
    return false;
  }

  const long double anchor_ns =
      static_cast<long double>(anchor_stamp.nanoseconds());
  const long double offset_ns =
      static_cast<long double>(params.level0_gravity_time_offset_ms) * 1.0e6L;
  const long double max_age_ns =
      static_cast<long double>(params.level0_gravity_max_age_ms) * 1.0e6L;
  const GravityAlignmentSample* before = nullptr;
  const GravityAlignmentSample* after = nullptr;
  const GravityAlignmentSample* exact = nullptr;
  const GravityAlignmentSample* nearest = nullptr;
  long double before_aligned_ns = 0.0L;
  long double after_aligned_ns = 0.0L;
  long double nearest_distance_ns = std::numeric_limits<long double>::infinity();
  int64_t previous_stamp_ns = 0;
  bool has_previous_stamp = false;

  for (const GravityAlignmentSample& sample : history) {
    if (sample.stamp.get_clock_type() != anchor_stamp.get_clock_type()) {
      return false;
    }
    const int64_t sample_stamp_ns = sample.stamp.nanoseconds();
    if (has_previous_stamp && sample_stamp_ns < previous_stamp_ns) {
      return false;
    }
    previous_stamp_ns = sample_stamp_ns;
    has_previous_stamp = true;
    if (!sample.valid || !isRotationMatrix(sample.R_level_base)) {
      continue;
    }

    const long double aligned_ns =
        static_cast<long double>(sample_stamp_ns) + offset_ns;
    const long double distance_ns = std::abs(aligned_ns - anchor_ns);
    if (distance_ns < nearest_distance_ns) {
      nearest = &sample;
      nearest_distance_ns = distance_ns;
    }
    if (aligned_ns == anchor_ns) {
      exact = &sample;
    } else if (aligned_ns < anchor_ns) {
      before = &sample;
      before_aligned_ns = aligned_ns;
    } else if (after == nullptr) {
      after = &sample;
      after_aligned_ns = aligned_ns;
    }
  }

  if (exact != nullptr) {
    selected_sample = *exact;
    return true;
  }

  if (before != nullptr && after != nullptr &&
      before->source == after->source &&
      anchor_ns - before_aligned_ns <= max_age_ns &&
      after_aligned_ns - anchor_ns <= max_age_ns) {
    const long double interval_ns = after_aligned_ns - before_aligned_ns;
    if (interval_ns > 0.0L) {
      const double fraction = static_cast<double>(
          (anchor_ns - before_aligned_ns) / interval_ns);
      Eigen::Matrix3d interpolated_rotation;
      if (slerpShortestRotation(Eigen::Quaterniond(before->R_level_base),
                                Eigen::Quaterniond(after->R_level_base),
                                fraction, interpolated_rotation)) {
        const long double selected_stamp_ns = anchor_ns - offset_ns;
        if (selected_stamp_ns >=
                static_cast<long double>(std::numeric_limits<int64_t>::min()) &&
            selected_stamp_ns <=
                static_cast<long double>(std::numeric_limits<int64_t>::max())) {
          selected_sample = *before;
          selected_sample.stamp = rclcpp::Time(
              static_cast<int64_t>(std::llround(selected_stamp_ns)),
              anchor_stamp.get_clock_type());
          selected_sample.R_level_base = interpolated_rotation;
          selected_sample.roll_deg =
              before->roll_deg + fraction * (after->roll_deg - before->roll_deg);
          selected_sample.pitch_deg = before->pitch_deg +
              fraction * (after->pitch_deg - before->pitch_deg);
          selected_sample.uncertainty_deg = conservativeMaximum(
              before->uncertainty_deg, after->uncertainty_deg);
          selected_sample.angular_velocity_rps = conservativeMaximum(
              before->angular_velocity_rps, after->angular_velocity_rps);
          selected_sample.accel_norm_residual_mps2 = conservativeMaximum(
              before->accel_norm_residual_mps2,
              after->accel_norm_residual_mps2);
          selected_sample.valid = true;
          return true;
        }
      }
    }
  }

  if (nearest != nullptr && nearest_distance_ns <= max_age_ns) {
    selected_sample = *nearest;
    return true;
  }
  return false;
}

GravityGateResult evaluateGravitySample(
    const rclcpp::Time& anchor_stamp,
    const GravityAlignmentSample* sample,
    const GLParams& params) {
  GravityGateResult result;
  if (sample == nullptr || !sample->valid ||
      sample->stamp.get_clock_type() != anchor_stamp.get_clock_type() ||
      !isRotationMatrix(sample->R_level_base)) {
    return result;
  }

  const long double offset_ns =
      static_cast<long double>(params.level0_gravity_time_offset_ms) * 1.0e6L;
  const long double delta_ns =
      static_cast<long double>(sample->stamp.nanoseconds()) + offset_ns -
      static_cast<long double>(anchor_stamp.nanoseconds());
  result.age_ms = static_cast<double>(std::abs(delta_ns) / 1.0e6L);
  if (!std::isfinite(result.age_ms) ||
      result.age_ms > params.level0_gravity_max_age_ms) {
    result.reject_reason = GLAnchorRejectReason::GravityStale;
    return result;
  }
  if (!finiteNonNegative(sample->uncertainty_deg) ||
      sample->uncertainty_deg > params.level0_gravity_max_uncertainty_deg) {
    result.reject_reason = GLAnchorRejectReason::GravityUncertain;
    return result;
  }
  const double tilt_limit = params.level0_gravity_align_enabled
      ? params.level0_anchor_max_abs_rp_deg
      : params.level0_unaligned_max_abs_rp_deg;
  if (!std::isfinite(sample->roll_deg) || !std::isfinite(sample->pitch_deg) ||
      std::abs(sample->roll_deg) > tilt_limit ||
      std::abs(sample->pitch_deg) > tilt_limit) {
    result.reject_reason = GLAnchorRejectReason::TiltExceeded;
    return result;
  }
  if (!finiteNonNegative(sample->angular_velocity_rps) ||
      sample->angular_velocity_rps >
          params.level0_anchor_max_angular_velocity_rps) {
    result.reject_reason = GLAnchorRejectReason::AngularVelocityExceeded;
    return result;
  }
  if (!finiteNonNegative(sample->accel_norm_residual_mps2) ||
      sample->accel_norm_residual_mps2 >
          params.level0_gravity_max_accel_residual_mps2) {
    result.reject_reason = GLAnchorRejectReason::AccelerationResidualExceeded;
    return result;
  }

  result.accepted = true;
  result.reject_reason = GLAnchorRejectReason::None;
  if (params.level0_gravity_align_enabled) {
    result.R_level_base = sample->R_level_base;
  }
  return result;
}

void accumulateGravityGateSummary(
    const GravityAlignmentSample* sample,
    const GravityGateResult& gate,
    GLSummaryCounts& summary) {
  summary.anchor_gravity_age_ms = gate.age_ms;
  if (sample != nullptr) {
    summary.anchor_gravity_uncertainty_deg = sample->uncertainty_deg;
    summary.anchor_roll_deg = sample->roll_deg;
    summary.anchor_pitch_deg = sample->pitch_deg;
    summary.anchor_angular_velocity_rps = sample->angular_velocity_rps;
    summary.anchor_accel_norm_residual_mps2 =
        sample->accel_norm_residual_mps2;
  }

  switch (gate.reject_reason) {
    case GLAnchorRejectReason::None:
      break;
    case GLAnchorRejectReason::GravityUnavailable:
      ++summary.anchor_rejected_gravity_unavailable;
      break;
    case GLAnchorRejectReason::GravityStale:
      ++summary.anchor_rejected_gravity_stale;
      break;
    case GLAnchorRejectReason::GravityUncertain:
      ++summary.anchor_rejected_gravity_uncertain;
      break;
    case GLAnchorRejectReason::TiltExceeded:
      ++summary.anchor_rejected_rp;
      break;
    case GLAnchorRejectReason::AngularVelocityExceeded:
      ++summary.anchor_rejected_angular_velocity;
      break;
    case GLAnchorRejectReason::AccelerationResidualExceeded:
      ++summary.anchor_rejected_accel_residual;
      break;
    case GLAnchorRejectReason::MotionWindowExceeded:
      ++summary.anchor_rejected_motion_window;
      break;
    case GLAnchorRejectReason::TooFewValidPoints:
      ++summary.anchor_rejected_too_few_points;
      break;
  }
}

bool composeMapBaseSeed(const Eigen::Matrix4d& T_map_level,
                        const Eigen::Matrix3d& R_level_base,
                        Eigen::Matrix4d& T_map_base) {
  T_map_base = Eigen::Matrix4d::Identity();
  if (!T_map_level.allFinite() || !isRotationMatrix(R_level_base)) {
    return false;
  }
  Eigen::Matrix4d T_level_base = Eigen::Matrix4d::Identity();
  T_level_base.block<3, 3>(0, 0) = R_level_base;
  T_map_base = T_map_level * T_level_base;
  return T_map_base.allFinite();
}

}  // namespace localization
