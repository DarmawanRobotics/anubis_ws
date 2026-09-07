#ifndef GLOBAL_LOCALIZATION_GRAVITY_HPP
#define GLOBAL_LOCALIZATION_GRAVITY_HPP

#include <cstddef>
#include <deque>
#include <vector>

#include <Eigen/Geometry>
#include <rclcpp/time.hpp>

#include "localization/global_localization_types.hpp"

namespace localization {

struct GravityGateResult {
  bool accepted = false;
  GLAnchorRejectReason reject_reason = GLAnchorRejectReason::GravityUnavailable;
  double age_ms = -1.0;
  Eigen::Matrix3d R_level_base = Eigen::Matrix3d::Identity();
};

class RawImuGravityFilter {
public:
  void configure(double window_s, int min_samples,
                 double max_angular_velocity_rps,
                 double max_accel_residual_mps2);

  void reset();

  bool addSample(const rclcpp::Time& stamp,
                 const Eigen::Vector3d& acceleration_base_mps2,
                 const Eigen::Vector3d& angular_velocity_base_rps,
                 GravityAlignmentSample& sample);

  std::size_t sampleCount() const;

private:
  struct Entry {
    rclcpp::Time stamp;
    Eigen::Vector3d up_base = Eigen::Vector3d::UnitZ();
    double angular_velocity_rps = 0.0;
    double accel_norm_residual_mps2 = 0.0;
  };

  void popFront();

  double window_s_ = 0.30;
  int min_samples_ = 30;
  double max_angular_velocity_rps_ = 0.10;
  double max_accel_residual_mps2_ = 1.5;
  std::deque<Entry> entries_;
  Eigen::Vector3d up_sum_ = Eigen::Vector3d::Zero();
};

bool makeLevelRotationFromUp(const Eigen::Vector3d& up_base,
                             Eigen::Matrix3d& R_level_base);

bool makeGravitySampleFromOrientation(
    const rclcpp::Time& stamp,
    const Eigen::Quaterniond& world_from_sensor,
    const Eigen::Vector3d& orientation_variance,
    const Eigen::Matrix3d& body_from_sensor,
    const Eigen::Vector3d& acceleration_body,
    const Eigen::Vector3d& angular_velocity_body,
    const std::string& source,
    GravityAlignmentSample& sample);

bool slerpShortestRotation(const Eigen::Quaterniond& first,
                           const Eigen::Quaterniond& second,
                           double fraction,
                           Eigen::Matrix3d& interpolated_rotation);

bool selectGravitySampleForAnchor(
    const std::vector<GravityAlignmentSample>& history,
    const rclcpp::Time& anchor_stamp,
    const GLParams& params,
    GravityAlignmentSample& selected_sample);

GravityGateResult evaluateGravitySample(
    const rclcpp::Time& anchor_stamp,
    const GravityAlignmentSample* sample,
    const GLParams& params);

void accumulateGravityGateSummary(
    const GravityAlignmentSample* sample,
    const GravityGateResult& gate,
    GLSummaryCounts& summary);

bool composeMapBaseSeed(const Eigen::Matrix4d& T_map_level,
                        const Eigen::Matrix3d& R_level_base,
                        Eigen::Matrix4d& T_map_base);

}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_GRAVITY_HPP
