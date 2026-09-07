#ifndef GLOBAL_LOCALIZATION_HPP
#define GLOBAL_LOCALIZATION_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "localization/global_localization_math.hpp"
#include "localization/global_localization_types.hpp"
#include "scan_descriptor/database.hpp"

namespace localization {

class GlobalLocalization {
public:
  GlobalLocalization();
  ~GlobalLocalization();

  void init(const Eigen::Matrix4d& initial_pose);

  bool performGlobalLocalization(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& global_map,
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& current_cloud,
      const Eigen::Matrix4d& initial_trans,
      Eigen::Matrix4d& final_pose);

  void performGlobalLocalization(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& global_map,
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& current_cloud,
      const Eigen::Matrix4d& initial_trans,
      const Eigen::Matrix4d* ukf_pose,
      const Eigen::Matrix4d* odom_pose,
      const GravityAlignmentSample* gravity_sample,
      GlobalLocalizationResult& result,
      GLSummaryCounts* summary = nullptr);

  void setParams(const GLParams& params);

  void setBaseFromLidar(const Eigen::Matrix4d& T_base_lidar);

  void setDescriptorDatabase(
      const std::shared_ptr<const scan_descriptor::DescriptorDatabase>& database);

private:
  bool preprocessCloud(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& input,
      pcl::PointCloud<pcl::PointXYZI>::Ptr& output) const;

  void runSingleSeedIcp(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& source,
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& target,
      GlobalPoseCandidate& candidate) const;

  Eigen::Matrix4d initial_pose_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d base_from_lidar_ = Eigen::Matrix4d::Identity();
  GLParams params_;
  std::shared_ptr<const scan_descriptor::DescriptorDatabase> descriptor_database_;
  rclcpp::Logger logger_ = rclcpp::get_logger("global_localization");
};

}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_HPP
