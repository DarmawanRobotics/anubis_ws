// hdl localizaton 
#include <mutex>
#include <memory>
#include <iostream>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <deque>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <std_srvs/srv/empty.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <pcl/filters/voxel_grid.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/voxel_grid_covariance_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <localization/pose_estimator.hpp>
#include <localization/pose_prediction.hpp>
#include <localization/sensor_data_validity.hpp>
#include <localization/static_imu_init.hpp>
#include <localization/tracking_health_monitor.hpp>
#include <localization/mode_state.h>
#include <localization/ndt_optimizer_diagnostics.hpp>
#include <localization/point_cloud_time_diagnostics.hpp>
#include <localization/global_localization.hpp>
#include <localization/global_localization_csv.hpp>
#include <localization/global_localization_gravity.hpp>
#include <localization/global_localization_integration.hpp>
#include <localization/global_localization_validation.hpp>
#include <localization/global_localization_static_confirm.hpp>
#include <localization/map_artifact_manifest.hpp>
#include <scan_descriptor/database.hpp>

#include <anubis_localization/msg/scan_matching_status.hpp>
#include <anubis_interfaces/srv/load_map.hpp>
#include <anubis_interfaces/srv/localization_state.hpp>
#include <anubis_interfaces/msg/localization.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using namespace std;

namespace localization {

class HdlLocalizationNode : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNode(const rclcpp::NodeOptions& options) : Node("localization", options) {
    tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
    // Dedicated thread + listener so we can read dog_sdk odom→base_link and publish
    // corrective map→odom (avoids stacking full pose × SDK odom).
    tf_buffer->setUsingDedicatedThread(true);
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    robot_odom_frame_id              = declare_parameter<std::string>("robot_odom_frame_id", "odom");
    odom_child_frame_id              = declare_parameter<std::string>("odom_child_frame_id", "base_link");
    send_tf_transforms               = declare_parameter<bool>("send_tf_transforms", false);
    cool_time_duration               = declare_parameter<double>("cool_time_duration", 0.5);
    reg_method                       = declare_parameter<std::string>("reg_method", "NDT_OMP");
    ndt_neighbor_search_method       = declare_parameter<std::string>("ndt_neighbor_search_method", "DIRECT7");
    ndt_neighbor_search_radius       = declare_parameter<double>("ndt_neighbor_search_radius", 2.0);
    ndt_resolution                   = declare_parameter<double>("ndt_resolution", 0.3);
    enable_robot_odometry_prediction = declare_parameter<bool>("enable_robot_odometry_prediction", true);

    use_imu     = declare_parameter<bool>("use_imu", true);
    invert_acc  = declare_parameter<bool>("invert_acc", false);
    invert_gyro = declare_parameter<bool>("invert_gyro", false);
    enable_vertical_velocity_prediction_ = declare_parameter<bool>(
      "enable_vertical_velocity_prediction", false);
    // imu static init params
    imu_init_time_         = static_cast<float>(declare_parameter<double>("imu_init_time", 3.0));
    imu_init_queue_size_   = declare_parameter<int>("imu_init_queue_size", 600);
    imu_init_max_gyro_var_ = static_cast<float>(declare_parameter<double>("imu_init_max_gyro_var", 0.05));
    imu_init_max_acce_var_ = static_cast<float>(declare_parameter<double>("imu_init_max_acce_var", 0.2));

    std::string imu_topic               = declare_parameter<std::string>("imu_topic", "/livox/imu");
    std::string points_topic            = declare_parameter<std::string>("points_topic", "/livox/lidar");
    std::string odom_topic              = declare_parameter<std::string>("odom_topic", "/odom/localization_odom");
    std::string aligned_points_topic    = declare_parameter<std::string>("aligned_points_topic", "/aligned_points");
    std::string status_topic            = declare_parameter<std::string>("status_topic", "/status");
    std::string localization_info_topic = declare_parameter<std::string>("localization_info_topic", "/localization_info");
    std::string global_map_points_topic = declare_parameter<std::string>("global_map_points_topic", "/global_map_points");
    initial_pcd_map_path_                = declare_parameter<std::string>("initial_pcd_map_path", "");

    // Load numeric parameters
    imu_data_filter_num_      = declare_parameter<int>("imu_data_filter_num", 5);
    // If a delayed IMU leaves a LiDAR frame without any usable prediction, use the
    // constant-velocity UKF propagation so repeated NDT corrections cannot collapse
    // the position covariance to zero.
    imu_stale_fallback_enabled_ = declare_parameter<bool>(
      "imu_stale_fallback_enabled", true);
    imu_stale_fallback_max_dt_s_ = declare_parameter<double>(
      "imu_stale_fallback_max_dt_s", 0.10);
    // R14: scan motion compensation (de-warping). A Livox frame is accumulated
    // over ~100 ms; while the dog walks, body rotation distorts the in-frame
    // cloud (10° rock at 3 m range ≈ 0.5 m point displacement — matches the
    // observed obs-pred gaps that tripped the innovation gate). Mid-360 point
    // timestamps show that header.stamp is the scan START, so de-warp keeps the
    // cloud in that same reference time instead of shifting it to scan end.
    motion_compensation_enable_ = declare_parameter<bool>("motion_compensation_enable", false);
    motion_compensation_slices_ = declare_parameter<int>("motion_compensation_slices", 10);
    robot_odom_topic_ = declare_parameter<std::string>("robot_odom_topic", "/odom");
    robot_odom_velocity_max_age_s_ = declare_parameter<double>(
      "robot_odom_velocity_max_age_s", 0.25);
    localization_rejection_timeout_s_ = declare_parameter<double>(
      "localization_rejection_timeout_s", 2.0);
    localization_recovery_valid_frames_ = declare_parameter<int>(
      "localization_recovery_valid_frames", 3);
    ndt_optimizer_diagnostics_enabled_ = declare_parameter<bool>(
      "ndt_optimizer_diagnostics_enabled", true);
    ndt_detailed_diagnostics_enabled_ = declare_parameter<bool>(
      "ndt_detailed_diagnostics_enabled", false);
    sensor_time_diagnostics_enabled_ = declare_parameter<bool>(
      "sensor_time_diagnostics_enabled", true);
    diagnostics_healthy_interval_ms_ = std::max<int64_t>(
      1, declare_parameter<int64_t>("diagnostics_healthy_interval_ms", 1000));
    tracking_health_.configure(
      localization_rejection_timeout_s_, localization_recovery_valid_frames_);
    globalmap_voxel_size_     = static_cast<float>(declare_parameter<double>("globalmap_voxel_size", 0.1));
    points_voxel_filter_size_ = static_cast<float>(declare_parameter<double>("points_voxel_filter_size", 0.1));
    
    min_valid_count_           = declare_parameter<int>("min_valid_count", 5);
    buffer_size_               = declare_parameter<int>("buffer_size", 10);
    localization_odom_frame_id = declare_parameter<std::string>("localization_odom_frame_id", "base_link");

    // IMU rotation matrix parameters 
    std::vector<double> init_imu_R;
    init_imu_R.reserve(9);
    declare_parameter<std::vector<double>>("init_R", std::vector<double>{});
    get_parameter("init_R", init_imu_R);
    
    // Fixed LiDAR-to-base installation transform parameters.
    std::vector<double> init_T;
    init_T.reserve(16);
    declare_parameter<std::vector<double>>("init_T", std::vector<double>{});
    get_parameter("init_T", init_T);

    // map 系重力方向：地图 z 轴与真实重力的偏差，由地图地面平面离线拟合得到。
    // 空或非法时退回 (0,0,1)。禁止用运行时 NDT 姿态改写——实测会被污染。
    std::vector<double> map_gravity_dir;
    map_gravity_dir.reserve(3);
    declare_parameter<std::vector<double>>("map_gravity_direction", std::vector<double>{});
    get_parameter("map_gravity_direction", map_gravity_dir);

    // NDT 配准的点云高度上限（见 filterByHeight 的说明）。<=0 关闭过滤。
    declare_parameter<float>("ndt_max_point_height", ndt_max_point_height_);
    get_parameter("ndt_max_point_height", ndt_max_point_height_);

    if (map_gravity_dir.size() == 3) {
      const Eigen::Vector3f dir(
        static_cast<float>(map_gravity_dir[0]),
        static_cast<float>(map_gravity_dir[1]),
        static_cast<float>(map_gravity_dir[2]));
      if (dir.allFinite() && dir.norm() > 1e-6f) {
        map_gravity_direction_ = dir.normalized();
      } else {
        RCLCPP_WARN(get_logger(),
          "map_gravity_direction 非法，使用默认 (0,0,1)");
      }
    } else if (!map_gravity_dir.empty()) {
      RCLCPP_WARN(get_logger(),
        "map_gravity_direction 需要 3 个元素，实际 %zu 个，使用默认 (0,0,1)",
        map_gravity_dir.size());
    }

    // Initial pose initialization parameters
    declare_parameter<int>("init_match_count_threshold", 5);
    get_parameter("init_match_count_threshold", init_match_count_threshold_);
    declare_parameter<float>("init_match_score_threshold", 0.15);
    get_parameter("init_match_score_threshold", init_match_score_threshold_);
    // Round 10: pose jump threshold for init consistency check (meters)
    declare_parameter<float>("init_match_pose_jump_threshold", 0.3);
    get_parameter("init_match_pose_jump_threshold", init_match_pose_jump_threshold_);
    declare_parameter<float>("ndt_score_threshold", 0.5);
    get_parameter("ndt_score_threshold", ndt_score_threshold_);
    
    // Global localization parameters
    declare_parameter<bool>("use_global_localization_init", true);
    get_parameter("use_global_localization_init", use_global_localization_init_);
    
    declare_parameter<float>("init_pose_change_threshold", 0.01);
    get_parameter("init_pose_change_threshold", init_pose_change_threshold_);
    
    declare_parameter<float>("init_quat_change_threshold", 0.01);
    get_parameter("init_quat_change_threshold", init_quat_change_threshold_);
    
    declare_parameter<float>("global_localization_timeout", 10.0);
    get_parameter("global_localization_timeout", global_localization_timeout_);

    static_imu_init_.SetParam(imu_init_time_, imu_init_queue_size_, imu_init_max_gyro_var_, imu_init_max_acce_var_);
    
    if (init_imu_R.size() != 9) {
      throw std::invalid_argument(
        "init_R is required and must contain exactly 9 row-major values");
    }
    init_rotation_matrix_ << init_imu_R[0], init_imu_R[1], init_imu_R[2],
                             init_imu_R[3], init_imu_R[4], init_imu_R[5],
                             init_imu_R[6], init_imu_R[7], init_imu_R[8];
    const Eigen::Matrix3f imu_rotation_orthogonality =
      init_rotation_matrix_.transpose() * init_rotation_matrix_ -
      Eigen::Matrix3f::Identity();
    if (!init_rotation_matrix_.allFinite() ||
        imu_rotation_orthogonality.cwiseAbs().maxCoeff() > 1e-5f ||
        std::abs(init_rotation_matrix_.determinant() - 1.0f) > 1e-5f) {
      throw std::invalid_argument("init_R must be a finite rotation matrix");
    }
    RCLCPP_INFO(get_logger(), "Fixed IMU-to-base rotation loaded from config");

    if (init_T.size() != 16) {
      throw std::invalid_argument(
        "init_T is required and must contain exactly 16 row-major values");
    }
    lidar_to_base_transform_ << init_T[0], init_T[1], init_T[2], init_T[3],
                                init_T[4], init_T[5], init_T[6], init_T[7],
                                init_T[8], init_T[9], init_T[10], init_T[11],
                                init_T[12], init_T[13], init_T[14], init_T[15];
    const Eigen::Matrix3f lidar_rotation =
      lidar_to_base_transform_.block<3, 3>(0, 0);
    const Eigen::Matrix3f lidar_rotation_orthogonality =
      lidar_rotation.transpose() * lidar_rotation - Eigen::Matrix3f::Identity();
    if (!lidar_to_base_transform_.allFinite() ||
        lidar_rotation_orthogonality.cwiseAbs().maxCoeff() > 1e-5f ||
        std::abs(lidar_rotation.determinant() - 1.0f) > 1e-5f ||
        !lidar_to_base_transform_.row(3).isApprox(
          Eigen::RowVector4f(0.0f, 0.0f, 0.0f, 1.0f), 1e-6f)) {
      throw std::invalid_argument("init_T must be a finite rigid transform");
    }
    // The current Mid-360 reports LiDAR and built-in IMU axes aligned
    // (R_imu_lidar=I), so R_base_imu and R_base_lidar must be identical.
    // Refuse a mixed configuration instead of silently feeding NDT and UKF
    // measurements expressed in different body frames.
    const float lidar_imu_rotation_error =
      (lidar_rotation - init_rotation_matrix_).cwiseAbs().maxCoeff();
    if (lidar_imu_rotation_error > 1e-5f) {
      throw std::invalid_argument(
        "init_R and init_T rotation must match for the aligned Mid-360 LiDAR/IMU axes");
    }
    RCLCPP_INFO(get_logger(), "Fixed LiDAR-to-base transform loaded from config");
    RCLCPP_INFO(get_logger(),
      "[EXTRINSIC CONFIG] imu_to_base_source=config_init_R "
      "pointcloud_source=config_init_T rotation_consistency_max_abs=%.9f "
      "direction=p_base=T_base_lidar*p_lidar "
      "imu_to_base_R=[%.6f,%.6f,%.6f;%.6f,%.6f,%.6f;%.6f,%.6f,%.6f] "
      "pointcloud_R=[%.6f,%.6f,%.6f;%.6f,%.6f,%.6f;%.6f,%.6f,%.6f] "
      "pointcloud_t=[%.6f,%.6f,%.6f]",
      lidar_imu_rotation_error,
      init_rotation_matrix_(0, 0), init_rotation_matrix_(0, 1), init_rotation_matrix_(0, 2),
      init_rotation_matrix_(1, 0), init_rotation_matrix_(1, 1), init_rotation_matrix_(1, 2),
      init_rotation_matrix_(2, 0), init_rotation_matrix_(2, 1), init_rotation_matrix_(2, 2),
      lidar_to_base_transform_(0, 0), lidar_to_base_transform_(0, 1), lidar_to_base_transform_(0, 2),
      lidar_to_base_transform_(1, 0), lidar_to_base_transform_(1, 1), lidar_to_base_transform_(1, 2),
      lidar_to_base_transform_(2, 0), lidar_to_base_transform_(2, 1), lidar_to_base_transform_(2, 2),
      lidar_to_base_transform_(0, 3), lidar_to_base_transform_(1, 3), lidar_to_base_transform_(2, 3));
    // Log loaded parameters for verification
    RCLCPP_INFO(get_logger(),
                "Loaded parameters:\n"
                "  robot_odom_frame_id: %s\n"
                "  odom_child_frame_id: %s\n"
                "  localization_odom_frame_id: %s\n"
                "  use_imu: %s\n"
                "  invert_acc: %s\n"
                "  invert_gyro: %s\n"
                "  imu_topic: %s\n"
                "  points_topic: %s\n"
                "  odom_topic: %s\n"
                "  aligned_points_topic: %s\n"
                "  status_topic: %s\n"
                "  localization_info_topic: %s\n"
                "  global_map_points_topic: %s\n"
                "  send_tf_transforms: %s\n"
                "  enable_robot_odometry_prediction: %s\n"
                "  reg_method: %s\n"
                "  ndt_neighbor_search_method: %s\n"
                "  ndt_neighbor_search_radius: %.3f\n"
                "  ndt_resolution: %.3f\n"
                "  imu_data_filter_num: %d\n"
                "  globalmap_voxel_size: %.3f\n"
                "  points_voxel_filter_size: %.3f\n"
                "  min_valid_count: %d\n"
                "  buffer_size: %d\n"
                "  imu_init_time: %.1f\n"
                "  imu_init_queue_size: %d\n"
                "  imu_init_max_gyro_var: %.3f\n"
                "  imu_init_max_acce_var: %.3f",
                robot_odom_frame_id.c_str(),
                odom_child_frame_id.c_str(),
                localization_odom_frame_id.c_str(),
                use_imu ? "true" : "false",
                invert_acc ? "true" : "false",
                invert_gyro ? "true" : "false",
                imu_topic.c_str(),
                points_topic.c_str(),
                odom_topic.c_str(),
                aligned_points_topic.c_str(),
                status_topic.c_str(),
                localization_info_topic.c_str(),
                global_map_points_topic.c_str(),
                send_tf_transforms ? "true" : "false",
                enable_robot_odometry_prediction ? "true" : "false",
                reg_method.c_str(),
                ndt_neighbor_search_method.c_str(),
                ndt_neighbor_search_radius,
                ndt_resolution,
                imu_data_filter_num_,
                globalmap_voxel_size_,
                points_voxel_filter_size_,
                min_valid_count_,
                buffer_size_,
                imu_init_time_,
                imu_init_queue_size_,
                imu_init_max_gyro_var_,
                imu_init_max_acce_var_);

    global_map_points_ptr_.reset(new pcl::PointCloud<PointT>());
    if (use_imu) {
      RCLCPP_INFO(get_logger(), "enable imu-based prediction");
      correct_imu_data_ptr_ = std::make_shared<sensor_msgs::msg::Imu>();
      imu_sub               = create_subscription<sensor_msgs::msg::Imu>(imu_topic, 256, std::bind(&HdlLocalizationNode::imu_callback, this, std::placeholders::_1));
    }
    points_sub      = create_subscription<sensor_msgs::msg::PointCloud2>(points_topic, 5, std::bind(&HdlLocalizationNode::points_callback, this, std::placeholders::_1));
    robot_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      robot_odom_topic_, 20,
      std::bind(&HdlLocalizationNode::robotOdomCallback, this, std::placeholders::_1));
    initialpose_sub =
      create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 8, std::bind(&HdlLocalizationNode::initialpose_callback, this, std::placeholders::_1));

    localization_lidar_info_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(100), // 10Hz 
                std::bind(&HdlLocalizationNode::PublishLidarLocalizationInfo, this));
    odom_publish_timer_            = this->create_wall_timer(
                std::chrono::milliseconds(50), // 20Hz
                std::bind(&HdlLocalizationNode::PublishOdomTimer, this));
    pose_pub    = create_publisher<nav_msgs::msg::Odometry>(odom_topic, 5);
    aligned_pub = create_publisher<sensor_msgs::msg::PointCloud2>(aligned_points_topic, 5);
    status_pub  = create_publisher<anubis_localization::msg::ScanMatchingStatus>(status_topic, 5);

    localization_state_srv_ = this->create_service<anubis_interfaces::srv::LocalizationState>(
            "/localization_state/service",
            std::bind(&HdlLocalizationNode::LocalizationStateCallback, this, std::placeholders::_1, std::placeholders::_2));
    load_map_service_ptr_   = create_service<anubis_interfaces::srv::LoadMap>(
            "/load_map_service", std::bind(&HdlLocalizationNode::LoadMapCallBack, this, std::placeholders::_1, std::placeholders::_2));

    // --- AprilTag anchor correction ---
    // [PATCH -- ArUco -> AprilTag migration] Detection lives in the
    // separate perception node (anubis_perception, wrapping apriltag_ros);
    // this only consumes apriltag_msgs/AprilTagDetectionArray and, on a
    // validated, confirmed discrepancy, re-anchors the estimator the same
    // way a confirmed GL pose does (see applyAprilTagAnchorCorrection
    // below) -- PoseEstimator does not expose a lighter incremental
    // absolute-pose update, so this reuses the one vetted mechanism that
    // exists rather than improvising new UKF surgery. Because
    // apriltag_ros carries no pose in its detection message (only pixel
    // corners/homography -- the pose is on /tf instead), and because
    // anubis_description publishes the full base_link->...->
    // camera_color_optical_frame chain, T_map_base is now obtained via a
    // single TF lookup (base_link -> "tag<family>:<id>") composed by
    // tf2 across that chain, combined with the known T_map_tag from the
    // apriltag_anchors.yaml reference file loaded at map-load time. There
    // is therefore no separately-maintained base_to_camera_T extrinsic
    // parameter anymore -- same reasoning as mapping_alg.cpp's
    // AprilTagObservation.
    apriltag_enable_ = this->declare_parameter<bool>("apriltag.enable", false);
    apriltag_detection_topic_ = this->declare_parameter<std::string>(
        "apriltag.detection_topic", "/perception/apriltag/detections");
    // hamming/decision_margin replace the old reprojection_error_px/
    // require_depth_validation gate -- see mapping_alg.cpp's equivalent
    // comment for why ArUco's PnP-derived gate has no AprilTag analog.
    apriltag_max_hamming_ = this->declare_parameter<int>("apriltag.max_hamming", 0);
    apriltag_min_decision_margin_ = this->declare_parameter<double>(
        "apriltag.min_decision_margin", 50.0);
    apriltag_max_jump_m_ = this->declare_parameter<double>("apriltag.max_jump_m", 0.5);
    apriltag_consistency_min_hits_ = this->declare_parameter<int>(
        "apriltag.consistency_min_hits", 3);
    apriltag_consistency_window_s_ = this->declare_parameter<double>(
        "apriltag.consistency_window_s", 2.0);
    apriltag_family_ = this->declare_parameter<std::string>(
        "apriltag.tag_family", "36h11");
    apriltag_base_frame_id_ = this->declare_parameter<std::string>(
        "apriltag.base_frame_id", "base_link");
    if (apriltag_enable_) {
      sub_apriltag_ptr_ = this->create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>(
          apriltag_detection_topic_, rclcpp::QoS(50),
          std::bind(&HdlLocalizationNode::aprilTagDetectionCallBack, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "AprilTag anchor correction enabled on '%s' (family=%s)",
                  apriltag_detection_topic_.c_str(), apriltag_family_.c_str());
    }


    localization_info_pub_ = this->create_publisher<anubis_interfaces::msg::Localization>(localization_info_topic, 10);
    global_map_pub_        = this->create_publisher<sensor_msgs::msg::PointCloud2>(global_map_points_topic, 1);
    const auto gl_diagnostic_qos = rclcpp::QoS(rclcpp::KeepLast(1))
      .reliable()
      .transient_local();
    localization_valid_pub_ = create_publisher<std_msgs::msg::Bool>(
      "/localization/valid", gl_diagnostic_qos);
    request_initialpose_pub_ = create_publisher<std_msgs::msg::Bool>(
      "/localization/request_initialpose", gl_diagnostic_qos);
    publishGlobalLocalizationDiagnostics(true);

    initialize_params();
    raw_points_ptr_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    // Initialize sensor data validity tracking
    last_lidar_data_time_ = get_clock()->now();
    last_imu_data_time_   = get_clock()->now();
    // Initialize buffer, mark all as invalid initially
    for (int i = 0; i < buffer_size_; i++) {
      lidar_status_buffer_.push_back(false);
      imu_status_buffer_.push_back(false);
    }
    // Initialize pose history
    last_valid_pose_time_   = get_clock()->now();
    has_valid_pose_history_ = false;
    
    // Initialize confidence management
    last_confidence_update_time_ = get_clock()->now();
    current_confidence_          = 0.0;
    is_extrapolating_            = false;
    
    log_counter_  = 0;
    log_interval_ = 10;
    
    RCLCPP_INFO(get_logger(), "Sensor data validity tracking initialized with buffer size: %d, min valid count: %d", 
                buffer_size_, min_valid_count_);

    if (!initial_pcd_map_path_.empty()) {
      std::string load_message;
      update_map_flag_.store(true);
      const bool loaded = loadGlobalMapFromFile(initial_pcd_map_path_, load_message);
      update_map_flag_.store(false);
      if (loaded) {
        RCLCPP_INFO(get_logger(), "Initial PCD map loaded: %s", load_message.c_str());
      } else {
        RCLCPP_ERROR(get_logger(), "Initial PCD map load failed: %s", load_message.c_str());
      }
    }
  }

private:
  localization::StaticIMUInit static_imu_init_;
  float imu_init_time_ = 3.0f;
  int imu_init_queue_size_ = 300;
  float imu_init_max_gyro_var_ = 0.05f;
  float imu_init_max_acce_var_ = 0.2f;
  // Initial pose initialization parameters
  int   init_match_count_threshold_       = 5;
  float init_match_score_threshold_       = 0.2f;  // upper bound: reject catastrophic frames
  float init_match_pose_jump_threshold_   = 0.3f;  // max allowed frame-to-frame pose jump (m)
  float ndt_score_threshold_              = 0.5f;  // reject NDT matches with fitness score above this
  // Initial pose initialization state variables
  int init_match_count_ = 0;
  // Round 10: pose consistency tracking for init verification
  Eigen::Vector3f last_verify_pos_{0, 0, 0};
  bool has_init_verify_pose_ = false;
  pcl::Registration<PointT, PointT>::Ptr create_registration() {
    if (reg_method == "NDT_OMP") {
      RCLCPP_INFO(get_logger(), "NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);
      ndt->setMaximumIterations(ndt_max_iterations_);
      ndt->setIterationDiagnosticsEnabled(ndt_detailed_diagnostics_enabled_);
      if (ndt_neighbor_search_method == "DIRECT1") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          RCLCPP_INFO(get_logger(), "search_method KDTREE is selected");
        } else {
          RCLCPP_WARN(get_logger(), "invalid search method was given");
          RCLCPP_WARN(get_logger(), "default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      // [2026-08-17] 保留一个 typed 弱引用，仅用于读迭代次数做诊断：
      // 基类 pcl::Registration 没有 getFinalNumIteration()。迭代若常打满
      // setMaximumIterations(15)，说明初值差或 ndt_resolution 与场景不匹配，
      // 是 NDT 段耗时高的直接原因。
      ndt_diag_ = ndt;
      return ndt;
    }
    RCLCPP_ERROR_STREAM(get_logger(), "unknown registration method:" << reg_method);
    return nullptr;
  }

  void loadGlobalLocalizationParams() {
    auto load = [this](const char* name, auto& value) {
      using Value = std::decay_t<decltype(value)>;
      value = declare_parameter<Value>(name, value);
    };
    auto load_read_only = [this](const char* name, auto& value,
                                 const char* description) {
      using Value = std::decay_t<decltype(value)>;
      rcl_interfaces::msg::ParameterDescriptor descriptor;
      descriptor.description = description;
      descriptor.read_only = true;
      value = declare_parameter<Value>(name, value, descriptor);
    };

    load("level0_grid_stride_xy", gl_params_.level0_grid_stride_xy);
    load("gl_recall_source", gl_params_.gl_recall_source);
    load("retrieval_fallback_to_grid", gl_params_.retrieval_fallback_to_grid);
    load("retrieval_topk", gl_params_.retrieval_topk);
    load("retrieval_nms_xy", gl_params_.retrieval_nms_xy);
    load("retrieval_max_distance", gl_params_.retrieval_max_distance);
    load("retrieval_min_points", gl_params_.retrieval_min_points);
    load("retrieval_yaw_half_range_deg", gl_params_.retrieval_yaw_half_range_deg);
    load("retrieval_yaw_step_deg", gl_params_.retrieval_yaw_step_deg);
    load("retrieval_search_radius_xy", gl_params_.retrieval_search_radius_xy);
    load("retrieval_seed_stride_xy", gl_params_.retrieval_seed_stride_xy);
    load("level0_yaw_samples", gl_params_.level0_yaw_samples);
    load("level0_top_regions", gl_params_.level0_top_regions);
    load("level0_distance_field_resolution", gl_params_.level0_distance_field_resolution);
    load("level0_max_obstacle_distance", gl_params_.level0_max_obstacle_distance);
    load_read_only("level0_base_ground_offset", gl_params_.level0_base_ground_offset,
                   "Calibrated base-to-ground offset used by M2b; immutable after startup");
    load("level0_min_point_height", gl_params_.level0_min_point_height);
    load("level0_max_point_height", gl_params_.level0_max_point_height);
    load("level0_min_range", gl_params_.level0_min_range);
    load("level0_max_range", gl_params_.level0_max_range);
    load("level0_max_scan_points", gl_params_.level0_max_scan_points);
    load("level0_min_valid_points", gl_params_.level0_min_valid_points);
    load("level0_gravity_align_enabled", gl_params_.level0_gravity_align_enabled);
    load("level0_raw_imu_gravity_filter_enabled", gl_params_.level0_raw_imu_gravity_filter_enabled);
    load("level0_raw_imu_gravity_window_s", gl_params_.level0_raw_imu_gravity_window_s);
    load("level0_raw_imu_gravity_max_angular_velocity_rps",
         gl_params_.level0_raw_imu_gravity_max_angular_velocity_rps);
    load("level0_raw_imu_gravity_min_samples", gl_params_.level0_raw_imu_gravity_min_samples);
    load("level0_gravity_time_offset_ms", gl_params_.level0_gravity_time_offset_ms);
    load("level0_gravity_max_age_ms", gl_params_.level0_gravity_max_age_ms);
    load("level0_gravity_max_uncertainty_deg", gl_params_.level0_gravity_max_uncertainty_deg);
    load("level0_anchor_max_abs_rp_deg", gl_params_.level0_anchor_max_abs_rp_deg);
    load("level0_unaligned_max_abs_rp_deg", gl_params_.level0_unaligned_max_abs_rp_deg);
    load("level0_anchor_max_angular_velocity_rps", gl_params_.level0_anchor_max_angular_velocity_rps);
    load("level0_gravity_max_accel_residual_mps2", gl_params_.level0_gravity_max_accel_residual_mps2);
    load("level0_anchor_max_linear_velocity_mps", gl_params_.level0_anchor_max_linear_velocity_mps);
    load("level0_anchor_recapture_limit", gl_params_.level0_anchor_recapture_limit);
    load("level0_min_valid_projection_ratio", gl_params_.level0_min_valid_projection_ratio);
    load("level0_invalid_projection_penalty", gl_params_.level0_invalid_projection_penalty);
    load("level0_boundary_padding_m", gl_params_.level0_boundary_padding_m);
    load("level0_max_boundary_unknown_ratio", gl_params_.level0_max_boundary_unknown_ratio);
    load("level0_region_nms_xy", gl_params_.level0_region_nms_xy);
    load("level0_region_nms_yaw_deg", gl_params_.level0_region_nms_yaw_deg);
    load("level1_search_radius_xy", gl_params_.level1_search_radius_xy);
    load("level1_seed_stride_xy", gl_params_.level1_seed_stride_xy);
    load("level1_yaw_step_deg", gl_params_.level1_yaw_step_deg);
    load("max_candidates_total", gl_params_.max_candidates_total);
    load("max_refine_candidates", gl_params_.max_refine_candidates);
    load("seed_dedup_xy", gl_params_.seed_dedup_xy);
    load("seed_dedup_yaw_deg", gl_params_.seed_dedup_yaw_deg);
    load("grid_z_percentile", gl_params_.grid_z_percentile);
    load("coarse_leaf_map", gl_params_.coarse_leaf_map);
    load("coarse_leaf_src", gl_params_.coarse_leaf_src);
    load("coarse_max_iter", gl_params_.coarse_max_iter);
    load("coarse_roi_radius", gl_params_.coarse_roi_radius);
    load("coarse_max_corr_dist", gl_params_.coarse_max_corr_dist);
    load("coarse_min_overlap", gl_params_.coarse_min_overlap);
    load("coarse_min_inliers", gl_params_.coarse_min_inliers);
    load("coarse_max_p90_residual", gl_params_.coarse_max_p90_residual);
    load("refine_leaf_src", gl_params_.refine_leaf_src);
    load("refine_max_iter", gl_params_.refine_max_iter);
    load("refine_fitness_threshold", gl_params_.refine_fitness_threshold);
    load("max_correction_xy", gl_params_.max_correction_xy);
    load("max_correction_z", gl_params_.max_correction_z);
    load("max_yaw_delta_deg", gl_params_.max_yaw_delta_deg);
    load("max_rp_delta_deg", gl_params_.max_rp_delta_deg);
    load("max_abs_roll_deg", gl_params_.max_abs_roll_deg);
    load("max_abs_pitch_deg", gl_params_.max_abs_pitch_deg);
    load("cluster_xy", gl_params_.cluster_xy);
    load("cluster_yaw_deg", gl_params_.cluster_yaw_deg);
    load("support_seed_xy", gl_params_.support_seed_xy);
    load("min_support_groups", gl_params_.min_support_groups);
    load("ambiguous_pose_distance", gl_params_.ambiguous_pose_distance);
    load("ambiguous_score_ratio", gl_params_.ambiguous_score_ratio);
    load("gl_confirm_frames", gl_params_.gl_confirm_frames);
    load("gl_confirm_xy_tol", gl_params_.gl_confirm_xy_tol);
    load("gl_confirm_yaw_tol_deg", gl_params_.gl_confirm_yaw_tol_deg);
    load("gl_probe_period_s", gl_params_.gl_probe_period_s);
    load("gl_attempt_timeout_s", gl_params_.gl_attempt_timeout_s);
    load("min_retry_interval_s", gl_params_.min_retry_interval_s);
    load("per_call_deadline_ms", gl_params_.per_call_deadline_ms);
    load("episode_timeout_s", gl_params_.episode_timeout_s);
    load("max_scan_age_after_gl_s", gl_params_.max_scan_age_after_gl_s);
    load("shadow_gl_budget_ms", gl_params_.shadow_gl_budget_ms);
    load("shadow_gl_period_s", gl_params_.shadow_gl_period_s);
    load_read_only("allow_degraded_fallback", gl_params_.allow_degraded_fallback,
                   "Whether degraded global-localization fallback is allowed; immutable after startup");
    load_read_only("flat_single_level_map", gl_params_.flat_single_level_map,
                   "Whether the loaded map is approved as a single-level map; immutable after startup");
    load_read_only("auto_confirm_enabled", gl_params_.auto_confirm_enabled,
                   "Request automatic M2b confirmation; immutable after startup");
    load_read_only("auto_confirm_verify_timeout_s",
                   gl_params_.auto_confirm_verify_timeout_s,
                   "Automatic-confirmation verification timeout; immutable after startup");
    load_read_only("m2b_approval_id", gl_params_.m2b_approval_id,
                   "M2b approval evidence identifier; immutable after startup");
    load("advisory_suggestion_enabled", gl_params_.advisory_suggestion_enabled);
    load("advisory_min_confidence", gl_params_.advisory_min_confidence);
    load("advisory_calibration_id", gl_params_.advisory_calibration_id);
    load("dump_candidates_csv", gl_params_.dump_candidates_csv);
    load("candidates_csv_path", gl_params_.candidates_csv_path);
    load("candidates_csv_episode_id", gl_params_.candidates_csv_episode_id);
    load("candidate_debug_log", gl_params_.candidate_debug_log);
    const auto parameter_overrides =
      get_node_parameters_interface()->get_parameter_overrides();
    const bool official_runtime_explicit =
      parameter_overrides.find("gl_episode_runtime_enabled") !=
      parameter_overrides.end();
    const bool legacy_runtime_explicit =
      parameter_overrides.find("gl_auto_init_enabled") !=
      parameter_overrides.end();
    rcl_interfaces::msg::ParameterDescriptor runtime_descriptor;
    runtime_descriptor.description =
      "Enable the M2b global-localization episode runtime; immutable after startup";
    runtime_descriptor.read_only = true;
    gl_online_shadow_enabled_ = declare_parameter<bool>(
      "gl_online_shadow_enabled", false, runtime_descriptor);
    const bool official_runtime_value = declare_parameter<bool>(
      "gl_episode_runtime_enabled", false, runtime_descriptor);
    const bool legacy_runtime_value = declare_parameter<bool>(
      "gl_auto_init_enabled", false, runtime_descriptor);
    const GLEpisodeRuntimeResolution runtime_resolution =
      resolveGLEpisodeRuntimeEnabled(
        official_runtime_explicit
          ? std::optional<bool>(official_runtime_value)
          : std::nullopt,
        legacy_runtime_explicit
          ? std::optional<bool>(legacy_runtime_value)
          : std::nullopt);
    gl_episode_runtime_enabled_ = runtime_resolution.enabled;
    gl_runtime_parameter_conflict_ = runtime_resolution.conflict;
    if (runtime_resolution.conflict) {
      RCLCPP_ERROR(
        get_logger(),
        "Conflicting global-localization runtime parameters: "
        "gl_episode_runtime_enabled=%s, deprecated gl_auto_init_enabled=%s; "
        "effective episode runtime=false",
        official_runtime_value ? "true" : "false",
        legacy_runtime_value ? "true" : "false");
    } else if (runtime_resolution.legacy_parameter_used) {
      RCLCPP_WARN(
        get_logger(),
        "gl_auto_init_enabled is deprecated; use gl_episode_runtime_enabled "
        "(effective=%s)",
        gl_episode_runtime_enabled_ ? "true" : "false");
    }
    gl_relocalize_on_tracking_failure_ = declare_parameter<bool>(
      "gl_relocalize_on_tracking_failure", false);
    gl_odom_debug_log_ = declare_parameter<bool>("gl_odom_debug_log", false);
    gl_odom_debug_log_interval_s_ = declare_parameter<double>(
      "gl_odom_debug_log_interval_s", 1.0);
  }

  void initialize_params() {
    voxel_filter_ptr_->setLeafSize(points_voxel_filter_size_, points_voxel_filter_size_, points_voxel_filter_size_);
    registration = create_registration();
    loadGlobalLocalizationParams();
    gl_raw_imu_gravity_filter_.configure(
      gl_params_.level0_raw_imu_gravity_window_s,
      gl_params_.level0_raw_imu_gravity_min_samples,
      gl_params_.level0_raw_imu_gravity_max_angular_velocity_rps,
      gl_params_.level0_gravity_max_accel_residual_mps2);

    // Initialize global localization
    if (use_global_localization_init_) {
      try {
        global_localization_ptr_ = std::make_shared<GlobalLocalization>();
        global_localization_ptr_->setParams(gl_params_);
        global_localization_ptr_->setBaseFromLidar(
          lidar_to_base_transform_.cast<double>());
        if (gl_params_.dump_candidates_csv) {
          gl_candidates_csv_writer_ = std::make_unique<GLCandidateCsvWriter>(
              std::filesystem::path(gl_params_.candidates_csv_path));
          RCLCPP_INFO(
            get_logger(),
            "Global localization candidate CSV enabled: episode=%s path=%s",
            gl_params_.candidates_csv_episode_id.c_str(),
            gl_params_.candidates_csv_path.c_str());
        }
        const bool auto_confirm_effective =
          automaticApprovalContractSatisfied(gl_params_);
        RCLCPP_INFO(
          get_logger(),
          "Global localization initialized: online_shadow=%s "
          "episode_runtime_effective=%s auto_confirm_requested=%s "
          "auto_confirm_effective=%s flat_map=%s tracking_failure_relocalize=%s",
          gl_online_shadow_enabled_ ? "true" : "false",
          gl_episode_runtime_enabled_ ? "true" : "false",
          gl_params_.auto_confirm_enabled ? "true" : "false",
          auto_confirm_effective ? "true" : "false",
          gl_params_.flat_single_level_map ? "true" : "false",
          gl_relocalize_on_tracking_failure_ ? "true" : "false");
        if (gl_params_.auto_confirm_enabled && !auto_confirm_effective) {
          const std::vector<std::string> contract_errors =
            validateAutoConfirmContract(gl_params_);
          RCLCPP_ERROR(
            get_logger(),
            "auto_confirm_requested=true auto_confirm_effective=false; "
            "GL episode and decision-shadow remain enabled, but confirmed "
            "poses will not be applied");
          for (const std::string& error : contract_errors) {
            RCLCPP_ERROR(get_logger(), "[GL auto-confirm contract] %s",
                         error.c_str());
          }
        }
        RCLCPP_INFO(
          get_logger(),
          "Global localization recall: source=%s topk=%d nms_xy=%.2f fallback_to_grid=%s",
          gl_params_.gl_recall_source.c_str(), gl_params_.retrieval_topk,
          gl_params_.retrieval_nms_xy,
          gl_params_.retrieval_fallback_to_grid ? "true" : "false");
        RCLCPP_INFO(
          get_logger(),
          "Global localization static confirm: frames=%d xy_tol=%.2fm yaw_tol=%.1fdeg probe_period=%.2fs timeout=%.1fs",
          gl_params_.gl_confirm_frames, gl_params_.gl_confirm_xy_tol,
          gl_params_.gl_confirm_yaw_tol_deg, gl_params_.gl_probe_period_s,
          gl_params_.gl_attempt_timeout_s);
        RCLCPP_INFO(
          get_logger(),
          "Global localization budgets: per_call_deadline_ms=%.3f episode_timeout_s=%.3f shadow_gl_budget_ms=%.3f",
          gl_params_.per_call_deadline_ms,
          gl_params_.episode_timeout_s,
          gl_params_.shadow_gl_budget_ms);
        RCLCPP_INFO(
          get_logger(),
          "Raw IMU gravity filter configured: enabled=%s window=%.3fs min_samples=%d",
          gl_params_.level0_raw_imu_gravity_filter_enabled ? "true" : "false",
          gl_params_.level0_raw_imu_gravity_window_s,
          gl_params_.level0_raw_imu_gravity_min_samples);
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "Failed to initialize global localization: %s", e.what());
        use_global_localization_init_ = false;
      }
    }
    // initialize pose estimator
    specify_init_pose_ = declare_parameter<bool>("specify_init_pose", true);
    if (specify_init_pose_) {
      RCLCPP_INFO(get_logger(), "initialize pose estimator with specified parameters!!"); 
      init_pos_x_ = declare_parameter<double>("init_pos_x", 0.0);
      init_pos_y_ = declare_parameter<double>("init_pos_y", 0.0);
      init_pos_z_ = declare_parameter<double>("init_pos_z", 0.0);
      init_ori_w_ = declare_parameter<double>("init_ori_w", 1.0);
      init_ori_x_ = declare_parameter<double>("init_ori_x", 0.0);
      init_ori_y_ = declare_parameter<double>("init_ori_y", 0.0);
      init_ori_z_ = declare_parameter<double>("init_ori_z", 0.0);
  
      Eigen::Vector3f config_pos(init_pos_x_, init_pos_y_, init_pos_z_);
      Eigen::Quaternionf config_quat(init_ori_w_, init_ori_x_, init_ori_y_, init_ori_z_);
      last_init_pos_ = config_pos;
      last_init_quat_ = config_quat;
      has_set_init_pose_ = true;
      last_pose_source_ = "config";
      // Static gravity initialization applies roll/pitch once IMU init completes.
      Eigen::Quaternionf initial_quat = last_init_quat_;
      pose_estimator.reset(new localization::PoseEstimator(
        registration,
        get_clock()->now(),
        last_init_pos_,
        initial_quat,
        cool_time_duration,
        ndt_score_threshold_,
        !enable_vertical_velocity_prediction_
      ));
      // static_imu_init_ may already have finished (e.g. re-init on restart) by the
      // time this estimator is created; a fresh PoseEstimator always starts with
      // zero bias, so re-apply immediately if a result is already available.
      applyImuBiasIfReady();
      pose_estimator->clear_trust_anchor();
      // Log Euler angles from the configured pose. Static gravity initialization
      // may subsequently replace roll/pitch while preserving this map yaw.
      Eigen::Vector3f euler_deg = initial_quat.toRotationMatrix().eulerAngles(0, 1, 2) * 180.0 / M_PI;
      RCLCPP_INFO(get_logger(), "Initial pose estimator created - Position: [%.3f, %.3f, %.3f], "
                   "Euler(deg): roll=%.2f pitch=%.2f yaw=%.2f",
                   last_init_pos_.x(), last_init_pos_.y(), last_init_pos_.z(),
                   euler_deg.x(), euler_deg.y(), euler_deg.z());
    }
  }

private:
  struct StreamTimingState {
    bool seen = false;
    int64_t stamp_ns = 0;
    int64_t receipt_ns = 0;
    double stamp_gap_ms = 0.0;
    double receipt_gap_ms = 0.0;
    double receipt_age_ms = 0.0;
    uint64_t out_of_order_count = 0;
  };

  bool updateStreamTiming(
      StreamTimingState& state, const int64_t stamp_ns,
      const int64_t receipt_ns) {
    std::lock_guard<std::mutex> lock(sensor_time_mutex_);
    if (state.seen) {
      state.stamp_gap_ms = static_cast<double>(stamp_ns - state.stamp_ns) * 1e-6;
      state.receipt_gap_ms =
        static_cast<double>(receipt_ns - state.receipt_ns) * 1e-6;
      if (stamp_ns <= state.stamp_ns) {
        ++state.out_of_order_count;
        state.receipt_ns = receipt_ns;
        state.receipt_age_ms =
          static_cast<double>(receipt_ns - stamp_ns) * 1e-6;
        return false;
      }
    }
    state.seen = true;
    state.stamp_ns = stamp_ns;
    state.receipt_ns = receipt_ns;
    state.receipt_age_ms = static_cast<double>(receipt_ns - stamp_ns) * 1e-6;
    return true;
  }

  void logSensorTimeDiagnostics(
      const sensor_msgs::msg::PointCloud2& cloud,
      const PointCloudTimeDiagnostics& point_time,
      const double parse_ms,
      const int64_t callback_ns) {
    if (!sensor_time_diagnostics_enabled_) {
      return;
    }

    StreamTimingState lidar;
    StreamTimingState imu;
    StreamTimingState odom;
    {
      std::lock_guard<std::mutex> lock(sensor_time_mutex_);
      lidar = lidar_timing_;
      imu = imu_timing_;
      odom = odom_timing_;
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double header_ns = static_cast<double>(
      rclcpp::Time(cloud.header.stamp).nanoseconds());
    const double point_first_ms = point_time.valid
      ? (point_time.first_stamp_ns - header_ns) * 1e-6 : nan;
    const double point_last_ms = point_time.valid
      ? (point_time.last_stamp_ns - header_ns) * 1e-6 : nan;
    const double point_span_ms = point_time.valid
      ? (point_time.max_stamp_ns - point_time.min_stamp_ns) * 1e-6 : nan;
    const double point_end_age_ms = point_time.valid
      ? (static_cast<double>(callback_ns) - point_time.max_stamp_ns) * 1e-6 : nan;

    const char* header_reference = "unknown";
    if (point_time.valid && std::abs(point_first_ms) <= 5.0) {
      header_reference = "scan_start";
    } else if (point_time.valid && std::abs(point_last_ms) <= 5.0) {
      header_reference = "scan_end";
    }

    const double imu_to_start_ms = imu.seen && point_time.valid
      ? (static_cast<double>(imu.stamp_ns) - point_time.min_stamp_ns) * 1e-6 : nan;
    const double imu_to_end_ms = imu.seen && point_time.valid
      ? (static_cast<double>(imu.stamp_ns) - point_time.max_stamp_ns) * 1e-6 : nan;
    const double odom_to_start_ms = odom.seen && point_time.valid
      ? (static_cast<double>(odom.stamp_ns) - point_time.min_stamp_ns) * 1e-6 : nan;
    const double odom_to_end_ms = odom.seen && point_time.valid
      ? (static_cast<double>(odom.stamp_ns) - point_time.max_stamp_ns) * 1e-6 : nan;

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), diagnostics_healthy_interval_ms_,
      "[TIME LIDAR] header_ref=%s header_age=%.2fms first-header=%+.2fms "
      "last-header=%+.2fms span=%.2fms point_end_age=%.2fms "
      "stamp_gap=%.2fms receipt_gap=%.2fms point_ts=%zu regressions=%zu "
      "max_back=%.3fms max_forward=%.3fms parse=%.2fms",
      header_reference, lidar.receipt_age_ms, point_first_ms, point_last_ms,
      point_span_ms, point_end_age_ms, lidar.stamp_gap_ms,
      lidar.receipt_gap_ms, point_time.valid_points,
      point_time.timestamp_regressions,
      point_time.max_timestamp_regression_ns * 1e-6,
      point_time.max_forward_step_ns * 1e-6, parse_ms);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), diagnostics_healthy_interval_ms_,
      "[TIME SYNC] imu_seen=%d imu-start=%+.2fms imu-end=%+.2fms "
      "imu_age=%.2fms imu_gap=%.2fms imu_receipt_gap=%.2fms imu_ooo=%lu | "
      "odom_seen=%d odom-start=%+.2fms odom-end=%+.2fms odom_age=%.2fms "
      "odom_gap=%.2fms odom_receipt_gap=%.2fms odom_ooo=%lu odom_stamp_source=jetson_publish",
      imu.seen ? 1 : 0, imu_to_start_ms, imu_to_end_ms, imu.receipt_age_ms,
      imu.stamp_gap_ms, imu.receipt_gap_ms,
      static_cast<unsigned long>(imu.out_of_order_count),
      odom.seen ? 1 : 0, odom_to_start_ms, odom_to_end_ms,
      odom.receipt_age_ms, odom.stamp_gap_ms, odom.receipt_gap_ms,
      static_cast<unsigned long>(odom.out_of_order_count));

    if (!point_time.field_present || !point_time.field_supported) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "[TIME] PointCloud2 timestamp field unavailable or not FLOAT64; "
        "per-point LiDAR timing cannot be verified (present=%d supported=%d)",
        point_time.field_present ? 1 : 0,
        point_time.field_supported ? 1 : 0);
    } else if (point_time.nonfinite_points > 0) {
      RCLCPP_WARN(
        get_logger(),
        "[TIME] Non-finite per-point LiDAR timestamps: nonfinite=%zu valid=%zu",
        point_time.nonfinite_points, point_time.valid_points);
    } else if (point_time.timestamp_regressions > 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "[TIME] PointCloud2 point order is not timestamp-monotonic: "
        "regressions=%zu max_back=%.3fms. Index-based deskew timing is approximate.",
        point_time.timestamp_regressions,
        point_time.max_timestamp_regression_ns * 1e-6);
    }
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
    const int64_t imu_stamp_ns = rclcpp::Time(imu_msg->header.stamp).nanoseconds();
    int64_t latest_stamp_ns = latest_imu_stamp_ns_.load(std::memory_order_relaxed);
    while (imu_stamp_ns > latest_stamp_ns &&
           !latest_imu_stamp_ns_.compare_exchange_weak(
             latest_stamp_ns, imu_stamp_ns, std::memory_order_relaxed)) {
    }
    if (sensor_time_diagnostics_enabled_) {
      const int64_t receipt_ns = get_clock()->now().nanoseconds();
      const int64_t stamp_ns = imu_stamp_ns;
      if (!updateStreamTiming(imu_timing_, stamp_ns, receipt_ns)) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[TIME] Out-of-order IMU timestamp detected");
      }
    }
    // DEBUG (2026-07-23): print raw pre-transform accel/gyro once, to verify whether
    // /livox/imu linear_acceleration is already in m/s^2 (should read ~9.8 on the
    // gravity axis at rest) or in g (should read ~1.0), before we scale it by 9.81 below.
    // Remove once the unit has been confirmed from a real static test log.
    RCLCPP_INFO_ONCE(get_logger(),
      "[IMU DEBUG] raw (pre-transform) linear_acceleration=[%.4f, %.4f, %.4f] angular_velocity=[%.4f, %.4f, %.4f]",
      imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z,
      imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);

    const double acc_sign = invert_acc ? -1.0 : 1.0;
    const double gyro_sign = invert_gyro ? -1.0 : 1.0;
    correct_imu_data_ptr_ = imu_msg;
    Eigen::Vector3f acceleration(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    // Apply the fixed IMU-to-base extrinsic and convert g to m/s^2.
    acceleration = static_cast<float>(acc_sign) *
      init_rotation_matrix_ * acceleration * 9.81f;
    correct_imu_data_ptr_->linear_acceleration.x = acceleration.x();
    correct_imu_data_ptr_->linear_acceleration.y = acceleration.y();
    correct_imu_data_ptr_->linear_acceleration.z = acceleration.z();
    Eigen::Vector3f angular_velocity(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    // Apply the same fixed IMU-to-base extrinsic to angular velocity.
    angular_velocity = static_cast<float>(gyro_sign) *
      init_rotation_matrix_ * angular_velocity;
    correct_imu_data_ptr_->angular_velocity.x = angular_velocity.x();
    correct_imu_data_ptr_->angular_velocity.y = angular_velocity.y();
    correct_imu_data_ptr_->angular_velocity.z = angular_velocity.z();
    recordGravityAlignmentSample(
      *imu_msg, acceleration.cast<double>(), angular_velocity.cast<double>());
    // Update latest IMU angular velocity for extrapolation
    latest_angular_velocity_ = angular_velocity;
    updateImuStatus(true);
    if (!static_imu_init_.InitSuccess()) {
      static_imu_init_.AddIMUData(correct_imu_data_ptr_);
      if (static_imu_init_.InitSuccess()) {
        const Eigen::Vector3d gyro_mean = static_imu_init_.GetInitBg();
        const Eigen::Vector3d gyro_var = static_imu_init_.GetCovGyro();
        const Eigen::Vector3d acc_mean = static_imu_init_.GetInitBa();
        const Eigen::Vector3d acc_var = static_imu_init_.GetCovAcce();
        RCLCPP_INFO(get_logger(),
          "[IMU INIT] result=success attempts=%zu samples=%zu duration=%.3fs "
          "gyro_mean=[%.7f,%.7f,%.7f] gyro_var=[%.3e,%.3e,%.3e] "
          "gyro_var_norm=%.3e gyro_limit=%.3e gyro_gate=enforced "
          "acc_mean=[%.5f,%.5f,%.5f] acc_norm=%.5f "
          "acc_var=[%.3e,%.3e,%.3e] acc_var_norm=%.3e acc_limit=%.3e "
          "acc_gate=diagnostic_only acc_would_pass=%d frame=base_after_init_R",
          static_imu_init_.GetInitAttemptCount(),
          static_imu_init_.GetLastAttemptSampleCount(),
          static_imu_init_.GetLastAttemptDurationSeconds(),
          gyro_mean.x(), gyro_mean.y(), gyro_mean.z(),
          gyro_var.x(), gyro_var.y(), gyro_var.z(), gyro_var.norm(),
          imu_init_max_gyro_var_,
          acc_mean.x(), acc_mean.y(), acc_mean.z(), acc_mean.norm(),
          acc_var.x(), acc_var.y(), acc_var.z(), acc_var.norm(),
          imu_init_max_acce_var_, acc_var.norm() <= imu_init_max_acce_var_ ? 1 : 0);
        RCLCPP_INFO(get_logger(),
          "Static gravity estimate ready; init_R remains the fixed IMU-to-base extrinsic");
        std::lock_guard<std::mutex> lock(pose_estimator_mutex);
        applyImuBiasIfReady();
      }
    } else {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      imu_data.push_back(correct_imu_data_ptr_);
    }
  }

  /**
   * @brief Push the base-frame biases estimated by static IMU initialization into
   * the pose estimator. Must be called whenever a new PoseEstimator is created after
   * static_imu_init_ has already succeeded because a new estimator starts at zero bias.
   *
   * imu_callback applies the fixed IMU-to-base extrinsic before samples reach
   * StaticIMUInit. Keep gravity and bias calculations in that base frame here; the
   * runtime model performs the only remaining body-to-map rotation.
   */
  void applyImuBiasIfReady() {
    if (!pose_estimator) {
      return;
    }
    // map 系重力方向是地图的固定属性，与 IMU 静态初始化是否完成无关 —— 无条件
    // 先设。本函数在每个 PoseEstimator 构造点之后都会被调用，因此这里设置可以
    // 保证新建的估计器一定拿到基准（新增构造点时也不会漏）。
    pose_estimator->set_map_gravity_direction(map_gravity_direction_);
    if (!static_imu_init_.InitSuccess()) {
      return;
    }
    // StaticIMUInit receives the already transformed IMU sample, so its gravity and
    // bias estimates are in base frame. Do not apply init_R a second time here.
    Eigen::Vector3f gravity_dir     = static_imu_init_.GetGravity().cast<float>();
    Eigen::Vector3f gravity_body    = gravity_dir * 9.80665f;
    Eigen::Vector3f acc_bias  = static_imu_init_.GetInitBa().cast<float>() + gravity_body;
    Eigen::Vector3f gyro_bias = static_imu_init_.GetInitBg().cast<float>();

    pose_estimator->set_initial_biases(acc_bias, gyro_bias);

    // Gravity initializes roll/pitch only. Preserve the map yaw supplied by GL,
    // RViz, or config; PoseSystem performs the only body-to-map rotation at runtime.
    const Eigen::Vector3f up_body = -gravity_dir.normalized();
    const float yaw = rotationToRpy(pose_estimator->quat().toRotationMatrix()).z();
    const Eigen::Quaternionf initial_orientation =
      orientationFromGravityPreservingYaw(up_body, yaw);
    pose_estimator->set_orientation(initial_orientation);

    // [2026-08-17] 用同一个已收敛的重力方向播种锚定的低通滤波器。
    // 不播种的话，新建的 PoseEstimator 需要 20 帧预热才能启用重力改写，而落位后
    // 第一次 correct() 就发生在预热期内——实测四次落位无一例外：首帧 grav=0，
    // NDT 把 applyImuBiasIfReady 刚设好的 pitch 从 -5.7° 拉到 +0.6°，锚定上线后
    // 才报 gerr=10~14°，此时污染已进 UKF。
    // 这里设置的 orientation 与播种的重力方向同源，两者必须一起生效。
    pose_estimator->seed_gravity_estimate(up_body, get_clock()->now());

    RCLCPP_INFO(get_logger(),
      "Applied base-frame IMU initialization: acc_bias=[%.4f, %.4f, %.4f], "
      "gyro_bias=[%.4f, %.4f, %.4f], gravity_dir=[%.4f, %.4f, %.4f]",
      acc_bias.x(), acc_bias.y(), acc_bias.z(),
      gyro_bias.x(), gyro_bias.y(), gyro_bias.z(),
      gravity_dir.x(), gravity_dir.y(), gravity_dir.z());
  }

  void robotOdomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
    if (sensor_time_diagnostics_enabled_) {
      const int64_t receipt_ns = get_clock()->now().nanoseconds();
      const int64_t stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
      if (!updateStreamTiming(odom_timing_, stamp_ns, receipt_ns)) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[TIME] Out-of-order SDK odom timestamp detected");
      }
    }
    const Eigen::Vector3f velocity(
      msg->twist.twist.linear.x,
      msg->twist.twist.linear.y,
      msg->twist.twist.linear.z);
    if (!velocity.allFinite()) {
      return;
    }
    std::lock_guard<std::mutex> lock(robot_odom_mutex_);
    latest_robot_odom_velocity_body_ = velocity;
    latest_robot_odom_stamp_ = rclcpp::Time(msg->header.stamp);
    has_robot_odom_velocity_ = true;
  }

  bool lookupRobotOdomPose(
      const rclcpp::Time& stamp, Eigen::Matrix4f& odom_pose) const {
    odom_pose = Eigen::Matrix4f::Identity();
    if (robot_odom_frame_id == odom_child_frame_id) {
      return true;
    }
    try {
      if (!tf_buffer->canTransform(
          robot_odom_frame_id, odom_child_frame_id, stamp,
          tf2::durationFromSec(0.0))) {
        return false;
      }
      const geometry_msgs::msg::TransformStamped transform =
        tf_buffer->lookupTransform(
          robot_odom_frame_id, odom_child_frame_id, stamp);
      odom_pose = tf2::transformToEigen(transform).cast<float>().matrix();
      return odom_pose.allFinite();
    } catch (const tf2::TransformException&) {
      return false;
    }
  }

  // R14: de-warp a base-frame scan so every point is expressed in base_link at
  // the scan START time (the Mid-360 header timestamp). Rationale: a Livox frame accumulates over ~100 ms; while
  // the dog walks, body rotation distorts the in-frame cloud (10 deg rock at 3 m
  // range ≈ 0.5 m displacement of far points — the same magnitude as the
  // obs-pred gaps that tripped the innovation gate during navigation).
  //
  // Method: points_callback first applies T_base_lidar. Integrate the buffered
  // gyro (also transformed into base frame by imu_callback; units rad/s, sign handled like the
  // predict loop) from scan start to scan end, then rotate each point from its
  // capture time back to scan start. Points are assumed to arrive in capture order
  // (Livox publishes in scan order), so slice-by-index ≈ slice-by-time at
  // 10 ms resolution. Translation uses fresh SDK odometry velocity in the odom
  // child frame. If that velocity is stale, translation compensation is skipped.
  //
  // Safety: if the IMU window is too sparse (driver hiccup / CPU stall) the scan
  // is left untouched — never compensate with garbage data.
  void motionCompensateScan(
      pcl::PointCloud<PointT>::Ptr cloud, const rclcpp::Time& scan_stamp,
      const PointCloudTimeDiagnostics* point_time = nullptr) {
    // This diagnostic deliberately reports the *assumed* deskew window next to
    // the per-point window measured from PointCloud2.  It is intentionally
    // read-only: no timing source, transform, or NDT input is changed here.
    dsk_ms_win_ = 0.0;
    dsk_ms_gyro_ = 0.0;
    dsk_ms_slice_ = 0.0;
    dsk_ms_pts_ = 0.0;
    dsk_win_size_ = 0;
    dsk_imu_buf_at_deskew_ = 0;
    dsk_skipped_ = true;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double measured_period_s = last_scan_period_;
    const double header_ns = static_cast<double>(scan_stamp.nanoseconds());
    const bool point_timing_valid = point_time != nullptr && point_time->valid;
    const double point_start_ms = point_timing_valid
      ? (point_time->min_stamp_ns - header_ns) * 1e-6 : nan;
    const double point_end_ms = point_timing_valid
      ? (point_time->max_stamp_ns - header_ns) * 1e-6 : nan;
    const double point_span_ms = point_timing_valid
      ? (point_time->max_stamp_ns - point_time->min_stamp_ns) * 1e-6 : nan;
    // Livox CustomMsg -> PointCloud2 conversion preserves absolute per-point
    // timestamps. Real bags consistently show [header, header + ~100 ms], i.e.
    // header is scan start. Keep NDT/UKF time semantics unchanged by de-warping
    // to that start time. When point timing is unavailable, use the measured
    // inter-frame period as the scan duration.
    double scan_start_offset_s = 0.0;
    double scan_end_offset_s = measured_period_s;
    if (point_timing_valid) {
      const double candidate_start_s = (point_time->min_stamp_ns - header_ns) * 1e-9;
      const double candidate_end_s = (point_time->max_stamp_ns - header_ns) * 1e-9;
      const double candidate_period_s = candidate_end_s - candidate_start_s;
      if (std::isfinite(candidate_start_s) && std::isfinite(candidate_end_s) &&
          candidate_period_s > 0.05 && candidate_period_s < 0.5) {
        scan_start_offset_s = candidate_start_s;
        scan_end_offset_s = candidate_end_s;
      }
    }
    const double period_s = scan_end_offset_s - scan_start_offset_s;
    const rclcpp::Time t0 =
      scan_stamp + rclcpp::Duration::from_seconds(scan_start_offset_s);
    const rclcpp::Time t1 =
      scan_stamp + rclcpp::Duration::from_seconds(scan_end_offset_s);
    const double assumed_start_ms = scan_start_offset_s * 1000.0;
    const double assumed_end_ms = scan_end_offset_s * 1000.0;
    const double window_start_error_ms = point_timing_valid
      ? assumed_start_ms - point_start_ms : nan;
    const double window_end_error_ms = point_timing_valid
      ? assumed_end_ms - point_end_ms : nan;
    const char* header_reference = "unknown";
    if (point_timing_valid && std::abs(point_time->first_stamp_ns - header_ns) <= 5e6) {
      header_reference = "scan_start";
    } else if (point_timing_valid && std::abs(point_time->last_stamp_ns - header_ns) <= 5e6) {
      header_reference = "scan_end";
    }

    auto log_deskew_timing = [&, this](
        const char* outcome, const size_t imu_count,
        const double imu_first_ms, const double imu_last_ms,
        const double gyro_path_deg, const bool odom_valid,
        const double odom_age_ms, const double translation_m) {
      if (!sensor_time_diagnostics_enabled_) {
        return;
      }
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), diagnostics_healthy_interval_ms_,
        "[DESKEW TIME] outcome=%s frame=base_link ref=%s stamp_ns=%lld period=%.2fms "
        "assumed=[%+.2f,%+.2f]ms point=[%+.2f,%+.2f]ms span=%.2fms "
        "window_err=[%+.2f,%+.2f]ms imu=[%+.2f,%+.2f]ms n=%zu "
        "gyro_path=%.2fdeg odom=%d age=%+.2fms trans=%.3fm",
        outcome, header_reference,
        static_cast<long long>(scan_stamp.nanoseconds()), period_s * 1000.0,
        assumed_start_ms, assumed_end_ms, point_start_ms, point_end_ms,
        point_span_ms, window_start_error_ms, window_end_error_ms,
        imu_first_ms, imu_last_ms, imu_count, gyro_path_deg,
        odom_valid ? 1 : 0, odom_age_ms, translation_m);

      // A 20 ms mismatch is already large relative to the 100 ms Livox frame;
      // make it an explicit warning so an A/B run cannot be misread as an NDT
      // numerical failure when the upstream time window is wrong.
      if (motion_compensation_enable_ && use_imu && point_timing_valid &&
          (std::abs(window_start_error_ms) > 20.0 ||
           std::abs(window_end_error_ms) > 20.0)) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[DESKEW TIME MISMATCH] assumed window is not the measured scan window: "
          "assumed=[%+.2f,%+.2f]ms point=[%+.2f,%+.2f]ms error=[%+.2f,%+.2f]ms",
          assumed_start_ms, assumed_end_ms, point_start_ms, point_end_ms,
          window_start_error_ms, window_end_error_ms);
      }
    };

    if (!motion_compensation_enable_ || !use_imu) {
      log_deskew_timing("disabled", 0, nan, nan, 0.0, false, nan, 0.0);
      return;
    }
    const size_t n = cloud->size();
    const int slices = motion_compensation_slices_;
    if (slices < 2 || n < static_cast<size_t>(slices) * 2) {
      log_deskew_timing("invalid_shape", 0, nan, nan, 0.0, false, nan, 0.0);
      return;
    }

    // [2026-08-17] 去畸变四段细分计时。实测本函数 p50=39.5ms，而 2 万点做 10 段
    // 刚体变换理论上应 <1ms —— 差 40 倍，原因静态看不出来，只能实测定位：
    //   win_copy  大 → 与 200Hz 的 imu_callback 抢 imu_data_mutex
    //   gyro_int  大 → 陀螺积分循环（window 样本数异常多？）
    //   slice_pre 大 → 10 段变换预计算里对 cum 的线性搜索
    //   pt_loop   大 → 点循环本身（缓存/整数除法/ARM 时钟）
    const auto t_dsk0 = std::chrono::steady_clock::now();

    // Copy the IMU samples inside the measured [scan_start, scan_end] window
    // under the same lock ordering
    // as the predict loop (pose_estimator_mutex -> imu_data_mutex).
    std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> win;
    {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      win.reserve(32);
      for (const auto& imu : imu_data) {
        const rclcpp::Time t(imu->header.stamp);
        if (t >= t0 && t <= t1) {
          win.push_back(imu);
        }
      }
    }
    const auto t_dsk1 = std::chrono::steady_clock::now();
    dsk_win_size_ = win.size();
    dsk_ms_win_ = std::chrono::duration<double, std::milli>(t_dsk1 - t_dsk0).count();
    double imu_first_ms = nan;
    double imu_last_ms = nan;
    if (!win.empty()) {
      imu_first_ms =
        (rclcpp::Time(win.front()->header.stamp).nanoseconds() - header_ns) * 1e-6;
      imu_last_ms =
        (rclcpp::Time(win.back()->header.stamp).nanoseconds() - header_ns) * 1e-6;
    }
    if (win.size() < static_cast<size_t>(slices)) {
      log_deskew_timing(
        "imu_sparse", win.size(), imu_first_ms, imu_last_ms, 0.0,
        false, nan, 0.0);
      return;  // too sparse: skip compensation for this frame
    }
    dsk_skipped_ = false;

    // Forward gyro integration from scan start to scan end. R_cum(t) maps a vector
    // from body frame at time t into body frame at t0: v_t0 = R_cum(t) * v_t.
    struct CumSample { double t; Eigen::Matrix3f R; };
    std::vector<CumSample> cum;
    cum.reserve(win.size() + 1);

    Eigen::Matrix3f R_cum = Eigen::Matrix3f::Identity();
    double gyro_path_rad = 0.0;
    double t_prev = (rclcpp::Time(win.front()->header.stamp) - t0).seconds();
    cum.push_back({t_prev, R_cum});
    for (size_t i = 1; i < win.size(); ++i) {
      const Eigen::Vector3f gyro(
        win[i - 1]->angular_velocity.x,
        win[i - 1]->angular_velocity.y,
        win[i - 1]->angular_velocity.z);
      const double t_i = (rclcpp::Time(win[i]->header.stamp) - t0).seconds();
      const double dt = t_i - t_prev;
      if (dt > 1e-6) {
        const float ang = gyro.norm() * static_cast<float>(dt);
        if (ang > 1e-6f) {
          R_cum = R_cum * Eigen::AngleAxisf(ang, gyro.normalized()).toRotationMatrix();
          gyro_path_rad += std::abs(static_cast<double>(ang));
        }
      }
      t_prev = t_i;
      cum.push_back({t_i, R_cum});
    }
    // Tail step from the last sample to scan end.
    {
      const Eigen::Vector3f gyro(
        win.back()->angular_velocity.x,
        win.back()->angular_velocity.y,
        win.back()->angular_velocity.z);
      const double dt = period_s - t_prev;
      if (dt > 1e-6) {
        const float ang = gyro.norm() * static_cast<float>(dt);
        if (ang > 1e-6f) {
          R_cum = R_cum * Eigen::AngleAxisf(ang, gyro.normalized()).toRotationMatrix();
          gyro_path_rad += std::abs(static_cast<double>(ang));
        }
      }
    }
    const auto t_dsk2 = std::chrono::steady_clock::now();
    dsk_ms_gyro_ = std::chrono::duration<double, std::milli>(t_dsk2 - t_dsk1).count();

    // Translation comes only from SDK odometry. Falling back to the main UKF here
    // recreates the positive feedback this compensation is intended to break.
    Eigen::Vector3f v_body_reference = Eigen::Vector3f::Zero();
    bool odom_compensation_valid = false;
    double odom_age_ms = nan;
    {
      std::lock_guard<std::mutex> lock(robot_odom_mutex_);
      if (has_robot_odom_velocity_) {
        const double age = (scan_stamp - latest_robot_odom_stamp_).seconds();
        odom_age_ms = age * 1000.0;
        // age < 0 表示 odom 比本帧雷达**更新**，这是正常时序而非过期：SDK odom
        // 200Hz、雷达 10Hz，odom 领先 50~100ms 是常态。原判据 `age >= 0.0` 把它
        // 当过期丢弃，实测 61 次 age=-0.06~-0.08s 被误杀 —— 站立时无害，行走时
        // 恰恰跳过了最需要的平移运动补偿。
        // 领先量按同一个上限约束：超前太多同样说明时钟异常。
        if (age >= -robot_odom_velocity_max_age_s_ &&
            age <= robot_odom_velocity_max_age_s_) {
          v_body_reference.x() = latest_robot_odom_velocity_body_.x();
          v_body_reference.y() = latest_robot_odom_velocity_body_.y();
          odom_compensation_valid = true;
        } else {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "SDK odom velocity is stale (age=%.3fs); scan translation compensation skipped",
            age);
        }
      }
    }

    // Precompute per-slice transforms: R_delta maps from body frame at the slice
    // midpoint time t_k back to body frame at scan start (R_cum(t_k)), plus the
    // constant-velocity translation term. R_cum(t_k) comes from the nearest IMU
    // integration sample.
    std::vector<Eigen::Matrix3f> slice_R(static_cast<size_t>(slices));
    std::vector<Eigen::Vector3f> slice_t(static_cast<size_t>(slices));
    double max_slice_translation_m = 0.0;
    for (int k = 0; k < slices; ++k) {
      const double t_k = (static_cast<double>(k) + 0.5) * period_s / slices;
      const Eigen::Matrix3f R_k = [&]() {
        size_t best = 0;
        double best_d = std::abs(cum[0].t - t_k);
        for (size_t j = 1; j < cum.size(); ++j) {
          const double d = std::abs(cum[j].t - t_k);
          if (d < best_d) { best_d = d; best = j; }
        }
        return cum[best].R;
      }();
      slice_R[static_cast<size_t>(k)] = R_k;
      slice_t[static_cast<size_t>(k)] =
        v_body_reference * static_cast<float>(t_k);
      max_slice_translation_m = std::max(
        max_slice_translation_m,
        static_cast<double>(slice_t[static_cast<size_t>(k)].norm()));
    }

    // Assign each point to a slice by capture order (Livox publishes in scan
    // order) and apply the precomputed transform.
    const auto t_dsk3 = std::chrono::steady_clock::now();
    dsk_ms_slice_ = std::chrono::duration<double, std::milli>(t_dsk3 - t_dsk2).count();
    for (size_t i = 0; i < n; ++i) {
      const size_t k = std::min<size_t>(static_cast<size_t>(slices - 1), i * slices / n);
      auto& pt = cloud->points[i];
      const Eigen::Vector3f p_comp =
        slice_R[k] * Eigen::Vector3f(pt.x, pt.y, pt.z) + slice_t[k];
      pt.x = p_comp.x();
      pt.y = p_comp.y();
      pt.z = p_comp.z();
    }
    dsk_ms_pts_ = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_dsk3).count();
    log_deskew_timing(
      "applied", win.size(), imu_first_ms, imu_last_ms,
      gyro_path_rad * 180.0 / M_PI, odom_compensation_valid, odom_age_ms,
      max_slice_translation_m);
  }

  void points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr points_msg) {
    auto start = std::chrono::high_resolution_clock::now();
    PointCloudTimeDiagnostics point_time;
    double point_time_parse_ms = 0.0;

    // Correct de-warp timing needs the PointCloud2 timestamp field even when
    // verbose sensor diagnostics are disabled. Parsing costs ~0.2 ms per frame
    // in the recorded Mid-360 bags and is skipped when neither feature uses it.
    if (sensor_time_diagnostics_enabled_ || motion_compensation_enable_) {
      const auto parse_begin = std::chrono::steady_clock::now();
      point_time = analyzePointCloudTimestamps(*points_msg);
      point_time_parse_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - parse_begin).count();
    }

    if (sensor_time_diagnostics_enabled_) {
      const int64_t callback_ns = get_clock()->now().nanoseconds();
      const int64_t header_ns = rclcpp::Time(points_msg->header.stamp).nanoseconds();
      const bool stamp_ordered =
        updateStreamTiming(lidar_timing_, header_ns, callback_ns);
      logSensorTimeDiagnostics(
        *points_msg, point_time, point_time_parse_ms, callback_ns);
      if (!stamp_ordered) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[TIME] Out-of-order LiDAR frame timestamp detected");
      }
    }

    // [2026-08-17] 管线滞后归因。三个量分别指向不同的责任方：
    //   entry_age = now − header.stamp
    //       传感器出数到本回调开始处理之间的总延迟 = Livox 驱动 + DDS 传输 + 订阅队列排队
    //   stamp_gap = 相邻两帧 header.stamp 之差（应恒为 100ms）
    //       只反映 Livox 出数节奏，与我们无关 → 偏离 100ms 即 Livox 侧不稳
    //   wall_gap  = 相邻两次回调进入的墙钟之差
    //       我们的实际处理节奏 → 大于 stamp_gap 即"跟不上"，差值就是每帧欠的量
    // 判据：
    //   entry_age 大 + 处理快 + 队列空  → Livox/传输侧
    //   entry_age 大 + 处理慢          → 我们处理慢导致排队（滞后会累积）
    //   stamp_gap 抖动                 → Livox 出数不稳
    {
      const rclcpp::Time hdr(points_msg->header.stamp);
      const auto wall_now = std::chrono::steady_clock::now();
      lag_entry_age_ms_ = (get_clock()->now() - hdr).seconds() * 1000.0;
      if (has_lag_prev_) {
        lag_stamp_gap_ms_ = (hdr - lag_prev_stamp_).seconds() * 1000.0;
        lag_wall_gap_ms_ = std::chrono::duration<double, std::milli>(
          wall_now - lag_prev_wall_).count();
        // 每帧欠的量累加即为管线累计滞后（丢帧时会被重置，形成锯齿）
        lag_cum_ms_ += (lag_wall_gap_ms_ - lag_stamp_gap_ms_);
        if (lag_cum_ms_ < 0.0) {
          lag_cum_ms_ = 0.0;
        }
      }
      lag_prev_stamp_ = hdr;
      lag_prev_wall_ = wall_now;
      has_lag_prev_ = true;
    }

    // R11 P1-⑥: skip stale / out-of-order LiDAR frames (do not hard-set queue=1)
    {
      const rclcpp::Time stamp(points_msg->header.stamp);
      const rclcpp::Time now = get_clock()->now();
      if (has_last_lidar_stamp_ && stamp <= last_lidar_stamp_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Skipping out-of-order/stale LiDAR stamp (stamp<=last)");
        return;
      }
      const double age = (now - stamp).seconds();
      if (age > 0.50) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "Skipping stale LiDAR (age=%.3fs > 0.50s)", age);
        return;
      }
      if (has_last_lidar_stamp_) {
        // R14: track the measured scan period for the de-warp window. Sanity
        // bounds (0.05..0.5 s) keep a dropped-frame gap from corrupting it.
        const double period = (stamp - last_lidar_stamp_).seconds();
        if (period > 0.05 && period < 0.5) {
          last_scan_period_ = period;
        }
      }
      last_lidar_stamp_ = stamp;
      has_last_lidar_stamp_ = true;
    }
    // Only a fresh, monotonic frame is evidence that the LiDAR stream is alive.
    updateLidarStatus(true);

    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex); 
    if (use_imu && !static_imu_init_.InitSuccess()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0, "Radar CallBack Waiting for IMU Initial !!!");
      publishExtrapolatedOdom(points_msg->header.stamp);
      return;
    }
    if (!pose_estimator) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0, "Radar CallBack Waiting for Initial Pose Input!!");
      pubDefaultLocalizationOdom(points_msg->header.stamp);
      return;
    }
    if (!global_map_points_ptr_ || global_map_points_ptr_->empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0, "Radar CallBack Waiting for Globalmap Input!!");
      pubDefaultLocalizationOdom(points_msg->header.stamp);
      return;
    }
    if (!isSensorDataValid()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1.0, "Sensor data invalid, using extrapolation!");
      sensor_invalid_latched_ = true;
      publishGlobalLocalizationDiagnostics();
      publishExtrapolatedOdom(points_msg->header.stamp);
      return;
    }
  
    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    raw_points_ptr_->clear();
    pcl::fromROSMsg(*points_msg, *raw_points_ptr_);

    if (raw_points_ptr_->empty()) {
      RCLCPP_ERROR(get_logger(), "cloud is empty!!");
      return;
    }

    // Safety net: remove any NaN/Inf that made it through (shouldn't happen now).
    // NOTE: removeNaNFromPointCloud's 3rd arg is the kept-point index mapping
    // (cloud_out[i] = cloud_in[index[i]]), NOT the removed points.  Removed count
    // must be computed as before - after.  Also force is_dense=false, otherwise
    // PCL short-circuits and never actually checks for NaN.
    {
      const size_t before = raw_points_ptr_->size();
      raw_points_ptr_->is_dense = false;
      std::vector<int> keep_indices;
      pcl::removeNaNFromPointCloud(*raw_points_ptr_, *raw_points_ptr_, keep_indices);
      const size_t removed = before - raw_points_ptr_->size();
      if (removed > 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0,
          "Removed %zu NaN/Inf points from input cloud (total=%zu)", removed, before);
      }
    }

    // Apply the fixed LiDAR->base extrinsic before de-warping. The buffered IMU
    // gyro and SDK odometry twist are both expressed in base_link, so applying
    // them to livox_frame points would be a frame error whenever init_T is not
    // identity. De-warp still runs before downsampling because VoxelGrid reorders
    // output by voxel index,
    // which breaks the "point order ≈ capture order" assumption the slice
    // assignment relies on — so compensation must run on the raw cloud.
    //
    // [2026-08-17] 分段计时：回调 p50=112ms 超出 100ms 周期，导致滞后持续增长
    // → 丢帧 → NDT 初值突跳 → 拒帧 → GL。必须先知道这 112ms 花在哪，再谈优化。
    const size_t pts_raw = raw_points_ptr_->size();
    const auto t_extrinsic_begin = std::chrono::steady_clock::now();
    TransformPoints(raw_points_ptr_, pcl_cloud);
    const auto t_deskew_begin = std::chrono::steady_clock::now();
    motionCompensateScan(
      pcl_cloud, rclcpp::Time(points_msg->header.stamp), &point_time);

    const auto t_voxel_begin = std::chrono::steady_clock::now();
    raw_points_ptr_ = downsample(pcl_cloud);
    const auto t_voxel_end = std::chrono::steady_clock::now();
    stage_ms_extrinsic_ = std::chrono::duration<double, std::milli>(
      t_deskew_begin - t_extrinsic_begin).count();
    stage_ms_deskew_ = std::chrono::duration<double, std::milli>(
      t_voxel_begin - t_deskew_begin).count();
    stage_ms_voxel_ = std::chrono::duration<double, std::milli>(
      t_voxel_end - t_voxel_begin).count();
    stage_pts_raw_ = pts_raw;
    stage_pts_filtered_ = raw_points_ptr_ ? raw_points_ptr_->size() : 0;
    // last_scan = filtered;
    
    const rclcpp::Time scan_stamp(points_msg->header.stamp);
    // ------------------------------------------------------------------
    // Global localization: descriptor retrieval + static (motion-free) confirm
    //
    // [2026-08-14] The motion-driven temporal vote was removed. Retrieval
    // recall is cheap enough to repeat in place, so a pose is confirmed when
    // gl_confirm_frames consecutive INDEPENDENT probes agree, instead of the
    // robot having to walk/rotate between votes. Geometric verification and
    // the "is the runner-up somewhere else nearly as good?" ambiguity check
    // still run inside performGlobalLocalization (selectBestCluster).
    // ------------------------------------------------------------------
    if (globalLocalizationRuntimeEnabled() && !is_init_success_) {
      if (!gl_episode_state_.active && gl_once_gate_) {
        beginGLEpisode(gl_episode_state_, ++gl_episode_id_counter_,
                       get_clock()->now(), gl_params_);
        resetStaticConfirm(gl_static_confirm_);
        gl_summary_ = GLSummaryCounts{};
        gl_request_initialpose_ = false;
        publishGlobalLocalizationDiagnostics(true);
        RCLCPP_INFO(get_logger(),
                    "[GL] episode %lu started (retrieval recall + %d-frame static confirm)",
                    static_cast<unsigned long>(gl_episode_state_.episode_id),
                    gl_params_.gl_confirm_frames);
      }

      if (gl_episode_state_.active) {
        if (checkGLEpisodeTimeout(gl_episode_state_, get_clock()->now(), gl_params_)) {
          is_init_success_ = false;
          localization_state_ = 1;
          gl_request_initialpose_ = true;
          GlobalLocalizationResult timeout_result;
          timeout_result.status = GLStatus::NoCandidate;
          timeout_result.budget_exceeded = true;
          dumpGlobalLocalizationDecision(scan_stamp, timeout_result);
          publishGlobalLocalizationDiagnostics(true);
          RCLCPP_WARN(get_logger(),
                      "[GL] episode %lu timed out; localization stays invalid and RViz initialpose is required",
                      static_cast<unsigned long>(gl_episode_state_.episode_id));
          return;
        }

        const StaticConfirmParams confirm_params = staticConfirmParams();
        if (shouldRunStaticProbe(gl_static_confirm_, scan_stamp, confirm_params)) {
          GlobalLocalizationResult probe_result;
          if (runGlobalLocalizationShadowProbe(
                  raw_points_ptr_, scan_stamp, nullptr, &probe_result)) {
            // [2026-08-14 实测] 狗站立时身体自然摇摆会周期性打断重力滤波
            // 窗口。重力不可用是环境问题、不是位姿证据 —— 跳过本次探针
            // 且不清零确认链，否则连续一致计数永远凑不满（实测成功/失败
            // 以约 3 秒周期交替，agree 每次被清零）。
            if (probe_result.anchor_reject_reason !=
                GLAnchorRejectReason::GravityUnavailable) {
              const StaticConfirmOutcome outcome = ingestStaticProbeResult(
                  gl_static_confirm_, probe_result, scan_stamp, confirm_params,
                  gl_summary_);
            RCLCPP_INFO(
                get_logger(),
                "[GL confirm] probe=%d status=%s agree=%d/%d best_score=%.4f support=%d",
                gl_static_confirm_.probe_count, glStatusName(probe_result.status),
                gl_static_confirm_.agree_count, confirm_params.confirm_frames,
                probe_result.best_score, gl_summary_.best_support_group_count);

            if (outcome == StaticConfirmOutcome::Confirmed) {
              GlobalLocalizationResult confirmed = probe_result;
              confirmed.status = GLStatus::Confirmed;
              confirmed.success = true;
              const GLIntegrationAction action = decideGLIntegrationAction(
                  confirmed, gl_params_, gl_episode_state_.active, false);
              dumpGlobalLocalizationDecision(scan_stamp, confirmed);
              if (action.accept_pose_for_init && action.run_init_verification) {
                if (!applyConfirmedGlobalPose(confirmed, scan_stamp)) {
                  RCLCPP_ERROR(
                      get_logger(),
                      "[GL confirm] confirmed pose failed node-level application checks; episode stays active");
                  resetStaticConfirm(gl_static_confirm_);
                }
              } else if (action.decision_shadow) {
                RCLCPP_INFO(
                    get_logger(),
                    "[GL confirm] hypothetically confirmed (decision-shadow); "
                    "auto_confirm_requested=%d auto_confirm_effective=%d; no pose applied",
                    gl_params_.auto_confirm_enabled ? 1 : 0,
                    automaticApprovalContractSatisfied(gl_params_) ? 1 : 0);
                resetStaticConfirm(gl_static_confirm_);
              } else {
                RCLCPP_WARN(
                    get_logger(),
                    "[GL confirm] confirmed result rejected by integration contract (invalid=%d stale=%d)",
                    action.invalid_confirmed_result ? 1 : 0,
                    action.stale_confirmed_result ? 1 : 0);
                resetStaticConfirm(gl_static_confirm_);
              }
            } else if (outcome == StaticConfirmOutcome::Timeout) {
              endGLEpisode(gl_episode_state_, GLStatus::NoCandidate);
              gl_request_initialpose_ = true;
              publishGlobalLocalizationDiagnostics(true);
              RCLCPP_WARN(
                  get_logger(),
                  "[GL confirm] no agreeing pose within %.1fs (%d probes); RViz initialpose required. "
                  "Move the robot a couple of metres and restart if the scene is ambiguous.",
                  confirm_params.attempt_timeout_s, gl_static_confirm_.probe_count);
            }
            }
          }
        }

        // The episode owns the scan while active: the main NDT/UKF path and
        // every map pose/TF output stay frozen until verification passes.
        if (gl_episode_state_.active) {
          return;
        }
      }

      if (gl_episode_state_.phase == GLEpisodePhase::AwaitingPrior) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 30000,
            "[GL] awaiting external prior (RViz initialpose); legacy NDT auto-init suppressed");
        return;
      }
    }
    
    if (!is_init_success_) {
      if (gl_auto_confirm_pending_verification_) {
        const rclcpp::Time verification_now = get_clock()->now();
        const bool invalid_verification_clock =
          gl_auto_confirm_verification_start_time_.nanoseconds() <= 0 ||
          verification_now.get_clock_type() !=
            gl_auto_confirm_verification_start_time_.get_clock_type() ||
          verification_now < gl_auto_confirm_verification_start_time_;
        const bool verification_timed_out = invalid_verification_clock ||
          (verification_now - gl_auto_confirm_verification_start_time_).seconds() >
            gl_params_.auto_confirm_verify_timeout_s;
        if (verification_timed_out) {
          pose_estimator->reset_velocity();
          {
            std::lock_guard<std::mutex> lock(imu_data_mutex);
            imu_data.clear();
          }
          gl_auto_confirm_pending_verification_ = false;
          gl_request_initialpose_ = true;
          gl_once_gate_ = false;
          endGLEpisode(gl_episode_state_, GLStatus::NoCandidate);
          localization_state_ = 1;
          RCLCPP_ERROR(
            get_logger(),
            "[GL APPLY] automatic pose did not pass NDT verification within %.2fs; localization remains invalid and RViz initialpose is required",
            gl_params_.auto_confirm_verify_timeout_s);
          publishGlobalLocalizationDiagnostics(true);
          return;
        }
      }
      localization_state_ = 1;
      PoseEstimator::MatchResult init_result = pose_estimator->GetMatchState();
      RCLCPP_INFO(get_logger(), "init_result.is_converged_ : %d", init_result.is_converged_);
      RCLCPP_INFO(get_logger(), "init_result.fitness_score_ : %f", init_result.fitness_score_);
      // Round 10: combined init criteria — relaxed score upper bound + pose consistency.
      // Pure score < 0.5 is too strict for this environment (NDT score at correct
      // position is naturally 0.6-0.9 due to map geometry). A loose score cap rejects
      // only catastrophic frames (score > init_match_score_threshold_, e.g. 9840 at 87m).
      // Pose jump detection ensures we only accept STABLE positions — if the NDT result
      // jumps more than init_match_pose_jump_threshold_ between frames, we're not stable.
      // On a stationary robot this catches wrong local minima that wander frame-to-frame.
      Eigen::Vector3f cur_pos = pose_estimator->pos();
      // Always update the reference pose so the next frame can compute a real jump.
      // Note: has_init_verify_pose_ is reset when the PoseEstimator is recreated
      // (global localization success / RViz initialpose), so the first frame after
      // recreation always uses the fallback and fails the pose check — that's intentional:
      // we need at least one frame to establish the reference.
      float pose_jump = has_init_verify_pose_ ? (cur_pos - last_verify_pos_).norm()
                                                : init_match_pose_jump_threshold_ * 2.0f;
      last_verify_pos_ = cur_pos;
      has_init_verify_pose_ = true;
      bool score_ok = init_result.is_converged_
                      && init_result.fitness_score_ < init_match_score_threshold_;
      bool pose_ok = pose_jump < init_match_pose_jump_threshold_;
      if (score_ok && pose_ok) {
        init_match_count_++;
        RCLCPP_INFO(get_logger(), "Init match count: %d/%d (score: %.4f, jump: %.3fm, Z: %.3f)",
                    init_match_count_, init_match_count_threshold_,
                    init_result.fitness_score_, pose_jump, cur_pos.z());
        if (init_match_count_ >= init_match_count_threshold_) {
          is_init_success_ = true;
          localization_state_ = 2;
          tracking_health_.resetHealthy();
          consecutive_ndt_rejections_ = 0;
          RCLCPP_INFO(get_logger(), "Init Pose Successful!!!");
          if (gl_auto_confirm_pending_verification_) {
            RCLCPP_WARN(
              get_logger(),
              "[GL APPLY] automatic pose passed consecutive NDT verification; localization output is now enabled (apply_count=%lu)",
              static_cast<unsigned long>(gl_auto_confirm_apply_count_));
            gl_auto_confirm_pending_verification_ = false;
            gl_auto_confirm_verification_start_time_ =
              rclcpp::Time(int64_t{0}, get_clock()->get_clock_type());
          }
          init_match_count_ = 0;
          // Start innovation gate warmup so UKF can converge from GL initial guess
          // before hard physics gates engage.
          if (pose_estimator) {
            const int previous_remaining =
              pose_estimator->tracking_gate_warmup_remaining();
            const int previous_consumed =
              pose_estimator->tracking_gate_warmup_consumed();
            pose_estimator->start_tracking_gate_warmup();
            RCLCPP_INFO(
              get_logger(),
              "[WARMUP] restarted after Init success: previous_remaining=%d "
              "consumed_total=%d new_remaining=%d restarts=%d",
              previous_remaining, previous_consumed,
              pose_estimator->tracking_gate_warmup_remaining(),
              pose_estimator->tracking_gate_warmup_restarts());
          }
          publishGlobalLocalizationDiagnostics();
        }
      } else {
        init_match_count_ = 0;
        RCLCPP_WARN(get_logger(),
          "Init match criteria not met (score=%.4f %s, jump=%.3fm %s), resetting counter",
          init_result.fitness_score_, score_ok ? "OK" : "FAIL",
          pose_jump, pose_ok ? "OK" : "FAIL");
      }
      RCLCPP_INFO(get_logger(), "Wait Init Pose!!! Current count: %d/%d, Z: %.3f",
                  init_match_count_, init_match_count_threshold_, cur_pos.z());
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    // Track whether this frame really executed an IMU-backed UKF prediction.
    // A non-empty buffer is not sufficient: the filter stride and timestamp guards
    // can still result in zero predict calls.
    bool imu_prediction_used = false;
    bool imu_fallback_active = false;
    double imu_fallback_dt_s = 0.0;
    const int64_t latest_imu_ns = latest_imu_stamp_ns_.load(std::memory_order_relaxed);
    const int64_t lidar_stamp_ns = rclcpp::Time(stamp).nanoseconds();
    const double imu_age_ms = latest_imu_ns > 0
      ? (static_cast<double>(lidar_stamp_ns - latest_imu_ns) * 1e-6)
      : std::numeric_limits<double>::quiet_NaN();
    const long predict_count_before = pose_estimator->predict_count();

    // predict
    if (!use_imu) {
      pose_estimator->predict(stamp);
    } else {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      // Round 10: during init verification, skip IMU prediction entirely.
      // The UKF is frozen at the global-localized position. Without this,
      // even a single successful NDT correction can impart velocity via the
      // Kalman gain, and subsequent rejected frames (score > threshold) cannot
      // correct the accumulated drift — causing the UKF to diverge permanently.
      if (!is_init_success_) {
        imu_data.clear();
      } else {
        // R11 P0-④: drop IMU older than 0.15s; synthesize one step only on real backlog
        // Normal LiDAR period (~100ms @ 200Hz IMU) yields ~20 samples — that is NOT backlog.
        constexpr double kMaxImuAge = 0.15;
        constexpr size_t kMaxImuCatchupSteps = 40;  // ~0.2s @ 200Hz; was 10 (false-triggered every frame)
        const rclcpp::Time lidar_stamp(stamp);
        const rclcpp::Time cutoff = lidar_stamp - rclcpp::Duration::from_seconds(kMaxImuAge);

        imu_data.erase(
          std::remove_if(imu_data.begin(), imu_data.end(),
            [&](const sensor_msgs::msg::Imu::ConstSharedPtr& imu) {
              return rclcpp::Time(imu->header.stamp) < cutoff;
            }),
          imu_data.end());

        size_t usable = 0;
        for (; usable < imu_data.size(); ++usable) {
          if (lidar_stamp < rclcpp::Time(imu_data[usable]->header.stamp)) {
            break;
          }
        }

        // [2026-08-17] IMU 供给诊断：区分"IMU 侧时间戳问题"与"我们处理慢的后果"。
        //   imu_lead > 0  ：IMU 比雷达新（正常，200Hz vs 10Hz）
        //   imu_lead < 0  ：IMU 落后于雷达 → Livox IMU 时间戳/传输侧问题
        //   usable 很小   ：多半是本帧处理慢，早于 lidar-0.15s 的样本被 cutoff 砍掉
        //   usable < 5    ：predict 一次都不会被调用（num % imu_data_filter_num_ 到不了）
        imu_buf_size_ = imu_data.size();
        imu_usable_ = usable;
        imu_lead_ms_ = imu_data.empty() ? 0.0
          : (rclcpp::Time(imu_data.back()->header.stamp) - lidar_stamp).seconds() * 1000.0;

        double imu_span = 0.0;
        if (usable > 0) {
          imu_span = (rclcpp::Time(imu_data[usable - 1]->header.stamp) -
                      rclcpp::Time(imu_data[0]->header.stamp)).seconds();
        }

        auto predict_one = [&](const sensor_msgs::msg::Imu::ConstSharedPtr& imu, bool use_explicit_dt, double dt) {
          const auto& acc = imu->linear_acceleration;
          const auto& gyro = imu->angular_velocity;
          const Eigen::Vector3f acc_v(acc.x, acc.y, acc.z);
          const Eigen::Vector3f gyro_v(gyro.x, gyro.y, gyro.z);
          if (use_explicit_dt) {
            pose_estimator->predict_with_dt(imu->header.stamp, acc_v, gyro_v, dt);
          } else {
            pose_estimator->predict(imu->header.stamp, acc_v, gyro_v);
          }
          // R11 P0-①: do NOT publish_odometry here (avoids TF/INFO storm during catch-up)
        };

        if (usable > kMaxImuCatchupSteps || imu_span > kMaxImuAge) {
          predict_one(imu_data[usable - 1], true, kMaxImuAge);
          RCLCPP_WARN(get_logger(),
            "[IMU Q] catch-up synthesize: usable=%zu span=%.3fs dt_cap=%.2fs "
            "ukf_pos=[%.3f,%.3f,%.3f] ukf_vel=[%.3f,%.3f,%.3f]",
            usable, imu_span, kMaxImuAge,
            pose_estimator->pos().x(), pose_estimator->pos().y(), pose_estimator->pos().z(),
            pose_estimator->vel().x(), pose_estimator->vel().y(), pose_estimator->vel().z());
          imu_data.erase(imu_data.begin(), imu_data.begin() + static_cast<std::ptrdiff_t>(usable));
        } else {
          int num = 0;
          auto imu_iter = imu_data.begin();
          for (; imu_iter != imu_data.end(); ++imu_iter) {
            if (lidar_stamp < rclcpp::Time((*imu_iter)->header.stamp)) {
              break;
            }
            num++;
            if (!(num % imu_data_filter_num_)) {
              predict_one(*imu_iter, false, 0.0);
            }
          }
          imu_data.erase(imu_data.begin(), imu_iter);
        }
      }
      imu_prediction_used = pose_estimator->predict_count() > predict_count_before;
    }

    // The IMU publisher can be delayed by more than the LiDAR period (the recorded
    // bag reaches 1.2--1.5 s). Do not let NDT correct repeatedly without any
    // intervening prediction: that makes Ppos monotonically collapse and drives the
    // Kalman gain to zero. PoseSystem::f(state) is a constant-velocity, no-input
    // propagation, so it adds uncertainty without fabricating acceleration.
    if (use_imu && is_init_success_ && imu_stale_fallback_enabled_ &&
        !imu_prediction_used) {
      const double max_fallback_dt = std::clamp(
        imu_stale_fallback_max_dt_s_, 0.02, 0.15);
      imu_fallback_dt_s = std::clamp(last_scan_period_, 0.02, max_fallback_dt);
      pose_estimator->predict_without_imu(rclcpp::Time(stamp), imu_fallback_dt_s);
      imu_fallback_active = true;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[IMU FALLBACK] no IMU predict for LiDAR frame: age=%+.1fms buf=%zu usable=%zu "
        "dt=%.3fs predict_count=%ld->%ld; constant-velocity covariance propagation enabled",
        imu_age_ms, imu_buf_size_, imu_usable_, imu_fallback_dt_s,
        predict_count_before, pose_estimator->predict_count());
    }

    pose_estimator->set_prediction_diagnostic(
      imu_prediction_used, imu_age_ms, imu_fallback_active, imu_fallback_dt_s);

    // odometry-based prediction
    rclcpp::Time last_correction_time = pose_estimator->last_correction_time();
    if (enable_robot_odometry_prediction && last_correction_time != rclcpp::Time((int64_t)0, get_clock()->get_clock_type())) {
      geometry_msgs::msg::TransformStamped odom_delta;
      if (tf_buffer->canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, rclcpp::Duration(std::chrono::milliseconds(0)))) {
        odom_delta =
          tf_buffer->lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, rclcpp::Duration(std::chrono::milliseconds(0)));
      }
      if (odom_delta.header.stamp == rclcpp::Time((int64_t)0, get_clock()->get_clock_type())) {
        pose_estimator->clear_odom_prediction();
        RCLCPP_WARN_STREAM(get_logger(), "failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
      } else {
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        pose_estimator->predict_odom(delta.cast<float>().matrix());
      }
    }
    if (pose_estimator->has_trust_anchor()) {
      Eigen::Matrix4f current_odom_pose;
      if (lookupRobotOdomPose(stamp, current_odom_pose)) {
        pose_estimator->update_trust_odom(current_odom_pose);
      } else {
        pose_estimator->suspend_trust_odom();
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[TRUST] current %s->%s transform unavailable; trust tightening suspended for this frame",
          robot_odom_frame_id.c_str(), odom_child_frame_id.c_str());
      }
    }
    // correct
    // DIAG (2026-07-23): capture the pure IMU-predicted pose right before NDT correction,
    // so we can log how much NDT disagrees with dead-reckoning on every single frame.
    // This gives a quantitative, greppable substitute for "does the LiDAR point cloud
    // visually stay glued to the map while walking" when RViz is too laggy to judge by eye:
    //   - small & stable delta_pos/delta_yaw while walking -> NDT tracking is healthy,
    //     a wrong walking direction is more likely an odom->base_link heading/SDK issue.
    //   - delta_pos/delta_yaw growing or spiking during motion -> NDT itself is losing
    //     track / fighting the prediction, i.e. a motion-time registration problem.
    Eigen::Matrix4f pred_pose_diag = pose_estimator->matrix();
    // [2026-08-17] 只给 NDT 截天花板；raw_points_ptr_ 保持原样供 GL 使用。
    // 两侧（地图与扫描）必须用同一个阈值，否则点云分布不一致会引入新的配准偏差。
    const auto ndt_scan = filterByHeight(raw_points_ptr_, ndt_max_point_height_);
    stage_pts_ndt_ = ndt_scan ? ndt_scan->size() : 0;
    // [2026-08-31 走廊漂移修复] 把对齐到本帧的 SDK 机体速度喂给
    // pose_estimator,correct() 内做 UKF 速度一致性门(静止归零/超速钳制)。
    {
      std::lock_guard<std::mutex> lock(robot_odom_mutex_);
      if (has_robot_odom_velocity_) {
        const double age =
          (rclcpp::Time(stamp) - latest_robot_odom_stamp_).seconds();
        if (age >= -robot_odom_velocity_max_age_s_ &&
            age <= robot_odom_velocity_max_age_s_) {
          pose_estimator->set_robot_odom_velocity(
            latest_robot_odom_velocity_body_, age);
        } else {
          pose_estimator->clear_robot_odom_velocity();
        }
      } else {
        pose_estimator->clear_robot_odom_velocity();
      }
    }
    const auto t_ndt_begin = std::chrono::steady_clock::now();
    auto aligned = pose_estimator->correct(stamp, ndt_scan);
    stage_ms_ndt_ = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_ndt_begin).count();

    Eigen::Matrix4f corr_pose_diag = pose_estimator->matrix();
    Eigen::Vector3f pred_p = pred_pose_diag.block<3, 1>(0, 3);
    Eigen::Vector3f corr_p = corr_pose_diag.block<3, 1>(0, 3);
    float delta_pos = (corr_p - pred_p).norm();
    Eigen::Quaternionf pred_q(pred_pose_diag.block<3, 3>(0, 0));
    Eigen::Quaternionf corr_q(corr_pose_diag.block<3, 3>(0, 0));
    float pred_yaw = quaternionToNormalizedRPY(pred_q).z();
    float corr_yaw = quaternionToNormalizedRPY(corr_q).z();
    float delta_yaw = corr_yaw - pred_yaw;
    if (delta_yaw > M_PI) delta_yaw -= 2.0f * M_PI;
    if (delta_yaw < -M_PI) delta_yaw += 2.0f * M_PI;

    PoseEstimator::MatchResult match_result = pose_estimator->GetMatchState();
    const Eigen::Vector3f ukf_vel = pose_estimator->vel();
    const float vel_norm = ukf_vel.norm();
    const bool ndt_accepted =
      match_result.is_converged_ && match_result.fitness_score_ <= ndt_score_threshold_;

    int ndt_iters = -1;
    double ndt_trans_prob = -1.0;
    double ndt_nvtl = -1.0;
    double ndt_res = -1.0;
    int ndt_threads = -1;
    bool ndt_solver_converged = false;
    bool ndt_optimizer_available = false;
    NdtOptimizerDiagnostics ndt_optimizer;
    pclomp::NdtIterationDiagnostics ndt_iteration_diagnostics;
    if (auto ndt_locked = ndt_diag_.lock()) {
      ndt_iters = ndt_locked->getFinalNumIteration();
      ndt_trans_prob = ndt_locked->getTransformationProbability();
      ndt_nvtl = ndt_locked->getNearestVoxelTransformationLikelihood();
      ndt_res = ndt_locked->getResolution();
      ndt_threads = ndt_locked->getNumThreads();
      ndt_solver_converged = ndt_locked->hasConverged();
      if (ndt_detailed_diagnostics_enabled_) {
        ndt_iteration_diagnostics = ndt_locked->getIterationDiagnostics();
      }

      if (ndt_optimizer_diagnostics_enabled_) {
        auto transformations = ndt_locked->getFinalTransformationArray();
        const Eigen::Matrix4f final_transformation =
          ndt_locked->getFinalTransformation();
        if (!transformations.empty() &&
            !transformations.back().isApprox(final_transformation, 1e-5f)) {
          transformations.push_back(final_transformation);
        }

        // ndt_omp does not refresh its exported hessian on the zero-step early
        // return. Do not report a potentially stale matrix when no iteration
        // transform was recorded for this frame.
        Eigen::Matrix<double, 6, 6> hessian =
          Eigen::Matrix<double, 6, 6>::Constant(
            std::numeric_limits<double>::quiet_NaN());
        if (transformations.size() >= 2) {
          hessian = ndt_locked->getHessian();
        }
        ndt_optimizer = analyzeNdtOptimizer(hessian, transformations);
        ndt_optimizer_available = true;
      }
    }

    // [2026-08-16] 位姿轨迹诊断（1Hz 限频）。
    //
    // 加这条的原因：连续两轮板端测试都无法回答"z 到底是稳定的常量偏移，还是还在
    // 持续漂移"——日志里只有事件（拒帧/落位/告警），没有任何状态量的时间序列，
    // 排查等于半盲。z/roll/pitch 恰恰是本项目最脆弱的三个自由度（室内平面环境
    // NDT 对它们没有几何约束），必须能看到它们随时间怎么走。
    //
    // 字段说明：
    //   z/roll/pitch  当前位姿（roll/pitch 已由重力锚定，正常应贴近 map 重力基准）
    //   dz_ref        相对落位后捕获的地面参考的 z 漂移，是判断"稳定 vs 漂移"的主指标
    //   |v|           UKF 线速度模长，静止时应 <0.05；变大即幽灵速度复现
    //   score         NDT fitness，注意：分数好**不代表** z/pitch 正确
    //   acc           本帧 NDT 是否被接受
    {
      const Eigen::Vector3f rpy_now = quaternionToNormalizedRPY(corr_q);
      float ref_z = 0.0f;
      const bool has_ref = pose_estimator->ground_reference_z(ref_z);
      char dz_buf[32];
      if (has_ref) {
        snprintf(dz_buf, sizeof(dz_buf), "%+.3f", corr_p.z() - ref_z);
      } else {
        snprintf(dz_buf, sizeof(dz_buf), "n/a");
      }
      const auto& td = pose_estimator->tracking_diag();
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[POSE DIAG] xyz=[%.3f,%.3f,%.3f] rpy=[%.2f,%.2f,%.2f]deg dz_ref=%s "
        "|v|=%.3f score=%.4f acc=%d warmup=%d | "
        "grav=%d gerr=%.1f grate=%.2f gmag=%.2f | "
        "innov dxy=%.3f dz=%+.3f dyaw=%+.2f | "
        "occ ready=%d points=%d would_reject=%d",
        corr_p.x(), corr_p.y(), corr_p.z(),
        rpy_now.x() * 180.0f / static_cast<float>(M_PI),
        rpy_now.y() * 180.0f / static_cast<float>(M_PI),
        rpy_now.z() * 180.0f / static_cast<float>(M_PI),
        dz_buf, vel_norm, match_result.fitness_score_,
        ndt_accepted ? 1 : 0,
        pose_estimator->tracking_gate_warmup_remaining(),
        td.gravity_applied ? 1 : 0, td.gravity_err_deg,
        td.gravity_accept, td.gravity_mag_err,
        td.innov_dxy, td.innov_dz, td.innov_dyaw_deg,
        td.occupancy_ready ? 1 : 0, td.occupancy_points,
        td.occupancy_blocked ? 1 : 0);

      if (td.attitude_chain_valid) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), diagnostics_healthy_interval_ms_,
          "[ATTITUDE CHAIN] gravity_applied=%d "
          "pred=[%+.2f,%+.2f,%+.2f]deg "
          "raw_ndt=[%+.2f,%+.2f,%+.2f]deg "
          "gravity_obs=[%+.2f,%+.2f,%+.2f]deg "
          "fused=[%+.2f,%+.2f,%+.2f]deg raw_gravity_err=%.2fdeg",
          td.gravity_applied ? 1 : 0,
          td.pred_roll_deg, td.pred_pitch_deg, td.pred_yaw_deg,
          td.ndt_roll_deg, td.ndt_pitch_deg, td.ndt_yaw_deg,
          td.gravity_obs_roll_deg, td.gravity_obs_pitch_deg,
          td.gravity_obs_yaw_deg,
          rpy_now.x() * 180.0f / static_cast<float>(M_PI),
          rpy_now.y() * 180.0f / static_cast<float>(M_PI),
          rpy_now.z() * 180.0f / static_cast<float>(M_PI),
          td.gravity_err_deg);
      }

      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[TRUST DIAG] valid=%d deviation_xy=%.3f allowed_xy=%.3f "
        "xy_gate=%.3f travel=%.3f consecutive=%d | "
        "warmup_remaining=%d consumed_total=%d restarts=%d",
        td.trust_valid ? 1 : 0, td.trust_deviation_xy,
        td.trust_allowed_xy, td.trust_gate_xy, td.trust_travel_m,
        td.trust_exceed_count,
        pose_estimator->tracking_gate_warmup_remaining(),
        pose_estimator->tracking_gate_warmup_consumed(),
        pose_estimator->tracking_gate_warmup_restarts());

      // 预测来源分解：三个 z 并排，直接看出是谁在往下走。
      // odom_dz 持续为负 → 腿式里程计的 z 在漂；imu_z 与 used_z 一致且平稳 →
      // 预测侧健康，问题在 NDT。
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[PRED DIAG] imu_z=%+.3f odom_z=%+.3f used_z=%+.3f odom_dz=%+.3f valid=%d | "
        "imu_yaw=%+.2f odom_yaw=%+.2f",
        td.pred_imu_z, td.pred_odom_z, td.pred_used_z, td.pred_odom_dz,
        td.odom_pred_valid ? 1 : 0,
        td.pred_imu_yaw_deg, td.pred_odom_yaw_deg);

      // UKF 健康度。
      // [2026-08-16] 限频从 5s 降到 1s：上一轮 5s 采样在"姿态开始甩但 NDT 还在收"
      // 的窗口留下盲区，无法证明模长塌缩发生在姿态发散**之前**还是之后。
      // predicts 是最关键的新字段——实测出现过整个滤波器实例生命周期内 predict
      // 一次都没跑（点云回调阻塞 3~4s 所致），而 correct 仍在跑，四元数模长随即
      // 无阻尼塌缩到 0.14。它若长时间不增长，就是管线停摆的直接证据。
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[UKF DIAG] quat_norm=%.4f quat_cov_tr=%.5f sigma_ang=%.1fdeg "
        "predicts=%ld llt_fail=%ld",
        td.ukf_quat_norm, td.ukf_quat_cov_tr, td.ukf_sigma_quat_dev,
        td.ukf_predicts, td.ukf_llt_failures);

      // 分段耗时：直接定位 112ms 的构成。other = 总耗时 − 已计时的三段，
      // 包含 GL、发布、TF、诊断本身等。
      const double stage_total =
        std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - start).count();
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[STAGE] extrinsic=%.1f deskew=%.1f voxel=%.1f ndt=%.1f other=%.1f total=%.1fms | "
        "pts %zu->%zu->ndt %zu",
        stage_ms_extrinsic_, stage_ms_deskew_, stage_ms_voxel_, stage_ms_ndt_,
        stage_total - stage_ms_extrinsic_ - stage_ms_deskew_ -
          stage_ms_voxel_ - stage_ms_ndt_,
        stage_total, stage_pts_raw_, stage_pts_filtered_, stage_pts_ndt_);

      // NDT 明细。目的：解释 ndt 段为何要 76ms / 迭代为何会到 17（上限设的是 15）。
      //   set_src / align 分开：若 set_src 不可忽略，说明每帧在重建源点云结构。
      //   threads：确认 OpenMP 是否真的按 6 线程跑（构造时取 omp_get_max_threads）。
      //   trans_prob / nvtl：NDT 自身的配准质量度量，与 fitness_score 是不同的量；
      //     迭代打满但 trans_prob 仍高 → 是收敛判据太严而非匹配不上。
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[NDT] iters=%d/%d set_src=%.2f align=%.1f score=%.1fms | res=%.2f threads=%d "
        "trans_prob=%.4f nvtl=%.4f solver_converged=%d accepted=%d src_pts=%zu",
        ndt_iters, ndt_max_iterations_,
        td.ndt_setsrc_ms, td.ndt_align_ms, td.ndt_score_ms,
        ndt_res, ndt_threads, ndt_trans_prob, ndt_nvtl,
        ndt_solver_converged ? 1 : 0, ndt_accepted ? 1 : 0,
        stage_pts_filtered_);

      if (ndt_optimizer_available) {
        if (ndt_accepted) {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), diagnostics_healthy_interval_ms_,
            "[NDT OPT] sensor_ns=%ld solver_converged=%d accepted=1 iters=%d/%d transforms=%zu "
            "init=[%.3f,%.3f,%.3f] final=[%.3f,%.3f,%.3f] "
            "corr_local=[%+.3f,%+.3f,%+.3f] corr_rpy=[%+.2f,%+.2f,%+.2f]deg",
            static_cast<long>(rclcpp::Time(stamp).nanoseconds()),
            ndt_solver_converged ? 1 : 0, ndt_iters, ndt_max_iterations_,
            ndt_optimizer.transformation_count,
            ndt_optimizer.initial_xyz[0], ndt_optimizer.initial_xyz[1],
            ndt_optimizer.initial_xyz[2], ndt_optimizer.final_xyz[0],
            ndt_optimizer.final_xyz[1], ndt_optimizer.final_xyz[2],
            ndt_optimizer.correction_xyz_local[0],
            ndt_optimizer.correction_xyz_local[1],
            ndt_optimizer.correction_xyz_local[2],
            ndt_optimizer.correction_rpy_deg[0],
            ndt_optimizer.correction_rpy_deg[1],
            ndt_optimizer.correction_rpy_deg[2]);
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), diagnostics_healthy_interval_ms_,
            "[NDT CURV] valid=%d eig=[%+.2e,%+.2e,%+.2e,%+.2e,%+.2e,%+.2e] "
            "inertia=[+%d,-%d,0:%d] abs_cond=%.3e weak_abs=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
            "path_t=%.3fm net_t=%.3fm ratio=%.2f path_r=%.2fdeg "
            "last_step=[%.4fm,%.3fdeg]",
            ndt_optimizer.hessian_valid ? 1 : 0,
            ndt_optimizer.hessian_eigenvalues[0],
            ndt_optimizer.hessian_eigenvalues[1],
            ndt_optimizer.hessian_eigenvalues[2],
            ndt_optimizer.hessian_eigenvalues[3],
            ndt_optimizer.hessian_eigenvalues[4],
            ndt_optimizer.hessian_eigenvalues[5],
            ndt_optimizer.hessian_positive, ndt_optimizer.hessian_negative,
            ndt_optimizer.hessian_near_zero,
            ndt_optimizer.hessian_abs_condition,
            std::abs(ndt_optimizer.weakest_direction[0]),
            std::abs(ndt_optimizer.weakest_direction[1]),
            std::abs(ndt_optimizer.weakest_direction[2]),
            std::abs(ndt_optimizer.weakest_direction[3]),
            std::abs(ndt_optimizer.weakest_direction[4]),
            std::abs(ndt_optimizer.weakest_direction[5]),
            ndt_optimizer.iteration_translation_path_m,
            ndt_optimizer.net_translation_m,
            ndt_optimizer.translation_path_ratio,
            ndt_optimizer.iteration_rotation_path_deg,
            ndt_optimizer.final_step_translation_m,
            ndt_optimizer.final_step_rotation_deg);
        } else {
          RCLCPP_WARN(
            get_logger(),
            "[NDT OPT REJECT] sensor_ns=%ld solver_converged=%d accepted=0 iters=%d/%d transforms=%zu "
            "score=%.4f init=[%.3f,%.3f,%.3f] final=[%.3f,%.3f,%.3f] "
            "corr_local=[%+.3f,%+.3f,%+.3f] corr_rpy=[%+.2f,%+.2f,%+.2f]deg",
            static_cast<long>(rclcpp::Time(stamp).nanoseconds()),
            ndt_solver_converged ? 1 : 0, ndt_iters, ndt_max_iterations_,
            ndt_optimizer.transformation_count, match_result.fitness_score_,
            ndt_optimizer.initial_xyz[0], ndt_optimizer.initial_xyz[1],
            ndt_optimizer.initial_xyz[2], ndt_optimizer.final_xyz[0],
            ndt_optimizer.final_xyz[1], ndt_optimizer.final_xyz[2],
            ndt_optimizer.correction_xyz_local[0],
            ndt_optimizer.correction_xyz_local[1],
            ndt_optimizer.correction_xyz_local[2],
            ndt_optimizer.correction_rpy_deg[0],
            ndt_optimizer.correction_rpy_deg[1],
            ndt_optimizer.correction_rpy_deg[2]);
          RCLCPP_WARN(
            get_logger(),
            "[NDT CURV REJECT] valid=%d eig=[%+.2e,%+.2e,%+.2e,%+.2e,%+.2e,%+.2e] "
            "inertia=[+%d,-%d,0:%d] abs_cond=%.3e weak_abs=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
            "path_t=%.3fm net_t=%.3fm ratio=%.2f path_r=%.2fdeg "
            "last_step=[%.4fm,%.3fdeg]",
            ndt_optimizer.hessian_valid ? 1 : 0,
            ndt_optimizer.hessian_eigenvalues[0],
            ndt_optimizer.hessian_eigenvalues[1],
            ndt_optimizer.hessian_eigenvalues[2],
            ndt_optimizer.hessian_eigenvalues[3],
            ndt_optimizer.hessian_eigenvalues[4],
            ndt_optimizer.hessian_eigenvalues[5],
            ndt_optimizer.hessian_positive, ndt_optimizer.hessian_negative,
            ndt_optimizer.hessian_near_zero,
            ndt_optimizer.hessian_abs_condition,
            std::abs(ndt_optimizer.weakest_direction[0]),
            std::abs(ndt_optimizer.weakest_direction[1]),
            std::abs(ndt_optimizer.weakest_direction[2]),
            std::abs(ndt_optimizer.weakest_direction[3]),
            std::abs(ndt_optimizer.weakest_direction[4]),
            std::abs(ndt_optimizer.weakest_direction[5]),
            ndt_optimizer.iteration_translation_path_m,
            ndt_optimizer.net_translation_m,
            ndt_optimizer.translation_path_ratio,
            ndt_optimizer.iteration_rotation_path_deg,
            ndt_optimizer.final_step_translation_m,
            ndt_optimizer.final_step_rotation_deg);
        }
      }

      if (ndt_detailed_diagnostics_enabled_ && td.position_chain_valid) {
        const long sensor_ns = static_cast<long>(rclcpp::Time(stamp).nanoseconds());
        RCLCPP_INFO(
          get_logger(),
          "[NDT CHAIN] sensor_ns=%ld accepted=%d odom_valid=%d "
          "imu_pred=%d imu_fallback=%d imu_age_ms=%+.1f fallback_dt_ms=%.1f "
          "imu=[%.6f,%.6f,%.6f] odom=[%.6f,%.6f,%.6f] "
          "init=[%.6f,%.6f,%.6f] raw=[%.6f,%.6f,%.6f] "
          "ukf_pre=[%.6f,%.6f,%.6f] ukf_post=[%.6f,%.6f,%.6f] "
          "v_pre=[%.6f,%.6f,%.6f] v_post=[%.6f,%.6f,%.6f] "
          "Ppos=[%.3e,%.3e,%.3e] Rpos=%.3e maha=%.6f motion=%.4f corrected=%d",
          sensor_ns, ndt_accepted ? 1 : 0, td.odom_pred_valid ? 1 : 0,
          td.imu_prediction_available ? 1 : 0,
          td.imu_fallback_active ? 1 : 0,
          td.imu_age_ms, td.imu_fallback_dt_ms,
          td.pred_imu_x, td.pred_imu_y, td.pred_imu_z_full,
          td.pred_odom_x, td.pred_odom_y, td.pred_odom_z_full,
          td.pred_used_x, td.pred_used_y, td.pred_used_z_full,
          td.ndt_x, td.ndt_y, td.ndt_z,
          td.ukf_pre_x, td.ukf_pre_y, td.ukf_pre_z,
          td.ukf_post_x, td.ukf_post_y, td.ukf_post_z,
          td.ukf_pre_vx, td.ukf_pre_vy, td.ukf_pre_vz,
          td.ukf_post_vx, td.ukf_post_vy, td.ukf_post_vz,
          td.ukf_pos_cov_x, td.ukf_pos_cov_y, td.ukf_pos_cov_z,
          td.ukf_measurement_pos_variance, td.ukf_mahalanobis_sq,
          td.ukf_motion_scale, td.ukf_correct_applied ? 1 : 0);

        for (const auto& iteration : ndt_iteration_diagnostics) {
          const NdtOptimizerDiagnostics curvature =
            analyzeNdtOptimizer(iteration.hessian, {});
          const double hit_ratio = iteration.source_points > 0
            ? static_cast<double>(iteration.points_with_neighborhood) /
              static_cast<double>(iteration.source_points)
            : 0.0;
          const double neighbors_per_hit = iteration.points_with_neighborhood > 0
            ? static_cast<double>(iteration.neighborhood_count) /
              static_cast<double>(iteration.points_with_neighborhood)
            : 0.0;
          const double score_per_source = iteration.source_points > 0
            ? iteration.score / static_cast<double>(iteration.source_points)
            : 0.0;
          const double raw_step_t = iteration.raw_newton_step.head<3>().norm();
          const double raw_step_r_deg =
            iteration.raw_newton_step.tail<3>().norm() * 180.0 / M_PI;
          const double applied_step_t = iteration.applied_step.head<3>().norm();
          const double applied_step_r_deg =
            iteration.applied_step.tail<3>().norm() * 180.0 / M_PI;
          RCLCPP_INFO(
            get_logger(),
            "[NDT ITER] sensor_ns=%ld k=%d/%d "
            "pose=[%.6f,%.6f,%.6f,%+.4f,%+.4f,%+.4f] "
            "score=%.9e score_src=%.9e matched=%zu/%zu hit=%.4f "
            "neighbors=%d per_hit=%.3f grad_norm=%.6e "
            "grad=[%+.3e,%+.3e,%+.3e,%+.3e,%+.3e,%+.3e] "
            "raw_step=[%.6fm,%.4fdeg] applied=[%.6fm,%.4fdeg] "
            "applied_vec=[%+.6f,%+.6f,%+.6f,%+.5f,%+.5f,%+.5f] "
            "cond=%.6e weak_abs=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]",
            sensor_ns, iteration.iteration, ndt_iters,
            iteration.pose[0], iteration.pose[1], iteration.pose[2],
            iteration.pose[3] * 180.0 / M_PI,
            iteration.pose[4] * 180.0 / M_PI,
            iteration.pose[5] * 180.0 / M_PI,
            iteration.score, score_per_source,
            iteration.points_with_neighborhood, iteration.source_points,
            hit_ratio, iteration.neighborhood_count, neighbors_per_hit,
            iteration.gradient.norm(),
            iteration.gradient[0], iteration.gradient[1], iteration.gradient[2],
            iteration.gradient[3], iteration.gradient[4], iteration.gradient[5],
            raw_step_t, raw_step_r_deg, applied_step_t, applied_step_r_deg,
            iteration.applied_step[0], iteration.applied_step[1],
            iteration.applied_step[2], iteration.applied_step[3],
            iteration.applied_step[4], iteration.applied_step[5],
            curvature.hessian_abs_condition,
            std::abs(curvature.weakest_direction[0]),
            std::abs(curvature.weakest_direction[1]),
            std::abs(curvature.weakest_direction[2]),
            std::abs(curvature.weakest_direction[3]),
            std::abs(curvature.weakest_direction[4]),
            std::abs(curvature.weakest_direction[5]));
        }
      }

      // 管线滞后归因（判据见 points_callback 入口处的注释）
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[LAG] entry_age=%.1f stamp_gap=%.1f wall_gap=%.1f drift=%+.1f cum=%.0fms",
        lag_entry_age_ms_, lag_stamp_gap_ms_, lag_wall_gap_ms_,
        lag_wall_gap_ms_ - lag_stamp_gap_ms_, lag_cum_ms_);

      // IMU 供给：usable<5 时本帧 predict 一次都不会被调用
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[IMU SUP] buf=%zu usable=%zu lead=%+.1fms age=%+.1fms "
        "expect_predicts=%zu pred_used=%d fallback=%d fallback_dt=%.1fms",
        imu_buf_size_, imu_usable_, imu_lead_ms_,
        imu_age_ms,
        imu_usable_ / static_cast<size_t>(std::max(1, imu_data_filter_num_)),
        imu_prediction_used ? 1 : 0, imu_fallback_active ? 1 : 0,
        imu_fallback_dt_s * 1000.0);

      // 去畸变细分：四段之和应≈stage_ms_deskew_，谁大就是谁的问题
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[DESKEW] win=%.2f gyro=%.2f slice=%.2f pts=%.2f | sum=%.1f measured=%.1fms "
        "| win_size=%zu skipped=%d",
        dsk_ms_win_, dsk_ms_gyro_, dsk_ms_slice_, dsk_ms_pts_,
        dsk_ms_win_ + dsk_ms_gyro_ + dsk_ms_slice_ + dsk_ms_pts_,
        stage_ms_deskew_, dsk_win_size_, dsk_skipped_ ? 1 : 0);
    }

    // Tracking health state machine: a short rejection freezes the last map->odom
    // while publishing valid=false. Only a timeout enters global relocalization.
    {
      const bool ndt_rejected = !ndt_accepted;
      if (is_init_success_) {
        if (ndt_rejected) {
          consecutive_ndt_rejections_++;
        } else {
          consecutive_ndt_rejections_ = 0;
        }

        const auto health = tracking_health_.observe(
          ndt_accepted, rclcpp::Time(stamp).seconds());
        localization_state_ = health.localization_valid ? 3 : 4;
        if (health.state_changed &&
            health.state == TrackingHealthMonitor::State::Grace) {
          pose_estimator->reset_velocity();
          std::lock_guard<std::mutex> lock(imu_data_mutex);
          imu_data.clear();
          RCLCPP_WARN(
            get_logger(),
            "NDT tracking entered rejection grace: velocity reset, map->odom frozen");
        }
        const bool sensor_recovered =
          sensor_invalid_latched_ && health.localization_valid && ndt_accepted;
        if (sensor_recovered) {
          sensor_invalid_latched_ = false;
        }
        if (health.state_changed || !health.localization_valid || sensor_recovered) {
          publishGlobalLocalizationDiagnostics(health.state_changed || sensor_recovered);
        }

        if (health.enter_relocalization) {
          const bool gl_runtime_enabled = globalLocalizationRuntimeEnabled();
          const bool start_gl_relocalization =
            gl_runtime_enabled && gl_relocalize_on_tracking_failure_;
          RCLCPP_ERROR(get_logger(),
            "NDT tracking exceeded %.2fs rejection grace after %d rejected frames. "
            "Recovery: velocity reset + %s. "
            "pos=[%.1f,%.1f,%.1f] score=%.1f vel=[%.2f,%.2f,%.2f]",
            localization_rejection_timeout_s_, consecutive_ndt_rejections_,
            start_gl_relocalization
              ? "global localization"
              : "manual initialpose request (tracking-failure GL disabled)",
            corr_p.x(), corr_p.y(), corr_p.z(),
            match_result.fitness_score_,
            ukf_vel.x(), ukf_vel.y(), ukf_vel.z());
          pose_estimator->reset_velocity();
          {
            std::lock_guard<std::mutex> lock(imu_data_mutex);
            imu_data.clear();
          }
          // Position must be recovered by an approved GL path or manual initialpose;
          // velocity-only reset is not sufficient after the grace timeout.
          is_init_success_ = false;
          resetGlobalLocalizationEpisodeState();
          if (!start_gl_relocalization) {
            // Keep the existing AwaitingPrior gate closed until /initialpose.
            // Otherwise the legacy NDT init-verification path continues from the
            // failed pose even though diagnostics explicitly requested an
            // external prior.
            endGLEpisode(gl_episode_state_, GLStatus::NoCandidate);
          }
          gl_once_gate_ = start_gl_relocalization;
          gl_request_initialpose_ = !start_gl_relocalization;
          publishGlobalLocalizationDiagnostics();
          init_match_count_ = 0;
          has_init_verify_pose_ = false;
          consecutive_ndt_rejections_ = 0;
          localization_state_ = 1;
        }
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    last_timeout_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (last_timeout_ > 80) {
      RCLCPP_INFO(get_logger(), "!!!point cloud callback time cost > 80ms, = %d ms\n", last_timeout_);
    }

    // R11.1 deep trace: compact per-frame quality line.
    // Always log interesting frames; throttle healthy frames to ~2Hz to keep logs analyzable.
    {
      const bool interesting =
        !ndt_accepted ||
        match_result.fitness_score_ >= 0.5f ||
        delta_pos > 0.25f ||
        std::abs(delta_yaw) > 0.15f ||
        std::abs(ukf_vel.z()) > 0.4f ||
        vel_norm > 1.0f ||
        consecutive_ndt_rejections_ > 0 ||
        last_timeout_ > 80;

      const rclcpp::Time now_log = get_clock()->now();
      const bool throttle_ok =
        !has_last_ndt_q_log_ ||
        (now_log - last_ndt_q_log_time_).seconds() >= 0.5;

      // [2026-08-11] Per-frame NDT tracking log disabled on request: keep only
      // the GL candidate / vote / confirmation logs. Kept as reference so the
      // line can be re-enabled for offline deep dives.
      // if (interesting || throttle_ok) {
      //   RCLCPP_INFO(
      //     get_logger(),
      //     "[NDT Q] score=%.4f conv=%d accept=%d rej_streak=%d "
      //     "pos=[%.3f,%.3f,%.3f] yaw=%.3f pred=[%.3f,%.3f,%.3f] pred_yaw=%.3f "
      //     "dpos=%.4f dyaw=%.4f vel=[%.3f,%.3f,%.3f] |v|=%.3f vz=%.3f "
      //     "motion=%.2f warmup=%d cost_ms=%d state=%d",
      //     match_result.fitness_score_,
      //     match_result.is_converged_ ? 1 : 0,
      //     ndt_accepted ? 1 : 0,
      //     consecutive_ndt_rejections_,
      //     corr_p.x(), corr_p.y(), corr_p.z(), corr_yaw,
      //     pred_p.x(), pred_p.y(), pred_p.z(), pred_yaw,
      //     delta_pos, delta_yaw,
      //     ukf_vel.x(), ukf_vel.y(), ukf_vel.z(),
      //     vel_norm, ukf_vel.z(),
      //     pose_estimator->motion_scale(),
      //     pose_estimator->tracking_gate_warmup_remaining(),
      //     last_timeout_,
      //     localization_state_);
      //     last_ndt_q_log_time_ = now_log;
      //     has_last_ndt_q_log_ = true;
      // }
      (void)interesting;
      (void)throttle_ok;

      // Full verbose line kept at DEBUG for offline deep dives if needed
      RCLCPP_DEBUG(
        get_logger(),
        "[NDT DIAG] ndt_score=%.4f converged=%d pos=[%.3f,%.3f,%.3f] yaw=%.3f "
        "pred_pos=[%.3f,%.3f,%.3f] pred_yaw=%.3f delta_pos=%.4f delta_yaw=%.4f",
        match_result.fitness_score_, match_result.is_converged_,
        corr_p.x(), corr_p.y(), corr_p.z(), corr_yaw,
        pred_p.x(), pred_p.y(), pred_p.z(), pred_yaw,
        delta_pos, delta_yaw);
    }

    if (aligned_pub->get_subscription_count()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      sensor_msgs::msg::PointCloud2 aligned_msg;
      pcl::toROSMsg(*aligned, aligned_msg);
      aligned_pub->publish(aligned_msg);
    }
    // Update pose history for extrapolation
    if (ndt_accepted) {
      Eigen::Matrix4f current_pose = pose_estimator->matrix();
      Eigen::Vector3f current_velocity = getCurrentVelocity(current_pose, points_msg->header.stamp);
      Eigen::Vector3f current_angular_velocity = getCurrentAngularVelocity(current_pose, points_msg->header.stamp);
      updatePoseHistory(current_pose, current_velocity, current_angular_velocity, points_msg->header.stamp);
      is_extrapolating_ = false;
      current_confidence_ = 1.0;
      last_confidence_update_time_ = points_msg->header.stamp;
    }
    publish_odometry(points_msg->header.stamp, pose_estimator->matrix());
  }

  /**
   * @brief callback for initial pose input 
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_msg) {
    RCLCPP_INFO(get_logger(), "initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const bool initialpose_requested = gl_request_initialpose_;
    resetGlobalLocalizationEpisodeState();
  
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    Eigen::Vector3f new_pos(p.x, p.y, p.z);
    Eigen::Quaternionf new_quat(q.w, q.x, q.y, q.z);
    
    const float pos_change = has_set_init_pose_ ?
      (new_pos - last_init_pos_).norm() : 0.0f;
    const float quat_change = has_set_init_pose_ ?
      std::abs(1.0f - std::abs(last_init_quat_.dot(new_quat))) : 0.0f;
    const bool pose_changed = initialPoseRequiresReset(
      has_set_init_pose_, is_init_success_, initialpose_requested,
      pos_change, quat_change, init_pose_change_threshold_,
      init_quat_change_threshold_);
    if (has_set_init_pose_ && is_init_success_ && !initialpose_requested && pose_changed) {
      RCLCPP_INFO(
        get_logger(), "Pose change detected - Position: %.3f m, Orientation: %.3f",
        pos_change, quat_change);
    }
    
    if (pose_changed) {
      last_init_pos_ = new_pos;
      last_init_quat_ = new_quat;
      has_set_init_pose_ = true;
      last_pose_source_ = "Callback";
      // Keep yaw from RViz; static gravity initialization supplies roll/pitch.
      Eigen::Quaternionf initial_quat = last_init_quat_;
      pose_estimator.reset(new localization::PoseEstimator(
        registration, get_clock()->now(), last_init_pos_, initial_quat,
        cool_time_duration, ndt_score_threshold_,
        !enable_vertical_velocity_prediction_));
      applyImuBiasIfReady();
      pose_estimator->clear_trust_anchor();
      // restart init verification and disable GL for this round
      is_init_success_ = false;
      init_match_count_ = 0;
      has_init_verify_pose_ = false;  // Round 10: reset pose consistency tracker
      localization_state_ = 1;
      gl_once_gate_ = false;
      gl_request_initialpose_ = false;
      publishGlobalLocalizationDiagnostics();
      RCLCPP_INFO(get_logger(), "New initial pose set from RViz - Position: [%.3f, %.3f, %.3f], Quaternion: [%.3f, %.3f, %.3f, %.3f]",
                   last_init_pos_.x(), last_init_pos_.y(), last_init_pos_.z(), last_init_quat_.w(), last_init_quat_.x(), last_init_quat_.y(), last_init_quat_.z());
      RCLCPP_INFO(get_logger(), "Localization will restart with new pose");
    } else {
      RCLCPP_INFO(get_logger(), "Pose unchanged, no action needed");
    }
  }

  pcl::PointCloud<PointT>::Ptr downsample(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    if (!voxel_filter_ptr_) {
      return cloud;
    }
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxel_filter_ptr_->setInputCloud(cloud);
    voxel_filter_ptr_->filter(*filtered);
    filtered->header = cloud->header;
    return filtered;
  }

  /**
   * @brief 截掉高处（天花板）点，仅用于 NDT 配准
   *
   * [2026-08-17] 依据实测：地图中 z>1.5m 的点占 40.3%，但它们沿走廊均匀延伸、
   * 几乎不含纵向特征。NDT 是所有点一起最小化，这 40% 的无区分度点把有区分度的
   * 墙面/近地面特征稀释掉了。
   * 离线量化（三个实测落位点平均，沿 y 平移 0.5m 的平均最近邻残差）：
   *     无上限 0.0491m → 2.0m: 0.0714(+45%) → 1.8m: 0.0787(+60%)
   *     → **1.5m: 0.0840(+71%)** → 1.2m: 0.0828(+69%) → 1.0m: 0.0823(+68%)
   * 1.5m 是拐点，再往下砍会损失有用的墙面点、收益反而下降。
   * 残差越大 = 沿该方向平移越容易被察觉 = 越不容易滑走。
   *
   * ⚠️ 只能用于 NDT。GL 的描述子库是用**未过滤**的地图建的，过滤后送进 GL 会
   * 让检索失配，因此保留独立的未过滤副本给 GL。
   *
   * 高度参考系说明：地图点在 map 系、实时扫描在雷达系。实测地面在 map 系约
   * z=-0.25~-0.30，而机体/雷达落位在 map z≈0，两者近似对齐，故同一阈值对两侧
   * 都近似表示"传感器高度以上 N 米"。这依赖 flat_single_level_map=true。
   */
  pcl::PointCloud<PointT>::Ptr filterByHeight(
      const pcl::PointCloud<PointT>::Ptr& cloud, float z_max) const {
    if (!cloud || z_max <= 0.0f) {
      return cloud;
    }
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    filtered->reserve(cloud->size());
    for (const auto& pt : cloud->points) {
      if (std::isfinite(pt.z) && pt.z <= z_max) {
        filtered->push_back(pt);
      }
    }
    filtered->header = cloud->header;
    filtered->width = static_cast<uint32_t>(filtered->size());
    filtered->height = 1;
    filtered->is_dense = false;
    return filtered;
  }

  void publish_odometry(const rclcpp::Time& stamp, const Eigen::Matrix4f& pose) {
    const bool freeze_tracking_tf =
      is_init_success_ && tracking_health_.freezeTf() && isSensorDataValid();
    const GLOutputGate output_gate = decideGLOutputGate(
      gl_episode_state_, localizationOutputEligible() || freeze_tracking_tf);
    if (!output_gate.publish_localization_odom) {
      RCLCPP_DEBUG_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "[GL gate] localization odom/TF suppressed: episode_active=%d init_verified=%d",
        gl_episode_state_.active ? 1 : 0,
        localizationOutputEligible() ? 1 : 0);
      return;
    }

    if (freeze_tracking_tf && !has_last_valid_localization_pose_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Tracking is degraded but no valid pose is cached; odom/TF suppressed");
      return;
    }

    const Eigen::Matrix4f& output_pose = freeze_tracking_tf
      ? last_valid_localization_pose_
      : pose;
    if (!freeze_tracking_tf) {
      last_valid_localization_pose_ = pose;
      has_last_valid_localization_pose_ = true;
    }

    // Final sanity gate: refuse to publish if UKF position is unreasonably far from map origin
    // (indicates UKF corruption that survived all upstream guards)
    Eigen::Vector3f pos_check(
      output_pose(0, 3), output_pose(1, 3), output_pose(2, 3));
    if (pos_check.norm() > 50.0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[publish_odometry] SUPPRESSED: UKF position norm=%.1f > 50m, skipping publish to protect navigation",
        pos_check.norm());
      return;
    }
    RCLCPP_DEBUG(
      get_logger(),
      "[publish_odometry] stamp_ns=%ld now_ns=%ld send_tf_transforms=%s frame_id=%s child_frame_id=%s pose_xyz=[%.3f, %.3f, %.3f]",
      static_cast<long>(stamp.nanoseconds()),
      static_cast<long>(get_clock()->now().nanoseconds()),
      send_tf_transforms ? "true" : "false",
      robot_odom_frame_id.c_str(),
      localization_odom_frame_id.c_str(),
      output_pose(0, 3), output_pose(1, 3), output_pose(2, 3));
    if (send_tf_transforms && output_gate.publish_map_tf) {
      // Use current time for TF publication to avoid TF_OLD_DATA rejection by nav2.
      rclcpp::Time tf_now = get_clock()->now();

      if (freeze_tracking_tf) {
        if (has_last_valid_map_to_odom_) {
          geometry_msgs::msg::TransformStamped frozen_tf = last_valid_map_to_odom_;
          frozen_tf.header.stamp = tf_now;
          tf_broadcaster->sendTransform(frozen_tf);
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Localization invalid during rejection grace; republishing frozen map->%s TF",
            robot_odom_frame_id.c_str());
        } else {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Localization invalid during rejection grace but no map->odom TF is cached");
        }
      } else {
        // map→odom is only valid when the matching odom→base_link transform is available.
        const bool same_odom_frames = (robot_odom_frame_id == odom_child_frame_id);
        bool can_lookup_odom_child = false;
        if (!same_odom_frames &&
            tf_buffer->_frameExists(robot_odom_frame_id) &&
            tf_buffer->_frameExists(odom_child_frame_id)) {
          try {
            can_lookup_odom_child = tf_buffer->canTransform(
              robot_odom_frame_id, odom_child_frame_id,
              stamp,
              tf2::durationFromSec(0.0));
          } catch (const tf2::TransformException &) {
            can_lookup_odom_child = false;
          }
        }

        if (can_lookup_odom_child) {
          RCLCPP_DEBUG(
            get_logger(),
            "[publish_odometry] canTransform(%s <- %s) = true",
            robot_odom_frame_id.c_str(), odom_child_frame_id.c_str());
          try {
            geometry_msgs::msg::TransformStamped map_wrt_frame =
              tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
            map_wrt_frame.header.stamp = stamp;
            map_wrt_frame.header.frame_id = odom_child_frame_id;
            map_wrt_frame.child_frame_id = "map";

            geometry_msgs::msg::TransformStamped frame_wrt_odom = tf_buffer->lookupTransform(
              robot_odom_frame_id,
              odom_child_frame_id,
              stamp);

            geometry_msgs::msg::TransformStamped map_wrt_odom;
            tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

            tf2::Transform odom_wrt_map;
            tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
            odom_wrt_map = odom_wrt_map.inverse();

            geometry_msgs::msg::TransformStamped odom_trans;
            odom_trans.transform = tf2::toMsg(odom_wrt_map);
            odom_trans.header.stamp = tf_now;
            odom_trans.header.frame_id = "map";
            odom_trans.child_frame_id = robot_odom_frame_id;

            tf_broadcaster->sendTransform(odom_trans);
            last_valid_map_to_odom_ = odom_trans;
            has_last_valid_map_to_odom_ = true;
            RCLCPP_DEBUG(
              get_logger(),
              "[publish_odometry] broadcast corrective TF map -> %s",
              robot_odom_frame_id.c_str());
          } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 5000,
              "[publish_odometry] odom TF lookup failed at scan stamp; corrective TF suppressed: %s",
              ex.what());
          }
        } else if (!same_odom_frames) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "[publish_odometry] canTransform(%s <- %s) = false "
            "(is the odometry source publishing odom->base_link?). "
            "Corrective map->%s TF suppressed",
            robot_odom_frame_id.c_str(), odom_child_frame_id.c_str(),
            robot_odom_frame_id.c_str());
        }
      }
    }
    // publish the transform
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";
    odom.pose.pose = tf2::toMsg(Eigen::Isometry3d(output_pose.cast<double>()));
    odom.child_frame_id = localization_odom_frame_id;
    // Use confidence value directly, 1.0 means completely reliable, 0.0 means completely unreliable
    double confidence = getCurrentConfidence();
    odom.pose.covariance[0] = confidence;  // Use confidence value directly
    const Eigen::Vector3f published_velocity =
      (!freeze_tracking_tf && pose_estimator)
      ? pose_estimator->vel()
      : Eigen::Vector3f::Zero();
    odom.twist.twist.linear.x = published_velocity.x();
    odom.twist.twist.linear.y = published_velocity.y();
    odom.twist.twist.linear.z = published_velocity.z();
    odom.twist.twist.angular.z =
      freeze_tracking_tf ? 0.0 : latest_angular_velocity_.z();
    pose_pub->publish(odom);
    RCLCPP_DEBUG(
      get_logger(),
      "[publish_odometry] published odom header_ns=%ld frame_id=%s child_frame_id=%s",
      static_cast<long>(static_cast<long long>(odom.header.stamp.sec) * 1000000000LL + odom.header.stamp.nanosec),
      odom.header.frame_id.c_str(),
      odom.child_frame_id.c_str());
  }

  /**
   * @brief Publish default localization information (position is 0)
   * @param stamp timestamp
   */
  void pubDefaultLocalizationOdom(const rclcpp::Time& stamp) {
    (void)stamp;
    is_extrapolating_ = false;
    current_confidence_ = 0.0;
    RCLCPP_DEBUG_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Localization pose is invalid; identity odom/TF publication is suppressed");
  }

  void recordGravityAlignmentSample(
      const sensor_msgs::msg::Imu& imu_msg,
      const Eigen::Vector3d& acceleration_body,
      const Eigen::Vector3d& angular_velocity_body) {
    // GL consumes the same base-frame cloud as NDT. The raw-orientation fallback
    // must therefore use acceleration/gyro after the fixed IMU->base rotation;
    // otherwise the mounting pitch is mistaken for body tilt and trips the GL
    // anchor gate before any candidate is evaluated.
    GravityAlignmentSample raw_sample;
    const bool raw_sample_valid =
      gl_params_.level0_raw_imu_gravity_filter_enabled &&
      gl_raw_imu_gravity_filter_.addSample(
        rclcpp::Time(imu_msg.header.stamp), acceleration_body,
        angular_velocity_body, raw_sample);

    const auto& orientation = imu_msg.orientation;
    const Eigen::Quaterniond world_from_sensor(
      orientation.w, orientation.x, orientation.y, orientation.z);
    const Eigen::Matrix3d body_from_sensor =
      init_rotation_matrix_.cast<double>();
    GravityAlignmentSample fused_sample;
    const bool fused_sample_valid = makeGravitySampleFromOrientation(
        rclcpp::Time(imu_msg.header.stamp), world_from_sensor,
        Eigen::Vector3d(
          imu_msg.orientation_covariance[0],
          imu_msg.orientation_covariance[4],
          imu_msg.orientation_covariance[8]),
        body_from_sensor, acceleration_body, angular_velocity_body,
        "imu_orientation", fused_sample);
    if (!fused_sample_valid && !raw_sample_valid) {
      return;
    }
    const GravityAlignmentSample& sample =
      fused_sample_valid ? fused_sample : raw_sample;
    if (!fused_sample_valid) {
      RCLCPP_INFO_ONCE(
        get_logger(),
        "Raw IMU gravity fallback ready: frame=base_after_init_R "
        "roll=%.3fdeg pitch=%.3fdeg uncertainty=%.3fdeg samples=%zu",
        sample.roll_deg, sample.pitch_deg, sample.uncertainty_deg,
        gl_raw_imu_gravity_filter_.sampleCount());
    }

    std::lock_guard<std::mutex> lock(gl_gravity_history_mutex_);
    if (!gl_gravity_history_.empty()) {
      const GravityAlignmentSample& last = gl_gravity_history_.back();
      if (last.stamp.get_clock_type() != sample.stamp.get_clock_type() ||
          sample.stamp <= last.stamp) {
        return;
      }
    }
    gl_gravity_history_.push_back(sample);
    while (gl_gravity_history_.size() > 512U) {
      gl_gravity_history_.pop_front();
    }
    while (!gl_gravity_history_.empty() &&
        (sample.stamp - gl_gravity_history_.front().stamp).seconds() > 2.0) {
      gl_gravity_history_.pop_front();
    }
  }

  bool selectGravityForScan(const rclcpp::Time& scan_stamp,
                            GravityAlignmentSample& selected) {
    std::vector<GravityAlignmentSample> history;
    {
      std::lock_guard<std::mutex> lock(gl_gravity_history_mutex_);
      history.assign(gl_gravity_history_.begin(), gl_gravity_history_.end());
    }
    return selectGravitySampleForAnchor(
      history, scan_stamp, gl_params_, selected);
  }

  bool runGlobalLocalizationShadowProbe(
      const pcl::PointCloud<PointT>::Ptr& current_cloud,
      const rclcpp::Time& scan_stamp,
      const Eigen::Matrix4d* initial_pose_override = nullptr,
      GlobalLocalizationResult* result_output = nullptr,
      bool export_probe = true) {
    if (!gl_online_shadow_enabled_ || !gl_episode_runtime_enabled_) {
      RCLCPP_INFO_ONCE(
        get_logger(),
        "Global localization episode runtime is disabled; no fallback shadow "
        "probe or pose evaluation will be performed");
      return false;
    }
    if (!global_localization_ptr_ || !global_map_points_ptr_) {
      RCLCPP_WARN(get_logger(), "Global localization not available");
      return false;
    }
    global_localization_in_progress_ = true;
    global_localization_start_time_ = get_clock()->now();
    const auto started = std::chrono::steady_clock::now();
    bool probe_completed = false;
    try {
      Eigen::Vector3f init_pos;
      Eigen::Quaternionf init_quat;
      if (!getCurrentInitPose(init_pos, init_quat)) {
        RCLCPP_WARN(get_logger(), "Failed to get initial pose for GL shadow");
        global_localization_in_progress_ = false;
        return false;
      }

      Eigen::Matrix4d initial_trans = Eigen::Matrix4d::Identity();
      if (initial_pose_override != nullptr && initial_pose_override->allFinite()) {
        initial_trans = *initial_pose_override;
      } else {
        initial_trans.block<3, 1>(0, 3) = init_pos.cast<double>();
        initial_trans.block<3, 3>(0, 0) = init_quat.toRotationMatrix().cast<double>();
      }

      GravityAlignmentSample gravity;
      const bool gravity_selected = selectGravityForScan(scan_stamp, gravity);
      const GravityAlignmentSample* gravity_ptr = nullptr;
      if (gravity_selected) {
        gravity_ptr = &gravity;
      }
      if (initial_pose_override != nullptr) {
        double gravity_age_ms = -1.0;
        if (gravity_selected &&
            gravity.stamp.get_clock_type() == scan_stamp.get_clock_type()) {
          gravity_age_ms =
              std::abs((gravity.stamp - scan_stamp).seconds()) * 1000.0;
        }
        // A temporal probe with an unusable gravity sample cannot produce
        // candidates — performGlobalLocalization rejects it before the
        // distance-field build — so running it would only ingest a
        // guaranteed-empty NoCandidate and waste the motion trigger. Defer
        // instead: the last-temporal odom stays un-updated, keeping the
        // trigger armed, and the probe retries once the robot has been still
        // long enough for the raw-IMU gravity filter to converge.
        bool gravity_usable = gravity_selected && gravity.valid;
        GLAnchorRejectReason gravity_reject =
            GLAnchorRejectReason::GravityUnavailable;
        if (gravity_usable) {
          const GravityGateResult gravity_gate =
              evaluateGravitySample(scan_stamp, &gravity, gl_params_);
          gravity_usable = gravity_gate.accepted;
          gravity_reject = gravity_gate.reject_reason;
        }
        if (gravity_selected) {
          RCLCPP_INFO(
              get_logger(),
              "[GL GRAVITY] probe=temporal selected=1 valid=%d usable=%d source=%s age_ms=%.3f uncertainty_deg=%.3f",
              gravity.valid ? 1 : 0, gravity_usable ? 1 : 0,
              gravity.source.c_str(), gravity_age_ms,
              gravity.uncertainty_deg);
        }
        if (!gravity_usable) {
          RCLCPP_INFO_THROTTLE(
              get_logger(), *get_clock(), 1000,
              "[GL GRAVITY] probe=temporal usable=0 selected=%d valid=%d reject_reason=%d; deferring probe and preserving motion evidence",
              gravity_selected ? 1 : 0,
              gravity_selected && gravity.valid ? 1 : 0,
              static_cast<int>(gravity_reject));
          global_localization_in_progress_ = false;
          return false;
        }
      }

      Eigen::Matrix4d ukf_pose;
      const Eigen::Matrix4d* ukf_pose_ptr = nullptr;
      if (has_valid_pose_history_ && pose_estimator) {
        ukf_pose = pose_estimator->matrix().cast<double>();
        ukf_pose_ptr = &ukf_pose;
      }

      GlobalLocalizationResult result;
      global_localization_ptr_->performGlobalLocalization(
        global_map_points_ptr_, current_cloud, initial_trans, ukf_pose_ptr,
        nullptr, gravity_ptr, result, &gl_summary_);
      probe_completed = true;
      result.budget_exceeded = gl_episode_state_.budget_exceeded;
      result.anchor_invalid = gl_episode_state_.anchor_invalid;
      const double cost_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      gl_summary_.processing_ms = cost_ms;
      if (!std::isfinite(gl_params_.per_call_deadline_ms) ||
          cost_ms > gl_params_.per_call_deadline_ms) {
        result.budget_exceeded = true;
        gl_episode_state_.budget_exceeded = true;
        RCLCPP_WARN(
            get_logger(),
            "[GL budget] processing_ms=%.3f exceeded per_call_deadline_ms=%.3f; result is diagnostic-only",
            cost_ms, gl_params_.per_call_deadline_ms);
      }
      if (result_output != nullptr) {
        *result_output = result;
      }
      if (export_probe && gl_candidates_csv_writer_) {
        GLCandidateCsvContext context;
        context.episode_id = gl_params_.candidates_csv_episode_id;
        context.scan_stamp_ns = scan_stamp.nanoseconds();
        context.processing_ms = cost_ms;
        std::string csv_error;
        const bool include_decision =
            !globalLocalizationRuntimeEnabled();
        if (!gl_candidates_csv_writer_->appendProbe(
                context, result, gl_summary_, include_decision, csv_error)) {
          RCLCPP_ERROR(
            get_logger(), "Failed to append GL candidate CSV: %s",
            csv_error.c_str());
        }
      }
      RCLCPP_INFO(
        get_logger(), "%s",
        formatGLSummary(result, gl_summary_, gl_episode_state_, cost_ms).c_str());
      if (result.status == GLStatus::Confirmed || result.success) {
        RCLCPP_ERROR(
          get_logger(),
          "GL online shadow produced a usable pose unexpectedly; result ignored");
      }
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Global localization shadow exception: %s", e.what());
    }
    global_localization_in_progress_ = false;
    return probe_completed;
  }

  void dumpGlobalLocalizationDecision(
      const rclcpp::Time& scan_stamp,
      const GlobalLocalizationResult& result) {
    if (!gl_candidates_csv_writer_) {
      return;
    }
    GLCandidateCsvContext context;
    context.episode_id = gl_params_.candidates_csv_episode_id;
    context.scan_stamp_ns = scan_stamp.nanoseconds();
    context.processing_ms = gl_summary_.processing_ms;
    std::string csv_error;
    if (!gl_candidates_csv_writer_->appendDecision(
            context, result, gl_summary_, csv_error)) {
      RCLCPP_ERROR(
        get_logger(), "Failed to append GL decision CSV: %s",
        csv_error.c_str());
    }
  }

  bool applyConfirmedGlobalPose(
      const GlobalLocalizationResult& result,
      const rclcpp::Time& scan_stamp) {
    const GLIntegrationAction action = decideGLIntegrationAction(
      result, gl_params_, gl_episode_state_.active, false);
    if (!globalLocalizationRuntimeEnabled() ||
        !action.accept_pose_for_init || !action.run_init_verification) {
      RCLCPP_ERROR(
        get_logger(),
        "[GL APPLY] blocked by node-level runtime/approval contract: "
        "runtime=%d episode_active=%d auto_confirm_effective=%d "
        "accept_pose=%d run_verification=%d",
        globalLocalizationRuntimeEnabled() ? 1 : 0,
        gl_episode_state_.active ? 1 : 0,
        automaticApprovalContractSatisfied(gl_params_) ? 1 : 0,
        action.accept_pose_for_init ? 1 : 0,
        action.run_init_verification ? 1 : 0);
      return false;
    }
    if (result.status != GLStatus::Confirmed || !result.success ||
        !result.final_pose.allFinite() || scan_stamp.nanoseconds() <= 0 ||
        !registration) {
      return false;
    }

    const Eigen::Matrix3d rotation = result.final_pose.block<3, 3>(0, 0);
    Eigen::Quaterniond pose_quaternion(rotation);
    if (!pose_quaternion.coeffs().allFinite() ||
        !std::isfinite(pose_quaternion.norm()) ||
        pose_quaternion.norm() < 1e-6) {
      return false;
    }
    pose_quaternion.normalize();

    const Eigen::Vector3f new_position =
        result.final_pose.block<3, 1>(0, 3).cast<float>();
    const Eigen::Quaternionf new_orientation =
        pose_quaternion.cast<float>();
    if (!new_position.allFinite() || !new_orientation.coeffs().allFinite()) {
      return false;
    }

    // The GL episode owns all scans while active. Discard IMU samples queued
    // against the previous estimator before installing the confirmed pose.
    {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      imu_data.clear();
    }

    last_init_pos_ = new_position;
    last_init_quat_ = new_orientation;
    has_set_init_pose_ = true;
    last_pose_source_ = "GlobalLocalizationAutoConfirm";
    pose_estimator.reset(new localization::PoseEstimator(
      registration, scan_stamp, last_init_pos_, last_init_quat_,
      cool_time_duration, ndt_score_threshold_,
      !enable_vertical_velocity_prediction_));
    applyImuBiasIfReady();
    Eigen::Matrix4f anchor_map_pose = Eigen::Matrix4f::Identity();
    anchor_map_pose.block<3, 3>(0, 0) = last_init_quat_.toRotationMatrix();
    anchor_map_pose.block<3, 1>(0, 3) = last_init_pos_;
    Eigen::Matrix4f anchor_odom_pose;
    if (lookupRobotOdomPose(scan_stamp, anchor_odom_pose)) {
      pose_estimator->set_trust_anchor(anchor_map_pose, anchor_odom_pose);
    } else {
      pose_estimator->clear_trust_anchor();
      RCLCPP_WARN(
        get_logger(),
        "[TRUST] GL pose installed without %s->%s transform at scan time; "
        "trust region disabled for this estimator",
        robot_odom_frame_id.c_str(), odom_child_frame_id.c_str());
    }

    is_init_success_ = false;
    init_match_count_ = 0;
    has_init_verify_pose_ = false;
    consecutive_ndt_rejections_ = 0;
    localization_state_ = 1;
    has_valid_pose_history_ = false;
    is_extrapolating_ = false;
    current_confidence_ = 0.0;
    last_valid_pose_time_ = scan_stamp;
    gl_request_initialpose_ = false;
    gl_once_gate_ = false;
    gl_auto_confirm_pending_verification_ = true;
    gl_auto_confirm_verification_start_time_ = get_clock()->now();
    ++gl_auto_confirm_apply_count_;
    endGLEpisode(gl_episode_state_, GLStatus::Confirmed);

    RCLCPP_WARN(
      get_logger(),
      "[GL APPLY] confirmed pose installed for NDT verification: xyz=[%.3f, %.3f, %.3f] approval_id=%s apply_count=%lu; odom/TF remain suppressed until verification passes",
      last_init_pos_.x(), last_init_pos_.y(), last_init_pos_.z(),
      gl_params_.m2b_approval_id.c_str(),
      static_cast<unsigned long>(gl_auto_confirm_apply_count_));
    publishGlobalLocalizationDiagnostics(true);
    return true;
  }

  bool getCurrentInitPoseForMatrix(Eigen::Matrix4d& pose) {
    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;
    if (!getCurrentInitPose(position, orientation)) {
      return false;
    }
    pose = Eigen::Matrix4d::Identity();
    pose.block<3, 1>(0, 3) = position.cast<double>();
    pose.block<3, 3>(0, 0) = orientation.toRotationMatrix().cast<double>();
    return pose.allFinite();
  }

  bool globalLocalizationRuntimeEnabled() const {
    return gl_episode_runtime_enabled_ && gl_online_shadow_enabled_ &&
      use_global_localization_init_ && global_localization_ptr_ != nullptr;
  }

  StaticConfirmParams staticConfirmParams() const {
    StaticConfirmParams params;
    params.confirm_frames = gl_params_.gl_confirm_frames;
    params.confirm_xy_tol = gl_params_.gl_confirm_xy_tol;
    params.confirm_yaw_tol_deg = gl_params_.gl_confirm_yaw_tol_deg;
    params.probe_period_s = gl_params_.gl_probe_period_s;
    params.attempt_timeout_s = gl_params_.gl_attempt_timeout_s;
    return params;
  }

  /**
   * @brief get current initial pose
   * @param pos output position
   * @param quat output orientation
   * @return true if success
   */
  bool getCurrentInitPose(Eigen::Vector3f& pos, Eigen::Quaternionf& quat) {
    if (has_set_init_pose_) {
      pos = last_init_pos_;
      quat = last_init_quat_;
      return true;
    }
    if (specify_init_pose_) {
      pos = Eigen::Vector3f(init_pos_x_, init_pos_y_, init_pos_z_);
      quat = Eigen::Quaternionf(init_ori_w_, init_ori_x_, init_ori_y_, init_ori_z_);
      return true;
    }
    pos = Eigen::Vector3f::Zero();
    quat = Eigen::Quaternionf::Identity();
    return true;
  }
  
  /**
   * @brief Check if lidar data is valid
   * @return true: lidar data is valid, false: lidar data is invalid
   */
  bool isLidarDataValid() {
    if (lidar_status_buffer_.size() < min_valid_count_) { return false; }
    int valid_count = std::count(lidar_status_buffer_.begin(), lidar_status_buffer_.end(), true);
    return valid_count >= min_valid_count_;
  }

  /**
   * @brief Check if IMU data is valid
   * @return true: IMU data is valid, false: IMU data is invalid
   */
  bool isImuDataValid() {
    if (!use_imu) { return false; }
    if (imu_status_buffer_.size() < min_valid_count_) { return false; }
    int valid_count = std::count(imu_status_buffer_.begin(), imu_status_buffer_.end(), true);
    return valid_count >= min_valid_count_;
  }

  /**
   * @brief Check if the sensors required by the active localization path are valid
   * @return true: fresh LiDAR and (when enabled) IMU data are valid
   */
  bool isSensorDataValid() {
    return sensorDataValid(use_imu, isLidarDataValid(), isImuDataValid());
  }

  /**
   * @brief Update lidar data status
   * @param is_valid whether data is valid
   */
  void updateLidarStatus(bool is_valid) {
    lidar_status_buffer_.push_back(is_valid);
    if (lidar_status_buffer_.size() > buffer_size_) {
      lidar_status_buffer_.pop_front();
    }
    if (is_valid) {
      last_lidar_data_time_ = get_clock()->now();
    }
  }

  /**
   * @brief Update IMU data status
   * @param is_valid whether data is valid
   */
  void updateImuStatus(bool is_valid) {
    imu_status_buffer_.push_back(is_valid);
    if (imu_status_buffer_.size() > buffer_size_) {
      imu_status_buffer_.pop_front();
    }
    if (is_valid) {
      last_imu_data_time_ = get_clock()->now();
    }
  }
  
  // Transform points with the fixed LiDAR-to-base installation extrinsic.
  void TransformPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out) {
      cloud_out->clear();
      int point_size = cloud_in->points.size();
      cloud_out->resize(point_size);
  #pragma omp parallel for
      for (int i = 0; i < cloud_in->points.size(); ++i) {
          pcl::PointXYZI  point;
          Eigen::Vector4f p_start(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z, 1.0);
          Eigen::Vector4f p_result(lidar_to_base_transform_ * p_start);
          point.x              = p_result(0);
          point.y              = p_result(1);
          point.z              = p_result(2);
          point.intensity      = cloud_in->points[i].intensity;
          cloud_out->points[i] = point;
      }
  }

  /**
   * @brief Update pose history for extrapolation
   * @param pose current pose matrix
   * @param velocity current velocity
   * @param angular_velocity current angular velocity
   * @param current_time current time
   */
  void updatePoseHistory(const Eigen::Matrix4f& pose, const Eigen::Vector3f& velocity, 
                         const Eigen::Vector3f& angular_velocity, const rclcpp::Time& current_time) {
    last_pose_ = pose;
    last_velocity_ = velocity;
    last_angular_velocity_ = angular_velocity;
    last_valid_pose_time_ = current_time;
    has_valid_pose_history_ = true;
  }

  /**
   * @brief Get current velocity (prioritize UKF state, extrapolation as backup)
   * @param current_pose current pose
   * @param current_time current time
   * @return current velocity
   */
  Eigen::Vector3f getCurrentVelocity(const Eigen::Matrix4f& current_pose, const rclcpp::Time& current_time) {
    if (pose_estimator) {
      return pose_estimator->vel();
    }
    if (has_valid_pose_history_) {
      double dt = (current_time - last_valid_pose_time_).seconds();
      if (dt > 0.0) {
        Eigen::Vector3f current_position = current_pose.block<3, 1>(0, 3);
        Eigen::Vector3f last_position = last_pose_.block<3, 1>(0, 3);
        Eigen::Vector3f position_delta = current_position - last_position;
        return position_delta / dt;
      }
    }
    return last_velocity_;
  }

  /**
   * @brief Get current angular velocity (prioritize IMU data, extrapolation as backup)
   * @param current_pose current pose
   * @param current_time current time
   * @return current angular velocity
   */
  Eigen::Vector3f getCurrentAngularVelocity(const Eigen::Matrix4f& current_pose, const rclcpp::Time& current_time) {
    if (latest_angular_velocity_.norm() > 0.0) {
      return latest_angular_velocity_;
    }
    if (has_valid_pose_history_) {
      double dt = (current_time - last_valid_pose_time_).seconds();
      if (dt > 0.0) {
        Eigen::Matrix3f current_rotation = current_pose.block<3, 3>(0, 0);
        Eigen::Matrix3f last_rotation = last_pose_.block<3, 3>(0, 0);
        Eigen::Matrix3f relative_rotation = current_rotation * last_rotation.transpose();
        Eigen::AngleAxisf angle_axis(relative_rotation);
        Eigen::Vector3f angular_velocity = angle_axis.axis() * angle_axis.angle() / dt;
        return angular_velocity;
      }
    }
    return last_angular_velocity_;
  }

  /**
   * @brief Get sensor status statistics information
   * @return string containing sensor status statistics information
   */
  std::string getSensorStatusInfo() const {
    std::stringstream ss;
    // Lidar status statistics
    int lidar_valid_count = std::count(lidar_status_buffer_.begin(), lidar_status_buffer_.end(), true);
    int lidar_total_count = lidar_status_buffer_.size();
    double lidar_valid_ratio = lidar_total_count > 0 ? (double)lidar_valid_count / lidar_total_count : 0.0; 
    // IMU status statistics
    int imu_valid_count = std::count(imu_status_buffer_.begin(), imu_status_buffer_.end(), true);
    int imu_total_count = imu_status_buffer_.size();
    double imu_valid_ratio = imu_total_count > 0 ? (double)imu_valid_count / imu_total_count : 0.0;
    
    ss << "Lidar: " << lidar_valid_count << "/" << lidar_total_count 
       << " (" << std::fixed << std::setprecision(1) << (lidar_valid_ratio * 100.0) << "%)"
       << " | IMU: " << imu_valid_count << "/" << imu_total_count 
       << " (" << std::fixed << std::setprecision(1) << (imu_valid_ratio * 100.0) << "%)"
       << " | Pose History: " << (has_valid_pose_history_ ? "Available" : "Not Available")
       << " | Confidence: " << std::fixed << std::setprecision(2) << current_confidence_; 
    return ss.str();
  }

  /**
   * @brief Update confidence
   * @param current_time current time
   */
  void updateConfidence(const rclcpp::Time& current_time) {
    if (!is_init_success_) {
      current_confidence_ = 0.0;
    } else if (is_extrapolating_) {
      // When extrapolating, confidence decays exponentially
      double dt = (current_time - last_confidence_update_time_).seconds();
      if (dt > 0.0) {
        current_confidence_ *= std::exp(-confidence_decay_rate_ * dt);
        if (current_confidence_ < 0.01) {
          current_confidence_ = 0.01;
        }
      }
    } else {
      current_confidence_ = 1.0;
    } 
    last_confidence_update_time_ = current_time;
  }

  /**
   * @brief Get current confidence
   * @return current confidence value
   */
  double getCurrentConfidence() const { return current_confidence_; }

  /**
   * @brief Control log printing frequency
   * @param message message to print
   * @param force_print whether to force print (ignore counter)
   */
  void controlledLogInfo(const std::string& message, bool force_print = false) {
    log_counter_++;
    if (force_print || log_counter_ % log_interval_ == 0) {
      RCLCPP_INFO(get_logger(), "%s", message.c_str());
      log_counter_ = 0;  // Reset counter
    }
  }

  /**
   * @brief Convert quaternion to normalized Euler angles (ZYX order, consistent with Eigen)
   * @param quat quaternion (w, x, y, z)
   * @return normalized Euler angles [roll, pitch, yaw] (ZYX order)
   */
  Eigen::Vector3f quaternionToNormalizedRPY(const Eigen::Quaternionf& quat) {
    Eigen::Quaternionf normalized_quat = quat.normalized();
    float w = normalized_quat.w();
    float x = normalized_quat.x();
    float y = normalized_quat.y();
    float z = normalized_quat.z();
    // 0=Z-axis(yaw), 1=Y-axis(pitch), 2=X-axis(roll)
    float roll, pitch, yaw;
    // Calculate Roll 
    float sinr_cosp = 2.0f * (w * x + y * z);
    float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
    roll = std::atan2(sinr_cosp, cosr_cosp);
    // Calculate Pitch 
    float sinp = 2.0f * (w * y - x * z);
    if (std::abs(sinp) >= 1.0f) {
      // Handle gimbal lock case (pitch = ±90°)
      pitch = std::copysign(M_PI / 2.0f, sinp);
      roll = 0.0f;
      yaw = 2.0f * std::atan2(x, w);
    } else {
      pitch = std::asin(sinp);
      // Calculate Yaw 
      float siny_cosp = 2.0f * (w * z + x * y);
      float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
      yaw = std::atan2(siny_cosp, cosy_cosp);
    }

    if (roll > M_PI / 2.0f) {
      roll -= M_PI;
      pitch = M_PI - pitch;
      yaw += M_PI;
    } else if (roll < -M_PI / 2.0f) {
      roll += M_PI;
      pitch = -M_PI - pitch;
      yaw += M_PI;
    }
    if (pitch > M_PI / 2.0f) {
      pitch = M_PI - pitch;
      roll += M_PI;
      yaw += M_PI;
    } else if (pitch < -M_PI / 2.0f) {
      pitch = -M_PI - pitch;
      roll += M_PI;
      yaw += M_PI;
    }
    if (yaw > M_PI) {
      yaw -= 2.0f * M_PI;
    } else if (yaw < -M_PI) {
      yaw += 2.0f * M_PI;
    }
    return Eigen::Vector3f(roll, pitch, yaw);
  }

  /**
   * @brief Extrapolate pose based on previous state
   * @param current_time current time
   * @return extrapolated pose matrix
   */
  Eigen::Matrix4f extrapolatePose(const rclcpp::Time& current_time) {
    if (!has_valid_pose_history_) {
      return Eigen::Matrix4f::Identity();
    }
    // Use fixed time step 0.05 seconds (20Hz)
    const double dt = 0.05;
    const double max_velocity = 1.0;        // Maximum velocity
    const double max_angular_velocity = 0.5; // Maximum angular velocity
    Eigen::Vector3f limited_velocity = last_velocity_;
    double velocity_norm = limited_velocity.norm();
    if (velocity_norm > max_velocity) {
      limited_velocity = limited_velocity.normalized() * max_velocity;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1.0, 
                           "Velocity limited from %.2f to %.2f m/s", velocity_norm, max_velocity);
    }
    
    // Limit angular velocity range
    Eigen::Vector3f limited_angular_velocity = last_angular_velocity_;
    double angular_velocity_norm = limited_angular_velocity.norm();
    if (angular_velocity_norm > max_angular_velocity) {
      limited_angular_velocity = limited_angular_velocity.normalized() * max_angular_velocity;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1.0, 
                           "Angular velocity limited from %.2f to %.2f rad/s", angular_velocity_norm, max_angular_velocity);
    }
    Eigen::Vector3f position_delta = limited_velocity * dt;
    Eigen::Vector3f last_position = last_pose_.block<3, 1>(0, 3);
    Eigen::Matrix3f last_rotation = last_pose_.block<3, 3>(0, 0);
    Eigen::Vector3f new_position = last_position + position_delta;
    Eigen::Vector3f angle_delta = limited_angular_velocity * dt;
    Eigen::Matrix3f delta_rotation = Eigen::Matrix3f::Identity();
    delta_rotation = Eigen::AngleAxisf(angle_delta.z(), Eigen::Vector3f::UnitZ()) *
                     Eigen::AngleAxisf(angle_delta.y(), Eigen::Vector3f::UnitY()) *
                     Eigen::AngleAxisf(angle_delta.x(), Eigen::Vector3f::UnitX());
    Eigen::Matrix3f new_rotation = last_rotation * delta_rotation;
    Eigen::Matrix4f extrapolated_pose = Eigen::Matrix4f::Identity();
    extrapolated_pose.block<3, 3>(0, 0) = new_rotation;
    extrapolated_pose.block<3, 1>(0, 3) = new_position;
    
    // Debug information
    RCLCPP_DEBUG(get_logger(), 
                 "Extrapolation: dt=%.3fs, vel=[%.2f,%.2f,%.2f], pos_delta=[%.2f,%.2f,%.2f], "
                 "new_pos=[%.2f,%.2f,%.2f]", 
                 dt, 
                 limited_velocity.x(), limited_velocity.y(), limited_velocity.z(),
                 position_delta.x(), position_delta.y(), position_delta.z(),
                 new_position.x(), new_position.y(), new_position.z());
    
    return extrapolated_pose;
  }

  /**
   * @brief Publish extrapolated localization information
   * @param stamp timestamp
   */
  void publishExtrapolatedOdom(const rclcpp::Time& stamp) {
    if (!has_valid_pose_history_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1.0, 
                           "No pose history available, publishing default pose");
      pubDefaultLocalizationOdom(stamp);
      return;
    }
    // Set extrapolation state to true, start confidence decay
    is_extrapolating_ = true;
    Eigen::Matrix4f extrapolated_pose = extrapolatePose(stamp);
    publish_odometry(stamp, extrapolated_pose);
    last_pose_ = extrapolated_pose;
    last_valid_pose_time_ = stamp; 
    double extrapolation_time = (stamp - last_valid_pose_time_).seconds();  
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1.0, 
                         "Published extrapolated pose and updated history (extrapolation time: %.2fs, "
                         "last velocity: [%.2f, %.2f, %.2f], "
                         "last angular velocity: [%.2f, %.2f, %.2f], "
                         "confidence: %.3f)", 
                         extrapolation_time,
                         last_velocity_.x(), last_velocity_.y(), last_velocity_.z(),
                         last_angular_velocity_.x(), last_angular_velocity_.y(), last_angular_velocity_.z(),
                         getCurrentConfidence());
  }

  void publish_scan_matching_status(const std_msgs::msg::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
    anubis_localization::msg::ScanMatchingStatus status;
    status.header = header;
    status.has_converged = registration->hasConverged();
    status.matching_error = registration->getFitnessScore();
    const double max_correspondence_dist = 0.5;

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for (int i = 0; i < aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if (k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();
    status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;
    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);
    std::vector<double> errors(6, 0.0);
    if (pose_estimator->wo_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::msg::String());
      status.prediction_labels.back().data = "without_pred";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
    }
    if (pose_estimator->imu_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::msg::String());
      status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
    }
    if (pose_estimator->odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::msg::String());
      status.prediction_labels.back().data = "odom";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
    }
    status_pub->publish(status);
  }

  void PublishLidarLocalizationInfo() {
        auto current_time = this->get_clock()->now();
        updateConfidence(current_time);
        if (lidar_status_buffer_.size() > 0) {
            double lidar_time_diff = (current_time - last_lidar_data_time_).seconds();
            if (lidar_time_diff > sensor_timeout_threshold_) {
                updateLidarStatus(false);
            }
        }  
        if (imu_status_buffer_.size() > 0) {
            double imu_time_diff = (current_time - last_imu_data_time_).seconds();
            if (imu_time_diff > sensor_timeout_threshold_) {
                updateImuStatus(false);
            }
        }
        publishGlobalLocalizationDiagnostics();
        anubis_interfaces::msg::Localization msg;
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "map";
        msg.type = "loc_state";

        if (!is_init_success_) {
            msg.status = 0;
            msg.coord_type = is_use_map_coord_ ? 0 : 1;
            msg.pos.x = msg.pos.y = msg.pos.z = 0.0;
            msg.rpy.x = msg.rpy.y = msg.rpy.z = 0.0;
            msg.vel.x = msg.vel.y = msg.vel.z = 0.0;
            msg.acc.x = msg.acc.y = msg.acc.z = 0.0;
            msg.gyro.x = msg.gyro.y = msg.gyro.z = 0.0;
            msg.speed = 0.0;
            current_confidence_ = 0.0;
            localization_info_pub_->publish(msg);
            controlledLogInfo("Sensor Status: " + getSensorStatusInfo());
            return;
        }
        if (!isSensorDataValid()) {
            // Sensor data invalid, use extrapolated pose, set status to 4 (localization failed)
            msg.status = 4;
            msg.coord_type = is_use_map_coord_ ? 0 : 1;          
            if (has_valid_pose_history_) {
                Eigen::Matrix4f extrapolated_pose = extrapolatePose(current_time);
                Eigen::Vector3f position = extrapolated_pose.block<3, 1>(0, 3);
                Eigen::Matrix3f rotation = extrapolated_pose.block<3, 3>(0, 0);        
                if (msg.coord_type == 0) {
                    msg.pos.x = position.x();
                    msg.pos.y = position.y();
                    msg.pos.z = position.z();
                }
                Eigen::Quaternionf quat(rotation);
                Eigen::Vector3f rpy = quaternionToNormalizedRPY(quat);
                msg.rpy.x = rpy(0); // roll
                msg.rpy.y = rpy(1); // pitch
                msg.rpy.z = rpy(2); // yaw
            } else {
                msg.pos.x = msg.pos.y = msg.pos.z = 0.0;
                msg.rpy.y = msg.rpy.z = 0.0;
            }  
            msg.vel.x = msg.vel.y = msg.vel.z = 0.0;
            msg.acc.x = msg.acc.y = msg.acc.z = 0.0;
            msg.gyro.x = msg.gyro.y = msg.gyro.z = 0.0;
            msg.speed = 0.0;
            
            localization_info_pub_->publish(msg);
            controlledLogInfo("Sensor Status: " + getSensorStatusInfo() + " | Publishing extrapolated pose with status 4");
            return;
        }
        if (localization_state_ == 0) {
            msg.status = 0;
        } else if (localization_state_ == 1) {
            msg.status = 1;
        } else if (localization_state_ == 2) {
            msg.status = 2;
        } else if (localization_state_ == 4) {
            msg.status = 4;
        } else {
            msg.status = 3;
        }
        msg.coord_type = is_use_map_coord_ ? 0 : 1; // 0 map coordinate, 1 latitude and longitude coordinate

        Eigen::VectorXf state = pose_estimator->GetCurrentUkfState();
        if (msg.coord_type == 0) {
            msg.pos.x = state(0); // pos.x
            msg.pos.y = state(1); // pos.y
            msg.pos.z = state(2); // pos.z
        }
        {
            Eigen::Matrix3f rotation = pose_estimator->matrix().block<3, 3>(0, 0);
            Eigen::Quaternionf quat(rotation);
            Eigen::Vector3f rpy = quaternionToNormalizedRPY(quat);
            msg.rpy.x = rpy(0); // roll
            msg.rpy.y = rpy(1); // pitch
            msg.rpy.z = rpy(2); // yaw
        }
        if (correct_imu_data_ptr_) {
            const auto &imu = *correct_imu_data_ptr_;
            msg.acc.x = imu.linear_acceleration.x;
            msg.acc.y = imu.linear_acceleration.y;
            msg.acc.z = imu.linear_acceleration.z;

            msg.gyro.x = imu.angular_velocity.x;
            msg.gyro.y = imu.angular_velocity.y;
            msg.gyro.z = imu.angular_velocity.z;
        } else {
            msg.acc.x = msg.acc.y = msg.acc.z = 0.0;
            msg.gyro.x = msg.gyro.y = msg.gyro.z = 0.0;
        }
        const Eigen::Vector3f vel = pose_estimator->vel();
        msg.vel.x = vel(0);
        msg.vel.y = vel(1);
        msg.vel.z = vel(2);
        msg.speed = vel.norm();
        localization_info_pub_->publish(msg);
        controlledLogInfo("Sensor Status: " + getSensorStatusInfo());
    }

    /**
     * @brief Odometry publishing timer callback function
     * Only provides supplementary odometry publishing when sensors fail or not initialized, does not interfere with original logic
     */
    void PublishOdomTimer() {
        if (!is_init_success_) {
            RCLCPP_DEBUG(get_logger(), "Localization not initialized, publishing default pose");
            pubDefaultLocalizationOdom(this->get_clock()->now());
            return;
        }  

        if (!isSensorDataValid()) {
            RCLCPP_DEBUG(get_logger(), "Sensor data invalid, checking pose history...");
            sensor_invalid_latched_ = true;
            publishGlobalLocalizationDiagnostics();
            auto current_time = this->get_clock()->now();
            if (has_valid_pose_history_) {
                double time_since_last_pose = (current_time - last_valid_pose_time_).seconds();
                RCLCPP_DEBUG(get_logger(), "Time since last pose: %.2fs, using extrapolation", time_since_last_pose);
                
                publishExtrapolatedOdom(current_time);
                return;
            } else {
                RCLCPP_DEBUG(get_logger(), "No valid pose history available, using default pose");
                pubDefaultLocalizationOdom(current_time);
                return;
            }
        }
        if (is_extrapolating_) {
            is_extrapolating_ = false;
            current_confidence_ = 1.0;
            RCLCPP_DEBUG(get_logger(), "Sensor data recovered, resetting extrapolation state, confidence: 1.0");
        }
        RCLCPP_DEBUG(get_logger(), "Sensor data valid, no supplementary odometry needed");
    }

    void LocalizationStateCallback(const std::shared_ptr<anubis_interfaces::srv::LocalizationState::Request> request,
            std::shared_ptr<anubis_interfaces::srv::LocalizationState::Response> response) {
         uint8_t receive_message = request->data;
        switch (receive_message) {
        case 0: 
            mode_state_.store(ModeState::INIT);
            response->success = true;
            response->message = "Initialized (Mode: INIT)";
            break;
        case 2:
            mode_state_.store(ModeState::READY);
            response->success = true;
            response->message = "Set READY State (will auto switch to ACTIVE when ready)";
            break;
        case 4: 
            mode_state_.store(ModeState::SUCCESS);
            response->success = true;
            response->message = "Localization stopped (Mode: SUCCESS)";
            break;
        default:
            response->success = false;
            response->message = "Invalid State";
            break;
        }
    }

    bool loadGlobalMapFromFile(const std::string& map_path, std::string& message) {
        if (map_path.empty()) {
            message = "PCD map path is empty";
            return false;
        }
        std::error_code filesystem_error;
        if (!std::filesystem::exists(std::filesystem::path(map_path), filesystem_error) ||
            filesystem_error) {
            message = "Map file does not exist or cannot be accessed: " + map_path;
            return false;
        }
        MapArtifactManifest artifact_manifest;
        std::string manifest_error;
        if (!validateMapArtifactManifest(
                std::filesystem::path(map_path), artifact_manifest,
                &manifest_error)) {
            message = "Map artifact manifest validation failed: " + manifest_error;
            return false;
        }
        RCLCPP_INFO(
            get_logger(), "Validated map artifact manifest generation=%s artifacts=%zu root=%s",
            artifact_manifest.generation.c_str(), artifact_manifest.artifacts.size(),
            artifact_manifest.artifact_root.string().c_str());
        if (!registration) {
            message = "Registration backend is not initialized";
            return false;
        }

        pcl::PointCloud<PointT>::Ptr loaded_map(new pcl::PointCloud<PointT>());
        const int load_result = pcl::io::loadPCDFile<PointT>(map_path, *loaded_map);
        if (load_result != 0) {
            message = "Failed to read PCD map: " + map_path;
            return false;
        }
        RCLCPP_INFO(get_logger(), "Loaded %zu raw map points from %s",
                    loaded_map->size(), map_path.c_str());

        const size_t before = loaded_map->size();
        loaded_map->is_dense = false;
        std::vector<int> keep_indices;
        pcl::removeNaNFromPointCloud(*loaded_map, *loaded_map, keep_indices);
        const size_t removed = before - loaded_map->size();
        if (removed > 0) {
            RCLCPP_WARN(get_logger(),
                        "Removed %zu NaN/Inf points from loaded map (total=%zu)",
                        removed, before);
        }
        if (loaded_map->empty()) {
            message = "Loaded PCD map contains no finite points: " + map_path;
            return false;
        }

        pcl::PointCloud<PointT>::Ptr filtered_map(new pcl::PointCloud<PointT>());
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(loaded_map);
        voxel.setLeafSize(globalmap_voxel_size_, globalmap_voxel_size_,
                          globalmap_voxel_size_);
        voxel.filter(*filtered_map);
        if (filtered_map->size() < 1000U) {
            message = "Global map is too small after filtering (" +
                std::to_string(filtered_map->size()) + " points): " + map_path;
            return false;
        }

        // Preflight the descriptor artifact before changing any active map
        // pointers.  The manifest binds an existing database to this exact
        // generation; a malformed database may only be bypassed when the
        // configured grid fallback explicitly permits degraded retrieval.
        std::shared_ptr<scan_descriptor::DescriptorDatabase>
            loaded_descriptor_database;
        std::filesystem::path descriptor_path;
        std::string descriptor_error;
        if (global_localization_ptr_) {
            const std::filesystem::path map_fs_path(map_path);
            descriptor_path = artifact_manifest.artifact_root /
                (map_fs_path.stem().string() + "_scd.bin");
            const std::filesystem::path descriptor_relative =
                descriptor_path.filename();
            std::error_code descriptor_status_error;
            const auto descriptor_status = std::filesystem::symlink_status(
                descriptor_path, descriptor_status_error);
            const bool descriptor_missing = descriptor_status_error ==
                std::make_error_code(std::errc::no_such_file_or_directory);
            if (descriptor_status_error && !descriptor_missing) {
                message = "Descriptor DB cannot be inspected: " +
                    descriptor_path.string() + ": " +
                    descriptor_status_error.message();
                return false;
            }
            if (!descriptor_missing) {
                if (std::filesystem::is_symlink(descriptor_status) ||
                    !std::filesystem::is_regular_file(descriptor_status) ||
                    !artifact_manifest.contains(descriptor_relative)) {
                    message = "Descriptor DB is not a manifest-bound regular artifact: " +
                        descriptor_path.string();
                    return false;
                }
                auto database =
                    std::make_shared<scan_descriptor::DescriptorDatabase>();
                if (database->load(descriptor_path.string(), &descriptor_error)) {
                    loaded_descriptor_database = std::move(database);
                } else if (gl_params_.gl_recall_source == "retrieval" &&
                           !gl_params_.retrieval_fallback_to_grid) {
                    message = "Manifest-bound descriptor DB failed to load: " +
                        descriptor_error;
                    return false;
                }
            } else if (gl_params_.gl_recall_source == "retrieval" &&
                       !gl_params_.retrieval_fallback_to_grid) {
                message = "Manifest-bound descriptor DB is missing: " +
                    descriptor_path.string();
                return false;
            }
        }

        // Only replace the active target after the new PCD has passed every
        // check, so a failed reload cannot destroy a working localization map.
        // [2026-08-17] global_map_points_ptr_ 保持**未过滤**——GL 的描述子库是
        // 用它建的，过滤会让检索失配。NDT 单独用一份截掉天花板的副本。
        global_map_points_ptr_ = filtered_map;
        ndt_map_points_ptr_ = filterByHeight(filtered_map, ndt_max_point_height_);
        registration->setInputTarget(ndt_map_points_ptr_);
        if (reg_method == "NDT_OMP" && !ndt_map_points_ptr_->empty()) {
            pclomp::VoxelGridCovariance<PointT> ndt_grid;
            const float resolution = static_cast<float>(ndt_resolution);
            ndt_grid.setLeafSize(resolution, resolution, resolution);
            ndt_grid.setInputCloud(ndt_map_points_ptr_);
            ndt_grid.filter(false);
            const int min_points = ndt_grid.getMinPointPerVoxel();
            const auto& leaves = ndt_grid.getLeaves();
            size_t valid_cells = 0;
            size_t points_in_valid_cells = 0;
            for (const auto& leaf_entry : leaves) {
                const int point_count = leaf_entry.second.getPointCount();
                if (point_count >= min_points) {
                    ++valid_cells;
                    points_in_valid_cells += static_cast<size_t>(point_count);
                }
            }
            const double valid_cell_percent = leaves.empty() ? 0.0 :
                100.0 * static_cast<double>(valid_cells) /
                static_cast<double>(leaves.size());
            const double represented_point_percent = ndt_map_points_ptr_->empty() ? 0.0 :
                100.0 * static_cast<double>(points_in_valid_cells) /
                static_cast<double>(ndt_map_points_ptr_->size());
            RCLCPP_INFO(get_logger(),
                "[NDT MAP] input_points=%zu prefilter_leaf=%.3fm ndt_resolution=%.3fm "
                "occupied_cells=%zu valid_cells=%zu min_points_per_cell=%d "
                "valid_cell_pct=%.2f points_in_valid_cells=%zu represented_point_pct=%.2f",
                ndt_map_points_ptr_->size(), globalmap_voxel_size_, ndt_resolution,
                leaves.size(), valid_cells, min_points, valid_cell_percent,
                points_in_valid_cells, represented_point_percent);
        }
        RCLCPP_INFO(get_logger(),
            "NDT target height-filtered: %zu -> %zu points (z_max=%.2fm, dropped %.1f%%); "
            "GL keeps the unfiltered %zu-point map",
            filtered_map->size(), ndt_map_points_ptr_->size(),
            ndt_max_point_height_,
            filtered_map->empty() ? 0.0 :
              100.0 * (1.0 - static_cast<double>(ndt_map_points_ptr_->size()) /
                             static_cast<double>(filtered_map->size())),
            global_map_points_ptr_->size());
        gl_map_id_ = map_path;
        descriptor_database_ = std::move(loaded_descriptor_database);
        if (global_localization_ptr_) {
            if (descriptor_database_) {
                global_localization_ptr_->setDescriptorDatabase(descriptor_database_);
                RCLCPP_INFO(get_logger(), "Descriptor DB loaded (%zu keyframes): %s",
                            descriptor_database_->size(), descriptor_path.string().c_str());
            } else {
                global_localization_ptr_->setDescriptorDatabase(nullptr);
                if (gl_params_.gl_recall_source == "retrieval") {
                    RCLCPP_WARN(get_logger(),
                                "Manifest-bound descriptor DB unavailable (%s); retrieval disabled and grid fallback=%s",
                                descriptor_error.empty() ? "artifact is absent" :
                                    descriptor_error.c_str(),
                                gl_params_.retrieval_fallback_to_grid ? "enabled" : "disabled");
                } else {
                    RCLCPP_DEBUG(get_logger(), "Descriptor DB not loaded while grid recall is active: %s",
                                 descriptor_error.empty() ? "artifact is absent" :
                                     descriptor_error.c_str());
                }
            }
        }
        Reset();

        if (global_map_pub_->get_subscription_count()) {
            pcl::PointCloud<PointT>::Ptr publish_map(new pcl::PointCloud<PointT>());
            pcl::VoxelGrid<PointT> publish_voxel;
            publish_voxel.setInputCloud(global_map_points_ptr_);
            publish_voxel.setLeafSize(0.5f, 0.5f, 0.5f);
            publish_voxel.filter(*publish_map);
            sensor_msgs::msg::PointCloud2 global_map_msg;
            pcl::toROSMsg(*publish_map, global_map_msg);
            global_map_msg.header.frame_id = "map";
            global_map_msg.header.stamp = get_clock()->now();
            global_map_pub_->publish(global_map_msg);
            RCLCPP_INFO(get_logger(), "Published downsampled global map");
        }

        message = "Map loaded successfully: " + map_path + " (" +
            std::to_string(global_map_points_ptr_->size()) + " filtered points)";
        return true;
    }

    void LoadMapCallBack(anubis_interfaces::srv::LoadMap::Request::SharedPtr request,
            anubis_interfaces::srv::LoadMap::Response::SharedPtr response) {
        std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
        update_map_flag_.store(true);
        try {
            response->success = loadGlobalMapFromFile(request->pcd_path, response->message);
        } catch (const std::exception& exception) {
            response->success = false;
            response->message = std::string("Unexpected map load failure: ") + exception.what();
        }
        update_map_flag_.store(false);

        if (response->success) {
            RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
            if (apriltag_enable_) {
                const std::filesystem::path pcd_path(request->pcd_path);
                loadAprilTagReferenceFile(pcd_path.parent_path().string());
                apriltag_pending_votes_.clear();
            }
        } else {
            RCLCPP_ERROR(get_logger(), "%s; previous map remains active",
                         response->message.c_str());
        }
    }

    // Parses the apriltag_anchors.yaml this same stack writes at
    // mapping save-time (see saveAprilTagMap in anubis_mapping's
    // mapping_alg.cpp). Deliberately a small hand-rolled parser tied to
    // that exact, fixed output format rather than a general YAML library
    // dependency this package doesn't otherwise have. [PATCH -- ArUco ->
    // AprilTag migration] Schema is unchanged from the old aruco.yaml on
    // purpose (id/position/orientation/validated) -- only the filename
    // and naming changed, so this regex parsing logic did not need to
    // change at all.
    bool loadAprilTagReferenceFile(const std::string& map_dir) {
        apriltag_reference_markers_.clear();
        const std::string path = map_dir + "/apriltag_anchors.yaml";
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            RCLCPP_INFO(
                get_logger(),
                "No apriltag_anchors.yaml at %s -- AprilTag anchor correction has no reference markers until one is loaded",
                path.c_str());
            return false;
        }

        static const std::regex id_re(R"(-\s*id:\s*(-?\d+))");
        static const std::regex pos_re(
            R"(position:\s*\{x:\s*([-\d.eE+]+),\s*y:\s*([-\d.eE+]+),\s*z:\s*([-\d.eE+]+)\})");
        static const std::regex ori_re(
            R"(orientation:\s*\{x:\s*([-\d.eE+]+),\s*y:\s*([-\d.eE+]+),\s*z:\s*([-\d.eE+]+),\s*w:\s*([-\d.eE+]+)\})");
        static const std::regex validated_re(R"(validated:\s*(true|false))");

        int current_id = std::numeric_limits<int>::min();
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
        bool has_pos = false;
        bool has_ori = false;
        bool validated = false;

        auto flush = [&]() {
            if (current_id != std::numeric_limits<int>::min() && has_pos && has_ori) {
                AprilTagMarkerRef ref;
                ref.T_map_marker.setIdentity();
                ref.T_map_marker.block<3, 3>(0, 0) = quat.normalized().toRotationMatrix();
                ref.T_map_marker.block<3, 1>(0, 3) = pos;
                ref.validated = validated;
                apriltag_reference_markers_[current_id] = ref;
            }
        };

        std::string line;
        while (std::getline(ifs, line)) {
            std::smatch m;
            if (std::regex_search(line, m, id_re)) {
                flush();
                current_id = std::stoi(m[1].str());
                has_pos = false;
                has_ori = false;
                validated = false;
                continue;
            }
            if (std::regex_search(line, m, pos_re)) {
                pos = Eigen::Vector3d(std::stod(m[1]), std::stod(m[2]), std::stod(m[3]));
                has_pos = true;
                continue;
            }
            if (std::regex_search(line, m, ori_re)) {
                // std::stod(m[4]) is w -- Eigen::Quaterniond ctor order is (w, x, y, z).
                quat = Eigen::Quaterniond(
                    std::stod(m[4]), std::stod(m[1]), std::stod(m[2]), std::stod(m[3]));
                has_ori = true;
                continue;
            }
            if (std::regex_search(line, m, validated_re)) {
                validated = (m[1].str() == "true");
                continue;
            }
        }
        flush();

        std::size_t n_validated = 0;
        for (const auto& kv : apriltag_reference_markers_) {
            if (kv.second.validated) {
                ++n_validated;
            }
        }
        RCLCPP_INFO(
            get_logger(), "Loaded apriltag_anchors.yaml: %zu marker id(s), %zu validated (%s)",
            apriltag_reference_markers_.size(), n_validated, path.c_str());
        return true;
    }

    void aprilTagDetectionCallBack(const apriltag_msgs::msg::AprilTagDetectionArray::UniquePtr msg) {
        if (msg->detections.empty() || apriltag_reference_markers_.empty()) {
            return;
        }
        const rclcpp::Time scan_stamp(msg->header.stamp);

        for (const auto& det : msg->detections) {
            if (det.family != apriltag_family_) {
                continue;
            }
            if (det.hamming > apriltag_max_hamming_) {
                continue;
            }
            if (det.decision_margin < apriltag_min_decision_margin_) {
                continue;
            }
            const auto ref_it = apriltag_reference_markers_.find(det.id);
            if (ref_it == apriltag_reference_markers_.end() || !ref_it->second.validated) {
                continue;
            }

            // [PATCH -- AprilTag migration] apriltag_ros publishes the tag
            // pose on /tf (child_frame_id "tag<family>:<id>"), not on this
            // message -- and because anubis_description publishes the
            // full base_link->...->camera_color_optical_frame chain, tf2
            // composes T_base_marker directly in one lookup. This
            // replaces BOTH the old T_camera_marker (read from the ArUco
            // message) AND the separately-maintained
            // aruco_base_to_camera_extrinsic_ matrix -- there is nothing
            // left to keep in sync with the URDF by hand.
            const std::string tag_frame = "tag" + apriltag_family_ + ":" + std::to_string(det.id);
            geometry_msgs::msg::TransformStamped transform;
            try {
                transform = tf_buffer->lookupTransform(
                    apriltag_base_frame_id_, tag_frame, msg->header.stamp,
                    rclcpp::Duration::from_seconds(0.1));
            } catch (const tf2::TransformException& ex) {
                RCLCPP_DEBUG(get_logger(), "[APRILTAG] id=%d: TF lookup %s -> %s failed: %s",
                             det.id, apriltag_base_frame_id_.c_str(), tag_frame.c_str(), ex.what());
                continue;
            }

            Eigen::Matrix4d T_base_marker = Eigen::Matrix4d::Identity();
            T_base_marker(0, 3) = transform.transform.translation.x;
            T_base_marker(1, 3) = transform.transform.translation.y;
            T_base_marker(2, 3) = transform.transform.translation.z;
            Eigen::Quaterniond q(
                transform.transform.rotation.w, transform.transform.rotation.x,
                transform.transform.rotation.y, transform.transform.rotation.z);
            if (!q.coeffs().allFinite() || q.norm() < 1e-6) {
                continue;
            }
            T_base_marker.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();

            // T_map_marker = T_map_base * T_base_marker
            // => T_map_base = T_map_marker * inv(T_base_marker)
            const Eigen::Matrix4d T_map_base_observed =
                ref_it->second.T_map_marker * T_base_marker.inverse();
            if (!T_map_base_observed.allFinite()) {
                continue;
            }

            std::lock_guard<std::mutex> lock(pose_estimator_mutex);
            if (!pose_estimator) {
                continue;
            }
            const Eigen::Vector3d observed_pos = T_map_base_observed.block<3, 1>(0, 3);
            const Eigen::Vector3f current_pos = pose_estimator->pos();
            const double dist = (observed_pos - current_pos.cast<double>()).norm();

            if (dist <= apriltag_max_jump_m_) {
                // Small, ordinary drift correction -- apply directly, no need
                // to wait for repeated votes.
                applyAprilTagAnchorCorrection(T_map_base_observed, scan_stamp, det.id, dist);
                apriltag_pending_votes_.erase(det.id);
                continue;
            }

            // Large discrepancy: require temporal agreement before trusting
            // it, same spirit as the false-lock disambiguation approach
            // used for global localization.
            auto& vote = apriltag_pending_votes_[det.id];
            if (vote.hits == 0 ||
                (scan_stamp - vote.first_seen).seconds() > apriltag_consistency_window_s_) {
                vote.hits = 0;
                vote.position_sum.setZero();
                vote.first_seen = scan_stamp;
            }
            vote.position_sum += observed_pos;
            vote.hits += 1;
            if (vote.hits >= apriltag_consistency_min_hits_) {
                const Eigen::Vector3d voted_pos =
                    vote.position_sum / static_cast<double>(vote.hits);
                Eigen::Matrix4d voted_pose = T_map_base_observed;
                voted_pose.block<3, 1>(0, 3) = voted_pos;
                RCLCPP_WARN(
                    get_logger(),
                    "[APRILTAG] id=%d confirmed a %.2fm discrepancy after %d consistent sightings -- re-anchoring",
                    det.id, dist, vote.hits);
                applyAprilTagAnchorCorrection(voted_pose, scan_stamp, det.id, dist);
                apriltag_pending_votes_.erase(det.id);
            } else {
                RCLCPP_WARN(
                    get_logger(),
                    "[APRILTAG] id=%d shows a %.2fm discrepancy (%d/%d votes so far), waiting for confirmation",
                    det.id, dist, vote.hits, apriltag_consistency_min_hits_);
            }
        }
    }

    // Re-anchors the estimator to a trusted external pose. Mirrors
    // applyConfirmedGlobalPose's install sequence as closely as this file
    // could verify (rebuild PoseEstimator + set_trust_anchor) --
    // PoseEstimator exposes no lighter incremental absolute-pose update.
    // CAVEAT: applyConfirmedGlobalPose's real body likely continues past
    // what informed this function (tracking-health-monitor resets, warmup
    // calls, GL episode bookkeeping) -- diff this against the actual
    // applyConfirmedGlobalPose in your file and port over anything missed
    // before trusting this in an unattended run. Caller must hold
    // pose_estimator_mutex.
    void applyAprilTagAnchorCorrection(
        const Eigen::Matrix4d& T_map_base, const rclcpp::Time& scan_stamp,
        int marker_id, double correction_distance_m) {
        const Eigen::Vector3f new_position = T_map_base.block<3, 1>(0, 3).cast<float>();
        Eigen::Quaterniond q(Eigen::Matrix3d(T_map_base.block<3, 3>(0, 0)));
        if (!q.coeffs().allFinite() || q.norm() < 1e-6) {
            RCLCPP_WARN(get_logger(), "[APRILTAG] id=%d correction has non-finite orientation, aborting", marker_id);
            return;
        }
        q.normalize();
        const Eigen::Quaternionf new_orientation = q.cast<float>();
        if (!new_position.allFinite() || !new_orientation.coeffs().allFinite()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(imu_data_mutex);
            imu_data.clear();
        }

        last_init_pos_ = new_position;
        last_init_quat_ = new_orientation;
        has_set_init_pose_ = true;
        last_pose_source_ = "AprilTagAnchorCorrection";
        pose_estimator.reset(new localization::PoseEstimator(
            registration, scan_stamp, last_init_pos_, last_init_quat_,
            cool_time_duration, ndt_score_threshold_,
            !enable_vertical_velocity_prediction_));
        applyImuBiasIfReady();

        Eigen::Matrix4f anchor_map_pose = Eigen::Matrix4f::Identity();
        anchor_map_pose.block<3, 3>(0, 0) = new_orientation.toRotationMatrix();
        anchor_map_pose.block<3, 1>(0, 3) = new_position;
        Eigen::Matrix4f anchor_odom_pose;
        if (lookupRobotOdomPose(scan_stamp, anchor_odom_pose)) {
            pose_estimator->set_trust_anchor(anchor_map_pose, anchor_odom_pose);
        } else {
            pose_estimator->clear_trust_anchor();
            RCLCPP_WARN(
                get_logger(),
                "[APRILTAG] id=%d re-anchor installed without %s->%s transform at scan time; trust region disabled",
                marker_id, robot_odom_frame_id.c_str(), odom_child_frame_id.c_str());
        }

        RCLCPP_WARN(
            get_logger(),
            "[APRILTAG] id=%d applied re-anchor, correction_distance=%.2fm new_pos=[%.3f %.3f %.3f]",
            marker_id, correction_distance_m, new_position.x(), new_position.y(), new_position.z());

        is_init_success_ = false;
        init_match_count_ = 0;
    }

    void Reset() {
        is_init_success_ = false;
        gl_once_gate_ = true;
        resetGlobalLocalizationEpisodeState();
        localization_state_ = 0; 
        init_match_count_ = 0;
        has_init_verify_pose_ = false;
        consecutive_ndt_rejections_ = 0;
        if (pose_estimator) {
          pose_estimator->reset_velocity();
        }
        {
          std::lock_guard<std::mutex> lock(imu_data_mutex);
          imu_data.clear();
        }
        lidar_status_buffer_.clear();
        imu_status_buffer_.clear();
        for (int i = 0; i < buffer_size_; i++) {
          lidar_status_buffer_.push_back(false);
          imu_status_buffer_.push_back(false);
        }
        has_valid_pose_history_ = false;
        publishGlobalLocalizationDiagnostics();
    }

    void resetGlobalLocalizationEpisodeState() {
        gl_episode_state_ = GLEpisodeState{};
        gl_summary_ = GLSummaryCounts{};
        resetStaticConfirm(gl_static_confirm_);
        gl_request_initialpose_ = false;
        gl_auto_confirm_pending_verification_ = false;
        gl_auto_confirm_verification_start_time_ =
          rclcpp::Time(int64_t{0}, get_clock()->get_clock_type());
    }

    bool localizationOutputEligible() {
        return is_init_success_ && localization_state_ != 4 &&
          tracking_health_.localizationValid() && isSensorDataValid();
    }

    void publishGlobalLocalizationDiagnostics(bool force = false) {
        if (!localization_valid_pub_ || !request_initialpose_pub_) {
          return;
        }
        const GLOutputGate output_gate = decideGLOutputGate(
          gl_episode_state_, localizationOutputEligible());
        const bool valid = output_gate.localization_valid;
        if (!force && gl_diagnostics_initialized_ &&
            valid == last_published_localization_valid_ &&
            gl_request_initialpose_ == last_published_request_initialpose_) {
          return;
        }

        std_msgs::msg::Bool valid_msg;
        valid_msg.data = valid;
        localization_valid_pub_->publish(valid_msg);
        std_msgs::msg::Bool request_msg;
        request_msg.data = gl_request_initialpose_;
        request_initialpose_pub_->publish(request_msg);
        last_published_localization_valid_ = valid;
        last_published_request_initialpose_ = gl_request_initialpose_;
        gl_diagnostics_initialized_ = true;
    }

private:
  std::string robot_odom_frame_id;
  std::string odom_child_frame_id;

  // --- AprilTag anchor correction state ---
  struct AprilTagMarkerRef {
    Eigen::Matrix4d T_map_marker = Eigen::Matrix4d::Identity();
    bool validated = false;
  };
  struct AprilTagPendingVote {
    Eigen::Vector3d position_sum = Eigen::Vector3d::Zero();
    int hits = 0;
    rclcpp::Time first_seen;
  };
  bool apriltag_enable_ = false;
  std::string apriltag_detection_topic_ = "/perception/apriltag/detections";
  // hamming = corrected bit count (0 is best); decision_margin = detector
  // confidence (higher is better). Replaces the old
  // reprojection_error_px/require_depth_validation gate -- see
  // mapping_alg.cpp's equivalent comment for why ArUco's PnP-derived
  // gate has no AprilTag analog.
  int apriltag_max_hamming_ = 0;
  double apriltag_min_decision_margin_ = 50.0;
  std::string apriltag_family_ = "36h11";
  std::string apriltag_base_frame_id_ = "base_link";
  // A correction within this distance of the current estimate is treated as
  // ordinary drift correction; anything larger requires temporal voting
  // (apriltag_consistency_min_hits_ agreeing detections within the window)
  // before being trusted, same spirit as the false-lock disambiguation
  // approach documented for global localization.
  double apriltag_max_jump_m_ = 0.5;
  int apriltag_consistency_min_hits_ = 3;
  double apriltag_consistency_window_s_ = 2.0;
  std::unordered_map<int, AprilTagMarkerRef> apriltag_reference_markers_;
  std::unordered_map<int, AprilTagPendingVote> apriltag_pending_votes_;
  rclcpp::Subscription<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr sub_apriltag_ptr_;
  std::string localization_odom_frame_id;
  bool send_tf_transforms;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;
  bool enable_vertical_velocity_prediction_ = false;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr                         imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr                 points_sub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr                       robot_odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr                 globalmap_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub;
  rclcpp::TimerBase::SharedPtr localization_lidar_info_timer_;
  rclcpp::TimerBase::SharedPtr odom_publish_timer_; 

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr               pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr         aligned_pub;
  rclcpp::Publisher<anubis_localization::msg::ScanMatchingStatus>::SharedPtr status_pub;
  rclcpp::Publisher<anubis_interfaces::msg::Localization>::SharedPtr    localization_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr         global_map_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr                   localization_valid_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr                   request_initialpose_pub_;

  std::unique_ptr<tf2_ros::Buffer>               tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener>    tf_listener;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> imu_data;
  // Stale samples are removed from imu_data before diagnostics are emitted, so
  // retain the newest received header separately for an accurate age report.
  std::atomic<int64_t> latest_imu_stamp_ns_{0};
  
  // transformation matrices
  Eigen::Matrix3f init_rotation_matrix_ = Eigen::Matrix3f::Identity();
  Eigen::Matrix4f lidar_to_base_transform_ = Eigen::Matrix4f::Identity();
  /// map 系重力方向（单位向量），config 的 map_gravity_direction，默认 (0,0,1)
  Eigen::Vector3f map_gravity_direction_ = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  /// NDT 专用的截高地图副本；global_map_points_ptr_ 保持未过滤供 GL 使用
  pcl::PointCloud<PointT>::Ptr ndt_map_points_ptr_ = nullptr;
  /// NDT 配准的点云高度上限（米，传感器高度以上）。<=0 关闭过滤。
  float ndt_max_point_height_ = 1.5f;
  size_t stage_pts_ndt_ = 0;   ///< 截高后送进 NDT 的扫描点数

  // ---- 管线滞后与分段耗时诊断（2026-08-17）--------------------------------
  // 目的：把"位置滞后"的责任分清是 Livox / IMU / NDT / 我们的处理，而不是猜。
  double lag_entry_age_ms_  = 0.0;   ///< now − header.stamp（驱动+传输+排队）
  double lag_stamp_gap_ms_  = 0.0;   ///< 相邻帧 header.stamp 之差（Livox 节奏）
  double lag_wall_gap_ms_   = 0.0;   ///< 相邻回调墙钟之差（我们的节奏）
  double lag_cum_ms_        = 0.0;   ///< 累计滞后（wall_gap−stamp_gap 的累加）
  rclcpp::Time lag_prev_stamp_;
  std::chrono::steady_clock::time_point lag_prev_wall_;
  bool   has_lag_prev_      = false;

  double stage_ms_extrinsic_ = 0.0;
  double stage_ms_deskew_   = 0.0;
  double stage_ms_voxel_    = 0.0;
  double stage_ms_ndt_      = 0.0;
  size_t stage_pts_raw_     = 0;
  size_t stage_pts_filtered_ = 0;

  size_t imu_buf_size_      = 0;     ///< IMU 缓冲区样本数
  size_t imu_usable_        = 0;     ///< 其中可用于本帧 predict 的数量（<5 则一次都不会 predict）
  double imu_lead_ms_       = 0.0;   ///< IMU 最新戳 − 雷达戳；为负说明 IMU 落后于雷达
  bool ndt_optimizer_diagnostics_enabled_ = true;
  bool ndt_detailed_diagnostics_enabled_ = false;
  bool sensor_time_diagnostics_enabled_ = true;
  int64_t diagnostics_healthy_interval_ms_ = 1000;
  std::mutex sensor_time_mutex_;
  StreamTimingState lidar_timing_;
  StreamTimingState imu_timing_;
  StreamTimingState odom_timing_;
  /// 仅用于读 NDT 迭代次数的弱引用（基类无此接口）
  std::weak_ptr<pclomp::NormalDistributionsTransform<PointT, PointT>> ndt_diag_;
  int    ndt_max_iterations_ = 15;   ///< 与 create_registration 里 setMaximumIterations 保持一致

  // 去畸变四段细分（本函数 p50=39.5ms，理论应 <1ms，原因待实测定位）
  double dsk_ms_win_    = 0.0;   ///< IMU 窗口拷贝（含 imu_data_mutex 等待）
  double dsk_ms_gyro_   = 0.0;   ///< 陀螺积分循环
  double dsk_ms_slice_  = 0.0;   ///< 10 段变换预计算
  double dsk_ms_pts_    = 0.0;   ///< 点循环
  size_t dsk_win_size_  = 0;
  size_t dsk_imu_buf_at_deskew_ = 0;
  bool   dsk_skipped_   = true;
  
  // Global localization and pose management
  std::shared_ptr<GlobalLocalization> global_localization_ptr_;
  std::shared_ptr<const scan_descriptor::DescriptorDatabase> descriptor_database_;
  // Pose caching and change detection
  Eigen::Vector3f last_init_pos_     = Eigen::Vector3f::Zero();
  Eigen::Quaternionf last_init_quat_ = Eigen::Quaternionf::Identity();
  bool has_set_init_pose_            = false;
  std::string last_pose_source_      = "none";
  // Configuration parameters for initial pose
  bool specify_init_pose_ = true;
  double init_pos_x_ = 0.0;
  double init_pos_y_ = 0.0;
  double init_pos_z_ = 0.0;
  double init_ori_w_ = 1.0;
  double init_ori_x_ = 0.0;
  double init_ori_y_ = 0.0;
  double init_ori_z_ = 0.0;
  // Global localization parameters
  bool use_global_localization_init_ = true;
  float init_pose_change_threshold_ = 0.01f;      
  float init_quat_change_threshold_ = 0.01f;      
  float global_localization_timeout_ = 10.0f;     
  // Global localization state
  bool global_localization_in_progress_ = false;
  rclcpp::Time global_localization_start_time_;
  bool gl_once_gate_ = true;
  GLParams gl_params_;
  std::unique_ptr<GLCandidateCsvWriter> gl_candidates_csv_writer_;
  bool gl_online_shadow_enabled_ = false;
  bool gl_episode_runtime_enabled_ = false;
  bool gl_runtime_parameter_conflict_ = false;
  bool gl_relocalize_on_tracking_failure_ = false;
  bool gl_odom_debug_log_ = false;
  double gl_odom_debug_log_interval_s_ = 1.0;
  uint64_t gl_episode_id_counter_ = 0;
  std::string gl_map_id_ = "runtime-map";
  // [2026-08-14] Static confirmation state (replaces the odom anchor, the
  // candidate bank, the temporal vote and the R16-R22 rotation-evidence
  // machinery that the motion-driven vote needed).
  StaticConfirmState gl_static_confirm_;
  GLEpisodeState gl_episode_state_;
  GLSummaryCounts gl_summary_;
  RawImuGravityFilter gl_raw_imu_gravity_filter_;
  std::mutex gl_gravity_history_mutex_;
  std::deque<GravityAlignmentSample> gl_gravity_history_;
  bool gl_request_initialpose_ = false;
  bool gl_auto_confirm_pending_verification_ = false;
  rclcpp::Time gl_auto_confirm_verification_start_time_{int64_t{0}, RCL_ROS_TIME};
  uint64_t gl_auto_confirm_apply_count_ = 0;
  bool gl_diagnostics_initialized_ = false;
  bool last_published_localization_valid_ = false;
  bool last_published_request_initialpose_ = false;
  // Sensor data validity tracking
  rclcpp::Time last_lidar_data_time_;
  rclcpp::Time last_imu_data_time_;
  std::deque<bool> lidar_status_buffer_;
  std::deque<bool> imu_status_buffer_;
  int min_valid_count_;      
  int buffer_size_;          
  double sensor_timeout_threshold_ = 1.0; 
  bool sensor_invalid_latched_ = false;

  // Store previous state for extrapolation
  Eigen::Vector3f last_velocity_{0.0, 0.0, 0.0};           // Previous velocity
  Eigen::Vector3f last_angular_velocity_{0.0, 0.0, 0.0};   // Previous angular velocity
  Eigen::Matrix4f last_pose_{Eigen::Matrix4f::Identity()};  // Previous pose matrix
  rclcpp::Time last_valid_pose_time_;                        // Time of last valid pose
  bool has_valid_pose_history_ = false;                      // Whether valid pose history exists
  
  // Store latest IMU data for extrapolation
  Eigen::Vector3f latest_angular_velocity_{0.0, 0.0, 0.0}; // Latest IMU angular velocity
  
  // Confidence management
  double current_confidence_ = 1.0;                    // Current confidence
  rclcpp::Time last_confidence_update_time_;           // Last confidence update time
  double confidence_decay_rate_ = 0.1;                 // Confidence decay rate (per second)
  bool is_extrapolating_ = false;                      // Whether currently extrapolating
  
  int log_counter_ = 0;                                // Log counter
  int log_interval_ = 10;                              // Log printing interval (print every 10 times)

  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Registration<PointT, PointT>::Ptr registration;
  pcl::PointCloud<PointT>::Ptr global_map_points_ptr_;
  pcl::VoxelGrid<PointT>::Ptr  voxel_filter_ptr_ = pcl::VoxelGrid<PointT>::Ptr(new pcl::VoxelGrid<PointT>());
  pcl::PointCloud<PointT>::Ptr raw_points_ptr_ = nullptr;  ///< Raw point cloud pointer.
  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<localization::PoseEstimator> pose_estimator;

  rclcpp::Service<anubis_interfaces::srv::LocalizationState>::SharedPtr localization_state_srv_;
  rclcpp::Service<anubis_interfaces::srv::LoadMap>::SharedPtr load_map_service_ptr_;
  // Parameters
  double cool_time_duration;
  std::string reg_method;
  std::string ndt_neighbor_search_method;
  double ndt_neighbor_search_radius;
  double ndt_resolution;
  std::string ndt_distance_mode;
  bool enable_robot_odometry_prediction;
  std::string robot_odom_topic_ = "/odom";
  double robot_odom_velocity_max_age_s_ = 0.25;
  int imu_data_filter_num_ = 5;  // Number of IMU data points to filter.
  bool imu_stale_fallback_enabled_ = true;
  double imu_stale_fallback_max_dt_s_ = 0.10;

  bool is_init_success_ = false;
  bool is_use_map_coord_ = true;

  // Tracking rejection grace / recovery state
  int consecutive_ndt_rejections_ = 0;
  double localization_rejection_timeout_s_ = 2.0;
  int localization_recovery_valid_frames_ = 3;
  TrackingHealthMonitor tracking_health_{2.0, 3};
  Eigen::Matrix4f last_valid_localization_pose_ = Eigen::Matrix4f::Identity();
  bool has_last_valid_localization_pose_ = false;
  geometry_msgs::msg::TransformStamped last_valid_map_to_odom_;
  bool has_last_valid_map_to_odom_ = false;
  std::mutex robot_odom_mutex_;
  Eigen::Vector3f latest_robot_odom_velocity_body_ = Eigen::Vector3f::Zero();
  rclcpp::Time latest_robot_odom_stamp_{int64_t{0}, RCL_ROS_TIME};
  bool has_robot_odom_velocity_ = false;
  rclcpp::Time last_lidar_stamp_;
  bool has_last_lidar_stamp_ = false;
  // R14 scan motion compensation state
  bool   motion_compensation_enable_ = false;
  int    motion_compensation_slices_ = 10;
  double last_scan_period_           = 0.1;   // nominal 10 Hz; updated per scan
  rclcpp::Time last_ndt_q_log_time_;
  bool has_last_ndt_q_log_ = false;
  int localization_state_ = 0;   // 0: not init, 1: initing, 2: init success, 3: continuous localization, 4: continuous localization failed
  sensor_msgs::msg::Imu::SharedPtr correct_imu_data_ptr_;
  std::atomic<bool> update_map_flag_{false};
  std::string initial_pcd_map_path_;
  float globalmap_voxel_size_ = 0.1;
  float points_voxel_filter_size_ = 0.1;
  int last_timeout_;
  std::atomic<ModeState>  mode_state_ = ModeState::INIT;
};
}  // namespace localization

int main(int argc, char** argv) {
  omp_set_num_threads(6);
  rclcpp::init(argc, argv);
  auto node = std::make_shared<localization::HdlLocalizationNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}