/**
 * @file mapping_alg.h
 * @brief
 * @author Liuzhao Li (liliuzhao@jushenzhiren.com)
 * @version 1.0
 * @date 2025-07-31
 * @copyright Copyright (C) 2025 具身智人(北京)科技有限公司
 */

#pragma once
#include "common/state_mode.h"
#include "ikd_tree/ikd_tree.h"
#include "pcd2grid.h"
#include "process/imu_process.h"
#include "process/lidar_process.h"
#include "so3_math.h"
#include "keyframe_store.h"
#include "loop_closure.h"
#include "map_z_drift_correction.h"
#include "lio_time_guard.h"

#include <Eigen/Core>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <functional>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <math.h>
#include <mtk_iekf/esekfom/esekfom.hpp>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <anubis_interfaces/srv/map_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>
#include <tf2_ros/transform_broadcaster.h>
#include <thread>
#include <unistd.h>
#include <visualization_msgs/msg/marker.hpp>
namespace anubis_mapping
{
    class MappingAlg : public rclcpp::Node
    {
    public:
        MappingAlg(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

        ~MappingAlg();

        void run();

        void reset();

    private:
        double get_time_sec(const builtin_interfaces::msg::Time& time);

        rclcpp::Time get_ros_time(double timestamp);

        void init();


        void pointsBody2World(PointType const* const pi, PointType* const po);

        void pointsBody2Imu(PointType const* const pi, PointType* const po);

        void points_cache_collect();

        void lasermap_fov_segment();

        void lidarCallBack(const sensor_msgs::msg::PointCloud2::UniquePtr msg);

        void imuCallBack(const sensor_msgs::msg::Imu::UniquePtr msg_in);

        bool syncData(MeasureGroup& meas);

        // A timestamp discontinuity invalidates the active mapping session.
        // This helper is deliberately one-way until reset() starts a fresh
        // session; there is no re-anchor path after a rejected frame.
        void mark_lio_time_guard_failed(const char* reason);

        void map_incremental();

        void pubWorldPoints(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull);

        void pubBodyPoints(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body);

        void pubMapPoints(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap);

        void stateCallBack(
            anubis_interfaces::srv::MapState::Request::SharedPtr request, anubis_interfaces::srv::MapState::Response::SharedPtr response);

        void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped,
            std::unique_ptr<tf2_ros::TransformBroadcaster>&                               tf_br);

        void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath);

        void map_publish_callback();

        void maybeAddKeyframe();
        scan_descriptor::Descriptor makeLoopDescriptor(
            const PointCloudType& cloud_lidar, const Eigen::Matrix4d& T_map_lidar) const;
        scan_descriptor::Descriptor makeGlobalLocalizationDescriptor(
            const PointCloudType& cloud_lidar, const Eigen::Matrix4d& T_map_base) const;
        bool saveDescriptorDatabase(
            const KeyframeStore::Snapshot& snapshot,
            const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& optimized_poses,
            const std::string& output_dir);
        bool levelMapForSave(
            PointCloudType& map_source,
            std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& optimized_poses);
        bool correctKeyframeZForSave(
            const KeyframeStore::Snapshot& snapshot,
            std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& optimized_poses,
            PointCloudType& map_source);
        Eigen::Matrix4d currentLidarPose() const;

        // --- AprilTag anchor integration -----------------------------------
        // Detection itself (image capture, AprilTag decoding) deliberately
        // lives in a separate, non-safety-critical node (anubis_perception,
        // wrapping apriltag_ros). This node only consumes the resulting
        // tag pose -- via a TF lookup, see AprilTagObservation below -- and
        // tags it to the nearest keyframe (by timestamp) so it can later be
        // resolved against the *loop-closure-corrected* keyframe pose at
        // save time, instead of the raw online LIO pose. Keeping detection
        // out of this class avoids adding image-processing latency or a new
        // failure surface to the 5 kHz mapping loop.
        struct AprilTagObservation
        {
            int marker_id = -1;
            uint64_t keyframe_id = 0;
            // [PATCH -- AprilTag migration] Previously T_lidar_marker was
            // computed as a hand-maintained extrinsic matrix
            // (aruco.lidar_to_camera_T) times the pose carried in the old
            // ArucoDetection message. apriltag_ros does not put a pose in
            // its detection message at all (only pixel corners + a
            // homography) -- it publishes the tag pose on /tf instead
            // (child_frame_id "tag<family>:<id>"). Since anubis_description
            // publishes the full base_link->livox_frame AND
            // base_link->camera_link->...->camera_color_optical_frame
            // chain, T_lidar_marker is now obtained with a single TF
            // lookup (tf_buffer_->lookupTransform(lidar_frame_id_,
            // "tag"+family+":"+id, stamp)) that tf2 composes across that
            // whole chain -- there is no separately-maintained extrinsic
            // matrix left to fall out of sync with the URDF.
            Eigen::Matrix4d T_lidar_marker = Eigen::Matrix4d::Identity();
        };

        void aprilTagDetectionCallBack(const apriltag_msgs::msg::AprilTagDetectionArray::UniquePtr msg);

        // Resolves every buffered observation against optimized_poses (the
        // same post-loop-closure/leveling poses used by
        // saveDescriptorDatabase) and writes apriltag_anchors.yaml into
        // output_dir. Call this from finish() in the same staging_dir used
        // by the descriptor/map artifacts, so it is covered by the
        // existing MapArtifactTransaction promote/rollback.
        bool saveAprilTagMap(
            const KeyframeStore::Snapshot& snapshot,
            const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& optimized_poses,
            const std::string& output_dir);

        void h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data);

        template <typename T>
        void pointBodyToWorld(const Eigen::Matrix<T, 3, 1>& pi, Eigen::Matrix<T, 3, 1>& po)
        {
            Vec3d p_body(pi[0], pi[1], pi[2]);
            Vec3d p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

            po[0] = p_global(0);
            po[1] = p_global(1);
            po[2] = p_global(2);
        }

        template <typename T>
        void set_posestamp(T& out)
        {
            out.pose.position.x    = state_point.pos(0);
            out.pose.position.y    = state_point.pos(1);
            out.pose.position.z    = state_point.pos(2);
            out.pose.orientation.x = geoQuat.x;
            out.pose.orientation.y = geoQuat.y;
            out.pose.orientation.z = geoQuat.z;
            out.pose.orientation.w = geoQuat.w;
        }

        inline double QuaternionToYaw(double x, double y, double z, double w)
        {
            return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
        }


    private:
        struct LioSyncDiagnostics
        {
            bool        valid                    = false;
            std::size_t lidar_queue_before       = 0;
            std::size_t imu_queue_before         = 0;
            std::size_t imu_queue_after          = 0;
            std::size_t imu_selected             = 0;
            std::size_t imu_before_lidar_begin   = 0;
            std::size_t imu_inside_lidar_scan    = 0;
            double      previous_lidar_end       = -1.0;
            double      imu_queue_front_before   = 0.0;
            double      imu_queue_back_before    = 0.0;
            double      imu_first_selected       = 0.0;
            double      imu_last_selected        = 0.0;
            double      imu_queue_front_after    = 0.0;
            double      point_offset_last_ms     = 0.0;
            double      point_offset_max_ms      = 0.0;
            double      scan_duration_s          = 0.0;
            bool        used_max_point_time      = false;
        };

        struct LioMatchDiagnostics
        {
            bool   valid                  = false;
            int    model_calls            = 0;
            int    initial_effective      = 0;
            int    final_effective        = 0;
            int    final_edge_effective   = 0;
            int    final_floor_normals    = 0;
            int    final_wall_normals     = 0;
            int    final_mixed_normals    = 0;
            double initial_residual_mean  = 0.0;
            double initial_residual_abs_mean = 0.0;
            double initial_residual_abs_p95  = 0.0;
            double final_residual_mean    = 0.0;
            double final_residual_abs_mean = 0.0;
            double final_residual_abs_p95  = 0.0;
            double final_residual_abs_max  = 0.0;
            double final_floor_residual_mean = 0.0;
            double final_floor_residual_abs_mean = 0.0;
            double final_wall_residual_mean = 0.0;
            double final_wall_residual_abs_mean = 0.0;
            double final_z_information     = 0.0;
            double final_z_rhs             = 0.0;
            double final_z_only_step       = 0.0;
            double initial_state_z        = 0.0;
            double initial_state_pitch_deg = 0.0;
            double final_model_state_z    = 0.0;
            double final_model_pitch_deg  = 0.0;
            double initial_condition      = 0.0;
            double final_condition        = 0.0;
            Eigen::Matrix<double, 6, 1> initial_weak =
                Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 6, 1> final_weak =
                Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 6, 1> final_hessian_diag =
                Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 6, 1> final_normalized_eigenvalues =
                Eigen::Matrix<double, 6, 1>::Zero();
        };

        bool extrinsic_est_en = true, path_en = true;

        float       res_last[100000]       = { 0.0 };
        float       DET_RANGE              = 300.0f;
        const float MOV_THRESHOLD          = 1.5f;
        double      time_diff_lidar_to_imu = 0.0;

        std::mutex              mtx_buffer;
        std::condition_variable sig_buffer;
        std::string             root_dir_ = ROOT_DIR;
        std::string             lid_topic, imu_topic;
        std::string             data_path_;

        double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
        bool   has_received_imu_timestamp_ = false;
        double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
        double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
        double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
        int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0;
        int    hessian_diag_counter_ = 0;
        bool   lio_trace_enabled_ = false;
        int    lio_trace_every_n_scans_ = 1;
        // 传感器时间连续性守卫:扫描间隙或帧内/跨帧 IMU 间隔超限即终止
        // 当前会话，不跨时间洞做状态预测积分(防止 LIO 轨迹发散)。
        bool   lio_time_guard_enabled_ = true;
        double lio_time_guard_max_scan_gap_ = 0.25;
        double lio_time_guard_max_imu_dt_ = 0.10;
        double lio_time_guard_max_scan_overlap_ = 0.002;
        // A rejected sensor-time frame invalidates the current mapping
        // session.  Continuing would make the next IMU prediction span the
        // blackout, so the operator must restart with a continuous stream.
        LioTimeGuardSessionState lio_time_guard_session_;
        std::uint64_t lio_frame_counter_ = 0;
        std::uint64_t current_lio_frame_id_ = 0;
        double current_lio_frame_time_ = 0.0;
        bool   trace_current_lio_frame_ = false;
        double previous_synced_lidar_end_time_ = -1.0;
        LioSyncDiagnostics lio_sync_diag_;
        LioMatchDiagnostics lio_match_diag_;
        int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0;
        bool   point_selected_surf[100000] = { 0 };
        bool   lidar_pushed = false;
        bool   flg_first_scan = true;
        bool   flg_EKF_inited = false;
        bool   pub_world_points_flag_ = false, pub_body_points_flag_ = false;
        bool   is_first_lidar = true;

        bool keyframe_enable_ = false;
        double keyframe_trans_threshold_ = 0.5;
        double keyframe_yaw_threshold_deg_ = 10.0;
        double keyframe_store_leaf_ = 0.1;
        // A separate dense cloud is retained for save-time ground evidence.
        // It is intentionally no coarser than half the PGM resolution so a
        // 5 cm grid is not fed by a 10 cm keyframe lattice.
        double keyframe_ground_store_leaf_ = 0.025;
        double keyframe_floor_z_map_ = -0.304;
        Eigen::Matrix4d lidar_to_base_extrinsic_ = Eigen::Matrix4d::Identity();
        // [2026-08-14] 2D 栅格图按关键帧轨迹外扩这么多米裁剪；<=0 关闭。
        double grid_crop_margin_ = 3.0;
        size_t keyframe_max_count_ = 5000;
        scan_descriptor::DescriptorConfig descriptor_config_;
        std::shared_ptr<KeyframeStore> keyframe_store_ = std::make_shared<KeyframeStore>();
        LoopClosureConfig loop_config_;
        std::unique_ptr<LoopClosure> loop_closure_;
        bool loop_log_once_ = false;

        bool map_leveling_enabled_ = true;
        // 保存时单层几何/整平质量门是否拦截。默认 false：整平/竖向修正/固定
        // 地面跨度门失败只记录诊断并按未修正几何保存（sidecar 记录
        // leveling_applied=false），以支持跨楼层等单层地面模型不适用的建图。
        // 回环/最终合成位姿安全门不受此开关影响，始终 fail-closed。
        bool map_leveling_enforce_gates_ = false;
        double map_leveling_max_correction_deg_ = 8.0;
        double map_leveling_min_inlier_ratio_ = 0.60;
        // Select the lower observed envelope per spatial cell for save-time
        // leveling without weakening the final quality gates.
        double map_leveling_sample_quantile_ = 0.05;
        double map_leveling_fallback_sample_quantile_ = 0.10;
        double map_leveling_converged_tilt_deg_ = 0.05;
        int map_leveling_max_iterations_ = 8;
        bool map_z_correction_enabled_ = true;
        double map_z_local_candidate_min_height_ = -0.65;
        double map_z_local_candidate_max_height_ = 0.65;
        int map_z_local_min_sample_cells_ = 15;
        double map_z_local_max_tilt_deg_ = 3.0;
        double map_z_local_max_residual_p95_ = 0.08;
        double map_z_local_min_range_ = 0.5;
        double map_z_local_max_range_ = 8.0;
        double map_z_local_max_source_distance_ = 2.5;
        int map_z_local_min_independent_sources_ = 2;
        int map_z_local_min_source_sample_cells_ = 3;
        // A single Mid-360 keyframe is sparse (the production replay uses
        // four scan lines).  When its local ground fit fails, use a bounded
        // temporal neighborhood of stored keyframes as the same spatial
        // observation.  The local quality gates remain unchanged.
        int map_z_local_keyframe_window_ = 6;
        MapZDriftCorrectionConfig map_z_correction_config_;
        std::vector<MapZDriftObservation> map_z_observations_;
        MapZDriftCorrectionResult map_z_correction_result_;
        double map_leveling_max_final_floor_span_m_ = 0.05;
        double map_z_final_floor_span_ = 0.0;
        Eigen::Matrix3d map_leveling_rotation_ = Eigen::Matrix3d::Identity();
        Eigen::Vector3d map_leveling_normal_ = Eigen::Vector3d::UnitZ();
        double map_leveling_floor_z_ = -0.304;
        double map_leveling_tilt_deg_ = 0.0;
        double map_leveling_inlier_ratio_ = 0.0;
        int map_leveling_iterations_ = 0;
        bool map_leveling_succeeded_ = false;
        double map_leveling_selected_quantile_ = 0.05;
        bool map_leveling_q10_attempted_ = false;
        bool map_leveling_q10_accepted_ = false;
        std::size_t map_leveling_seed_candidate_points_ = 0;
        std::size_t map_leveling_seed_sample_cells_ = 0;
        std::size_t map_leveling_seed_inlier_cells_ = 0;
        double map_leveling_seed_inlier_ratio_ = 0.0;
        double map_leveling_seed_residual_p95_ = 0.0;
        double map_leveling_seed_tilt_deg_ = 0.0;
        std::size_t map_leveling_refit_candidate_points_ = 0;
        std::size_t map_leveling_refit_sample_cells_ = 0;
        std::size_t map_leveling_refit_inlier_cells_ = 0;
        double map_leveling_refit_inlier_ratio_ = 0.0;
        double map_leveling_refit_residual_p95_ = 0.0;
        double map_leveling_refit_major_span_ = 0.0;
        double map_leveling_refit_minor_span_ = 0.0;
        std::string map_leveling_failure_reason_;
        // 最近一次整平质量门失败原因;不参与保存回滚,供跳过整平时
        // 的 WARN 日志与 sidecar failure_reason 使用。
        std::string map_leveling_last_gate_reason_;

        Pcd2GridOptions           pcd2pgm_options_;
        std::shared_ptr<Pcd2Grid> pcd2grid_ptr_;

        std::vector<vector<int>>  pointSearchInd_surf;
        std::vector<BoxPointType> cub_needrm;
        std::vector<PointVector>  Nearest_Points;
        std::vector<double>       extrinT;
        std::vector<double>       extrinR;
        std::deque<double>        time_buffer;
        std::deque<CloudPtr>      lidar_buffer;
        std::deque<ImuMessagePtr> imu_buffer;

        CloudPtr featsFromMap     = CloudPtr(new PointCloudType());
        CloudPtr feats_undistort  = CloudPtr(new PointCloudType());
        CloudPtr feats_down_body  = CloudPtr(new PointCloudType());
        CloudPtr feats_down_world = CloudPtr(new PointCloudType());
        CloudPtr normvec          = CloudPtr(new PointCloudType(100000, 1));
        CloudPtr laserCloudOri    = CloudPtr(new PointCloudType(100000, 1));
        CloudPtr corr_normvect    = CloudPtr(new PointCloudType(100000, 1));
        CloudPtr _featsArray      = CloudPtr(new PointCloudType());

        pcl::VoxelGrid<PointType> downSizeFilterSurf;
        pcl::VoxelGrid<PointType> downSizeFilterMap;
        // Edge points are voxelised separately from planar ones: a single
        // VoxelGrid averages every field inside a voxel, which would blend the
        // FeatureLabel stored in normal_x into a meaningless fraction.
        pcl::VoxelGrid<PointType> downSizeFilterEdge;

        // Scratch clouds for the split-by-label downsampling path.
        CloudPtr feats_surf_raw   = CloudPtr(new PointCloudType());
        CloudPtr feats_edge_raw   = CloudPtr(new PointCloudType());
        CloudPtr feats_surf_ds    = CloudPtr(new PointCloudType());
        CloudPtr feats_edge_ds    = CloudPtr(new PointCloudType());

        // Edge-aware residual tuning.
        bool   edge_constraint_enable = false;
        double filter_size_edge_min   = 0.1;
        double edge_residual_weight   = 0.5;
        int    edge_point_count       = 0;

        KD_TREE<PointType> ikdtree;

        Vec3d euler_cur;
        Vec3d position_last   = Zero3d;
        Vec3d Lidar_T_wrt_IMU = Zero3d;
        Mat3d Lidar_R_wrt_IMU = Eye3d;

        MeasureGroup                                 Measures;
        esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
        state_ikfom                                  state_point;
        vect3                                        pos_lid;

    public:
        bool finish();

    private:
        std::shared_ptr<Preprocess> p_pre = std::make_shared<Preprocess>();
        std::shared_ptr<ImuProcess> p_imu = std::make_shared<ImuProcess>();

    private:
        nav_msgs::msg::Path             path;
        nav_msgs::msg::Odometry         odomAftMapped;
        geometry_msgs::msg::Quaternion  geoQuat;
        geometry_msgs::msg::PoseStamped msg_body_pose;

        BoxPointType LocalMap_Points;
        bool         Localmap_Initialized = false;

        double lidar_mean_scantime = 0.0;
        int    scan_num            = 0;

        PointCloudType::Ptr pcl_wait_pub  = PointCloudType::Ptr(new PointCloudType());
        PointCloudType::Ptr pcl_wait_save = PointCloudType::Ptr(new PointCloudType());

        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pubLaserCloudFull_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pubLaserCloudFull_body_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pubLaserCloudMap_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          pubOdomAftMapped_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr              pubPath_;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr         sub_imu_ptr_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_ptr_;

        rclcpp::Service<anubis_interfaces::srv::MapState>::SharedPtr state_service_;

        // --- AprilTag anchor integration state ---
        bool apriltag_enable_ = false;
        std::string apriltag_detection_topic_ = "/perception/apriltag/detections";
        // Nearest-keyframe association is rejected beyond this time gap so a
        // detection made while the store had a stale/short snapshot cannot
        // silently attach to an unrelated keyframe.
        double apriltag_max_keyframe_time_gap_s_ = 1.0;
        int apriltag_min_observations_ = 3;
        double apriltag_max_position_deviation_m_ = 0.05;
        // Quality gate on apriltag_msgs/AprilTagDetection itself (hamming =
        // corrected bit count, 0 is best; decision_margin = detector
        // confidence, higher is better) -- replaces the old
        // reprojection_error_px/depth_validated gate, which depended on
        // ArUco's own PnP pose estimate that AprilTag detections don't
        // carry.
        int apriltag_max_hamming_ = 0;
        double apriltag_min_decision_margin_ = 50.0;
        std::string apriltag_family_ = "36h11";
        // Target frame for the TF lookup below -- normally "livox_frame".
        // No lidar-to-camera extrinsic parameter exists anymore; see the
        // comment on AprilTagObservation::T_lidar_marker in the struct
        // above for why.
        std::string apriltag_lidar_frame_id_ = "livox_frame";
        // Live per-sighting debug TF (odom -> apriltag_debug_<id>), for
        // visual sanity-checking during capture -- see the comment where
        // this is published in aprilTagDetectionCallBack() for why it is
        // NOT the same thing as the final map-frame anchor pose.
        bool apriltag_publish_debug_tf_ = true;
        rclcpp::Subscription<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr sub_apriltag_ptr_;
        std::mutex apriltag_mtx_;
        std::vector<AprilTagObservation> apriltag_observations_;
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

        std::atomic<SlamState> state_{ SlamState::STABLE };
        // ikd-tree owns a background rebuild thread and has no safe in-place
        // reset API.  Refuse a second mapping session in the same node instead
        // of silently mixing the previous map into the next one.
        bool mapping_session_started_ = false;

        std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
        rclcpp::TimerBase::SharedPtr                   map_pub_timer_;

        bool effect_pub_en = false, map_pub_en = false;
    };

}  // namespace anubis_mapping