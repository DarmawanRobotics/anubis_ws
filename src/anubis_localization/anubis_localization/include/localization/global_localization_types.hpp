#ifndef GLOBAL_LOCALIZATION_TYPES_HPP
#define GLOBAL_LOCALIZATION_TYPES_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/time.hpp>


namespace localization {

enum class GLStatus : uint8_t {
  Confirmed,
  PendingVote,
  Ambiguous,
  NoCandidate,
  LowConsensus,
};

enum class GLEpisodePhase : uint8_t {
  Idle,
  RecallAnchor,
  AwaitingPrior,
};

enum class GLAnchorRejectReason : uint8_t {
  None,
  GravityUnavailable,
  GravityStale,
  GravityUncertain,
  TiltExceeded,
  AngularVelocityExceeded,
  AccelerationResidualExceeded,
  MotionWindowExceeded,
  TooFewValidPoints,
};

struct GLParams {
  std::string gl_recall_source = "grid";
  bool retrieval_fallback_to_grid = true;
  int retrieval_topk = 3;
  double retrieval_nms_xy = 4.0;
  double retrieval_max_distance = 0.40;
  int retrieval_min_points = 300;
  double retrieval_yaw_half_range_deg = 10.0;
  double retrieval_yaw_step_deg = 5.0;
  double retrieval_search_radius_xy = 1.5;
  double retrieval_seed_stride_xy = 1.5;
  double level0_grid_stride_xy = 2.0;
  int level0_yaw_samples = 4;
  int level0_top_regions = 6;
  double level0_distance_field_resolution = 0.5;
  double level0_max_obstacle_distance = 3.0;
  double level0_base_ground_offset = -1.0;
  double level0_min_point_height = 0.15;
  double level0_max_point_height = 1.20;
  double level0_min_range = 1.0;
  double level0_max_range = 50.0;
  int level0_max_scan_points = 1000;
  int level0_min_valid_points = 200;
  bool level0_gravity_align_enabled = true;
  bool level0_raw_imu_gravity_filter_enabled = true;
  double level0_raw_imu_gravity_window_s = 0.30;
  // [2026-08-14] 重力滤波器自己的角速度门限（原借用已删除的
  // level0_anchor_low_dynamic_max_angular_velocity_rps）：IMU 样本只在机身
  // 角速度低于此值时参与重力估计。实测静止站立 0.01~0.02 rad/s。
  double level0_raw_imu_gravity_max_angular_velocity_rps = 0.05;
  int level0_raw_imu_gravity_min_samples = 30;
  double level0_gravity_time_offset_ms = 0.0;
  double level0_gravity_max_age_ms = 30.0;
  double level0_gravity_max_uncertainty_deg = 2.0;
  double level0_anchor_max_abs_rp_deg = 15.0;
  double level0_unaligned_max_abs_rp_deg = 3.0;
  double level0_anchor_max_angular_velocity_rps = 0.5;
  double level0_gravity_max_accel_residual_mps2 = 1.5;
  double level0_anchor_max_linear_velocity_mps = 0.10;
  int level0_anchor_recapture_limit = 5;
  double level0_min_valid_projection_ratio = 0.60;
  double level0_invalid_projection_penalty = 3.0;
  double level0_boundary_padding_m = 5.0;
  double level0_max_boundary_unknown_ratio = 0.40;
  double level0_region_nms_xy = 4.0;
  double level0_region_nms_yaw_deg = 30.0;
  double level1_search_radius_xy = 2.0;
  double level1_seed_stride_xy = 1.5;
  double level1_yaw_step_deg = 15.0;
  int max_candidates_total = 80;
  int max_refine_candidates = 12;
  double seed_dedup_xy = 0.2;
  double seed_dedup_yaw_deg = 3.0;
  double grid_z_percentile = 0.10;
  double coarse_leaf_map = 1.0;
  double coarse_leaf_src = 0.5;
  int coarse_max_iter = 15;
  double coarse_roi_radius = 15.0;
  double coarse_max_corr_dist = 2.0;
  double coarse_min_overlap = 0.10;
  int coarse_min_inliers = 100;
  double coarse_max_p90_residual = 1.0;
  double refine_leaf_src = 0.2;
  int refine_max_iter = 100;
  double refine_fitness_threshold = 0.25;
  double max_correction_xy = 2.5;
  double max_correction_z = 1.5;
  double max_yaw_delta_deg = 45.0;
  double max_rp_delta_deg = 10.0;
  double max_abs_roll_deg = 15.0;
  double max_abs_pitch_deg = 15.0;
  double cluster_xy = 0.5;
  double cluster_yaw_deg = 15.0;
  double support_seed_xy = 1.2;
  int min_support_groups = 2;
  double ambiguous_pose_distance = 2.0;
  double ambiguous_score_ratio = 1.15;
  // [2026-08-14] Static (motion-free) confirmation replaces the motion-driven
  // temporal vote. Retrieval recall is fast enough to repeat from the SAME
  // viewpoint, so confirmation now needs N consecutive independent probes to
  // agree instead of the robot walking/rotating between votes. See
  // global_localization_static_confirm.hpp for the full three-gate rationale.
  int gl_confirm_frames = 3;
  double gl_confirm_xy_tol = 0.30;
  double gl_confirm_yaw_tol_deg = 5.0;
  double gl_probe_period_s = 0.5;
  double gl_attempt_timeout_s = 30.0;
  double min_retry_interval_s = 0.5;
  double per_call_deadline_ms = 1000.0;
  double episode_timeout_s = 10.0;
  double max_scan_age_after_gl_s = 0.20;
  double shadow_gl_budget_ms = 50.0;
  double shadow_gl_period_s = 10.0;
  bool allow_degraded_fallback = false;
  bool flat_single_level_map = false;
  bool auto_confirm_enabled = false;
  double auto_confirm_verify_timeout_s = 5.0;
  std::string m2b_approval_id;
  bool advisory_suggestion_enabled = false;
  double advisory_min_confidence = -1.0;
  std::string advisory_calibration_id;
  bool dump_candidates_csv = false;
  std::string candidates_csv_path;
  std::string candidates_csv_episode_id;
  bool candidate_debug_log = false;
};

struct GLSummaryCounts {
  int retrieval_db_size = 0;
  int retrieval_match_count = 0;
  double retrieval_best_distance = 1e9;
  int seed_total = 0;
  int coarse_kept = 0;
  int refined = 0;
  int passed = 0;
  int rejected_conv = 0;
  int rejected_score = 0;
  int rejected_xy = 0;
  int rejected_z = 0;
  int rejected_yaw = 0;
  int rejected_rp = 0;
  int cluster_count = 0;
  int best_cluster_size = 0;
  int best_support_group_count = 0;
  int level = 0;
  int batch_index = 0;
  int batch_count = 0;
  int covered_cells = 0;
  int total_cells = 0;
  int candidate_bank_size = 0;
  int stale_scans_dropped = 0;
  int level0_input_points = 0;
  int level0_valid_points = 0;
  int level0_boundary_unknown_points = 0;
  int level0_invalid_projection_points = 0;
  uint64_t level0_lookup_count = 0;
  int level0_invalid_hypotheses = 0;
  int level1_seed_count = 0;
  int level1_batch_count = 0;
  int static_confirm_probe_count = 0;
  int static_confirm_agree_count = 0;
  int anchor_recapture_count = 0;
  int anchor_rejected_gravity_unavailable = 0;
  int anchor_rejected_gravity_stale = 0;
  int anchor_rejected_gravity_uncertain = 0;
  int anchor_rejected_rp = 0;
  int anchor_rejected_angular_velocity = 0;
  int anchor_rejected_accel_residual = 0;
  int anchor_rejected_motion_window = 0;
  int anchor_rejected_too_few_points = 0;
  double low_dynamic_wait_ms = 0.0;
  double anchor_linear_velocity_mps = -1.0;
  double motion_odom_age_ms = -1.0;
  double anchor_gravity_age_ms = -1.0;
  double anchor_gravity_clock_residual_ms = -1.0;
  double anchor_gravity_uncertainty_deg = -1.0;
  double anchor_roll_deg = 0.0;
  double anchor_pitch_deg = 0.0;
  double anchor_angular_velocity_rps = -1.0;
  double anchor_accel_norm_residual_mps2 = -1.0;
  double anchor_scan_duration_ms = -1.0;
  double anchor_intrascan_rp_span_deg = -1.0;
  double recall_motion_travel_m = 0.0;
  double processing_ms = 0.0;
  double await_evidence_ms = 0.0;
  bool level0_ground_unavailable = false;
};

struct GravityAlignmentSample {
  rclcpp::Time stamp;
  Eigen::Matrix3d R_level_base = Eigen::Matrix3d::Identity();
  double roll_deg = 0.0;
  double pitch_deg = 0.0;
  double uncertainty_deg = 1e9;
  double angular_velocity_rps = 1e9;
  double accel_norm_residual_mps2 = 1e9;
  std::string source;
  bool valid = false;
};

struct GlobalPoseCandidate {
  Eigen::Matrix4d seed_pose = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d final_pose = Eigen::Matrix4d::Identity();
  double fitness_score = 1e9;
  double coarse_score = 1e9;
  double level0_mean_truncated_distance = 1e9;
  double level0_valid_projection_ratio = 0.0;
  int level0_input_points = 0;
  int level0_valid_points = 0;
  double overlap_ratio = 0.0;
  int inlier_count = 0;
  double p90_residual = 1e9;
  double correction_xy = 0.0;
  double correction_z = 0.0;
  double yaw_delta_deg = 0.0;
  double roll_delta_deg = 0.0;
  double pitch_delta_deg = 0.0;
  double final_yaw_deg = 0.0;
  std::string source_type;
  int grid_cell_id = -1;
  int yaw_bin = -1;
  int rotation_batch = -1;
  std::string support_group;
  std::string evidence_id;
  std::string evidence_role;
  bool converged = false;
};

struct GlobalCluster {
  std::vector<int> member_indices;
  int size = 0;
  double best_score = 1e9;
  double mean_score = 1e9;
  double score_variance = 0.0;
  Eigen::Vector3d mean_pos = Eigen::Vector3d::Zero();
  double mean_yaw_deg = 0.0;
  int support_group_count = 0;
  Eigen::Matrix4d medoid_pose = Eigen::Matrix4d::Identity();
};

struct GlobalClusterVote {
  Eigen::Matrix4d medoid_pose_at_anchor = Eigen::Matrix4d::Identity();
  int hit_count = 0;
  int miss_count = 0;
  bool has_value = false;
};

struct GlobalLocalizationResult {
  GLStatus status = GLStatus::NoCandidate;
  bool success = false;
  bool ambiguous = false;
  bool budget_exceeded = false;
  bool waiting_low_dynamic = false;
  bool anchor_invalid = false;
  bool anchor_motion_exceeded = false;
  GLAnchorRejectReason anchor_reject_reason = GLAnchorRejectReason::None;
  Eigen::Matrix4d final_pose = Eigen::Matrix4d::Identity();
  double best_score = -1.0;
  int best_cluster_size = 0;
  int candidate_count = 0;
  int coarse_kept_count = 0;
  int refined_count = 0;
  int passed_filter_count = 0;
  int cluster_count = 0;
  int best_cluster_index = -1;
  // Level 0 top regions are exported as evaluator stage=coarse. They are
  // converted to T_map_base poses before being stored here.
  std::vector<GlobalPoseCandidate> level0_candidates;
  // Level 1 inputs retain their original seed_pose. final_pose and coarse
  // quality fields contain the result of coarse alignment when it ran.
  std::vector<GlobalPoseCandidate> seed_candidates;
  std::vector<GlobalPoseCandidate> candidates;
  std::vector<GlobalPoseCandidate> passed_candidates;
  std::vector<GlobalCluster> clusters;
};

}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_TYPES_HPP
