/**
 * @brief
 */

#pragma once
#include "common.h"
#include "pcd2grid_ground_plane.h"

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/srv/get_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>

namespace anubis_mapping
{
    struct Pcd2GridOptions
    {
        std::string file_name = "map";
        // All z thresholds below are heights above this map-frame floor.
        // MappingAlg supplies the same floor used by keyframe descriptors.
        double floor_z_map = -0.304;
        // [2026-08-26] 原值 0.05m；最新实测地图在自适应地面回退后仍有
        // 0.05~0.10m 地面点进入障碍带，离线同 PCD 对比提高到 0.10m 后
        // 占用格减少 1400（10.4%），未见墙线明显缺失。
        double thre_z_min = 0.10;
        double thre_z_max = 3.0;
        int flag_pass_through = 0;
        // Leaf size used for the saved PCD artifact.  The 2-D grid intentionally
        // consumes the dense map source, but the value is kept here so the
        // complete quantization contract can be recorded in the sidecar.
        double save_voxel = 0.1;
        // [2026-08-21] 原值 0.10m；与导航 costmap 的 0.05m 分辨率一致，
        // 减少窄门两侧各一个粗栅格造成的净空损失。
        double map_resolution = 0.05;
        double thre_radius = 0.15;
        // Minimum number of height-band points required before a cell can be
        // considered a persistent obstacle.  This is deliberately separate
        // from thres_point_count, which controls the 3-D radius filter.
        int min_points_occupied = 2;
        int thres_point_count = 2;
        int min_obstacle_frames = 2;
        int min_free_frames = 2;
        double free_to_obstacle_frame_ratio = 2.0;
        // [2026-08-14] 悬空结构判定：一个栅格里最低点离地若高于此值，说明它下方
        // 是空的（门洞上方的过梁、桌面、吊顶横梁），机器人可以从下面通过 ——
        // 判为可通行而不是障碍物。
        //
        // 为什么需要它：固定 z 上限不可靠。实测三道隔墙的过梁点在 1.95/1.99/
        // 2.09m（各房间天花板 1.95~2.45m 本就不同），z 上限 2.0 恰好切在这
        // 条带中间：隔墙1/2 的过梁落进带内 → 门口被判占用、通道从 1.3m 压到
        // 0.2-0.3m（机身 0.43m 过不去）；隔墙3 的过梁恰好高出 0.1m 躲过 →
        // 门口干净。用"最低点高度"判定则与过梁绝对高度无关。
        //
        // 当前机器狗含雷达最高约 0.49m。原值 0.60m；门楣边缘最低点实测
        // 约 0.593m，改为 0.55m 后仍保留约 0.06m 垂直余量。
        // 更高的门楣/桌面按悬空结构清除；<=0 表示关闭该判定。
        double overhead_clearance_z = 0.55;
        // [2026-08-29/30 门口混合格修复 总开关] false = 完全恢复旧的
        // "格内最低点 <= overhead_clearance_z 即占用"判定。true = 启用
        // 低点计数/帧数判定；当保存链带有关键帧扫描时，还必须通过下方
        // 的射线可见性门才允许把混合格清为空闲。
        bool low_point_gate_enabled = true;
        // [2026-08-29/30 门口混合格修复] 悬空判定的低点计数门限:格内
        // 低于 overhead_clearance_z 的点数 >= min_low_points_occupied 时
        // 判真实矮障碍(占用);零星低点(门槛条回波混进门楣格,如
        // 10~13cm 低点+2.4m 过梁)不再劫持整格。
        // 细东西可能只有 1~2 个占用点,此门限保持 2 不上调。
        int min_low_points_occupied = 2;
        // 低点占比门限:低点数/格内障碍点数 >= 此值(与计数门限同时满足)
        // 才判真实矮障碍。
        double overhead_min_low_ratio = 0.25;
        // [2026-08-30] 零星低点(个数 < min_low_points_occupied)的持续性
        // 门槛:低点在超过该帧数的不同关键帧中同一格持续出现才判阻挡
        // (可能是墙脚/门框);实测门口门轨 5~10 帧被正确放行,墙柱低点
        // 37 帧被保守保留。
        int overhead_low_frame_persistence = 10;
        // 当关键帧扫描证据可用时，混合格/门楣格只有在地面射线从该格
        // 后方穿过至少这么多帧、且达到下面的占比后才允许清除。这样可
        // 防止“先按低点门限清墙、再由射线穿墙”造成长条状碎片。
        int overhead_min_ray_crossing_frames = 2;
        double overhead_min_ray_crossing_ratio = 0.10;
        // 聚合地图里落在 [floor_z_map + ground_free_min_height,
        // floor_z_map + thre_z_min) 的点是地面观测，可证明对应 2D 栅格为空闲。
        bool ground_free_enabled = true;
        double ground_free_min_height = -0.15;
        // 只对逐关键帧地面回波做 2D 射线清空，并在已确认障碍前停止。
        bool raytrace_free_enabled = true;
        double raytrace_max_range = 10.0;
        // 仅用于日志：占用格内所有点都低于该离地高度时，记录为疑似地面漏点。
        double low_obstacle_diagnostic_height = 0.25;
        // 保存 PGM 时输出逐栅格证据 CSV。仅记录现有分类输入和结果，不改变
        // 高度滤波、半径滤波、射线清空或占用优先级。
        bool cell_evidence_diagnostics_enabled = true;
        // PCD 可能因 FAST-LIO 俯仰累计误差出现缓慢的 map.z 漂移。PGM 只在
        // 高度分类阶段使用拟合平面；GL 描述子仍使用固定 floor_z_map。
        bool ground_plane_enabled = true;
        // 地面拟合失败时的保存策略。true 为 fail-closed：拒绝保存；
        // false 记录 ERROR 后退回固定 floor_z_map 分类继续保存，
        // 用于跨楼层等单层地面模型本就不适用的场景。
        bool ground_plane_required = true;
        double ground_plane_candidate_min_height = -0.15;
        double ground_plane_candidate_max_height = 0.30;
        int ground_plane_min_sample_cells = 100;
        double ground_plane_min_inlier_ratio = 0.60;
        double ground_plane_max_tilt_deg = 3.0;
        double ground_plane_max_residual_p95 = 0.08;
        // Absolute fitted-floor deviation gate after save-time leveling. Keep
        // this tighter than the first-pass leveling search; residual, tilt and
        // curvature gates remain active as well.
        double ground_plane_max_floor_offset = 0.25;
        // When enabled, robust plane and quadratic-surface candidates are
        // evaluated independently; the lowest-residual candidate that passes
        // all quality gates is selected.
        bool ground_plane_quadratic_enabled = true;
        double ground_plane_quadratic_max_local_tilt_deg = 3.5;
        double ground_plane_quadratic_max_curvature = 0.02;
    };

    struct Pcd2GridScan
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Matrix4d T_map_lidar = Eigen::Matrix4d::Identity();
        PointCloudType::ConstPtr cloud_lidar;
    };

    using Pcd2GridScans =
        std::vector<Pcd2GridScan, Eigen::aligned_allocator<Pcd2GridScan>>;

    class Pcd2Grid
    {
    public:
        explicit Pcd2Grid(const Pcd2GridOptions &options);
        ~Pcd2Grid() {}

        bool run(
            const CloudPtr &map_points, const std::string &file_name,
            const Pcd2GridScans &scans = {});

    private:
        Pcd2GridOptions options_;
        GroundPlaneModel ground_plane_;

        void HeightFilter(const CloudPtr &pcd_cloud, CloudPtr &cloud_after_height);

        void RadiusOutlierFilter(const CloudPtr &pcd_cloud, CloudPtr &cloud_after_radius);

        bool SetMapTopicMsg(
            const CloudPtr &map_points, const CloudPtr &height_filtered_points,
            const CloudPtr &obstacle_points, const Pcd2GridScans &scans,
            nav_msgs::msg::OccupancyGrid &msg,
            const std::string &diagnostics_file_name);

        bool SavePGMAndYAML(const nav_msgs::msg::OccupancyGrid &msg, const std::string &name);
    };
}

using Pcd2Grid = anubis_mapping::Pcd2Grid;
