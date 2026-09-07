
#include "pcd2grid.h"
#include "pcd2grid_ground_plane.h"
#include "pcd2grid_height.h"
#include "pcd2grid_logic.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <vector>

namespace anubis_mapping
{
    Pcd2Grid::Pcd2Grid(const Pcd2GridOptions &options) : options_(options)
    {
    }

    bool Pcd2Grid::run(
        const CloudPtr &pcd_cloud, const std::string &file_name,
        const Pcd2GridScans &scans)
    {
        if (!pcd_cloud || pcd_cloud->empty())
        {
            RCLCPP_ERROR(rclcpp::get_logger("pcd2grid"), "Cannot create grid map from an empty point cloud");
            return false;
        }
        for (const auto& scan : scans)
        {
            if (!scan.cloud_lidar || !scan.T_map_lidar.allFinite() ||
                !scan.T_map_lidar.row(3).isApprox(
                    Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-6))
            {
                RCLCPP_ERROR(
                    rclcpp::get_logger("pcd2grid"),
                    "Invalid scan evidence: missing cloud or non-finite/non-homogeneous pose");
                return false;
            }
            const Eigen::Matrix3d scan_rotation =
                scan.T_map_lidar.block<3, 3>(0, 0);
            if ((scan_rotation.transpose() * scan_rotation -
                 Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1e-4 ||
                std::abs(scan_rotation.determinant() - 1.0) > 1e-4)
            {
                RCLCPP_ERROR(
                    rclcpp::get_logger("pcd2grid"),
                    "Invalid scan evidence pose rotation");
                return false;
            }
        }
        if (!valid_pcd2grid_height_contract(
                options_.floor_z_map, options_.thre_z_min,
                options_.thre_z_max, options_.overhead_clearance_z))
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("pcd2grid"),
                "Invalid height contract: floor_z_map=%.3f relative_z=[%.3f,%.3f]",
                options_.floor_z_map, options_.thre_z_min, options_.thre_z_max);
            return false;
        }
        if ((options_.ground_free_enabled || options_.raytrace_free_enabled) &&
            (!std::isfinite(options_.ground_free_min_height) ||
             options_.ground_free_min_height >= options_.thre_z_min))
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("pcd2grid"),
                "Invalid ground free-space band: relative_z=[%.3f,%.3f)",
                options_.ground_free_min_height, options_.thre_z_min);
            return false;
        }
        if (options_.raytrace_free_enabled &&
            (!std::isfinite(options_.raytrace_max_range) ||
             options_.raytrace_max_range <= 0.0))
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("pcd2grid"),
                "pcd2pgm.raytrace_max_range must be positive");
            return false;
        }
        if (!std::isfinite(options_.low_obstacle_diagnostic_height) ||
            options_.low_obstacle_diagnostic_height <= options_.thre_z_min)
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("pcd2grid"),
                "pcd2pgm.low_obstacle_diagnostic_height must exceed thre_z_min");
            return false;
        }
        if (!std::isfinite(options_.map_resolution) ||
            options_.map_resolution <= 0.0 ||
            !std::isfinite(options_.thre_radius) ||
            options_.thre_radius < 0.0 ||
            options_.min_points_occupied < 1 ||
            options_.min_obstacle_frames < 1 || options_.min_free_frames < 1 ||
            !std::isfinite(options_.free_to_obstacle_frame_ratio) ||
            options_.free_to_obstacle_frame_ratio < 0.0)
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("pcd2grid"),
                "Invalid PCD-to-grid numeric/evidence thresholds: "
                "resolution=%.6f radius=%.6f min_points=%d "
                "min_obstacle_frames=%d min_free_frames=%d ratio=%.3f",
                options_.map_resolution, options_.thre_radius,
                options_.min_points_occupied, options_.min_obstacle_frames,
                options_.min_free_frames,
                options_.free_to_obstacle_frame_ratio);
            return false;
        }
        GroundPlaneFitOptions plane_options;
        plane_options.fixed_floor_z = options_.floor_z_map;
        plane_options.candidate_min_height =
            options_.ground_plane_candidate_min_height;
        plane_options.candidate_max_height =
            options_.ground_plane_candidate_max_height;
        plane_options.min_sample_cells = static_cast<std::size_t>(
            std::max(0, options_.ground_plane_min_sample_cells));
        plane_options.min_inlier_ratio = options_.ground_plane_min_inlier_ratio;
        plane_options.max_tilt_deg = options_.ground_plane_max_tilt_deg;
        plane_options.max_residual_p95 =
            options_.ground_plane_max_residual_p95;
        plane_options.max_floor_offset =
            options_.ground_plane_max_floor_offset;
        plane_options.quadratic_enabled =
            options_.ground_plane_quadratic_enabled;
        plane_options.quadratic_max_local_tilt_deg =
            options_.ground_plane_quadratic_max_local_tilt_deg;
        plane_options.quadratic_max_curvature =
            options_.ground_plane_quadratic_max_curvature;

        std::vector<GroundPlanePoint> plane_points;
        plane_points.reserve(pcd_cloud->size());
        for (const auto& point : pcd_cloud->points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                continue;
            }
            plane_points.push_back({point.x, point.y, point.z});
        }
        if (options_.ground_plane_enabled)
        {
            ground_plane_ = fit_ground_plane(plane_points, plane_options);
        }
        else
        {
            ground_plane_.adaptive = false;
            ground_plane_.fixed_floor_z = options_.floor_z_map;
            ground_plane_.a = 0.0;
            ground_plane_.b = 0.0;
            ground_plane_.c = options_.floor_z_map;
            ground_plane_.fitted_floor_min = options_.floor_z_map;
            ground_plane_.fitted_floor_max = options_.floor_z_map;
            ground_plane_.fallback_reason = "disabled by configuration";
        }
        if (ground_plane_.adaptive)
        {
            const double max_floor_deviation = std::max(
                std::abs(ground_plane_.fitted_floor_min -
                    options_.floor_z_map),
                std::abs(ground_plane_.fitted_floor_max -
                    options_.floor_z_map));
            if (ground_plane_.quadratic)
            {
                RCLCPP_INFO(
                    rclcpp::get_logger("pcd2grid"),
                    "Ground quadratic accepted: center=[%.3f,%.3f] scale=%.3f "
                    "coeff=[%+.6f,%+.6f,%+.6f,%+.6f,%+.6f,%+.6f] "
                    "local_tilt_p99/max=[%.3f,%.3f]deg curvature=%.6f/m "
                    "floor_range=[%.3f,%.3f]m candidates=%zu samples=%zu "
                    "inliers=%zu (%.1f%%) residual_p95=%.3fm "
                    "median_offset_from_floor_z_map=%+.3fm "
                    "floor_deviation_max=%.3fm gate=%.3fm",
                    ground_plane_.center_x, ground_plane_.center_y,
                    ground_plane_.coordinate_scale, ground_plane_.a,
                    ground_plane_.b, ground_plane_.c, ground_plane_.q_xx,
                    ground_plane_.q_xy, ground_plane_.q_yy,
                    ground_plane_.tilt_deg, ground_plane_.max_local_tilt_deg,
                    ground_plane_.curvature, ground_plane_.fitted_floor_min,
                    ground_plane_.fitted_floor_max,
                    ground_plane_.candidate_points, ground_plane_.sample_cells,
                    ground_plane_.inlier_cells,
                    100.0 * ground_plane_.inlier_ratio,
                    ground_plane_.residual_p95,
                    ground_plane_.median_floor_offset,
                    max_floor_deviation,
                    options_.ground_plane_max_floor_offset);
            }
            else
            {
                RCLCPP_INFO(
                    rclcpp::get_logger("pcd2grid"),
                    "Ground plane accepted: map_z=%+.6f*x%+.6f*y%+.6f "
                    "tilt=%.3fdeg floor_range=[%.3f,%.3f]m candidates=%zu "
                    "samples=%zu inliers=%zu (%.1f%%) residual_p95=%.3fm "
                    "median_offset_from_floor_z_map=%+.3fm "
                    "floor_deviation_max=%.3fm gate=%.3fm",
                    ground_plane_.a, ground_plane_.b, ground_plane_.c,
                    ground_plane_.tilt_deg, ground_plane_.fitted_floor_min,
                    ground_plane_.fitted_floor_max,
                    ground_plane_.candidate_points, ground_plane_.sample_cells,
                    ground_plane_.inlier_cells,
                    100.0 * ground_plane_.inlier_ratio,
                    ground_plane_.residual_p95,
                    ground_plane_.median_floor_offset,
                    max_floor_deviation,
                    options_.ground_plane_max_floor_offset);
            }
            if (ground_plane_.tilt_deg > 0.5 ||
                ground_plane_.fitted_floor_max -
                        ground_plane_.fitted_floor_min > 0.10)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("pcd2grid"),
                    "Map floor varies by %.3fm (model=%s tilt_metric=%.3fdeg). "
                    "PGM height filtering "
                    "is compensating this map geometry; inspect FAST-LIO pitch, "
                    "IMU initialization and time synchronization.",
                    ground_plane_.fitted_floor_max -
                        ground_plane_.fitted_floor_min,
                    ground_plane_.model_name(), ground_plane_.tilt_deg);
            }
        }
        else
        {
            const double max_floor_deviation = std::max(
                std::abs(ground_plane_.fitted_floor_min -
                    options_.floor_z_map),
                std::abs(ground_plane_.fitted_floor_max -
                    options_.floor_z_map));
            RCLCPP_WARN(
                rclcpp::get_logger("pcd2grid"),
                "Ground plane fallback: reason=%s fixed_floor_z=%.3fm "
                "candidates=%zu samples=%zu inliers=%zu "
                "fitted_floor_range=[%.3f,%.3f]m "
                "floor_deviation_max=%.3fm gate=%.3fm",
                ground_plane_.fallback_reason.c_str(), options_.floor_z_map,
                ground_plane_.candidate_points, ground_plane_.sample_cells,
                ground_plane_.inlier_cells, ground_plane_.fitted_floor_min,
                ground_plane_.fitted_floor_max, max_floor_deviation,
                options_.ground_plane_max_floor_offset);
            if (options_.ground_plane_enabled)
            {
                if (!ground_plane_fallback_allowed(
                        options_.ground_plane_enabled,
                        options_.ground_plane_required,
                        ground_plane_.adaptive))
                {
                    RCLCPP_ERROR(
                        rclcpp::get_logger("pcd2grid"),
                        "Ground plane fitting failed; refusing to save a map "
                        "with a fixed-floor fallback: reason=%s",
                        ground_plane_.fallback_reason.c_str());
                    return false;
                }
                RCLCPP_ERROR(
                    rclcpp::get_logger("pcd2grid"),
                    "Ground plane fitting failed; saving with fixed-floor "
                    "fallback because ground_plane_required=false: reason=%s",
                    ground_plane_.fallback_reason.c_str());
            }
        }
        RCLCPP_INFO(
            rclcpp::get_logger("pcd2grid"),
            "PCD-to-grid height contract: floor_model=%s floor_z_map=%.3f "
            "relative_z=[%.3f,%.3f] overhead_clearance=%.3f ground_free=%s "
            "relative_ground_z=[%.3f,%.3f) raytrace=%s max_range=%.1f "
            "ground_plane_required=%s",
            ground_plane_.model_name(),
            options_.floor_z_map, options_.thre_z_min, options_.thre_z_max,
            options_.overhead_clearance_z,
            options_.ground_free_enabled ? "on" : "off",
            options_.ground_free_min_height, options_.thre_z_min,
            options_.raytrace_free_enabled ? "on" : "off",
            options_.raytrace_max_range,
            options_.ground_plane_required ? "true" : "false");
        if (options_.low_point_gate_enabled)
        {
            RCLCPP_INFO(
                rclcpp::get_logger("pcd2grid"),
                "Overhead visibility gate: %s scans=%zu min_cross_frames=%d "
                "min_cross_ratio=%.3f (ray tracing is conservative before "
                "final overhead classification)",
                scans.empty() ? "aggregate-fallback" : "keyframe-ray",
                scans.size(), options_.overhead_min_ray_crossing_frames,
                options_.overhead_min_ray_crossing_ratio);
        }

        CloudPtr cloud_after_pass_through = CloudPtr(new PointCloudType());
        CloudPtr cloud_after_radius = CloudPtr(new PointCloudType());
        nav_msgs::msg::OccupancyGrid map_topic_msg;

        HeightFilter(pcd_cloud, cloud_after_pass_through);
        RadiusOutlierFilter(cloud_after_pass_through, cloud_after_radius);
        if (!SetMapTopicMsg(
                pcd_cloud, cloud_after_pass_through, cloud_after_radius,
                scans, map_topic_msg,
                options_.cell_evidence_diagnostics_enabled
                    ? file_name + "_cell_evidence.csv" : std::string()))
        {
            return false;
        }
        return SavePGMAndYAML(map_topic_msg, file_name);
    }
    void Pcd2Grid::HeightFilter(
        const CloudPtr &pcd_cloud, CloudPtr &cloud_after_height)
    {
        cloud_after_height->clear();
        cloud_after_height->reserve(pcd_cloud->size());
        const bool keep_outside = options_.flag_pass_through != 0;
        for (const auto& point : pcd_cloud->points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                continue;
            }
            const double height = ground_plane_.height(
                point.x, point.y, point.z);
            const bool inside = height >= options_.thre_z_min &&
                height <= options_.thre_z_max;
            if (inside != keep_outside)
            {
                cloud_after_height->push_back(point);
            }
        }
    }

    void Pcd2Grid::RadiusOutlierFilter(const CloudPtr &pcd_cloud, CloudPtr &cloud_after_radius)
    {
        if (options_.thres_point_count <= 0 || options_.thre_radius <= 0.0)
        {
            *cloud_after_radius = *pcd_cloud;
            return;
        }
        pcl::RadiusOutlierRemoval<PointType> radiusoutlier;
        radiusoutlier.setInputCloud(pcd_cloud);
        radiusoutlier.setRadiusSearch(options_.thre_radius);
        radiusoutlier.setMinNeighborsInRadius(options_.thres_point_count);
        radiusoutlier.filter(*cloud_after_radius);
        // pcl::io::savePCDFile<PointType>(options_.file_name + "_radius_filter.pcd",
        //                                 *cloud_after_radius);
        // std::cout << "Point cloud size after radius filter: "
        //           << cloud_after_radius->points.size() << std::endl;
    }

    bool Pcd2Grid::SetMapTopicMsg(
        const CloudPtr &map_points, const CloudPtr &height_filtered_points,
        const CloudPtr &obstacle_points, const Pcd2GridScans &scans,
        nav_msgs::msg::OccupancyGrid &msg,
        const std::string &diagnostics_file_name)
    {
        msg.header.stamp = rclcpp::Clock().now();
        msg.header.frame_id = "map";
        msg.info.map_load_time = rclcpp::Clock().now();
        msg.info.resolution = options_.map_resolution;

        if (!map_points || map_points->points.empty())
        {
            RCLCPP_ERROR(rclcpp::get_logger("pcd2grid"), "Cannot create a grid without map points");
            return false;
        }
        if (options_.map_resolution <= 0.0)
        {
            RCLCPP_ERROR(rclcpp::get_logger("pcd2grid"), "map_resolution must be positive");
            return false;
        }

        double x_min = std::numeric_limits<double>::max();
        double x_max = std::numeric_limits<double>::lowest();
        double y_min = std::numeric_limits<double>::max();
        double y_max = std::numeric_limits<double>::lowest();
        auto update_bounds = [&](double x, double y)
        {
            if (!std::isfinite(x) || !std::isfinite(y))
            {
                return;
            }
            if (x < x_min)
                x_min = x;
            if (x > x_max)
                x_max = x;

            if (y < y_min)
                y_min = y;
            if (y > y_max)
                y_max = y;
        };
        // The raw map is the geometric extent of this conversion, independent
        // of which evidence channels are enabled.  Tying bounds to the
        // obstacle/ground switches can collapse a ground-only or ray-only map
        // to a single cell and silently discard valid free-space evidence.
        for (const auto &point : map_points->points)
        {
            update_bounds(point.x, point.y);
        }
        for (const auto &scan : scans)
        {
            if (scan.T_map_lidar.allFinite())
            {
                update_bounds(scan.T_map_lidar(0, 3), scan.T_map_lidar(1, 3));
            }
        }
        // Ray endpoints can be ground returns even when aggregate ground
        // evidence is disabled.  Include only valid ground endpoints within
        // the configured range so a distant wall/outlier cannot inflate the
        // grid extent while the ray traversal remains fully representable.
        if (options_.raytrace_free_enabled)
        {
            for (const auto &scan : scans)
            {
                if (!scan.T_map_lidar.allFinite() || !scan.cloud_lidar)
                {
                    continue;
                }
                for (const auto &point_lidar : scan.cloud_lidar->points)
                {
                    if (!std::isfinite(point_lidar.x) ||
                        !std::isfinite(point_lidar.y) ||
                        !std::isfinite(point_lidar.z))
                    {
                        continue;
                    }
                    const Eigen::Vector4d point_map = scan.T_map_lidar *
                        Eigen::Vector4d(
                            point_lidar.x, point_lidar.y, point_lidar.z, 1.0);
                    if (point_map.allFinite())
                    {
                        const double point_height = ground_plane_.height(
                            point_map.x(), point_map.y(), point_map.z());
                        const double range = std::hypot(
                            point_map.x() - scan.T_map_lidar(0, 3),
                            point_map.y() - scan.T_map_lidar(1, 3));
                        if (is_ground_free_height(
                                point_height, options_.ground_free_min_height,
                                options_.thre_z_min) &&
                            range <= options_.raytrace_max_range)
                        {
                            update_bounds(point_map.x(), point_map.y());
                        }
                    }
                }
            }
        }
        if (x_min > x_max || y_min > y_max)
        {
            RCLCPP_ERROR(rclcpp::get_logger("pcd2grid"), "Map contains no finite XY points");
            return false;
        }

        msg.info.origin.position.x = x_min;
        msg.info.origin.position.y = y_min;
        msg.info.origin.position.z = 0.0;
        msg.info.origin.orientation.w = 1.0;

        const double width_value =
            std::floor((x_max - x_min) / options_.map_resolution) + 1.0;
        const double height_value =
            std::floor((y_max - y_min) / options_.map_resolution) + 1.0;
        // The traversal and OpenCV writer use signed int indices.  Reject
        // malformed or physically unrepresentable extents before converting
        // to uint32_t or allocating the cell evidence arrays.
        const double max_dimension = static_cast<double>(
            std::numeric_limits<int>::max());
        if (!std::isfinite(width_value) || !std::isfinite(height_value) ||
            width_value < 1.0 || height_value < 1.0 ||
            width_value > max_dimension || height_value > max_dimension)
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("pcd2grid"),
                "Map extent is invalid or too large for grid indexing: "
                "width=%.3f height=%.3f resolution=%.6f",
                width_value, height_value, options_.map_resolution);
            return false;
        }
        const uint64_t width = static_cast<uint64_t>(width_value);
        const uint64_t height = static_cast<uint64_t>(height_value);
        if (height != 0U && width >
                std::numeric_limits<size_t>::max() / height)
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("pcd2grid"),
                "Map extent overflows the host allocation size: "
                "width=%llu height=%llu",
                static_cast<unsigned long long>(width),
                static_cast<unsigned long long>(height));
            return false;
        }
        msg.info.width = static_cast<uint32_t>(width);
        msg.info.height = static_cast<uint32_t>(height);
        const size_t cell_count = static_cast<size_t>(width) * height;
        msg.data.assign(cell_count, kGridUnknown);

        std::vector<uint32_t> raw_counts(cell_count, 0U);
        std::vector<uint32_t> height_filtered_counts(cell_count, 0U);
        std::vector<uint32_t> obstacle_counts(cell_count, 0U);
        // Store heights above the local fitted floor, not absolute map.z. This
        // keeps a gently tilted FAST-LIO map from turning its floor into walls.
        std::vector<float> obstacle_min_height(
            cell_count, std::numeric_limits<float>::max());
        std::vector<float> obstacle_max_height(
            cell_count, std::numeric_limits<float>::lowest());
        std::vector<uint8_t> ground_free(cell_count, 0U);
        std::vector<uint8_t> ray_free(cell_count, 0U);
        // A candidate cell contains enough height-filtered returns to need a
        // final occupied/overhead decision.  The semantic occupied mask is
        // kept separate from the ray-occlusion mask below: a high/mixed
        // candidate may ultimately be classified as overhead/free.  Only a
        // candidate already confirmed as a low obstacle is a hard ray blocker;
        // high candidates remain traversable for crossing evidence.
        std::vector<uint8_t> candidate_mask(cell_count, 0U);
        std::vector<uint8_t> occupied_mask(cell_count, 0U);
        // A low-obstacle candidate is a hard occlusion boundary.  This is
        // separate from occupied_mask only to make the ray policy explicit;
        // a high/mixed candidate may be crossed to prove an overhead return,
        // but free-space marking stops after the first candidate so a single
        // projected wall/lintel return cannot paint a long corridor.
        std::vector<uint8_t> ray_occlusion_mask(cell_count, 0U);
        std::vector<uint8_t> overhead_mask(cell_count, 0U);
        std::vector<uint32_t> obstacle_low_counts(cell_count, 0U);
        // 低点出现的帧数(按关键帧去重):用于"零星低点"与"多帧持续低点"
        // 的区分,后者即使占比低也保持占用。
        std::vector<uint32_t> obstacle_low_frame_counts(cell_count, 0U);
        std::vector<uint32_t> low_frame_stamp(cell_count, 0U);
        uint32_t low_frame_generation = 0U;
        std::vector<uint32_t> obstacle_frame_counts(cell_count, 0U);
        std::vector<uint32_t> ground_frame_counts(cell_count, 0U);
        std::vector<uint32_t> ray_frame_counts(cell_count, 0U);
        // Number of distinct keyframes whose ground ray continued beyond an
        // obstacle cell.  This is stronger than ray_free (which only records
        // cells before an obstacle) and is used to prove that a high return is
        // overhead rather than a wall hit.
        std::vector<uint32_t> overhead_ray_cross_frame_counts(
            cell_count, 0U);
        std::vector<uint32_t> overhead_ray_cross_stamp(cell_count, 0U);
        std::vector<std::vector<float>> obstacle_heights;
        if (!diagnostics_file_name.empty())
        {
            obstacle_heights.resize(cell_count);
        }

        auto cell_index = [&](double x, double y, int &i, int &j, size_t &index)
        {
            if (!std::isfinite(x) || !std::isfinite(y))
            {
                return false;
            }
            i = static_cast<int>(std::floor((x - x_min) / options_.map_resolution));
            j = static_cast<int>(std::floor((y - y_min) / options_.map_resolution));
            if (i < 0 || i >= static_cast<int>(msg.info.width) ||
                j < 0 || j >= static_cast<int>(msg.info.height))
            {
                return false;
            }
            index = static_cast<size_t>(i) +
                static_cast<size_t>(j) * msg.info.width;
            return true;
        };

        for (const auto &point : map_points->points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                continue;
            }
            int i = 0, j = 0;
            size_t index = 0;
            if (!cell_index(point.x, point.y, i, j, index))
            {
                continue;
            }
            raw_counts[index]++;
            if (options_.ground_free_enabled &&
                is_ground_free_height(
                    ground_plane_.height(point.x, point.y, point.z),
                    options_.ground_free_min_height, options_.thre_z_min))
            {
                ground_free[index] = 1U;
            }
        }
        if (height_filtered_points)
        {
            for (const auto &point : height_filtered_points->points)
            {
                if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                    !std::isfinite(point.z))
                {
                    continue;
                }
                int i = 0, j = 0;
                size_t index = 0;
                if (cell_index(point.x, point.y, i, j, index))
                {
                    height_filtered_counts[index]++;
                }
            }
        }
        if (obstacle_points)
        {
            for (const auto &point : obstacle_points->points)
            {
                if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                    !std::isfinite(point.z))
                {
                    continue;
                }
                int i = 0, j = 0;
                size_t index = 0;
                if (!cell_index(point.x, point.y, i, j, index))
                {
                    continue;
                }
                obstacle_counts[index]++;
                const float point_height = static_cast<float>(
                    ground_plane_.height(point.x, point.y, point.z));
                if (static_cast<double>(point_height) <=
                    options_.overhead_clearance_z)
                {
                    obstacle_low_counts[index]++;
                }
                obstacle_min_height[index] = std::min(
                    obstacle_min_height[index], point_height);
                obstacle_max_height[index] = std::max(
                    obstacle_max_height[index], point_height);
                if (!obstacle_heights.empty())
                {
                    obstacle_heights[index].push_back(point_height);
                }
            }
        }

        const uint32_t min_points_occupied = static_cast<uint32_t>(
            std::max(0, options_.min_points_occupied));
        const uint32_t min_low_points_gate = static_cast<uint32_t>(
            std::max(0, options_.min_low_points_occupied));
        // 低点帧数统计:逐关键帧扫描,落在障碍带 [crossable,
        // overhead_clearance_z] 内的点按格去重计数(一帧一格一次)。
        // 仅在低点门限开启时统计;关闭时不产生任何额外开销。
        if (options_.low_point_gate_enabled)
        for (const auto &scan : scans)
        {
            if (!scan.cloud_lidar)
            {
                continue;
            }
            ++low_frame_generation;
            const Eigen::Matrix4f T_map_lidar_f =
                scan.T_map_lidar.cast<float>();
            for (const auto &point : scan.cloud_lidar->points)
            {
                if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                    !std::isfinite(point.z))
                {
                    continue;
                }
                const Eigen::Vector4f point_map =
                    T_map_lidar_f *
                    Eigen::Vector4f(point.x, point.y, point.z, 1.0F);
                const float point_height = static_cast<float>(
                    ground_plane_.height(point_map.x(), point_map.y(),
                        point_map.z()));
                if (point_height < options_.thre_z_min ||
                    point_height > options_.overhead_clearance_z)
                {
                    continue;
                }
                int low_i = 0, low_j = 0;
                size_t low_index = 0;
                if (!cell_index(point_map.x(), point_map.y(), low_i, low_j,
                        low_index))
                {
                    continue;
                }
                if (low_frame_stamp[low_index] != low_frame_generation)
                {
                    low_frame_stamp[low_index] = low_frame_generation;
                    obstacle_low_frame_counts[low_index]++;
                }
            }
        }
        // With keyframe scans available, a ray records every high/mixed
        // candidate that it crosses, but stops at the first confirmed low
        // obstacle.  Free marking is limited to the segment before the first
        // candidate, so crossing evidence remains plentiful without allowing
        // a projected wall/lintel to paint a long free corridor.  When scans
        // are absent, retain the legacy aggregate-PCD behavior because no
        // visibility evidence can be computed.
        const bool overhead_visibility_gate_active =
            options_.low_point_gate_enabled && !scans.empty();
        for (size_t index = 0; index < cell_count; ++index)
        {
            if (obstacle_counts[index] < min_points_occupied)
            {
                continue;
            }
            candidate_mask[index] = 1U;
            bool low_blocks;
            if (overhead_visibility_gate_active)
            {
                // In the keyframe visibility mode, defer the high/mixed-cell
                // decision until rays have been traced.  A low-point profile
                // that independently meets the obstacle gate is the only
                // preliminary blocker; all other candidates remain
                // traversable so crossing evidence can be collected.
                low_blocks = low_points_form_obstacle(
                    obstacle_low_counts[index], obstacle_counts[index],
                    obstacle_low_frame_counts[index], min_low_points_gate,
                    options_.overhead_min_low_ratio,
                    static_cast<uint32_t>(
                        options_.overhead_low_frame_persistence));
            }
            else if (options_.low_point_gate_enabled)
            {
                // Aggregate-only fallback: preserve the pre-visibility gate
                // behavior when no keyframe poses/clouds are available.
                low_blocks = low_points_block_cell(
                    obstacle_low_counts[index], obstacle_counts[index],
                    obstacle_low_frame_counts[index], true,
                    min_low_points_gate, options_.overhead_min_low_ratio,
                    static_cast<uint32_t>(
                        options_.overhead_low_frame_persistence));
            }
            else
            {
                // 关闭低点门限:恢复旧"最低点 <= 净高即占用"判定。
                low_blocks = is_overhead_height(
                    obstacle_min_height[index], options_.overhead_clearance_z);
            }
            if (!low_blocks)
            {
                overhead_mask[index] = 1U;
            }
            else
            {
                occupied_mask[index] = 1U;
            }
            if (overhead_visibility_gate_active)
            {
                // Only a profile that independently meets the low-obstacle
                // gate may terminate a ray.  High/mixed candidates are still
                // visited so the ground endpoint can provide crossing
                // evidence for the final overhead decision.
                ray_occlusion_mask[index] = low_blocks ? 1U : 0U;
            }
        }

        size_t valid_ray_scans = 0;
        size_t ground_ray_endpoints = 0;
        size_t traced_rays = 0;
        size_t obstacle_blocked_rays = 0;
        if (!scans.empty())
        {
            std::vector<uint32_t> endpoint_seen(cell_count, 0U);
            std::vector<uint32_t> obstacle_frame_seen(cell_count, 0U);
            std::vector<uint32_t> ground_frame_seen(cell_count, 0U);
            std::vector<uint32_t> ray_frame_seen(cell_count, 0U);
            uint32_t scan_token = 0U;
            for (const auto &scan : scans)
            {
                if (!scan.T_map_lidar.allFinite() || !scan.cloud_lidar ||
                    scan.cloud_lidar->empty())
                {
                    continue;
                }
                int origin_i = 0, origin_j = 0;
                size_t origin_index = 0;
                const double origin_x = scan.T_map_lidar(0, 3);
                const double origin_y = scan.T_map_lidar(1, 3);
                if (!cell_index(
                        origin_x, origin_y, origin_i, origin_j, origin_index))
                {
                    continue;
                }
                scan_token++;
                if (options_.raytrace_free_enabled)
                {
                    valid_ray_scans++;
                }
                for (const auto &point_lidar : scan.cloud_lidar->points)
                {
                    if (!std::isfinite(point_lidar.x) ||
                        !std::isfinite(point_lidar.y) ||
                        !std::isfinite(point_lidar.z))
                    {
                        continue;
                    }
                    const Eigen::Vector4d point_map = scan.T_map_lidar *
                        Eigen::Vector4d(
                            point_lidar.x, point_lidar.y, point_lidar.z, 1.0);
                    if (!point_map.allFinite())
                    {
                        continue;
                    }
                    int endpoint_i = 0, endpoint_j = 0;
                    size_t endpoint_index = 0;
                    if (!cell_index(
                            point_map.x(), point_map.y(), endpoint_i,
                            endpoint_j, endpoint_index))
                    {
                        continue;
                    }
                    const double point_height = ground_plane_.height(
                        point_map.x(), point_map.y(), point_map.z());
                    const bool is_ground = is_ground_free_height(
                        point_height, options_.ground_free_min_height,
                        options_.thre_z_min);
                    if (options_.ground_free_enabled && is_ground &&
                        ground_frame_seen[endpoint_index] != scan_token)
                    {
                        ground_frame_seen[endpoint_index] = scan_token;
                        ground_frame_counts[endpoint_index]++;
                    }
                    const bool inside_obstacle_band =
                        point_height >= options_.thre_z_min &&
                        point_height <= options_.thre_z_max;
                    const bool passes_height_filter =
                        inside_obstacle_band !=
                        (options_.flag_pass_through != 0);
                    if (passes_height_filter &&
                        obstacle_frame_seen[endpoint_index] != scan_token)
                    {
                        obstacle_frame_seen[endpoint_index] = scan_token;
                        obstacle_frame_counts[endpoint_index]++;
                    }
                    if ((!options_.raytrace_free_enabled &&
                            !overhead_visibility_gate_active) || !is_ground)
                    {
                        continue;
                    }
                    const double dx = point_map.x() - origin_x;
                    const double dy = point_map.y() - origin_y;
                    if (std::hypot(dx, dy) > options_.raytrace_max_range)
                    {
                        continue;
                    }
                    if (endpoint_seen[endpoint_index] == scan_token)
                    {
                        continue;
                    }
                    endpoint_seen[endpoint_index] = scan_token;
                    ground_ray_endpoints++;
                    bool first_cell = true;
                    bool blocked = false;
                    // Once a ray has crossed the first high/mixed candidate,
                    // keep tracing only to collect crossing evidence.  Do
                    // not mark cells behind that candidate as free; this
                    // preserves the map boundary without reintroducing long
                    // free streaks.  A confirmed low obstacle terminates the
                    // ray immediately.
                    bool free_marking_open = true;
                    trace_grid_line(
                        origin_i, origin_j, endpoint_i, endpoint_j,
                        [&](int i, int j)
                        {
                            const size_t index = static_cast<size_t>(i) +
                                static_cast<size_t>(j) * msg.info.width;
                            if (overhead_visibility_gate_active &&
                                !first_cell && candidate_mask[index] != 0U)
                            {
                                // The endpoint is a ground return beyond this
                                // candidate.  That is useful visibility
                                // evidence.  A high/mixed candidate is not a
                                // hard blocker, but it closes free marking for
                                // the remainder of this ray.  A low candidate
                                // is a real obstacle and terminates the ray.
                                if (index != endpoint_index &&
                                    overhead_ray_cross_stamp[index] !=
                                        scan_token)
                                {
                                    overhead_ray_cross_stamp[index] = scan_token;
                                    overhead_ray_cross_frame_counts[index]++;
                                }
                                if (ray_occlusion_mask[index] != 0U &&
                                    index != endpoint_index)
                                {
                                    blocked = true;
                                    return false;
                                }
                                if (index != endpoint_index)
                                {
                                    free_marking_open = false;
                                }
                            }
                            first_cell = false;
                            if (options_.raytrace_free_enabled &&
                                free_marking_open)
                            {
                                ray_free[index] = 1U;
                                if (ray_frame_seen[index] != scan_token)
                                {
                                    ray_frame_seen[index] = scan_token;
                                    ray_frame_counts[index]++;
                                }
                            }
                            return true;
                        });
                    traced_rays++;
                    if (blocked)
                    {
                        obstacle_blocked_rays++;
                    }
                }
            }
        }

        int occupied_cells = 0, free_cells = 0, unknown_cells = 0;
        int overhead_cells = 0, ground_evidence_cells = 0, ray_evidence_cells = 0;
        int sparse_free_cells = 0;
        int unknown_no_raw = 0, unknown_height_filtered = 0;
        int unknown_radius_filtered = 0;
        size_t low_obstacle_cells = 0;
        size_t occupied_with_ground = 0;
        size_t occupied_with_ray = 0;
        size_t occupied_with_ground_and_ray = 0;
        size_t occupied_low_p90 = 0;
        size_t occupied_low_p90_with_free = 0;
        size_t occupied_low_p95_strong_free = 0;
        size_t mixed_overhead_cleared = 0;
        double mixed_overhead_min_x = 0.0;
        double mixed_overhead_max_x = 0.0;
        double mixed_overhead_min_y = 0.0;
        double mixed_overhead_max_y = 0.0;
        size_t overhead_visibility_candidates = 0U;
        size_t overhead_visibility_accepted = 0U;
        size_t overhead_visibility_rejected_low = 0U;
        size_t overhead_visibility_rejected_no_crossing = 0U;
        size_t mixed_visibility_accepted = 0U;
        std::array<size_t, 6> occupied_p90_bins{};
        std::array<size_t, 5> occupied_frame_bins{};
        std::vector<float> obstacle_p50(
            cell_count, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> obstacle_p90(
            cell_count, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> obstacle_p95(
            cell_count, std::numeric_limits<float>::quiet_NaN());
        auto percentile = [](const std::vector<float> &values, double ratio)
        {
            if (values.empty())
            {
                return std::numeric_limits<float>::quiet_NaN();
            }
            const double position = ratio *
                static_cast<double>(values.size() - 1U);
            const size_t lower = static_cast<size_t>(std::floor(position));
            const size_t upper = static_cast<size_t>(std::ceil(position));
            const double weight = position - static_cast<double>(lower);
            return static_cast<float>(
                values[lower] * (1.0 - weight) + values[upper] * weight);
        };
        for (size_t index = 0; index < obstacle_heights.size(); ++index)
        {
            auto &heights = obstacle_heights[index];
            if (heights.empty())
            {
                continue;
            }
            std::sort(heights.begin(), heights.end());
            obstacle_p50[index] = percentile(heights, 0.50);
            obstacle_p90[index] = percentile(heights, 0.90);
            obstacle_p95[index] = percentile(heights, 0.95);
        }
        struct LowObstacleSample
        {
            double x = 0.0;
            double y = 0.0;
            double min_height = 0.0;
            double max_height = 0.0;
            uint32_t points = 0U;
        };
        std::vector<LowObstacleSample> low_obstacle_samples;
        low_obstacle_samples.reserve(8);

        for (size_t index = 0; index < cell_count; ++index)
        {
            const bool has_free_evidence =
                ground_free[index] != 0U || ray_free[index] != 0U;
            if (overhead_visibility_gate_active &&
                obstacle_counts[index] >= min_points_occupied)
            {
                ++overhead_visibility_candidates;
                const bool low_obstacle = low_points_form_obstacle(
                    obstacle_low_counts[index], obstacle_counts[index],
                    obstacle_low_frame_counts[index], min_low_points_gate,
                    options_.overhead_min_low_ratio,
                    static_cast<uint32_t>(
                        options_.overhead_low_frame_persistence));
                const bool crossed = overhead_ray_crossing_sufficient(
                    overhead_ray_cross_frame_counts[index],
                    obstacle_frame_counts[index],
                    static_cast<uint32_t>(
                        options_.overhead_min_ray_crossing_frames),
                    options_.overhead_min_ray_crossing_ratio);
                if (low_obstacle)
                {
                    ++overhead_visibility_rejected_low;
                    msg.data[index] = kGridOccupied;
                }
                else if (crossed)
                {
                    ++overhead_visibility_accepted;
                    if (obstacle_low_counts[index] > 0U)
                    {
                        ++mixed_visibility_accepted;
                    }
                    overhead_mask[index] = 1U;
                    msg.data[index] = kGridFree;
                }
                else
                {
                    ++overhead_visibility_rejected_no_crossing;
                    // No crossing is not proof of a wall: a pure high return
                    // (and a weak mixed return with normal frame/free-space
                    // evidence) should retain the established low-point
                    // classifier result.  The ray-occlusion mask above keeps
                    // this fallback from creating long free corridors.
                    msg.data[index] = classify_pcd2grid_cell_height(
                        obstacle_counts[index], obstacle_low_counts[index],
                        obstacle_low_frame_counts[index],
                        options_.overhead_clearance_z, has_free_evidence,
                        min_points_occupied, obstacle_frame_counts[index],
                        ground_frame_counts[index], ray_frame_counts[index],
                        static_cast<uint32_t>(options_.min_obstacle_frames),
                        static_cast<uint32_t>(options_.min_free_frames),
                        options_.free_to_obstacle_frame_ratio,
                        min_low_points_gate, options_.overhead_min_low_ratio,
                        static_cast<uint32_t>(
                            options_.overhead_low_frame_persistence));
                    if (msg.data[index] == kGridFree)
                    {
                        overhead_mask[index] = 1U;
                    }
                }
            }
            else if (options_.low_point_gate_enabled)
            {
                msg.data[index] = classify_pcd2grid_cell_height(
                    obstacle_counts[index], obstacle_low_counts[index],
                    obstacle_low_frame_counts[index],
                    options_.overhead_clearance_z,
                    has_free_evidence, min_points_occupied,
                    obstacle_frame_counts[index], ground_frame_counts[index],
                    ray_frame_counts[index],
                    static_cast<uint32_t>(options_.min_obstacle_frames),
                    static_cast<uint32_t>(options_.min_free_frames),
                    options_.free_to_obstacle_frame_ratio,
                    min_low_points_gate, options_.overhead_min_low_ratio,
                    static_cast<uint32_t>(
                        options_.overhead_low_frame_persistence));
            }
            else
            {
                // 关闭低点门限:恢复旧"最低点"判定路径。
                msg.data[index] = classify_pcd2grid_cell_height(
                    obstacle_counts[index], obstacle_min_height[index],
                    options_.overhead_clearance_z,
                    has_free_evidence, min_points_occupied,
                    obstacle_frame_counts[index], ground_frame_counts[index],
                    ray_frame_counts[index],
                    static_cast<uint32_t>(options_.min_obstacle_frames),
                    static_cast<uint32_t>(options_.min_free_frames),
                    options_.free_to_obstacle_frame_ratio);
            }
            if (msg.data[index] == kGridFree &&
                obstacle_counts[index] >= min_points_occupied &&
                obstacle_low_counts[index] > 0U)
            {
                // 零星低点+主体悬空的混合格被放行:记入数量与坐标范围,
                // 供门口/过梁场景直接核对,无需手工翻证据 CSV。
                ++mixed_overhead_cleared;
                const uint32_t mixed_i = static_cast<uint32_t>(
                    index % msg.info.width);
                const uint32_t mixed_j = static_cast<uint32_t>(
                    index / msg.info.width);
                const double mixed_x = x_min + (static_cast<double>(mixed_i) + 0.5) *
                    options_.map_resolution;
                const double mixed_y = y_min + (static_cast<double>(mixed_j) + 0.5) *
                    options_.map_resolution;
                if (mixed_overhead_cleared == 1U)
                {
                    mixed_overhead_min_x = mixed_overhead_max_x = mixed_x;
                    mixed_overhead_min_y = mixed_overhead_max_y = mixed_y;
                }
                else
                {
                    mixed_overhead_min_x = std::min(mixed_overhead_min_x, mixed_x);
                    mixed_overhead_max_x = std::max(mixed_overhead_max_x, mixed_x);
                    mixed_overhead_min_y = std::min(mixed_overhead_min_y, mixed_y);
                    mixed_overhead_max_y = std::max(mixed_overhead_max_y, mixed_y);
                }
            }
            if (msg.data[index] == kGridOccupied)
            {
                occupied_cells++;
                const bool has_ground = ground_free[index] != 0U;
                const bool has_ray = ray_free[index] != 0U;
                occupied_with_ground += has_ground ? 1U : 0U;
                occupied_with_ray += has_ray ? 1U : 0U;
                occupied_with_ground_and_ray += has_ground && has_ray ? 1U : 0U;
                const float p90 = obstacle_p90[index];
                const float p95 = obstacle_p95[index];
                if (std::isfinite(p90))
                {
                    const size_t p90_bin = p90 <= 0.15F ? 0U :
                        p90 <= 0.20F ? 1U : p90 <= 0.25F ? 2U :
                        p90 <= 0.35F ? 3U : p90 <= 0.55F ? 4U : 5U;
                    occupied_p90_bins[p90_bin]++;
                    if (p90 <= options_.low_obstacle_diagnostic_height)
                    {
                        occupied_low_p90++;
                        occupied_low_p90_with_free +=
                            has_free_evidence ? 1U : 0U;
                    }
                }
                if (std::isfinite(p95) &&
                    p95 <= options_.low_obstacle_diagnostic_height &&
                    ground_frame_counts[index] >= 2U &&
                    ray_frame_counts[index] >= 2U &&
                    ground_frame_counts[index] >=
                        2U * std::max(1U, obstacle_frame_counts[index]))
                {
                    occupied_low_p95_strong_free++;
                }
                const uint32_t obstacle_frames = obstacle_frame_counts[index];
                const size_t frame_bin = obstacle_frames == 0U ? 0U :
                    obstacle_frames == 1U ? 1U : obstacle_frames == 2U ? 2U :
                    obstacle_frames <= 4U ? 3U : 4U;
                occupied_frame_bins[frame_bin]++;
                if (obstacle_max_height[index] <=
                    options_.low_obstacle_diagnostic_height)
                {
                    low_obstacle_cells++;
                    if (low_obstacle_samples.size() < 8U)
                    {
                        const uint32_t i = static_cast<uint32_t>(
                            index % msg.info.width);
                        const uint32_t j = static_cast<uint32_t>(
                            index / msg.info.width);
                        low_obstacle_samples.push_back({
                            x_min + (static_cast<double>(i) + 0.5) *
                                options_.map_resolution,
                            y_min + (static_cast<double>(j) + 0.5) *
                                options_.map_resolution,
                            obstacle_min_height[index],
                            obstacle_max_height[index],
                            obstacle_counts[index]});
                    }
                }
            }
            else if (msg.data[index] == kGridFree)
            {
                free_cells++;
                if (overhead_mask[index] != 0U)
                {
                    overhead_cells++;
                }
                if (ground_free[index] != 0U)
                {
                    ground_evidence_cells++;
                }
                if (ray_free[index] != 0U)
                {
                    ray_evidence_cells++;
                }
                if (obstacle_counts[index] > 0U &&
                    obstacle_counts[index] < min_points_occupied)
                {
                    sparse_free_cells++;
                }
            }
            else
            {
                unknown_cells++;
                if (raw_counts[index] == 0U)
                {
                    unknown_no_raw++;
                }
                else if (height_filtered_counts[index] == 0U)
                {
                    unknown_height_filtered++;
                }
                else if (obstacle_counts[index] == 0U)
                {
                    unknown_radius_filtered++;
                }
            }
        }

        RCLCPP_INFO(rclcpp::get_logger("rclcpp"),
            "Grid %dx%d, cells: occupied=%d free=%d unknown=%d "
            "(overhead-cleared=%d sparse-cleared=%d ground-evidence=%d "
            "ray-evidence=%d, "
            "clearance_above_floor=%.2fm floor_z_map=%.3fm)",
            msg.info.width, msg.info.height, occupied_cells, free_cells, unknown_cells,
            overhead_cells, sparse_free_cells, ground_evidence_cells,
            ray_evidence_cells,
            options_.overhead_clearance_z, options_.floor_z_map);
        if (mixed_overhead_cleared > 0U)
        {
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"),
                "Mixed-cell overhead-cleared: count=%zu bbox_x=[%.2f,%.2f]m "
                "bbox_y=[%.2f,%.2f]m (低点占比<%.2f 或 <%u 个的门槛/门楣混合格已放行)",
                mixed_overhead_cleared,
                mixed_overhead_min_x, mixed_overhead_max_x,
                mixed_overhead_min_y, mixed_overhead_max_y,
                options_.overhead_min_low_ratio, min_low_points_gate);
        }
        if (overhead_visibility_gate_active)
        {
            RCLCPP_INFO(
                rclcpp::get_logger("pcd2grid"),
                "Overhead visibility decision: candidates=%zu accepted=%zu "
                "mixed_accepted=%zu rejected_low=%zu rejected_no_crossing=%zu "
                "thresholds=[frames>=%d ratio>=%.3f]",
                overhead_visibility_candidates,
                overhead_visibility_accepted, mixed_visibility_accepted,
                overhead_visibility_rejected_low,
                overhead_visibility_rejected_no_crossing,
                options_.overhead_min_ray_crossing_frames,
                options_.overhead_min_ray_crossing_ratio);
        }
        RCLCPP_INFO(
            rclcpp::get_logger("pcd2grid"),
            "Free-space raytrace: scans=%zu/%zu ground_endpoints=%zu traced=%zu "
            "blocked_by_occupied=%zu; unknown reasons: no_raw_xy=%d "
            "height_filtered=%d radius_filtered=%d",
            valid_ray_scans, scans.size(), ground_ray_endpoints, traced_rays,
            obstacle_blocked_rays, unknown_no_raw, unknown_height_filtered,
            unknown_radius_filtered);
        if (!diagnostics_file_name.empty())
        {
            RCLCPP_INFO(
                rclcpp::get_logger("pcd2grid"),
                "Occupied evidence conflicts: total=%d ground=%zu ray=%zu "
                "ground_and_ray=%zu; p90_height_bins="
                "[<=0.15:%zu <=0.20:%zu <=0.25:%zu <=0.35:%zu "
                "<=0.55:%zu >0.55:%zu]; obstacle_frame_bins="
                "[0:%zu 1:%zu 2:%zu 3-4:%zu >=5:%zu]",
                occupied_cells, occupied_with_ground, occupied_with_ray,
                occupied_with_ground_and_ray,
                occupied_p90_bins[0], occupied_p90_bins[1],
                occupied_p90_bins[2], occupied_p90_bins[3],
                occupied_p90_bins[4], occupied_p90_bins[5],
                occupied_frame_bins[0], occupied_frame_bins[1],
                occupied_frame_bins[2], occupied_frame_bins[3],
                occupied_frame_bins[4]);
            RCLCPP_INFO(
                rclcpp::get_logger("pcd2grid"),
                "Occupied low-height shadow metrics: p90<=%.2fm=%zu "
                "with_free=%zu; p95<=%.2fm with ground_frames>=2, "
                "ray_frames>=2 and ground_frames>=2*obstacle_frames=%zu",
                options_.low_obstacle_diagnostic_height,
                occupied_low_p90, occupied_low_p90_with_free,
                options_.low_obstacle_diagnostic_height,
                occupied_low_p95_strong_free);

            std::ofstream diagnostics(
                diagnostics_file_name, std::ios::out | std::ios::trunc);
            if (!diagnostics.is_open())
            {
                RCLCPP_ERROR(
                    rclcpp::get_logger("pcd2grid"),
                    "Cannot open required cell evidence diagnostics: %s",
                    diagnostics_file_name.c_str());
                return false;
            }

            diagnostics << std::setprecision(9)
                << "cell_i,cell_j,map_x,map_y,class,raw_points,"
                   "height_filtered_points,obstacle_points,obstacle_low_points,min_height,"
                   "p50_height,p90_height,p95_height,max_height,"
                   "ground_free,ray_free,obstacle_frames,ground_frames,"
                   "ray_frames,overhead_ray_cross_frames\n";
            size_t diagnostic_rows = 0;
            for (size_t index = 0; index < cell_count; ++index)
            {
                if (raw_counts[index] == 0U &&
                    ground_free[index] == 0U && ray_free[index] == 0U)
                {
                    continue;
                }
                const uint32_t i = static_cast<uint32_t>(
                    index % msg.info.width);
                const uint32_t j = static_cast<uint32_t>(
                    index / msg.info.width);
                const bool has_obstacle_points =
                    obstacle_counts[index] > 0U;
                const float no_height =
                    std::numeric_limits<float>::quiet_NaN();
                diagnostics
                    << i << ',' << j << ','
                    << x_min + (static_cast<double>(i) + 0.5) *
                        options_.map_resolution << ','
                    << y_min + (static_cast<double>(j) + 0.5) *
                        options_.map_resolution << ','
                    << static_cast<int>(msg.data[index]) << ','
                    << raw_counts[index] << ','
                    << height_filtered_counts[index] << ','
                    << obstacle_counts[index] << ','
                    << obstacle_low_counts[index] << ','
                    << (has_obstacle_points
                        ? obstacle_min_height[index] : no_height) << ','
                    << obstacle_p50[index] << ','
                    << obstacle_p90[index] << ','
                    << obstacle_p95[index] << ','
                    << (has_obstacle_points
                        ? obstacle_max_height[index] : no_height) << ','
                    << static_cast<int>(ground_free[index]) << ','
                    << static_cast<int>(ray_free[index]) << ','
                    << obstacle_frame_counts[index] << ','
                    << ground_frame_counts[index] << ','
                    << ray_frame_counts[index] << ','
                    << overhead_ray_cross_frame_counts[index] << '\n';
                diagnostic_rows++;
            }
            diagnostics.flush();
            if (!diagnostics.good())
            {
                RCLCPP_ERROR(
                    rclcpp::get_logger("pcd2grid"),
                    "Failed while writing required cell evidence diagnostics: %s",
                    diagnostics_file_name.c_str());
                return false;
            }
            RCLCPP_INFO(
                rclcpp::get_logger("pcd2grid"),
                "Cell evidence diagnostics saved: %s (%zu rows)",
                diagnostics_file_name.c_str(), diagnostic_rows);
        }
        if (low_obstacle_cells > 0U)
        {
            RCLCPP_WARN(
                rclcpp::get_logger("pcd2grid"),
                "Low-only occupied candidates=%zu (all obstacle points <= %.2fm "
                "above floor); inspect these cells before changing floor_z_map",
                low_obstacle_cells, options_.low_obstacle_diagnostic_height);
            for (const auto &sample : low_obstacle_samples)
            {
                RCLCPP_WARN(
                    rclcpp::get_logger("pcd2grid"),
                    "Low obstacle candidate map_xy=[%.2f,%.2f] "
                    "height=[%.3f,%.3f]m points=%u",
                    sample.x, sample.y, sample.min_height,
                    sample.max_height, sample.points);
            }
        }
        return true;
    }

    bool Pcd2Grid::SavePGMAndYAML(const nav_msgs::msg::OccupancyGrid &msg, const std::string &name)
    {
        int width = msg.info.width;
        int height = msg.info.height;
        if (width <= 0 || height <= 0)
        {
            RCLCPP_ERROR(rclcpp::get_logger("pcd2grid"), "Refusing to save an invalid occupancy grid");
            return false;
        }
        const size_t width_size = static_cast<size_t>(width);
        const size_t height_size = static_cast<size_t>(height);
        if (height_size != 0U &&
            width_size > std::numeric_limits<size_t>::max() / height_size)
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("pcd2grid"),
                "Refusing to save an occupancy grid whose dimensions overflow host size");
            return false;
        }
        const size_t expected_cells = width_size * height_size;
        if (msg.data.size() != expected_cells)
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("pcd2grid"),
                "Refusing to save an occupancy grid with mismatched data size: "
                "expected=%zu actual=%zu",
                expected_cells, msg.data.size());
            return false;
        }
        if (!std::isfinite(msg.info.resolution) || msg.info.resolution <= 0.0 ||
            !std::isfinite(msg.info.origin.position.x) ||
            !std::isfinite(msg.info.origin.position.y) ||
            !std::isfinite(msg.info.origin.position.z) ||
            !std::isfinite(msg.info.origin.orientation.x) ||
            !std::isfinite(msg.info.origin.orientation.y) ||
            !std::isfinite(msg.info.origin.orientation.z) ||
            !std::isfinite(msg.info.origin.orientation.w))
        {
            RCLCPP_ERROR(
                rclcpp::get_logger("pcd2grid"),
                "Refusing to save an occupancy grid with non-finite metadata");
            return false;
        }
        if (name.empty())
        {
            RCLCPP_ERROR(rclcpp::get_logger("pcd2grid"),
                         "Refusing to save an occupancy grid with an empty path");
            return false;
        }

        // Save PGM
        cv::Mat image(height, width, CV_8UC1);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int8_t data = msg.data[x + y * width];
                if (data == -1)
                {
                    image.at<uchar>(y, x) = 205; // Unknown
                }
                else
                {
                    image.at<uchar>(y, x) = 255 - data * 255 / 100; // Occupied
                }
            }
        }
        cv::flip(image, image, 0); // resolve image mirroring issues
        std::string pgm_file = name + ".pgm";
        try
        {
            if (!cv::imwrite(pgm_file, image))
            {
                RCLCPP_ERROR(rclcpp::get_logger("pcd2grid"), "Failed to save PGM file: %s", pgm_file.c_str());
                return false;
            }
        }
        catch (const cv::Exception &exception)
        {
            RCLCPP_ERROR(rclcpp::get_logger("pcd2grid"), "Failed to save PGM file %s: %s", pgm_file.c_str(), exception.what());
            return false;
        }
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Saved PGM file: %s", pgm_file.c_str());

        // Save YAML
        std::string yaml_file = name + ".yaml";
        std::ofstream yaml_output(yaml_file);
        if (!yaml_output.is_open())
        {
            RCLCPP_ERROR(rclcpp::get_logger("pcd2grid"), "Failed to open YAML file: %s", yaml_file.c_str());
            return false;
        }
        yaml_output << "image: " << std::filesystem::path(pgm_file).filename().string() << std::endl;
        yaml_output << std::setprecision(12)
                    << "resolution: " << msg.info.resolution << std::endl;
        yaml_output << "origin: [" << msg.info.origin.position.x << ", "
                    << msg.info.origin.position.y << ", "
                    << msg.info.origin.position.z << "]" << std::endl;
        yaml_output << "negate: 0" << std::endl;
        yaml_output << "occupied_thresh: 0.65" << std::endl;
        yaml_output << "free_thresh: 0.196" << std::endl;
        yaml_output.close();
        if (!yaml_output)
        {
            RCLCPP_ERROR(rclcpp::get_logger("pcd2grid"), "Failed while writing YAML file: %s", yaml_file.c_str());
            return false;
        }

        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Saved YAML file: %s", yaml_file.c_str());
        return true;
    }
}
