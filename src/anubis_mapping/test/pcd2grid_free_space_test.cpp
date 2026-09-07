#include <chrono>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "pcd2grid.h"

namespace
{
    anubis_mapping::PointType make_point(float x, float y, float z)
    {
        anubis_mapping::PointType point;
        point.x = x;
        point.y = y;
        point.z = z;
        return point;
    }

    cv::Mat run_grid_for_test(
        const anubis_mapping::Pcd2GridOptions& options,
        const anubis_mapping::CloudPtr& map_points,
        const anubis_mapping::Pcd2GridScans& scans,
        bool& success)
    {
        const auto unique_id = std::chrono::steady_clock::now()
                                   .time_since_epoch()
                                   .count();
        const std::filesystem::path output_dir =
            std::filesystem::temp_directory_path() /
            ("pcd2grid_switch_test_" + std::to_string(unique_id));
        if (!std::filesystem::create_directories(output_dir))
        {
            success = false;
            return {};
        }
        const std::filesystem::path output_prefix = output_dir / "map";
        anubis_mapping::Pcd2Grid converter(options);
        success = converter.run(map_points, output_prefix.string(), scans);
        cv::Mat image;
        if (success)
        {
            image = cv::imread(
                (output_prefix.string() + ".pgm"), cv::IMREAD_GRAYSCALE).clone();
        }
        std::filesystem::remove_all(output_dir);
        return image;
    }
}

TEST(Pcd2GridFreeSpaceTest, RayStopsAtOccupiedCellAndGroundCannotClearIt)
{
    anubis_mapping::Pcd2GridOptions options;
    options.floor_z_map = 0.0;
    options.thre_z_min = 0.2;
    options.thre_z_max = 2.0;
    options.map_resolution = 1.0;
    options.thre_radius = 0.0;
    options.thres_point_count = 0;
    options.ground_free_enabled = true;
    options.ground_free_min_height = -0.1;
    options.raytrace_free_enabled = true;
    options.raytrace_max_range = 10.0;
    options.low_obstacle_diagnostic_height = 0.25;
    // This compact synthetic cloud intentionally has no broad ground-plane
    // coverage; production saves reject that fallback, while this test is
    // specifically exercising ray/free evidence classification.
    options.ground_plane_enabled = false;

    anubis_mapping::CloudPtr map_points(new anubis_mapping::PointCloudType());
    // [2026-08-29] 障碍点改为明确的低矮实体(0.30/0.35m,狗体带内):
    // 旧值 0.5/0.6 恰好跨在 overhead_clearance_z=0.55 两侧,低点计数门限
    // (>=2)按设计将其判为零星低点放行 —— 该边界场景由
    // LowPointGateClearsMixedLintelCells 专门覆盖,这里只测射线停止语义。
    map_points->push_back(make_point(2.0F, 0.0F, 0.30F));
    map_points->push_back(make_point(2.0F, 0.0F, 0.35F));
    map_points->push_back(make_point(2.0F, 0.0F, 0.0F));
    map_points->push_back(make_point(4.0F, 0.0F, 0.0F));

    anubis_mapping::CloudPtr scan_points(new anubis_mapping::PointCloudType());
    scan_points->push_back(make_point(4.0F, 0.0F, 0.0F));
    anubis_mapping::Pcd2GridScan scan;
    scan.cloud_lidar = scan_points;

    anubis_mapping::Pcd2GridScans scans;
    scans.push_back(scan);

    const auto unique_id = std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count();
    const std::filesystem::path output_dir =
        std::filesystem::temp_directory_path() /
        ("pcd2grid_free_space_test_" + std::to_string(unique_id));
    ASSERT_TRUE(std::filesystem::create_directories(output_dir));
    const std::filesystem::path output_prefix = output_dir / "map";

    anubis_mapping::Pcd2Grid converter(options);
    ASSERT_TRUE(converter.run(map_points, output_prefix.string(), scans));

    const cv::Mat image = cv::imread(
        (output_prefix.string() + ".pgm"), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());
    ASSERT_EQ(image.rows, 1);
    ASSERT_EQ(image.cols, 5);
    EXPECT_EQ(image.at<uint8_t>(0, 0), 255U);
    EXPECT_EQ(image.at<uint8_t>(0, 1), 255U);
    EXPECT_EQ(image.at<uint8_t>(0, 2), 0U);
    EXPECT_EQ(image.at<uint8_t>(0, 3), 205U);
    EXPECT_EQ(image.at<uint8_t>(0, 4), 255U);
    EXPECT_TRUE(std::filesystem::exists(
        output_prefix.string() + "_cell_evidence.csv"));

    EXPECT_TRUE(std::filesystem::remove_all(output_dir) > 0U);
}

TEST(Pcd2GridFreeSpaceTest, OverheadCellNeedsTwoRaysThatContinueBehindIt)
{
    anubis_mapping::Pcd2GridOptions options;
    options.floor_z_map = 0.0;
    options.thre_z_min = 0.2;
    options.thre_z_max = 2.0;
    options.overhead_clearance_z = 0.55;
    options.map_resolution = 1.0;
    options.thre_radius = 0.0;
    options.thres_point_count = 0;
    options.min_points_occupied = 2;
    options.ground_free_enabled = true;
    options.ground_free_min_height = -0.1;
    options.raytrace_free_enabled = true;
    options.low_point_gate_enabled = true;
    options.overhead_min_ray_crossing_frames = 2;
    options.overhead_min_ray_crossing_ratio = 0.0;
    options.ground_plane_enabled = false;
    options.cell_evidence_diagnostics_enabled = false;

    anubis_mapping::CloudPtr map_points(new anubis_mapping::PointCloudType());
    // Two high returns in the middle cell represent a lintel/overhead return.
    map_points->push_back(make_point(2.0F, 0.0F, 1.2F));
    map_points->push_back(make_point(2.0F, 0.0F, 1.3F));
    map_points->push_back(make_point(4.0F, 0.0F, 0.0F));

    anubis_mapping::CloudPtr scan_points(new anubis_mapping::PointCloudType());
    scan_points->push_back(make_point(4.0F, 0.0F, 0.0F));
    anubis_mapping::Pcd2GridScan scan;
    scan.cloud_lidar = scan_points;
    anubis_mapping::Pcd2GridScans scans{scan, scan};

    const auto unique_id = std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count();
    const std::filesystem::path output_dir =
        std::filesystem::temp_directory_path() /
        ("pcd2grid_overhead_visibility_test_" + std::to_string(unique_id));
    ASSERT_TRUE(std::filesystem::create_directories(output_dir));
    const std::filesystem::path output_prefix = output_dir / "map";

    anubis_mapping::Pcd2Grid converter(options);
    ASSERT_TRUE(converter.run(map_points, output_prefix.string(), scans));
    const cv::Mat image = cv::imread(
        output_prefix.string() + ".pgm", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());
    ASSERT_EQ(image.rows, 1);
    ASSERT_EQ(image.cols, 5);
    // The cell containing the lintel is cleared only after both scan rays
    // cross it; the endpoint cell remains free from ground evidence.
    EXPECT_EQ(image.at<uint8_t>(0, 2), 255U);
    EXPECT_EQ(image.at<uint8_t>(0, 4), 255U);

    EXPECT_TRUE(std::filesystem::remove_all(output_dir) > 0U);
}

TEST(Pcd2GridFreeSpaceTest, GroundEvidenceSwitchControlsGroundOnlyMap)
{
    anubis_mapping::Pcd2GridOptions options;
    options.floor_z_map = 0.0;
    options.thre_z_min = 0.2;
    options.thre_z_max = 2.0;
    options.map_resolution = 1.0;
    options.thre_radius = 0.0;
    options.thres_point_count = 0;
    options.min_points_occupied = 2;
    options.ground_free_min_height = -0.1;
    options.raytrace_free_enabled = false;
    options.low_obstacle_diagnostic_height = 0.25;
    options.ground_plane_enabled = false;
    options.cell_evidence_diagnostics_enabled = false;

    anubis_mapping::CloudPtr ground(new anubis_mapping::PointCloudType());
    ground->push_back(make_point(0.0F, 0.0F, 0.0F));

    options.ground_free_enabled = true;
    bool success = false;
    const cv::Mat with_ground = run_grid_for_test(options, ground, {}, success);
    ASSERT_TRUE(success);
    ASSERT_EQ(with_ground.rows, 1);
    ASSERT_EQ(with_ground.cols, 1);
    EXPECT_EQ(with_ground.at<uint8_t>(0, 0), 255U);

    options.ground_free_enabled = false;
    const cv::Mat without_ground = run_grid_for_test(options, ground, {}, success);
    ASSERT_TRUE(success);
    ASSERT_EQ(without_ground.rows, 1);
    ASSERT_EQ(without_ground.cols, 1);
    EXPECT_EQ(without_ground.at<uint8_t>(0, 0), 205U);
}

TEST(Pcd2GridFreeSpaceTest, RaytraceSwitchIsIndependentFromGroundEvidence)
{
    anubis_mapping::Pcd2GridOptions options;
    options.floor_z_map = 0.0;
    options.thre_z_min = 0.2;
    options.thre_z_max = 2.0;
    options.map_resolution = 1.0;
    options.thre_radius = 0.0;
    options.thres_point_count = 0;
    options.min_points_occupied = 2;
    options.ground_free_enabled = false;
    options.ground_free_min_height = -0.1;
    options.raytrace_max_range = 10.0;
    options.low_obstacle_diagnostic_height = 0.25;
    options.ground_plane_enabled = false;
    options.cell_evidence_diagnostics_enabled = false;

    anubis_mapping::CloudPtr ground(new anubis_mapping::PointCloudType());
    ground->push_back(make_point(2.0F, 0.0F, 0.0F));
    anubis_mapping::CloudPtr scan_cloud(new anubis_mapping::PointCloudType());
    scan_cloud->push_back(make_point(2.0F, 0.0F, 0.0F));
    anubis_mapping::Pcd2GridScan scan;
    scan.cloud_lidar = scan_cloud;
    anubis_mapping::Pcd2GridScans scans{scan};

    bool success = false;
    options.raytrace_free_enabled = true;
    const cv::Mat with_ray = run_grid_for_test(options, ground, scans, success);
    ASSERT_TRUE(success);
    ASSERT_EQ(with_ray.rows, 1);
    ASSERT_EQ(with_ray.cols, 3);
    for (int column = 0; column < with_ray.cols; ++column)
    {
        EXPECT_EQ(with_ray.at<uint8_t>(0, column), 255U);
    }

    options.raytrace_free_enabled = false;
    const cv::Mat without_ray = run_grid_for_test(options, ground, scans, success);
    ASSERT_TRUE(success);
    ASSERT_EQ(without_ray.rows, 1);
    ASSERT_EQ(without_ray.cols, 3);
    for (int column = 0; column < without_ray.cols; ++column)
    {
        EXPECT_EQ(without_ray.at<uint8_t>(0, column), 205U);
    }
}

TEST(Pcd2GridFreeSpaceTest, FiveCentGridKeepsDenseAndSparseCellsDistinct)
{
    anubis_mapping::Pcd2GridOptions options;
    options.floor_z_map = 0.0;
    options.thre_z_min = 0.1;
    options.thre_z_max = 2.0;
    options.map_resolution = 0.05;
    options.thre_radius = 0.0;
    options.thres_point_count = 0;
    options.min_points_occupied = 2;
    options.ground_free_enabled = false;
    options.raytrace_free_enabled = false;
    options.ground_plane_enabled = false;
    options.cell_evidence_diagnostics_enabled = false;

    // Two returns occupy the first 5 cm cell; the third is in the adjacent
    // cell and remains a sparse/free observation.
    anubis_mapping::CloudPtr points(new anubis_mapping::PointCloudType());
    points->push_back(make_point(0.000F, 0.000F, 0.20F));
    points->push_back(make_point(0.010F, 0.010F, 0.20F));
    points->push_back(make_point(0.060F, 0.000F, 0.20F));

    const auto unique_id = std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count();
    const std::filesystem::path output_dir =
        std::filesystem::temp_directory_path() /
        ("pcd2grid_five_cent_quantization_" + std::to_string(unique_id));
    ASSERT_TRUE(std::filesystem::create_directories(output_dir));
    const std::filesystem::path output_prefix = output_dir / "map";

    anubis_mapping::Pcd2Grid converter(options);
    ASSERT_TRUE(converter.run(points, output_prefix.string()));
    const cv::Mat image = cv::imread(
        output_prefix.string() + ".pgm", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());
    ASSERT_EQ(image.rows, 1);
    ASSERT_EQ(image.cols, 2);
    EXPECT_EQ(image.at<uint8_t>(0, 0), 0U);
    EXPECT_EQ(image.at<uint8_t>(0, 1), 255U);

    EXPECT_TRUE(std::filesystem::remove_all(output_dir) > 0U);
}

TEST(Pcd2GridFreeSpaceTest, RejectsUnrepresentableGridExtent)
{
    anubis_mapping::Pcd2GridOptions options;
    options.floor_z_map = 0.0;
    options.thre_z_min = 0.2;
    options.thre_z_max = 2.0;
    options.map_resolution = 0.05;
    options.thre_radius = 0.0;
    options.ground_plane_enabled = false;
    options.ground_free_enabled = false;
    options.raytrace_free_enabled = false;
    options.cell_evidence_diagnostics_enabled = false;

    anubis_mapping::CloudPtr points(new anubis_mapping::PointCloudType());
    points->push_back(make_point(-1.0e9F, 0.0F, 0.0F));
    points->push_back(make_point(1.0e9F, 0.0F, 0.0F));

    const auto unique_id = std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count();
    const std::filesystem::path output_dir =
        std::filesystem::temp_directory_path() /
        ("pcd2grid_extent_test_" + std::to_string(unique_id));
    ASSERT_TRUE(std::filesystem::create_directories(output_dir));
    const std::filesystem::path output_prefix = output_dir / "map";

    anubis_mapping::Pcd2Grid converter(options);
    EXPECT_FALSE(converter.run(points, output_prefix.string(), {}));
    EXPECT_FALSE(std::filesystem::exists(output_prefix.string() + ".pgm"));
    EXPECT_TRUE(std::filesystem::remove_all(output_dir) > 0U);
}
