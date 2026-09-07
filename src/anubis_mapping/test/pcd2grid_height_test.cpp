#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "pcd2grid_height.h"
#include "pcd2grid_ground_plane.h"
#include "pcd2grid_logic.h"

TEST(Pcd2GridHeightTest, ConvertsRelativeHeightsUsingSharedFloor)
{
    constexpr double floor_z_map = -0.304;

    ASSERT_TRUE(anubis_mapping::valid_pcd2grid_height_contract(
        floor_z_map, 0.05, 3.0, 1.2));
    EXPECT_DOUBLE_EQ(
        anubis_mapping::map_z_from_floor_height(floor_z_map, 0.05), -0.254);
    EXPECT_DOUBLE_EQ(
        anubis_mapping::map_z_from_floor_height(floor_z_map, 3.0), 2.696);
}

TEST(Pcd2GridHeightTest, ClearsOnlyCellsAboveRelativeOverheadHeight)
{
    constexpr double floor_z_map = -0.304;

    EXPECT_FALSE(anubis_mapping::is_overhead_cell(
        floor_z_map + 1.0, floor_z_map, 1.2));
    EXPECT_TRUE(anubis_mapping::is_overhead_cell(
        floor_z_map + 1.3, floor_z_map, 1.2));
    EXPECT_FALSE(anubis_mapping::is_overhead_cell(
        floor_z_map + 1.3, floor_z_map, 0.0));
}

TEST(Pcd2GridHeightTest, RejectsInvalidHeightContract)
{
    EXPECT_FALSE(anubis_mapping::valid_pcd2grid_height_contract(
        -0.304, 3.0, 0.05, 1.2));
    EXPECT_FALSE(anubis_mapping::valid_pcd2grid_height_contract(
        -0.304, 0.05, 3.0, std::nan("")));
}

TEST(Pcd2GridGroundPlaneTest, RejectsNonFiniteSampleQuantile)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    for (int x_index = 0; x_index < 4; ++x_index)
    {
        for (int y_index = 0; y_index < 4; ++y_index)
        {
            points.push_back({
                static_cast<double>(x_index), static_cast<double>(y_index),
                -0.304});
        }
    }
    anubis_mapping::GroundPlaneFitOptions options;
    options.fixed_floor_z = -0.304;
    options.sample_cell_size = 0.5;
    options.min_points_per_cell = 1;
    options.min_sample_cells = 3;
    options.min_xy_span = 1.0;
    options.sample_quantile = std::numeric_limits<double>::quiet_NaN();

    const auto model = anubis_mapping::fit_ground_plane(points, options);

    EXPECT_FALSE(model.adaptive);
    EXPECT_EQ(model.fallback_reason, "invalid fit options");
}

TEST(Pcd2GridGroundPlaneTest, RejectsNonFiniteQualityGateOptions)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    for (int x_index = 0; x_index < 4; ++x_index)
    {
        for (int y_index = 0; y_index < 4; ++y_index)
        {
            points.push_back({
                static_cast<double>(x_index), static_cast<double>(y_index),
                -0.304});
        }
    }
    const auto check_invalid = [&](auto configure)
    {
        anubis_mapping::GroundPlaneFitOptions options;
        options.fixed_floor_z = -0.304;
        options.sample_cell_size = 0.5;
        options.min_points_per_cell = 1;
        options.min_sample_cells = 3;
        configure(options);
        const auto model = anubis_mapping::fit_ground_plane(points, options);
        EXPECT_FALSE(model.adaptive);
        EXPECT_EQ(model.fallback_reason, "invalid fit options");
    };
    check_invalid([](anubis_mapping::GroundPlaneFitOptions& options) {
        options.min_xy_span = std::numeric_limits<double>::quiet_NaN();
    });
    check_invalid([](anubis_mapping::GroundPlaneFitOptions& options) {
        options.min_inlier_ratio = std::numeric_limits<double>::quiet_NaN();
    });
    check_invalid([](anubis_mapping::GroundPlaneFitOptions& options) {
        options.max_residual_p95 = std::numeric_limits<double>::quiet_NaN();
    });
}

TEST(Pcd2GridPathTest, AcceptsOnlyPortableRelativeBasenames)
{
    EXPECT_TRUE(anubis_mapping::is_safe_pcd2grid_basename("map"));
    EXPECT_TRUE(anubis_mapping::is_safe_pcd2grid_basename("floor_v2"));
    EXPECT_FALSE(anubis_mapping::is_safe_pcd2grid_basename(""));
    EXPECT_FALSE(anubis_mapping::is_safe_pcd2grid_basename("."));
    EXPECT_FALSE(anubis_mapping::is_safe_pcd2grid_basename(".."));
    EXPECT_FALSE(anubis_mapping::is_safe_pcd2grid_basename("../outside"));
    EXPECT_FALSE(anubis_mapping::is_safe_pcd2grid_basename("..\\outside"));
    EXPECT_FALSE(anubis_mapping::is_safe_pcd2grid_basename("C:\\outside"));
}

TEST(Pcd2GridLogicTest, UsesGroundBandAsFreeSpaceEvidence)
{
    constexpr double floor_z_map = -0.304;

    EXPECT_TRUE(anubis_mapping::is_ground_free_observation(
        floor_z_map, floor_z_map, -0.15, 0.05));
    EXPECT_TRUE(anubis_mapping::is_ground_free_observation(
        floor_z_map + 0.049, floor_z_map, -0.15, 0.05));
    EXPECT_FALSE(anubis_mapping::is_ground_free_observation(
        floor_z_map + 0.05, floor_z_map, -0.15, 0.05));
    EXPECT_FALSE(anubis_mapping::is_ground_free_observation(
        floor_z_map - 0.16, floor_z_map, -0.15, 0.05));
}

TEST(Pcd2GridLogicTest, OccupiedEvidenceHasPriorityOverFreeEvidence)
{
    constexpr double floor_z_map = -0.304;

    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell(
                  0, 0.0, floor_z_map, 1.2, false),
              anubis_mapping::kGridUnknown);
    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell(
                  1, floor_z_map + 0.1, floor_z_map, 1.2, false),
              anubis_mapping::kGridFree);
    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell(
                  1, floor_z_map + 0.1, floor_z_map, 1.2, true),
              anubis_mapping::kGridFree);
    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell(
                  2, floor_z_map + 0.1, floor_z_map, 1.2, true),
              anubis_mapping::kGridOccupied);
    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell(
                  2, floor_z_map + 1.3, floor_z_map, 1.2, false),
              anubis_mapping::kGridFree);
}

TEST(Pcd2GridLogicTest, RepeatedFreeFramesCanRejectSparseObstacleEvidence)
{
    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell_height(
                  2, 0.30, 0.55, true, 2,
                  1, 2, 2, 2, 2, 2.0),
              anubis_mapping::kGridFree);
    // Once an obstacle is persistent in the minimum number of frames, free
    // observations remain diagnostic evidence and cannot erase a wall.
    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell_height(
                  2, 0.30, 0.55, true, 2,
                  2, 20, 20, 2, 2, 2.0),
              anubis_mapping::kGridOccupied);
}

TEST(Pcd2GridLogicTest, LowPointGateClearsMixedLintelCells)
{
    // [2026-08-29/30 门口混合格修复·定版四条件验算] 低点计数(>=2)+占比
    // (>=0.25)同时满足,或低点帧数 >10 帧持续,或该格无空闲证据 → 阻挡;
    // 否则放行。实测数据:门口格 low=1/6(17%)、5~10 帧、有射线证据 → 放行;
    // 墙柱格 low=52/361(14%)、37 帧 → 阻挡;RayStops 夹具 {0.3,0.35} → 阻挡。

    // 规则①②:低点 2/8(25%>=25% 且 count>=2) → 阻挡(真实矮障碍)。
    EXPECT_TRUE(anubis_mapping::low_points_block_cell(
        2U, 8U, 0U, true, 2U, 0.25, 10U));
    // 规则③:低点 1 个但 12 帧持续(>10) → 阻挡(墙脚/门框)。
    EXPECT_TRUE(anubis_mapping::low_points_block_cell(
        1U, 10U, 12U, true, 2U, 0.25, 10U));
    // 规则④:1 个低点、8 帧,但该格无空闲证据 → 阻挡。
    EXPECT_TRUE(anubis_mapping::low_points_block_cell(
        1U, 10U, 0U, false, 2U, 0.25, 10U));
    // 实测门口格:低点 1/6(17%<25%),8 帧(<=10),有射线证据 → 放行。
    EXPECT_FALSE(anubis_mapping::low_points_block_cell(
        1U, 6U, 8U, true, 2U, 0.25, 10U));
    // 纯过梁(0 低点) → 不阻挡。
    EXPECT_FALSE(anubis_mapping::low_points_block_cell(
        0U, 4U, 37U, true, 2U, 0.25, 10U));
    // 空格子 → 不阻挡。
    EXPECT_FALSE(anubis_mapping::low_points_block_cell(
        0U, 0U, 0U, false, 2U, 0.25, 10U));

    // 分类重载:门口混合格(4 点,1 低点,8 帧,有射线证据) → 放行。
    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell_height(
                  4U, 1U, 8U, 0.55, true, 2,
                  3, 0, 0, 2, 2, 2.0, 2U, 0.25, 10U),
              anubis_mapping::kGridFree);
    // 纯矮障碍格(2 点全低) → 占用。
    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell_height(
                  2U, 2U, 2U, 0.55, true, 2,
                  2, 0, 0, 2, 2, 2.0, 2U, 0.25, 10U),
              anubis_mapping::kGridOccupied);
    // 零星低点但 12 帧持续(墙脚/门框) → 占用。
    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell_height(
                  4U, 1U, 12U, 0.55, true, 2,
                  3, 0, 0, 2, 2, 2.0, 2U, 0.25, 10U),
              anubis_mapping::kGridOccupied);
}

TEST(Pcd2GridLogicTest, OverheadVisibilityRequiresCrossingEvidence)
{
    // A ray that ends in a wall cell is not a clearance proof.
    EXPECT_FALSE(anubis_mapping::overhead_ray_crossing_sufficient(
        1U, 10U, 2U, 0.10));
    // Two independent crossings and a 20% crossing ratio are sufficient.
    EXPECT_TRUE(anubis_mapping::overhead_ray_crossing_sufficient(
        2U, 10U, 2U, 0.10));
    // A single accidental crossing cannot clear a frequently observed wall.
    EXPECT_FALSE(anubis_mapping::overhead_ray_crossing_sufficient(
        1U, 2U, 1U, 0.60));
    // A non-positive ratio intentionally disables only the ratio part; the
    // independent-frame minimum remains active.
    EXPECT_TRUE(anubis_mapping::overhead_ray_crossing_sufficient(
        2U, 100U, 2U, 0.0));

    // The low-point occupancy gate remains a hard veto even if visibility is
    // available: a dense/ persistent low return is a real obstacle.
    EXPECT_TRUE(anubis_mapping::low_points_form_obstacle(
        2U, 8U, 0U, 2U, 0.25, 10U));
    EXPECT_TRUE(anubis_mapping::low_points_form_obstacle(
        1U, 20U, 11U, 2U, 0.25, 10U));
    EXPECT_FALSE(anubis_mapping::low_points_form_obstacle(
        1U, 20U, 10U, 2U, 0.25, 10U));
    EXPECT_FALSE(anubis_mapping::low_points_form_obstacle(
        0U, 20U, 100U, 2U, 0.25, 10U));
}

TEST(Pcd2GridLogicTest, RejectsZeroMinimumOccupiedPointThreshold)
{
    EXPECT_FALSE(anubis_mapping::valid_min_points_occupied(0U));
    EXPECT_EQ(anubis_mapping::classify_pcd2grid_cell_height(
                  1U, 0.30, 0.55, false, 0U),
              anubis_mapping::kGridUnknown);
}

TEST(Pcd2GridLogicTest, GroundPlaneFallbackPolicyIsIndependentOfLevelingGate)
{
    // A disabled adaptive fit is rejected only when the PGM-specific
    // requirement is enabled.  The map-leveling save gate is intentionally
    // not an input to this pure policy function.
    EXPECT_FALSE(anubis_mapping::ground_plane_fallback_allowed(true, true, false));
    EXPECT_TRUE(anubis_mapping::ground_plane_fallback_allowed(true, false, false));
    EXPECT_TRUE(anubis_mapping::ground_plane_fallback_allowed(false, true, false));
    EXPECT_TRUE(anubis_mapping::ground_plane_fallback_allowed(true, true, true));
}

TEST(Pcd2GridLogicTest, RayTraversalIncludesEndpointsAndCanStop)
{
    std::vector<std::pair<int, int>> cells;
    anubis_mapping::trace_grid_line(0, 0, 4, 2, [&](int x, int y) {
        cells.emplace_back(x, y);
        return x < 2;
    });

    const std::vector<std::pair<int, int>> expected{{0, 0}, {1, 1}, {2, 1}};
    EXPECT_EQ(cells, expected);
}

TEST(Pcd2GridGroundPlaneTest, RecoversAGentlyTiltedFloorWithOutliers)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    for (int x_index = -20; x_index <= 20; ++x_index)
    {
        for (int y_index = -30; y_index <= 10; ++y_index)
        {
            const double x = 0.2 * x_index + 0.02;
            const double y = 0.2 * y_index + 0.03;
            const double floor = -0.006 * x - 0.013 * y - 0.292;
            points.push_back({x, y, floor - 0.005});
            points.push_back({x + 0.02, y + 0.01, floor + 0.006});
            if ((x_index + y_index) % 5 == 0)
            {
                points.push_back({x, y, floor + 0.22});
            }
        }
    }

    anubis_mapping::GroundPlaneFitOptions options;
    options.fixed_floor_z = -0.304;
    const auto model = anubis_mapping::fit_ground_plane(points, options);

    ASSERT_TRUE(model.adaptive) << model.fallback_reason;
    // Both a robust plane and a validated quadratic surface are valid outputs:
    // the production selector evaluates both and keeps the lower-residual
    // model.  Validate the geometry rather than coupling this regression to
    // the model order (quadratic coefficients use normalized coordinates).
    EXPECT_LT(model.residual_p95, options.max_residual_p95);
    EXPECT_LT(model.max_local_tilt_deg,
        options.quadratic_enabled
            ? options.quadratic_max_local_tilt_deg
            : options.max_tilt_deg);
    for (const auto& sample : std::vector<std::pair<double, double>>{
             {-2.0, -6.0}, {0.0, -2.0}, {3.0, 1.0}})
    {
        const double expected = -0.006 * sample.first -
            0.013 * sample.second - 0.292;
        EXPECT_NEAR(model.floor_z(sample.first, sample.second), expected, 0.02);
        EXPECT_NEAR(model.height(
            sample.first, sample.second,
            model.floor_z(sample.first, sample.second)), 0.0, 1e-9);
    }
}

TEST(Pcd2GridGroundPlaneTest, LowerEnvelopeQuantileRejectsElevatedReturns)
{
    constexpr double floor_z = -0.304;
    std::vector<anubis_mapping::GroundPlanePoint> points;
    for (int x_index = 0; x_index < 12; ++x_index)
    {
        for (int y_index = 0; y_index < 12; ++y_index)
        {
            const double x = 0.2 * static_cast<double>(x_index) + 0.02;
            const double y = 0.2 * static_cast<double>(y_index) + 0.02;
            const double floor = floor_z + 0.002 * x - 0.001 * y;
            points.push_back({x, y, floor - 0.003});
            points.push_back({x + 0.01, y + 0.01, floor + 0.002});
            points.push_back({x + 0.02, y, floor + 0.20});
            points.push_back({x, y + 0.02, floor + 0.20});
            points.push_back({x + 0.02, y + 0.02, floor + 0.20});
            points.push_back({x + 0.03, y, floor + 0.20});
            points.push_back({x, y + 0.03, floor + 0.20});
            points.push_back({x + 0.03, y + 0.01, floor + 0.20});
            points.push_back({x + 0.01, y + 0.03, floor + 0.20});
            points.push_back({x + 0.03, y + 0.03, floor + 0.20});
        }
    }

    anubis_mapping::GroundPlaneFitOptions lower_options;
    lower_options.fixed_floor_z = floor_z;
    lower_options.candidate_min_height = -0.10;
    lower_options.candidate_max_height = 0.40;
    lower_options.sample_cell_size = 0.20;
    lower_options.min_points_per_cell = 1;
    lower_options.sample_quantile = 0.05;
    lower_options.min_sample_cells = 100;
    lower_options.min_xy_span = 1.0;
    lower_options.quadratic_enabled = false;
    const auto lower = anubis_mapping::fit_ground_plane(points, lower_options);

    auto elevated_options = lower_options;
    elevated_options.sample_quantile = 0.20;
    const auto elevated = anubis_mapping::fit_ground_plane(points, elevated_options);

    ASSERT_TRUE(lower.adaptive) << lower.fallback_reason;
    ASSERT_TRUE(elevated.adaptive) << elevated.fallback_reason;
    EXPECT_NEAR(lower.floor_z(1.0, 1.0), floor_z, 0.01);
    EXPECT_GT(elevated.floor_z(1.0, 1.0) - lower.floor_z(1.0, 1.0), 0.02);
}

TEST(Pcd2GridGroundPlaneTest, FallsBackWhenCoverageIsTooSmall)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    for (int index = 0; index < 20; ++index)
    {
        points.push_back({0.01 * index, 0.01 * index, -0.304});
    }

    anubis_mapping::GroundPlaneFitOptions options;
    options.fixed_floor_z = -0.304;
    const auto model = anubis_mapping::fit_ground_plane(points, options);

    EXPECT_FALSE(model.adaptive);
    EXPECT_FALSE(model.fallback_reason.empty());
    EXPECT_DOUBLE_EQ(model.floor_z(50.0, -50.0), -0.304);
}

TEST(Pcd2GridGroundPlaneTest, EvaluatesQuadraticWhenPlaneSamplingGateEarlyExits)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    // Keep the number of cells below the production minimum while retaining
    // enough geometry for the quadratic candidate to run its own checks.
    for (int x_index = 0; x_index < 4; ++x_index)
    {
        for (int y_index = 0; y_index < 4; ++y_index)
        {
            const double x = 0.5 * static_cast<double>(x_index);
            const double y = 0.5 * static_cast<double>(y_index);
            points.push_back({x, y, -0.304 + 0.002 * x - 0.001 * y});
        }
    }

    anubis_mapping::GroundPlaneFitOptions options;
    options.fixed_floor_z = -0.304;
    options.sample_cell_size = 0.5;
    options.min_points_per_cell = 1;
    options.min_sample_cells = 100;
    options.min_xy_span = 1.0;
    options.quadratic_enabled = true;

    const auto model = anubis_mapping::fit_ground_plane(points, options);

    EXPECT_FALSE(model.adaptive);
    EXPECT_NE(model.fallback_reason.find("quadratic:"), std::string::npos);
    EXPECT_NE(
        model.fallback_reason.find("quadratic insufficient spatial sample cells"),
        std::string::npos);
}

TEST(Pcd2GridGroundPlaneTest, KeepsAcceptedPlaneWhenQuadraticCandidateFails)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    for (int x_index = -10; x_index <= 10; ++x_index)
    {
        for (int y_index = -10; y_index <= 10; ++y_index)
        {
            const double x = 0.2 * static_cast<double>(x_index) + 0.01;
            const double y = 0.2 * static_cast<double>(y_index) + 0.02;
            const double floor = -0.304 + 0.02 * x - 0.004 * y;
            points.push_back({x, y, floor - 0.002});
            points.push_back({x + 0.01, y + 0.01, floor + 0.002});
        }
    }

    anubis_mapping::GroundPlaneFitOptions options;
    options.fixed_floor_z = -0.304;
    options.quadratic_enabled = true;
    // The plane tilt is about 1.2 degrees, so this deliberately rejects the
    // quadratic candidate while leaving the plane quality gate permissive.
    options.quadratic_max_local_tilt_deg = 0.5;

    const auto model = anubis_mapping::fit_ground_plane(points, options);

    ASSERT_TRUE(model.adaptive) << model.fallback_reason;
    EXPECT_FALSE(model.quadratic);
    EXPECT_LT(model.residual_p95, options.max_residual_p95);
    EXPECT_NEAR(model.floor_z(1.0, -1.0),
        -0.304 + 0.02 * 1.0 - 0.004 * -1.0, 0.01);
}

TEST(Pcd2GridGroundPlaneTest, AcceptsLongDiagonalNarrowCoverage)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    const Eigen::Vector2d direction =
        Eigen::Vector2d(1.0, 1.0).normalized();
    const Eigen::Vector2d normal(-direction.y(), direction.x());
    for (int index = 0; index < 50; ++index)
    {
        const double distance = 0.025 * static_cast<double>(index);
        const Eigen::Vector2d center = direction * distance;
        for (const double offset : {-0.045, 0.045})
        {
            const Eigen::Vector2d xy = center + normal * offset;
            const double floor = -0.304 + 0.004 * xy.x() -
                0.003 * xy.y();
            points.push_back({xy.x(), xy.y(), floor});
        }
    }

    anubis_mapping::GroundPlaneFitOptions options;
    options.fixed_floor_z = -0.304;
    options.sample_cell_size = 0.05;
    options.min_points_per_cell = 1;
    options.min_sample_cells = 20;
    options.min_xy_span = 1.0;
    options.quadratic_enabled = false;

    const auto model = anubis_mapping::fit_ground_plane(points, options);

    ASSERT_TRUE(model.adaptive) << model.fallback_reason;
    EXPECT_NEAR(model.floor_z(0.8, 0.8),
        -0.304 + 0.004 * 0.8 - 0.003 * 0.8, 0.01);
}

TEST(Pcd2GridGroundPlaneTest, RejectsUnsafeTilt)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    for (int x_index = -20; x_index <= 20; ++x_index)
    {
        for (int y_index = -20; y_index <= 20; ++y_index)
        {
            const double x = 0.2 * x_index;
            const double y = 0.2 * y_index;
            const double floor = 0.08 * x - 0.304;
            points.push_back({x, y, floor});
            points.push_back({x + 0.01, y + 0.01, floor + 0.002});
        }
    }

    anubis_mapping::GroundPlaneFitOptions options;
    options.fixed_floor_z = -0.304;
    const auto model = anubis_mapping::fit_ground_plane(points, options);

    EXPECT_FALSE(model.adaptive);
    EXPECT_EQ(model.fallback_reason.find("tilt exceeds quality gate"), 0U);
}

TEST(Pcd2GridGroundPlaneTest, UsesQuadraticOnlyForSmoothResidualCurvature)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    for (int x_index = -30; x_index <= 30; ++x_index)
    {
        for (int y_index = -30; y_index <= 30; ++y_index)
        {
            const double x = 0.2 * x_index + 0.02;
            const double y = 0.2 * y_index + 0.03;
            const double floor = -0.36 + 0.004 * x * x + 0.003 * y * y;
            points.push_back({x, y, floor - 0.005});
            points.push_back({x + 0.02, y + 0.01, floor + 0.005});
            if ((x_index + y_index) % 7 == 0)
            {
                points.push_back({x, y, floor + 0.18});
            }
        }
    }

    anubis_mapping::GroundPlaneFitOptions options;
    options.fixed_floor_z = -0.304;
    const auto model = anubis_mapping::fit_ground_plane(points, options);

    ASSERT_TRUE(model.adaptive) << model.fallback_reason;
    ASSERT_TRUE(model.quadratic);
    EXPECT_LT(model.residual_p95, 0.02);
    EXPECT_LT(model.max_local_tilt_deg,
        options.quadratic_max_local_tilt_deg);
    EXPECT_LT(model.curvature, options.quadratic_max_curvature);
    EXPECT_NEAR(model.floor_z(4.0, -3.0),
        -0.36 + 0.004 * 16.0 + 0.003 * 9.0, 0.01);
}

TEST(Pcd2GridGroundPlaneTest, QuadraticFloorRangeUsesOnlyRobustInliers)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    for (int x_index = -20; x_index <= 20; ++x_index)
    {
        for (int y_index = -20; y_index <= 20; ++y_index)
        {
            const double x = 0.2 * x_index + 0.02;
            const double y = 0.2 * y_index + 0.03;
            const double floor = -0.34 + 0.003 * x * x + 0.002 * y * y;
            points.push_back({x, y, floor});
        }
    }
    // Sparse low-obstacle samples stay inside the broad candidate band but
    // are rejected by the robust quadratic fit.  They must not define the
    // reported floor extent or median gauge.
    points.push_back({-4.25, -4.25, 0.15});
    points.push_back({-4.24, -4.24, 0.15});
    points.push_back({4.25, 4.25, 0.15});
    points.push_back({4.24, 4.24, 0.15});

    anubis_mapping::GroundPlaneFitOptions options;
    options.fixed_floor_z = -0.304;
    options.min_points_per_cell = 1;
    options.candidate_min_height = -0.20;
    options.candidate_max_height = 0.60;
    options.max_floor_offset = 0.25;
    const auto model = anubis_mapping::fit_ground_plane(points, options);

    ASSERT_TRUE(model.adaptive) << model.fallback_reason;
    ASSERT_TRUE(model.quadratic);
    EXPECT_LT(model.fitted_floor_max, -0.20);
    EXPECT_GT(model.fitted_floor_min, -0.40);
    EXPECT_LT(std::abs(model.median_floor_offset), 0.08);
}

TEST(Pcd2GridGroundPlaneTest, RejectsQuadraticWithUnsafeCurvature)
{
    std::vector<anubis_mapping::GroundPlanePoint> points;
    for (int x_index = -15; x_index <= 15; ++x_index)
    {
        for (int y_index = -15; y_index <= 15; ++y_index)
        {
            const double x = 0.2 * x_index + 0.02;
            const double y = 0.2 * y_index + 0.03;
            const double floor = -0.40 + 0.02 * x * x;
            points.push_back({x, y, floor - 0.003});
            points.push_back({x + 0.02, y + 0.01, floor + 0.003});
        }
    }

    anubis_mapping::GroundPlaneFitOptions options;
    options.fixed_floor_z = -0.304;
    const auto model = anubis_mapping::fit_ground_plane(points, options);

    EXPECT_FALSE(model.adaptive);
    EXPECT_NE(model.fallback_reason.find("quadratic"), std::string::npos);
}
