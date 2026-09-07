#pragma once

#include <cstddef>
#include <cmath>
#include <Eigen/Core>
#include <string>
#include <vector>

namespace anubis_mapping
{
    struct GroundPlanePoint
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct GroundPlaneFitOptions
    {
        double fixed_floor_z = -0.304;
        double candidate_min_height = -0.15;
        double candidate_max_height = 0.30;
        double sample_cell_size = 0.20;
        int min_points_per_cell = 2;
        double sample_quantile = 0.20;
        std::size_t min_sample_cells = 100;
        double min_xy_span = 1.0;
        double min_inlier_ratio = 0.60;
        double residual_clip_min = 0.025;
        double residual_mad_scale = 3.0;
        double max_tilt_deg = 3.0;
        double max_residual_p95 = 0.08;
        // Save-time map leveling removes the rigid tilt first.  Keep the
        // fitted-floor offset gate at the post-leveling production bound.
        double max_floor_offset = 0.25;
        // Final inlier coverage must retain a meaningful fraction of the
        // observed footprint.  The minor-span floor prevents an exactly
        // collinear strip from being accepted while still allowing narrow
        // corridors.
        double min_inlier_span_fraction = 0.35;
        double min_inlier_minor_span = 0.05;
        bool quadratic_enabled = true;
        double quadratic_max_local_tilt_deg = 3.5;
        double quadratic_max_curvature = 0.02;
        int max_iterations = 8;
    };

    struct GroundPlaneModel
    {
        bool adaptive = false;
        bool quadratic = false;
        double fixed_floor_z = -0.304;
        double a = 0.0;
        double b = 0.0;
        double c = -0.304;
        double q_xx = 0.0;
        double q_xy = 0.0;
        double q_yy = 0.0;
        double center_x = 0.0;
        double center_y = 0.0;
        double coordinate_scale = 1.0;
        std::size_t candidate_points = 0;
        std::size_t sample_cells = 0;
        std::size_t inlier_cells = 0;
        double inlier_ratio = 0.0;
        double sample_major_span = 0.0;
        double sample_minor_span = 0.0;
        double inlier_major_span = 0.0;
        double inlier_minor_span = 0.0;
        double tilt_deg = 0.0;
        double max_local_tilt_deg = 0.0;
        double curvature = 0.0;
        double residual_p95 = 0.0;
        double fitted_floor_min = -0.304;
        double fitted_floor_max = -0.304;
        double median_floor_offset = 0.0;
        std::string fallback_reason;

        double floor_z(double x, double y) const
        {
            if (!adaptive)
            {
                return fixed_floor_z;
            }
            if (!quadratic)
            {
                return a * x + b * y + c;
            }
            const double normalized_x = (x - center_x) / coordinate_scale;
            const double normalized_y = (y - center_y) / coordinate_scale;
            return a * normalized_x + b * normalized_y + c +
                q_xx * normalized_x * normalized_x +
                q_xy * normalized_x * normalized_y +
                q_yy * normalized_y * normalized_y;
        }

        const char* model_name() const
        {
            if (!adaptive)
            {
                return "fixed_z";
            }
            return quadratic ? "adaptive_quadratic" : "adaptive_plane";
        }

        double height(double x, double y, double z) const
        {
            return z - floor_z(x, y);
        }

        // Unit normal of the fitted surface in map coordinates.  The normal
        // points toward +Z so it can be used directly for gravity leveling.
        Eigen::Vector3d normalAt(double x, double y) const
        {
            if (!adaptive)
            {
                return Eigen::Vector3d::UnitZ();
            }
            double gradient_x = a;
            double gradient_y = b;
            if (quadratic)
            {
                const double nx = (x - center_x) / coordinate_scale;
                const double ny = (y - center_y) / coordinate_scale;
                gradient_x = (a + 2.0 * q_xx * nx + q_xy * ny) /
                    coordinate_scale;
                gradient_y = (b + q_xy * nx + 2.0 * q_yy * ny) /
                    coordinate_scale;
            }
            Eigen::Vector3d normal(-gradient_x, -gradient_y, 1.0);
            const double norm = normal.norm();
            if (norm > 0.0 && std::isfinite(norm))
            {
                return normal / norm;
            }
            return Eigen::Vector3d::UnitZ();
        }
    };

    GroundPlaneModel fit_ground_plane(
        const std::vector<GroundPlanePoint>& points,
        const GroundPlaneFitOptions& options);
}  // namespace anubis_mapping
