#include "pcd2grid_ground_plane.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace anubis_mapping
{
    namespace
    {
        double quantile(std::vector<double> values, double fraction)
        {
            if (values.empty())
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            std::sort(values.begin(), values.end());
            const double position = fraction * static_cast<double>(values.size() - 1U);
            const std::size_t lower = static_cast<std::size_t>(std::floor(position));
            const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
            const double weight = position - static_cast<double>(lower);
            return values[lower] * (1.0 - weight) + values[upper] * weight;
        }

        struct XYCoverageStats
        {
            bool valid = false;
            double major_span = 0.0;
            double minor_span = 0.0;
        };

        XYCoverageStats compute_xy_coverage(
            const std::vector<GroundPlanePoint>& samples,
            const std::vector<uint8_t>* mask = nullptr)
        {
            XYCoverageStats result;
            std::vector<Eigen::Vector2d> selected;
            selected.reserve(samples.size());
            for (std::size_t index = 0; index < samples.size(); ++index)
            {
                if (mask != nullptr &&
                    (index >= mask->size() || (*mask)[index] == 0U))
                {
                    continue;
                }
                if (!std::isfinite(samples[index].x) ||
                    !std::isfinite(samples[index].y))
                {
                    continue;
                }
                selected.emplace_back(samples[index].x, samples[index].y);
            }
            if (selected.size() < 3U)
            {
                return result;
            }

            Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
            for (const auto& sample : selected)
            {
                centroid += sample;
            }
            centroid /= static_cast<double>(selected.size());
            Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
            for (const auto& sample : selected)
            {
                const Eigen::Vector2d delta = sample - centroid;
                covariance.noalias() += delta * delta.transpose();
            }
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(
                covariance, Eigen::ComputeEigenvectors);
            if (solver.info() != Eigen::Success ||
                !solver.eigenvalues().allFinite() ||
                !solver.eigenvectors().allFinite())
            {
                return result;
            }
            const Eigen::Matrix2d axes = solver.eigenvectors();
            double major_min = std::numeric_limits<double>::max();
            double major_max = std::numeric_limits<double>::lowest();
            double minor_min = std::numeric_limits<double>::max();
            double minor_max = std::numeric_limits<double>::lowest();
            for (const auto& sample : selected)
            {
                const Eigen::Vector2d delta = sample - centroid;
                const double minor = axes.col(0).dot(delta);
                const double major = axes.col(1).dot(delta);
                major_min = std::min(major_min, major);
                major_max = std::max(major_max, major);
                minor_min = std::min(minor_min, minor);
                minor_max = std::max(minor_max, minor);
            }
            result.major_span = major_max - major_min;
            result.minor_span = minor_max - minor_min;
            const double major_variance = solver.eigenvalues()(1);
            const double minor_variance = solver.eigenvalues()(0);
            result.valid = std::isfinite(result.major_span) &&
                std::isfinite(result.minor_span) &&
                std::isfinite(major_variance) && std::isfinite(minor_variance) &&
                major_variance > 0.0 && minor_variance >
                    std::numeric_limits<double>::epsilon() *
                        std::max(1.0, major_variance);
            return result;
        }

        bool solve_plane(
            const std::vector<GroundPlanePoint>& samples,
            const std::vector<uint8_t>& inliers, Eigen::Vector3d& coefficients)
        {
            Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
            Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
            std::size_t count = 0;
            for (std::size_t index = 0; index < samples.size(); ++index)
            {
                if (inliers[index] == 0U)
                {
                    continue;
                }
                const Eigen::Vector3d row(samples[index].x, samples[index].y, 1.0);
                normal.noalias() += row * row.transpose();
                rhs.noalias() += row * samples[index].z;
                ++count;
            }
            if (count < 3U)
            {
                return false;
            }
            const Eigen::LDLT<Eigen::Matrix3d> decomposition(normal);
            if (decomposition.info() != Eigen::Success)
            {
                return false;
            }
            coefficients = decomposition.solve(rhs);
            return decomposition.info() == Eigen::Success && coefficients.allFinite();
        }

        using QuadraticCoefficients = Eigen::Matrix<double, 6, 1>;
        using QuadraticNormalMatrix = Eigen::Matrix<double, 6, 6>;

        QuadraticCoefficients quadratic_row(
            const GroundPlanePoint& sample, double center_x, double center_y,
            double coordinate_scale)
        {
            const double x = (sample.x - center_x) / coordinate_scale;
            const double y = (sample.y - center_y) / coordinate_scale;
            QuadraticCoefficients row;
            row << x, y, 1.0, x * x, x * y, y * y;
            return row;
        }

        bool solve_quadratic(
            const std::vector<GroundPlanePoint>& samples,
            const std::vector<uint8_t>& inliers, double center_x,
            double center_y, double coordinate_scale,
            QuadraticCoefficients& coefficients)
        {
            QuadraticNormalMatrix normal = QuadraticNormalMatrix::Zero();
            QuadraticCoefficients rhs = QuadraticCoefficients::Zero();
            std::size_t count = 0;
            for (std::size_t index = 0; index < samples.size(); ++index)
            {
                if (inliers[index] == 0U)
                {
                    continue;
                }
                const QuadraticCoefficients row = quadratic_row(
                    samples[index], center_x, center_y, coordinate_scale);
                normal.noalias() += row * row.transpose();
                rhs.noalias() += row * samples[index].z;
                ++count;
            }
            if (count < 6U)
            {
                return false;
            }
            const Eigen::LDLT<QuadraticNormalMatrix> decomposition(normal);
            if (decomposition.info() != Eigen::Success)
            {
                return false;
            }
            coefficients = decomposition.solve(rhs);
            return decomposition.info() == Eigen::Success &&
                coefficients.allFinite();
        }

        GroundPlaneModel fallback_model(
            const GroundPlaneFitOptions& options, const std::string& reason)
        {
            GroundPlaneModel model;
            model.fixed_floor_z = options.fixed_floor_z;
            model.c = options.fixed_floor_z;
            model.fitted_floor_min = options.fixed_floor_z;
            model.fitted_floor_max = options.fixed_floor_z;
            model.fallback_reason = reason;
            return model;
        }

        GroundPlaneModel fit_quadratic_surface(
            const std::vector<GroundPlanePoint>& samples,
            const GroundPlaneFitOptions& options,
            std::size_t candidate_points)
        {
            GroundPlaneModel model = fallback_model(
                options, "quadratic surface not evaluated");
            model.quadratic = true;
            model.candidate_points = candidate_points;
            model.sample_cells = samples.size();
            if (samples.size() < options.min_sample_cells)
            {
                model.fallback_reason =
                    "quadratic insufficient spatial sample cells";
                return model;
            }
            const XYCoverageStats sample_coverage = compute_xy_coverage(samples);
            model.sample_major_span = sample_coverage.major_span;
            model.sample_minor_span = sample_coverage.minor_span;
            if (!sample_coverage.valid ||
                model.sample_major_span < options.min_xy_span ||
                model.sample_minor_span < options.min_inlier_minor_span)
            {
                model.fallback_reason = "quadratic insufficient xy coverage";
                return model;
            }
            if (!std::isfinite(options.quadratic_max_local_tilt_deg) ||
                options.quadratic_max_local_tilt_deg <= 0.0 ||
                !std::isfinite(options.quadratic_max_curvature) ||
                options.quadratic_max_curvature <= 0.0)
            {
                model.fallback_reason = "invalid quadratic fit options";
                return model;
            }

            double x_min = std::numeric_limits<double>::max();
            double x_max = std::numeric_limits<double>::lowest();
            double y_min = std::numeric_limits<double>::max();
            double y_max = std::numeric_limits<double>::lowest();
            for (const auto& sample : samples)
            {
                x_min = std::min(x_min, sample.x);
                x_max = std::max(x_max, sample.x);
                y_min = std::min(y_min, sample.y);
                y_max = std::max(y_max, sample.y);
            }
            model.center_x = 0.5 * (x_min + x_max);
            model.center_y = 0.5 * (y_min + y_max);
            model.coordinate_scale = std::max(
                1.0, 0.5 * std::max(x_max - x_min, y_max - y_min));

            std::vector<uint8_t> inliers(samples.size(), 1U);
            QuadraticCoefficients coefficients =
                QuadraticCoefficients::Zero();
            for (int iteration = 0; iteration < options.max_iterations; ++iteration)
            {
                if (!solve_quadratic(
                        samples, inliers, model.center_x, model.center_y,
                        model.coordinate_scale, coefficients))
                {
                    model.fallback_reason =
                        "quadratic normal matrix is singular";
                    return model;
                }
                std::vector<double> active_residuals;
                std::vector<double> residuals(samples.size(), 0.0);
                active_residuals.reserve(samples.size());
                for (std::size_t index = 0; index < samples.size(); ++index)
                {
                    const double fitted = quadratic_row(
                        samples[index], model.center_x, model.center_y,
                        model.coordinate_scale).dot(coefficients);
                    residuals[index] = samples[index].z - fitted;
                    if (inliers[index] != 0U)
                    {
                        active_residuals.push_back(residuals[index]);
                    }
                }
                if (active_residuals.empty())
                {
                    model.fallback_reason =
                        "quadratic robust iteration lost all inliers";
                    return model;
                }
                const double median = quantile(active_residuals, 0.50);
                std::vector<double> deviations;
                deviations.reserve(active_residuals.size());
                for (const double residual : active_residuals)
                {
                    deviations.push_back(std::abs(residual - median));
                }
                const double mad = quantile(std::move(deviations), 0.50);
                const double threshold = std::max(
                    options.residual_clip_min,
                    options.residual_mad_scale * 1.4826 * mad);
                std::vector<uint8_t> updated(samples.size(), 0U);
                for (std::size_t index = 0; index < samples.size(); ++index)
                {
                    updated[index] =
                        std::abs(residuals[index] - median) <= threshold
                            ? 1U : 0U;
                }
                if (updated == inliers)
                {
                    break;
                }
                inliers.swap(updated);
            }
            if (!solve_quadratic(
                    samples, inliers, model.center_x, model.center_y,
                    model.coordinate_scale, coefficients))
            {
                model.fallback_reason = "final quadratic solve failed";
                return model;
            }

            model.a = coefficients[0];
            model.b = coefficients[1];
            model.c = coefficients[2];
            model.q_xx = coefficients[3];
            model.q_xy = coefficients[4];
            model.q_yy = coefficients[5];
            model.inlier_cells = static_cast<std::size_t>(
                std::count(inliers.begin(), inliers.end(), 1U));
            model.inlier_ratio = static_cast<double>(model.inlier_cells) /
                static_cast<double>(samples.size());
            const XYCoverageStats inlier_coverage =
                compute_xy_coverage(samples, &inliers);
            model.inlier_major_span = inlier_coverage.major_span;
            model.inlier_minor_span = inlier_coverage.minor_span;

            std::vector<double> absolute_residuals;
            std::vector<double> local_tilts;
            std::vector<double> fitted_floors;
            absolute_residuals.reserve(model.inlier_cells);
            local_tilts.reserve(model.inlier_cells);
            fitted_floors.reserve(samples.size());
            for (std::size_t index = 0; index < samples.size(); ++index)
            {
                const double normalized_x =
                    (samples[index].x - model.center_x) /
                    model.coordinate_scale;
                const double normalized_y =
                    (samples[index].y - model.center_y) /
                    model.coordinate_scale;
                const double fitted =
                    model.a * normalized_x + model.b * normalized_y + model.c +
                    model.q_xx * normalized_x * normalized_x +
                    model.q_xy * normalized_x * normalized_y +
                    model.q_yy * normalized_y * normalized_y;
                if (inliers[index] == 0U)
                {
                    continue;
                }
                // Floor extent and gauge must describe the accepted ground
                // support only.  Including rejected wall/obstacle samples
                // here can expand the range or shift the median even though
                // the robust fit correctly excluded those samples.
                fitted_floors.push_back(fitted);
                absolute_residuals.push_back(
                    std::abs(samples[index].z - fitted));
                const double gradient_x =
                    (model.a + 2.0 * model.q_xx * normalized_x +
                     model.q_xy * normalized_y) /
                    model.coordinate_scale;
                const double gradient_y =
                    (model.b + model.q_xy * normalized_x +
                     2.0 * model.q_yy * normalized_y) /
                    model.coordinate_scale;
                local_tilts.push_back(
                    180.0 / std::acos(-1.0) *
                    std::atan(std::hypot(gradient_x, gradient_y)));
            }
            if (absolute_residuals.empty() || local_tilts.empty() ||
                fitted_floors.empty())
            {
                model.fallback_reason =
                    "quadratic fit produced no finite inlier metrics";
                return model;
            }
            model.residual_p95 = quantile(absolute_residuals, 0.95);
            model.tilt_deg = quantile(local_tilts, 0.99);
            model.max_local_tilt_deg =
                *std::max_element(local_tilts.begin(), local_tilts.end());
            const auto floor_extents = std::minmax_element(
                fitted_floors.begin(), fitted_floors.end());
            model.fitted_floor_min = *floor_extents.first;
            model.fitted_floor_max = *floor_extents.second;
            model.median_floor_offset =
                quantile(fitted_floors, 0.50) - options.fixed_floor_z;
            if (!std::isfinite(model.residual_p95) ||
                !std::isfinite(model.tilt_deg) ||
                !std::isfinite(model.max_local_tilt_deg) ||
                !std::isfinite(model.fitted_floor_min) ||
                !std::isfinite(model.fitted_floor_max) ||
                !std::isfinite(model.median_floor_offset) ||
                !std::isfinite(model.sample_major_span) ||
                !std::isfinite(model.sample_minor_span) ||
                !std::isfinite(model.inlier_major_span) ||
                !std::isfinite(model.inlier_minor_span))
            {
                model.fallback_reason =
                    "quadratic fit produced non-finite metrics";
                return model;
            }

            Eigen::Matrix2d hessian;
            const double inverse_scale_squared = 1.0 /
                (model.coordinate_scale * model.coordinate_scale);
            hessian << 2.0 * model.q_xx, model.q_xy,
                model.q_xy, 2.0 * model.q_yy;
            hessian *= inverse_scale_squared;
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(
                hessian, Eigen::EigenvaluesOnly);
            if (eigensolver.info() != Eigen::Success)
            {
                model.fallback_reason = "quadratic curvature solve failed";
                return model;
            }
            model.curvature = eigensolver.eigenvalues().cwiseAbs().maxCoeff();

            if (model.inlier_ratio < options.min_inlier_ratio)
            {
                model.fallback_reason =
                    "quadratic inlier ratio below quality gate";
                return model;
            }
            if (!inlier_coverage.valid ||
                model.inlier_major_span <
                    options.min_xy_span * options.min_inlier_span_fraction ||
                model.inlier_minor_span < options.min_inlier_minor_span)
            {
                model.fallback_reason =
                    "quadratic inlier xy coverage below quality gate";
                return model;
            }
            if (model.residual_p95 > options.max_residual_p95)
            {
                model.fallback_reason =
                    "quadratic residual p95 exceeds quality gate";
                return model;
            }
            if (model.max_local_tilt_deg >
                options.quadratic_max_local_tilt_deg)
            {
                model.fallback_reason =
                    "quadratic local tilt exceeds quality gate";
                return model;
            }
            if (model.curvature > options.quadratic_max_curvature)
            {
                model.fallback_reason =
                    "quadratic curvature exceeds quality gate";
                return model;
            }
            const double max_floor_deviation = std::max(
                std::abs(model.fitted_floor_min - options.fixed_floor_z),
                std::abs(model.fitted_floor_max - options.fixed_floor_z));
            if (std::abs(model.median_floor_offset) >
                    options.max_floor_offset ||
                max_floor_deviation > options.max_floor_offset)
            {
                model.fallback_reason =
                    "quadratic floor too far from configured floor";
                return model;
            }

            model.adaptive = true;
            model.fallback_reason.clear();
            return model;
        }
    }  // namespace

    GroundPlaneModel fit_ground_plane(
        const std::vector<GroundPlanePoint>& points,
        const GroundPlaneFitOptions& options)
    {
        GroundPlaneModel model = fallback_model(options, "not evaluated");
        if (!std::isfinite(options.fixed_floor_z) ||
            !std::isfinite(options.candidate_min_height) ||
            !std::isfinite(options.candidate_max_height) ||
            options.candidate_min_height >= options.candidate_max_height ||
            !std::isfinite(options.sample_cell_size) ||
            options.sample_cell_size <= 0.0 || options.min_points_per_cell < 1 ||
            !std::isfinite(options.sample_quantile) ||
            options.sample_quantile < 0.0 || options.sample_quantile > 1.0 ||
            options.min_sample_cells < 3U ||
            !std::isfinite(options.min_xy_span) || options.min_xy_span <= 0.0 ||
            !std::isfinite(options.min_inlier_ratio) ||
            options.min_inlier_ratio <= 0.0 || options.min_inlier_ratio > 1.0 ||
            !std::isfinite(options.residual_clip_min) ||
            options.residual_clip_min <= 0.0 ||
            !std::isfinite(options.residual_mad_scale) ||
            options.residual_mad_scale <= 0.0 ||
            !std::isfinite(options.max_tilt_deg) || options.max_tilt_deg <= 0.0 ||
            !std::isfinite(options.max_residual_p95) ||
            options.max_residual_p95 <= 0.0 ||
            !std::isfinite(options.max_floor_offset) ||
            options.max_floor_offset <= 0.0 || options.max_iterations < 1 ||
            !std::isfinite(options.min_inlier_span_fraction) ||
            options.min_inlier_span_fraction <= 0.0 ||
            options.min_inlier_span_fraction > 1.0 ||
            !std::isfinite(options.min_inlier_minor_span) ||
            options.min_inlier_minor_span <= 0.0 ||
            (options.quadratic_enabled &&
             (!std::isfinite(options.quadratic_max_local_tilt_deg) ||
              options.quadratic_max_local_tilt_deg <= 0.0 ||
              !std::isfinite(options.quadratic_max_curvature) ||
              options.quadratic_max_curvature <= 0.0)))
        {
            model.fallback_reason = "invalid fit options";
            return model;
        }

        using CellKey = std::pair<int64_t, int64_t>;
        std::map<CellKey, std::vector<GroundPlanePoint>> cells;
        const double candidate_min_z =
            options.fixed_floor_z + options.candidate_min_height;
        const double candidate_max_z =
            options.fixed_floor_z + options.candidate_max_height;
        for (const auto& point : points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                !std::isfinite(point.z) || point.z < candidate_min_z ||
                point.z >= candidate_max_z)
            {
                continue;
            }
            const CellKey key{
                static_cast<int64_t>(std::floor(point.x / options.sample_cell_size)),
                static_cast<int64_t>(std::floor(point.y / options.sample_cell_size))};
            cells[key].push_back(point);
            ++model.candidate_points;
        }

        std::vector<GroundPlanePoint> samples;
        samples.reserve(cells.size());
        for (const auto& entry : cells)
        {
            if (entry.second.size() <
                static_cast<std::size_t>(options.min_points_per_cell))
            {
                continue;
            }
            std::vector<double> xs;
            std::vector<double> ys;
            std::vector<double> zs;
            xs.reserve(entry.second.size());
            ys.reserve(entry.second.size());
            zs.reserve(entry.second.size());
            for (const auto& point : entry.second)
            {
                xs.push_back(point.x);
                ys.push_back(point.y);
                zs.push_back(point.z);
            }
            samples.push_back({
                quantile(std::move(xs), 0.50),
                quantile(std::move(ys), 0.50),
                quantile(std::move(zs), options.sample_quantile)});
        }
        model.sample_cells = samples.size();

        // Plane and quadratic fitting are independent candidates.  Keep the
        // selector available even when the plane path cannot produce a
        // solution, so a future change to the quadratic gates cannot be
        // silently bypassed by an early plane return.
        const auto select_with_quadratic =
            [&](GroundPlaneModel candidate,
                const std::string& plane_failure) -> GroundPlaneModel
        {
            GroundPlaneModel quadratic = fallback_model(
                options, "quadratic fitting disabled");
            if (options.quadratic_enabled)
            {
                quadratic = fit_quadratic_surface(
                    samples, options, candidate.candidate_points);
            }
            if (quadratic.adaptive && candidate.adaptive)
            {
                // Both models passed their independent quality gates.  The
                // robust residual is the deterministic tie-breaker.
                return quadratic.residual_p95 < candidate.residual_p95
                    ? quadratic : candidate;
            }
            if (quadratic.adaptive)
            {
                return quadratic;
            }
            if (plane_failure.empty() && candidate.adaptive)
            {
                // The plane already passed all of its gates.  A rejected
                // higher-order candidate must not turn that valid result into
                // a fixed-floor fallback.
                return candidate;
            }
            candidate.adaptive = false;
            if (!plane_failure.empty())
            {
                candidate.fallback_reason = plane_failure;
                if (options.quadratic_enabled &&
                    !quadratic.fallback_reason.empty())
                {
                    candidate.fallback_reason += "; quadratic: " +
                        quadratic.fallback_reason;
                }
            }
            return candidate;
        };

        if (samples.size() < options.min_sample_cells)
        {
            return select_with_quadratic(
                model, "insufficient spatial sample cells");
        }

        const XYCoverageStats sample_coverage = compute_xy_coverage(samples);
        model.sample_major_span = sample_coverage.major_span;
        model.sample_minor_span = sample_coverage.minor_span;
        // Coverage is orientation-independent: a long narrow or diagonal
        // corridor can have one axis-aligned span below min_xy_span while
        // still providing enough spatial extent for a stable plane fit.
        // Keep a non-zero minor span requirement so an exactly collinear set
        // is rejected before the normal-equation solve.
        if (!sample_coverage.valid ||
            model.sample_major_span < options.min_xy_span ||
            model.sample_minor_span < options.min_inlier_minor_span)
        {
            return select_with_quadratic(model, "insufficient xy coverage");
        }

        std::vector<uint8_t> inliers(samples.size(), 1U);
        Eigen::Vector3d coefficients = Eigen::Vector3d::Zero();
        for (int iteration = 0; iteration < options.max_iterations; ++iteration)
        {
            if (!solve_plane(samples, inliers, coefficients))
            {
                return select_with_quadratic(
                    model, "plane normal matrix is singular");
            }
            std::vector<double> active_residuals;
            std::vector<double> residuals(samples.size(), 0.0);
            active_residuals.reserve(samples.size());
            for (std::size_t index = 0; index < samples.size(); ++index)
            {
                residuals[index] = samples[index].z -
                    (coefficients.x() * samples[index].x +
                     coefficients.y() * samples[index].y + coefficients.z());
                if (inliers[index] != 0U)
                {
                    active_residuals.push_back(residuals[index]);
                }
            }
            const double median = quantile(active_residuals, 0.50);
            std::vector<double> deviations;
            deviations.reserve(active_residuals.size());
            for (const double residual : active_residuals)
            {
                deviations.push_back(std::abs(residual - median));
            }
            const double mad = quantile(std::move(deviations), 0.50);
            const double threshold = std::max(
                options.residual_clip_min,
                options.residual_mad_scale * 1.4826 * mad);
            std::vector<uint8_t> updated(samples.size(), 0U);
            for (std::size_t index = 0; index < samples.size(); ++index)
            {
                updated[index] =
                    std::abs(residuals[index] - median) <= threshold ? 1U : 0U;
            }
            if (updated == inliers)
            {
                break;
            }
            inliers.swap(updated);
        }
        if (!solve_plane(samples, inliers, coefficients))
        {
            return select_with_quadratic(model, "final plane solve failed");
        }

        model.a = coefficients.x();
        model.b = coefficients.y();
        model.c = coefficients.z();
        model.inlier_cells = static_cast<std::size_t>(
            std::count(inliers.begin(), inliers.end(), 1U));
        model.inlier_ratio = static_cast<double>(model.inlier_cells) /
            static_cast<double>(samples.size());
        const XYCoverageStats inlier_coverage =
            compute_xy_coverage(samples, &inliers);
        model.inlier_major_span = inlier_coverage.major_span;
        model.inlier_minor_span = inlier_coverage.minor_span;
        model.tilt_deg = 180.0 / std::acos(-1.0) *
            std::atan(std::hypot(model.a, model.b));
        model.max_local_tilt_deg = model.tilt_deg;

        std::vector<double> absolute_residuals;
        std::vector<double> fitted_floors;
        absolute_residuals.reserve(model.inlier_cells);
        fitted_floors.reserve(model.inlier_cells);
        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            if (inliers[index] == 0U)
            {
                continue;
            }
            const double fitted = model.a * samples[index].x +
                model.b * samples[index].y + model.c;
            fitted_floors.push_back(fitted);
            absolute_residuals.push_back(std::abs(samples[index].z - fitted));
        }
        if (absolute_residuals.empty() || fitted_floors.empty())
        {
            return select_with_quadratic(
                model, "plane fit produced no finite inlier metrics");
        }
        model.residual_p95 = quantile(std::move(absolute_residuals), 0.95);
        const auto floor_extents = std::minmax_element(
            fitted_floors.begin(), fitted_floors.end());
        model.fitted_floor_min = *floor_extents.first;
        model.fitted_floor_max = *floor_extents.second;
        model.median_floor_offset =
            quantile(std::move(fitted_floors), 0.50) - options.fixed_floor_z;
        if (!std::isfinite(model.residual_p95) ||
            !std::isfinite(model.tilt_deg) ||
            !std::isfinite(model.fitted_floor_min) ||
            !std::isfinite(model.fitted_floor_max) ||
            !std::isfinite(model.median_floor_offset) ||
            !std::isfinite(model.sample_major_span) ||
            !std::isfinite(model.sample_minor_span) ||
            !std::isfinite(model.inlier_major_span) ||
            !std::isfinite(model.inlier_minor_span))
        {
            return select_with_quadratic(
                model, "plane fit produced non-finite metrics");
        }

        std::string plane_failure;
        const double max_floor_deviation = std::max(
            std::abs(model.fitted_floor_min - options.fixed_floor_z),
            std::abs(model.fitted_floor_max - options.fixed_floor_z));
        if (model.inlier_ratio < options.min_inlier_ratio)
        {
            plane_failure = "inlier ratio below quality gate";
        }
        else if (!inlier_coverage.valid ||
                 model.inlier_major_span <
                     options.min_xy_span * options.min_inlier_span_fraction ||
                 model.inlier_minor_span < options.min_inlier_minor_span)
        {
            plane_failure = "inlier xy coverage below quality gate";
        }
        else if (model.tilt_deg > options.max_tilt_deg)
        {
            plane_failure = "tilt exceeds quality gate";
        }
        else if (max_floor_deviation > options.max_floor_offset)
        {
            plane_failure = "fitted floor too far from configured floor";
        }
        else if (model.residual_p95 > options.max_residual_p95)
        {
            plane_failure = "residual p95 exceeds quality gate";
        }

        if (plane_failure.empty())
        {
            model.adaptive = true;
            model.fallback_reason.clear();
        }

        return select_with_quadratic(model, plane_failure);
    }
}  // namespace anubis_mapping
