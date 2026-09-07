#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

#include "pcd2grid_height.h"

namespace anubis_mapping
{
    // Ground-plane fitting is an optional classifier enhancement, but when it
    // is enabled its fallback policy is an independent contract. In
    // particular, map_leveling.enforce_quality_gates controls save-time
    // leveling/geometry gates and must not silently change this decision;
    // final pose-safety validation is a separate unconditional contract.
    inline bool ground_plane_fallback_allowed(
        bool ground_plane_enabled, bool ground_plane_required,
        bool fitted_adaptive) noexcept
    {
        return !ground_plane_enabled || fitted_adaptive || !ground_plane_required;
    }

    constexpr int8_t kGridUnknown = -1;
    constexpr int8_t kGridFree = 0;
    constexpr int8_t kGridOccupied = 100;

    inline bool valid_min_points_occupied(uint32_t value)
    {
        return value >= 1U;
    }

    // MappingAlg places the configured grid prefix below a transaction
    // staging directory.  Keep that prefix to one portable path component so
    // an absolute path or a parent traversal cannot escape the staging tree.
    inline bool is_safe_pcd2grid_basename(const std::string& value)
    {
        if (value.empty() || value == "." || value == "..")
        {
            return false;
        }
        if (value.find_first_of("/\\:") != std::string::npos ||
            value.find('\0') != std::string::npos)
        {
            return false;
        }
        for (const unsigned char character : value)
        {
            if (character < 0x20U || character == 0x7fU)
            {
                return false;
            }
        }
        return true;
    }

    inline bool is_ground_free_height(
        double point_height, double min_height, double obstacle_min_height)
    {
        return std::isfinite(point_height) && std::isfinite(min_height) &&
            std::isfinite(obstacle_min_height) &&
            min_height < obstacle_min_height && point_height >= min_height &&
            point_height < obstacle_min_height;
    }

    inline bool is_ground_free_observation(
        double point_map_z, double floor_z_map,
        double min_height, double obstacle_min_height)
    {
        if (!std::isfinite(point_map_z) || !std::isfinite(floor_z_map) ||
            !std::isfinite(min_height) || !std::isfinite(obstacle_min_height) ||
            min_height >= obstacle_min_height)
        {
            return false;
        }
        // Compare in map coordinates so a point exactly on the obstacle lower
        // bound cannot enter both bands due to subtract-then-compare rounding.
        return point_map_z >= map_z_from_floor_height(floor_z_map, min_height) &&
            point_map_z <
                map_z_from_floor_height(floor_z_map, obstacle_min_height);
    }

    inline int8_t classify_pcd2grid_cell_height(
        uint32_t obstacle_point_count, double min_obstacle_height,
        double overhead_clearance, bool has_free_space_evidence,
        uint32_t min_points_occupied = 2U)
    {
        if (!valid_min_points_occupied(min_points_occupied))
        {
            return kGridUnknown;
        }
        if (obstacle_point_count >= min_points_occupied)
        {
            return is_overhead_height(
                       min_obstacle_height, overhead_clearance)
                ? kGridFree
                : kGridOccupied;
        }
        if (obstacle_point_count > 0U)
        {
            return kGridFree;
        }
        return has_free_space_evidence ? kGridFree : kGridUnknown;
    }

    // Extended classifier used when per-keyframe evidence is available.  A
    // small number of aggregate points is not enough to override repeated
    // free observations; conversely, a wall observed in multiple frames keeps
    // occupied precedence.  Zero frame counts retain the legacy aggregate
    // behavior for offline PCDs that do not carry scan poses.
    inline int8_t classify_pcd2grid_cell_height(
        uint32_t obstacle_point_count, double min_obstacle_height,
        double overhead_clearance, bool has_free_space_evidence,
        uint32_t min_points_occupied, uint32_t obstacle_frames,
        uint32_t ground_frames, uint32_t ray_frames,
        uint32_t min_obstacle_frames, uint32_t min_free_frames,
        double free_to_obstacle_frame_ratio)
    {
        if (!valid_min_points_occupied(min_points_occupied))
        {
            return kGridUnknown;
        }
        if (obstacle_point_count >= min_points_occupied)
        {
            if (is_overhead_height(min_obstacle_height, overhead_clearance))
            {
                return kGridFree;
            }
            const bool has_frame_evidence = obstacle_frames > 0U;
            const bool persistent_obstacle = !has_frame_evidence ||
                obstacle_frames >= std::max(1U, min_obstacle_frames);
            const uint32_t free_frames = std::max(ground_frames, ray_frames);
            const bool strong_free = has_frame_evidence &&
                free_frames >= std::max(1U, min_free_frames) &&
                (free_to_obstacle_frame_ratio <= 0.0 ||
                 static_cast<double>(free_frames) >=
                     free_to_obstacle_frame_ratio *
                         static_cast<double>(std::max(1U, obstacle_frames)));
            if (strong_free && obstacle_frames <
                    std::max(1U, min_obstacle_frames))
            {
                return kGridFree;
            }
            return persistent_obstacle ? kGridOccupied : kGridFree;
        }
        if (obstacle_point_count > 0U)
        {
            return kGridFree;
        }
        return has_free_space_evidence ? kGridFree : kGridUnknown;
    }

    inline int8_t classify_pcd2grid_cell(
        uint32_t obstacle_point_count, double min_obstacle_map_z,
        double floor_z_map, double overhead_clearance,
        bool has_free_space_evidence, uint32_t min_points_occupied = 2U)
    {
        return classify_pcd2grid_cell_height(
            obstacle_point_count, min_obstacle_map_z - floor_z_map,
            overhead_clearance, has_free_space_evidence,
            min_points_occupied);
    }

    // [2026-08-30 门口混合格修复·定版] 低点是否构成阻挡(占用)。四个条件
    // 共存,任一满足即阻挡:
    //   ① 低点计数:low_count >= min_low_points(默认 2 —— 细东西可能只有
    //      1~2 个占用点,不上调);
    //      且 ② 低点占比:low_count/total >= min_low_ratio(默认 0.25)
    //      —— 同时满足说明低点构成实体矮障碍;
    //   ③ 低点持续性:low_frames > low_frame_persistence(默认 10)——
    //      即使数量少/占比低,多帧同位置持续出现也优先保留(可能是墙脚/
    //      门框);实测门口门轨 5~10 帧被放行,墙柱低点 37 帧被保留;
    //   ④ 该格没有地面/射线空闲证据 —— 有空闲证据证明可通行。
    // 边界约定:低点定义为 height <= overhead_clearance_z(与"最低点 > 0.55
    // 才悬空"的既有边界一致,0.55 整点按低点计);total 为高度/半径滤波后的
    // obstacle_counts;高度使用相对地面平面的 h,不是绝对 map.z。
    inline bool low_points_block_cell(
        uint32_t low_count, uint32_t total, uint32_t low_frames,
        bool has_free_evidence, uint32_t min_low_points,
        double min_low_ratio, uint32_t low_frame_persistence) noexcept
    {
        if (low_count == 0U)
        {
            return false;
        }
        if (low_count >= min_low_points &&
            static_cast<double>(low_count) /
                    static_cast<double>(std::max(1U, total)) >=
                min_low_ratio)
        {
            return true;
        }
        return low_frames > low_frame_persistence || !has_free_evidence;
    }

    // A low-point gate only tells us that the vertical profile is ambiguous;
    // it does not prove that the high returns are overhead.  This predicate
    // contains the part of the gate that is independent of free-space
    // evidence.  The caller combines it with a visibility (ray-crossing)
    // observation before clearing a cell.
    inline bool low_points_form_obstacle(
        uint32_t low_count, uint32_t total, uint32_t low_frames,
        uint32_t min_low_points, double min_low_ratio,
        uint32_t low_frame_persistence) noexcept
    {
        if (low_count == 0U)
        {
            return false;
        }
        const bool count_and_ratio =
            low_count >= min_low_points &&
            static_cast<double>(low_count) /
                    static_cast<double>(std::max(1U, total)) >=
                min_low_ratio;
        return count_and_ratio || low_frames > low_frame_persistence;
    }

    // A ground return can be used to clear a projected high-return cell only
    // when the ray actually continues beyond that cell.  A ray that merely
    // terminates in the cell is consistent with a wall base and is therefore
    // deliberately not counted.  Both a minimum number of independent scan
    // frames and a ratio to the obstacle-observation frames are required so a
    // single accidental pass cannot punch a hole through a wall.
    inline bool overhead_ray_crossing_sufficient(
        uint32_t crossed_frames, uint32_t obstacle_frames,
        uint32_t min_crossed_frames, double min_crossed_ratio) noexcept
    {
        if (crossed_frames < std::max(1U, min_crossed_frames))
        {
            return false;
        }
        if (!std::isfinite(min_crossed_ratio) || min_crossed_ratio <= 0.0)
        {
            return true;
        }
        return static_cast<double>(crossed_frames) >=
            min_crossed_ratio *
                static_cast<double>(std::max(1U, obstacle_frames));
    }

    inline int8_t classify_pcd2grid_cell(
        uint32_t obstacle_point_count, double min_obstacle_map_z,
        double floor_z_map, double overhead_clearance,
        bool has_free_space_evidence, uint32_t min_points_occupied,
        uint32_t obstacle_frames, uint32_t ground_frames,
        uint32_t ray_frames, uint32_t min_obstacle_frames,
        uint32_t min_free_frames, double free_to_obstacle_frame_ratio)
    {
        return classify_pcd2grid_cell_height(
            obstacle_point_count, min_obstacle_map_z - floor_z_map,
            overhead_clearance, has_free_space_evidence,
            min_points_occupied, obstacle_frames, ground_frames, ray_frames,
            min_obstacle_frames, min_free_frames,
            free_to_obstacle_frame_ratio);
    }

    // Low-point-gate variant of the frame-evidence classifier: the overhead
    // decision uses the low-point count/ratio/persistence/free-evidence gate
    // instead of the raw minimum height, so a stray threshold return mixed
    // into a lintel cell no longer blocks the whole 2-D column.
    inline int8_t classify_pcd2grid_cell_height(
        uint32_t obstacle_point_count, uint32_t low_point_count,
        uint32_t low_point_frames, double overhead_clearance,
        bool has_free_space_evidence, uint32_t min_points_occupied,
        uint32_t obstacle_frames, uint32_t ground_frames,
        uint32_t ray_frames, uint32_t min_obstacle_frames,
        uint32_t min_free_frames, double free_to_obstacle_frame_ratio,
        uint32_t min_low_points, double min_low_ratio,
        uint32_t low_frame_persistence)
    {
        if (!valid_min_points_occupied(min_points_occupied))
        {
            return kGridUnknown;
        }
        if (obstacle_point_count >= min_points_occupied)
        {
            if (!low_points_block_cell(
                    low_point_count, obstacle_point_count, low_point_frames,
                    has_free_space_evidence, min_low_points, min_low_ratio,
                    low_frame_persistence))
            {
                return kGridFree;
            }
            const bool has_frame_evidence = obstacle_frames > 0U;
            const bool persistent_obstacle = !has_frame_evidence ||
                obstacle_frames >= std::max(1U, min_obstacle_frames);
            const uint32_t free_frames = std::max(ground_frames, ray_frames);
            const bool strong_free = has_frame_evidence &&
                free_frames >= std::max(1U, min_free_frames) &&
                (free_to_obstacle_frame_ratio <= 0.0 ||
                 static_cast<double>(free_frames) >=
                     free_to_obstacle_frame_ratio *
                         static_cast<double>(std::max(1U, obstacle_frames)));
            if (strong_free && obstacle_frames <
                    std::max(1U, min_obstacle_frames))
            {
                return kGridFree;
            }
            return persistent_obstacle ? kGridOccupied : kGridFree;
        }
        if (obstacle_point_count > 0U)
        {
            return kGridFree;
        }
        return has_free_space_evidence ? kGridFree : kGridUnknown;
    }

    // Integer Bresenham traversal. Returning false from visitor stops at the
    // current cell, which lets callers conservatively stop rays at obstacles.
    template <typename Visitor>
    void trace_grid_line(
        int x0, int y0, int x1, int y1, Visitor&& visitor)
    {
        const int dx = std::abs(x1 - x0);
        const int sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0);
        const int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;

        while (true)
        {
            if (!visitor(x0, y0))
            {
                return;
            }
            if (x0 == x1 && y0 == y1)
            {
                return;
            }
            const int twice_error = 2 * error;
            if (twice_error >= dy)
            {
                error += dy;
                x0 += sx;
            }
            if (twice_error <= dx)
            {
                error += dx;
                y0 += sy;
            }
        }
    }
}  // namespace anubis_mapping
