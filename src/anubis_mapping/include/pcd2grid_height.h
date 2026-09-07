#pragma once

#include <cmath>

namespace anubis_mapping
{
    inline bool valid_pcd2grid_height_contract(
        double floor_z_map, double min_height, double max_height,
        double overhead_clearance)
    {
        return std::isfinite(floor_z_map) && std::isfinite(min_height) &&
            std::isfinite(max_height) && min_height < max_height &&
            std::isfinite(overhead_clearance);
    }

    inline double map_z_from_floor_height(double floor_z_map, double height)
    {
        return floor_z_map + height;
    }

    inline bool is_overhead_cell(
        double min_point_map_z, double floor_z_map,
        double overhead_clearance)
    {
        return overhead_clearance > 0.0 &&
            min_point_map_z - floor_z_map > overhead_clearance;
    }

    inline bool is_overhead_height(
        double min_point_height, double overhead_clearance)
    {
        return overhead_clearance > 0.0 &&
            min_point_height > overhead_clearance;
    }
}  // namespace anubis_mapping
