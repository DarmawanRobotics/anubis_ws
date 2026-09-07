#ifndef LEVEL0_DISTANCE_FIELD_HPP
#define LEVEL0_DISTANCE_FIELD_HPP

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "localization/global_localization_types.hpp"

namespace localization {

enum class Level0ProjectionClass : uint8_t {
  Valid,
  BoundaryUnknown,
  Invalid,
};

struct Level0ScanPreparationStats {
  int input_points = 0;
  int finite_points = 0;
  int height_points = 0;
  int range_points = 0;
  int sampled_points = 0;
  bool valid = false;
};

struct Level0Score {
  bool valid = false;
  int input_point_count = 0;
  int valid_projection_points = 0;
  int boundary_unknown_points = 0;
  int invalid_projection_points = 0;
  uint64_t lookup_count = 0;
  double valid_projection_ratio = 0.0;
  double boundary_unknown_ratio = 0.0;
  double invalid_projection_ratio = 0.0;
  double mean_truncated_distance = std::numeric_limits<double>::infinity();
  double score = std::numeric_limits<double>::infinity();
};

bool applyLevel0ScoreToCandidate(const Level0Score& score,
                                 GlobalPoseCandidate& candidate);

void accumulateLevel0Summary(const Level0Score& score,
                             GLSummaryCounts& summary);

class Level0DistanceField {
public:
  bool build(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& global_map,
             const GLParams& params);

  bool prepareScan(
      const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& scan_base,
      const Eigen::Matrix3d& R_level_base,
      double base_ground_offset,
      const GLParams& params,
      pcl::PointCloud<pcl::PointXYZI>::Ptr& scan_level,
      Level0ScanPreparationStats& stats) const;

  bool scoreHypothesis(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& scan_level,
                       const Eigen::Matrix4d& T_map_level,
                       const GLParams& params,
                       Level0Score& score) const;

  bool empty() const { return distance_cells_.empty(); }
  int width() const { return width_; }
  int height() const { return height_; }
  double resolution() const { return resolution_; }
  double floorZ() const { return floor_z_; }
  double minX() const { return min_x_; }
  double minY() const { return min_y_; }
  double maxX() const { return max_x_; }
  double maxY() const { return max_y_; }

private:
  int indexFor(double x, double y) const;
  Level0ProjectionClass classify(double x, double y) const;
  double distanceAt(int index) const;

  double resolution_ = 0.0;
  double floor_z_ = 0.0;
  double min_x_ = 0.0;
  double min_y_ = 0.0;
  double max_x_ = 0.0;
  double max_y_ = 0.0;
  double boundary_padding_m_ = 0.0;
  int width_ = 0;
  int height_ = 0;
  std::vector<double> distance_cells_;
};

}  // namespace localization

#endif  // LEVEL0_DISTANCE_FIELD_HPP
