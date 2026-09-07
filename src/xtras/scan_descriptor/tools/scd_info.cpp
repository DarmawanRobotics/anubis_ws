#include <iomanip>
#include <iostream>
#include <limits>

#include "scan_descriptor/database.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: scd_info <map_scd.bin>\n";
    return 2;
  }
  scan_descriptor::DescriptorDatabase database;
  std::string error;
  if (!database.load(argv[1], &error)) {
    std::cerr << "load failed: " << error << "\n";
    return 1;
  }
  const auto& config = database.config();
  Eigen::Vector3d minimum = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::infinity());
  Eigen::Vector3d maximum = Eigen::Vector3d::Constant(
      -std::numeric_limits<double>::infinity());
  for (size_t index = 0; index < database.size(); ++index) {
    const Eigen::Vector3d position =
        database.at(index).T_map_lidar.block<3, 1>(0, 3);
    minimum = minimum.cwiseMin(position);
    maximum = maximum.cwiseMax(position);
  }
  std::cout << "format=SCDB version=1 normalization=ground_relative\n"
            << "rings=" << config.ring_count
            << " sectors=" << config.sector_count
            << " range=[" << config.min_range << "," << config.max_range << "]"
            << " z=[" << config.z_min << "," << config.z_max << "]\n"
            << "keyframes=" << database.size() << "\n";
  if (!database.empty()) {
    std::cout << std::fixed << std::setprecision(3)
              << "bbox_min=" << minimum.transpose() << "\n"
              << "bbox_max=" << maximum.transpose() << "\n";
  }
  return 0;
}
