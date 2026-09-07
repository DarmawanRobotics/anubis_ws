cmake_minimum_required(VERSION 3.10)
project(anubis_mapping)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(apriltag_msgs REQUIRED)
find_package(anubis_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters kdtree search)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(scan_descriptor REQUIRED)

# GTSAM and OpenSSL are not consistently available as a rosdep key across
# platforms in this workspace's history -- found directly, matching how
# they were installed (apt: libgtsam-dev, libssl-dev). See package.xml.
find_package(GTSAM REQUIRED)
find_package(OpenSSL REQUIRED)

# [PATCH -- fixes a real class of build error, not a style preference]
# darmawan_ws's own CMakeLists.txt defines ONE shared dependency list and
# applies it to every single target uniformly, rather than a precise
# per-target list. This package initially tried the "precise" approach
# instead, and it kept surfacing one missing dependency at a time across
# several real colcon builds (pcl_conversions, nav_msgs, sensor_msgs on
# pcd2grid_node; rclcpp on a test target that directly includes
# pcd2grid.h; etc.) -- each miss only visible by actually compiling on a
# real ROS2 install, which the environment used to write this package
# did not have. Matching the original's "one list, apply everywhere"
# pattern trades a small amount of precision (some targets link a few
# packages they don't strictly use) for not repeating that same class of
# error target-by-target.
set(ANUBIS_MAPPING_ROS_DEPS
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  tf2
  tf2_ros
  pcl_conversions
  apriltag_msgs
  anubis_interfaces
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS})

# ------------------------------------------------------------------------
# Vendored FAST-LIO2 + ikd-Tree + MTK core (src/slam/src's original,
# unmodified except the one project-wide namespace rename applied to
# every ported file: robot::slam -> anubis_mapping).
# ------------------------------------------------------------------------
add_library(anubis_mapping_lio_core
  src/process/imu_process.cpp
  src/process/lidar_process.cpp
  src/use_ikfom.cpp
  src/ikd_tree/ikd_tree.cpp
  src/common/math.cpp
)
target_link_libraries(anubis_mapping_lio_core
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)

# ------------------------------------------------------------------------
# Ported custom modules -- each verified by compiling + running its
# existing gtest alongside the port (see docs/memos for the log; 120
# tests passing as of this package's first cut).
# ------------------------------------------------------------------------
add_library(anubis_mapping_modules
  src/keyframe_store.cpp
  src/loop_closure.cpp
  src/map_z_drift_correction.cpp
  src/map_leveling_sidecar.cpp
  src/map_artifact_digest.cpp
  src/map_artifact_transaction.cpp
  src/pcd2grid_ground_plane.cpp
)
target_link_libraries(anubis_mapping_modules
  ${PCL_LIBRARIES}
  scan_descriptor::scan_descriptor
  gtsam
  OpenSSL::Crypto
)

# [2026-08-14 in darmawan_ws] Offline PCD -> occupancy-grid re-cut tool
# (fixes the "floor painted as a wall" issue). Standalone CLI, not part
# of the live mapping node -- entry point ported from
# darmawan_ws's test/pcd2pgm_tool.cpp.
add_executable(pcd2grid_node
  src/pcd2pgm_tool.cpp
  src/pcd2grid.cpp
  src/pcd2grid_ground_plane.cpp
)
target_link_libraries(pcd2grid_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
ament_target_dependencies(pcd2grid_node ${ANUBIS_MAPPING_ROS_DEPS})


# ------------------------------------------------------------------------
# mapping_alg: the orchestrating node. Wires the FAST-LIO2 core +
# custom modules above together, plus the AprilTag anchor integration
# (rewritten from darmawan_ws's ArUco version -- apriltag_ros has no
# pose field on its detection message, so the tag pose is obtained via
# a TF lookup instead; see the [PATCH] comments in mapping_alg.cpp).
# This is the one file in this package that was NOT bulk-copied and
# NOT compile-verified in a sandbox (no rclcpp/PCL-ROS integration
# available there) -- a real `colcon build` is this file's first
# actual compile. See README.md.
# ------------------------------------------------------------------------
add_executable(mapping_alg_node src/mapping_alg.cpp src/mapping_node_main.cpp)
target_include_directories(mapping_alg_node PRIVATE include)
target_link_libraries(mapping_alg_node
  anubis_mapping_lio_core
  anubis_mapping_modules
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)
ament_target_dependencies(mapping_alg_node ${ANUBIS_MAPPING_ROS_DEPS})

install(TARGETS anubis_mapping_lio_core anubis_mapping_modules pcd2grid_node mapping_alg_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)

# ------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)

  # Pure logic, no PCL/GTSAM -- lio_time_guard and imu_init_gate are
  # header-only.
  ament_add_gtest(test_lio_time_guard test/lio_time_guard_test.cpp)
  ament_add_gtest(test_imu_init_gate test/imu_init_gate_test.cpp)

  ament_add_gtest(test_keyframe_store test/keyframe_store_test.cpp
    src/keyframe_store.cpp)
  target_link_libraries(test_keyframe_store
    ${PCL_LIBRARIES} scan_descriptor::scan_descriptor)

  ament_add_gtest(test_map_z_drift_correction
    test/map_z_drift_correction_test.cpp src/map_z_drift_correction.cpp)
  target_link_libraries(test_map_z_drift_correction ${PCL_LIBRARIES})

  ament_add_gtest(test_map_artifact_digest
    test/map_artifact_digest_test.cpp src/map_artifact_digest.cpp
    src/map_leveling_sidecar.cpp)
  target_link_libraries(test_map_artifact_digest
    ${PCL_LIBRARIES} OpenSSL::Crypto)

  ament_add_gtest(test_map_leveling_sidecar
    test/map_leveling_sidecar_test.cpp src/map_leveling_sidecar.cpp
    src/map_artifact_digest.cpp)
  target_link_libraries(test_map_leveling_sidecar
    ${PCL_LIBRARIES} OpenSSL::Crypto)

  ament_add_gtest(test_map_artifact_transaction
    test/map_artifact_transaction_test.cpp src/map_artifact_transaction.cpp
    src/map_artifact_digest.cpp src/map_leveling_sidecar.cpp)
  target_link_libraries(test_map_artifact_transaction
    ${PCL_LIBRARIES} OpenSSL::Crypto)

  ament_add_gtest(test_loop_closure test/loop_closure_test.cpp
    src/loop_closure.cpp src/keyframe_store.cpp)
  target_link_libraries(test_loop_closure
    ${PCL_LIBRARIES} scan_descriptor::scan_descriptor gtsam)

  ament_add_gtest(test_pcd2grid_height test/pcd2grid_height_test.cpp
    src/pcd2grid_ground_plane.cpp)
  target_link_libraries(test_pcd2grid_height ${PCL_LIBRARIES})

  ament_add_gtest(test_pcd2grid_free_space test/pcd2grid_free_space_test.cpp
    src/pcd2grid_ground_plane.cpp)
  target_link_libraries(test_pcd2grid_free_space ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
  ament_target_dependencies(test_pcd2grid_free_space ${ANUBIS_MAPPING_ROS_DEPS})
endif()

ament_export_include_directories(include)
ament_export_libraries(anubis_mapping_lio_core anubis_mapping_modules)
ament_package()