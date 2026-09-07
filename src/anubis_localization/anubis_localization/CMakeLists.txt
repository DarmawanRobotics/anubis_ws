cmake_minimum_required(VERSION 3.5)
project(anubis_localization)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

cmake_policy(SET CMP0074 NEW)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ndt_omp REQUIRED)
find_package(fast_gicp REQUIRED)
find_package(apriltag_msgs REQUIRED)
find_package(anubis_interfaces REQUIRED)
find_package(scan_descriptor REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# [PATCH] One shared dependency list applied to every ROS-dependent
# target below, matching the original file's own pattern (it used a
# single ${dependencies} list project-wide) -- deliberately not
# "precise per-target" lists, which cost anubis_mapping several rounds
# of one-missing-dependency-at-a-time colcon build failures before this
# lesson was learned. robots_dog_msgs -> anubis_interfaces,
# apriltag_msgs added (ArUco -> AprilTag migration); everything else
# unchanged from the original ${dependencies} list.
set(ANUBIS_LOCALIZATION_ROS_DEPS
  rclcpp
  std_srvs
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  pcl_ros
  pcl_conversions
  sensor_msgs
  geometry_msgs
  nav_msgs
  ndt_omp
  fast_gicp
  apriltag_msgs
  anubis_interfaces
  scan_descriptor
)

########################
## message generation ##
########################
rosidl_generate_interfaces(${PROJECT_NAME}
  msg/ScanMatchingStatus.msg
DEPENDENCIES
  std_msgs
  geometry_msgs
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp
)
ament_export_dependencies(rosidl_default_runtime)

###########
## Build ##
###########

add_library(localization_lib SHARED
  src/localization/pose_estimator.cpp
)
ament_target_dependencies(localization_lib
  ${ANUBIS_LOCALIZATION_ROS_DEPS}
)
target_link_libraries(localization_lib "${cpp_typesupport_target}")
target_include_directories(localization_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(localization_node
  apps/localization_nodelet.cpp
  src/localization/pose_estimator.cpp
  src/localization/static_imu_init.cpp
  src/localization/global_localization.cpp
  src/localization/global_localization_csv.cpp
  src/localization/global_localization_gravity.cpp
  src/localization/global_localization_refinement.cpp
  src/localization/global_localization_spatial.cpp
  src/localization/global_localization_episode.cpp
  src/localization/global_localization_static_confirm.cpp
  src/localization/global_localization_integration.cpp
  src/localization/map_artifact_manifest.cpp
  src/localization/level0_distance_field.cpp
  src/localization/level0_global_search.cpp
  src/localization/level1_coarse_search.cpp
  src/localization/retrieval_recall.cpp
)
ament_target_dependencies(localization_node
  rcl_interfaces
  std_msgs
  ${ANUBIS_LOCALIZATION_ROS_DEPS}
)
target_include_directories(localization_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(localization_node
  "${cpp_typesupport_target}"
  OpenSSL::Crypto
)

install(
  TARGETS
    localization_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(
  TARGETS
    localization_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    launch
    rviz
    config
  DESTINATION share/${PROJECT_NAME}
)

install(
        FILES ../run_localization.sh
        DESTINATION .
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(PROGRAMS
  tools/global_localization_eval.py
  tools/tracking_fault_injector.py
  tools/tracking_safety_bag_eval.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  ament_add_gtest(pose_prediction_test
    test/pose_prediction_test.cpp
  )
  target_include_directories(pose_prediction_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(pose_prediction_test Eigen3::Eigen)

  ament_add_gtest(tracking_health_monitor_test
    test/tracking_health_monitor_test.cpp
  )
  target_include_directories(tracking_health_monitor_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  ament_add_gtest(sensor_data_validity_test
    test/sensor_data_validity_test.cpp
  )
  target_include_directories(sensor_data_validity_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  ament_add_gtest(ndt_optimizer_diagnostics_test
    test/ndt_optimizer_diagnostics_test.cpp
  )
  target_include_directories(ndt_optimizer_diagnostics_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(ndt_optimizer_diagnostics_test Eigen3::Eigen)

  ament_add_gtest(point_cloud_time_diagnostics_test
    test/point_cloud_time_diagnostics_test.cpp
  )
  target_include_directories(point_cloud_time_diagnostics_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(point_cloud_time_diagnostics_test sensor_msgs)

  ament_add_gtest(global_localization_math_test
    test/global_localization_math_test.cpp
  )
  target_include_directories(global_localization_math_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(global_localization_math_test Eigen3::Eigen)

  ament_add_gtest(global_localization_types_test
    test/global_localization_types_test.cpp
  )
  target_include_directories(global_localization_types_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(global_localization_types_test rclcpp)
  target_link_libraries(global_localization_types_test Eigen3::Eigen)

  ament_add_gtest(global_localization_csv_test
    test/global_localization_csv_test.cpp
    src/localization/global_localization_csv.cpp
  )
  target_include_directories(global_localization_csv_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(global_localization_csv_test rclcpp)
  target_link_libraries(global_localization_csv_test Eigen3::Eigen)

  ament_add_gtest(global_localization_interface_test
    test/global_localization_interface_test.cpp
    src/localization/global_localization.cpp
    src/localization/global_localization_gravity.cpp
    src/localization/global_localization_refinement.cpp
    src/localization/global_localization_spatial.cpp
    src/localization/level0_distance_field.cpp
    src/localization/level0_global_search.cpp
    src/localization/level1_coarse_search.cpp
    src/localization/retrieval_recall.cpp
  )
  target_include_directories(global_localization_interface_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(global_localization_interface_test
    rclcpp
    pcl_ros
    scan_descriptor
  )
  target_link_libraries(global_localization_interface_test Eigen3::Eigen)

  ament_add_gtest(retrieval_recall_test
    test/retrieval_recall_test.cpp
    src/localization/retrieval_recall.cpp
    src/localization/global_localization_gravity.cpp
    src/localization/level0_distance_field.cpp
  )
  target_include_directories(retrieval_recall_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_target_dependencies(retrieval_recall_test rclcpp pcl_ros scan_descriptor)
  target_link_libraries(retrieval_recall_test Eigen3::Eigen)

  ament_add_gtest(level0_distance_field_test
    test/level0_distance_field_test.cpp
    src/localization/level0_distance_field.cpp
  )
  target_include_directories(level0_distance_field_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(level0_distance_field_test
    rclcpp
    pcl_ros
  )
  target_link_libraries(level0_distance_field_test Eigen3::Eigen)

  ament_add_gtest(global_localization_gravity_test
    test/global_localization_gravity_test.cpp
    src/localization/global_localization_gravity.cpp
  )
  target_include_directories(global_localization_gravity_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(global_localization_gravity_test rclcpp)
  target_link_libraries(global_localization_gravity_test Eigen3::Eigen)

  ament_add_gtest(level0_global_search_test
    test/level0_global_search_test.cpp
    src/localization/level0_distance_field.cpp
    src/localization/level0_global_search.cpp
  )
  target_include_directories(level0_global_search_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(level0_global_search_test
    rclcpp
    pcl_ros
  )
  target_link_libraries(level0_global_search_test Eigen3::Eigen)

  ament_add_gtest(level1_coarse_search_test
    test/level1_coarse_search_test.cpp
    src/localization/global_localization_gravity.cpp
    src/localization/level0_distance_field.cpp
    src/localization/level1_coarse_search.cpp
  )
  target_include_directories(level1_coarse_search_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(level1_coarse_search_test
    rclcpp
    pcl_ros
  )
  target_link_libraries(level1_coarse_search_test Eigen3::Eigen)

  ament_add_gtest(global_localization_refinement_test
    test/global_localization_refinement_test.cpp
    src/localization/global_localization_refinement.cpp
  )
  target_include_directories(global_localization_refinement_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(global_localization_refinement_test
    rclcpp
    pcl_ros
  )
  target_link_libraries(global_localization_refinement_test Eigen3::Eigen)

  ament_add_gtest(global_localization_spatial_test
    test/global_localization_spatial_test.cpp
    src/localization/global_localization_spatial.cpp
  )
  target_include_directories(global_localization_spatial_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(global_localization_spatial_test rclcpp)
  target_link_libraries(global_localization_spatial_test Eigen3::Eigen)

  ament_add_gtest(global_localization_integration_test
    test/global_localization_integration_test.cpp
    src/localization/global_localization_episode.cpp
    src/localization/global_localization_integration.cpp
  )
  target_include_directories(global_localization_integration_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(global_localization_integration_test rclcpp)
  target_link_libraries(global_localization_integration_test Eigen3::Eigen)

  ament_add_gtest(global_localization_static_confirm_test
    test/global_localization_static_confirm_test.cpp
    src/localization/global_localization_static_confirm.cpp
  )
  target_include_directories(global_localization_static_confirm_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_target_dependencies(global_localization_static_confirm_test rclcpp)
  target_link_libraries(global_localization_static_confirm_test Eigen3::Eigen)

  ament_add_gtest(map_artifact_manifest_test
    test/map_artifact_manifest_test.cpp
    src/localization/map_artifact_manifest.cpp
  )
  target_include_directories(map_artifact_manifest_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(map_artifact_manifest_test OpenSSL::Crypto)

  add_test(
    NAME global_localization_eval_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_global_localization_eval.py
  )
  set_tests_properties(global_localization_eval_test PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools"
  )
  ament_add_pytest_test(
    extrinsic_config_consistency_test
    test/test_extrinsic_config_consistency.py
    TIMEOUT 30
  )
endif()

ament_package()
