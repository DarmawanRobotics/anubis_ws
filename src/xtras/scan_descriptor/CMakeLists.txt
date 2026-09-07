cmake_minimum_required(VERSION 3.8)
project(scan_descriptor)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/descriptor.cpp
  src/database.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

add_executable(scd_info tools/scd_info.cpp)
target_link_libraries(scd_info PRIVATE ${PROJECT_NAME})

# scd_info must NOT be exported: ament_target_dependencies() in consumer
# packages links every exported target, and an executable is not a library
# (observed link failure: "-lscan_descriptor::scd_info: No such file").
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS scd_info
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_libraries(${PROJECT_NAME})
ament_export_include_directories(include)
ament_export_dependencies(Eigen3)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(descriptor_test test/descriptor_test.cpp)
  target_link_libraries(descriptor_test ${PROJECT_NAME})
  ament_add_gtest(database_test test/database_test.cpp)
  target_link_libraries(database_test ${PROJECT_NAME})
endif()

ament_package()
