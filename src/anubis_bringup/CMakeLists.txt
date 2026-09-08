cmake_minimum_required(VERSION 3.8)
project(anubis_bringup)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/_common.sh
  scripts/bringup_mapping.sh
  scripts/bringup_autonomous.sh
  scripts/stop_all.sh
  scripts/estop.sh
  scripts/set_map.sh
  DESTINATION share/${PROJECT_NAME}/scripts
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
