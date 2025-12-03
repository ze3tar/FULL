cmake_minimum_required(VERSION 3.8)
project(rm65_moveit_plugin)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_planning_interface REQUIRED)
find_package(rm65_planner REQUIRED)
find_package(rm65_kinematics REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(full_integration REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  src/apf_rrt_planner_plugin.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  pluginlib
  moveit_core
  moveit_planning_interface
  rm65_planner
  rm65_kinematics
  Eigen3
  rclcpp
  ament_index_cpp
  yaml-cpp
  full_integration
)

target_link_libraries(${PROJECT_NAME} ${moveit_core_LIBRARIES} rm65_planner rm65_kinematics)
target_link_libraries(${PROJECT_NAME} yaml-cpp)

pluginlib_export_plugin_description_file(moveit_core plugin_description.xml)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
