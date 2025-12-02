cmake_minimum_required(VERSION 3.8)
project(rm65_moveit_plugin)

find_package(ament_cmake REQUIRED)
find_package(moveit_core QUIET)
find_package(moveit_ros_planning QUIET)
find_package(pluginlib QUIET)
find_package(rm65_planner REQUIRED)

if(NOT moveit_core_FOUND OR NOT moveit_ros_planning_FOUND OR NOT pluginlib_FOUND)
  message(WARNING "MoveIt dependencies not found; skipping build of rm65_moveit_plugin.")
  ament_package()
  return()
endif()

include_directories(include)
add_library(${PROJECT_NAME} SHARED src/apf_rrt_planner_plugin.cpp)
target_link_libraries(${PROJECT_NAME} moveit_core moveit_ros_planning rm65_planner)
pluginlib_export_plugin_description_file(moveit_core plugin_description.xml)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(moveit_core moveit_ros_planning pluginlib rm65_planner)
ament_package()
