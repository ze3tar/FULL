cmake_minimum_required(VERSION 3.8)
project(rm65_planner)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rm65_kinematics REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME} src/apf_rrt_planner.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen rm65_kinematics)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_package()
