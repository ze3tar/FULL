cmake_minimum_required(VERSION 3.8)
project(rm65_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(full_interfaces REQUIRED)
find_package(rm65_kinematics REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  src/apf_rrt_planner.cpp
)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen rm65_kinematics)
ament_target_dependencies(${PROJECT_NAME} Eigen3)

target_include_directories(${PROJECT_NAME} PUBLIC include)

add_executable(rm65_plan_path_server src/plan_path_server.cpp)
target_link_libraries(rm65_plan_path_server ${PROJECT_NAME})
ament_target_dependencies(rm65_plan_path_server rclcpp full_interfaces trajectory_msgs sensor_msgs Eigen3)

install(TARGETS ${PROJECT_NAME} rm65_plan_path_server
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(Eigen3 rm65_kinematics)

ament_package()
