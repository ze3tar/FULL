cmake_minimum_required(VERSION 3.8)
project(FULL_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PlanPath.srv"
  DEPENDENCIES sensor_msgs trajectory_msgs
)

ament_package()
