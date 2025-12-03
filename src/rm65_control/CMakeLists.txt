cmake_minimum_required(VERSION 3.8)
project(rm65_control)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(FILES
  ros2_controllers.yaml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
