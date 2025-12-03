# ROS 2 Humble Porting Report for RM65 Stack

This document captures the ROS 2 Humble migration status for the RM65 manipulation stack and shows the expected ROS 2 layouts, manifests, build files, and launch configuration per package. Each section points to the in-repo files that implement the migration so you can cross-check and extend the stack.

## Workspace Layout
Place all packages under `ros2_ws/src` (the repository already follows this pattern under `src/`). Key packages:
- `full_interfaces` – ROS 2 interface definitions (srv for path planning).
- `rm65_description` – URDF and meshes (MoveIt2-ready).
- `rm65_control` – ros2_control YAML and launch for controller manager.
- `rm65_kinematics` – Eigen-based forward/inverse kinematics utilities used by the planner plugin.
- `rm65_moveit_config` – MoveIt2 configuration, controllers, and demo launch.
- `rm65_moveit_plugin` – APF-RRT MoveIt2 planning context plugin.
- `rm65_planner` – Standalone APF-RRT server node powered by `rclcpp` and custom interfaces.
- `full_rviz_bridge` – Visualization helpers (launch + RViz configs).
- `full_integration` – System-level launch for combining controllers, MoveIt2, planner, and RViz2.

## Package Manifests (package.xml)
- Interfaces renamed to lowercase ROS 2 style: see `src/full_interfaces/package.xml` for the canonical interface manifest using `rosidl_default_generators` and `rosidl_interface_packages` membership.
- Planner consumes the renamed interfaces: `src/rm65_planner/package.xml` depends on `full_interfaces`, `rclcpp`, `Eigen3`, `rm65_kinematics`, `trajectory_msgs`, and `sensor_msgs`.
- Control, MoveIt2 config, kinematics, and plugin packages already use format 3 manifests with `ament_cmake` and ROS 2 dependencies.

## Build Files
- Interfaces: `src/full_interfaces/CMakeLists.txt` invokes `rosidl_generate_interfaces` for `srv/PlanPath.srv` and exports the package via `ament_package()`.
- Planner: `src/rm65_planner/CMakeLists.txt` finds `full_interfaces`, `rclcpp`, `Eigen3`, `rm65_kinematics`, and message packages; it builds `rm65_plan_path_server` against the shared APF-RRT library and exports the include directory and library for downstream use.
- MoveIt2 plugin: `src/rm65_moveit_plugin/CMakeLists.txt` builds the planning context plugin and installs `plugin_description.xml` plus YAML config for the planning pipeline.
- Control: `src/rm65_control/CMakeLists.txt` installs ros2_control YAML and launch files using `ament_cmake`.

## Interfaces
- Service definition: `src/full_interfaces/srv/PlanPath.srv` remains unchanged, now generated under the lowercase package namespace (`full_interfaces::srv::PlanPath`). Include it via `#include "full_interfaces/srv/plan_path.hpp"` in C++ nodes.

## Planner Node
- The APF-RRT server uses `rclcpp` and the renamed interfaces; see `src/rm65_planner/src/plan_path_server.cpp` for the ROS 2 service wiring and trajectory construction. It exposes `/rm65/plan_path` to accept start/goal joint states and returns a `trajectory_msgs/JointTrajectory`.

## MoveIt2 Integration
- The MoveIt2 configuration package (`src/rm65_moveit_config`) contains ROS 2 controller bindings (`config/moveit_controllers.yaml`), planning pipeline configuration (`config/moveit_planners.yaml`), and demo launch (`launch/demo.launch.py`). Use this package as the anchor for the Motion Planning Framework.
- The custom MoveIt2 planning plugin in `src/rm65_moveit_plugin` registers via `plugin_description.xml` and reads settings from `config/plugin_settings.yaml` to slot into the MoveIt2 planning pipeline.

## ros2_control
- Controller composition is defined in `src/rm65_control/ros2_controllers.yaml` and launched through `launch/rm65_control.launch.py`. Point the MoveIt2 controllers file at the exported controller names when deploying with real hardware or simulation.

## Full-System Launch
- A unified launch entry point `full_integration/rm65_full_system.launch.py` (under `src/full_integration/launch`) should start:
  - The robot description and state publishers.
  - The ros2_control controller manager and joint state broadcaster.
  - The MoveIt2 `move_group` node with the APF-RRT plugin enabled.
  - The APF-RRT planning server (`rm65_plan_path_server`).
  - RViz2 configured from `full_rviz_bridge`.

## Verification Steps
1. Source ROS 2: `source /opt/ros/humble/setup.bash`.
2. Build the workspace: `colcon build --symlink-install` (from the workspace root).
3. Source overlays: `source install/setup.bash`.
4. Launch MoveIt2 demo: `ros2 launch rm65_moveit_config demo.launch.py`.
5. Launch the full system (controllers + MoveIt2 + planner + RViz2): `ros2 launch full_integration rm65_full_system.launch.py`.

Following these instructions keeps the RM65 stack consistent with ROS 2 Humble expectations while preserving the custom APF-RRT planning capabilities.
