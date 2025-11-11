# ROS Integration Guide

This guide walks through deploying the ML-enhanced APF-RRT planner in a ROS
/ MoveIt workspace using the provided bridge utilities.

## Prerequisites
- ROS Noetic or ROS 2 Foxy + MoveIt (adjust launch arguments as needed)
- Robot description for RM65-6F loaded via `robot_state_publisher`
- Python 3.8+ environment with this repository installed
- Required Python packages (`pip install -r requirements.txt` if available)

## 1. Environment Setup
1. Clone the repository into your catkin/colcon workspace `src/` directory.
2. Install dependencies (inside a virtualenv is recommended):
   ```bash
   pip install numpy torch gymnasium stable-baselines3 matplotlib pandas yaml
   ```
3. Build your workspace (`catkin_make` or `colcon build`).
4. Source the workspace (`source devel/setup.bash` or `install/setup.bash`).

## 2. Exporting Paths for ROS
Run the enhanced baseline script to generate ROS-friendly waypoints:
```bash
python3 baseline_enhanced.py -- (script prompts)  # or run_experiment() manually
```
Outputs of interest:
- `path_improved_ros.yaml`, `path_improved_pruned_ros.yaml`
- `obstacles.csv`, `obstacles.marker`

## 3. Launching the Planner Bridge
1. Start the MoveIt pipeline with the RM65-6F configuration and RViz:
   ```bash
   roslaunch apf_rrt_planner.launch use_sim_time:=true
   ```
2. In a separate terminal, run the bridge node to publish trajectories:
   ```bash
   rosrun ros_moveit_bridge.py --path path_improved_pruned_ros.yaml
   ```
   The bridge:
   - Parses YAML waypoints into `moveit_msgs/RobotTrajectory`
   - Publishes visualization markers for obstacles and path
   - Optionally accepts live updates from the RL planner (see next section)

## 4. Integrating the RL Planner
Use the CLI to execute the PPO-enhanced planner and stream results to ROS:
```bash
python3 rl_enhanced_apf_rrt.py test --model models/best_model.zip --plot
```
- To run continuous replanning with dynamic obstacles, extend the bridge to
  subscribe to predictions from `DynamicObstacleManager` and publish updated
  trajectories via `/move_group/goal`.

## 5. Validation Checklist
- [ ] RViz shows the planned path (blue) and original nodes (grey scatter)
- [ ] Move group successfully executes the trajectory
- [ ] Dynamic obstacle topics (`/predicted_obstacles`) update at 20 Hz
- [ ] Planning time per replanning cycle < 50 ms (see benchmark CLI)

## 6. Troubleshooting
- **Missing markers** – verify `obstacles.marker` is loaded and frame IDs match
  the MoveIt planning frame (`world` by default).
- **Planner stalls** – ensure PPO checkpoints exist in `models/` and match the
  `ScenarioConfig` difficulty used in deployment.
- **Import errors** – run `python3 quick_test.py` to validate the environment and
  follow the installation hints printed for missing dependencies.

## 7. Next Steps
- Automate exports with a ROS action server wrapping `RLEnhancedPlanner`.
- Connect the LSTM predictor by publishing `/dynamic_obstacles` predictions for
  the bridge to consume.
- Capture benchmark logs with `comprehensive_comparison.py` for regression
  testing before hardware trials.
