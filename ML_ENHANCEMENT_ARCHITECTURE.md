# ML Enhancement Architecture

This document explains how the APF-guided RRT baseline is extended with machine
learning components.  It complements the inline documentation within each
module.

## High-Level Flow

```mermaid
graph TD
    A[Scenario Config] --> B[APF-RRT Gym Environment]
    B --> C[PPO Policy (train_agent)]
    C --> D[Trained Agent Weights]
    D --> E[RLEnhancedPlanner.plan]
    E --> F[PSOPathSmoother]
    E --> G[ObstaclePredictor]
    G --> E
    F --> H[ROS MoveIt Bridge]
```

1. **Scenario sampling** – `ScenarioConfig` defines obstacle density, dynamics,
   joint limits, and seeds.
2. **Gym environment** – `APFRRTEnv` exposes planner state, collision metrics,
   and APF parameters as part of the observation vector.
3. **PPO training** – `train_agent` sets up vectorised Gym environments,
   Stable-Baselines3 PPO, and reward checkpoints for Colab/GPU sessions.
4. **Planner execution** – `RLEnhancedPlanner.plan` streams state through the
   trained policy, performs APF-guided expansion, and reports metrics.
5. **Obstacle prediction** – `ObstaclePredictor` delivers multi-step forecasts
   that feed into dynamic obstacle handling before each planning iteration.
6. **Path smoothing** – `PSOPathSmoother` refines the resulting waypoint chain
   while respecting collision, joint limit, and velocity constraints.
7. **ROS bridge** – `ros_moveit_bridge.py` converts planned trajectories into
   ROS action goals and publishes RViz visualisation markers.

## Module Responsibilities

### `rl_enhanced_apf_rrt.py`
- Houses environment dynamics, PPO helpers, benchmarking utilities, and the
  `RLEnhancedPlanner` runtime wrapper.
- Provides backwards compatible aliases (`RLEnhancedAPF_RRT`,
  `APF_RRT_Environment`) for older scripts.
- Exposes CLI entry points for training, testing, and benchmarking.

### `config_space_apf_rrt.py`
- Wraps the Gym environment into a Colab-friendly API (`ConfigSpaceAPF_RRT`).
- Supplies helper functions for training (`train_colab_ready_agent`) and
  evaluation (`evaluate_agent`).

### `obstacle_predictor.py`
- Implements dataset utilities, LSTM model definition, training loop, and a
  high-level predictor interface plus `DynamicObstacleManager` for deployment.

### `pso_path_smoother.py`
- Provides a configurable PSO optimiser capable of enforcing obstacle
  clearances, joint limits, and velocity bounds.
- Includes `visualize_smoothing` for 3D inspection of original vs smoothed
  paths.

### `comprehensive_comparison.py`
- Runs benchmarks across planners, integrates optional PSO smoothing, and
  produces plots/JSON reports for downstream analysis.

### `ros_moveit_bridge.py`
- Translates planner outputs into MoveIt motion goals, handles TF frames, and
  publishes RViz markers for obstacles and trajectories.

## Data Artefacts
- `models/` – PPO checkpoints, including `best_model.zip` and `final_model.zip`.
- `path_points_*.csv` – waypoint dumps from baseline experiments.
- `path_*_ros.yaml` – ROS-friendly exports ready for MoveIt execution.
- `benchmarks/*.json` – stored benchmark runs via `comprehensive_comparison.py`.

## Training Tips
- Prefer GPU execution (`torch.cuda.is_available()`) for the PPO agent.  The
  default hyper-parameters target 300k timesteps in roughly 2 hours on a
  T4/V100 GPU.
- Use `ConfigSpaceSettings(seed=...)` to reproduce benchmark environments when
  comparing baseline and RL-enhanced planners.
- Enable dynamic probability during evaluation to test policy robustness before
  integrating LSTM predictions.
