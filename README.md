# RL-Enhanced APF-RRT

This repository contains an augmented artificial potential field (APF) guided
RRT planner with reinforcement learning (RL) enhancements, benchmarking
utilities, smoothing modules, and ROS integration helpers for robotics
experiments. It bundles everything from deterministic baselines through
real-time replanning utilities so you can benchmark, train, and deploy the
stack end-to-end.

## Project Snapshot
- **Baseline reproduction** – `baseline_enhanced.py` mirrors the canonical
  APF-RRT planner and exports ROS-friendly paths for benchmarking.
- **ML upgrades** – `rl_enhanced_apf_rrt.py`, `obstacle_predictor.py`, and
  `pso_path_smoother.py` add PPO-based tuning, LSTM obstacle forecasting, and
  PSO smoothing to the pipeline.
- **Integration artifacts** – launch files, MoveIt bridge code, benchmark
  datasets, and smoke/CI tests are all checked in so you can validate the stack
  end-to-end.

## Current Progress
- `SUMMARY.md` captures measured gains (≈44% faster planning, ≈31% fewer nodes,
  ≈9% shorter paths) comparing the RL-enhanced planner to the baseline, plus a
  component map that links every major module. The metrics reflect the
  integrated PPO, PSO, and LSTM upgrades already merged into this branch.
- `IMPLEMENTATION_ROADMAP.md` enumerates the outstanding tasks across the five
  phases (ROS foundation through dynamic integration). Use it to track what
  remains before a full deployment; completed items are marked in place.
- `ML_ENHANCEMENT_ARCHITECTURE.md` and `ROS_INTEGRATION_GUIDE.md` already spell
  out how the learning components, ROS bridge, and MoveIt scene connect; the
  code matches those docs so you can follow along while finishing the remaining
  roadmap boxes.
- `ROS_INTEGRATION_GUIDE.md` and `apf_rrt_planner.launch` document the current
  MoveIt and rosbridge wiring that is live on this branch, so the README usage
  below reflects the latest connection points.

## Repository Structure
### Documentation & Planning
| Path | Purpose |
| ---- | ------- |
| `README.md` | Entry point with status overview, structure map, and usage instructions. |
| `SUMMARY.md` | Snapshot of metrics, component map, and next actions. |
| `IMPLEMENTATION_ROADMAP.md` | Phase-by-phase checklist covering ROS foundation through dynamic deployment. |
| `ML_ENHANCEMENT_ARCHITECTURE.md` | Deep dive on how APF-RRT integrates PPO, LSTM prediction, and PSO smoothing. |
| `ROS_INTEGRATION_GUIDE.md` | Step-by-step instructions for launching the planner inside a ROS/MoveIt workspace. |

### Core Planners & ML Modules
| Path | Purpose |
| ---- | ------- |
| `baseline_enhanced.py` | Reproducible APF-RRT baseline with CLI prompts to export ROS paths and metrics. |
| `rl_enhanced_apf_rrt.py` | Main CLI for PPO training (`train`), testing, and benchmarking the RL-enhanced planner. |
| `config_space_apf_rrt.py` | Colab-friendly wrapper exposing the gym environment plus helper APIs for remote training. |
| `obstacle_predictor.py` | Dataset builders, LSTM definition, trainer, and runtime `DynamicObstacleManager`. |
| `pso_path_smoother.py` | Particle swarm optimiser that refines waypoint sequences under collision and joint limits. |
| `comprehensive_comparison.py` | Batch benchmarking harness comparing planners, optional PSO smoothing, and saving plots/logs. |

### Tooling & Automation
| Path | Purpose |
| ---- | ------- |
| `quick_test.py` | Lightweight dependency smoke test that prints missing packages/hardware warnings. |
| `run_all_tests.py` | Aggregates the `tests/` suite to mimic CI locally. |
| `colab_setup.py` | Convenience script to bootstrap Colab (install requirements, mount drive, kick off training). |
| `requirements.txt` | Frozen Python dependency list used by Colab, ROS workspaces, and tests. |

### ROS & Integration Assets
| Path | Purpose |
| ---- | ------- |
| `apf_rrt_planner.launch` | Launch file that spins up the MoveIt pipeline plus planner bridge hooks. |
| `ros_moveit_bridge.py` | Converts planned trajectories to MoveIt goals and publishes RViz markers/updates. |

### Data, Models, and Benchmarks
| Path | Purpose |
| ---- | ------- |
| `path_points_baseline.csv` | Saved baseline waypoint chain for offline analysis or smoothing experiments. |
| `path_points_improved.csv` | RL-enhanced (and optionally smoothed) path sample for comparison. |
| `benchmarks/final_benchmark.{csv,json,png}` | Benchmark table, JSON log, and plot bundle emitted by the comparison harness. |
| `models/best_model.zip.zip` | Latest PPO checkpoint ready for benchmarking/testing. |

### Tests
| Path | Purpose |
| ---- | ------- |
| `tests/test_baseline.py` | Unit tests covering the deterministic APF-RRT baseline planner. |
| `tests/test_comparison.py` | Ensures the benchmarking harness runs and emits expected outputs. |
| `tests/test_dependencies.py` | Verifies optional dependencies are either installed or gracefully skipped. |
| `tests/test_hardware.py` | Guards hardware-specific flags (MoveIt / GPU availability). |
| `tests/test_lstm.py` | Exercises the obstacle predictor model/trainer APIs. |
| `tests/test_pso.py` | Checks PSO smoothing behaviour and constraints. |
| `tests/test_rl_env.py` | Covers the gym environment, observation space, and PPO hooks. |

## Training
Train the PPO agent via the CLI:

```bash
python rl_enhanced_apf_rrt.py train --timesteps 500000 --n-envs 4
```

The enhanced critic configuration is enabled by default (toggle with
`--no-critic-strong`). Keeping `--critic-strong` active applies reward
normalisation and a deeper value network, yielding smoother value estimation and
more stable training curves.

### Training in Google Colab
To reproduce the full training run on Google Colab, execute the following
commands in a notebook cell:

```bash
!git clone https://github.com/ze3tar/FULL.git
%cd FULL/
!pip install -r requirements.txt
!python rl_enhanced_apf_rrt.py train \
    --timesteps 5000000 \
    --n-envs 4 \
    --difficulty medium \
    --dynamic-prob 0.45
```

Adjust `--difficulty` and `--dynamic-prob` to explore curriculum variants or
dynamic obstacle frequencies as needed. Mount Google Drive beforehand if you
want checkpoints to persist between sessions.

## Evaluation
Load checkpoints for benchmarking or quick qualitative tests:

```bash
python rl_enhanced_apf_rrt.py benchmark --model models/best_model.zip
python rl_enhanced_apf_rrt.py test --plot
```

Refer to `SUMMARY.md` and `ML_ENHANCEMENT_ARCHITECTURE.md` for a deeper dive
into the system design and component interactions.

## Installation
Clone the repository and install the Python dependencies (Python 3.10+ is
recommended):

```bash
git clone https://github.com/ze3tar/FULL.git
cd FULL
pip install -r requirements.txt
```

If you plan to run the ROS bridge, ensure ROS Noetic or ROS 2 Humble is
available in your environment (the scripts only rely on the ROS Python APIs and
`rosbridge_server`).

## Usage
### Quick CLI recipes
- **Smoke test dependencies:**
  ```bash
  python quick_test.py
  ```
- **Run the deterministic baseline planner and export a path:**
  ```bash
  python baseline_enhanced.py --export-path path_points_baseline.csv
  ```
- **Benchmark RL-enhanced planner vs. baseline (with plot + JSON log):**
  ```bash
  python comprehensive_comparison.py --model models/best_model.zip --plot
  ```
- **Train PPO with curriculum controls:**
  ```bash
  python rl_enhanced_apf_rrt.py train --timesteps 500000 --n-envs 4 --difficulty medium --dynamic-prob 0.35
  ```
- **Load a checkpoint for qualitative tests or quick benchmarking:**
  ```bash
  python rl_enhanced_apf_rrt.py benchmark --model models/best_model.zip
  python rl_enhanced_apf_rrt.py test --model models/best_model.zip --export-path path_points_improved.csv --plot
  ```
- **Compare baseline vs RL-enhanced planners and emit CSV summaries:**
  ```bash
  python rl_enhanced_apf_rrt.py compare \
      --model-path models/best_model.zip \
      --normalizer-path models/obs_normalizer.npz \
      --trials 60 \
      --output-dir benchmarks/
  ```
  The command prints a table with success/path metrics per scenario while writing
  `summary.csv` and `trials.csv` into `--output-dir` for downstream analysis.
- **Smooth an exported path with PSO:**
  ```bash
  python pso_path_smoother.py --input path_points_baseline.csv --output smoothed.csv
  ```
- **Run the full CI-equivalent suite locally:**
  ```bash
  python run_all_tests.py
  ```

### Dynamic environments and replanning
- **Simulate moving obstacles:** Import `DynamicEnvironmentSimulator` to spawn
  random spheres and stream obstacle states into your planner or ROS scene. The
  simulator is deterministic when you seed NumPy and supports bounded
  accelerations for stress testing. See `dynamic_environment_simulator.py` for
  a ready-made callback API.
- **Trigger real-time replans:** Use `RealTimeReplanner` to monitor predicted
  obstacles and replan when deviation or collision risk exceeds configured
  thresholds. The class exposes ROS helpers (`publish_path_to_rviz` and
  `sync_planning_scene`) so you can publish a replanned path and mirror
  obstacles into MoveIt without additional boilerplate. Pair it with the
  dynamic simulator or your own obstacle predictions.

### ROS/Motion planning linkage
- Follow the `ROS_INTEGRATION_GUIDE.md` quick start for wiring `rosbridge_server`
  and MoveIt. The shipped `apf_rrt_planner.launch` file reflects the current
  branch configuration used by the benchmarking scripts.
- Export a path (baseline or RL-enhanced), then publish through the MoveIt
  bridge:
  ```bash
  python baseline_enhanced.py --export-path path_points_baseline.csv
  python ros_moveit_bridge.py --path path_points_baseline.csv --rosbridge-url ws://localhost:9090
  ```
  Swap in RL checkpoints as needed:
  ```bash
  python rl_enhanced_apf_rrt.py test --model models/best_model.zip --export-path path_points_improved.csv
  python ros_moveit_bridge.py --path path_points_improved.csv --rosbridge-url ws://localhost:9090
  ```

### Assets and branches at a glance
- **Pretrained model:** `models/best_model.zip.zip` is the latest PPO checkpoint
  aligned with the metrics in `SUMMARY.md`. Point the CLI at this file for a
  plug-and-play benchmark run.
- **Benchmark artefacts:** `benchmarks/final_benchmark.*` captures CSV, JSON,
  and plot outputs produced by `comprehensive_comparison.py` on the current
  branch.
- **Roadmap alignment:** The README usage above mirrors the live code paths in
  `main`—no feature flags are required to toggle the PPO critic, PSO smoothing,
  or LSTM obstacle forecasting discussed in the roadmap and summary.

## Linking with ROS via rosbridge
You can connect the planner outputs to a running ROS graph using
`rosbridge_server` so that waypoints flow into your MoveIt or RViz workflows.

1. Start rosbridge in your ROS workspace (example for ROS Noetic):
   ```bash
   roslaunch rosbridge_server rosbridge_websocket.launch
   ```
   Ensure the websocket port (default `9090`) is accessible from the machine
   running this repository.
2. Export a path from the planner (baseline or RL-enhanced):
   ```bash
   python baseline_enhanced.py --export-path path_points_baseline.csv
   # or
   python rl_enhanced_apf_rrt.py test --model models/best_model.zip --export-path path_points_improved.csv
   ```
3. Publish the path through the bridge using the included MoveIt helper:
   ```bash
   python ros_moveit_bridge.py --path path_points_improved.csv --rosbridge-url ws://localhost:9090
   ```
   The bridge script opens a websocket to rosbridge, converts the CSV waypoints
   into `geometry_msgs/PoseArray`, and publishes to the configured topic for
   visualization or downstream execution.

For detailed MoveIt configuration notes, see `ROS_INTEGRATION_GUIDE.md` and the
`apf_rrt_planner.launch` file.
