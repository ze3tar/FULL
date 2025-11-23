# RL-Enhanced APF-RRT

This repository contains an augmented artificial potential field (APF) guided
RRT planner with reinforcement learning (RL) enhancements, benchmarking
utilities, smoothing modules, and ROS integration helpers for robotics
experiments.

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
  component map that links every major module.
- `IMPLEMENTATION_ROADMAP.md` enumerates the outstanding tasks across the five
  phases (ROS foundation through dynamic integration). Use it to track what
  remains before a full deployment.
- `ML_ENHANCEMENT_ARCHITECTURE.md` and `ROS_INTEGRATION_GUIDE.md` already spell
  out how the learning components, ROS bridge, and MoveIt scene connect; the
  code matches those docs so you can follow along while finishing the remaining
  roadmap boxes.

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
