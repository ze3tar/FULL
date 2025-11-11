# Implementation Roadmap

This roadmap expands on the quick reference in `SUMMARY.md` and provides task
breakdowns for each milestone.

## Phase 1 – Foundation (ROS Integration)
- [ ] Import RM65-6F URDF into MoveIt and validate joint limits.
- [ ] Finalise `config_space_apf_rrt.py` environment reproduction of the paper.
- [ ] Validate static obstacle scenarios in RViz using exported YAML paths.
- [ ] Record baseline metrics matching the reference results.

## Phase 2 – RL Enhancement
- [ ] Generate ≥10,000 training episodes via `train_agent`.
- [ ] Run PPO training on GPU (expect 2–3 days for convergence to target metrics).
- [ ] Evaluate the trained agent (`rl_enhanced_apf_rrt.py benchmark`) and log
      improvements in planning time, nodes, and path length.

## Phase 3 – Obstacle Prediction
- [ ] Generate synthetic obstacle trajectories (`generate_training_data`).
- [ ] Train `ObstaclePredictorLSTM` to reach ≥85% 3-step accuracy.
- [ ] Integrate `DynamicObstacleManager` into the planner loop for forecasts.

## Phase 4 – PSO Smoothing
- [ ] Benchmark PSO smoother on stored paths and tune coefficients.
- [ ] Integrate smoothing pipeline into `comprehensive_comparison.py`.
- [ ] Validate 15–20% reduction in path cost and 50% curvature reduction.

## Phase 5 – Dynamic Environments & Integration
- [ ] Implement `dynamic_environment_simulator.py` for moving obstacle scenarios.
- [ ] Add `realtime_replanner.py` for continuous replanning hooks.
- [ ] Combine RL planner, predictor, and PSO smoothing in ROS/MoveIt demos.
- [ ] Target metrics: ≥85% success in dynamic scenes, ~50 ms replanning, ≥98%
      collision avoidance.

## Deliverables Checklist
- [ ] Architecture documentation (`ML_ENHANCEMENT_ARCHITECTURE.md`).
- [ ] ROS deployment guide (`ROS_INTEGRATION_GUIDE.md`).
- [ ] Benchmark reports (`benchmarks/*.json`, `comparison_plots.png`).
- [ ] Trained PPO model and LSTM weights stored in `models/`.
