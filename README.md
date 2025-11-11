# RL-Enhanced APF-RRT

This repository contains an augmented artificial potential field (APF) guided RRT
planner with reinforcement learning enhancements, benchmarking utilities, and
integration helpers for robotics experiments.

## Training

Train the PPO agent via the CLI:

```bash
python rl_enhanced_apf_rrt.py train --timesteps 500000 --n-envs 4
```

The enhanced critic configuration is enabled by default (toggle with
`--no-critic-strong`). Keeping `--critic-strong` active applies reward
normalisation and a deeper value network, yielding smoother value estimation and
more stable training curves.

## Evaluation

Load checkpoints for benchmarking or quick qualitative tests:

```bash
python rl_enhanced_apf_rrt.py benchmark --model models/best_model.zip
python rl_enhanced_apf_rrt.py test --plot
```

Refer to `SUMMARY.md` and `ML_ENHANCEMENT_ARCHITECTURE.md` for a deeper dive
into the system design and component interactions.
