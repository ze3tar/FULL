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

### Training in Google Colab

To reproduce the full training run on Google Colab, execute the following
commands in a notebook cell:

```bash
!git clone https://github.com/<your-fork>/RL-Enhanced-APF-RRT.git
%cd RL-Enhanced-APF-RRT
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
