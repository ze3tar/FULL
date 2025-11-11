#!/usr/bin/env python3
"""
APF-guided RRT in Configuration Space (Non-ROS version)
For PPO-based training and simulation outside ROS
"""

import math
import random
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ConfigSpaceAPF_RRT(gym.Env):
    """APF-guided RRT environment compatible with PPO"""

    def __init__(self):
        super(ConfigSpaceAPF_RRT, self).__init__()

        # 6-DOF robot (same joint limits as RM65-6F)
        self.n_joints = 6
        self.q_min = np.array([-np.pi, -np.pi/2, -np.pi, -np.pi, -np.pi, -np.pi])
        self.q_max = np.array([ np.pi,  np.pi/2,  np.pi,  np.pi,  np.pi,  np.pi])

        # Algorithm parameters
        self.step_size = 0.3
        self.goal_threshold = 0.05
        self.max_step_size = 0.1  # RL action magnitude cap

        # APF parameters
        self.K_att = 1.0
        self.K_rep = 0.3
        self.d0 = 1.5

        # Obstacles (simple static spheres)
        self.obstacles = [
            (np.array([0.5, 0.7, -0.3, 0.0, 0.3, 0.0]), 0.5),
            (np.array([0.2, 0.5, -0.5, -0.1, 0.5, 0.1]), 0.4)
        ]

        # Gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints * 2,), dtype=np.float32)

        self.state = None
        self.goal = None
        self.reset()

    # -------------------------------------------------
    # Core RRT math utilities
    # -------------------------------------------------
    def config_distance(self, q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))

    def compute_apf_force(self, q_near, q_goal):
        """Attractive + repulsive potential field forces"""
        v_att = np.array(q_goal) - np.array(q_near)
        d_att = np.linalg.norm(v_att)
        F_att = self.K_att * (v_att / (d_att + 1e-9)) if d_att > 0 else np.zeros(self.n_joints)

        F_rep = np.zeros(self.n_joints)
        for q_obs, r in self.obstacles:
            v_obs = np.array(q_near) - np.array(q_obs)
            d_obs = np.linalg.norm(v_obs) - r
            if d_obs <= self.d0 and d_obs > 1e-6:
                mag = self.K_rep * (1.0 / (d_obs ** 2)) * (1.0/d_obs - 1.0/self.d0)
                F_rep += mag * (v_obs / (np.linalg.norm(v_obs) + 1e-9))

        return F_att + F_rep

    # -------------------------------------------------
    # Gym environment interface
    # -------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(-1, 1, size=self.n_joints)
        self.goal = np.zeros(self.n_joints)
        obs = np.concatenate([self.state, self.goal])
        return obs, {}

    def step(self, action):
        # Limit action magnitude
        action = np.clip(action, -1, 1) * self.max_step_size
        self.state = np.clip(self.state + action, -1.0, 1.0)

        # Compute distance to goal
        distance = np.linalg.norm(self.state - self.goal)

        # Reward design
        reward = -distance - 0.1 * np.linalg.norm(action)
        done = False

        # Success condition
        if distance < self.goal_threshold:
            reward += 100.0
            done = True

        # Clip episode length
        truncated = False

        # Observation = current + goal
        obs = np.concatenate([self.state, self.goal])

        return obs, reward, done, truncated, {}

    def render(self):
        print(f"State: {self.state}, Goal dist: {np.linalg.norm(self.state - self.goal):.3f}")


# -------------------------------------------------
# Manual test
# -------------------------------------------------
if __name__ == "__main__":
    env = ConfigSpaceAPF_RRT()
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        if done:
            print("Reached goal!")
            break
    print(f"Episode reward: {total_reward:.2f}")
