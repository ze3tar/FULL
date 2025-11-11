#!/usr/bin/env python3
"""
RL-Enhanced APF-RRT Planner using PPO (Proximal Policy Optimization)
Learns optimal APF parameters and sampling strategies for improved performance
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import time
import pickle

# Import your baseline APF-RRT
import sys
sys.path.append('.')
from config_space_apf_rrt import ConfigSpaceAPF_RRT


class APF_RRT_Environment(gym.Env):
    """
    Gym environment for training RL agent to optimize APF-RRT parameters
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, difficulty='medium', max_steps=100):
        super(APF_RRT_Environment, self).__init__()
        
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.current_step = 0
        
        # State space: [distance_to_goal, angle_to_goal, min_obstacle_dist,
        #                obstacle_density, K_att, K_rep, d0, step_size, nodes_count]
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, 0, 0, 0.1, 0.05, 0.5, 0.1, 0]),
            high=np.array([10, np.pi, 5, 1.0, 5.0, 2.0, 3.0, 1.0, 5000]),
            dtype=np.float32
        )
        
        # Action space: [delta_K_att, delta_K_rep, delta_d0, delta_step_size, goal_bias]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.3, -0.5, -0.2, -0.1]),
            high=np.array([0.5, 0.3, 0.5, 0.2, 0.1]),
            dtype=np.float32
        )
        
        # APF parameters (will be adjusted by RL)
        self.K_att = 1.0
        self.K_rep = 0.3
        self.d0 = 1.5
        self.step_size = 0.3
        self.goal_bias = 0.07
        
        # Planning state
        self.q_current = None
        self.q_goal = None
        self.obstacles = []
        self.nodes = []
        self.path_length = 0
        self.planning_time = 0
        
        self._setup_environment()
    
    def _setup_environment(self):
        """Generate random planning scenario based on difficulty"""
        # Random start and goal configurations (6-DOF)
        self.q_start = np.random.uniform(-np.pi, np.pi, 6)
        self.q_goal = np.random.uniform(-np.pi, np.pi, 6)
        self.q_current = self.q_start.copy()
        
        # Generate obstacles based on difficulty
        n_obstacles = {'easy': 2, 'medium': 4, 'hard': 6}[self.difficulty]
        self.obstacles = self._generate_obstacles(n_obstacles)
        
        # Reset metrics
        self.nodes = [self.q_start]
        self.path_length = 0
        self.planning_time = 0
    
    def _generate_obstacles(self, n):
        """Generate random obstacle positions in configuration space"""
        obstacles = []
        for _ in range(n):
            # Random configuration that's an obstacle
            q_obs = np.random.uniform(-np.pi, np.pi, 6)
            radius = np.random.uniform(0.3, 0.8)
            obstacles.append((q_obs, radius))
        return obstacles
    
    def _get_state(self):
        """Compute current state for RL agent"""
        # Distance to goal in config space
        d_to_goal = np.linalg.norm(self.q_current - self.q_goal)
        
        # Angle to goal (using first 3 joints for direction)
        v_to_goal = self.q_goal[:3] - self.q_current[:3]
        angle_to_goal = np.arctan2(v_to_goal[1], v_to_goal[0])
        
        # Minimum obstacle distance
        min_obs_dist = min([np.linalg.norm(self.q_current - q_obs) - r 
                           for q_obs, r in self.obstacles] + [10.0])
        
        # Obstacle density (obstacles within influence distance)
        nearby_obstacles = sum([1 for q_obs, r in self.obstacles 
                               if np.linalg.norm(self.q_current - q_obs) < self.d0])
        obstacle_density = nearby_obstacles / max(len(self.obstacles), 1)
        
        # Current parameters
        state = np.array([
            d_to_goal / 10.0,        # Normalized
            angle_to_goal / np.pi,
            min_obs_dist / 5.0,
            obstacle_density,
            self.K_att,
            self.K_rep,
            self.d0,
            self.step_size,
            len(self.nodes) / 1000.0  # Normalized node count
        ], dtype=np.float32)
        
        return state
    
    def _compute_apf_force(self):
        """Compute APF forces with current parameters"""
        # Attractive force
        v_att = self.q_goal - self.q_current
        d_att = np.linalg.norm(v_att)
        F_att = (self.K_att * (v_att / (d_att + 1e-9))) if d_att > 0 else np.zeros(6)
        
        # Repulsive force
        F_rep = np.zeros(6)
        for q_obs, r in self.obstacles:
            v_obs = self.q_current - q_obs
            d_obs = np.linalg.norm(v_obs) - r
            
            if d_obs <= self.d0 and d_obs > 1e-6:
                mag = self.K_rep * (1.0 / (d_obs ** 2)) * (1.0/d_obs - 1.0/self.d0)
                F_rep += mag * (v_obs / (np.linalg.norm(v_obs) + 1e-9))
        
        return F_att + F_rep
    
    def _check_collision(self, q):
        """Check if configuration collides with obstacles"""
        for q_obs, r in self.obstacles:
            if np.linalg.norm(q - q_obs) < r:
                return True
        return False
    
    def step(self, action):
        """Execute one planning step with RL-adjusted parameters"""
        self.current_step += 1
        
        # Apply RL action to adjust parameters
        self.K_att = np.clip(self.K_att + action[0], 0.1, 5.0)
        self.K_rep = np.clip(self.K_rep + action[1], 0.05, 2.0)
        self.d0 = np.clip(self.d0 + action[2], 0.5, 3.0)
        self.step_size = np.clip(self.step_size + action[3], 0.1, 1.0)
        self.goal_bias = np.clip(self.goal_bias + action[4], 0.0, 0.3)
        
        # Sample next configuration (goal-biased)
        if np.random.random() < self.goal_bias:
            q_rand = self.q_goal
        else:
            q_rand = np.random.uniform(-np.pi, np.pi, 6)
        
        # Find nearest node
        distances = [np.linalg.norm(n - q_rand) for n in self.nodes]
        q_near = self.nodes[np.argmin(distances)]
        
        # Compute direction with APF
        F_total = self._compute_apf_force()
        v_rand = q_rand - q_near
        v_rand_unit = v_rand / (np.linalg.norm(v_rand) + 1e-9)
        v_goal = self.q_goal - q_near
        v_goal_unit = v_goal / (np.linalg.norm(v_goal) + 1e-9)
        
        # Combine directions (weighted)
        combined = 0.5 * v_rand_unit + 0.3 * v_goal_unit + 0.2 * (F_total / (np.linalg.norm(F_total) + 1e-9))
        direction = combined / (np.linalg.norm(combined) + 1e-9)
        
        # Generate new configuration
        q_new = q_near + direction * self.step_size
        q_new = np.clip(q_new, -np.pi, np.pi)  # Joint limits
        
        # Compute reward
        reward = 0
        done = False
        
        # Check collision
        if self._check_collision(q_new):
            reward -= 5  # Collision penalty
        else:
            # Add node
            self.nodes.append(q_new)
            self.q_current = q_new
            
            # Progress reward
            old_dist = np.linalg.norm(q_near - self.q_goal)
            new_dist = np.linalg.norm(q_new - self.q_goal)
            progress = old_dist - new_dist
            reward += 10 * progress
            
            # Check if goal reached
            if new_dist < 0.2:
                reward += 100  # Success bonus
                done = True
                
            # Efficiency rewards
            reward -= 0.01 * len(self.nodes)  # Penalize too many nodes
            
            # Obstacle clearance bonus
            min_clearance = min([np.linalg.norm(q_new - q_obs) - r 
                                for q_obs, r in self.obstacles] + [10.0])
            if min_clearance > 0:
                reward += 2 * np.tanh(min_clearance)
        
        # Time penalty
        reward -= 0.1
        
        # Episode termination conditions
        if self.current_step >= self.max_steps:
            done = True
            if np.linalg.norm(self.q_current - self.q_goal) > 0.5:
                reward -= 50  # Failed to reach goal
        
        state = self._get_state()
        info = {
            'nodes': len(self.nodes),
            'distance_to_goal': np.linalg.norm(self.q_current - self.q_goal),
            'K_att': self.K_att,
            'K_rep': self.K_rep
        }
        
        return state, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        self.current_step = 0
        self._setup_environment()
        return self._get_state(), {}
    
    def render(self, mode='human'):
        """Render current state (optional)"""
        pass


class TrainingCallback(BaseCallback):
    """Callback for monitoring training progress"""
    
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Evaluate policy
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f}")
                self.model.save(f"{self.save_path}/best_model")
        
        return True


def train_rl_agent(total_timesteps=100000, n_envs=4):
    """
    Train PPO agent to optimize APF-RRT parameters
    
    Args:
        total_timesteps: Number of training steps
        n_envs: Number of parallel environments
    """
    print("="*60)
    print("Training RL Agent for APF-RRT Parameter Optimization")
    print("="*60)
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: APF_RRT_Environment(difficulty='medium') for _ in range(n_envs)])
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./ppo_apf_rrt_tensorboard/"
    )
    
    # Create callback
    callback = TrainingCallback(check_freq=1000, save_path="./models")
    
    # Train
    print(f"\nTraining for {total_timesteps} timesteps with {n_envs} parallel environments...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Save final model
    model.save("./models/final_model")
    print("Model saved to ./models/final_model")
    
    return model


class RLEnhancedAPF_RRT:
    """
    APF-RRT planner enhanced with trained RL agent
    """
    
    def __init__(self, model_path="./models/best_model.zip"):
        """Load trained RL model"""
        self.model = PPO.load(model_path)
        print(f"Loaded RL model from {model_path}")
        
        # Default parameters (will be adjusted by RL)
        self.K_att = 1.0
        self.K_rep = 0.3
        self.d0 = 1.5
        self.step_size = 0.3
        self.goal_bias = 0.07
    
    def plan(self, q_start, q_goal, obstacles, max_iters=5000):
        """
        Plan path using RL-enhanced APF-RRT
        
        Args:
            q_start: Start configuration (6-DOF)
            q_goal: Goal configuration (6-DOF)
            obstacles: List of (q_obs, radius) tuples
            max_iters: Maximum iterations
        
        Returns:
            path: List of configurations
            nodes: All explored nodes
            runtime: Planning time
            metrics: Planning metrics
        """
        print(f"\nRL-Enhanced Planning from {q_start} to {q_goal}")
        
        nodes = [q_start]
        parents = {0: None}
        start_time = time.time()
        
        # Initialize environment for RL decisions
        q_current = q_start
        
        for iteration in range(max_iters):
            # Get current state for RL
            state = self._get_state(q_current, q_goal, obstacles, nodes)
            
            # RL agent decides parameter adjustments
            action, _ = self.model.predict(state, deterministic=True)
            
            # Apply RL action
            self._apply_action(action)
            
            # Sample configuration (goal-biased)
            if np.random.random() < self.goal_bias:
                q_rand = q_goal
            else:
                q_rand = np.random.uniform(-np.pi, np.pi, 6)
            
            # Find nearest node
            distances = [np.linalg.norm(n - q_rand) for n in nodes]
            idx_near = int(np.argmin(distances))
            q_near = nodes[idx_near]
            
            # Compute APF-guided direction
            F_total = self._compute_apf_force(q_near, q_goal, obstacles)
            v_rand = q_rand - q_near
            v_rand_unit = v_rand / (np.linalg.norm(v_rand) + 1e-9)
            v_goal = q_goal - q_near
            v_goal_unit = v_goal / (np.linalg.norm(v_goal) + 1e-9)
            
            # Combine directions
            combined = (0.5 * v_rand_unit + 0.3 * v_goal_unit + 
                       0.2 * (F_total / (np.linalg.norm(F_total) + 1e-9)))
            direction = combined / (np.linalg.norm(combined) + 1e-9)
            
            # Generate new configuration
            q_new = q_near + direction * self.step_size
            q_new = np.clip(q_new, -np.pi, np.pi)
            
            # Check collision
            if not self._check_collision(q_new, obstacles):
                nodes.append(q_new)
                parents[len(nodes) - 1] = idx_near
                q_current = q_new
                
                # Check if goal reached
                if np.linalg.norm(q_new - q_goal) < 0.2:
                    print(f"✓ Goal reached in {iteration} iterations!")
                    
                    # Reconstruct path
                    path = [q_goal]
                    cur = len(nodes) - 1
                    while cur is not None:
                        path.append(nodes[cur])
                        cur = parents[cur]
                    path.reverse()
                    
                    runtime = time.time() - start_time
                    metrics = {
                        'nodes': len(nodes),
                        'iterations': iteration,
                        'K_att_final': self.K_att,
                        'K_rep_final': self.K_rep
                    }
                    
                    return path, nodes, runtime, metrics
        
        runtime = time.time() - start_time
        print(f"✗ Failed to reach goal after {max_iters} iterations")
        return None, nodes, runtime, {}
    
    def _get_state(self, q_current, q_goal, obstacles, nodes):
        """Get state for RL agent"""
        d_to_goal = np.linalg.norm(q_current - q_goal)
        v_to_goal = q_goal[:3] - q_current[:3]
        angle_to_goal = np.arctan2(v_to_goal[1], v_to_goal[0])
        
        min_obs_dist = min([np.linalg.norm(q_current - q_obs) - r 
                           for q_obs, r in obstacles] + [10.0])
        nearby_obstacles = sum([1 for q_obs, r in obstacles 
                               if np.linalg.norm(q_current - q_obs) < self.d0])
        obstacle_density = nearby_obstacles / max(len(obstacles), 1)
        
        state = np.array([
            d_to_goal / 10.0,
            angle_to_goal / np.pi,
            min_obs_dist / 5.0,
            obstacle_density,
            self.K_att,
            self.K_rep,
            self.d0,
            self.step_size,
            len(nodes) / 1000.0
        ], dtype=np.float32)
        
        return state
    
    def _apply_action(self, action):
        """Apply RL action to adjust parameters"""
        self.K_att = np.clip(self.K_att + action[0], 0.1, 5.0)
        self.K_rep = np.clip(self.K_rep + action[1], 0.05, 2.0)
        self.d0 = np.clip(self.d0 + action[2], 0.5, 3.0)
        self.step_size = np.clip(self.step_size + action[3], 0.1, 1.0)
        self.goal_bias = np.clip(self.goal_bias + action[4], 0.0, 0.3)
    
    def _compute_apf_force(self, q_near, q_goal, obstacles):
        """Compute APF force"""
        v_att = q_goal - q_near
        d_att = np.linalg.norm(v_att)
        F_att = (self.K_att * (v_att / (d_att + 1e-9))) if d_att > 0 else np.zeros(6)
        
        F_rep = np.zeros(6)
        for q_obs, r in obstacles:
            v_obs = q_near - q_obs
            d_obs = np.linalg.norm(v_obs) - r
            
            if d_obs <= self.d0 and d_obs > 1e-6:
                mag = self.K_rep * (1.0 / (d_obs ** 2)) * (1.0/d_obs - 1.0/self.d0)
                F_rep += mag * (v_obs / (np.linalg.norm(v_obs) + 1e-9))
        
        return F_att + F_rep
    
    def _check_collision(self, q, obstacles):
        """Check collision"""
        for q_obs, r in obstacles:
            if np.linalg.norm(q - q_obs) < r:
                return True
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RL-Enhanced APF-RRT')
    parser.add_argument('--mode', choices=['train', 'test'], default='test',
                       help='Train new model or test existing one')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Training timesteps')
    parser.add_argument('--model', type=str, default='./models/best_model.zip',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting RL training...")
        model = train_rl_agent(total_timesteps=args.timesteps)
        print("\n✓ Training complete! Run with --mode test to evaluate.")
        
    else:  # test mode
        print("Testing RL-enhanced planner...")
        planner = RLEnhancedAPF_RRT(model_path=args.model)
        
        # Test scenario
        q_start = np.array([0.8, 1.4, -0.7, -0.2, 0.7, 0.1])
        q_goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        obstacles = [
            (np.array([0.5, 0.7, -0.3, 0.0, 0.3, 0.0]), 0.5),
            (np.array([0.2, 0.5, -0.5, -0.1, 0.5, 0.1]), 0.4)
        ]
        
        path, nodes, runtime, metrics = planner.plan(q_start, q_goal, obstacles)
        
        if path:
            print(f"\n✓ Success!")
            print(f"  Nodes: {metrics['nodes']}")
            print(f"  Runtime: {runtime:.3f}s")
            print(f"  Final K_att: {metrics['K_att_final']:.3f}")
            print(f"  Final K_rep: {metrics['K_rep_final']:.3f}")
        else:
            print(f"\n✗ Failed to find path")
