#!/usr/bin/env python3
"""
PSO (Particle Swarm Optimization) Path Smoother
Post-processes paths to minimize length, curvature, and ensure smoothness
"""

import dataclasses
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

@dataclasses.dataclass
class PSOHyperparameters:
    """Container for PSO tunables to simplify Task 4.2 hyperparameter sweeps."""

    n_particles: int = 30
    max_iters: int = 50
    w: float = 0.65
    c1: float = 1.6
    c2: float = 1.8
    collision_penalty: float = 12.0
    smoothness_penalty: float = 0.65
    curvature_penalty: float = 0.5
    length_penalty: float = 1.0
    joint_limit_penalty: float = 5.0
    velocity_penalty: float = 2.0
    collision_margin: float = 0.1
    max_velocity: float = 2.0


class PSOPathSmoother:
    """
    Particle Swarm Optimization for path smoothing and optimization
    """
    
    def __init__(
        self,
        n_particles: int = 30,
        max_iters: int = 50,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        verbose: bool = True,
        collision_margin: float = 0.05,
        max_velocity: float = 2.0,
    ):
        """Create a PSO smoother with explicit safety knobs."""

        self.verbose = verbose
        self.n_particles = n_particles
        self.max_iters = max_iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.collision_margin = collision_margin
        self.max_velocity = max_velocity

        # Cost function weights
        self.alpha_length = 1.0       # Path length weight
        self.beta_smoothness = 0.5    # Smoothness weight
        self.gamma_collision = 10.0   # Collision penalty weight
        self.delta_joint_limit = 5.0  # Joint limit violation weight
        self.epsilon_velocity = 2.0   # Max velocity constraint weight
        self.zeta_curvature = 0.4     # Peak curvature penalty
    
    def smooth(
        self,
        path: Sequence[Sequence[float]],
        obstacles: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
        joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_endpoints: bool = True,
        verbose: bool = True,
        return_trace: bool = False,
    ):
        """
        Smooth path using PSO
        
        Args:
            path: Original path as array (n_waypoints, n_dims)
            obstacles: List of (center, radius) tuples
            joint_limits: (q_min, q_max) arrays for joint limits
            fixed_endpoints: Whether to keep start/end fixed
            verbose: Print progress
        
        Returns:
            smoothed_path: Optimized path
            best_cost: Final cost value
            metrics: Dictionary of metrics
        """
        if verbose:
            print("\n" + "="*60)
            print("PSO Path Smoothing")
            print("="*60)
        
        path = np.array(path)
        n_waypoints, n_dims = path.shape
        
        if fixed_endpoints:
            # Only optimize interior waypoints
            optimize_indices = range(1, n_waypoints - 1)
            n_optimize = len(optimize_indices)
        else:
            optimize_indices = range(n_waypoints)
            n_optimize = n_waypoints
        
        if n_optimize == 0:
            return path, 0, {}
        
        # Flatten path for optimization (only interior waypoints)
        def path_to_vector(p):
            return p[optimize_indices].flatten()
        
        def vector_to_path(v):
            new_path = path.copy()
            new_path[optimize_indices] = v.reshape(n_optimize, n_dims)
            return new_path
        
        # Initialize particles
        particles: List[np.ndarray] = []
        velocities: List[np.ndarray] = []
        personal_best_positions: List[np.ndarray] = []
        personal_best_costs: List[float] = []
        trace: List[float] = []
        
        initial_vector = path_to_vector(path)
        search_range = 0.25  # Search within ±25% of original positions
        
        for i in range(self.n_particles):
            # Random perturbation around initial path
            if i == 0:
                # First particle is the original path
                particle = initial_vector.copy()
            else:
                noise = np.random.uniform(-search_range, search_range, 
                                        size=initial_vector.shape)
                particle = initial_vector + noise
            
            particles.append(particle)
            velocities.append(np.zeros_like(particle))
            personal_best_positions.append(particle.copy())
            
            # Evaluate initial cost
            cost = self._evaluate_cost(vector_to_path(particle), 
                                      obstacles, joint_limits)
            personal_best_costs.append(cost)
        
        # Initialize global best
        global_best_idx = np.argmin(personal_best_costs)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_cost = personal_best_costs[global_best_idx]
        
        if verbose:
            print(f"Initial cost: {global_best_cost:.4f}")
            print(f"Optimizing {n_optimize} waypoints with {self.n_particles} particles...")
        
        start_time = time.time()
        
        # PSO iterations
        for iteration in range(self.max_iters):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.random(2)
                cognitive = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social = self.c2 * r2 * (global_best_position - particles[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # Update position
                particles[i] += velocities[i]
                
                # Evaluate cost
                current_path = vector_to_path(particles[i])
                cost = self._evaluate_cost(current_path, obstacles, joint_limits)
                
                # Update personal best
                if cost < personal_best_costs[i]:
                    personal_best_costs[i] = cost
                    personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best
                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best_position = particles[i].copy()

            trace.append(float(global_best_cost))
            # Print progress
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{self.max_iters}: "
                      f"Best cost = {global_best_cost:.4f}")
        
        runtime = time.time() - start_time
        
        # Construct final smoothed path
        smoothed_path = vector_to_path(global_best_position)
        
        # Compute metrics
        original_cost = self._evaluate_cost(path, obstacles, joint_limits)
        improvement = ((original_cost - global_best_cost) / (original_cost + 1e-9)) * 100
        
        metrics = {
            'original_cost': original_cost,
            'final_cost': global_best_cost,
            'improvement_percent': improvement,
            'runtime': runtime,
            'original_length': self._compute_length(path),
            'final_length': self._compute_length(smoothed_path),
            'original_smoothness': self._compute_smoothness(path),
            'final_smoothness': self._compute_smoothness(smoothed_path),
            'original_max_curvature': self._compute_max_curvature(path),
            'final_max_curvature': self._compute_max_curvature(smoothed_path),
            'original_clearance': self._compute_min_clearance(path, obstacles) if obstacles else np.inf,
            'final_clearance': self._compute_min_clearance(smoothed_path, obstacles) if obstacles else np.inf,
            'cost_trace': trace,
        }
        
        if verbose:
            print(f"\n✓ Smoothing complete in {runtime:.2f}s")
            print(f"  Cost: {original_cost:.4f} → {global_best_cost:.4f} ({improvement:.1f}% improvement)")
            print(f"  Length: {metrics['original_length']:.2f} → {metrics['final_length']:.2f}")
            print(f"  Smoothness: {metrics['original_smoothness']:.4f} → {metrics['final_smoothness']:.4f}")
        
        if return_trace:
            metrics['cost_trace'] = trace

        return smoothed_path, global_best_cost, metrics
    
    def _evaluate_cost(self, path, obstacles, joint_limits):
        """
        Evaluate path cost (lower is better)
        
        Cost = α*length + β*smoothness_penalty + γ*collision_penalty +
               δ*joint_limit_penalty + ε*velocity_penalty + ζ*max_curvature
        """
        cost = 0
        
        # 1. Path length
        length = self._compute_length(path)
        cost += self.alpha_length * length
        
        # 2. Smoothness (curvature penalty)
        smoothness_penalty = self._compute_smoothness(path)
        cost += self.beta_smoothness * smoothness_penalty

        # 2b. Peak curvature term keeps endpoints gentle
        max_curvature = self._compute_max_curvature(path)
        cost += self.zeta_curvature * max_curvature
        
        # 3. Collision penalty
        if obstacles is not None:
            collision_penalty = self._compute_collision_penalty(path, obstacles)
            cost += self.gamma_collision * collision_penalty
        
        # 4. Joint limit violations
        if joint_limits is not None:
            joint_penalty = self._compute_joint_limit_penalty(path, joint_limits)
            cost += self.delta_joint_limit * joint_penalty
        
        # 5. Maximum velocity constraint
        velocity_penalty = self._compute_velocity_penalty(path)
        cost += self.epsilon_velocity * velocity_penalty

        # 6. Encourage clearance buffer
        if obstacles is not None:
            clearance = self._compute_min_clearance(path, obstacles)
            if clearance < self.collision_margin:
                cost += self.gamma_collision * (self.collision_margin - clearance) ** 2

        return cost
    
    def _compute_length(self, path):
        """Compute total path length"""
        if len(path) < 2:
            return 0
        
        dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.sum(dists)
    
    def _compute_smoothness(self, path):
        """
        Compute smoothness penalty based on curvature
        Lower is smoother
        """
        if len(path) < 3:
            return 0
        
        # Compute angles between consecutive segments
        segments = np.diff(path, axis=0)
        
        # Normalize segments
        norms = np.linalg.norm(segments, axis=1, keepdims=True)
        norms = np.where(norms > 1e-6, norms, 1.0)  # Avoid division by zero
        segments_normalized = segments / norms
        
        # Compute angles (using dot product)
        angles = []
        for i in range(len(segments_normalized) - 1):
            cos_angle = np.dot(segments_normalized[i], segments_normalized[i+1])
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        # Smoothness penalty is sum of squared angles
        return np.sum(np.array(angles) ** 2)

    def _compute_max_curvature(self, path: np.ndarray) -> float:
        """Return the maximum turning angle observed along the path."""

        if len(path) < 3:
            return 0.0

        segments = np.diff(path, axis=0)
        norms = np.linalg.norm(segments, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        segments_normalized = segments / norms

        max_angle = 0.0
        for i in range(len(segments_normalized) - 1):
            cos_angle = np.dot(segments_normalized[i], segments_normalized[i+1])
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            max_angle = max(max_angle, float(angle))

        return max_angle
    
    def _compute_collision_penalty(self, path, obstacles):
        """
        Compute collision penalty
        Penalizes waypoints that are too close to obstacles
        """
        penalty = 0
        safety_margin = 1.0 + self.collision_margin  # Safety factor
        
        for waypoint in path:
            for obs_center, obs_radius in obstacles:
                distance = np.linalg.norm(waypoint - obs_center) - obs_radius
                
                if distance < 0:
                    # Inside obstacle - heavy penalty
                    penalty += 100 * abs(distance)
                elif distance < obs_radius * safety_margin:
                    # Too close - moderate penalty
                    penalty += (obs_radius * safety_margin - distance) ** 2
        
        # Also sample along segments to catch near-misses between waypoints
        if len(path) >= 2:
            dense_points = self._sample_along_path(path, samples_per_segment=4)
            for waypoint in dense_points:
                for obs_center, obs_radius in obstacles:
                    distance = np.linalg.norm(waypoint - obs_center) - obs_radius
                    if distance < 0:
                        penalty += 50 * abs(distance)
                    elif distance < obs_radius * safety_margin:
                        penalty += (obs_radius * safety_margin - distance) ** 2

        return penalty

    def _compute_min_clearance(self, path: np.ndarray, obstacles: Sequence[Tuple[np.ndarray, float]]) -> float:
        """Return the minimum clearance to any obstacle along the path."""

        if obstacles is None or len(obstacles) == 0:
            return float("inf")

        dense_points = self._sample_along_path(path, samples_per_segment=4)
        min_clearance = float("inf")
        for waypoint in dense_points:
            for obs_center, obs_radius in obstacles:
                distance = np.linalg.norm(waypoint - obs_center) - obs_radius
                min_clearance = min(min_clearance, float(distance))
        return min_clearance

    def _sample_along_path(self, path: np.ndarray, samples_per_segment: int = 3) -> np.ndarray:
        """Return interpolated points along path segments for collision checks."""

        if len(path) < 2:
            return path

        samples: List[np.ndarray] = []
        for p_start, p_end in zip(path[:-1], path[1:]):
            for ratio in np.linspace(0.0, 1.0, samples_per_segment, endpoint=True):
                samples.append(p_start * (1 - ratio) + p_end * ratio)
        return np.asarray(samples)
    
    def _compute_joint_limit_penalty(self, path, joint_limits):
        """
        Penalize configurations outside joint limits
        """
        q_min, q_max = joint_limits
        penalty = 0
        
        for q in path:
            # Lower limit violations
            violations_low = np.maximum(0, q_min - q)
            penalty += np.sum(violations_low ** 2)
            
            # Upper limit violations
            violations_high = np.maximum(0, q - q_max)
            penalty += np.sum(violations_high ** 2)
        
        return penalty
    
    def _compute_velocity_penalty(self, path, max_velocity=None):
        """
        Penalize excessive velocities between waypoints
        """
        if len(path) < 2:
            return 0
        
        velocities = np.diff(path, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)

        # Penalize velocities exceeding maximum
        max_velocity = max_velocity or self.max_velocity
        excess_velocities = np.maximum(0, velocity_magnitudes - max_velocity)
        return np.sum(excess_velocities ** 2)
    
    def set_weights(self, alpha=None, beta=None, gamma=None, delta=None, epsilon=None):
        """Update cost function weights"""
        if alpha is not None:
            self.alpha_length = alpha
        if beta is not None:
            self.beta_smoothness = beta
        if gamma is not None:
            self.gamma_collision = gamma
        if delta is not None:
            self.delta_joint_limit = delta
        if epsilon is not None:
            self.epsilon_velocity = epsilon

    def apply_hyperparameters(self, params: PSOHyperparameters) -> None:
        """Copy values from a :class:`PSOHyperparameters` bundle."""

        self.n_particles = params.n_particles
        self.max_iters = params.max_iters
        self.w = params.w
        self.c1 = params.c1
        self.c2 = params.c2
        self.alpha_length = params.length_penalty
        self.beta_smoothness = params.smoothness_penalty
        self.gamma_collision = params.collision_penalty
        self.delta_joint_limit = params.joint_limit_penalty
        self.epsilon_velocity = params.velocity_penalty
        self.zeta_curvature = params.curvature_penalty
        self.collision_margin = params.collision_margin
        self.max_velocity = params.max_velocity

    def tune_from_environment(
        self,
        path: np.ndarray,
        obstacles: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
    ) -> PSOHyperparameters:
        """Return heuristically tuned hyperparameters based on path roughness."""

        roughness = self._compute_smoothness(path)
        curvature = self._compute_max_curvature(path)
        clutter_score = 0.0
        if obstacles:
            for _, radius in obstacles:
                clutter_score += radius
        clutter_score = clutter_score / max(1, len(obstacles or [])) if obstacles else 0.0

        params = PSOHyperparameters()
        params.n_particles = int(max(14, min(42, 22 + roughness * 4 + curvature * 8)))
        params.max_iters = int(max(30, min(90, 35 + curvature * 20)))
        params.collision_penalty = 10.0 + clutter_score * 0.1
        params.smoothness_penalty = 0.5 + 0.15 * curvature
        params.curvature_penalty = 0.4 + 0.2 * curvature
        params.collision_margin = max(self.collision_margin, 0.05 + 0.002 * clutter_score)

        return params


def visualize_smoothing(original_path, smoothed_path, obstacles=None, title="PSO smoothing", overlay=True, save_path='pso_smoothing_result.png', show=False):
    """Visualize original vs smoothed path with optional overlay for RViz parity."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    original_path = np.array(original_path)
    smoothed_path = np.array(smoothed_path)

    if overlay:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        axes = [ax]
    else:
        fig = plt.figure(figsize=(15, 6))
        axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]

    def draw_path(ax, path, color, label):
        ax.plot(path[:, 0], path[:, 1], path[:, 2], f"{color}-o", linewidth=2, markersize=5, label=label)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], c='green', s=80, marker='o', label='Start')
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c='purple', s=80, marker='*', label='Goal')

    def draw_obstacles(ax):
        if obstacles:
            for obs_center, obs_radius in obstacles:
                u = np.linspace(0, 2*np.pi, 20)
                v = np.linspace(0, np.pi, 10)
                x = obs_center[0] + obs_radius * np.outer(np.cos(u), np.sin(v))
                y = obs_center[1] + obs_radius * np.outer(np.sin(u), np.sin(v))
                z = obs_center[2] + obs_radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='gray', alpha=0.25)

    # Draw plots
    if overlay:
        ax = axes[0]
        draw_obstacles(ax)
        draw_path(ax, original_path, 'r', 'Original Path')
        draw_path(ax, smoothed_path, 'b', 'Smoothed Path')
        ax.set_title(title)
    else:
        draw_obstacles(axes[0]); draw_path(axes[0], original_path, 'r', 'Original Path'); axes[0].set_title('Original Path')
        draw_obstacles(axes[1]); draw_path(axes[1], smoothed_path, 'b', 'Smoothed Path'); axes[1].set_title('PSO-Smoothed Path')

    for ax in axes:
        ax.set_xlabel('Joint 1'); ax.set_ylabel('Joint 2'); ax.set_zlabel('Joint 3')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")
    if show:
        plt.show()


if __name__ == "__main__":
    print("Testing PSO Path Smoother")
    
    # Generate test path with some roughness
    np.random.seed(42)
    n_waypoints = 10
    n_dims = 6  # 6-DOF
    
    # Create a rough path
    t = np.linspace(0, 1, n_waypoints)
    smooth_base = np.column_stack([
        np.sin(2 * np.pi * t),
        np.cos(2 * np.pi * t),
        t,
        np.sin(4 * np.pi * t),
        np.cos(4 * np.pi * t),
        1 - t
    ])
    
    # Add noise to make it rough
    noise = np.random.normal(0, 0.1, smooth_base.shape)
    rough_path = smooth_base + noise
    
    # Define obstacles
    obstacles = [
        (np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.5]), 0.3),
        (np.array([-0.3, 0.3, 0.3, 0.2, -0.2, 0.3]), 0.25)
    ]
    
    # Define joint limits
    joint_limits = (
        np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]),
        np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
    )
    
    # Create smoother
    smoother = PSOPathSmoother(n_particles=30, max_iters=50)
    
    # Smooth path
    smoothed_path, cost, metrics = smoother.smooth(
        rough_path, 
        obstacles=obstacles,
        joint_limits=joint_limits,
        fixed_endpoints=True,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Metrics:")
    print("="*60)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Visualize (using first 3 dimensions)
    visualize_smoothing(
        rough_path[:, :3],
        smoothed_path[:, :3],
        obstacles=[(obs_center[:3], radius) for obs_center, radius in obstacles]
    )
    
    print("\n✓ Test complete!")
