#!/usr/bin/env python3
"""RL-enhanced APF-RRT planner.

This module provides everything that is required to train, evaluate and
visualise an artificial potential field (APF) guided RRT motion planner in a
stand-alone environment.  The code is structured so it can be dropped into a
Google Colab notebook without modifications – all heavy imports are guarded and
GPU support is automatically detected.

Key features
------------
* Clean separation between environment dynamics, RL training utilities and
  evaluation helpers.
* Efficient vectorised state computations to keep the Gym environment light.
* Optional multi-processing vectorised environments for faster PPO training on
  Colab (or locally).
* Matplotlib based 3D visualisation of the explored tree and final path.
* Command line interface supporting ``train`` and ``test`` modes.

The environment models a 6-DoF configuration space.  The first three joints are
used when rendering 3D plots which is sufficient to understand exploration
behaviour while keeping the visualisation legible.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

try:  # pragma: no cover - optional dependency guard
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover - TensorFlow is optional
    tf = None  # type: ignore

import numpy as np
import torch
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from path_exporter import export_path

# Matplotlib is an optional dependency; importing lazily keeps the module usable
# without it (e.g. on headless Colab runtimes before ``pip install matplotlib``).
try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
except Exception:  # pragma: no cover - optional dependency
    plt = None


# ---------------------------------------------------------------------------
# Success criterion
# ---------------------------------------------------------------------------

DEFAULT_GOAL_TOLERANCE = 0.2


def is_success(ee_pos: np.ndarray, goal_pos: np.ndarray, threshold: float = DEFAULT_GOAL_TOLERANCE) -> bool:
    """Return True if end-effector reached goal."""

    return np.linalg.norm(ee_pos - goal_pos) < threshold


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlannerParameters:
    """Tunables for the APF-RRT planner.

    The RL agent learns increments on top of these parameters.  Bounds are used
    both when sampling actions inside the environment and when applying the
    trained agent during deployment.
    """

    attractive_gain: float = 1.0
    repulsive_gain: float = 0.3
    influence_distance: float = 1.5
    step_size: float = 0.3
    goal_bias: float = 0.07

    attractive_range: Tuple[float, float] = (0.1, 6.0)
    repulsive_range: Tuple[float, float] = (0.05, 3.0)
    influence_range: Tuple[float, float] = (0.5, 3.5)
    step_range: Tuple[float, float] = (0.1, 1.2)
    goal_bias_range: Tuple[float, float] = (0.0, 0.4)

    def apply_delta(self, delta: Sequence[float]) -> None:
        """Apply an action delta and clamp to configured ranges."""

        (da, dr, dd, ds, db) = delta
        self.attractive_gain = np.clip(
            self.attractive_gain + da, *self.attractive_range
        )
        self.repulsive_gain = np.clip(
            self.repulsive_gain + dr, *self.repulsive_range
        )
        self.influence_distance = np.clip(
            self.influence_distance + dd, *self.influence_range
        )
        self.step_size = np.clip(self.step_size + ds, *self.step_range)
        self.goal_bias = np.clip(self.goal_bias + db, *self.goal_bias_range)

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.attractive_gain,
                self.repulsive_gain,
                self.influence_distance,
                self.step_size,
                self.goal_bias,
            ],
            dtype=np.float32,
        )


@dataclass
class ScenarioConfig:
    """Random scenario generator options."""

    difficulty: str = "medium"  # easy / medium / hard
    max_steps: int = 128
    joint_min: float = -math.pi
    joint_max: float = math.pi
    n_joints: int = 6
    # Shared across training and evaluation; matches ROS launch default goal_radius.
    goal_tolerance: float = DEFAULT_GOAL_TOLERANCE
    dynamic_probability: float = 0.45
    obstacle_speed_range: Tuple[float, float] = (0.05, 0.35)
    dynamic_time_step: float = 0.08

    @property
    def obstacle_count(self) -> int:
        return {"easy": 2, "medium": 4, "hard": 6}[self.difficulty]

    def sample_configuration(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.joint_min, self.joint_max, self.n_joints)


@dataclass
class ObstacleState:
    """State of an obstacle in joint space."""

    centre: np.ndarray
    radius: float
    velocity: np.ndarray

    def copy(self) -> "ObstacleState":
        return ObstacleState(self.centre.copy(), float(self.radius), self.velocity.copy())

    def advance(self, joint_min: float, joint_max: float, dt: float) -> None:
        """Integrate obstacle motion with reflective boundaries."""

        self.centre += self.velocity * dt
        for idx, value in enumerate(self.centre):
            if value < joint_min:
                overflow = joint_min - value
                self.centre[idx] = joint_min + overflow
                self.velocity[idx] *= -1
            elif value > joint_max:
                overflow = value - joint_max
                self.centre[idx] = joint_max - overflow
                self.velocity[idx] *= -1


@dataclass
class PlanResult:
    success: bool
    collision: bool
    path: Optional[List[np.ndarray]]
    planning_time: float
    num_nodes: int
    path_length: float
    parents: Optional[Dict[int, Optional[int]]] = None


# ---------------------------------------------------------------------------
# Incremental RRT planner
# ---------------------------------------------------------------------------


class RRTPlanner:
    """Lightweight RRT planner supporting incremental expansion."""

    def __init__(self, env: "APFRRTEnv") -> None:
        self.env = env
        self.tree: List[np.ndarray] = []
        self.parents: Dict[int, Optional[int]] = {}
        self.start: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None

    def reset(self, start: np.ndarray, goal: np.ndarray) -> None:
        self.start = start.copy()
        self.goal = goal.copy()
        self.tree = [self.start]
        self.parents = {0: None}

    def sample(self) -> np.ndarray:
        return self.env._sample_random_configuration()

    def nearest(self, q_rand: np.ndarray) -> Tuple[int, np.ndarray]:
        dists = [np.linalg.norm(node - q_rand) for node in self.tree]
        idx = int(np.argmin(dists))
        return idx, self.tree[idx]

    def steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        direction = self.env._compute_direction(q_near, q_rand)
        return np.clip(
            q_near + direction * self.env.parameters.step_size,
            self.env.scenario.joint_min,
            self.env.scenario.joint_max,
        )

    def in_collision(self, q_new: np.ndarray) -> bool:
        return self.env._in_collision(q_new)

    def incremental_step(self) -> Tuple[np.ndarray, bool, bool, Optional[List[np.ndarray]]]:
        # sample one point
        q_rand = self.sample()
        # find nearest
        idx_near, q_near = self.nearest(q_rand)
        # steer
        direction = self.env._compute_direction(q_near, q_rand)
        step = min(self.env.parameters.step_size, np.linalg.norm(q_rand - q_near))
        q_new = np.clip(
            q_near + direction * step,
            self.env.scenario.joint_min,
            self.env.scenario.joint_max,
        )
        # check collision
        if self.in_collision(q_new):
            return q_near, True, False, None
        # add node
        self.tree.append(q_new)
        self.parents[len(self.tree) - 1] = idx_near
        path: Optional[List[np.ndarray]] = None
        reached = self.env.goal_reached(q_new)
        if reached:
            path = self.env._reconstruct_path(self.parents, len(self.tree) - 1, self.tree, self.goal)
        return q_new, False, reached, path


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

Obstacle = ObstacleState

if TYPE_CHECKING:  # pragma: no cover - optional dependency
    from obstacle_predictor import DynamicObstacleManager


class APFRRTEnv(Env):
    """Gymnasium environment exposing APF-RRT planning dynamics."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario: ScenarioConfig,
        parameters: Optional[PlannerParameters] = None,
        seed: Optional[int] = None,
        debug: bool = False,
        dynamic_manager: Optional["DynamicObstacleManager"] = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.parameters = parameters or PlannerParameters()
        self.rng = np.random.default_rng(seed)
        self.debug = debug
        self.planner = RRTPlanner(self)
        self.prev_dist_to_goal: float = 0.0
        self.dynamic_manager = dynamic_manager

        # Observation encodes planner state + tunables so PPO can correlate them
        # with progress: distance, heading, obstacle info, parameter vector.
        low = np.array(
            [0.0, -1.0, 0.0, 0.0] + [p[0] for p in self._parameter_bounds()],
            dtype=np.float32,
        )
        high = np.array(
            [1.0, 1.0, 1.0, 1.0] + [p[1] for p in self._parameter_bounds()],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action is applied as small parameter deltas.
        self.action_space = spaces.Box(
            low=np.array([-0.4, -0.4, -0.5, -0.25, -0.15], dtype=np.float32),
            high=np.array([0.4, 0.4, 0.5, 0.25, 0.15], dtype=np.float32),
        )

        self.q_start: np.ndarray = np.zeros(self.scenario.n_joints)
        self.q_goal: np.ndarray = np.zeros(self.scenario.n_joints)
        self.q_current: np.ndarray = np.zeros(self.scenario.n_joints)
        self.current_node: np.ndarray = np.zeros(self.scenario.n_joints)
        self.goal_position: np.ndarray = np.zeros(3, dtype=np.float64)
        self.obstacles: List[Obstacle] = []
        self._obstacle_ids: List[str] = []
        self._dynamic_active = False
        self.nodes: List[np.ndarray] = []
        self._step_index = 0
        self._last_path: Optional[List[np.ndarray]] = None
        self._last_parents: Optional[Dict[int, Optional[int]]] = None
        self._last_motion_dir = np.zeros(self.scenario.n_joints, dtype=np.float64)
        self._stuck_steps = 0

        self.reset()

    # -- Env API -----------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.parameters = PlannerParameters()
        self._dynamic_active = self.rng.random() < self.scenario.dynamic_probability
        self.q_start = self.scenario.sample_configuration(self.rng)
        self.q_goal = self.scenario.sample_configuration(self.rng)
        self.q_current = self.q_start.copy()
        self.current_node = self.q_start.copy()
        self.goal_position = self.workspace_position(self.q_goal)
        self.nodes = [self.q_start.copy()]
        self.obstacles = self._generate_obstacles(self.scenario.obstacle_count)
        self._obstacle_ids = [f"obs_{idx}" for idx in range(len(self.obstacles))]
        if self.dynamic_manager:
            self.dynamic_manager.reset()
            self._record_obstacles(timestamp=0.0)
        self._step_index = 0
        self.planner.reset(self.q_start, self.q_goal)
        self.nodes = self.planner.tree
        self._last_parents = None
        self.prev_dist_to_goal = self._distance_to_goal(self.q_start)
        self._last_motion_dir = np.zeros(self.scenario.n_joints, dtype=np.float64)
        self._stuck_steps = 0
        return self._get_state(), {}

    def step(self, action: np.ndarray):
        step_start = time.perf_counter()
        self._step_index += 1

        q_new, progress, collided, reached, dist, movement, dir_change = self._incremental_planner_step(action)
        reward = self._compute_reward(dist, progress, collided, reached, movement, dir_change)
        done = collided or reached
        truncated = False
        obs = self._get_observation(q_new)
        info: Dict[str, float] = {
            "progress": progress,
            "collision": collided,
            "reached_goal": reached,
            "step_ms": float((time.perf_counter() - step_start) * 1_000.0),
            "clearance": float(self._minimum_clearance(self.q_current)),
            "distance_to_goal": float(dist),
            "is_success": reached,
        }
        if self.debug:
            print(
                f"[DEBUG] Params: {self.parameters.to_array()} | Reached: {reached} "
                f"| Collision: {collided} | Progress: {progress:.4f} | Nodes: {len(self.nodes)}"
            )

        return obs, float(reward), bool(done), bool(truncated), info

    def render(self):  # pragma: no cover - simple debug rendering
        print(
            f"Step {self._step_index} – dist_to_goal: {self._distance_to_goal(self.q_current):.3f} "
            f"nodes: {len(self.nodes)}"
        )

    # -- Planning utilities ------------------------------------------------
    def _apply_action_parameters(self, action: np.ndarray) -> None:
        self.parameters.apply_delta(action)

    def _check_collision(self, q_new: np.ndarray) -> bool:
        return self._in_collision(q_new)

    def _get_observation(self, q_new: np.ndarray) -> np.ndarray:
        self.q_current = q_new.copy()
        self.current_node = q_new.copy()
        return self._get_state()

    def _compute_reward(
        self,
        dist: float,
        progress: float,
        collided: bool,
        reached_goal: bool,
        movement: float,
        direction_change: float,
    ) -> float:
        norm_improvement = progress / max(self.prev_dist_to_goal, 1e-6)
        progress_reward = 5.0 * norm_improvement
        movement_reward = 0.5 * movement
        oscillation_penalty = -0.1 * direction_change
        stuck_penalty = -0.5 if self._stuck_steps >= 3 else 0.0
        collision_penalty = -80.0 if collided else 0.0
        goal_bonus = 200.0 if reached_goal else 0.0

        reward = (
            progress_reward
            + movement_reward
            + goal_bonus
            + collision_penalty
            + oscillation_penalty
            + stuck_penalty
            - 0.01
        )

        self.prev_dist_to_goal = dist
        return float(reward)

    def _incremental_planner_step(self, action: np.ndarray):
        previous_q = self.q_current.copy()
        # apply APF + RRT parameters from the action
        self._apply_action_parameters(action)
        # perform ONE incremental RRT node expansion
        q_new, collided, reached, path = self.planner.incremental_step()
        # compute movement metrics
        movement_vec = q_new - previous_q
        movement = float(np.linalg.norm(movement_vec))
        motion_dir = movement_vec / (movement + 1e-9)
        direction_change = float(np.linalg.norm(motion_dir - self._last_motion_dir))
        self._last_motion_dir = motion_dir
        if movement < self.parameters.step_size * 0.1:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        # update internal state
        self.current_node = q_new
        self.q_current = q_new.copy()
        # compute distances and progress
        dist = self._distance_to_goal(q_new)
        progress = self.prev_dist_to_goal - dist
        # compute collisions
        collided = collided or self._check_collision(q_new)
        # check success
        if reached and path is not None:
            self._last_path = path
        else:
            self._advance_obstacles()
        return q_new, progress, collided, reached, dist, movement, direction_change

    def _parameter_bounds(self) -> List[Tuple[float, float]]:
        params = self.parameters
        return [
            params.attractive_range,
            params.repulsive_range,
            params.influence_range,
            params.step_range,
            params.goal_bias_range,
        ]

    def _generate_obstacles(self, n: int) -> List[Obstacle]:
        obstacles: List[Obstacle] = []
        for _ in range(n):
            centre = self.scenario.sample_configuration(self.rng)
            radius = float(self.rng.uniform(0.3, 0.8))
            if self._dynamic_active and self.scenario.obstacle_speed_range[1] > 0:
                direction = self.rng.normal(size=self.scenario.n_joints)
                direction /= np.linalg.norm(direction) + 1e-9
                speed = self.rng.uniform(*self.scenario.obstacle_speed_range)
                velocity = direction * speed
            else:
                velocity = np.zeros(self.scenario.n_joints)
            obstacles.append(ObstacleState(centre, radius, velocity))
        return obstacles

    def _record_obstacles(self, timestamp: float) -> None:
        """Update the dynamic manager with current obstacle state."""

        if self.dynamic_manager is None:
            return

        for obs_id, obstacle in zip(self._obstacle_ids, self.obstacles):
            self.dynamic_manager.update_obstacle(obs_id, obstacle.centre, obstacle.velocity, timestamp)

    def _effective_obstacles(self, step: int = 1) -> List[Tuple[np.ndarray, float]]:
        """Return obstacle centres optionally replaced by predicted futures."""

        if self.dynamic_manager is None:
            return [(obstacle.centre, obstacle.radius) for obstacle in self.obstacles]

        predicted_positions = self.dynamic_manager.get_all_predicted_positions(step)
        effective: List[Tuple[np.ndarray, float]] = []
        for obs_id, obstacle in zip(self._obstacle_ids, self.obstacles):
            centre = predicted_positions.get(obs_id, obstacle.centre)
            effective.append((centre, obstacle.radius))
        return effective

    def _sample_random_configuration(self) -> np.ndarray:
        if self.rng.random() < self.parameters.goal_bias:
            return self.q_goal
        return self.scenario.sample_configuration(self.rng)

    def _find_nearest_node(self, target: np.ndarray) -> Tuple[int, np.ndarray]:
        dists = [np.linalg.norm(node - target) for node in self.nodes]
        index = int(np.argmin(dists))
        return index, self.nodes[index]

    def _compute_direction(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        total_force = self._apf_force(q_near)
        towards_rand = q_rand - q_near
        towards_goal = self.q_goal - q_near

        apf_component = total_force / (np.linalg.norm(total_force) + 1e-9)
        apf_component *= 0.5  # damp APF so RRT steering remains dominant

        components = np.stack(
            [
                towards_rand / (np.linalg.norm(towards_rand) + 1e-9),
                towards_goal / (np.linalg.norm(towards_goal) + 1e-9),
                apf_component,
            ]
        )
        weights = np.array([0.55, 0.35, 0.10], dtype=np.float32)
        direction = (weights[:, None] * components).sum(axis=0)
        return direction / (np.linalg.norm(direction) + 1e-9)

    def _apf_force(self, q: np.ndarray) -> np.ndarray:
        params = self.parameters
        v_att = self.q_goal - q
        d_att = np.linalg.norm(v_att)
        f_att = params.attractive_gain * (v_att / (d_att + 1e-9)) if d_att > 0 else np.zeros_like(q)

        f_rep = np.zeros_like(q)
        for centre, radius in self._effective_obstacles(step=1):
            diff = q - centre
            dist = np.linalg.norm(diff) - radius
            if 0.0 < dist <= params.influence_distance:
                magnitude = params.repulsive_gain * ((1.0 / dist**2) * (1.0 / dist - 1.0 / params.influence_distance))
                f_rep += magnitude * (diff / (np.linalg.norm(diff) + 1e-9))
        return f_att + f_rep

    def _minimum_clearance(self, q: np.ndarray) -> float:
        distances = [np.linalg.norm(q - centre) - radius for centre, radius in self._effective_obstacles()]
        return min(distances) if distances else 10.0

    def _in_collision(self, q: np.ndarray) -> bool:
        return any(np.linalg.norm(q - centre) < radius for centre, radius in self._effective_obstacles())

    def _advance_obstacles(self) -> None:
        if not self._dynamic_active:
            return
        for obstacle in self.obstacles:
            obstacle.advance(
                self.scenario.joint_min,
                self.scenario.joint_max,
                self.scenario.dynamic_time_step,
            )
        self._record_obstacles(timestamp=float(self._step_index * self.scenario.dynamic_time_step))

    def _reconstruct_path(
        self,
        parents: Dict[int, Optional[int]],
        goal_index: int,
        nodes: Sequence[np.ndarray],
        q_goal: np.ndarray,
    ) -> List[np.ndarray]:
        path = [q_goal.copy()]
        current = goal_index
        while current is not None:
            path.append(nodes[current])
            current = parents[current]
        path.reverse()
        return path

    def workspace_position(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=np.float64)[:3]

    def _distance_to_goal(self, q: np.ndarray) -> float:
        return float(np.linalg.norm(self.workspace_position(q) - self.goal_position))

    def goal_reached(self, q: np.ndarray) -> bool:
        """Unified success check used by training and benchmarks."""

        return self._distance_to_goal(q) <= self.scenario.goal_tolerance

    # NOTE: only used for evaluation, never for RL training
    def _run_planning_episode(self) -> PlanResult:
        self.planner.reset(self.q_start, self.q_goal)
        self.nodes = self.planner.tree
        parents: Dict[int, Optional[int]] = {0: None}
        self.q_current = self.q_start.copy()
        self._last_path = None
        self._last_parents = parents
        self.goal_position = self.workspace_position(self.q_goal)

        collision = False
        success = False
        path_length = 0.0
        path: Optional[List[np.ndarray]] = None
        start_time = time.perf_counter()

        for _ in range(self.scenario.max_steps):
            q_rand = self._sample_random_configuration()
            idx_near, q_near = self._find_nearest_node(q_rand)
            direction = self._compute_direction(q_near, q_rand)
            step = min(self.parameters.step_size, np.linalg.norm(q_rand - q_near))
            q_new = np.clip(
                q_near + direction * step,
                self.scenario.joint_min,
                self.scenario.joint_max,
            )

            if self._in_collision(q_new):
                collision = True
                break

            self.nodes.append(q_new)
            parents[len(self.nodes) - 1] = idx_near
            path_length += float(
                np.linalg.norm(
                    self.workspace_position(q_new) - self.workspace_position(q_near)
                )
            )
            self.q_current = q_new

            if self.goal_reached(q_new):
                success = True
                path = self._reconstruct_path(parents, len(self.nodes) - 1, self.nodes, self.q_goal)
                break

            self._advance_obstacles()

        planning_time = float(time.perf_counter() - start_time)
        if path is not None:
            path_length = sum(
                float(
                    np.linalg.norm(
                        self.workspace_position(path[i + 1])
                        - self.workspace_position(path[i])
                    )
                )
                for i in range(len(path) - 1)
            )
        self._last_path = path
        return PlanResult(
            success=success,
            collision=collision,
            path=path,
            planning_time=planning_time,
            num_nodes=len(self.nodes),
            path_length=path_length,
            parents=parents,
        )

    def _get_state(self) -> np.ndarray:
        dist = self._distance_to_goal(self.q_current)
        heading_vector = self.workspace_position(self.q_goal) - self.workspace_position(self.q_current)
        heading = math.atan2(heading_vector[1], heading_vector[0]) / math.pi
        min_clearance = self._minimum_clearance(self.q_current)
        local_density = sum(
            1
            for obstacle in self.obstacles
            if np.linalg.norm(self.q_current - obstacle.centre) < self.parameters.influence_distance
        )
        density = local_density / max(len(self.obstacles), 1)
        normalised = np.array(
            [
                np.clip(dist / 10.0, 0.0, 1.0),
                np.clip(heading, -1.0, 1.0),
                np.clip(min_clearance / 5.0, 0.0, 1.0),
                np.clip(density, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([normalised, self.parameters.to_array()])


# ---------------------------------------------------------------------------
# Callbacks & helpers
# ---------------------------------------------------------------------------


class ObservationNormalizer:
    """Utility for applying observation normalisation offline."""

    def __init__(
        self,
        mean: np.ndarray,
        var: np.ndarray,
        clip: float,
        epsilon: float = 1e-8,
    ) -> None:
        self.mean = mean.astype(np.float32)
        self.var = var.astype(np.float32)
        self.clip = float(clip)
        self.epsilon = float(epsilon)

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        normalised = (observation - self.mean) / np.sqrt(self.var + self.epsilon)
        if self.clip > 0:
            normalised = np.clip(normalised, -self.clip, self.clip)
        return normalised.astype(np.float32)

    @classmethod
    def from_file(cls, path: Path) -> "ObservationNormalizer":
        with np.load(path) as data:
            epsilon = float(data["epsilon"]) if "epsilon" in data else 1e-8
            return cls(
                mean=data["mean"],
                var=data["var"],
                clip=float(data["clip"]),
                epsilon=epsilon,
            )


class RewardCheckpoint(BaseCallback):
    """Simple callback that stores the best model by mean reward."""

    def __init__(self, check_freq: int, save_path: Path, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        if not self.model.ep_info_buffer:
            return True

        rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
        mean_reward = float(np.mean(rewards))

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose:
                print(f"New best mean reward: {mean_reward:.2f}")
            self.model.save(self.save_path / "best_model")
        return True


class SuccessRateCallback(BaseCallback):
    """Compute success rate using the unified is_success flag."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_successes: List[bool] = []

    def _on_step(self) -> bool:
        infos: Sequence[Dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            if "is_success" in info:
                self.episode_successes.append(bool(info.get("is_success", False)))
        return True

    def _on_rollout_end(self) -> bool:
        if self.episode_successes:
            success_rate = float(np.mean(self.episode_successes))
            self.logger.record("rollout/success_rate", success_rate)
            if self.verbose:
                print(f"Rollout success rate: {success_rate:.3f}")
        self.episode_successes.clear()
        return True


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkMetrics:
    """Aggregate statistics over evaluation episodes."""

    success_rate: float
    collision_rate: float
    avg_planning_time: float
    avg_path_length: float
    avg_nodes: float
    total_episodes: int
    dynamic: bool

    def as_dict(self) -> Dict[str, float]:
        return {
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "avg_planning_time": self.avg_planning_time,
            "avg_path_length": self.avg_path_length,
            "avg_nodes": self.avg_nodes,
            "total_episodes": float(self.total_episodes),
            "dynamic": float(self.dynamic),
        }


# ---------------------------------------------------------------------------
# Training / evaluation entry points
# ---------------------------------------------------------------------------


def make_vec_env(
    scenario: ScenarioConfig,
    n_envs: int,
    seed: int,
    use_subprocess: bool = True,
    normalize: bool = False,
    vecnormalize_kwargs: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Union[DummyVecEnv, VecNormalize]:
    """Create a vectorised environment for PPO training."""

    def _factory(rank: int):
        def _init():
            env = APFRRTEnv(scenario, seed=seed + rank, debug=debug)
            return Monitor(env)

        return _init

    env_fns = [_factory(i) for i in range(n_envs)]
    if n_envs == 1:
        vec_env: Union[DummyVecEnv, SubprocVecEnv] = DummyVecEnv(env_fns)
    elif use_subprocess:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    if normalize:
        kwargs = dict(norm_obs=True, norm_reward=True, clip_obs=10.0)
        if vecnormalize_kwargs:
            kwargs.update(vecnormalize_kwargs)
        return VecNormalize(vec_env, **kwargs)
    return vec_env


def train_agent(
    total_timesteps: int = 200_000,
    n_envs: int = 4,
    difficulty: str = "medium",
    dynamic_probability: float = 0.45,
    obstacle_speed_range: Tuple[float, float] = (0.05, 0.35),
    log_dir: Path = Path("./models"),
    seed: int = 42,
    critic_strong: bool = True,
    debug: bool = False,
) -> PPO:
    """Train a PPO agent; tailored defaults for Google Colab."""

    log_dir.mkdir(parents=True, exist_ok=True)
    scenario = ScenarioConfig(
        difficulty=difficulty,
        dynamic_probability=dynamic_probability,
        obstacle_speed_range=obstacle_speed_range,
    )
    vec_env = make_vec_env(
        scenario,
        n_envs=n_envs,
        seed=seed,
        normalize=critic_strong,
        debug=debug,
    )

    if critic_strong and not isinstance(vec_env, VecNormalize):
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if critic_strong:
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[128, 128, 64]))
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=15,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=1.2,
            clip_range_vf=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
            device="cuda" if torch.cuda.is_available() else "auto",
            seed=seed,
        )
    else:
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
            device="cuda" if torch.cuda.is_available() else "auto",
            seed=seed,
        )

    checkpoint_callback = RewardCheckpoint(
        check_freq=5_000 // n_envs,
        save_path=log_dir,
        verbose=1,
    )
    success_callback = SuccessRateCallback(verbose=1 if debug else 0)
    callback = CallbackList([checkpoint_callback, success_callback])

    print("=" * 70)
    print("Training PPO agent for APF-RRT parameter optimisation")
    print("Using device:", model.device)
    print("Parallel environments:", n_envs)
    print("Total timesteps:", total_timesteps)
    print("=" * 70)

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    duration = time.time() - start
    print(f"Training finished in {duration / 60:.1f} minutes")

    if critic_strong and isinstance(vec_env, VecNormalize):
        vec_env.training = False
        vec_env.norm_reward = False
        stats_path = log_dir / "obs_normalizer.npz"
        np.savez(
            stats_path,
            mean=vec_env.obs_rms.mean,
            var=vec_env.obs_rms.var,
            clip=np.array(vec_env.clip_obs, dtype=np.float32),
            epsilon=np.array(getattr(vec_env, "epsilon", 1e-8), dtype=np.float32),
        )
        print(f"Saved observation normalisation statistics to {stats_path}")

    final_path = log_dir / "final_model"
    model.save(final_path)
    print(f"Saved final model to {final_path}")
    return model


def load_agent(model_path: Path) -> Tuple[PPO, Optional[ObservationNormalizer]]:
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / "best_model.zip"
    model = PPO.load(model_path)

    normalizer_path = model_path.parent / "obs_normalizer.npz"
    normalizer: Optional[ObservationNormalizer]
    if normalizer_path.exists():
        normalizer = ObservationNormalizer.from_file(normalizer_path)
    else:
        normalizer = None
    return model, normalizer


def benchmark_agent(
    agent: PPO,
    normalizer: Optional[ObservationNormalizer] = None,
    n_episodes: int = 40,
    difficulty: str = "medium",
    dynamic: bool = False,
    seed: int = 123,
    seeds: Optional[Sequence[int]] = None,
) -> BenchmarkMetrics:
    """Evaluate an agent and summarise success / collision metrics."""

    seeds_to_use: Sequence[int] = list(seeds) if seeds is not None else [seed]

    successes = 0
    collision_episodes = 0
    planning_times: List[float] = []
    path_lengths: List[float] = []
    node_counts: List[float] = []
    total_episodes = 0

    for eval_seed in seeds_to_use:
        scenario = ScenarioConfig(
            difficulty=difficulty,
            dynamic_probability=1.0 if dynamic else 0.0,
        )
        env = APFRRTEnv(scenario, seed=eval_seed)
        planner = RLEnhancedPlanner(agent=agent, scenario=scenario, normalizer=normalizer)

        for episode_idx in range(n_episodes):
            total_episodes += 1
            env.reset()

            result = planner.plan(env.q_start, env.q_goal, env.obstacles)
            info: Dict[str, Any] = result

            if info.get("success"):
                successes += 1
                print(f"SUCCESS at seed={eval_seed} episode={episode_idx}")
            if info.get("collision"):
                collision_episodes += 1

            planning_times.append(float(info.get("planning_time", 0.0)))
            path_lengths.append(float(info.get("path_length", 0.0)))
            node_counts.append(float(info.get("num_nodes", 0.0)))

    return BenchmarkMetrics(
        success_rate=successes / max(total_episodes, 1),
        collision_rate=collision_episodes / max(total_episodes, 1),
        avg_planning_time=float(np.mean(planning_times)) if planning_times else 0.0,
        avg_path_length=float(np.mean(path_lengths)) if path_lengths else 0.0,
        avg_nodes=float(np.mean(node_counts)) if node_counts else 0.0,
        total_episodes=total_episodes,
        dynamic=dynamic,
    )


# ---------------------------------------------------------------------------
# Planner using a trained agent
# ---------------------------------------------------------------------------


class RLEnhancedPlanner:
    """Plan paths with a trained PPO agent adjusting APF parameters."""

    def __init__(
        self,
        agent: Optional[PPO] = None,
        config: Optional[Dict[str, Any]] = None,
        scenario: Optional[ScenarioConfig] = None,
        normalizer: Optional[ObservationNormalizer] = None,
    ) -> None:
        self.agent = agent
        self.config: Dict[str, Any] = config or {}
        self.parameters = PlannerParameters()
        self.scenario = scenario or ScenarioConfig(dynamic_probability=0.0)
        self.normalizer = normalizer

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        obstacles: Sequence[Union[Obstacle, Tuple[np.ndarray, float]]],
        max_iters: int = 5_000,
        scenario: Optional[ScenarioConfig] = None,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        if self.agent is None:
            raise ValueError(
                "Agent not initialized. Please provide a trained RL agent or set "
                "agent=None for benchmarks."
            )

        scenario_cfg = scenario or self.scenario
        max_attempts = max(1, int(self.config.get("max_attempts", max_attempts)))
        base_seed = self.config.get("seed")

        base_obstacles: List[ObstacleState] = [
            self._to_obstacle_state(item, scenario_cfg.n_joints) for item in obstacles
        ]

        last_nodes: List[np.ndarray] = []
        last_metrics: Dict[str, float] = {}
        last_plan_time = 0.0

        for attempt in range(max_attempts):
            attempt_seed: Optional[int]
            if base_seed is not None:
                attempt_seed = int(base_seed) + attempt
            else:
                attempt_seed = None

            env = APFRRTEnv(
                scenario_cfg,
                seed=attempt_seed,
                debug=bool(self.config.get("debug", False)),
            )
            env.q_start = q_start.copy()
            env.q_goal = q_goal.copy()
            env.q_current = q_start.copy()
            env.goal_position = env.workspace_position(env.q_goal)
            env.obstacles = [obstacle.copy() for obstacle in base_obstacles]
            env._dynamic_active = bool(
                scenario_cfg.dynamic_probability > 0.0
                and any(np.linalg.norm(obs.velocity) > 0 for obs in env.obstacles)
            )
            env.nodes = [q_start.copy()]
            env.parameters = PlannerParameters()
            env._step_index = 0
            env.prev_dist_to_goal = env._distance_to_goal(env.q_start)

            state = env._get_state()
            if self.normalizer is not None:
                state = self.normalizer.normalize(state)
            action, _ = self.agent.predict(state, deterministic=True)
            env.parameters.apply_delta(action)

            result = env._run_planning_episode()
            plan_time = result.planning_time
            metrics = self._build_metrics(env, env.scenario.max_steps)
            metrics.update(
                {
                    "restart_attempts": float(attempt + 1),
                    "success": float(result.success),
                    "collision": float(result.collision),
                    "path_length": result.path_length,
                    "planning_time": result.planning_time,
                }
            )

            if result.success and result.path is None and env._last_path is not None:
                result.path = env._last_path

            if result.success and result.path is not None and env.goal_reached(result.path[-1]):
                path = result.path
                parents = result.parents or env._last_parents or {0: None}
                if len(path) < 2:
                    path = [q_start.copy(), q_goal.copy()]
                assert len(path) > 1, "Successful plans must include at least start and goal"
                total_path_length = float(self._compute_path_length(path))
                tree_nodes = list(env.nodes)
                planning_time = float(plan_time)
                path_points = [tuple(env.workspace_position(p)[:2]) for p in path]
                params = metrics.get("final_params")
                serialisable_params = params.tolist() if hasattr(params, "tolist") else params
                path_metadata = {
                    "nodes": float(len(tree_nodes)),
                    "iterations": float(metrics.get("iterations", len(tree_nodes))),
                    "planning_time": float(planning_time),
                    "parameters": serialisable_params,
                }
                return {
                    "success": True,
                    "path": path,
                    "path_length": float(total_path_length),
                    "num_nodes": float(len(tree_nodes)),
                    "planning_time": float(planning_time),
                    "collision": result.collision,
                    "metrics": metrics,
                    "nodes": tree_nodes,
                    "parents": parents,
                    "path_points": path_points,
                    "metadata": path_metadata,
                }

            last_plan_time = plan_time
            last_nodes = env.nodes
            last_metrics = metrics

        fallback_path: List[np.ndarray]
        if env._last_path:
            fallback_path = env._last_path
        elif last_nodes:
            fallback_path = list(last_nodes)
        else:
            fallback_path = [q_start.copy(), q_goal.copy()]

        total_path_length = float(self._compute_path_length(fallback_path))
        tree_nodes = list(last_nodes) if last_nodes else list(fallback_path)
        planning_time = float(last_plan_time)
        path_points = [tuple(env.workspace_position(p)[:2]) for p in fallback_path]
        params = last_metrics.get("final_params")
        serialisable_params = params.tolist() if hasattr(params, "tolist") else params
        path_metadata = {
            "nodes": float(len(tree_nodes)),
            "iterations": float(last_metrics.get("iterations", len(tree_nodes))),
            "planning_time": float(planning_time),
            "parameters": serialisable_params,
        }

        return {
            "success": False,
            "error": "Failed to find path",
            "path": fallback_path,
            "path_length": float(total_path_length),
            "num_nodes": float(len(tree_nodes)),
            "planning_time": float(planning_time),
            "collision": bool(last_metrics.get("collision", False)),
            "metrics": last_metrics,
            "nodes": tree_nodes,
            "parents": env._last_parents or {0: None},
            "path_points": path_points,
            "metadata": path_metadata,
        }

    def _compute_path_length(self, path: Sequence[np.ndarray]) -> float:
        if not path or len(path) < 2:
            return 0.0
        return float(
            np.sum(
                np.linalg.norm(
                    np.array(path[i], dtype=np.float64)[:3]
                    - np.array(path[i - 1], dtype=np.float64)[:3]
                )
                for i in range(1, len(path))
            )
        )

    @staticmethod
    def _build_metrics(env: APFRRTEnv, iterations: int) -> Dict[str, float]:
        params = env.parameters
        final_params = params.to_array()
        return {
            "iterations": float(iterations),
            "nodes": float(len(env.nodes)),
            "final_params": final_params,
            "dynamic": float(env._dynamic_active),
            "K_att_final": float(params.attractive_gain),
            "K_rep_final": float(params.repulsive_gain),
            "influence_distance_final": float(params.influence_distance),
            "step_size_final": float(params.step_size),
            "goal_bias_final": float(params.goal_bias),
        }

    @staticmethod
    def _to_obstacle_state(
        obstacle: Union[Obstacle, Tuple[np.ndarray, float]],
        n_joints: int,
    ) -> ObstacleState:
        if isinstance(obstacle, ObstacleState):
            return obstacle.copy()
        centre, radius = obstacle
        centre_arr = np.asarray(centre, dtype=np.float32).copy()
        if centre_arr.shape[0] != n_joints:
            raise ValueError("Obstacle dimension mismatch with scenario joints")
        return ObstacleState(centre_arr, float(radius), np.zeros(n_joints, dtype=np.float32))


# ---------------------------------------------------------------------------
# Backwards compatibility aliases
# ---------------------------------------------------------------------------


class RLEnhancedAPF_RRT(RLEnhancedPlanner):
    """Compatibility wrapper preserving the legacy class name."""


class APF_RRT_Environment(APFRRTEnv):
    """Compatibility wrapper used by older quick-test scripts."""


# ---------------------------------------------------------------------------
# Visualisation utilities
# ---------------------------------------------------------------------------


def plot_3d_path(
    nodes: Sequence[np.ndarray],
    path: Optional[Sequence[np.ndarray]] = None,
    obstacles: Optional[Sequence[Obstacle]] = None,
    parents: Optional[Dict[int, Optional[int]]] = None,
    *,
    show_tree: bool = True,
    show_obstacles: bool = True,
    show_apf: bool = False,
    apf_fn: Optional[Any] = None,
    apf_domain: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
    use_contours: bool = False,
    show: bool = True,
    save_path: Optional[Path] = None,
    elevation: Optional[float] = None,
    azimuth: Optional[float] = None,
) -> None:
    """Visualise the exploration tree and final path in 3D with publication styling."""

    if plt is None:
        raise ImportError("matplotlib is required for 3D visualisation")

    from visualization_3d import plot_apf_field_3d, plot_obstacles_3d, plot_path_3d, plot_rrt_tree_3d

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if show_tree and nodes and parents:
        plot_rrt_tree_3d(nodes, parents, ax)
    elif show_tree and nodes:
        plot_rrt_tree_3d(nodes, {idx: idx - 1 for idx in range(len(nodes))}, ax)

    if obstacles and show_obstacles:
        plot_obstacles_3d(obstacles, ax)

    if path:
        plot_path_3d(path, ax)

    if nodes:
        coords = np.asarray(nodes)
        mins = coords[:, :3].min(axis=0)
        maxs = coords[:, :3].max(axis=0)
        span = np.maximum(maxs - mins, 1e-3)
        padding = span * 0.2
        ax.set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
        ax.set_ylim(mins[1] - padding[1], maxs[1] + padding[1])
        ax.set_zlim(mins[2] - padding[2], maxs[2] + padding[2])

    if show_apf and apf_fn and apf_domain:
        plot_apf_field_3d(apf_fn, apf_domain, ax, use_contours=use_contours)

    if nodes:
        start = np.asarray(nodes[0])[:3]
        ax.scatter(*start, s=120, c="green", depthshade=True, label="Start")
    if path:
        goal = np.asarray(path[-1])[:3]
        ax.scatter(*goal, s=140, c="gold", depthshade=True, label="Goal")

    ax.set_title("APF-RRT exploration (first 3 joints)")
    ax.set_xlabel("Joint 1")
    ax.set_ylabel("Joint 2")
    ax.set_zlabel("Joint 3")
    ax.legend(loc="upper right", frameon=False)
    ax.set_facecolor("white")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)
    if elevation is not None:
        ax.view_init(elev=elevation, azim=ax.azim if azimuth is None else azimuth)
    if azimuth is not None and elevation is None:
        ax.view_init(elev=ax.elev, azim=azimuth)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL-enhanced APF-RRT planner")
    subparsers = parser.add_subparsers(dest="mode", required=False)

    train_parser = subparsers.add_parser("train", help="Train a new PPO agent")
    train_parser.add_argument("--timesteps", type=int, default=200_000)
    train_parser.add_argument("--n-envs", type=int, default=4)
    train_parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    train_parser.add_argument("--dynamic-prob", type=float, default=0.45, help="Probability of dynamic obstacles during training")
    train_parser.add_argument(
        "--obstacle-speed-min",
        type=float,
        default=0.05,
        help="Minimum obstacle speed for dynamic scenarios",
    )
    train_parser.add_argument(
        "--obstacle-speed-max",
        type=float,
        default=0.35,
        help="Maximum obstacle speed for dynamic scenarios",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--log-dir", type=Path, default=Path("./models"))
    train_parser.add_argument(
        "--critic-strong",
        dest="critic_strong",
        action="store_true",
        help="Enable the enhanced critic architecture and reward normalisation",
    )
    train_parser.add_argument(
        "--no-critic-strong",
        dest="critic_strong",
        action="store_false",
        help="Disable the enhanced critic configuration (revert to legacy settings)",
    )
    train_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging and parameter traces during training",
    )
    train_parser.set_defaults(critic_strong=True)

    test_parser = subparsers.add_parser("test", help="Evaluate with a trained agent")
    test_parser.add_argument("--model", type=Path, default=Path("./models/best_model.zip"))
    test_parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    test_parser.add_argument("--dynamic", action="store_true", help="Use dynamic obstacle scenarios")
    test_parser.add_argument("--plot", action="store_true", help="Show legacy 3D visualisation (alias for --plot-3d)")
    test_parser.add_argument("--plot-3d", action="store_true", help="Show publication-style 3D visualisation")
    test_parser.add_argument(
        "--show-tree",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render the RRT tree (can be slow for large trees)",
    )
    test_parser.add_argument(
        "--show-obstacles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render 3D obstacle extrusions",
    )
    test_parser.add_argument(
        "--show-apf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Visualise APF gradients as 3D arrows/contours",
    )
    test_parser.add_argument(
        "--save-fig",
        type=Path,
        nargs="?",
        const=Path("apf_rrt_figure"),
        default=None,
        help="Save the 3D visualisation as high-DPI PNG and PDF",
    )
    test_parser.add_argument("--elevation", type=float, default=None, help="Camera elevation for the 3D view")
    test_parser.add_argument("--azimuth", type=float, default=None, help="Camera azimuth for the 3D view")
    test_parser.add_argument(
        "--restarts",
        type=int,
        default=3,
        help="Number of planner restarts if the initial attempt fails",
    )
    test_parser.add_argument("--ros-publish", action="store_true", help="Publish nav_msgs/Path to ROS")
    test_parser.add_argument(
        "--moveit-execute",
        action="store_true",
        help="Execute the planned path through MoveIt",
    )
    test_parser.add_argument(
        "--save-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist the planned path and metadata to disk",
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="Report success / collision metrics")
    benchmark_parser.add_argument("--model", type=Path, default=Path("./models/best_model.zip"))
    benchmark_parser.add_argument("--episodes", type=int, default=40)
    benchmark_parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    benchmark_parser.add_argument("--seed", type=int, default=123)
    benchmark_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional list of seeds to evaluate; overrides --seed when provided",
    )
    benchmark_parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Also evaluate on dynamic obstacle scenarios",
    )

    parser.set_defaults(mode="test")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "train":
        obstacle_speed_range = (args.obstacle_speed_min, args.obstacle_speed_max)
        if obstacle_speed_range[0] > obstacle_speed_range[1]:
            raise ValueError("Minimum obstacle speed must not exceed maximum speed")
        train_agent(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            difficulty=args.difficulty,
            dynamic_probability=args.dynamic_prob,
            obstacle_speed_range=obstacle_speed_range,
            log_dir=args.log_dir,
            seed=args.seed,
            critic_strong=args.critic_strong,
            debug=args.debug,
        )
        return

    if args.mode == "benchmark":
        agent, normalizer = load_agent(args.model)
        scenario_flags = [False]
        if args.dynamic:
            scenario_flags.append(True)

        results = {}
        for dynamic_flag in scenario_flags:
            metrics = benchmark_agent(
                agent,
                normalizer,
                n_episodes=args.episodes,
                difficulty=args.difficulty,
                dynamic=dynamic_flag,
                seed=args.seed,
                seeds=args.seeds,
            )
            results["Dynamic" if dynamic_flag else "Static"] = metrics

        headers = ["Metric"] + list(results.keys())
        print(" | ".join(headers))
        print(" | ".join(["---"] * len(headers)))
        def _format_rate(metric_name: str) -> str:
            cells = [metric_name]
            for metrics in results.values():
                value = metrics.success_rate if metric_name == "Success Rate" else (1.0 - metrics.collision_rate)
                cells.append(f"{value * 100:.1f}%")
            return " | ".join(cells)

        print(_format_rate("Success Rate"))
        print(_format_rate("Collision Avoidance"))

        time_row = ["Avg Planning Time"]
        for metrics in results.values():
            time_row.append(f"{metrics.avg_planning_time:.3f}s")
        print(" | ".join(time_row))

        length_row = ["Avg Path Length"]
        for metrics in results.values():
            length_row.append(f"{metrics.avg_path_length:.3f}")
        print(" | ".join(length_row))

        nodes_row = ["Avg Nodes"]
        for metrics in results.values():
            nodes_row.append(f"{metrics.avg_nodes:.1f}")
        print(" | ".join(nodes_row))
        return

    agent, normalizer = load_agent(args.model)
    scenario = ScenarioConfig(
        difficulty=args.difficulty,
        dynamic_probability=1.0 if getattr(args, "dynamic", False) else 0.0,
    )
    planner = RLEnhancedPlanner(agent, scenario=scenario, normalizer=normalizer)

    q_start = np.array([0.8, 1.2, -0.6, -0.4, 0.5, 0.2])
    q_goal = np.zeros(6)
    base_obstacles: Sequence[Tuple[np.ndarray, float]] = [
        (np.array([0.5, 0.7, -0.3, 0.0, 0.3, 0.0]), 0.5),
        (np.array([0.2, 0.5, -0.5, -0.1, 0.5, 0.1]), 0.4),
    ]

    rng = np.random.default_rng(123)
    obstacles: List[Obstacle] = []
    for centre, radius in base_obstacles:
        if scenario.dynamic_probability > 0.0:
            direction = rng.normal(size=scenario.n_joints)
            direction /= np.linalg.norm(direction) + 1e-9
            speed = np.mean(scenario.obstacle_speed_range)
            velocity = direction * speed
        else:
            velocity = np.zeros(scenario.n_joints)
        obstacles.append(ObstacleState(centre.copy(), radius, velocity))

    result = planner.plan(
        q_start,
        q_goal,
        obstacles,
        max_attempts=args.restarts,
    )
    if not result.get("success", False):
        print("✗ Failed to find a collision-free path")
    else:
        path = result.get("path", [])
        nodes = result.get("nodes", [])
        metrics = result.get("metrics", {})
        path_points = result.get("path_points", [tuple(np.asarray(p)[:2]) for p in path])
        params = metrics.get("final_params")
        serialisable_params = params.tolist() if hasattr(params, "tolist") else params
        metadata = result.get(
            "metadata",
            {
                "nodes": len(nodes),
                "iterations": metrics.get("iterations", len(nodes)),
                "planning_time": result.get("planning_time", 0.0),
                "parameters": serialisable_params,
            },
        )
        print("✓ Path found")
        print(f"Iterations: {metrics.get('iterations', len(nodes))}")
        print(f"Nodes: {metrics.get('nodes', len(nodes))}")
        print(f"Planning time: {result.get('planning_time', 0.0):.3f}s")
        print(f"Dynamic scenario: {bool(metrics.get('dynamic', False))}")
        print("Final parameters:", metrics.get("final_params"))
        if getattr(args, "save_path", True):
            export_path(path_points, metadata)
        ros_bridge = None
        if getattr(args, "ros_publish", False) or getattr(args, "moveit_execute", False):
            try:
                from ros_moveit_bridge import APFRRT_ROSBridge

                ros_bridge = APFRRT_ROSBridge()
            except Exception as exc:  # pragma: no cover - ROS optional
                print(f"Failed to initialise ROS bridge: {exc}")
        if ros_bridge and getattr(args, "ros_publish", False):
            try:
                ros_bridge.publish_path(path_points)
            except Exception as exc:  # pragma: no cover - ROS optional
                print(f"Failed to publish path to ROS: {exc}")
        if ros_bridge and getattr(args, "moveit_execute", False):
            try:
                ros_bridge.send_to_moveit(path_points)
            except Exception as exc:  # pragma: no cover - ROS optional
                print(f"Failed to execute path in MoveIt: {exc}")
        plot_requested = bool(getattr(args, "plot_3d", False) or getattr(args, "plot", False))
        if plot_requested:
            final_params = metrics.get("final_params")
            params_array = np.asarray(final_params) if final_params is not None else None
            params = PlannerParameters()
            if params_array is not None and params_array.shape[0] >= 5:
                (
                    params.attractive_gain,
                    params.repulsive_gain,
                    params.influence_distance,
                    params.step_size,
                    params.goal_bias,
                ) = params_array[:5]

            def _apf_field(sample: np.ndarray) -> np.ndarray:
                q_full = np.zeros_like(q_goal, dtype=np.float64)
                q_full[:3] = sample[:3]
                if q_full.shape[0] > q_goal.shape[0]:
                    q_full = q_full[: q_goal.shape[0]]
                v_att = q_goal - q_full
                d_att = np.linalg.norm(v_att)
                f_att = params.attractive_gain * (v_att / (d_att + 1e-9)) if d_att > 0 else np.zeros_like(q_full)
                f_rep = np.zeros_like(q_full)
                for obstacle in obstacles:
                    centre = obstacle.centre if hasattr(obstacle, "centre") else np.asarray(obstacle[0])
                    radius = obstacle.radius if hasattr(obstacle, "radius") else float(obstacle[1])
                    diff = q_full - centre
                    dist = np.linalg.norm(diff) - radius
                    if 0.0 < dist <= params.influence_distance:
                        magnitude = params.repulsive_gain * (
                            (1.0 / dist**2) * (1.0 / dist - 1.0 / params.influence_distance)
                        )
                        f_rep += magnitude * (diff / (np.linalg.norm(diff) + 1e-9))
                return (f_att + f_rep)[:3]

            coords = np.asarray(nodes) if nodes else np.asarray(path)
            if coords.size == 0:
                coords = np.zeros((1, 3))
            coords3 = coords[:, :3]
            mins = coords3.min(axis=0)
            maxs = coords3.max(axis=0)
            span = np.maximum(maxs - mins, 1e-3)
            padding = span * 0.3
            domain = tuple(
                (float(mins[i] - padding[i]), float(maxs[i] + padding[i])) for i in range(3)
            )

            plot_3d_path(
                nodes,
                path,
                obstacles,
                parents=result.get("parents"),
                show_tree=bool(getattr(args, "show_tree", False)),
                show_obstacles=bool(getattr(args, "show_obstacles", True)),
                show_apf=bool(getattr(args, "show_apf", False)),
                apf_fn=_apf_field,
                apf_domain=domain,  # type: ignore[arg-type]
                use_contours=True,
                show=True,
                save_path=getattr(args, "save_fig", None),
                elevation=getattr(args, "elevation", None),
                azimuth=getattr(args, "azimuth", None),
            )


if __name__ == "__main__":
    main()
