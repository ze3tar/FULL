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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:  # pragma: no cover - optional dependency guard
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover - TensorFlow is optional
    tf = None  # type: ignore

import numpy as np
import torch
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import explained_variance
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# Matplotlib is an optional dependency; importing lazily keeps the module usable
# without it (e.g. on headless Colab runtimes before ``pip install matplotlib``).
try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
except Exception:  # pragma: no cover - optional dependency
    plt = None


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
    goal_tolerance: float = 0.18
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

Obstacle = ObstacleState


class APFRRTEnv(Env):
    """Gymnasium environment exposing APF-RRT planning dynamics."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario: ScenarioConfig,
        parameters: Optional[PlannerParameters] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.parameters = parameters or PlannerParameters()
        self.rng = np.random.default_rng(seed)

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
        self.obstacles: List[Obstacle] = []
        self._dynamic_active = False
        self.nodes: List[np.ndarray] = []
        self._step_index = 0

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
        self.nodes = [self.q_start.copy()]
        self.obstacles = self._generate_obstacles(self.scenario.obstacle_count)
        self._step_index = 0
        return self._get_state(), {}

    def step(self, action: np.ndarray):
        step_start = time.perf_counter()
        self._step_index += 1
        self.parameters.apply_delta(action)

        q_rand = self._sample_random_configuration()
        idx_near, q_near = self._find_nearest_node(q_rand)

        direction = self._compute_direction(q_near, q_rand)
        q_new = np.clip(
            q_near + direction * self.parameters.step_size,
            self.scenario.joint_min,
            self.scenario.joint_max,
        )

        done = False
        truncated = False
        info: Dict[str, float] = {}
        reward = -0.1  # default time penalty
        collision_flag = False

        if self._in_collision(q_new):
            reward -= 5.0
            collision_flag = True
        else:
            self.nodes.append(q_new)
            self.q_current = q_new

            old_dist = np.linalg.norm(q_near - self.q_goal)
            new_dist = np.linalg.norm(q_new - self.q_goal)
            progress = old_dist - new_dist
            reward += 9.0 * progress

            min_clearance = self._minimum_clearance(q_new)
            reward += 2.0 * math.tanh(max(min_clearance - 0.1, 0.0))

            if new_dist < self.scenario.goal_tolerance:
                reward += 120.0
                done = True
                info["reached_goal"] = 1.0

        reward -= 0.01 * len(self.nodes)

        if collision_flag:
            info["collision"] = 1.0

        if self._step_index >= self.scenario.max_steps and not done:
            done = True
            truncated = True
            info["timeout"] = 1.0
            reward -= 40.0

        self._advance_obstacles()

        info["clearance"] = float(self._minimum_clearance(self.q_current))
        info["dynamic"] = float(self._dynamic_active)
        info["nearest_index"] = float(idx_near)
        info["step_ms"] = float((time.perf_counter() - step_start) * 1_000.0)
        return self._get_state(), reward, done, truncated, info

    def render(self):  # pragma: no cover - simple debug rendering
        print(
            f"Step {self._step_index} – dist_to_goal: {np.linalg.norm(self.q_current - self.q_goal):.3f} "
            f"nodes: {len(self.nodes)}"
        )

    # -- Planning utilities ------------------------------------------------
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

        components = np.stack(
            [
                towards_rand / (np.linalg.norm(towards_rand) + 1e-9),
                towards_goal / (np.linalg.norm(towards_goal) + 1e-9),
                total_force / (np.linalg.norm(total_force) + 1e-9),
            ]
        )
        weights = np.array([0.45, 0.35, 0.20], dtype=np.float32)
        direction = (weights[:, None] * components).sum(axis=0)
        return direction / (np.linalg.norm(direction) + 1e-9)

    def _apf_force(self, q: np.ndarray) -> np.ndarray:
        params = self.parameters
        v_att = self.q_goal - q
        d_att = np.linalg.norm(v_att)
        f_att = params.attractive_gain * (v_att / (d_att + 1e-9)) if d_att > 0 else np.zeros_like(q)

        f_rep = np.zeros_like(q)
        for obstacle in self.obstacles:
            diff = q - obstacle.centre
            dist = np.linalg.norm(diff) - obstacle.radius
            if 0.0 < dist <= params.influence_distance:
                magnitude = params.repulsive_gain * ((1.0 / dist**2) * (1.0 / dist - 1.0 / params.influence_distance))
                f_rep += magnitude * (diff / (np.linalg.norm(diff) + 1e-9))
        return f_att + f_rep

    def _minimum_clearance(self, q: np.ndarray) -> float:
        distances = [np.linalg.norm(q - obstacle.centre) - obstacle.radius for obstacle in self.obstacles]
        return min(distances) if distances else 10.0

    def _in_collision(self, q: np.ndarray) -> bool:
        return any(np.linalg.norm(q - obstacle.centre) < obstacle.radius for obstacle in self.obstacles)

    def _advance_obstacles(self) -> None:
        if not self._dynamic_active:
            return
        for obstacle in self.obstacles:
            obstacle.advance(
                self.scenario.joint_min,
                self.scenario.joint_max,
                self.scenario.dynamic_time_step,
            )

    def _get_state(self) -> np.ndarray:
        dist = np.linalg.norm(self.q_current - self.q_goal)
        heading_vector = self.q_goal[:3] - self.q_current[:3]
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


class ValueDiagnosticsCallback(BaseCallback):
    """Logs the critic explained variance each rollout to monitor learning."""

    def _on_rollout_end(self) -> bool:
        values = self.model.rollout_buffer.values.flatten()
        returns = self.model.rollout_buffer.returns.flatten()
        variance = explained_variance(returns, values)
        self.logger.record("diagnostics/explained_variance", float(variance))
        if self.verbose > 0:
            print(f"Explained variance: {variance:.3f}")
        return True


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkMetrics:
    """Aggregate statistics over evaluation episodes."""

    success_rate: float
    collision_free_rate: float
    avg_replanning_time_ms: float
    dynamic: bool

    def as_dict(self) -> Dict[str, float]:
        return {
            "success_rate": self.success_rate,
            "collision_free_rate": self.collision_free_rate,
            "avg_replanning_time_ms": self.avg_replanning_time_ms,
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
) -> Union[DummyVecEnv, VecNormalize]:
    """Create a vectorised environment for PPO training."""

    def _factory(rank: int):
        def _init():
            env = APFRRTEnv(scenario, seed=seed + rank)
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
    )

    if critic_strong:
        policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[128, 128, 64])])
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
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048 // n_envs,
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
    diagnostics_callback = ValueDiagnosticsCallback(verbose=1)
    callback = CallbackList([checkpoint_callback, diagnostics_callback])

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
) -> BenchmarkMetrics:
    """Evaluate an agent and summarise success / collision metrics."""

    scenario = ScenarioConfig(
        difficulty=difficulty,
        dynamic_probability=1.0 if dynamic else 0.0,
    )
    env = APFRRTEnv(scenario, seed=seed)

    successes = 0
    collision_episodes = 0
    replanning_times: List[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        if normalizer is not None:
            agent_obs = normalizer.normalize(obs)
        else:
            agent_obs = obs
        done = False
        truncated = False
        episode_collided = False
        step_times: List[float] = []

        while not (done or truncated):
            action, _ = agent.predict(agent_obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            if normalizer is not None:
                agent_obs = normalizer.normalize(obs)
            else:
                agent_obs = obs
            step_times.append(info.get("step_ms", 0.0))
            if info.get("collision", 0.0):
                episode_collided = True

        if info.get("reached_goal", 0.0):
            successes += 1
        if episode_collided:
            collision_episodes += 1
        if step_times:
            replanning_times.append(float(np.mean(step_times)))

    if not replanning_times:
        replanning_times.append(0.0)

    return BenchmarkMetrics(
        success_rate=successes / max(n_episodes, 1),
        collision_free_rate=1.0 - collision_episodes / max(n_episodes, 1),
        avg_replanning_time_ms=float(np.mean(replanning_times)),
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
    ) -> Tuple[Optional[List[np.ndarray]], List[np.ndarray], float, Dict[str, float]]:
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

            env = APFRRTEnv(scenario_cfg, seed=attempt_seed)
            env.q_start = q_start.copy()
            env.q_goal = q_goal.copy()
            env.q_current = q_start.copy()
            env.obstacles = [obstacle.copy() for obstacle in base_obstacles]
            env._dynamic_active = bool(
                scenario_cfg.dynamic_probability > 0.0
                and any(np.linalg.norm(obs.velocity) > 0 for obs in env.obstacles)
            )
            env.nodes = [q_start.copy()]
            env.parameters = PlannerParameters()
            env._step_index = 0

            parents: Dict[int, Optional[int]] = {0: None}
            start_time = time.perf_counter()

            for iteration in range(max_iters):
                state = env._get_state()
                if self.normalizer is not None:
                    state = self.normalizer.normalize(state)
                action, _ = self.agent.predict(state, deterministic=True)
                env.parameters.apply_delta(action)

                q_rand = env._sample_random_configuration()
                idx_near, q_near = env._find_nearest_node(q_rand)
                direction = env._compute_direction(q_near, q_rand)
                q_new = np.clip(
                    q_near + direction * env.parameters.step_size,
                    env.scenario.joint_min,
                    env.scenario.joint_max,
                )

                if env._in_collision(q_new):
                    continue

                env.nodes.append(q_new)
                parents[len(env.nodes) - 1] = idx_near

                if np.linalg.norm(q_new - env.q_goal) < 0.2:
                    path = self._reconstruct_path(
                        parents, len(env.nodes) - 1, env.nodes, env.q_goal
                    )
                    plan_time = float(time.perf_counter() - start_time)
                    metrics = self._build_metrics(env, iteration + 1)
                    metrics["restart_attempts"] = float(attempt + 1)
                    return path, env.nodes, plan_time, metrics

            last_plan_time = float(time.perf_counter() - start_time)
            last_nodes = env.nodes
            last_metrics = self._build_metrics(env, max_iters)
            last_metrics["restart_attempts"] = float(attempt + 1)

        return None, last_nodes, last_plan_time, last_metrics

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
    def _reconstruct_path(
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
    show: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """Visualise the exploration tree and final path in 3D."""

    if plt is None:
        raise ImportError("matplotlib is required for 3D visualisation")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if nodes:
        coords = np.array(nodes)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=8, alpha=0.3, label="Tree")

    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], "r-", linewidth=2, label="Path")

    if obstacles:
        for obstacle in obstacles:
            if isinstance(obstacle, ObstacleState):
                centre = obstacle.centre
                radius = obstacle.radius
            else:
                centre, radius = obstacle
            u, v = np.mgrid[0 : 2 * np.pi : 12j, 0 : np.pi : 6j]
            x = centre[0] + radius * np.cos(u) * np.sin(v)
            y = centre[1] + radius * np.sin(u) * np.sin(v)
            z = centre[2] + radius * np.cos(v)
            ax.plot_surface(x, y, z, color="grey", alpha=0.2)

    ax.set_title("APF-RRT exploration (first 3 joints)")
    ax.set_xlabel("Joint 1")
    ax.set_ylabel("Joint 2")
    ax.set_zlabel("Joint 3")
    ax.legend(loc="upper right")
    ax.grid(True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
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
    train_parser.set_defaults(critic_strong=True)

    test_parser = subparsers.add_parser("test", help="Evaluate with a trained agent")
    test_parser.add_argument("--model", type=Path, default=Path("./models/best_model.zip"))
    test_parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    test_parser.add_argument("--dynamic", action="store_true", help="Use dynamic obstacle scenarios")
    test_parser.add_argument("--plot", action="store_true", help="Show 3D visualisation")
    test_parser.add_argument(
        "--restarts",
        type=int,
        default=3,
        help="Number of planner restarts if the initial attempt fails",
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="Report success / collision metrics")
    benchmark_parser.add_argument("--model", type=Path, default=Path("./models/best_model.zip"))
    benchmark_parser.add_argument("--episodes", type=int, default=40)
    benchmark_parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    benchmark_parser.add_argument("--seed", type=int, default=123)
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
            )
            results["Dynamic" if dynamic_flag else "Static"] = metrics

        headers = ["Metric"] + list(results.keys())
        print(" | ".join(headers))
        print(" | ".join(["---"] * len(headers)))
        def _format_rate(metric_name: str) -> str:
            cells = [metric_name]
            for metrics in results.values():
                value = metrics.success_rate if metric_name == "Success Rate" else metrics.collision_free_rate
                cells.append(f"{value * 100:.1f}%")
            return " | ".join(cells)

        print(_format_rate("Success Rate"))
        print(_format_rate("Collision Avoidance"))

        time_row = ["Avg Replanning Time"]
        for metrics in results.values():
            time_row.append(f"{metrics.avg_replanning_time_ms:.1f}ms")
        print(" | ".join(time_row))
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

    path, nodes, plan_time, metrics = planner.plan(
        q_start,
        q_goal,
        obstacles,
        max_attempts=args.restarts,
    )
    if path is None:
        print("✗ Failed to find a collision-free path")
    else:
        print("✓ Path found")
        print(f"Iterations: {metrics['iterations']}")
        print(f"Nodes: {metrics['nodes']}")
        print(f"Planning time: {plan_time:.3f}s")
        print(f"Dynamic scenario: {bool(metrics['dynamic'])}")
        print("Final parameters:", metrics["final_params"])
        if getattr(args, "plot", False):
            plot_3d_path(nodes, path, obstacles, show=True)


if __name__ == "__main__":
    main()
