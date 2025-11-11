#!/usr/bin/env python3
"""High-level wrapper for training APF-guided RRT policies in Gymnasium."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from gymnasium import Env

from rl_enhanced_apf_rrt import (
    APFRRTEnv,
    BenchmarkMetrics,
    ScenarioConfig,
    benchmark_agent,
    train_agent,
)


@dataclass
class ConfigSpaceSettings:
    """Convenience container mirroring :class:`ScenarioConfig` options."""

    difficulty: str = "medium"
    dynamic_probability: float = 0.45
    obstacle_speed_min: float = 0.05
    obstacle_speed_max: float = 0.35
    seed: Optional[int] = None

    def to_scenario(self) -> ScenarioConfig:
        return ScenarioConfig(
            difficulty=self.difficulty,
            dynamic_probability=self.dynamic_probability,
            obstacle_speed_range=(self.obstacle_speed_min, self.obstacle_speed_max),
        )


class ConfigSpaceAPF_RRT(Env):
    """Drop-in Gymnasium environment for PPO training."""

    metadata = APFRRTEnv.metadata

    def __init__(self, settings: Optional[ConfigSpaceSettings] = None) -> None:
        self.settings = settings or ConfigSpaceSettings()
        self._env = APFRRTEnv(self.settings.to_scenario(), seed=self.settings.seed)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    # ------------------------------------------------------------------
    # Gymnasium API delegates
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray):  # type: ignore[override]
        return self._env.step(action)

    def render(self):  # pragma: no cover - passthrough
        self._env.render()

    def close(self):  # pragma: no cover - passthrough
        self._env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def scenario(self) -> ScenarioConfig:
        return self._env.scenario

    @property
    def native_env(self) -> APFRRTEnv:
        return self._env


def train_colab_ready_agent(
    total_timesteps: int = 300_000,
    n_envs: int = 4,
    settings: Optional[ConfigSpaceSettings] = None,
    log_dir: str = "./models",
    seed: int = 42,
):
    """Thin wrapper around :func:`train_agent` with sensible defaults."""

    settings = settings or ConfigSpaceSettings()
    effective_seed = settings.seed if settings.seed is not None else seed
    return train_agent(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        difficulty=settings.difficulty,
        dynamic_probability=settings.dynamic_probability,
        obstacle_speed_range=(settings.obstacle_speed_min, settings.obstacle_speed_max),
        log_dir=Path(log_dir),
        seed=effective_seed,
        critic_strong=True,
    )


def evaluate_agent(
    agent,
    episodes: int = 40,
    settings: Optional[ConfigSpaceSettings] = None,
    include_dynamic: bool = True,
) -> Sequence[BenchmarkMetrics]:
    """Convenience benchmarking in static and dynamic obstacle regimes."""

    settings = settings or ConfigSpaceSettings()
    if isinstance(agent, tuple):
        policy, normalizer = agent
    else:
        policy, normalizer = agent, None

    results = [
        benchmark_agent(
            policy,
            normalizer,
            n_episodes=episodes,
            difficulty=settings.difficulty,
            dynamic=False,
            seed=settings.seed or 0,
        )
    ]
    if include_dynamic:
        results.append(
            benchmark_agent(
                policy,
                normalizer,
                n_episodes=episodes,
                difficulty=settings.difficulty,
                dynamic=True,
                seed=(settings.seed or 0) + 1,
            )
        )
    return results


if __name__ == "__main__":  # pragma: no cover - quick smoke test
    env = ConfigSpaceAPF_RRT()
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"Collected {total_reward:.2f} reward over 10 random steps")
