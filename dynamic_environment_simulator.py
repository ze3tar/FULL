#!/usr/bin/env python3
"""Lightweight dynamic environment simulator for on-the-fly obstacle updates."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ObstacleState:
    """Runtime state of a spherical obstacle."""

    obstacle_id: str
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "ObstacleState":
        return ObstacleState(
            obstacle_id=self.obstacle_id,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            radius=float(self.radius),
            acceleration=self.acceleration.copy(),
        )


class DynamicEnvironmentSimulator:
    """Generate and broadcast dynamic obstacle fields in real time."""

    def __init__(
        self,
        frequency_hz: float = 10.0,
        world_bounds: Tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        self.frequency_hz = max(frequency_hz, 0.5)
        self.world_bounds = world_bounds
        self.obstacles: List[ObstacleState] = []
        self.subscribers: List[Callable[[Sequence[ObstacleState]], None]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Obstacle creation helpers
    # ------------------------------------------------------------------
    def add_random_spheres(
        self,
        count: int = 5,
        speed_range: Tuple[float, float] = (0.05, 0.35),
        radius_range: Tuple[float, float] = (0.1, 0.6),
    ) -> None:
        """Spawn random moving spheres with bounded initial velocities."""

        low, high = self.world_bounds
        for idx in range(count):
            position = np.random.uniform(low, high, size=3)
            speed = np.random.uniform(*speed_range)
            direction = np.random.uniform(-1, 1, size=3)
            direction /= np.linalg.norm(direction) + 1e-8
            velocity = direction * speed
            radius = float(np.random.uniform(*radius_range))

            self.obstacles.append(
                ObstacleState(
                    obstacle_id=f"rand_{len(self.obstacles)+1}",
                    position=position,
                    velocity=velocity,
                    radius=radius,
                )
            )

    def add_accelerating_object(
        self,
        obstacle_id: str,
        position: Sequence[float],
        velocity: Sequence[float],
        acceleration: Sequence[float],
        radius: float = 0.2,
    ) -> None:
        """Register an obstacle with explicit acceleration dynamics."""

        self.obstacles.append(
            ObstacleState(
                obstacle_id=obstacle_id,
                position=np.asarray(position, dtype=float),
                velocity=np.asarray(velocity, dtype=float),
                radius=float(radius),
                acceleration=np.asarray(acceleration, dtype=float),
            )
        )

    # ------------------------------------------------------------------
    # Runtime management
    # ------------------------------------------------------------------
    def subscribe(self, callback: Callable[[Sequence[ObstacleState]], None]) -> None:
        """Register a callback that receives obstacle snapshots."""

        self.subscribers.append(callback)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run_loop(self) -> None:
        period = 1.0 / self.frequency_hz
        last_time = time.time()
        while self._running:
            now = time.time()
            dt = now - last_time
            last_time = now

            with self._lock:
                self._step(dt)
                snapshot = [obs.copy() for obs in self.obstacles]

            for cb in self.subscribers:
                cb(snapshot)

            elapsed = time.time() - now
            time.sleep(max(0.0, period - elapsed))

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        for obs in self.obstacles:
            obs.velocity += obs.acceleration * dt
            obs.position += obs.velocity * dt
            self._handle_bounds(obs)

        self._resolve_interactions()

    def _handle_bounds(self, obs: ObstacleState) -> None:
        low, high = self.world_bounds
        for axis in range(3):
            if obs.position[axis] - obs.radius < low:
                obs.position[axis] = low + obs.radius
                obs.velocity[axis] = abs(obs.velocity[axis])
            elif obs.position[axis] + obs.radius > high:
                obs.position[axis] = high - obs.radius
                obs.velocity[axis] = -abs(obs.velocity[axis])

    def _resolve_interactions(self) -> None:
        for i, obs_a in enumerate(self.obstacles):
            for obs_b in self.obstacles[i + 1 :]:
                delta = obs_b.position - obs_a.position
                dist = np.linalg.norm(delta)
                min_dist = obs_a.radius + obs_b.radius
                if dist < min_dist and dist > 1e-6:
                    normal = delta / dist
                    overlap = min_dist - dist
                    obs_a.position -= normal * overlap / 2
                    obs_b.position += normal * overlap / 2
                    obs_a.velocity, obs_b.velocity = obs_b.velocity.copy(), obs_a.velocity.copy()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def capture_snapshot(self) -> List[ObstacleState]:
        with self._lock:
            return [obs.copy() for obs in self.obstacles]

    def run_steps(self, n_steps: int, dt: float = 0.1) -> List[List[ObstacleState]]:
        """Deterministic stepping useful for unit tests or offline rollouts."""

        snapshots: List[List[ObstacleState]] = []
        with self._lock:
            for _ in range(n_steps):
                self._step(dt)
                snapshots.append([obs.copy() for obs in self.obstacles])
        return snapshots


__all__ = ["ObstacleState", "DynamicEnvironmentSimulator"]
