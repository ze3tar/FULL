#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINT_MIN = np.array([-3.107, -2.269, -2.356, -3.107, -2.234, -6.283], dtype=float)
JOINT_MAX = np.array([3.107, 2.269, 2.356, 3.107, 2.234, 6.283], dtype=float)
VELOCITY_LIMITS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)


def _clamp_positions(points: np.ndarray) -> np.ndarray:
    return np.clip(points, JOINT_MIN, JOINT_MAX)


def _enforce_velocity(points: np.ndarray, dt: float) -> np.ndarray:
    capped = points.copy()
    for i in range(1, len(points)):
        delta = capped[i] - capped[i - 1]
        for j in range(6):
            max_delta = VELOCITY_LIMITS[j] * dt
            delta[j] = max(min(delta[j], max_delta), -max_delta)
        capped[i] = capped[i - 1] + delta
    return capped


class TrajectoryRefiner:
    def __init__(self, dt: float = 0.1, smoothing_steps: int = 80, alpha: float = 0.1):
        self.dt = dt
        self.smoothing_steps = smoothing_steps
        self.alpha = alpha

    def refine(self, traj: JointTrajectory, model: Optional[object] = None) -> JointTrajectory:
        if not traj.points:
            return traj

        points = np.array([p.positions for p in traj.points], dtype=float)
        points = _clamp_positions(points)

                                    
        if model is not None:
            try:
                import torch

                tensor_in = torch.tensor(points, dtype=torch.float32)
                delta = model(tensor_in).detach().cpu().numpy()
                if delta.shape == points.shape:
                    points = points + delta
                    points = _clamp_positions(points)
            except Exception:
                pass

                                                            
        for _ in range(self.smoothing_steps):
            grad = np.zeros_like(points)
            grad[1:-1] = 2 * points[1:-1] - points[:-2] - points[2:]
            points[1:-1] -= self.alpha * grad[1:-1]
            points = _clamp_positions(points)
            points = _enforce_velocity(points, self.dt)

        refined = JointTrajectory()
        refined.joint_names = traj.joint_names
        time = 0.0
        for row in points:
            pt = JointTrajectoryPoint()
            pt.positions = row.tolist()
            pt.velocities = [0.0] * 6
            time += self.dt
            pt.time_from_start = Duration(seconds=time).to_msg()
            refined.points.append(pt)
        return refined


class DummyNode(Node):
    """A helper node for standalone testing of the refiner."""

    def __init__(self):
        super().__init__("trajectory_refiner_test")

    def run_example(self):
        traj = JointTrajectory()
        traj.joint_names = [f"joint{i+1}" for i in range(6)]
        for k in range(10):
            pt = JointTrajectoryPoint()
            pt.positions = [math.sin(0.1 * k)] * 6
            pt.time_from_start = Duration(seconds=0.1 * (k + 1)).to_msg()
            traj.points.append(pt)
        refiner = TrajectoryRefiner()
        refined = refiner.refine(traj)
        self.get_logger().info(f"Refined trajectory has {len(refined.points)} points")


def main():
    rclpy.init()
    node = DummyNode()
    node.run_example()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
