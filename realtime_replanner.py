#!/usr/bin/env python3
"""Real-time replanning engine that couples prediction + APF-RRT + PSO."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from dynamic_environment_simulator import ObstacleState
from pso_path_smoother import PSOPathSmoother


@dataclass
class ReplanDecision:
    """Container describing why and how a replan happened."""

    triggered: bool
    reason: str = "idle"
    path: Optional[np.ndarray] = None
    smooth_path: Optional[np.ndarray] = None
    latency_ms: float = 0.0


class RealTimeReplanner:
    """Monitor live obstacles and trigger fast replans when needed."""

    def __init__(
        self,
        planner: Optional[Callable[[np.ndarray, np.ndarray, Sequence[ObstacleState]], np.ndarray]] = None,
        smoother: Optional[PSOPathSmoother] = None,
        collision_margin: float = 0.2,
        deviation_threshold: float = 0.35,
    ) -> None:
        self.planner = planner or self._fallback_planner
        self.smoother = smoother or PSOPathSmoother(verbose=False)
        self.collision_margin = collision_margin
        self.deviation_threshold = deviation_threshold
        self.latest_prediction: List[ObstacleState] = []

    # ------------------------------------------------------------------
    # Interfaces
    # ------------------------------------------------------------------
    def update_predictions(self, obstacles: Sequence[ObstacleState]) -> None:
        self.latest_prediction = [obs.copy() for obs in obstacles]

    def evaluate_and_replan(
        self,
        current_pose: Sequence[float],
        reference_path: np.ndarray,
        goal: Sequence[float],
    ) -> ReplanDecision:
        start_time = time.time()
        current = np.asarray(current_pose, dtype=float)

        deviation = np.linalg.norm(current - reference_path[0])
        if deviation > self.deviation_threshold:
            reason = f"deviation {deviation:.3f}m exceeds threshold {self.deviation_threshold}m"
            return self._replan(reason, current, np.asarray(goal, dtype=float), start_time)

        if self._predict_collision(reference_path):
            return self._replan("predicted collision", current, np.asarray(goal, dtype=float), start_time)

        return ReplanDecision(triggered=False, reason="stable")

    # ------------------------------------------------------------------
    # Planning pipeline
    # ------------------------------------------------------------------
    def _replan(self, reason: str, start: np.ndarray, goal: np.ndarray, start_time: float) -> ReplanDecision:
        raw_path = self.planner(start, goal, self.latest_prediction)
        smooth_path, _, _ = self.smoother.smooth(raw_path, obstacles=self._to_spheres())
        latency_ms = (time.time() - start_time) * 1000.0
        return ReplanDecision(
            triggered=True,
            reason=reason,
            path=raw_path,
            smooth_path=smooth_path,
            latency_ms=latency_ms,
        )

    def _predict_collision(self, path: np.ndarray) -> bool:
        for obs in self.latest_prediction:
            for waypoint in path:
                if np.linalg.norm(waypoint - obs.position) < obs.radius + self.collision_margin:
                    return True
        return False

    def _fallback_planner(
        self, start: np.ndarray, goal: np.ndarray, obstacles: Sequence[ObstacleState]
    ) -> np.ndarray:
        """Straight-line fallback that mimics APF-RRT contract."""

        _ = obstacles
        samples = max(5, int(np.linalg.norm(goal - start) * 5))
        return np.linspace(start, goal, samples)

    def _to_spheres(self) -> List[Tuple[np.ndarray, float]]:
        return [(obs.position, obs.radius + self.collision_margin) for obs in self.latest_prediction]

    # ------------------------------------------------------------------
    # ROS helpers
    # ------------------------------------------------------------------
    def publish_path_to_rviz(self, path: np.ndarray, frame_id: str = "map") -> None:
        """Publish a path to RViz without requiring callers to manage ROS boilerplate."""

        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped
        import rospy

        if not rospy.core.is_initialized():
            rospy.init_node("realtime_replanner", anonymous=True)

        pub = rospy.Publisher("/replanner/path", Path, queue_size=1)
        msg = Path()
        msg.header.frame_id = frame_id
        msg.header.stamp = rospy.Time.now()

        for waypoint in path:
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.pose.position.x = float(waypoint[0])
            pose.pose.position.y = float(waypoint[1])
            pose.pose.position.z = float(waypoint[2]) if len(waypoint) > 2 else 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        pub.publish(msg)

    def sync_planning_scene(self, obstacles: Optional[Sequence[ObstacleState]] = None) -> None:
        """Push obstacle geometries into the MoveIt planning scene."""

        obstacles = list(obstacles or self.latest_prediction)
        import rospy
        from moveit_commander import PlanningSceneInterface
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose

        if not rospy.core.is_initialized():
            rospy.init_node("replanner_scene_sync", anonymous=True)

        scene = PlanningSceneInterface()
        for obs in obstacles:
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.SPHERE
            primitive.dimensions = [float(obs.radius)]

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = [float(v) for v in obs.position]
            pose.orientation.w = 1.0
            scene.attach_object(link="base_link", object_pose=pose, shape=primitive, touch_links=[])


__all__ = ["RealTimeReplanner", "ReplanDecision"]
