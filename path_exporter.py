"""Utilities for exporting planner paths and ROS interoperability."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np


def _normalise_point(point: Sequence[float]) -> Tuple[float, ...]:
    arr = np.asarray(point, dtype=np.float64)
    return tuple(float(x) for x in arr)


def export_path(path_points: Iterable[Sequence[float]], metadata: Dict[str, Any] | None = None) -> Dict[str, Path]:
    """Persist the planned path alongside run metadata.

    Args:
        path_points: Iterable of coordinates along the planned path.
        metadata: Optional metadata including iterations, nodes, planning_time, and parameters.

    Returns:
        Dictionary containing the CSV and JSON file paths.
    """

    base_dir = Path("outputs/paths")
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = base_dir / f"path_{timestamp}.csv"
    json_path = base_dir / f"path_{timestamp}.json"

    normalised_points = [_normalise_point(point) for point in path_points]
    dims = len(normalised_points[0]) if normalised_points else 2
    headers = ["x", "y", "z"][:dims]

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for point in normalised_points:
            writer.writerow(list(point[:dims]))

    meta = metadata or {}
    meta_payload = {
        "iterations": meta.get("iterations"),
        "nodes": meta.get("nodes"),
        "planning_time": meta.get("planning_time"),
        "parameters": meta.get("parameters"),
    }
    json_path.write_text(json.dumps(meta_payload, indent=2))

    return {"csv": csv_path, "metadata": json_path}


def convert_to_ros_path(path_points: Iterable[Sequence[float]]):
    """Create a ``nav_msgs/Path`` from a list of XY points."""

    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    import rospy

    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = "map"

    poses = []
    for point in path_points:
        x, y, *rest = _normalise_point(point)
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = rest[0] if rest else 0.0
        pose.pose.orientation.w = 1.0
        poses.append(pose)

    path_msg.poses = poses
    return path_msg
