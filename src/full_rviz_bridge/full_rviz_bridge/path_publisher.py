#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from moveit_msgs.msg import DisplayTrajectory
from trajectory_msgs.msg import JointTrajectory


class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.apf_path_pub = self.create_publisher(Path, '/rm65/apf_rrt_path', 10)
        self.ppo_path_pub = self.create_publisher(Path, '/rm65/ppo_path', 10)
        self.apf_marker_pub = self.create_publisher(Marker, '/rm65/apf_rrt_path_marker', 10)
        self.ppo_marker_pub = self.create_publisher(Marker, '/rm65/ppo_path_marker', 10)

        self.create_subscription(DisplayTrajectory, '/move_group/display_planned_path', self.handle_moveit_path, 10)
        self.create_subscription(JointTrajectory, '/ppo/refined_trajectory', self.handle_ppo_path, 10)

    @staticmethod
    def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return Rz @ Ry @ Rx

    @classmethod
    def _make_transform(cls, translation, rpy):
        T = np.eye(4)
        T[:3, :3] = cls._rpy_matrix(rpy[0], rpy[1], rpy[2])
        T[:3, 3] = translation
        return T

    @classmethod
    def _rot_z(cls, angle: float):
        return cls._make_transform([0.0, 0.0, 0.0], [0.0, 0.0, angle])

    @classmethod
    def _forward_transform(cls, q: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T = T @ cls._make_transform([0.0, 0.0, 0.2405], [0.0, 0.0, 0.0])
        T = T @ cls._rot_z(q[0])
        T = T @ cls._make_transform([0.0, 0.0, 0.0], [1.5708, -1.5708, 0.0])
        T = T @ cls._rot_z(q[1])
        T = T @ cls._make_transform([0.256, 0.0, 0.0], [0.0, 0.0, 1.5708])
        T = T @ cls._rot_z(q[2])
        T = T @ cls._make_transform([0.0, -0.21, 0.0], [1.5708, 0.0, 0.0])
        T = T @ cls._rot_z(q[3])
        T = T @ cls._make_transform([0.0, 0.0, 0.0], [-1.5708, 0.0, 0.0])
        T = T @ cls._rot_z(q[4])
        T = T @ cls._make_transform([0.0, -0.1725, 0.0], [1.5708, 0.0, 0.0])
        T = T @ cls._rot_z(q[5])
        return T[:3, 3]

    def _compute_ee_position(self, positions: list[float]) -> np.ndarray:
        if len(positions) < 6:
            return np.zeros(3)
        q = np.array(positions[:6], dtype=float)
        return self._forward_transform(q)

    def _joint_traj_to_path(self, traj: JointTrajectory) -> Path:
        path = Path()
        path.header.frame_id = 'base_link'
        path.header.stamp = self.get_clock().now().to_msg()
        for pt in traj.points:
            pose = PoseStamped()
            pose.header = path.header
            pos = self._compute_ee_position(pt.positions)
            pose.pose.position.x = float(pos[0])
            pose.pose.position.y = float(pos[1])
            pose.pose.position.z = float(pos[2])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def _path_to_marker(self, path: Path, frame_id: str, r: float, g: float, b: float) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.points = [Point(x=p.pose.position.x, y=p.pose.position.y, z=p.pose.position.z) for p in path.poses]
        return marker

    def handle_moveit_path(self, msg: DisplayTrajectory):
        if not msg.trajectory:
            return
        traj = msg.trajectory[0].joint_trajectory
        path = self._joint_traj_to_path(traj)
        self.apf_path_pub.publish(path)
        marker = self._path_to_marker(path, path.header.frame_id, 0.0, 0.5, 1.0)
        self.apf_marker_pub.publish(marker)
        self.get_logger().info('Published APF-RRT path from MoveIt')

    def handle_ppo_path(self, traj: JointTrajectory):
        path = self._joint_traj_to_path(traj)
        self.ppo_path_pub.publish(path)
        marker = self._path_to_marker(path, path.header.frame_id, 1.0, 0.3, 0.0)
        self.ppo_marker_pub.publish(marker)
        self.get_logger().info('Published PPO refined path')


def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
