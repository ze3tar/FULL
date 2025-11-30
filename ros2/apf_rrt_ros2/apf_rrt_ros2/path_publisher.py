from pathlib import Path
from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node


class APFRRTPathPublisher(Node):
    """Publish APF-RRT CSV waypoints to ROS 2 topics.

    The node loads a CSV path exported by the planners (three columns: ``x,y,z``
    in millimetres) and republishes the path as both ``nav_msgs/Path`` and
    ``geometry_msgs/PoseArray`` for easy RViz and MoveIt 2 consumption.
    """

    def __init__(self) -> None:
        super().__init__("apf_rrt_path_publisher")

        self.declare_parameter("path_file", "path_points_improved.csv")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate", 1.0)
        self.declare_parameter("publish_pose_array", True)

        self.frame_id: str = self.get_parameter("frame_id").get_parameter_value().string_value
        path_file_param = self.get_parameter("path_file").get_parameter_value().string_value
        publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value
        self.publish_pose_array = (
            self.get_parameter("publish_pose_array").get_parameter_value().bool_value
        )

        self.waypoints = self._load_waypoints(Path(path_file_param))
        if not self.waypoints:
            raise ValueError("No waypoints were loaded from the specified CSV file.")

        self.path_pub = self.create_publisher(PathMsg, "apf_rrt/path", 10)
        self.pose_array_pub = None
        if self.publish_pose_array:
            self.pose_array_pub = self.create_publisher(PoseArray, "apf_rrt/pose_array", 10)

        if publish_rate > 0:
            period = 1.0 / publish_rate
            self.timer = self.create_timer(period, self.publish_messages)
        else:
            self.publish_messages()
            self.timer = None

    def _load_waypoints(self, csv_path: Path) -> List[Tuple[float, float, float]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        waypoints: List[Tuple[float, float, float]] = []
        for row in data:
            x = float(row[0]) * 0.001
            y = float(row[1]) * 0.001 if row.size > 1 else 0.0
            z = float(row[2]) * 0.001 if row.size > 2 else 0.0
            waypoints.append((x, y, z))

        return waypoints

    def publish_messages(self) -> None:
        now = self.get_clock().now().to_msg()

        path_msg = PathMsg()
        path_msg.header.stamp = now
        path_msg.header.frame_id = self.frame_id

        pose_array = PoseArray()
        pose_array.header.stamp = now
        pose_array.header.frame_id = self.frame_id

        for waypoint in self.waypoints:
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            pose.pose.position.z = waypoint[2]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

            if self.publish_pose_array:
                pose_array.poses.append(pose.pose)

        self.path_pub.publish(path_msg)
        if self.pose_array_pub and self.publish_pose_array:
            self.pose_array_pub.publish(pose_array)

        self.get_logger().info(
            "Published %d poses to apf_rrt/path%s",
            len(path_msg.poses),
            " and apf_rrt/pose_array" if self.publish_pose_array else "",
        )

    def destroy_node(self) -> bool:
        if self.timer is not None:
            self.timer.cancel()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = APFRRTPathPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
