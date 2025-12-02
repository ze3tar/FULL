#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import csv
import os

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher = self.create_publisher(Path, 'ppo_path', 10)
        self.timer = self.create_timer(1.0, self.publish_path)
        self.csv_path = os.path.expanduser("~/FULL/path_points_improved.csv")

    def publish_path(self):
        path = Path()
        path.header.frame_id = "world"

        if not os.path.exists(self.csv_path):
            self.get_logger().warn(f"Path file not found: {self.csv_path}")
            return

        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # skip x,y,z

                for row in reader:
                    x, y, z = map(float, row)
                    pose = PoseStamped()
                    pose.header.frame_id = "world"
                    pose.pose.position.x = x / 1000.0
                    pose.pose.position.y = y / 1000.0
                    pose.pose.position.z = z / 1000.0
                    path.poses.append(pose)

        except Exception as e:
            self.get_logger().error(f"Error reading csv: {e}")
            return

        self.publisher.publish(path)
        self.get_logger().info("Published PPO path to RViz2")

def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
