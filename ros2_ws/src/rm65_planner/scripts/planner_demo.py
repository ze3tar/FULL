#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rm65_planner.apf_rrt_planner import APFRRTPlanner, Sphere
import ast


class PlannerDemo(Node):
    def __init__(self):
        super().__init__('planner_demo')
        self.declare_parameter('start', '[0,0,0,0,0,0]')
        self.declare_parameter('goal', '[0,0,0,0,0,0]')
        start = ast.literal_eval(self.get_parameter('start').get_parameter_value().string_value)
        goal = ast.literal_eval(self.get_parameter('goal').get_parameter_value().string_value)
        planner = APFRRTPlanner()
        path = planner.plan(start, goal, [])
        self.get_logger().info(f"Planned path with {len(path)} waypoints")


def main(args=None):
    rclpy.init(args=args)
    node = PlannerDemo()
    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
