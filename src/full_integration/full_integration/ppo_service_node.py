#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

from full_integration.srv import RefineTrajectory
from full_integration.trajectory_refiner import TrajectoryRefiner


class PPOServiceNode(Node):
    def __init__(self):
        super().__init__('ppo_service_node')
        self.declare_parameter('model_path', '')
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('use_model', True)

        self.model = self._load_model(self.get_parameter('model_path').get_parameter_value().string_value,
                                      self.get_parameter('use_model').get_parameter_value().bool_value)
        dt = self.get_parameter('dt').get_parameter_value().double_value
        self.refiner = TrajectoryRefiner(dt=dt)

        self.service = self.create_service(RefineTrajectory, '/refine_trajectory', self.handle_request)
        self.get_logger().info('PPO trajectory refinement service ready on /refine_trajectory')

    def _load_model(self, path: str, use_model: bool) -> Optional[object]:
        if not use_model:
            return None
        if not path:
            self.get_logger().warn('No PPO model path provided; running without neural refinement')
            return None
        if not os.path.exists(path):
            self.get_logger().warn(f'Model path {path} does not exist; running without neural refinement')
            return None
        try:
            import torch

            self.get_logger().info(f'Loading PPO model from {path}')
            return torch.jit.load(path)
        except Exception as exc:
            self.get_logger().error(f'Failed to load PPO model: {exc}')
            return None

    def handle_request(self, request: RefineTrajectory.Request, response: RefineTrajectory.Response):
        try:
            refined = self.refiner.refine(request.input, self.model)
            response.output = refined
            response.success = len(refined.points) > 0
            response.message = 'Refined' if response.success else 'Empty trajectory'
        except Exception as exc:
            self.get_logger().error(f'Refinement failed: {exc}')
            response.success = False
            response.message = f'Error: {exc}'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PPOServiceNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
