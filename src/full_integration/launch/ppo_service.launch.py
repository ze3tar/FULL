from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='full_integration',
            executable='ppo_service_node',
            name='ppo_service_node',
            output='screen',
            parameters=[{'model_path': '', 'use_model': True, 'dt': 0.1}],
        )
    ])
