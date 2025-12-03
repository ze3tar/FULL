from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rm65_planner',
            executable='rm65_plan_path_server',
            name='rm65_plan_path_server',
            output='screen'
        )
    ])
