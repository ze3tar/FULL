from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    start_arg = DeclareLaunchArgument('start', default_value='[0,0,0,0,0,0]')
    goal_arg = DeclareLaunchArgument('goal', default_value='[0,0,0,0,0,0]')
    start = LaunchConfiguration('start')
    goal = LaunchConfiguration('goal')

    node = Node(
        package='rm65_planner',
        executable='planner_demo.py',
        name='rm65_planner_demo',
        parameters=[{'start': start, 'goal': goal}]
    )
    return LaunchDescription([start_arg, goal_arg, node])
