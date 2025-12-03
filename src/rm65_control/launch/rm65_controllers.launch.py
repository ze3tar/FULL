from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command


def generate_launch_description():
    robot_description_arg = DeclareLaunchArgument(
        'robot_description_file',
        default_value=PathJoinSubstitution([
            get_package_share_directory('rm65_description'),
            'urdf',
            'rm_65_6f_description.urdf',
        ]),
        description='Absolute path to the RM65 URDF file.'
    )

    robot_description = {"robot_description": Command(['cat', LaunchConfiguration('robot_description_file')])}

    controllers_yaml = PathJoinSubstitution([
        get_package_share_directory('rm65_control'),
        'ros2_controllers.yaml',
    ])

    return LaunchDescription([
        robot_description_arg,
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[robot_description, controllers_yaml],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['rm65_controller'],
            output='screen',
        ),
    ])
