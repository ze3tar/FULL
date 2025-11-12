from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('full_visualizer')

    urdf_path = os.path.join(package_share, 'urdf', 'full_robot.urdf')
    rviz_config_path = os.path.join(package_share, 'rviz', 'full.rviz')

    with open(urdf_path, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    joint_state_pub = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
    )

    codex_node = Node(
        package='full_visualizer',
        executable='chatgpt_codex_node',
        name='chatgpt_codex_node',
        output='screen',
    )

    return LaunchDescription([
        joint_state_pub,
        robot_state_pub,
        rviz,
        codex_node,
    ])
