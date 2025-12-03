from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('rm65_moveit_config'),
                'launch',
                'demo.launch.py',
            ])
        )
    )

    ppo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('full_integration'),
                'launch',
                'ppo_service.launch.py',
            ])
        )
    )

    rviz_bridge = Node(
        package='full_rviz_bridge',
        executable='path_publisher',
        output='screen',
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
    )

    return LaunchDescription([
        static_tf,
        moveit_launch,
        ppo_launch,
        rviz_bridge,
    ])
