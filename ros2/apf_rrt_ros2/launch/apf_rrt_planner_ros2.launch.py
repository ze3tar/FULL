from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    path_arg = DeclareLaunchArgument(
        "path_file",
        default_value="path_points_improved.csv",
        description="CSV file containing APF-RRT waypoints (x,y,z in millimetres)",
    )

    frame_id_arg = DeclareLaunchArgument(
        "frame_id",
        default_value="map",
        description="Frame ID to stamp on published poses",
    )

    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate",
        default_value="1.0",
        description="Rate (Hz) to republish the path. Set to 0 to publish once.",
    )

    publish_pose_array_arg = DeclareLaunchArgument(
        "publish_pose_array",
        default_value="true",
        description="Whether to also publish a PoseArray alongside nav_msgs/Path.",
    )

    path_publisher_node = Node(
        package="apf_rrt_ros2",
        executable="apf_rrt_path_publisher",
        name="apf_rrt_path_publisher",
        parameters=[
            {
                "path_file": LaunchConfiguration("path_file"),
                "frame_id": LaunchConfiguration("frame_id"),
                "publish_rate": LaunchConfiguration("publish_rate"),
                "publish_pose_array": LaunchConfiguration("publish_pose_array"),
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            path_arg,
            frame_id_arg,
            publish_rate_arg,
            publish_pose_array_arg,
            path_publisher_node,
        ]
    )
