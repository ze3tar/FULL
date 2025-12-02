from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Absolute path to your URDF
    urdf_path = "/root/ros2_ws/src/rm65_description/urdf/rm_65_6f_description.urdf"

    robot_description = ParameterValue(
        open(urdf_path).read(),
        value_type=str,
    )

    return LaunchDescription([

        # Publishes /joint_states so robot_state_publisher can create TF
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
            output="screen",
        ),

        # Standard name: /robot_state_publisher
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),

        # Your PPO path publisher
        Node(
            package="full_rviz_bridge",
            executable="path_publisher",
            name="path_publisher",
            output="screen",
        ),

        # RViz2
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
        ),
    ])
