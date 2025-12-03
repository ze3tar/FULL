from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("rm65", package_name="rm65_moveit_config").to_moveit_configs()
    moveit_config.planning_pipelines = moveit_config.load_yaml(
        "rm65_moveit_config", "config/moveit_planners.yaml"
    )

    controllers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [get_package_share_directory('rm65_control'), 'launch', 'rm65_controllers.launch.py']
            )
        )
    )

    demo_launch = generate_demo_launch(moveit_config)
    return LaunchDescription([controllers_launch, demo_launch])
