from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("rm65", package_name="rm65_moveit_config").to_moveit_configs()
    moveit_config.planning_pipelines = moveit_config.load_yaml(
        "rm65_moveit_config", "config/moveit_planners.yaml"
    )
    return generate_move_group_launch(moveit_config)
