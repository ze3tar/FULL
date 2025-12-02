from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_setup_assistant_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("rm65", package_name="rm65_moveit_config").to_moveit_configs()
    return generate_setup_assistant_launch(moveit_config)
