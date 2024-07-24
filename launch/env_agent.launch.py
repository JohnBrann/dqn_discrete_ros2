import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare the launch argument for is_training
    is_training_arg = DeclareLaunchArgument(
        'is_training',
        default_value='false',
        description='Flag to enable training mode'
    )

    load_model_arg = DeclareLaunchArgument(
        'load_model',
        default_value=None,
        description='Flag to tell which model to test'

    )

    # Define the YAML configuration file path
    config_file = os.path.join(
        get_package_share_directory('dqn_discrete_ros2'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        is_training_arg,
        Node(
            package='dqn_discrete_ros2',
            executable='ros_gym_environment_helper',
            name='ros_gym_env_helper',
            output='screen',
            parameters=[config_file, {'is_training': LaunchConfiguration('is_training')}]
        ),
        Node(
            package='dqn_discrete_ros2',
            executable='agent',
            name='agent',
            output='screen',
            parameters=[config_file, {'is_training': LaunchConfiguration('is_training')}, {'load_model': LaunchConfiguration('load_model')}]
        )
    ])