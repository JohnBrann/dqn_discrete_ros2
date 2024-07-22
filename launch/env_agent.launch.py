from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('dqn_discrete_ros2'),
        'config',
        'params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='dqn_discrete_ros2',
            executable='ros_gym_environment_helper',
            name='ros_gym_env_helper',
            output='screen',
            parameters=[config]),
        Node(
            package='dqn_discrete_ros2',
            executable='agent',
            name='agent',
            output='screen',
            parameters=[config]),
    ])