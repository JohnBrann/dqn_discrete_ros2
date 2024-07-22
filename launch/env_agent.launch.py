from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dqn_discrete_ros2',
            executable='ros_gym_environment_helper',
            output='screen'),
        Node(
            package='dqn_discrete_ros2',
            executable='agent',
            output='screen'),
    ])
