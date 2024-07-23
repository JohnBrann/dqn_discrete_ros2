import gymnasium as gym
import rclpy
from rclpy.node import Node
import yaml
import argparse
from rclpy.parameter import Parameter

from model_msgs.srv import EnvReset, EnvSetup, EnvStep

class RosGymEnvHelper(Node):
    def __init__(self):
        super().__init__('ros_gym_env_helper')

         # Declare parameters
        self.declare_parameter('env_id', 'CartPole-v1')
        self.declare_parameter('is_training', False)
        # self.declare_parameter('parameters.ros_gym_env_helper.env_make_params', 'default_value')

        # Get parameter values
        self.env_id = self.get_parameter('env_id').get_parameter_value().string_value
        self.is_training = self.get_parameter('is_training').get_parameter_value().bool_value
        # self.env_make_params = self.get_parameter('parameters.ros_gym_env_helper.env_make_params').get_parameter_value().string_value

        # Initialize the gym environment
        self.env = gym.make(self.env_id, render_mode=None if self.is_training else 'human')#, **self.env_make_params)
        self.env.reset()

        # Create services
        self.env_setup_server = self.create_service(EnvSetup, 'env_setup', self.setup_callback)
        self.env_reset_server = self.create_service(EnvReset, 'env_reset', self.reset_callback)
        self.env_step_server = self.create_service(EnvStep, 'env_step', self.step_callback)

    def setup_callback(self, request, response):
        response.state_dim = self.env.observation_space.shape[0]
        response.action_dim = self.env.action_space.n.item()
        self.get_logger().info(f'Sending environment information to agent: state_dim={response.state_dim}, action_dim={response.action_dim}')
        return response

    def reset_callback(self, request, response):
        #self.get_logger().info(f'Received reset request...')
        obs, _ = self.env.reset()
        response.state = [float(x) for x in obs]
        # response.cart_pos = obs[0].item()
        # response.cart_velocity = obs[1].item()
        # response.pole_angle = obs[2].item()
        # response.pole_angular_velocity = obs[3].item()
        return response

    def step_callback(self, request, response):
        #self.get_logger().info(f'Received step request...')
        obs, reward, terminated, truncated, info = self.env.step(request.action)
        response.state = [float(x) for x in obs]
        # response.cart_pos = obs[0].item()
        # response.cart_velocity = obs[1].item()
        # response.pole_angle = obs[2].item()
        # response.pole_angular_velocity = obs[3].item()
        response.reward = reward
        response.terminated = terminated
        response.truncated = truncated
        #self.get_logger().info(f'Step message = {response}')
        return response

def main(args=None):
    rclpy.init(args=args)
    ros_gym_env_helper = RosGymEnvHelper()

    rclpy.spin(ros_gym_env_helper)

    ros_gym_env_helper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()