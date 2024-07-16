import gymnasium as gym
import rclpy
from rclpy.node import Node
from state_publisher import StatePublisher
from random_action_publisher import RandomActionPublisher
from action_subscriber import ActionSubscriber
import yaml
import argparse
from model_msgs.srv import env_reset


class RosGymEnvHelper(Node):
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set
        self.is_training = hyperparameters['is_training']
        self.env_id = hyperparameters['env_id']
        self.env_make_params = hyperparameters.get('env_make_params', {})

        
        # also contains the only instance of the environment 
        self.env = gym.make(self.env_id, render_mode=None if self.is_training else 'human', **self.env_make_params)
        self.env.reset()  # Initialize the environment
        # declare new action publisher
        # declare random action subscriber
        # declare state_subscriber
        self.state_publisher = StatePublisher()
        self.random_action_publisher = RandomActionPublisher()
        self.action_subscriber = ActionSubscriber()
        
    def next_step(self):
        next_step = self.env.step(self.action_subscriber.step_data())
        self.state_publisher.publish_state(next_step)

    def publish_random_action(self):
        self.random_action_publisher.publish_action(self.env.action_space.sample())

    
    def reset_env(self):
        pass

class EnvResetServer(Node):
    def __init__(self, ros_gym_env_helper):
        super.__init__('env_reset_server')
        self.ros_gym_env_helper = ros_gym_env_helper
        self.srv = self.create_service(env_reset, 'env_reset', self.)

        # Reset the environment and get the initial state
    def reset_callback(self, request, response):
        self.ros_gym_env_helper.reset_env()
        response.is_reset = True
        return response
def main(args=None):
    rclpy.init(args=args)
    ros_gym_env_helper = RosGymEnvHelper()

    rclpy.spin(ros_gym_env_helper)

    ros_gym_env_helper.destroy_node()

    rclpy.shutdown()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    args = parser.parse_args()
    env = RosGymEnvHelper(hyperparameter_set=args.hyperparameters)
    main()