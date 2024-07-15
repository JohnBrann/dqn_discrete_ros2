import gymnasium as gym
import rclpy
from rclpy.node import Node
from state_publisher import StatePublisher
from random_action_publisher import RandomActionPublisher
from action_subscriber import ActionSubscriber

class RosGymEnvHelper(Node):
    def __init__(self):
        # also contains the only instance of the environment 
        self.env = gym.make("CartPole-v1", render_mode="human")
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





def main(args=None):
    rclpy.init(args=args)
    ros_gym_env_helper = RosGymEnvHelper()

    rclpy.spin(ros_gym_env_helper)

    ros_gym_env_helper.destroy_node()

    rclpy.shutdown()
    
if __name__ == '__main__':
    main()