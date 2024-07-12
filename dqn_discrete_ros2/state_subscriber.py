import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from model_msgs.msg import CartpoleState
import numpy as np


class StateSubscriber(Node):

    def __init__(self):
        super().__init__('state_subscriber')
        self.subscription = self.create_subscription(
            CartpoleState,
            'environment_state',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.state_msg = None

    def listener_callback(self, msg):
        self.state_msg = msg

    # Returns information (state, reward, terminated, truncated) of a given step
    def step_data(self):
        state = np.array([self.state_msg.cart_pos, self.state_msg.cart_velocity, self.state_msg.pole_angle, self.state_msg.pole_angular_velocity])
        # Converts current state information for cartpole into a numpy array to match formatting used by Agent without ROS
        self.step_data = (state, self.state_msg.reward, self.state_msg.terminated, self.state_msg.truncated)
        return self.step_data


def main(args=None):
    rclpy.init(args=args)

    state_subscriber = StateSubscriber()

    rclpy.spin(state_subscriber)

    state_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
