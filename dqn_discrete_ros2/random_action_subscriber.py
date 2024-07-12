import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from model_msgs.msg import CartpoleState
import numpy as np


class RandomActionSubscriber(Node):

    def __init__(self):
        super().__init__('random_action_subscriber')
        self.subscription = self.create_subscription(
            CartpoleState,
            'random_action',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.action = None

    def listener_callback(self, msg):
        self.action = msg.force_direction
        self.get_logger().info(f'Recieved Random Action: {self.action}')

    # Returns information (state, reward, terminated, truncated) of a given step
    def step_data(self):
        return self.action


def main(args=None):
    rclpy.init(args=args)

    random_action_subscriber = RandomActionSubscriber()

    rclpy.spin(random_action_subscriber)

    random_action_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
