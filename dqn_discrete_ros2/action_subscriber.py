import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from model_msgs.msg import CartpoleState
import numpy as np


class ActionSubscriber(Node):

    def __init__(self):
        super().__init__('action_subscriber')
        self.subscription = self.create_subscription(
            CartpoleState,
            'actions',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.action = None

    def listener_callback(self, msg):
        self.action = msg.force_direction
        self.get_logger().info(f'Recieved Action: {self.action}')

    # Returns information (state, reward, terminated, truncated) of a given step
    def step_data(self):
        return self.action


def main(args=None):
    rclpy.init(args=args)

    action_subscriber = ActionSubscriber()

    rclpy.spin(action_subscriber)

    action_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
