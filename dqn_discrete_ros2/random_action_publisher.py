import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from model_msgs.msg import CartpoleAction
import numpy as np


class RandomActionPublisher(Node):
    def __init__(self):
        super().__init__('random_action_publisher')
        self.publisher_ = self.create_publisher(CartpoleAction, '/random_action', 10)

    def publish_action(self, action):
        msg = CartpoleAction()
        msg.force_direction = action


        # Publish message
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing Random Action: {msg}')


def main(args=None):
    rclpy.init(args=args)
    random_action_publisher = RandomActionPublisher()

    # Main loop for publishing actions
    try:
        while rclpy.ok():
            random_action_publisher.publish_state()
    except KeyboardInterrupt:
        pass
    finally:
        random_action_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()