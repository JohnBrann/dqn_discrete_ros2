import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from model_msgs.msg import CartpoleState
import numpy as np


class ActionPublisher(Node):
    def __init__(self):
        super().__init__('action_publisher')
        self.publisher_ = self.create_publisher(CartpoleAction, '/actions', 10)

    def publish_action(self):
        dsfsd



def main(args=None):
    rclpy.init(args=args)
    action_publisher = ActionPublisher()

    # Main loop for publishing actions
    try:
        while rclpy.ok():
            action_publisher.publish_state()
    except KeyboardInterrupt:
        pass
    finally:
        action_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()