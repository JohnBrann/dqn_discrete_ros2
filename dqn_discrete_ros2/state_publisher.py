import gymnasium as gym
import rclpy
from rclpy.node import Node
from model_msgs.msg import CartpoleState


class StatePublisher(Node):

    def __init__(self):
        super().__init__('state_publisher_node')
        self.publisher_ = self.create_publisher(CartpoleState, 'environment_state', 10)

    def publish_state(self, step):
        # Get the current state
        current_state, reward, terminated, truncated, info = step #self.env.step(self.env.action_space.sample())

        # Create CartpoleState message
        msg = CartpoleState()
        msg.cart_pos = current_state[0].item()
        msg.cart_velocity = current_state[1].item()
        msg.pole_angle = current_state[2].item()
        msg.pole_angular_velocity = current_state[3].item()
        msg.reward = reward
        msg.terminated = terminated
        msg.truncated = truncated

        # Publish message
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.cart_pos, msg.cart_velocity, msg.pole_angle, msg.pole_angular_velocity, msg.reward, msg.terminated, msg.truncated}')


def main(args=None):
    rclpy.init(args=args)
    environment_publisher = StatePublisher()

    # Main loop for publishing during training
    try:
        while rclpy.ok():
            environment_publisher.publish_state()
            #rclpy.spin_once(environment_publisher)  # Handle callbacks
    except KeyboardInterrupt:
        pass
    finally:
        environment_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()