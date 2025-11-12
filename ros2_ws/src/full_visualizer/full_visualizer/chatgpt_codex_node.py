import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ChatGPTCodexNode(Node):
    """Dummy node that publishes placeholder commands for future AI control."""

    def __init__(self) -> None:
        super().__init__('chatgpt_codex_node')
        self.publisher_ = self.create_publisher(String, '/chatgpt_codex_cmd', 10)
        self.timer = self.create_timer(3.0, self.timer_callback)
        self.get_logger().info('ChatGPT Codex node started.')

    def timer_callback(self) -> None:
        msg = String()
        msg.data = 'Codex command: Hello Robot!'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ChatGPTCodexNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
