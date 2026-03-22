import rclpy
import math
from rclpy.node import Node

class CameraTraj(Node):
    def __init__(self):
        super().__init__('CameraTraj')

        self.get_logger().info("Ready!")

def main(args=None):
    rclpy.init(args=args)
    node = CameraTraj()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
