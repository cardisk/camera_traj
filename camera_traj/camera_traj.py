import rclpy
import math
from rclpy.node import Node

class CameraTraj(Node):
    def __init__(self):
        super().__init__('CameraTraj')

        self.declare_parameters("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameters("yolo_topic",  "/cone_detection/output")

        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.yolo_topic  = self.get_parameter("yolo_topic").get_parameter_value().string_value

        # self.subscription = self.create_subscription(
        #     PoseStamped,
        #     '/zed/zed_node/pose',
        #     self.zed_slam_callback,
        #     10)

        # self.publisher = self.create_publisher(
        #     PoseStampedFRT,
        #     '/orb_slam3/camera_pose',
        #     10)

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
