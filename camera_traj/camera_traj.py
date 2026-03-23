import rclpy
import math
from rclpy.node import Node

from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import Image
from driverless_msgs.msg import BoundingBoxes

import numpy as np
from cv_bridge import CvBridge

from dataclasses import dataclass

@dataclass
class MapPoint:
    """
    x axis positive at front
    y axis positive at right
    color of the cone
    """
    x: int
    y: int
    color: str

    def from_camera_point(point_3d):
        pass

class RollingMap:
    def __init__(self, rmsth):
        self.safety_threshold: float = rmsth
        self.cone_map: list[MapPoint] = []

    def add_to_map(self, point_3d):
        pass

class CameraTraj(Node):
    def __init__(self):
        super().__init__('CameraTraj')

        self.bridge = CvBridge()

        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("yolo_topic",  "/cone_detection/output")

        self.declare_parameter("rolling_map_safety_threshold",  0.5)
        self.rmsth = self.get_parameter("rolling_map_safety_threshold").double_value

        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.yolo_topic  = self.get_parameter("yolo_topic").get_parameter_value().string_value

        self.cone_map = RollingMap(self.rmsth)

        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.yolo_sub  = Subscriber(self, BoundingBoxes, self.yolo_topic)

        queue_size = 10
        max_difference_in_seconds = 0.1

        self.sync = ApproximateTimeSynchronizer(
            [self.depth_sub, self.yolo_sub],
            queue_size,
            max_difference_in_seconds
        )

        self.sync.registerCallback(self.cone_extractor)

        self.get_logger().info("Ready!")

    def cone_extractor(self, depth_msg, yolo_msg):
        try:
            depth_array = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
            return

        height, width = depth_array.shape

        for det in yolo_msg:
            object_class = det["color"]

            xmin, ymin = det["BB"][0]
            xmax, ymax = det["BB"][1]

            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(width, int(xmax))
            ymax = min(height, int(ymax))

            if xmin >= xmax or ymin >= ymax:
                self.get_logger().warn(f"Invalid bounding box for {object_class}")
                continue

            w = xmax - xmin
            h = ymax - ymin

            if w < 5 or h < 5:
                self.get_logger().warn(f"Bounding box too small for {object_class}")
                continue

            patch_xmin = int(xmin + (w * 0.35))
            patch_xmax = int(xmax - (w * 0.35))
            patch_ymin = int(ymin + (h * 0.80))
            patch_ymax = ymax

            patch_xmin = max(0, min(patch_xmin, width - 1))
            patch_xmax = max(patch_xmin + 1, min(patch_xmax, width))
            patch_ymin = max(0, min(patch_ymin, height - 1))
            patch_ymax = max(patch_ymin + 1, min(patch_ymax, height))

            depth_roi = depth_array[patch_ymin:patch_ymax, patch_xmin:patch_xmax]
            valid_depths = depth_roi[~np.isnan(depth_roi) & ~np.isinf(depth_roi) & (depth_roi > 0.0)]

            if valid_depths.size > 0:
                distance = float(np.percentile(valid_depths, 20))
                self.get_logger().info(f"Detected {object_class} at distance: {distance:.2f}m")
            else:
                self.get_logger().info(f"Detected {object_class}, but depth is invalid.")

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
