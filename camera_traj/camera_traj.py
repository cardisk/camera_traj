import math
from dataclasses import dataclass

import numpy as np
import rclpy
from cv_bridge import CvBridge
from driverless_msgs.msg import BoundingBoxes
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class MapPoint:
    """
    x axis positive at front
    y axis positive at right
    color of the cone
    hit_count is the number of times the cone is seen

    distance from the ground is ignored
    """

    x: float
    y: float
    color: str
    hit_count: int = 1

    @classmethod
    def from_camera_point(
        cls, u: int, v: int, distance: float, camera_info, color: str
    ):
        """
        Optical frame has Z in front, X at right and Y down
        """

        # Camera intrinsics K matrix parameters
        fx = camera_info.k[0]
        cx = camera_info.k[2]

        # Proiezione inversa sull'asse X ottico (destra/sinistra)
        # Inverse projection
        x_opt = (u - cx) * distance / fx
        z_opt = distance

        map_x = z_opt
        map_y = x_opt

        return cls(x=map_x, y=map_y, color=color, hit_count=1)


class RollingMap:
    def __init__(self, rmsth, efa):
        self.safety_threshold: float = rmsth
        self.cone_map: list[MapPoint] = []
        self.ema_filter_alpha = efa

    def add_to_map(self, new_point: MapPoint):
        best_dist = float("inf")
        best_match = None

        for cone in self.cone_map:
            if cone.color == new_point.color:
                dist = math.hypot(cone.x - new_point.x, cone.y - new_point.y)
                if dist < best_dist:
                    best_dist = dist
                    best_match = cone

        if best_match is not None and best_dist <= self.safety_threshold:
            # Exponential Moving Average (EMA) updating
            best_match.x = (self.ema_filter_alpha * best_match.x) + (
                (1.0 - self.ema_filter_alpha) * new_point.x
            )
            best_match.y = (self.ema_filter_alpha * best_match.y) + (
                (1.0 - self.ema_filter_alpha) * new_point.y
            )
            best_match.hit_count += 1
        else:
            self.cone_map.append(new_point)


class CameraTraj(Node):
    def __init__(self):
        super().__init__("CameraTraj")

        self.bridge = CvBridge()
        self.camera_info = None

        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("yolo_topic", "/cone_detection/output")
        self.declare_parameter("camera_info_topic", "/zed/zed_node/depth/camera_info")

        self.declare_parameter("rolling_map_safety_threshold", 0.5)
        self.rmsth = self.get_parameter("rolling_map_safety_threshold").double_value

        self.declare_parameter("rolling_map_ema_filter_alpha", 0.6)
        self.rmsth = self.get_parameter("rolling_map_ema_filter_alpha").double_value

        self.depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        self.yolo_topic = (
            self.get_parameter("yolo_topic").get_parameter_value().string_value
        )
        self.camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )

        self.cone_map = RollingMap(self.rmsth)

        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.info_callback, 10
        )

        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.yolo_sub = Subscriber(self, BoundingBoxes, self.yolo_topic)

        queue_size = 10
        max_difference_in_seconds = 0.1

        self.sync = ApproximateTimeSynchronizer(
            [self.depth_sub, self.yolo_sub], queue_size, max_difference_in_seconds
        )

        self.sync.registerCallback(self.cone_extractor)

        self.get_logger().info("Ready!")

    def info_callback(self, camera_info_msg):
        if self.camera_info is None:
            self.camera_info = camera_info_msg
            self.get_logger().info("Camera Intrinsics received!")

    def cone_extractor(self, depth_msg, yolo_msg):
        if self.camera_info is None:
            return

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
            valid_depths = depth_roi[
                ~np.isnan(depth_roi) & ~np.isinf(depth_roi) & (depth_roi > 0.0)
            ]

            if valid_depths.size > 0:
                distance = float(np.percentile(valid_depths, 25))

                u_pixel = int((xmin + xmax) / 2)
                v_pixel = int(ymax)

                new_map_point = MapPoint.from_camera_point(
                    u=u_pixel,
                    v=v_pixel,
                    distance=distance,
                    camera_info=self.camera_info,
                    color=object_class,
                )

                self.cone_map.add_to_map(new_map_point)
                self.get_logger().info(
                    f"Added {object_class} to Map at X: {new_map_point.x:.2f}, Y: {new_map_point.y:.2f}"
                )
            else:
                self.get_logger().info(
                    f"Detected {object_class}, but depth is invalid."
                )


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


if __name__ == "__main__":
    main()
