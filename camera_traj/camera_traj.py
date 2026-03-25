import math
from dataclasses import dataclass

import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from message_filters import ApproximateTimeSynchronizer, Subscriber

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
from driverless_msgs.msg import BoundingBoxes

from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener


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


class RollingMap:
    def __init__(self, rmsth, efa, rmcb, rmcm):
        self.safety_threshold: float = rmsth
        self.cone_map: list[MapPoint] = []
        self.ema_filter_alpha = efa
        self.cull_behind_th = rmcb
        self.cull_max_th = rmcm

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

    def cull_map(self, transform_to_car, world_frame):
        """
        Remove the cones too far behind the car

        transform_to_car: world_frame to camera_frame
        """
        kept_cones = []

        for cone in self.cone_map:
            # Crafting a ROS2 message to the transformation
            p_odom = PointStamped()
            # Here there is no need to use the complete header to do the transformation
            p_odom.header.frame_id = world_frame
            p_odom.point.x = cone.x
            p_odom.point.y = cone.y
            p_odom.point.z = 0.0

            p_car = do_transform_point(p_odom, transform_to_car)

            # Culling the cones
            total_distance = math.hypot(p_car.point.x, p_car.point.y)

            if p_car.point.x > self.cull_behind_th and total_distance < self.cull_max_th:
                kept_cones.append(cone)

        self.cone_map = kept_cones


class CameraTraj(Node):
    """
    ROS2 node that calculates the trajectory from the
    camera frames received
    """

    def __init__(self):
        super().__init__("CameraTraj")

        # Class fields
        self.bridge = CvBridge()
        self.camera_info = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ROS2 launch parameters
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("yolo_topic", "/cone_detection/output")
        self.declare_parameter("camera_info_topic", "/zed/zed_node/depth/camera_info")
        self.declare_parameter("rolling_map_safety_threshold", 0.5)
        self.declare_parameter("rolling_map_ema_filter_alpha", 0.6)
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("car_frame", "zed_camera_link")
        self.declare_parameter("cull_distance_behind", -2.0)
        self.declare_parameter("cull_distance_max", 10.0)

        self.rmsth = self.get_parameter("rolling_map_safety_threshold").double_value
        self.rmefa = self.get_parameter("rolling_map_ema_filter_alpha").double_value
        self.rmcb = self.get_parameter("cull_distance_behind").double_value
        self.rmcm = self.get_parameter("cull_distance_max").double_value

        self.depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )

        self.yolo_topic = (
            self.get_parameter("yolo_topic").get_parameter_value().string_value
        )

        self.camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )

        self.car_frame = self.get_parameter("car_frame").get_parameter_value().string_value
        self.world_frame = (
            self.get_parameter("world_frame").get_parameter_value().string_value
        )

        # RollingMap initialization
        self.cone_map = RollingMap(
            self.rmsth,
            self.rmefa,
            self.rmcb,
            self.rmcm
        )

        # Camera intrinsics subscriber to take camera information
        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.info_callback, 10
        )

        # Synchronization of YOLO and Depth messages
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

        # Transforming the ROS2 message into usable data
        try:
            depth_array = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
            return

        # Camera frame shape
        height, width = depth_array.shape

        for det in yolo_msg:
            object_class = det["color"]

            xmin, ymin = det["BB"][0]
            xmax, ymax = det["BB"][1]

            # Clamping the BB if at the image borders
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(width, int(xmax))
            ymax = min(height, int(ymax))

            if xmin >= xmax or ymin >= ymax:
                self.get_logger().warn(f"Invalid bounding box for {object_class}")
                continue

            # BB size
            w = xmax - xmin
            h = ymax - ymin

            if w < 5 or h < 5:
                self.get_logger().warn(f"Bounding box too small for {object_class}")
                continue

            # BB resizing
            patch_xmin = int(xmin + (w * 0.35))
            patch_xmax = int(xmax - (w * 0.35))
            patch_ymin = int(ymin + (h * 0.80))
            patch_ymax = ymax

            patch_xmin = max(0, min(patch_xmin, width - 1))
            patch_xmax = max(patch_xmin + 1, min(patch_xmax, width))
            patch_ymin = max(0, min(patch_ymin, height - 1))
            patch_ymax = max(patch_ymin + 1, min(patch_ymax, height))

            # Using the BB as a mask over the depth
            depth_roi = depth_array[patch_ymin:patch_ymax, patch_xmin:patch_xmax]
            valid_depths = depth_roi[
                ~np.isnan(depth_roi) & ~np.isinf(depth_roi) & (depth_roi > 0.0)
            ]

            if valid_depths.size > 0:
                # Using the percentiles to extract depth information
                # 25% is more ore less the correct value estimation
                distance = float(np.percentile(valid_depths, 25))

                # Bottom center point inside the resiczed BB
                u_pixel = int((xmin + xmax) / 2)
                v_pixel = int(ymax)

                # Camera intrinsics
                fx = self.camera_info.k[0]
                fy = self.camera_info.k[4]
                cx = self.camera_info.k[2]
                cy = self.camera_info.k[5]

                # 3D point inside optical frame
                x_opt = (u_pixel - cx) * distance / fx
                y_opt = (v_pixel - cy) * distance / fy
                z_opt = distance

                # Crafting the PointStamped to do the transformation
                point_cam = PointStamped()

                # IMPORTANT: frame_id and timestamp must be the same as the depth image
                point_cam.header.frame_id = depth_msg.header.frame_id
                point_cam.header.stamp = depth_msg.header.stamp
                point_cam.point.x = x_opt
                point_cam.point.y = y_opt
                point_cam.point.z = z_opt

                try:
                    # Kindly asking to the TF2 package to transform the point
                    # - lookup_transform(target_frame, source_frame, timeout)
                    transform = self.tf_buffer.lookup_transform(
                        self.world_frame,
                        point_cam.header.frame_id,
                        point_cam.header.stamp,
                        rclpy.duration.Duration(seconds=0.1)
                    )

                    # Applying the transformation to the point
                    point_world = do_transform_point(point_cam, transform)

                    # Adding the new point inside the map
                    new_map_point = MapPoint(
                        x=point_world.point.x, y=point_world.point.y, color=object_class
                    )

                    self.cone_map.add_to_map(new_map_point)

                    self.get_logger().info(
                        f"Mapping {object_class} in {self.world_frame} -> X: {new_map_point.x:.2f}, Y: {new_map_point.y:.2f}"
                    )

                except Exception as e:
                    self.get_logger().warn(
                        f"Could not transform in {self.world_frame}: {e}"
                    )

            else:
                self.get_logger().info(
                    f"Detected {object_class}, but depth is invalid."
                )

        try:
            # lookup_transform(target_frame, source_frame, timeout)
            # rclpy.time.Time() to get the most recent transformation
            transform_to_car = self.tf_buffer.lookup_transform(
                self.car_frame,
                self.world_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )

            # Cleaning
            self.cone_map.cull_map(transform_to_car, self.world_frame)

            self.get_logger().info(f"Map updated, active cones: {len(self.cone_map.cone_map)}")

        except Exception as e:
            self.get_logger().warn(f"Culling failed. Could not get TF from {self.world_frame} to {self.car_frame}: {e}")


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
