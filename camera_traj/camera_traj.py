import math

import scipy.interpolate as spi
from scipy.spatial import Delaunay

import json
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from rclpy.qos import qos_profile_sensor_data

from message_filters import ApproximateTimeSynchronizer, Subscriber

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray
from driverless_msgs.msg import BoundingBoxes, Trajectory

from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener

from .rolling_map import MapPoint, RollingMap


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

        # ROS2 parameters

        # Do not need this because it's a special parameter declared by default
        # self.declare_parameter("use_sim_time", "False")
        self.declare_parameter("pure_local_trajectory", False)
        self.declare_parameter("debug_output", True)

        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("yolo_topic", "/cone_detection/output")
        self.declare_parameter("camera_info_topic", "/zed/zed_node/depth/camera_info")
        self.declare_parameter("rolling_map_debug_topic", "/camera_traj/debug/rolling_map")
        self.declare_parameter("trajectory_debug_topic", "/camera_traj/debug/trajectory")
        self.declare_parameter("output_topic", "/camera_traj/output")
        self.declare_parameter("rolling_map_safety_threshold", 1.5)
        self.declare_parameter("rolling_map_ema_filter_alpha", 0.3)
        self.declare_parameter("rolling_map_hit_count_threshold", 3)
        self.declare_parameter("cull_distance_behind", -2.0)
        self.declare_parameter("cull_distance_max", 15.0)
        self.declare_parameter("delaunay_min_distance", 2.0)
        self.declare_parameter("delaunay_max_distance", 8.0)
        self.declare_parameter("spline_smoothing", 3.0)
        self.declare_parameter("spline_degree", 3)
        self.declare_parameter("spline_sampling_resolution", 0.5)
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("car_frame", "zed_camera_link")

        # Flags
        self.use_sim_time          = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        self.pure_local_trajectory = self.get_parameter("pure_local_trajectory").get_parameter_value().bool_value
        self.debug_output          = self.get_parameter("debug_output").get_parameter_value().bool_value

        # Settings
        self.rmsth  = self.get_parameter("rolling_map_safety_threshold").get_parameter_value().double_value
        self.rmefa  = self.get_parameter("rolling_map_ema_filter_alpha").get_parameter_value().double_value
        self.rmhcth = self.get_parameter("rolling_map_hit_count_threshold").get_parameter_value().integer_value

        self.rmcb   = self.get_parameter("cull_distance_behind").get_parameter_value().double_value
        self.rmcm   = self.get_parameter("cull_distance_max").get_parameter_value().double_value

        self.delaunay_min_distance = self.get_parameter("delaunay_min_distance").get_parameter_value().double_value
        self.delaunay_max_distance = self.get_parameter("delaunay_max_distance").get_parameter_value().double_value

        self.spline_smoothing           = self.get_parameter("spline_smoothing").get_parameter_value().double_value
        self.spline_degree              = self.get_parameter("spline_degree").get_parameter_value().integer_value
        self.spline_sampling_resolution = self.get_parameter("spline_sampling_resolution").get_parameter_value().double_value

        self.car_frame   = self.get_parameter("car_frame").get_parameter_value().string_value
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value

        # Topics
        self.depth_topic             = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.yolo_topic              = self.get_parameter("yolo_topic").get_parameter_value().string_value
        self.camera_info_topic       = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.rolling_map_debug_topic = self.get_parameter("rolling_map_debug_topic").get_parameter_value().string_value
        self.trajectory_debug_topic  = self.get_parameter("trajectory_debug_topic").get_parameter_value().string_value
        self.output_topic            = self.get_parameter("output_topic").get_parameter_value().string_value

        if self.pure_local_trajectory:
            self.rmhcth = 1
            self.rmcb   = 0.0
            self.world_frame = self.car_frame

        self.rolling_map = RollingMap(
            self.rmsth,
            self.rmefa,
            self.rmcb,
            self.rmcm
        )

        self.get_logger().info("- Flags -----------------------------------------------------------------")
        self.get_logger().info(f"  * simulation timestamps: {self.use_sim_time}")
        self.get_logger().info(f"  * pure local trajectory: {self.pure_local_trajectory}")
        self.get_logger().info(f"  * output debug topics: {self.debug_output}")
        self.get_logger().info("-------------------------------------------------------------------------")

        self.get_logger().info("")

        self.get_logger().info("- Settings --------------------------------------------------------------")
        self.get_logger().info(f"  * rolling mapping safety threshold: {self.rmsth}m")
        self.get_logger().info(f"  * rolling mapping EMA alpha: {self.rmefa}")
        self.get_logger().info(f"  * rolling mapping hit count threshold: {self.rmhcth}")
        self.get_logger().info("")
        self.get_logger().info(f"  * rolling mapping cull distance behind car: {self.rmcb}m")
        self.get_logger().info(f"  * rolling mapping cull distance ahead car: {self.rmcm}m")
        self.get_logger().info("")
        self.get_logger().info(f"  * Delaunay minimum distance: {self.delaunay_min_distance}m")
        self.get_logger().info(f"  * Delaunay maximum distance: {self.delaunay_max_distance}m")
        self.get_logger().info(f"  * spline smoothing: {self.spline_smoothing}")
        self.get_logger().info(f"  * spline degree: {self.spline_degree}")
        self.get_logger().info(f"  * spline sampling resolution: {self.spline_sampling_resolution}m")
        self.get_logger().info("")
        self.get_logger().info(f"  * TF car frame: {self.car_frame}")
        self.get_logger().info(f"  * TF world frame: {self.world_frame}")
        self.get_logger().info("-------------------------------------------------------------------------")

        self.get_logger().info("")

        self.get_logger().info("- Topics ----------------------------------------------------------------")
        self.get_logger().info(f"  * depth: {self.depth_topic}")
        self.get_logger().info(f"  * camera_info: {self.camera_info_topic}")
        self.get_logger().info(f"  * yolo: {self.yolo_topic}")
        self.get_logger().info("")
        self.get_logger().info(f"  * output: {self.output_topic}")

        if self.debug_output:
            self.get_logger().info("")
            self.get_logger().info(f"  * rolling map: {self.rolling_map_debug_topic}")
            self.get_logger().info(f"  * trajectory: {self.trajectory_debug_topic}")

        self.get_logger().info("-------------------------------------------------------------------------")

        # Camera intrinsics subscriber to take camera information
        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.info_callback, 10
        )

        # Synchronization of YOLO and Depth messages
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.yolo_sub = Subscriber(self, BoundingBoxes, self.yolo_topic, qos_profile=qos_profile_sensor_data)

        queue_size = 50
        max_difference_in_seconds = 0.2

        self.sync = ApproximateTimeSynchronizer(
            [self.depth_sub, self.yolo_sub], queue_size, max_difference_in_seconds
        )

        self.sync.registerCallback(self.cone_extractor)

        self.output_publisher = self.create_publisher(Trajectory, self.output_topic, 10)

        if self.debug_output:
            self.rolling_map_marker_pub = self.create_publisher(MarkerArray, self.rolling_map_debug_topic, 10)
            self.previous_marker_count = 0

            self.trajectory_marker_pub = self.create_publisher(Marker, self.trajectory_debug_topic, 10)

        if self.use_sim_time:
            self.get_logger().info("")
            self.get_logger().warn("-------------------------------------------------------------------------")
            self.get_logger().warn("Using simulation time from /clock topic")
            self.get_logger().warn("Use this only when running this node with a bag started with --clock flag")
            self.get_logger().warn("-------------------------------------------------------------------------")

        self.get_logger().info("")
        self.get_logger().info("Ready!")

    def info_callback(self, camera_info_msg):
        if self.camera_info is None:
            self.camera_info = camera_info_msg
            self.get_logger().info("")
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

        yolo = json.loads(yolo_msg.json)

        if self.pure_local_trajectory:
            self.rolling_map.cone_map.clear()

        for det in yolo:
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
            patch_xmin = int(xmin + (w * 0.25))
            patch_xmax = int(xmax - (w * 0.25))
            patch_ymin = int(ymin + (h * 0.50))
            patch_ymax = int(ymax - (h * 0.05))

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
                # 50% is the median, it should remove the noise
                distance = float(np.percentile(valid_depths, 50))

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

                # [IMPORTANT] frame_id and timestamp must be the same as the depth image
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
                        rclpy.time.Time(),  # get the latest transformation
                        rclpy.duration.Duration(seconds=0.2)
                    )

                    point_world = do_transform_point(point_cam, transform)

                    new_map_point = MapPoint(
                        x=point_world.point.x, y=point_world.point.y, color=object_class
                    )

                    self.cone_map.add_to_map(new_map_point)

                except Exception as e:
                    self.get_logger().warn(
                        f"Could not transform in {self.world_frame}: {e}"
                    )

            else:
                self.get_logger().info(
                    f"Detected {object_class}, but depth is invalid."
                )

        try:
            # Kindly asking to the TF2 package to transform the point
            # - lookup_transform(target_frame, source_frame, timeout)
            transform_to_car = self.tf_buffer.lookup_transform(
                self.car_frame,
                self.world_frame,
                rclpy.time.Time(),  # get the latest transformation
                rclpy.duration.Duration(seconds=0.2)
            )

            # Cleaning the map
            self.rolling_map.cull_map(transform_to_car, self.world_frame)

            self.get_logger().info("")
            self.get_logger().info(f"Map updated, active cones: {len(self.cone_map.cone_map)}")

            if self.debug_output:
                self.publish_map_markers()

        except Exception as e:
            self.get_logger().warn(f"Culling failed. Could not get TF from {self.world_frame} to {self.car_frame}: {e}")
            return

        midpoints = self.calculate_centerline(self.rmhcth)
        midpoints_len = len(midpoints)

        if midpoints_len > 0:
            self.publish_trajectory(midpoints, transform_to_car)

        else:
            self.get_logger().warn("Could not calculate a new trajectory, no midpoints found inside the map")

    def publish_map_markers(self):
        marker_array = MarkerArray()
        current_count = len(self.rolling_map.cone_map)

        # A new Marker for each cone
        for i, cone in enumerate(self.cone_map.cone_map):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cones"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            # Z-Fighting: z value a little bit higher to avoid ground collisions
            marker.pose.position.x = cone.x
            marker.pose.position.y = cone.y
            marker.pose.position.z = 0.15

            # Standard orientation
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.3

            color = ColorRGBA(a=1.0)

            if cone.color == "blue_cone":
                color.b = 1.0
            elif cone.color == "yellow_cone":
                color.r = 1.0
                color.g = 1.0
            elif cone.color == "orange_cone":
                color.r = 0.8
                color.g = 0.5
            elif cone.color == "large_orange_cone":
                color.r = 1.0
                color.g = 0.1
            else:
                color.r = 1.0
                color.g = 1.0
                color.b = 1.0

            marker.color = color
            marker_array.markers.append(marker)

        # Deleting cones discarded by the rolling map culling
        for i in range(current_count, self.previous_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = self.world_frame
            delete_marker.ns = "cones"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        self.previous_marker_count = current_count
        self.rolling_map_marker_pub.publish(marker_array)

    def calculate_centerline(self, min_hit_count):
        left_cones = []
        right_cones = []

        for cone in self.rolling_map.cone_map:
            if cone.hit_count < min_hit_count:
                continue

            if cone.color == "blue_cone":
                left_cones.append([cone.x, cone.y])

            elif cone.color == "yellow_cone":
                right_cones.append([cone.x, cone.y])

            elif cone.color in ["orange_cone", "large_orange_cone"]:
                if cone.y > 0.0:  # Positive Y = Left (REP 103)
                    left_cones.append([cone.x, cone.y])

                else:
                    right_cones.append([cone.x, cone.y])

        if len(left_cones) < 1 or len(right_cones) < 1:
            return []

        all_cones = np.array(left_cones + right_cones)
        sides = np.array([0]*len(left_cones) + [1]*len(right_cones))

        try:
            tri = Delaunay(all_cones)

        except Exception:
            return []

        midpoints = []
        for simplex in tri.simplices:
            edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]

            for p1_idx, p2_idx in edges:
                if sides[p1_idx] != sides[p2_idx]:
                    p1 = all_cones[p1_idx]
                    p2 = all_cones[p2_idx]
                    dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])

                    if self.delaunay_min_distance < dist < self.delaunay_max_distance:
                        midpoints.append([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0])

        if len(midpoints) > 0:
            midpoints = np.unique(np.array(midpoints), axis=0).tolist()

        return midpoints

    def smooth_and_resample(self, ordered_points, resolution=0.5):
        if len(ordered_points) < 4:
            return ordered_points

        pts = np.array(ordered_points)
        x = pts[:, 0]
        y = pts[:, 1]

        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        num_points = max(int(np.sum(dists) / resolution), 2)

        try:
            tck, _u = spi.splprep([x, y], s=self.spline_smoothing, k=self.spline_degree)
            u_new = np.linspace(0, 1.0, num_points)
            new_x, new_y = spi.splev(u_new, tck)
            return np.vstack((new_x, new_y)).T.tolist()

        except Exception:
            return ordered_points

    def get_curvature(self, p1, p2, p3):
        area = 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
        a = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        b = math.hypot(p2[0]-p3[0], p2[1]-p3[1])
        c = math.hypot(p3[0]-p1[0], p3[1]-p1[1])

        if a * b * c == 0.0:
            return 0.0

        curvature = (4.0 * area) / (a * b * c)
        cross_z = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])

        return curvature * (1.0 if cross_z > 0 else -1.0)

    def publish_trajectory(self, midpoints, transform_to_car):
        if len(midpoints) < 3:
            return

        # Find the starting point of the trajectory (min local X)
        start_point = None
        min_local_x = float('inf')

        for p in midpoints:
            p_world = PointStamped()
            p_world.header.frame_id = self.world_frame
            p_world.point.x = float(p[0])
            p_world.point.y = float(p[1])
            p_world.point.z = 0.0

            p_local = do_transform_point(p_world, transform_to_car)

            if p_local.point.x < min_local_x:
                min_local_x = p_local.point.x
                start_point = p

        # Spatial ordering
        current_pos = start_point
        ordered_points = [current_pos]
        unvisited = [p for p in midpoints if not np.array_equal(p, current_pos)]

        MIN_DIST_BETWEEN_WAYPOINTS = 0.5

        while unvisited:
            distances = [math.hypot(p[0] - current_pos[0], p[1] - current_pos[1]) for p in unvisited]
            closest_idx = np.argmin(distances)
            closest_point = unvisited.pop(closest_idx)

            if math.hypot(closest_point[0] - current_pos[0], closest_point[1] - current_pos[1]) > MIN_DIST_BETWEEN_WAYPOINTS:
                ordered_points.append(closest_point)
                current_pos = closest_point

        # Smoothing and resampling
        dense_points = self.smooth_and_resample(ordered_points, resolution=self.spline_sampling_resolution)

        if len(dense_points) < 3:
            return

        # Message and physics calc
        traj_msg = Trajectory()
        traj_msg.header.frame_id = self.world_frame
        traj_msg.header.stamp = self.get_clock().now().to_msg()

        MAX_LAT_ACCEL = 5.0
        MAX_SPEED = 15.0
        MIN_SPEED = 3.0

        k_array = [0.0] * len(dense_points)
        v_array = [0.0] * len(dense_points)

        for i in range(1, len(dense_points) - 1):
            k = self.get_curvature(dense_points[i-1], dense_points[i], dense_points[i+1])
            k_array[i] = k
            abs_k = abs(k)
            v_array[i] = max(MIN_SPEED, min(MAX_SPEED, math.sqrt(MAX_LAT_ACCEL / abs_k) if abs_k >= 1e-4 else MAX_SPEED))

        k_array[0] = k_array[1]
        v_array[0] = v_array[1]
        k_array[-1] = k_array[-2]
        v_array[-1] = v_array[-2]

        for i, p in enumerate(dense_points):
            pt = Point()
            pt.x, pt.y, pt.z = float(p[0]), float(p[1]), 0.0
            traj_msg.trajectory.append(pt)
            traj_msg.curvatures.append(float(k_array[i]))
            traj_msg.velocities.append(float(v_array[i]))

        self.output_publisher.publish(traj_msg)

        # Debug
        if self.debug_output:
            vis_msg = Marker()
            vis_msg.header = traj_msg.header
            vis_msg.ns = "trajectory"
            vis_msg.id = 0
            vis_msg.type = Marker.LINE_STRIP
            vis_msg.action = Marker.ADD
            vis_msg.pose.orientation.w = 1.0
            vis_msg.scale.x = 0.1
            vis_msg.color.g = 1.0 # Green
            vis_msg.color.a = 1.0
            vis_msg.points = traj_msg.trajectory
            self.trajectory_marker_pub.publish(vis_msg)


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
