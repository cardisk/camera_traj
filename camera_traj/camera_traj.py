import math

import json
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from rclpy.qos import qos_profile_sensor_data

from message_filters import ApproximateTimeSynchronizer, Subscriber

from std_msgs.msg import ColorRGBA, Bool
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray
from driverless_msgs.msg import Mission, BoundingBoxes, Trajectory, Frame

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from .state import node_state
from .rolling_map import MapPoint, RollingMap
from . import perception as prc
from . import geometry as geom
from . import missions as miss


class CameraTraj(Node):
    """
    ROS2 node that calculates the trajectory from the
    camera frames received
    """

    def __init__(self):
        super().__init__(
            "CameraTraj",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )

        # Class fields
        self.bridge = CvBridge()
        self.camera_info = None

        self.lap_counter = 0
        self.last_lap_time = self.get_clock().now()
        self.mission_finished = False
        self.mission = None
        self.max_lateral_acceleration = 0
        self.max_speed = 0
        self.min_speed = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Declaring parameters
        self.p = {
            name: param.value
            for name, param in self.get_parameters_by_prefix('').items()
        }

        node_state.params = self.p

        # Flags
        self.use_sim_time          = self.p["use_sim_time"]
        self.pure_local_trajectory = self.p["pure_local_trajectory"]
        self.debug_output          = self.p["debug_output"]

        # Settings
        self.node_output_frequecy_hz = self.p["node_output_frequecy_hz"]
        self.depth_extraction_algorithm = str(self.p["depth_extraction_algorithm"]).lower()

        self.rmsth  = self.p["rolling_map_safety_threshold"]
        self.rmefa  = self.p["rolling_map_ema_filter_alpha"]
        self.rmhcth = self.p["rolling_map_hit_count_threshold"]
        self.rmmcth = self.p["rolling_map_miss_count_threshold"]
        self.rmcfr  = self.p["camera_fov_rad"]

        self.rmcb   = self.p["cull_distance_behind"]
        self.rmcm   = self.p["cull_distance_max"]

        self.delaunay_min_distance = self.p["delaunay_min_distance"]
        self.delaunay_max_distance = self.p["delaunay_max_distance"]

        self.spline_smoothing           = self.p["spline_smoothing"]
        self.spline_degree              = self.p["spline_degree"]
        self.spline_sampling_resolution = self.p["spline_sampling_resolution"]

        self.car_frame   = self.p["car_frame"]
        self.world_frame = self.p["world_frame"]

        self.target_laps = self.p["target_laps"] + 1
        self.lap_cooldown_sec = self.p["lap_cooldown_sec"]
        self.lap_lateral_bound = self.p["lap_lateral_bound"]
        self.mission_selected_topic = self.p["mission_selected_topic"]
        self.mission_finish_topic = self.p["mission_finish_topic"]

        self.mission_sub = self.create_subscription(
            Mission, self.mission_selected_topic, self.get_mission_selected, 10
        )

        # Publisher
        self.mission_finish_pub = self.create_publisher(Frame, self.mission_finish_topic, 10)

        # Topics
        self.depth_topic             = self.p["depth_topic"]
        self.yolo_topic              = self.p["yolo_topic"]
        self.camera_info_topic       = self.p["camera_info_topic"]
        self.rolling_map_debug_topic = self.p["rolling_map_debug_topic"]
        self.trajectory_debug_topic  = self.p["trajectory_debug_topic"]
        self.output_topic            = self.p["output_topic"]

        if self.pure_local_trajectory:
            self.rmhcth = 1
            self.rmcb   = 0.0
            self.world_frame = self.car_frame

        self.rolling_map = RollingMap(
            self.rmsth,
            self.rmefa,
            self.rmhcth,
            self.rmmcth,
            self.rmcfr,
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
        self.get_logger().info(f"  * node_output_frequecy_hz: {self.node_output_frequecy_hz}")
        self.get_logger().info(f"  * depth_extraction_algorithm: {self.depth_extraction_algorithm}")
        self.get_logger().info("")
        self.get_logger().info(f"  * rolling mapping safety threshold: {self.rmsth}m")
        self.get_logger().info(f"  * rolling mapping EMA alpha: {self.rmefa}")
        self.get_logger().info(f"  * rolling mapping hit count threshold: {self.rmhcth}")
        self.get_logger().info(f"  * rolling mapping miss count threshold: {self.rmmcth}")
        self.get_logger().info("")
        self.get_logger().info(f"  * camera FOV radiants: {self.rmcfr}rad")
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
            CameraInfo, self.camera_info_topic, self.get_camera_info, 10
        )

        # Synchronization of YOLO and Depth messages
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.yolo_sub = Subscriber(self, BoundingBoxes, self.yolo_topic, qos_profile=qos_profile_sensor_data)

        queue_size = 200
        max_difference_in_seconds = 0.2

        self.sync = ApproximateTimeSynchronizer(
            [self.depth_sub, self.yolo_sub], queue_size, max_difference_in_seconds
        )

        self.sync.registerCallback(self.acquire_cones_from_images)

        self.output_publisher = self.create_publisher(Trajectory, self.output_topic, 10)
        self.create_timer(1.0 / self.node_output_frequecy_hz, self.publish_trajectory)

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

    def get_mission_selected(self, mission_msg):
        if self.mission is not None:
            return

        self.mission = miss.get_mission_from_msg(mission_msg.id)

        match self.mission:
            case miss.Mission.EBSTest:
                self.get_logger().info("Mission: EBS Test")
                self.target_laps = self.p["ebs_test.target_laps"] + 1
                self.lap_cooldown_sec = self.p["ebs_test.lap_cooldown_sec"]
                self.max_lateral_acceleration = self.p["ebs_test.max_lateral_acceleration"]
                self.max_speed = self.p["ebs_test.max_speed"]
                self.min_speed = self.p["ebs_test.min_speed"]

            case miss.Mission.Acceleration:
                self.get_logger().info("Mission: Acceleration")
                self.target_laps = self.p["acceleration.target_laps"] + 1
                self.lap_cooldown_sec = self.p["acceleration.lap_cooldown_sec"]
                self.max_lateral_acceleration = self.p["acceleration.max_lateral_acceleration"]
                self.max_speed = self.p["acceleration.max_speed"]
                self.min_speed = self.p["acceleration.min_speed"]

            case miss.Mission.Autocross:
                self.get_logger().info("Mission: Autocross")
                self.target_laps = self.p["autocross.target_laps"] + 1
                self.lap_cooldown_sec = self.p["autocross.lap_cooldown_sec"]
                self.max_lateral_acceleration = self.p["autocross.max_lateral_acceleration"]
                self.max_speed = self.p["autocross.max_speed"]
                self.min_speed = self.p["autocross.min_speed"]

            case miss.Mission.Trackdrive:
                self.get_logger().info("Mission: Trackdrive")
                self.target_laps = self.p["trackdrive.target_laps"] + 1
                self.lap_cooldown_sec = self.p["trackdrive.lap_cooldown_sec"]
                self.max_lateral_acceleration = self.p["trackdrive.max_lateral_acceleration"]
                self.max_speed = self.p["trackdrive.max_speed"]
                self.min_speed = self.p["trackdrive.min_speed"]


    def get_camera_info(self, camera_info_msg):
        if self.camera_info is None:
            # Intrinsic camera matrix for the raw (distorted) images.
            #     [fx  0 cx]
            # K = [ 0 fy cy]
            #     [ 0  0  1]
            # Projects 3D points in the camera coordinate frame to 2D pixel
            # coordinates using the focal lengths (fx, fy) and principal point
            # (cx, cy).
            k = camera_info_msg.k
            self.camera_info = prc.CameraInfo(k[0], k[4], k[2], k[5])
            self.get_logger().info("")
            self.get_logger().info("Camera Intrinsics received!")
            # FIX: BUG?!?!?!
            # self.info_sub.unregister()

    def acquire_cones_from_images(self, depth_msg, yolo_msg):
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

        self.rolling_map.prepare_update()

        for det in yolo:
            distance = 0

            match self.depth_extraction_algorithm:
                case "bimodal":
                    try:
                        bb = prc.bounding_box_from_detection(det, width, height)
                    except Exception as e:
                        self.get_logger().warn(f"Could not get a valid BoundingBox: {e}")
                        continue

                    try:
                        depth_bb = prc.get_masked_depth_for_bounding_box(depth_array, bb)
                    except Exception as e:
                        self.get_logger().warn(f"Could not get a valid Depth BoundingBox: {e}")
                        continue

                    try:
                        if depth_bb.size > 0:
                            distance = prc.get_bimodal_distance(depth_bb)
                        else:
                            self.get_logger().warn(f"Detected {bb.color} but could not get any depth data")
                            continue
                    except Exception as e:
                        self.get_logger().warn(f"Could not get a valid Depth distance: {e}")
                        continue

                case "patching":
                    try:
                        bb = prc.patched_bounding_box_from_detection(det, width, height)
                    except Exception as e:
                        self.get_logger().warn(f"Could not get a valid BoundingBox: {e}")
                        continue

                    try:
                        depth_patched_bb = prc.get_masked_depth_for_bounding_box(depth_array, bb)
                    except Exception as e:
                        self.get_logger().warn(f"Could not get a valid Depth BoundingBox: {e}")
                        continue

                    try:
                        if depth_patched_bb.size > 0:
                            distance = prc.get_median_distance(depth_patched_bb)
                        else:
                            self.get_logger().warn(f"Detected {bb.color} but could not get any depth data")
                            continue
                    except Exception as e:
                        self.get_logger().warn(f"Could not get a valid Depth distance: {e}")
                        continue

                case _:
                    raise Exception(f"Invalid depth extration algorithm: {self.depth_extraction_algorithm}")

            # Bottom center point inside the BB
            # [IMPORTANT] the points used here are not the ones associated
            # with the real z depth. This can be a source of issues
            point2d     = prc.Point2D((bb.p1.x + bb.p2.x) / 2, bb.p2.y)
            point3d_opt = prc.project_point2d_into_3d_optical_frame(point2d, distance, self.camera_info)

            try:
                point3d_world = prc.transform_point3d(
                    self.tf_buffer,
                    point3d_opt,
                    depth_msg.header.stamp,
                    depth_msg.header.frame_id,
                    self.world_frame
                )

                new_map_point = MapPoint(
                    x=point3d_world.x, y=point3d_world.y, color=bb.color, color_votes={}
                )

                self.rolling_map.add_to_map(new_map_point)

            except Exception as e:
                self.get_logger().warn(
                    f"Could not transform in {self.world_frame}: {e}"
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

            if not self.mission_finished:
                current_time = self.get_clock().now()

                for cone in self.rolling_map.cone_map:
                    if cone.color == "large_orange_cone" and not cone.counted_for_lap:

                        # Point into car frame to see local X
                        p_odom = PointStamped()
                        p_odom.header.frame_id = self.world_frame
                        p_odom.point.x = cone.x
                        p_odom.point.y = cone.y

                        p_car = do_transform_point(p_odom, transform_to_car)

                        if p_car.point.x < self.p["cull_distance_behind"] and abs(p_car.point.y) < self.lap_lateral_bound:

                            # Temporal cooldown
                            if (current_time - self.last_lap_time).nanoseconds > (self.lap_cooldown_sec * 1e9):
                                self.lap_counter += 1
                                self.last_lap_time = current_time

                                # TODO: send lap counter here

                                if self.lap_counter >= self.target_laps:
                                    self.mission_finished = True

                            cone.counted_for_lap = True

            # Cleaning the map
            self.rolling_map.apply_decay(transform_to_car)
            self.rolling_map.cull_map(transform_to_car, self.world_frame)

            self.get_logger().info("")
            self.get_logger().info(f"Map updated, active cones: {len(self.rolling_map.cone_map)}")

            if self.debug_output:
                self.publish_active_map_as_marker_array()

        except Exception as e:
            self.get_logger().warn(f"Culling failed. Could not get TF from {self.world_frame} to {self.car_frame}: {e}")
            return

    def publish_active_map_as_marker_array(self):
        marker_array = MarkerArray()
        current_count = len(self.rolling_map.cone_map)

        # A new Marker for each cone
        for i, cone in enumerate(self.rolling_map.cone_map):
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

            if cone.color == "large_orange_cone":
                marker.scale.x = 0.4
                marker.scale.y = 0.4
                marker.scale.z = 0.3
            else:
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
                color.r = 1.0
                color.g = 0.6
            elif cone.color == "large_orange_cone":
                color.r = 1.0
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


    def publish_trajectory(self):
        current_time = self.get_clock().now()

        if self.mission_finished and (current_time - self.last_lap_time).nanoseconds > (node_state.params["mission_finished_shutdown_cmd_delay_sec"] * 1e9):
            finish_msg = Frame()
            finish_msg.id = 351
            finish_msg.dlc = 1
            finish_msg.data = [2, 0, 0, 0, 0, 0, 0, 0]
            self.mission_finish_pub.publish(finish_msg)

        try:
            node_state.last_transform_to_car = self.tf_buffer.lookup_transform(
                self.car_frame,
                self.world_frame,
                rclpy.time.Time(),  # get the latest transformation
                rclpy.duration.Duration(seconds=0.2)
            )
        except Exception:
            self.get_logger().warn(f"No {self.car_frame}, pass")

        try:
            node_state.last_transform_to_world = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.car_frame,
                rclpy.time.Time(),  # get the latest transformation
                rclpy.duration.Duration(seconds=0.2)
            )
        except Exception:
            self.get_logger().warn(f"No {self.world_frame}, pass")

        if node_state.last_transform_to_car is None or node_state.last_transform_to_world is None:
            self.get_logger().warn("Nothing to work with...")
            return

        track = geom.find_track_inside_map(self.rolling_map)
        unordered_midpoint = geom.find_unordered_midpoints_inside_track(track)
        ordered_midpoint = geom.order_midpoints(unordered_midpoint)

        car_pos = [
            node_state.last_transform_to_world.transform.translation.x,
            node_state.last_transform_to_world.transform.translation.y
        ]

        # Fallback starting point: if the first point is too far (> 2.5m), prepend the car
        if len(ordered_midpoint) > 0:
            first_point = np.array(ordered_midpoint[0])
            dist_to_first = np.linalg.norm(first_point - np.array(car_pos))
            if dist_to_first > 2.5:
                ordered_midpoint.insert(0, car_pos)

        midpoints_with_car = geom.project_car_on_midline(ordered_midpoint, car_pos)

        extended_midpoints = geom.extend_trajectory_linearly(
            midpoints_with_car,
            node_state.params["trajectory_extension_meters"],
            node_state.params["spline_sampling_resolution"]
        )

        extended_spline = geom.smooth_midline_with_spline(extended_midpoints)

        if len(extended_spline) <= 1:
            self.get_logger().warn("No points inside spline, skipping...")
            return

        # Message and physics calc
        traj_msg = Trajectory()
        traj_msg.header.frame_id = self.world_frame
        traj_msg.header.stamp = self.get_clock().now().to_msg()

        MAX_LAT_ACCEL = self.max_lateral_acceleration
        MAX_SPEED = self.max_speed
        MIN_SPEED = self.min_speed

        k_array = [0.0] * len(extended_spline)
        v_array = [0.0] * len(extended_spline)

        for i in range(1, len(extended_spline) - 1):
            k = geom.get_curvature(extended_spline[i-1], extended_spline[i], extended_spline[i+1])
            k_array[i] = k
            abs_k = abs(k)

            if self.mission_finished:
                v_array[i] = 0
            else:
                v_array[i] = max(MIN_SPEED, min(MAX_SPEED, math.sqrt(MAX_LAT_ACCEL / abs_k) if abs_k >= 1e-4 else MAX_SPEED))

        k_array[0] = k_array[1]
        v_array[0] = v_array[1]
        k_array[-1] = k_array[-2]
        v_array[-1] = v_array[-2]

        for i, p in enumerate(extended_spline):
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
