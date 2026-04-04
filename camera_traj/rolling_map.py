import math
from dataclasses import dataclass

from geometry_msgs.msg import PointStamped

from tf2_geometry_msgs import do_transform_point


@dataclass
class MapPoint:
    """
    Standard coordinates REP 103
    x axis positive at front
    y axis positive at left

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
