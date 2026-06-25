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
    color_votes is useful in case of missclassifications
    hit_count is the number of times the cone is seen
    miss_count is the number of times the cone is not seen
    seen_in_current_step is to know when to increment the miss_count

    distance from the ground is ignored
    """

    x: float
    y: float
    color: str
    color_votes: dict
    hit_count: int = 1
    miss_count: int = 0
    seen_in_current_step : bool = False
    counted_for_lap: bool = False


class RollingMap:
    def __init__(self, rmsth, efa, rmhth, rmmth, cfr, rmcb, rmcm):
        self.safety_threshold: float = rmsth
        self.cone_map: list[MapPoint] = []
        self.ema_filter_alpha = efa
        self.cone_hit_th = rmhth
        self.cone_miss_th = rmmth
        self.camera_fov_rad = cfr
        self.cull_behind_th = rmcb
        self.cull_max_th = rmcm

    def prepare_update(self):
        for cone in self.cone_map:
            cone.seen_in_current_step = False

    def add_to_map(self, new_point: MapPoint):
        best_dist = float("inf")
        best_match = None

        for cone in self.cone_map:
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
            best_match.color_votes[new_point.color] = best_match.color_votes.get(new_point.color, 0) + 1
            best_match.color = max(best_match.color_votes, key=best_match.color_votes.get)
            best_match.hit_count += 1
            best_match.seen_in_current_step = True
            # reward the cone by decrementing the miss_count
            best_match.miss_count = max(0, best_match.miss_count - 1)
        else:
            new_point.seen_in_current_step = True
            self.cone_map.append(new_point)

    def apply_decay(self, transform_to_car):
        """
        ZED2i has a FOV of more or less 110 degrees (1.9rad)
        """
        kept_cones = []

        for cone in self.cone_map:
            if cone.seen_in_current_step:
                kept_cones.append(cone)
                continue

            # Technically the yolo didn't see the cone, checking
            # if it was in the FOV of the camera

            p_odom = PointStamped()
            p_odom.point.x = cone.x
            p_odom.point.y = cone.y
            p_car = do_transform_point(p_odom, transform_to_car)

            x = p_car.point.x
            y = p_car.point.y
            dist = math.hypot(x, y)

            # Check if it's in front of the camera and inside the FOV
            if x > 0.0 and dist < self.cull_max_th:
                angle = math.atan2(y, x)
                if abs(angle) < (self.camera_fov_rad / 2.0):
                    cone.miss_count += 1

            if cone.miss_count <= self.cone_miss_th:
                kept_cones.append(cone)

        self.cone_map = kept_cones

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
