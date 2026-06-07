import math
from time import clock_getres
import numpy as np

import scipy.interpolate as spi
from scipy.spatial import Delaunay

from dataclasses import dataclass

from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

from .rolling_map import RollingMap


@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class Point3D:
    x: float
    y: float
    z: float


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


class Track:
    left_cones: list[Point2D] = []
    right_cones: list[Point2D] = []
    orange_cones: list[Point2D] = []
    large_orange_cones: list[Point2D] = []


def find_track_inside_map(rolling_map: RollingMap) -> Track:
    track = Track()

    for cone in rolling_map.cone_map:
        if cone.hit_count < rolling_map.cone_hit_th:
            continue

        match cone.color:
            case "blue_cone":
                track.left_cones.append(Point2D(cone.x, cone.y))

            case "yellow_cone":
                track.right_cones.append(Point2D(cone.x, cone.y))

            case "orange_cone":
                track.orange_cones.append(Point2D(cone.x, cone.y))

            case "large_orange_cone":
                track.large_orange_cones.append(Point2D(cone.x, cone.y))

            case _:
                raise Exception(f"Unknown cone color: {cone.color}")

    return track


def find_unordered_midpoints_inside_track(track: Track, delaunay_min_distance: float,
        delaunay_max_distance: float, is_starting: bool = False, transform_to_world = None) -> list:

    if is_starting and transform_to_world is not None:
        if len(track.large_orange_cones) >= 2:
            p1 = track.large_orange_cones[0]
            p2 = track.large_orange_cones[1]
            mid_world_x = (p1.x + p2.x) / 2.0
            mid_world_y = (p1.y + p2.y) / 2.0

            car_x = transform_to_world.transform.translation.x
            car_y = transform_to_world.transform.translation.y

            dx = mid_world_x - car_x
            dy = mid_world_y - car_y
            dist = math.hypot(dx, dy)

            if dist > 0.0:
                dir_x = dx / dist
                dir_y = dy / dist

                target_length = 15.0 # meters
                step = 0.5 # meters
                num_points = int(target_length / step)

                straight_line = []
                for i in range(num_points + 1):
                    straight_line.append([
                        car_x + dir_x * (i * step),
                        car_y + dir_y * (i * step)
                    ])

                return straight_line

        # Fallback
        return []

    if len(track.left_cones) < 3 and len(track.right_cones) < 3:
        if transform_to_world is not None:
            # Straight line 15 meters
            target_length = 15.0 # meters
            step = 0.5 # meters
            num_points = int(target_length / step)

            straight_line = []
            for i in range(num_points + 1):
                p_local = PointStamped()
                # TODO: refactor frame_id naming
                p_local.header.frame_id = 'car_frame'
                p_local.point.x = float(i * step)
                p_local.point.y = 0.0
                p_local.point.z = 0.0

                p_world = do_transform_point(p_local, transform_to_world)
                straight_line.append([p_world.point.x, p_world.point.y])

            return straight_line

        # Fallback
        return []

    all_cones = np.array(track.left_cones + track.right_cones)
    sides = np.array([0] * len(track.left_cones) + [1] * len(track.right_cones))

    try:
        triangulation = Delaunay(all_cones)
    except Exception:
        return []

    midpoints = []
    for simplex in triangulation.simplices:
        edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]

        for p1_idx, p2_idx in edges:
            if sides[p1_idx] != sides[p2_idx]:
                p1 = all_cones[p1_idx]
                p2 = all_cones[p2_idx]
                dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])

                if delaunay_min_distance < dist < delaunay_max_distance:
                    midpoints.append([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0])

    if len(track.large_orange_cones) >= 2:
        p1 = track.large_orange_cones[0]
        p2 = track.large_orange_cones[1]
        midpoints.append([(p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0])

    if len(midpoints) > 0:
        midpoints = np.unique(np.array(midpoints), axis=0).tolist()

    return midpoints


def order_midpoints(midpoints: list, world_frame, transform_to_car, min_distance_between_midpoints: float = 0.5) -> list:
    if not midpoints:
        return []

    # Find the starting point of the trajectory (min local X)
    start_point: tuple = ()
    min_local_x = float('inf')

    for p in midpoints:
        p_world = PointStamped()
        p_world.header.frame_id = world_frame
        p_world.point.x = float(p[0])
        p_world.point.y = float(p[1])
        p_world.point.z = 0.0

        p_local = do_transform_point(p_world, transform_to_car)

        if p_local.point.x < min_local_x:
            min_local_x = p_local.point.x
            start_point = p

    # Spatial ordering (Nearest Neighbour)
    current_pos: tuple = start_point
    ordered_points: list = [current_pos]
    unvisited = [p for p in midpoints if not np.array_equal(p, current_pos)]

    while unvisited:
        distances = [math.hypot(p[0] - current_pos[0], p[1] - current_pos[1]) for p in unvisited]
        closest_idx = np.argmin(distances)
        closest_point = unvisited.pop(closest_idx)

        if math.hypot(closest_point[0] - current_pos[0], closest_point[1] - current_pos[1]) > min_distance_between_midpoints:
            ordered_points.append(closest_point)
            current_pos = closest_point

    return np.array(ordered_points)


def smooth_midline_with_spline(midpoints: list, degree: int, smoothing_factor: int, sampling_resolution: float = 0.5) -> list:
    if len(midpoints) < degree + 1:
        return midpoints

    pts = np.array(midpoints)
    x = pts[:, 0]
    y = pts[:, 1]

    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    num_points = max(int(np.sum(dists) / sampling_resolution), 2)

    try:
        tck, _u = spi.splprep([x, y], s=smoothing_factor, k=degree)
        u_new = np.linspace(0, 1.0, num_points)
        new_x, new_y = spi.splev(u_new, tck)
        return np.vstack((new_x, new_y)).T.tolist()

    except Exception:
        return midpoints
