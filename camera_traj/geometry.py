import math
import numpy as np

import scipy.interpolate as spi
from scipy.spatial import Delaunay

from dataclasses import dataclass

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


# Points must be already ordered
def find_midline_inside_track(track: Track, delaunay_min_distance: float,
        delaunay_max_distance: float, is_starting: bool = False) -> list[Point2D]:
    if is_starting:
        return []

    if len(track.left_cones) < 3 and len(track.right_cones) < 3:
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

    if len(track.large_orange_cones) == 2:
        p1 = track.large_orange_cones[0]
        p2 = track.large_orange_cones[1]
        midpoints.append([(p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0])

    if len(midpoints) > 0:
        midpoints = np.unique(np.array(midpoints), axis=0).tolist()

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

    return ordered_points


def smooth_midline_with_spline(waypoints: list[Point2D]) -> list[Point2D]:
    return []


def calculate_centerline(self, min_hit_count, transform_to_car):
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
            p_world = PointStamped()
            p_world.header.frame_id = self.world_frame
            p_world.point.x = cone.x
            p_world.point.y = cone.y
            p_world.point.z = 0.0

            p_local = do_transform_point(p_world, transform_to_car)

            if p_local.point.y > 0.0:  # Positive Y = Left (REP 103)
                left_cones.append([cone.x, cone.y])

            else:
                right_cones.append([cone.x, cone.y])

    if len(left_cones) < 1 or len(right_cones) < 1:
        return []

    if len(left_cones) < 3 and len(right_cones) < 3:
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
