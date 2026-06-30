import math
from time import clock_getres
import numpy as np

import scipy.interpolate as spi
from scipy.spatial import Delaunay

from dataclasses import dataclass, field

from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

from .rolling_map import RollingMap
from .state import node_state
from .missions import Mission


@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class Point3D:
    x: float
    y: float
    z: float


def get_curvature(p1, p2, p3):
    area = 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
    a = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
    b = math.hypot(p2[0]-p3[0], p2[1]-p3[1])
    c = math.hypot(p3[0]-p1[0], p3[1]-p1[1])

    if a * b * c == 0.0:
        return 0.0

    curvature = (4.0 * area) / (a * b * c)
    cross_z = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])

    return curvature * (1.0 if cross_z > 0 else -1.0)


@dataclass
class Track:
    left_cones: list[Point2D]         = field(default_factory=list)
    right_cones: list[Point2D]        = field(default_factory=list)
    orange_cones: list[Point2D]       = field(default_factory=list)
    large_orange_cones: list[Point2D] = field(default_factory=list)


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
                if (node_state.mission == Mission.Acceleration):
                    p_world = PointStamped()
                    # Here there is no need to use the complete header to do the transformation
                    p_world.header.frame_id = node_state.params["world_frame"]
                    p_world.point.x = cone.x
                    p_world.point.y = cone.y
                    p_world.point.z = 0.0

                    p_car = do_transform_point(p_world, node_state.last_transform_to_car)

                    if p_car.point.y > 0:
                        track.left_cones.append(Point2D(cone.x, cone.y))
                    elif p_car.point.y < 0:
                        track.right_cones.append(Point2D(cone.x, cone.y))
                else:
                    track.orange_cones.append(Point2D(cone.x, cone.y))

            case "large_orange_cone":
                track.large_orange_cones.append(Point2D(cone.x, cone.y))

            case _:
                raise Exception(f"Unknown cone color: {cone.color}")

    return track


def find_unordered_midpoints_inside_track(track: Track) -> list:
    if len(track.left_cones) < 2 and len(track.right_cones) < 2:
            # Straight line 15 meters
            #
            # TODO: make these as parameters!!!
            target_length = 15.0 # meters
            step = 0.5 # meters
            num_points = int(target_length / step)

            straight_line = []
            for i in range(num_points + 1):
                p_local = PointStamped()
                p_local.header.frame_id = node_state.params["car_frame"]
                p_local.point.x = float(i * step)
                p_local.point.y = 0.0
                p_local.point.z = 0.0

                p_world = do_transform_point(p_local, node_state.last_transform_to_world)
                straight_line.append([p_world.point.x, p_world.point.y])

            return straight_line

    # Nx2 shape numpy array
    all_cones_list = [[c.x, c.y] for c in (track.left_cones + track.right_cones)]
    all_cones = np.array(all_cones_list)

    sides = np.array([0] * len(track.left_cones) + [1] * len(track.right_cones))

    try:
        triangulation = Delaunay(all_cones)
    except Exception:
        # TODO: if something happens please throw!!!
        return []

    midpoints = []

    # Appending the car position.
    # Pure Pursuit controller prefers a trajectory near the car
    # midpoints.append([node_state.last_transform_to_world.transform.translation.x,
    #                     node_state.last_transform_to_world.transform.translation.y])

    for simplex in triangulation.simplices:
        edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]

        for p1_idx, p2_idx in edges:
            if sides[p1_idx] != sides[p2_idx]:
                p1 = all_cones[p1_idx]
                p2 = all_cones[p2_idx]
                dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])

                if node_state.params["delaunay_min_distance"] < dist < node_state.params["delaunay_max_distance"]:
                    midpoints.append([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0])

    if len(track.large_orange_cones) >= 2:
        p1 = track.large_orange_cones[0]
        p2 = track.large_orange_cones[1]
        midpoints.append([(p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0])

    if len(midpoints) > 0:
        midpoints = np.unique(np.array(midpoints), axis=0).tolist()

    return midpoints


# TODO: make min_distance_between_midpoints a parameter!!!
def order_midpoints(midpoints: list, min_distance_between_midpoints: float = 0.5) -> list:
    if len(midpoints) == 0:
        return []

    filtered_midpoints = []
    start_point = None
    min_dist_to_car = float('inf')

    # Get car position in world coordinates to initialize our direction vector
    car_pos_world = np.array([
        node_state.last_transform_to_world.transform.translation.x,
        node_state.last_transform_to_world.transform.translation.y
    ])

    # Filter out points behind the car and find the best starting point
    for p in midpoints:
        p_world = PointStamped()
        p_world.header.frame_id = node_state.params["world_frame"]
        p_world.point.x = float(p[0])
        p_world.point.y = float(p[1])
        p_world.point.z = 0.0

        p_local = do_transform_point(p_world, node_state.last_transform_to_car)

        # Discard points behind the car.
        # Using a -0.5m buffer to keep points perfectly aligned with the front axle
        #
        # TODO: make this a parameter!!!
        if p_local.point.x > -0.5:
            filtered_midpoints.append(p)

            # The start point is the closest valid midpoint to the car
            dist_to_car = math.hypot(p_local.point.x, p_local.point.y)
            if dist_to_car < min_dist_to_car:
                min_dist_to_car = dist_to_car
                start_point = p

    if start_point is None:
        return []

    # Spatial ordering (Nearest Neighbour with directional penalty)
    current_pos = np.array(start_point)
    ordered_points = [current_pos.tolist()]

    # Initial direction vector looking from the car to the starting point
    current_dir = current_pos - car_pos_world
    norm_dir = np.linalg.norm(current_dir)
    if norm_dir > 1e-4:
        current_dir = current_dir / norm_dir
    else:
        current_dir = np.array([1.0, 0.0]) # Fallback looking forward

    # Remove the start point from the unvisited list
    unvisited = [np.array(p) for p in filtered_midpoints if not np.array_equal(p, start_point)]

    while unvisited:
        best_idx = -1
        best_score = float('inf')

        for i, p in enumerate(unvisited):
            diff = p - current_pos
            dist = np.linalg.norm(diff)

            if dist == 0:
                continue

            direction_to_p = diff / dist

            # Directional penalty using dot product
            # 1.0 = perfectly straight, 0.0 = 90 degrees turn, -1.0 = opposite direction
            dot_prod = np.dot(current_dir, direction_to_p)

            # Base score is purely the distance
            score = dist

            # STRICT REJECTION: Prevent backward loops
            # If the point is more than 90 degrees backward, discard it entirely
            if dot_prod < -0.1:
                continue

            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx == -1:
            break

        closest_point = unvisited.pop(best_idx)

        # Accept the point only if it respects the minimum required distance
        if np.linalg.norm(closest_point - current_pos) > min_distance_between_midpoints:
            ordered_points.append(closest_point.tolist())

            # Update current direction using a fast Exponential Moving Average (EMA)
            # This smooths out the directional tracking along the path
            new_dir = closest_point - current_pos
            new_dir = new_dir / np.linalg.norm(new_dir)

            current_dir = (0.5 * current_dir) + (0.5 * new_dir)
            current_dir = current_dir / np.linalg.norm(current_dir)

            current_pos = closest_point

    return ordered_points


def project_car_on_midline(ordered_midpoints: list, car_pos: list) -> list:
    if len(ordered_midpoints) < 2:
        return ordered_midpoints

    pts = np.array(ordered_midpoints)
    p_car = np.array(car_pos)

    best_dist = float('inf')
    best_projection = None
    best_index = 0

    # Find the nearest segment to the car
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]

        v = p2 - p1
        w = p_car - p1

        v_norm_sq = np.dot(v, v)
        if v_norm_sq == 0.0:
            continue

        # Calculate the projection limited to the segment
        # Changed from 0.0 to -3.0 to allow projecting backwards if the car is behind the first points
        t = np.clip(np.dot(w, v) / v_norm_sq, -3.0, 1.0)
        projection = p1 + t * v

        dist = np.linalg.norm(p_car - projection)
        if dist < best_dist:
            best_dist = dist
            best_projection = projection
            best_index = i

    if best_projection is None:
        return ordered_midpoints

    new_midpoints = [best_projection.tolist()] + pts[best_index + 1:].tolist()

    return new_midpoints


def extend_trajectory_linearly(points: list, extension_meters: float, step_size: float) -> list:
    if len(points) < 2 or extension_meters <= 0.0:
        return points

    p_last = np.array(points[-1])
    p_prev = np.array(points[-2])

    direction = p_last - p_prev
    norm = np.linalg.norm(direction)

    if norm == 0.0:
        return points

    unit_direction = direction / norm
    num_extra_points = int(extension_meters / step_size)

    extended_points = list(points)
    for i in range(1, num_extra_points + 1):
        new_point = p_last + unit_direction * (i * step_size)
        extended_points.append(new_point.tolist())

    return extended_points


def smooth_midline_with_spline(midpoints: list) -> list:
    degree = node_state.params["spline_degree"]

    if len(midpoints) < degree + 1:
        return midpoints

    pts = np.array(midpoints)
    x = pts[:, 0]
    y = pts[:, 1]

    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    num_points = max(int(np.sum(dists) / node_state.params["spline_sampling_resolution"]), 2)

    try:
        tck, _u = spi.splprep([x, y], s=node_state.params["spline_smoothing"], k=degree)
        u_new = np.linspace(0, 1.0, num_points)
        new_x, new_y = spi.splev(u_new, tck)
        return np.vstack((new_x, new_y)).T.tolist()

    except Exception:
        return midpoints
