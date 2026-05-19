import math

from dataclasses import dataclass

from .rolling_map import MapPoint, RollingMap


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
    # orange_cones: list[Point2D] = []
    # large_orange_cones: list[Point2D] = []
    unordered_midpoints: list[Point2D] = []


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

            case "orange_cone" | "large_orange_cone":
                # TODO: need to find a way to consider them
                pass

            case _:
                raise Exception(f"Unknown cone color: {cone.color}")

    return track


def find_midline_inside_track(track: Track):
    pass

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
