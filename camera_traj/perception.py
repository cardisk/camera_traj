import numpy as np

from dataclasses import dataclass


@dataclass
class CameraInfo:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class Point3D:
    x: float
    y: float
    z: float


def project_point_into_3d_optical_frame(point2d, depth, camera_info)
    x_opt = (point2d.x - camera_info.cx) * depth / camera_info.fx
    y_opt = (point2d.y - camera_info.cy) * depth / camera_info.fy
    z_opt = depth
    return Point3D(x_opt, y_opt, z_opt)

@dataclass
class BoundingBox:
    p1: Point2D
    p2: Point2D
    w: float
    h: float
    color: str


def bounding_box_from_detection(det, img_w, img_h):
    color = det["color"]

    xmin, ymin = det["BB"][0]
    p1 = Point2D(
        max(0, int(xmin)),
        max(0, int(ymin))
    )

    xmax, ymax = det["BB"][1]
    p2 = Point2D(
        min(img_w, int(xmax)),
        min(img_h, int(ymax))
    )

    # Something went horribly wrong during the cone detection
    if p1.x >= p2.x or p1.y >= p2.y:
        raise Exception(f"Invalid detection for '{color}', shape is inverted")

    # BB size
    w = p2.x - p1.x
    h = p2.y - p1.y

    # Detection is too small to work with
    if w < 5 or h < 5:
        raise Exception(f"Detection for '{color}' is too small to work with")

    return BoundingBox(p1, p2, w, h, color)


def patched_bounding_box_from_detection(det, img_w, img_h):
    bb = bounding_box_from_detection(det, img_w, img_h)
    bb.p1.x = int(bb.p1.x + (bb.w * 0.25))
    bb.p2.x = int(bb.p2.x - (bb.w * 0.25))
    bb.p1.y = int(bb.p1.y + (bb.h * 0.50))
    bb.p2.y = int(bb.p2.y - (bb.h * 0.05))

    bb.p1.x = max(0, min(bb.p1.x, bb.w - 1))
    bb.p2.x = max(bb.p1.x + 1, min(bb.p2.x, bb.w))
    bb.p1.y = max(0, min(bb.p1.y, bb.h - 1))
    bb.p2.y = max(bb.p2.y + 1, min(bb.p2.y, bb.h))

    return bb


def get_masked_depth_for_bounding_box(depth_img, bb):
    depth_roi = depth_img[bb.p1.y:bb.p2.y, bb.p1.x:bb.p2.x]
    return depth_roi[
        ~np.isnan(depth_roi) & ~np.isinf(depth_roi) & (depth_roi > 0.0)
    ]


def get_bimodal_distance(data, iterations=5):
    if data.size < 10:
        return np.nan

    # The two ends of the distribution
    mu1 = np.min(data)
    mu2 = np.max(data)

    for _ in range(iterations):
        # Every point goes to the nearest center
        dist1 = np.abs(data - mu1)
        dist2 = np.abs(data - mu2)
        cluster1_mask = dist1 < dist2

        # Updating the centers
        if np.any(cluster1_mask) and np.any(~cluster1_mask):
            mu1 = np.mean(data[cluster1_mask])
            mu2 = np.mean(data[~cluster1_mask])
        else:
            break

    # The cone is always the smallest mu
    return float(min(mu1, mu2))


def get_median_distance(data):
    return float(np.percentile(data, 50))
