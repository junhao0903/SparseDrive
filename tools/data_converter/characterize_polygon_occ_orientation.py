import math
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data_converter.occ_to_polygon_occ import (
    build_ego_to_lidar_transform,
    clip_polygon_to_roi,
    contour_to_metric,
    transform_points,
)


def legacy_swapped_contour_to_metric(
    contour: np.ndarray,
    x_min: float,
    y_min: float,
    x_step: float,
    y_step: float,
) -> np.ndarray:
    """Control implementation for the old col->x / row->y interpretation."""
    contour = contour.reshape(-1, 2).astype(np.float32)
    cols = contour[:, 0]
    rows = contour[:, 1]
    xs = x_min + (cols + 0.5) * x_step
    ys = y_min + (rows + 0.5) * y_step
    return np.stack([xs, ys], axis=-1)


def transposed_inverse_transform(
    points: np.ndarray,
    sample: dict,
) -> np.ndarray:
    """Control implementation for the common transpose mistake in ego->LiDAR."""
    rot = Quaternion(sample["lidar2ego_rotation"]).rotation_matrix[:2, :2].astype(np.float32)
    translation = np.asarray(sample["lidar2ego_translation"][:2], dtype=np.float32)
    wrong_matrix = rot.T
    wrong_offset = -translation @ rot.T
    return transform_points(points, wrong_matrix, wrong_offset)


def make_z_rotation_quaternion(yaw_degrees: float):
    yaw_radians = math.radians(yaw_degrees)
    return [math.cos(yaw_radians / 2.0), 0.0, 0.0, math.sin(yaw_radians / 2.0)]


def assert_allclose(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-5):
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(
            f"{name} mismatch\nactual=\n{actual}\nexpected=\n{expected}"
        )


def assert_not_allclose(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-5):
    if np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"{name} unexpectedly matched the reference output")


def polygon_bounds(points: np.ndarray):
    return np.array(
        [
            np.min(points[:, 0]),
            np.max(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 1]),
        ],
        dtype=np.float32,
    )


def characterize_identity_case():
    contour = np.array([[[1, 4]], [[3, 4]], [[3, 7]], [[1, 7]]], dtype=np.int32)
    expected = np.array(
        [[-1.5, 4.5], [-3.5, 4.5], [-3.5, 7.5], [-1.5, 7.5]], dtype=np.float32
    )

    current = contour_to_metric(contour, x_min=0.0, y_min=0.0, x_step=1.0, y_step=1.0)
    legacy = legacy_swapped_contour_to_metric(
        contour, x_min=0.0, y_min=0.0, x_step=1.0, y_step=1.0
    )

    sample = {
        "lidar2ego_translation": [0.0, 0.0, 0.0],
        "lidar2ego_rotation": make_z_rotation_quaternion(0.0),
    }
    matrix, offset = build_ego_to_lidar_transform(sample)
    transformed = transform_points(current, matrix, offset)

    assert_allclose("identity contour_to_metric", current, expected)
    assert_allclose("identity build_ego_to_lidar_transform", transformed, expected)
    assert_not_allclose("legacy det/map-incompatible contour mapping", legacy, expected)

    return {
        "case": "identity",
        "reference": expected.tolist(),
        "current": current.tolist(),
        "legacy_swapped": legacy.tolist(),
    }


def characterize_rotated_case():
    contour = np.array([[[1, 4]], [[4, 4]], [[4, 8]], [[1, 8]]], dtype=np.int32)
    x_min = -1.0
    y_min = -6.0
    x_step = 2.0
    y_step = 0.5
    sample = {
        "lidar2ego_translation": [2.0, -1.0, 0.0],
        "lidar2ego_rotation": make_z_rotation_quaternion(90.0),
    }

    current_metric = contour_to_metric(contour, x_min, y_min, x_step, y_step)
    legacy_metric = legacy_swapped_contour_to_metric(contour, x_min, y_min, x_step, y_step)
    matrix, offset = build_ego_to_lidar_transform(sample)
    current_lidar = transform_points(current_metric, matrix, offset)
    legacy_lidar = transform_points(legacy_metric, matrix, offset)
    transposed_lidar = transposed_inverse_transform(current_metric, sample)

    rot = Quaternion(sample["lidar2ego_rotation"]).rotation_matrix[:2, :2].astype(np.float32)
    translation = np.asarray(sample["lidar2ego_translation"][:2], dtype=np.float32)
    reference_lidar = (current_metric - translation) @ rot

    assert_allclose("rotated build_ego_to_lidar_transform", current_lidar, reference_lidar)
    assert_not_allclose("rotated legacy det/map-incompatible contour mapping", legacy_lidar, reference_lidar)
    assert_not_allclose("rotated transposed inverse transform", transposed_lidar, reference_lidar)

    return {
        "case": "rotated",
        "reference_lidar": reference_lidar.tolist(),
        "current_lidar": current_lidar.tolist(),
        "legacy_lidar": legacy_lidar.tolist(),
        "transposed_lidar": transposed_lidar.tolist(),
    }


def characterize_roi_clipping_order():
    occ_polygon = np.array(
        [[8.0, -1.0], [12.0, -1.0], [12.0, 1.0], [8.0, 1.0]], dtype=np.float32
    )
    sample = {
        "lidar2ego_translation": [0.0, 0.0, 0.0],
        "lidar2ego_rotation": make_z_rotation_quaternion(90.0),
    }
    roi = dict(roi_x_min=-2.0, roi_x_max=2.0, roi_y_min=-10.0, roi_y_max=-6.0)

    matrix, offset = build_ego_to_lidar_transform(sample)
    transformed_first = transform_points(occ_polygon, matrix, offset)
    clipped_after_transform = clip_polygon_to_roi(transformed_first, **roi)
    clipped_before_transform = clip_polygon_to_roi(occ_polygon, **roi)

    if len(clipped_after_transform) != 1:
        raise AssertionError(
            f"expected one clipped polygon after transform, got {len(clipped_after_transform)}"
        )
    if clipped_before_transform:
        raise AssertionError("source-frame clipping unexpectedly produced a target-frame ROI hit")

    expected_bounds = np.array([-1.0, 1.0, -10.0, -8.0], dtype=np.float32)
    assert_allclose(
        "ROI clip after transform bounds",
        polygon_bounds(clipped_after_transform[0]),
        expected_bounds,
    )

    return {
        "case": "roi_clip_order",
        "post_transform_bounds": polygon_bounds(clipped_after_transform[0]).tolist(),
        "pre_transform_num_parts": len(clipped_before_transform),
    }


def main():
    identity = characterize_identity_case()
    rotated = characterize_rotated_case()
    roi = characterize_roi_clipping_order()

    print("PASS identity:", identity)
    print("PASS rotated:", rotated)
    print("PASS roi_clip_order:", roi)
    print(
        "VERDICT: build_ego_to_lidar_transform() matches the analytic ego->LiDAR inverse "
        "in both identity and rotated cases. The det/map alignment lives in contour_to_metric(); "
        "legacy non-rotated/unswapped mappings are det/map-incompatible, while the live transform helper is sound. "
        "ROI clipping is only authoritative after transforming into the target LiDAR frame."
    )


if __name__ == "__main__":
    main()
