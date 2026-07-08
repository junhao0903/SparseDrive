from typing import Dict, Iterable, List, Optional

import numpy as np
from shapely.geometry import Polygon


def signed_polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_clockwise(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if signed_polygon_area(points) > 0:
        return points[::-1].copy()
    return points.copy()


def normalize_polygon_start(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points.copy()
    order = np.lexsort((points[:, 1], points[:, 0]))
    start_idx = int(order[0])
    return np.roll(points, -start_idx, axis=0)


def canonicalize_polygon(points: np.ndarray, clockwise: bool = True) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points.copy()
    if np.allclose(points[0], points[-1], atol=1e-5):
        points = points[:-1]
    if clockwise:
        points = ensure_clockwise(points)
    else:
        cw_points = ensure_clockwise(points)
        points = cw_points[::-1].copy()
    return normalize_polygon_start(points)


def sample_closed_polygon(points: np.ndarray, num_points: int = 32) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if len(points) < 3:
        raise ValueError("polygon requires at least 3 points")
    if np.allclose(points[0], points[-1], atol=1e-5):
        points = points[:-1]

    closed = np.concatenate([points, points[:1]], axis=0)
    edges = closed[1:] - closed[:-1]
    edge_lengths = np.linalg.norm(edges, axis=1)
    perimeter = float(edge_lengths.sum())
    if perimeter <= 1e-6:
        return np.repeat(points[:1], num_points, axis=0)

    cumulative = np.concatenate(
        [np.array([0.0], dtype=np.float32), np.cumsum(edge_lengths)]
    )
    target_distances = (
        np.arange(num_points, dtype=np.float32) * perimeter / num_points
    )

    sampled = []
    edge_idx = 0
    for distance in target_distances:
        while (
            edge_idx < len(edge_lengths) - 1
            and cumulative[edge_idx + 1] <= distance
        ):
            edge_idx += 1
        edge_length = edge_lengths[edge_idx]
        if edge_length <= 1e-8:
            sampled.append(closed[edge_idx].copy())
            continue
        alpha = (distance - cumulative[edge_idx]) / edge_length
        sampled.append(closed[edge_idx] + alpha * edges[edge_idx])
    return np.asarray(sampled, dtype=np.float32)


def polygon_validity_stats(points: np.ndarray) -> Dict[str, float]:
    points = np.asarray(points, dtype=np.float32)
    if np.allclose(points[0], points[-1], atol=1e-5):
        points = points[:-1]
    stats = {
        "num_points": int(len(points)),
        "signed_area": float(signed_polygon_area(points)),
        "area": 0.0,
        "perimeter": 0.0,
        "is_valid": 0.0,
    }
    if len(points) < 3:
        return stats
    polygon = Polygon(points)
    stats["area"] = float(polygon.area)
    stats["perimeter"] = float(polygon.length)
    stats["is_valid"] = float(polygon.is_valid and not polygon.is_empty)
    return stats


def normalize_polygon_annotation_item(
    item: Dict,
    expected_num_points: Optional[int] = None,
) -> Optional[Dict]:
    if not isinstance(item, dict):
        return None
    label = item.get("label", item.get("category_id"))
    polygon = item.get("polygon", item.get("points", item.get("vertices")))
    if label is None or polygon is None:
        return None

    polygon = np.asarray(polygon, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        return None
    polygon = canonicalize_polygon(polygon)
    if expected_num_points is not None and len(polygon) != expected_num_points:
        return None

    stats = polygon_validity_stats(polygon)
    if stats["is_valid"] < 1.0 or stats["area"] <= 0.0:
        return None
    return {
        "label": int(label),
        "polygon": polygon.tolist(),
        "meta": item.get("meta", {}),
    }


def normalize_polygon_annotation_list(
    items: Iterable[Dict],
    expected_num_points: Optional[int] = None,
) -> List[Dict]:
    normalized = []
    for item in items:
        normalized_item = normalize_polygon_annotation_item(
            item, expected_num_points=expected_num_points
        )
        if normalized_item is not None:
            normalized.append(normalized_item)
    return normalized
