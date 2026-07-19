from typing import Dict, Iterable, List, Optional

import cv2
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
    """把 polygon 顶点序列规范化为唯一、稳定的表示。

    为什么需要这个步骤：
    同一个几何 polygon，可以有很多“等价写法”：

    1. 起点不同
       [P0, P1, P2, P3]
       [P1, P2, P3, P0]

    2. 方向不同
       顺时针 vs 逆时针

    3. 有些数据会把闭合点重复写一遍
       [P0, P1, ..., Pn, P0]

    这些表示在几何上是同一个 polygon，但如果直接拿去做固定顺序点回归，
    L1/point loss 会把它们当成完全不同的目标，导致监督非常不稳定。

    本函数做三件事：
    1. 去掉末尾重复的闭合点
    2. 统一方向（默认顺时针）
    3. 统一起点（使用字典序最小的点作为第一个点）

    输出仍然是 [N, 2]，但现在它是一个“canonical representation”，
    可以更稳定地参与匹配、监督和缓存。
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points.copy()

    # 某些 polygon 库/数据导出会把首点再重复放到最后一个位置，形成显式闭合。
    # 这里把它移除，避免后面误认为多了一个真实顶点。
    if np.allclose(points[0], points[-1], atol=1e-5):
        points = points[:-1]

    # 统一方向。默认使用顺时针表示；如果调用方想保留逆时针，也可以传
    # clockwise=False，此时逻辑是“先转成顺时针，再整体反过来”。
    if clockwise:
        points = ensure_clockwise(points)
    else:
        cw_points = ensure_clockwise(points)
        points = cw_points[::-1].copy()

    # 统一起点。这里选字典序最小的点（先比 x，再比 y）作为 index 0。
    # 这样同一个 polygon 即使原始起点不同，最终也会落到同一套顶点顺序上。
    return normalize_polygon_start(points)


def sample_closed_polygon(points: np.ndarray, num_points: int = 32) -> np.ndarray:
    """沿闭合 polygon 周长均匀采样固定数量的点。

    输入：
        points: [N, 2] 的有序 polygon 顶点，可以闭合也可以不闭合。
                如果首尾相同，函数会自动去掉最后一个重复点。
        num_points: 最终采样点数，默认 32。

    输出：
        [num_points, 2] 的采样结果。

    设计原因：
    - 原始 contour 点数是可变的，无法直接喂给固定维度 head。
    - 直接按“索引均匀采样”会让点密度依赖 contour 离散化方式，结果不稳定。
    - 正确做法是按“几何周长距离”均匀采样，这样采样点在空间上更均匀。

    这里的采样逻辑是：
    1. 把 polygon 看成闭合折线
    2. 计算每条边的长度
    3. 计算总周长 perimeter
    4. 在 [0, perimeter) 上等间距放置 num_points 个目标距离
    5. 对每个目标距离，找到它落在哪条边上，再做线性插值

    这保证：
    - 不管原始 contour 有多少点，输出维度固定
    - 简单形状和复杂形状都能用统一接口表示
    - 采样更接近几何等距，而不是像素索引等距
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if len(points) < 3:
        raise ValueError("polygon requires at least 3 points")

    # 如果输入已经显式闭合（首尾重复），先去掉最后一个重复点。
    # 后面会自己再构造闭合边，不需要保留重复点。
    if np.allclose(points[0], points[-1], atol=1e-5):
        points = points[:-1]

    # 构造显式闭合折线：最后一个点连回第一个点。
    closed = np.concatenate([points, points[:1]], axis=0)
    edges = closed[1:] - closed[:-1]
    edge_lengths = np.linalg.norm(edges, axis=1)
    perimeter = float(edge_lengths.sum())

    # 退化情况：如果周长接近 0，说明 polygon 基本塌缩成一点或几乎不可用。
    # 这里退化返回重复点，避免后续出现除零错误；上层通常还会再做有效性过滤。
    if perimeter <= 1e-6:
        return np.repeat(points[:1], num_points, axis=0)

    # cumulative[i] 表示从起点走到第 i 条边起点时的累计周长。
    cumulative = np.concatenate(
        [np.array([0.0], dtype=np.float32), np.cumsum(edge_lengths)]
    )

    # 目标采样距离均匀分布在 [0, perimeter) 上。
    # 注意这里不包含 perimeter 本身，因为 perimeter 对应的点和 0 是同一个位置。
    target_distances = (
        np.arange(num_points, dtype=np.float32) * perimeter / num_points
    )

    sampled = []
    edge_idx = 0
    for distance in target_distances:
        # 找到当前 distance 落在哪一条边上。
        while (
            edge_idx < len(edge_lengths) - 1
            and cumulative[edge_idx + 1] <= distance
        ):
            edge_idx += 1
        edge_length = edge_lengths[edge_idx]

        # 某条边非常短时，直接取边起点，避免数值不稳定。
        if edge_length <= 1e-8:
            sampled.append(closed[edge_idx].copy())
            continue

        # alpha 是 distance 在这条边上的相对位置比例，取值通常在 [0, 1)。
        alpha = (distance - cumulative[edge_idx]) / edge_length

        # 在线段上做线性插值，得到真正的采样点坐标。
        sampled.append(closed[edge_idx] + alpha * edges[edge_idx])
    return np.asarray(sampled, dtype=np.float32)


def sample_polygon_preserve_vertices(
    points: np.ndarray,
    num_points: int = 32,
) -> np.ndarray:
    """Sample a closed polygon while preserving original vertices when possible.

    Strategy:
    1. Extract a compact set of meaningful vertices within the point budget.
       When the raw contour has too many vertices, use adaptive Douglas-Peucker
       simplification to keep the most important corners/endpoints first.
    2. Distribute the remaining point budget to edges proportionally to edge
       length.

    This is mainly intended for Polygon OCC, where sharp corners / ROI clipping
    intersection points are often more important than spending many samples on
    long straight edges.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if len(points) < 3:
        raise ValueError("polygon requires at least 3 points")

    if np.allclose(points[0], points[-1], atol=1e-5):
        points = points[:-1]

    points = simplify_polygon_vertices_to_budget(points, max_vertices=num_points)

    num_vertices = len(points)
    if num_vertices == num_points:
        return points.copy()

    closed = np.concatenate([points, points[:1]], axis=0)
    edges = closed[1:] - closed[:-1]
    edge_lengths = np.linalg.norm(edges, axis=1)
    perimeter = float(edge_lengths.sum())
    if perimeter <= 1e-6:
        return np.repeat(points[:1], num_points, axis=0)

    remain = num_points - num_vertices
    raw_alloc = edge_lengths / perimeter * remain
    edge_alloc = np.floor(raw_alloc).astype(np.int64)
    leftover = int(remain - edge_alloc.sum())
    if leftover > 0:
        residual_order = np.argsort(-(raw_alloc - edge_alloc))
        edge_alloc[residual_order[:leftover]] += 1

    sampled = []
    for edge_idx in range(num_vertices):
        start = closed[edge_idx]
        edge = edges[edge_idx]
        sampled.append(start.copy())

        num_edge_points = int(edge_alloc[edge_idx])
        if num_edge_points <= 0:
            continue
        for point_idx in range(num_edge_points):
            alpha = (point_idx + 1) / (num_edge_points + 1)
            sampled.append(start + alpha * edge)

    sampled = np.asarray(sampled, dtype=np.float32)
    if len(sampled) != num_points:
        raise ValueError(
            f"sample_polygon_preserve_vertices expected {num_points} points, got {len(sampled)}"
        )
    return sampled


def simplify_polygon_vertices_to_budget(
    points: np.ndarray,
    max_vertices: int,
) -> np.ndarray:
    """Reduce a polygon to at most `max_vertices` meaningful vertices.

    This keeps sharp turns / clipping corners preferentially by using adaptive
    Douglas-Peucker simplification on the closed polygon contour until the
    number of retained vertices fits the target budget.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if len(points) < 3:
        raise ValueError("polygon requires at least 3 points")

    if np.allclose(points[0], points[-1], atol=1e-5):
        points = points[:-1]

    if len(points) <= max_vertices:
        return points.copy()

    closed = np.concatenate([points, points[:1]], axis=0)
    contour = closed.reshape(-1, 1, 2).astype(np.float32)
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        return points[:max_vertices].copy()

    low, high = 0.0, perimeter
    best = None
    for _ in range(32):
        epsilon = 0.5 * (low + high)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        if np.allclose(approx[0], approx[-1], atol=1e-5):
            approx = approx[:-1]

        if len(approx) > max_vertices:
            low = epsilon
            continue

        if len(approx) >= 3:
            best = approx.astype(np.float32)
        high = epsilon

    if best is not None:
        return best

    # Extremely degenerate fallback: retain a uniform subset of original
    # vertices. This path should be rare, but keeps the function total.
    keep_idx = np.linspace(0, len(points), num=max_vertices, endpoint=False)
    keep_idx = np.floor(keep_idx).astype(np.int64)
    keep_idx = np.clip(keep_idx, 0, len(points) - 1)
    return points[keep_idx].copy()


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
