import argparse
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

from projects.mmdet3d_plugin.datasets.map_utils.polygon_occ_utils import (
    canonicalize_polygon,
    polygon_validity_stats,
    sample_closed_polygon,
    sample_polygon_preserve_vertices,
)


def load_mapping(path: Optional[str]) -> Optional[Dict[int, int]]:
    """Load an optional source-id -> target-id class remapping file."""
    # class_mapping 用来把原始 occupancy 语义 id 映射到 Polygon OCC 自己的
    # 类别体系。例如可以把多个 vehicle 子类并到同一个 "vehicle region"。
    if path is None:
        return None
    raw = mmcv.load(path)
    mapping = {}
    for k, v in raw.items():
        if not str(k).isdigit():
            continue
        mapping[int(k)] = int(v)
    return mapping


def load_bbox_mapping(path: Optional[str]) -> Optional[Dict[str, int]]:
    """Load optional detection-class-name -> target-id mapping for hybrid GT."""
    if path is None:
        return None
    raw = mmcv.load(path)
    mapping = {}
    for k, v in raw.items():
        if str(k).startswith("__"):
            continue
        mapping[str(k)] = int(v)
    return mapping


def parse_ignore_ids(raw: str) -> List[int]:
    """Parse comma-separated source ids that should be dropped before conversion."""
    # 常见用法是忽略 0（free/unknown/background），避免把“空区域”错误地
    # 转成 polygon。
    raw = raw.strip()
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def iter_manifest_samples(data) -> Iterable[Dict]:
    """Iterate manifest samples from the standard {'samples': [...]} format."""
    # manifest 是上游 build_occ3d_manifest.py 生成的中间文件。
    # 它的目标很简单：按 sample_token 给出每个样本的 occupancy 文件路径，
    # 让本脚本专注做“occupancy -> polygon”的转换，而不用再关心数据集索引。
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list):
        raise ValueError("manifest must be a list or dict with 'samples'")
    for item in data:
        if not isinstance(item, dict):
            continue
        yield item


def load_array(path: str, key: Optional[str] = None) -> np.ndarray:
    """Load a numpy array from .npy or .npz, with optional key inference.

    The goal is to support several occupancy exporters without hard-coding one
    exact field name. When `key` is absent, we try a short list of common names.
    """
    # 兼容两类文件：
    # 1. .npy 直接就是一个 ndarray
    # 2. .npz 包含多个字段，例如 Occ3D 常见的 semantics/mask_camera/mask_lidar
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        return data
    if key is not None:
        return data[key]
    if len(data.files) == 1:
        return data[data.files[0]]
    for candidate in ["occ", "occupancy", "semantic", "mask", "masks", "bev"]:
        if candidate in data.files:
            return data[candidate]
    raise ValueError(f"Could not infer array key for {path}")


def reduce_occupancy_to_bev(
    occ: np.ndarray,
    source_ids: List[int],
    project_axis: int,
) -> np.ndarray:
    """Collapse a 3D occupancy tensor into a 2D BEV mask for one semantic group."""
    # 这里的策略是“只要沿着投影轴上任意一个 voxel 命中该类，就在 BEV 上记为 1”。
    # 这是 Polygon OCC V1 最简单也最稳的投影方式之一。
    selected = np.isin(occ, source_ids)
    return np.any(selected, axis=project_axis)


def build_class_masks(
    array: np.ndarray,
    input_type: str,
    class_mapping: Optional[Dict[int, int]],
    project_axis: int,
    ignore_source_ids: List[int],
) -> Dict[int, np.ndarray]:
    """Convert input occupancy representation into per-class binary BEV masks.

    Supported inputs:
    - label_map: 2D semantic id grid
    - mask_stack: [C, H, W] binary or probability masks
    - occupancy: 3D semantic id tensor reduced along `project_axis`
    """
    # 最终目标是统一成：
    #   {target_class_id: binary_bev_mask}
    # 后续无论原始输入是 2D 还是 3D，都走同一套“连通域 -> contour -> polygon”逻辑。
    if input_type == "label_map":
        # 情况 1：输入已经是二维语义标签图，每个像素直接存一个类别 id。
        if array.ndim != 2:
            raise ValueError("label_map input must be 2D")
        masks = OrderedDict()
        unique_ids = sorted(int(x) for x in np.unique(array))
        for source_id in unique_ids:
            if source_id in ignore_source_ids:
                continue
            if class_mapping is None:
                target_id = source_id
            elif source_id in class_mapping:
                target_id = class_mapping[source_id]
            else:
                continue
            masks.setdefault(target_id, np.zeros_like(array, dtype=bool))
            masks[target_id] |= array == source_id
        return masks

    if input_type == "mask_stack":
        # 情况 2：输入是 [C, H, W] 的类别 mask 堆栈，每个通道代表一个类别。
        if array.ndim != 3:
            raise ValueError("mask_stack input must be 3D [C, H, W]")
        masks = OrderedDict()
        for source_id in range(array.shape[0]):
            if source_id in ignore_source_ids:
                continue
            if class_mapping is None:
                target_id = source_id
            elif source_id in class_mapping:
                target_id = class_mapping[source_id]
            else:
                continue
            masks.setdefault(target_id, np.zeros(array.shape[1:], dtype=bool))
            masks[target_id] |= array[source_id].astype(bool)
        return masks

    if input_type == "occupancy":
        # 情况 3：输入是三维 occupancy 语义体。这里不直接做 polygon，先按类
        # 投影成 BEV mask，再交给后续统一流程。
        if array.ndim != 3:
            raise ValueError("occupancy input must be 3D")
        if class_mapping is None:
            # 如果用户没有显式给映射，就默认“原始类 id -> 相同的目标类 id”。
            source_ids = sorted(int(x) for x in np.unique(array))
            class_mapping = {
                source_id: source_id
                for source_id in source_ids
                if source_id not in ignore_source_ids
            }
        grouped = OrderedDict()
        reverse = OrderedDict()
        for source_id, target_id in class_mapping.items():
            if source_id in ignore_source_ids:
                continue
            # reverse 把“多个 source class -> 一个 target class”的情况聚合起来。
            reverse.setdefault(target_id, []).append(source_id)
        for target_id, source_ids in reverse.items():
            grouped[target_id] = reduce_occupancy_to_bev(
                array, source_ids, project_axis=project_axis
            )
        return grouped

    raise ValueError(f"Unsupported input type: {input_type}")


def rotate_mask_ccw90(mask: np.ndarray) -> np.ndarray:
    """Rotate a 2D BEV mask 90 degrees counterclockwise."""
    return np.rot90(np.asarray(mask), 1)


def rotate_points_ccw90(points: np.ndarray) -> np.ndarray:
    """Rotate 2D points 90 degrees counterclockwise around the origin."""
    points = np.asarray(points, dtype=np.float32)
    return np.stack([-points[:, 1], points[:, 0]], axis=-1)


def fill_binary_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary mask while preserving exterior background."""
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if mask_u8.size == 0:
        return mask_u8.astype(bool)

    flood = mask_u8.copy()
    flood_mask = np.zeros((mask_u8.shape[0] + 2, mask_u8.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, holes)
    return filled > 0


def postprocess_driveable_surface_mask(
    mask: np.ndarray,
    fill_holes: bool,
    close_kernel: int,
) -> np.ndarray:
    """Repair small voids/gaps in driveable_surface before contour extraction."""
    output = np.asarray(mask, dtype=bool)
    if fill_holes:
        output = fill_binary_mask_holes(output)

    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        output = cv2.morphologyEx(
            output.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            kernel,
        ) > 0
    return output


def contour_to_metric(
    contour: np.ndarray,
    x_min: float,
    y_min: float,
    x_step: float,
    y_step: float,
) -> np.ndarray:
    """Convert OpenCV contour pixels into SparseDrive BEV metric coordinates.

    Occ3D/nuScenes-style BEV semantics are treated here as:
      - source x: forward
      - source y: left

    SparseDrive det/map working BEV convention is:
      - target x: right
      - target y: forward

    So after recovering source metric coordinates from `(row=x_src, col=y_src)`,
    we convert them with:
      x_dst = -y_src
      y_dst =  x_src
    """
    contour = contour.reshape(-1, 2).astype(np.float32)
    cols = contour[:, 0]
    rows = contour[:, 1]
    src_xs = x_min + (rows + 0.5) * x_step
    src_ys = y_min + (cols + 0.5) * y_step
    dst_xs = -src_ys
    dst_ys = src_xs
    return np.stack([dst_xs, dst_ys], axis=-1)


def build_ego_to_lidar_transform(
    sample: Dict,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build a row-vector 2D ego/Occ3D -> LiDAR transform from sample metadata."""
    translation = sample.get("lidar2ego_translation")
    rotation = sample.get("lidar2ego_rotation")
    if translation is None or rotation is None:
        return None, None
    lidar2ego_rot = Quaternion(rotation).rotation_matrix[:2, :2].astype(np.float32)
    lidar2ego_trans = np.asarray(translation[:2], dtype=np.float32)
    # Row-vector inverse of: p_ego = p_lidar @ R.T + t
    # is: p_lidar = (p_ego - t) @ R = p_ego @ R - t @ R.
    matrix = lidar2ego_rot
    offset = -lidar2ego_trans @ lidar2ego_rot
    return matrix, offset


def transform_points(points: np.ndarray, matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return points @ matrix + offset


def clip_polygon_to_roi(
    points: np.ndarray,
    roi_x_min: Optional[float],
    roi_x_max: Optional[float],
    roi_y_min: Optional[float],
    roi_y_max: Optional[float],
) -> List[np.ndarray]:
    """Clip a polygon to the target ROI and return one or more valid parts.

    Clipping is done before the final fixed-point sampling so the sampled points
    describe exactly the geometry the model is asked to predict inside the ROI.
    """
    # ----------------------------------------------------------------------
    # 这个函数的作用：
    #   把一个“已经处于米制坐标系中的 polygon”裁剪到目标 ROI 内。
    #
    # 为什么要裁剪：
    #   Occ3D 的原始空间范围比当前 SparseDrive 的局部 ROI 更大。
    #   如果不先裁剪，后续采样得到的 polygon 可能包含大量超出训练范围的边界，
    #   这样会让模型去拟合自己根本不需要预测的区域。
    #
    # 为什么要在“固定点采样之前”裁剪：
    #   正确顺序应该是：
    #       原始 polygon -> ROI 裁剪 -> 固定 32 点采样
    #   而不是：
    #       原始 polygon -> 先采样 32 点 -> 再裁剪
    #
    #   因为如果先采样再裁剪，32 个点中会有不少点落在 ROI 外，
    #   剩下 ROI 内的几何分布就会失真；
    #   先裁剪再采样，32 个点描述的才是真正需要预测的局部几何。
    #
    # 返回值为什么是 List[np.ndarray]：
    #   一个 polygon 和 ROI 相交后，不一定还是一个连通 polygon。
    #   例如一条很长的区域被 ROI 窗口切开，Shapely 可能返回 MultiPolygon。
    #   所以这里统一返回“多个有效 part”。
    # ----------------------------------------------------------------------

    # 如果输入显式闭合（首尾重复），先去掉最后一个重复点。
    # 这里希望交给 Shapely 的是“不重复首点”的顶点序列。
    if np.allclose(points[0], points[-1], atol=1e-5):
        points = points[:-1]

    # 先构造一个 Shapely Polygon，后续所有裁剪/修复都基于这个几何对象进行。
    polygon = Polygon(points)
    if polygon.is_empty or polygon.area <= 0:
        return []
    if not polygon.is_valid:
        # buffer(0) 是 Shapely 中常见的几何修复技巧。
        # 某些自交/退化边界在这里可以被自动修正；如果修正后仍然无效，
        # 就说明这个 polygon 本身不适合作为训练 GT。
        polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            return []

    if None not in (roi_x_min, roi_x_max, roi_y_min, roi_y_max):
        # ROI 使用轴对齐矩形表示。
        # 当前 ROI 的语义是“模型真正需要预测的局部 BEV 范围”。
        roi_polygon = box(roi_x_min, roi_y_min, roi_x_max, roi_y_max)

        # 取相交区域。结果类型可能是：
        #   - Polygon：裁剪后还是一整块
        #   - MultiPolygon：被切成多个不相连部分
        #   - GeometryCollection：混合结果，需要再过滤出 Polygon
        polygon = polygon.intersection(roi_polygon)
        if polygon.is_empty:
            return []

    # 把不同的 Shapely 返回类型统一展开成 polygon 列表，方便后面统一处理。
    if isinstance(polygon, Polygon):
        polygons = [polygon]
    elif isinstance(polygon, MultiPolygon):
        polygons = list(polygon.geoms)
    elif isinstance(polygon, GeometryCollection):
        polygons = [geom for geom in polygon.geoms if isinstance(geom, Polygon)]
    else:
        polygons = []

    outputs = []
    for poly in polygons:
        if poly.is_empty or poly.area <= 0:
            continue

        # Shapely exterior.coords 会返回显式闭合形式：最后一个点重复第一个点。
        # 这里把最后一个重复点去掉，和后续采样/规范化函数的输入格式保持一致。
        coords = np.asarray(poly.exterior.coords, dtype=np.float32)
        if len(coords) < 4:
            # 这里 < 4 的判断是因为闭合坐标至少应该是：
            #   P0, P1, P2, P0
            # 少于 4 个点说明它连一个最小三角形闭合都构不成。
            continue
        outputs.append(coords[:-1])

    # 每个输出 part 都还是“米制坐标下的 polygon 顶点序列”，
    # 还没有做固定点采样；调用方会对每个 part 分别采样成 32 点。
    return outputs


def box_to_corners_bev(box: np.ndarray) -> np.ndarray:
    """Convert SparseDrive/NuScenes GT box [x,y,z,l,w,h,yaw] to LiDAR BEV corners."""
    box = np.asarray(box, dtype=np.float32)
    if box.shape[0] < 7:
        raise ValueError("gt_boxes entries must have at least 7 values")
    center_x, center_y = box[0], box[1]
    length, width = box[3], box[4]
    yaw = box[6]

    local = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=np.float32,
    )
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    corners = local @ rot.T
    corners[:, 0] += center_x
    corners[:, 1] += center_y
    return corners


def boxes_to_polygons(
    gt_boxes: np.ndarray,
    gt_names: List[str],
    valid_flag: Optional[List[bool]],
    bbox_class_mapping: Dict[str, int],
    num_points: int,
    roi_x_min: Optional[float],
    roi_x_max: Optional[float],
    roi_y_min: Optional[float],
    roi_y_max: Optional[float],
) -> Dict[int, List[np.ndarray]]:
    """Convert selected GT boxes into fixed-length canonical footprint polygons."""
    grouped = OrderedDict()
    if valid_flag is None:
        valid_flag = [True] * len(gt_boxes)
    for box, name, is_valid in zip(gt_boxes, gt_names, valid_flag):
        if not is_valid:
            continue
        if name not in bbox_class_mapping:
            continue
        target_id = int(bbox_class_mapping[name])
        corners = box_to_corners_bev(np.asarray(box, dtype=np.float32))
        clipped_parts = clip_polygon_to_roi(
            corners, roi_x_min, roi_x_max, roi_y_min, roi_y_max
        )
        for part in clipped_parts:
            sampled = sample_polygon_preserve_vertices(part, num_points=num_points)
            canonical = canonicalize_polygon(sampled)
            stats = polygon_validity_stats(canonical)
            if stats["is_valid"] < 1.0 or stats["area"] <= 0.0:
                continue
            grouped.setdefault(target_id, [])
            grouped[target_id].append(canonical)
    return grouped


def binary_mask_to_polygons(
    mask: np.ndarray,
    num_points: int,
    min_pixels: int,
    simplify_ratio: float,
    x_min: float,
    y_min: float,
    x_step: float,
    y_step: float,
    roi_x_min: Optional[float],
    roi_x_max: Optional[float],
    roi_y_min: Optional[float],
    roi_y_max: Optional[float],
    point_transform_matrix: Optional[np.ndarray] = None,
    point_transform_offset: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Convert one binary class mask into canonical fixed-length polygons.

    Pipeline:
    connected components -> external contour -> simplify -> metric conversion
    -> fixed-point sampling -> canonical ordering -> validity filtering.
    """
    # ----------------------------------------------------------------------
    # 这个函数处理“单个类别”的二维 BEV 二值 mask。
    #
    # 输入:
    #   mask: [H, W]，某个目标类别在 BEV 上的占据情况。
    #         True / 1 表示该类别在这个网格上存在，False / 0 表示不存在。
    #
    # 输出:
    #   List[np.ndarray]，列表中每个元素都是一个 polygon，shape 为 [num_points, 2]。
    #
    # 为什么输出是“多个 polygon”而不是一个：
    #   同一类别在一帧里可能有多个彼此不连通的语义区域。
    #   例如：
    #   - 多辆车会形成多个 vehicle 区域
    #   - 多块 vegetation / terrain 会形成多个静态区域
    #
    # 整体处理思路是：
    #   1) 先做连通域分解，找到每一块独立区域
    #   2) 对每块区域提取主外轮廓
    #   3) 把像素坐标转成米制坐标
    #   4) 按 ROI 做裁剪
    #   5) 再把可变长轮廓统一采样成固定 num_points 个点
    #   6) 最后做 canonicalize，统一方向和起点
    # ----------------------------------------------------------------------
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max() <= 1:
        # OpenCV 的 connectedComponents / findContours 更习惯处理 0/255 的 uint8 mask。
        # 如果当前还是 0/1，就先放大到 0/255。
        mask_u8 = mask_u8 * 255

    # 8 连通域：对角相邻像素也视作同一区域。
    # 这通常比 4 连通更符合 BEV 占据区域的语义连续性。
    #
    # 返回:
    #   num_labels: 连通域总数（含背景 0）
    #   labels: 每个像素属于哪个连通域
    #   stats: 每个连通域的外接框和面积等统计量
    #   _    : 质心（这里不需要）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    polygons = []

    # 注意：label_idx = 0 总是背景，所以从 1 开始遍历真正的前景连通域。
    for label_idx in range(1, num_labels):
        area_pixels = int(stats[label_idx, cv2.CC_STAT_AREA])
        # 过滤掉太小的碎片。
        # 这是一个很关键的去噪步骤：如果不做，很多 1~2 个像素的小区域也会被
        # 强行转成 polygon，最终会让每帧 polygon 数量爆炸。
        if area_pixels < min_pixels:
            continue

        # 从 labels 中取出当前连通域对应的二值图。
        component = (labels == label_idx).astype(np.uint8) * 255

        # 只取外轮廓（RETR_EXTERNAL），先不建模 holes / inner rings。
        # 这是当前 V1 的最小方案：
        #   有洞的区域，会被视作“只有外边界”的一个 polygon。
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        # 理论上，一块干净的连通域通常应该只有一个主要外轮廓。
        # 但由于像素离散化、边缘毛刺、噪声、OpenCV 追边细节等原因，
        # findContours 仍然可能返回多个候选 contour。
        #
        # 这里保守地取面积最大的那个，假设它才是真正代表该语义区域的主边界。
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3 or cv2.contourArea(contour) <= 0:
            continue

        # 先根据轮廓周长按比例设定 Douglas-Peucker 的简化阈值。
        # simplify_ratio 越大，轮廓越平滑，但几何细节也会损失更多。
        perimeter = cv2.arcLength(contour, True)
        epsilon = simplify_ratio * perimeter
        # 在固定点数采样之前，先去掉栅格轮廓的“楼梯状”噪声。
        # 否则 32 个采样点会被很多无意义的小折线消耗掉。
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        # OpenCV contour 还是像素网格坐标，这里先转成 Occ3D/ego BEV 米制坐标。
        metric_pts = contour_to_metric(approx, x_min, y_min, x_step, y_step)
        if len(metric_pts) < 3:
            continue

        if point_transform_matrix is not None and point_transform_offset is not None:
            # 如果最终训练目标是 SparseDrive LiDAR frame，就必须先把 Occ3D
            # contour 转到 LiDAR frame，再做 ROI 裁剪和固定点采样。
            metric_pts = transform_points(
                metric_pts, point_transform_matrix, point_transform_offset
            )

        # 转成目标训练坐标系后，再按当前目标 ROI 做裁剪。
        # 这样输出的 polygon 才是真正落在 SparseDrive LiDAR-frame 训练 ROI 内的几何。
        #
        # 裁剪后可能出现：
        #   - 完全落在 ROI 外 -> 返回空
        #   - 一个 polygon 被裁成多个不连通片段 -> 返回多个 parts
        clipped_parts = clip_polygon_to_roi(
            metric_pts, roi_x_min, roi_x_max, roi_y_min, roi_y_max
        )
        if not clipped_parts:
            continue

        for clipped_pts in clipped_parts:
            unique_pts = np.unique(np.round(clipped_pts, decimals=6), axis=0)
            if len(unique_pts) < 3:
                continue
            # 无论原始轮廓最终有多少点，都统一采样到固定 num_points（默认 32）。
            # 这样模型 head 才能用固定维度回归输出 polygon。
            try:
                sampled = sample_polygon_preserve_vertices(
                    clipped_pts, num_points=num_points
                )
            except ValueError as exc:
                if "polygon requires at least 3 points" in str(exc):
                    continue
                raise

            # canonicalize：统一顶点方向和起点，避免同一个几何 polygon 只因为
            # 顺/逆时针或起点不同，就变成不同监督目标。
            canonical = canonicalize_polygon(sampled)

            # 做最后一道有效性检查，过滤掉裁剪或采样后退化的 polygon。
            stats_dict = polygon_validity_stats(canonical)
            if stats_dict["is_valid"] < 1.0 or stats_dict["area"] <= 0.0:
                continue
            polygons.append(canonical)

    # 返回当前类别在这一帧里的所有有效 polygon。
    return polygons


def convert_sample(
    sample: Dict,
    class_mapping: Optional[Dict[int, int]],
    bbox_class_mapping: Optional[Dict[str, int]],
    input_type: str,
    array_key: Optional[str],
    project_axis: int,
    ignore_source_ids: List[int],
    num_points: int,
    min_pixels: int,
    simplify_ratio: float,
    x_min: float,
    y_min: float,
    x_step: float,
    y_step: float,
    roi_x_min: Optional[float],
    roi_x_max: Optional[float],
    roi_y_min: Optional[float],
    roi_y_max: Optional[float],
    driveable_fill_holes: bool,
    driveable_close_kernel: int,
) -> Tuple[str, Dict[int, List[List[List[float]]]]]:
    """Convert one manifest sample into token-keyed Polygon OCC annotations."""
    # 返回值结构：
    #   sample_token,
    #   {
    #       target_class_id: [polygon0, polygon1, ...]
    #   }
    # 其中每个 polygon 都是 [num_points, 2] 的列表。
    sample_token = sample.get("sample_token", sample.get("token"))
    if sample_token is None:
        raise ValueError("sample is missing sample_token/token")
    path = sample.get("data_path", sample.get("path"))
    if path is None:
        raise ValueError("sample is missing data_path/path")
    array = load_array(path, key=array_key)
    masks = build_class_masks(
        array,
        input_type,
        class_mapping,
        project_axis,
        ignore_source_ids,
    )
    masks = OrderedDict(
        (target_id, rotate_mask_ccw90(mask)) for target_id, mask in masks.items()
    )
    occ_to_lidar_matrix, occ_to_lidar_offset = build_ego_to_lidar_transform(sample)
    result = OrderedDict()
    for target_id, mask in masks.items():
        # Each target id may produce multiple connected semantic regions, hence
        # multiple polygons under the same class key.
        # 同一个目标类别可能在同一帧里有多个区域，例如多个车辆 footprint，
        # 或多个互不连通的 terrain/vegetation 区域。
        class_simplify_ratio = simplify_ratio
        if int(target_id) == 6:
            # driveable_surface is the largest stuff region and the most
            # sensitive to local boundary smoothing near vehicles/curbs.
            # Repair small voids/gaps first, then keep its contour unsimplified
            # for better geometric fidelity.
            mask = postprocess_driveable_surface_mask(
                mask,
                fill_holes=driveable_fill_holes,
                close_kernel=driveable_close_kernel,
            )
            class_simplify_ratio = 0.0
        polygons = binary_mask_to_polygons(
            mask=mask,
            num_points=num_points,
            min_pixels=min_pixels,
            simplify_ratio=class_simplify_ratio,
            x_min=x_min,
            y_min=y_min,
            x_step=x_step,
            y_step=y_step,
            roi_x_min=roi_x_min,
            roi_x_max=roi_x_max,
            roi_y_min=roi_y_min,
            roi_y_max=roi_y_max,
            point_transform_matrix=occ_to_lidar_matrix,
            point_transform_offset=occ_to_lidar_offset,
        )
        if polygons:
            result[int(target_id)] = [polygon.tolist() for polygon in polygons]

    if bbox_class_mapping is not None and "gt_boxes" in sample and "gt_names" in sample:
        bbox_polygons = boxes_to_polygons(
            gt_boxes=np.asarray(sample["gt_boxes"], dtype=np.float32),
            gt_names=[str(x) for x in sample["gt_names"]],
            valid_flag=sample.get("valid_flag"),
            bbox_class_mapping=bbox_class_mapping,
            num_points=num_points,
            roi_x_min=roi_x_min,
            roi_x_max=roi_x_max,
            roi_y_min=roi_y_min,
            roi_y_max=roi_y_max,
        )
        for target_id, polygons in bbox_polygons.items():
            result.setdefault(int(target_id), [])
            result[int(target_id)].extend([polygon.tolist() for polygon in polygons])
    return sample_token, result


def summarize(results: Dict[str, Dict[int, List[List[List[float]]]]]) -> Dict:
    """Build lightweight dataset-level stats for smoke/debug inspection."""
    # 这里只做轻量统计，不做真正 metric。目的主要是：
    # - 判断 polygon 数量是否失控
    # - 判断所有样本是否都为空
    # - 粗看类别分布是否异常
    per_sample = []
    per_class = OrderedDict()
    for _, sample in results.items():
        total = 0
        for label, polygons in sample.items():
            per_class[label] = per_class.get(label, 0) + len(polygons)
            total += len(polygons)
        per_sample.append(total)
    return {
        "num_samples": len(results),
        "samples_with_polygons": int(sum(x > 0 for x in per_sample)),
        "mean_polygons_per_sample": float(np.mean(per_sample)) if per_sample else 0.0,
        "max_polygons_per_sample": int(max(per_sample)) if per_sample else 0,
        "class_counts": dict(per_class),
    }


def main():
    """CLI entry point for generic occupancy/BEV-mask -> Polygon OCC conversion."""
    # 这个脚本的定位是“通用转换器”：
    # 给它 manifest + occupancy/mask 数据，它就输出 token-keyed 的
    # polygon_occ sidecar 文件，供 dataset 在训练时按 token 对齐读取。
    parser = argparse.ArgumentParser(description="Convert occupancy or BEV masks to polygon_occ_annos")
    parser.add_argument("manifest", type=str, help="input manifest listing samples")
    parser.add_argument("output", type=str, help="output polygon occ annotation file")
    parser.add_argument(
        "--input-type",
        type=str,
        default="occupancy",
        choices=["occupancy", "label_map", "mask_stack"],
        help="input data representation",
    )
    parser.add_argument("--array-key", type=str, default=None, help="optional npz key")
    parser.add_argument("--class-mapping", type=str, default=None, help="json/pkl mapping source id -> target id")
    parser.add_argument(
        "--bbox-class-mapping",
        type=str,
        default=None,
        help="optional json/pkl mapping detection class name -> target id for hybrid GT",
    )
    parser.add_argument("--project-axis", type=int, default=0, help="axis to reduce for occupancy inputs")
    parser.add_argument(
        "--ignore-source-ids",
        type=str,
        default="0",
        help="comma separated source ids to ignore before class mapping",
    )
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--min-pixels", type=int, default=4)
    parser.add_argument("--simplify-ratio", type=float, default=0.005)
    parser.add_argument("--x-min", type=float, default=-40.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--x-step", type=float, default=0.4)
    parser.add_argument("--y-step", type=float, default=0.4)
    parser.add_argument("--roi-x-min", type=float, default=-15.0)
    parser.add_argument("--roi-x-max", type=float, default=15.0)
    parser.add_argument("--roi-y-min", type=float, default=-30.0)
    parser.add_argument("--roi-y-max", type=float, default=30.0)
    parser.add_argument(
        "--driveable-fill-holes",
        action="store_true",
        help="fill interior holes in driveable_surface masks before contour extraction",
    )
    parser.add_argument(
        "--driveable-close-kernel",
        type=int,
        default=0,
        help="optional morphology close kernel for driveable_surface; 0/1 disables",
    )
    parser.add_argument("--summary-path", type=str, default=None)
    args = parser.parse_args()

    class_mapping = load_mapping(args.class_mapping)
    bbox_class_mapping = load_bbox_mapping(args.bbox_class_mapping)
    ignore_source_ids = parse_ignore_ids(args.ignore_source_ids)
    manifest = mmcv.load(args.manifest)
    results = OrderedDict()
    for sample in iter_manifest_samples(manifest):
        # The converter is token-preserving: sample tokens from the manifest are
        # the exact keys written into the final polygon occ sidecar file.
        token, sample_result = convert_sample(
            sample=sample,
            class_mapping=class_mapping,
            bbox_class_mapping=bbox_class_mapping,
            input_type=args.input_type,
            array_key=args.array_key,
            project_axis=args.project_axis,
            ignore_source_ids=ignore_source_ids,
            num_points=args.num_points,
            min_pixels=args.min_pixels,
            simplify_ratio=args.simplify_ratio,
            x_min=args.x_min,
            y_min=args.y_min,
            x_step=args.x_step,
            y_step=args.y_step,
            roi_x_min=args.roi_x_min,
            roi_x_max=args.roi_x_max,
            roi_y_min=args.roi_y_min,
            roi_y_max=args.roi_y_max,
            driveable_fill_holes=args.driveable_fill_holes,
            driveable_close_kernel=args.driveable_close_kernel,
        )
        results[token] = sample_result

    output_dir = os.path.dirname(args.output)
    if output_dir:
        mmcv.mkdir_or_exist(output_dir)

    # 主输出只保留训练/评估真正需要的 token-keyed 标注内容。
    mmcv.dump({"results": results}, args.output)

    # Summary is kept separate from the main annotation file so training only
    # reads the token-keyed results without extra metadata.
    summary = summarize(results)
    if args.summary_path is not None:
        summary_dir = os.path.dirname(args.summary_path)
        if summary_dir:
            mmcv.mkdir_or_exist(summary_dir)
        mmcv.dump(summary, args.summary_path)
    print(summary)


if __name__ == "__main__":
    main()
