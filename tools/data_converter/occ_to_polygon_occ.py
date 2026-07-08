import argparse
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mmcv
import numpy as np

from projects.mmdet3d_plugin.datasets.map_utils.polygon_occ_utils import (
    canonicalize_polygon,
    polygon_validity_stats,
    sample_closed_polygon,
)


def load_mapping(path: Optional[str]) -> Optional[Dict[int, int]]:
    if path is None:
        return None
    raw = mmcv.load(path)
    return {int(k): int(v) for k, v in raw.items()}


def parse_ignore_ids(raw: str) -> List[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def iter_manifest_samples(data) -> Iterable[Dict]:
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list):
        raise ValueError("manifest must be a list or dict with 'samples'")
    for item in data:
        if not isinstance(item, dict):
            continue
        yield item


def load_array(path: str, key: Optional[str] = None) -> np.ndarray:
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
    selected = np.isin(occ, source_ids)
    return np.any(selected, axis=project_axis)


def build_class_masks(
    array: np.ndarray,
    input_type: str,
    class_mapping: Optional[Dict[int, int]],
    project_axis: int,
    ignore_source_ids: List[int],
) -> Dict[int, np.ndarray]:
    if input_type == "label_map":
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
        if array.ndim != 3:
            raise ValueError("occupancy input must be 3D")
        if class_mapping is None:
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
            reverse.setdefault(target_id, []).append(source_id)
        for target_id, source_ids in reverse.items():
            grouped[target_id] = reduce_occupancy_to_bev(
                array, source_ids, project_axis=project_axis
            )
        return grouped

    raise ValueError(f"Unsupported input type: {input_type}")


def contour_to_metric(
    contour: np.ndarray,
    x_min: float,
    y_min: float,
    x_step: float,
    y_step: float,
) -> np.ndarray:
    contour = contour.reshape(-1, 2).astype(np.float32)
    cols = contour[:, 0]
    rows = contour[:, 1]
    xs = x_min + (cols + 0.5) * x_step
    ys = y_min + (rows + 0.5) * y_step
    return np.stack([xs, ys], axis=-1)


def binary_mask_to_polygons(
    mask: np.ndarray,
    num_points: int,
    min_pixels: int,
    simplify_ratio: float,
    x_min: float,
    y_min: float,
    x_step: float,
    y_step: float,
) -> List[np.ndarray]:
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max() <= 1:
        mask_u8 = mask_u8 * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    polygons = []
    for label_idx in range(1, num_labels):
        area_pixels = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area_pixels < min_pixels:
            continue
        component = (labels == label_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3 or cv2.contourArea(contour) <= 0:
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = simplify_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        metric_pts = contour_to_metric(approx, x_min, y_min, x_step, y_step)
        if len(metric_pts) < 3:
            continue
        sampled = sample_closed_polygon(metric_pts, num_points=num_points)
        canonical = canonicalize_polygon(sampled)
        stats_dict = polygon_validity_stats(canonical)
        if stats_dict["is_valid"] < 1.0 or stats_dict["area"] <= 0.0:
            continue
        polygons.append(canonical)
    return polygons


def convert_sample(
    sample: Dict,
    class_mapping: Optional[Dict[int, int]],
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
) -> Tuple[str, Dict[int, List[List[List[float]]]]]:
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
    result = OrderedDict()
    for target_id, mask in masks.items():
        polygons = binary_mask_to_polygons(
            mask=mask,
            num_points=num_points,
            min_pixels=min_pixels,
            simplify_ratio=simplify_ratio,
            x_min=x_min,
            y_min=y_min,
            x_step=x_step,
            y_step=y_step,
        )
        if polygons:
            result[int(target_id)] = [polygon.tolist() for polygon in polygons]
    return sample_token, result


def summarize(results: Dict[str, Dict[int, List[List[List[float]]]]]) -> Dict:
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
    parser.add_argument("--x-min", type=float, default=-15.0)
    parser.add_argument("--y-min", type=float, default=-30.0)
    parser.add_argument("--x-step", type=float, default=0.5)
    parser.add_argument("--y-step", type=float, default=0.5)
    parser.add_argument("--summary-path", type=str, default=None)
    args = parser.parse_args()

    class_mapping = load_mapping(args.class_mapping)
    ignore_source_ids = parse_ignore_ids(args.ignore_source_ids)
    manifest = mmcv.load(args.manifest)
    results = OrderedDict()
    for sample in iter_manifest_samples(manifest):
        token, sample_result = convert_sample(
            sample=sample,
            class_mapping=class_mapping,
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
        )
        results[token] = sample_result

    output_dir = os.path.dirname(args.output)
    if output_dir:
        mmcv.mkdir_or_exist(output_dir)
    mmcv.dump({"results": results}, args.output)

    summary = summarize(results)
    if args.summary_path is not None:
        summary_dir = os.path.dirname(args.summary_path)
        if summary_dir:
            mmcv.mkdir_or_exist(summary_dir)
        mmcv.dump(summary, args.summary_path)
    print(summary)


if __name__ == "__main__":
    main()
