import argparse
import os
from collections import OrderedDict
from typing import Dict, Iterable, List

import mmcv
import numpy as np

from projects.mmdet3d_plugin.datasets.map_utils.polygon_occ_utils import (
    normalize_polygon_annotation_item,
    polygon_validity_stats,
)


def iter_input_samples(data) -> Iterable:
    if isinstance(data, dict):
        if "results" in data:
            data = data["results"]
        elif "annotations" in data:
            data = data["annotations"]

    if isinstance(data, dict):
        for sample_token, sample_annos in data.items():
            yield sample_token, sample_annos
        return

    if isinstance(data, list):
        for sample in data:
            if not isinstance(sample, dict):
                continue
            sample_token = sample.get("sample_token", sample.get("token"))
            if sample_token is None:
                continue
            sample_annos = sample.get("annotations", sample.get("polygons", []))
            yield sample_token, sample_annos
        return

    raise ValueError("Unsupported polygon occ input format")


def normalize_sample_annotations(sample_annos, expected_num_points=None) -> Dict[int, List[List[List[float]]]]:
    if isinstance(sample_annos, dict):
        if "annotations" in sample_annos:
            sample_annos = sample_annos["annotations"]
        elif "polygons" in sample_annos:
            sample_annos = sample_annos["polygons"]
        else:
            items = []
            for label, polygons in sample_annos.items():
                for polygon in polygons:
                    items.append({"label": label, "polygon": polygon})
            sample_annos = items

    grouped = OrderedDict()
    for item in sample_annos:
        normalized_item = normalize_polygon_annotation_item(
            item, expected_num_points=expected_num_points
        )
        if normalized_item is None:
            continue
        label = normalized_item["label"]
        grouped.setdefault(label, [])
        grouped[label].append(normalized_item["polygon"])
    return grouped


def summarize_annotations(results: Dict[str, Dict[int, List[List[List[float]]]]]):
    polygon_counts = []
    class_counts = OrderedDict()
    area_values = []
    for _, sample_annos in results.items():
        count = 0
        for label, polygons in sample_annos.items():
            class_counts[label] = class_counts.get(label, 0) + len(polygons)
            count += len(polygons)
            for polygon in polygons:
                stats = polygon_validity_stats(np.asarray(polygon, dtype=np.float32))
                area_values.append(stats["area"])
        polygon_counts.append(count)

    summary = {
        "num_samples": len(results),
        "samples_with_polygons": int(sum(count > 0 for count in polygon_counts)),
        "mean_polygons_per_sample": float(np.mean(polygon_counts)) if polygon_counts else 0.0,
        "max_polygons_per_sample": int(max(polygon_counts)) if polygon_counts else 0,
        "class_counts": dict(class_counts),
        "mean_polygon_area": float(np.mean(area_values)) if area_values else 0.0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Normalize Polygon OCC annotations")
    parser.add_argument("input", type=str, help="input annotation file")
    parser.add_argument("output", type=str, help="normalized output annotation file")
    parser.add_argument(
        "--num-points",
        type=int,
        default=32,
        help="expected polygon point count; use <=0 to disable strict check",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="optional summary json/pkl path",
    )
    args = parser.parse_args()

    expected_num_points = args.num_points if args.num_points > 0 else None
    data = mmcv.load(args.input)
    normalized = OrderedDict()
    for sample_token, sample_annos in iter_input_samples(data):
        normalized[sample_token] = normalize_sample_annotations(
            sample_annos, expected_num_points=expected_num_points
        )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        mmcv.mkdir_or_exist(output_dir)
    mmcv.dump({"results": normalized}, args.output)

    summary = summarize_annotations(normalized)
    if args.summary_path is not None:
        summary_dir = os.path.dirname(args.summary_path)
        if summary_dir:
            mmcv.mkdir_or_exist(summary_dir)
        mmcv.dump(summary, args.summary_path)
    print(summary)


if __name__ == "__main__":
    main()
