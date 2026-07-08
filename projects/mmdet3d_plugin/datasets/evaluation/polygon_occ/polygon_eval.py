from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import mmcv
import prettytable
from mmcv import Config
from mmcv.utils import print_log
from mmdet.datasets import build_dataset

from projects.mmdet3d_plugin.datasets.map_utils.polygon_occ_utils import (
    polygon_validity_stats,
)


class PolygonOccEvaluate(object):
    def __init__(self, dataset_cfg: Config) -> None:
        self.dataset = build_dataset(dataset_cfg)

    def _collect_gt(self) -> Dict[str, Dict[int, List[np.ndarray]]]:
        gts = {}
        for info in self.dataset.data_infos:
            token = info["token"]
            annos = info.get("polygon_occ_annos", {})
            sample = {}
            for label, polygons in annos.items():
                sample[int(label)] = [np.asarray(poly, dtype=np.float32) for poly in polygons]
            gts[token] = sample
        return gts

    def _summarize_sample(self, polygons: List[np.ndarray]) -> Dict[str, float]:
        if not polygons:
            return {
                "count": 0,
                "valid_ratio": 0.0,
                "mean_area": 0.0,
            }
        valid = 0
        areas = []
        for polygon in polygons:
            stats = polygon_validity_stats(np.asarray(polygon, dtype=np.float32))
            valid += int(stats["is_valid"] > 0)
            areas.append(stats["area"])
        return {
            "count": len(polygons),
            "valid_ratio": valid / max(len(polygons), 1),
            "mean_area": float(np.mean(areas)) if areas else 0.0,
        }

    def evaluate(self, result_path: str, logger: Optional[object] = None) -> Dict[str, float]:
        results = mmcv.load(result_path)["results"]
        gts = self._collect_gt()

        pred_counts, gt_counts = [], []
        pred_valid_ratios, gt_valid_ratios = [], []
        pred_areas, gt_areas = [], []
        covered_tokens = 0
        for token, gt in gts.items():
            pred = deepcopy(results.get(token, {"polygons": [], "scores": [], "labels": []}))
            pred_polygons = pred.get("polygons", pred.get("vectors", []))
            gt_polygons = []
            for polygons in gt.values():
                gt_polygons.extend(polygons)

            pred_summary = self._summarize_sample(pred_polygons)
            gt_summary = self._summarize_sample(gt_polygons)
            pred_counts.append(pred_summary["count"])
            gt_counts.append(gt_summary["count"])
            pred_valid_ratios.append(pred_summary["valid_ratio"])
            gt_valid_ratios.append(gt_summary["valid_ratio"])
            pred_areas.append(pred_summary["mean_area"])
            gt_areas.append(gt_summary["mean_area"])
            if pred_summary["count"] > 0:
                covered_tokens += 1

        result_dict = {
            "occ_num_samples": float(len(gts)),
            "occ_gt_mean_polygons": float(np.mean(gt_counts)) if gt_counts else 0.0,
            "occ_pred_mean_polygons": float(np.mean(pred_counts)) if pred_counts else 0.0,
            "occ_gt_valid_ratio": float(np.mean(gt_valid_ratios)) if gt_valid_ratios else 0.0,
            "occ_pred_valid_ratio": float(np.mean(pred_valid_ratios)) if pred_valid_ratios else 0.0,
            "occ_gt_mean_area": float(np.mean(gt_areas)) if gt_areas else 0.0,
            "occ_pred_mean_area": float(np.mean(pred_areas)) if pred_areas else 0.0,
            "occ_pred_coverage": covered_tokens / max(len(gts), 1),
        }

        table = prettytable.PrettyTable(["metric", "value"])
        for key, value in result_dict.items():
            table.add_row([key, round(value, 4)])
        print_log("\n" + str(table), logger=logger)
        return result_dict
