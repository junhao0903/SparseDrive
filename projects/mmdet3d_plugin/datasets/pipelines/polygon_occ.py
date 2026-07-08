from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, Polygon

from mmdet.datasets.builder import PIPELINES

from ..map_utils.polygon_occ_utils import canonicalize_polygon, sample_closed_polygon


@PIPELINES.register_module(force=True)
class VectorizePolygonOcc(object):
    def __init__(
        self,
        roi_size: Union[Tuple, List],
        normalize: bool,
        coords_dim: int = 2,
        sample_num: int = 32,
        closed_only: bool = True,
    ):
        self.roi_size = np.array(roi_size, dtype=np.float32)
        self.normalize = normalize
        self.coords_dim = coords_dim
        self.sample_num = sample_num
        self.closed_only = closed_only

    def __call__(self, input_dict):
        geom_dict = input_dict.get("polygon_occ_geoms")
        if geom_dict is None:
            geom_dict = input_dict.get("map_geoms")
        if geom_dict is None:
            return input_dict

        gt_labels, gt_pts = [], []
        for label, geom_list in geom_dict.items():
            for geom in geom_list:
                polygon = self.to_polygon(geom)
                if polygon is None:
                    continue
                pts = self.sample_polygon(polygon)
                if self.normalize:
                    pts = self.normalize_pts(pts)
                gt_labels.append(label)
                gt_pts.append(pts.astype(np.float32))

        if gt_pts:
            input_dict["gt_polygon_occ_labels"] = np.asarray(
                gt_labels, dtype=np.int64
            )
            input_dict["gt_polygon_occ_pts"] = np.stack(gt_pts, axis=0)
        else:
            input_dict["gt_polygon_occ_labels"] = np.zeros((0,), dtype=np.int64)
            input_dict["gt_polygon_occ_pts"] = np.zeros(
                (0, self.sample_num, self.coords_dim), dtype=np.float32
            )
        return input_dict

    def to_polygon(self, geom):
        if isinstance(geom, Polygon):
            return geom
        if not isinstance(geom, LineString):
            return None

        coords = np.asarray(geom.coords, dtype=np.float32)
        if len(coords) < 3:
            return None
        is_closed = np.allclose(coords[0], coords[-1], atol=1e-3)
        if not is_closed:
            if self.closed_only:
                return None
            coords = np.concatenate([coords, coords[:1]], axis=0)
        polygon = Polygon(coords)
        if polygon.is_empty or polygon.area <= 0:
            return None
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 0 or not polygon.is_valid:
                return None
        return polygon

    def sample_polygon(self, polygon: Polygon) -> NDArray:
        points = np.asarray(polygon.exterior.coords, dtype=np.float32)
        sampled = sample_closed_polygon(points, num_points=self.sample_num)
        sampled = sampled[:, : self.coords_dim]
        return canonicalize_polygon(sampled)

    def normalize_pts(self, pts: NDArray) -> NDArray:
        origin = -np.array([self.roi_size[0] / 2, self.roi_size[1] / 2])
        pts = pts.copy()
        pts[:, :2] = pts[:, :2] - origin
        pts[:, :2] = pts[:, :2] / (self.roi_size + 1e-5)
        return pts
