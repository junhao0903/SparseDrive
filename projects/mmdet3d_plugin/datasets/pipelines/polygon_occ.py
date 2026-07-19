from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, Polygon

from mmdet.datasets.builder import PIPELINES

from ..map_utils.polygon_occ_utils import (
    canonicalize_polygon,
    ensure_clockwise,
    normalize_polygon_start,
    sample_closed_polygon,
    sample_polygon_preserve_vertices,
)


@PIPELINES.register_module(force=True)
class VectorizePolygonOcc(object):
    def __init__(
        self,
        roi_size: Union[Tuple, List],
        normalize: bool,
        coords_dim: int = 2,
        sample_num: int = 32,
        closed_only: bool = True,
        permute: bool = False,
    ):
        self.roi_size = np.array(roi_size, dtype=np.float32)
        self.normalize = normalize
        self.coords_dim = coords_dim
        self.sample_num = sample_num
        self.closed_only = closed_only
        self.permute = permute

    def __call__(self, input_dict):
        geom_dict = input_dict.get("polygon_occ_geoms")
        if geom_dict is None:
            input_dict["gt_polygon_occ_labels"] = np.zeros((0,), dtype=np.int64)
            if self.permute:
                num_permute = 2 * self.sample_num
                input_dict["gt_polygon_occ_pts"] = np.zeros(
                    (0, num_permute, self.sample_num, self.coords_dim),
                    dtype=np.float32,
                )
            else:
                input_dict["gt_polygon_occ_pts"] = np.zeros(
                    (0, self.sample_num, self.coords_dim), dtype=np.float32
                )
            return input_dict

        gt_labels, gt_pts = [], []
        for label, geom_list in geom_dict.items():
            for geom in geom_list:
                pts = self.to_polygon_points(geom)
                if pts is None:
                    continue
                if self.normalize:
                    pts = self.normalize_pts(pts)
                gt_labels.append(label)
                if self.permute:
                    pts = self.permute_polygon(pts)
                gt_pts.append(pts.astype(np.float32))

        if gt_pts:
            input_dict["gt_polygon_occ_labels"] = np.asarray(
                gt_labels, dtype=np.int64
            )
            input_dict["gt_polygon_occ_pts"] = np.stack(gt_pts, axis=0)
        else:
            input_dict["gt_polygon_occ_labels"] = np.zeros((0,), dtype=np.int64)
            if self.permute:
                num_permute = 2 * self.sample_num
                input_dict["gt_polygon_occ_pts"] = np.zeros(
                    (0, num_permute, self.sample_num, self.coords_dim),
                    dtype=np.float32,
                )
            else:
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

    def to_polygon_points(self, geom):
        if isinstance(geom, np.ndarray):
            return self.process_array_polygon(geom)

        polygon = self.to_polygon(geom)
        if polygon is None:
            return None
        return self.sample_polygon(polygon)

    def process_array_polygon(self, coords: NDArray) -> NDArray:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != self.coords_dim or len(coords) < 3:
            return None

        polygon = Polygon(coords)
        if polygon.is_empty or polygon.area <= 0:
            return None
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 0 or not polygon.is_valid:
                return None

        if len(coords) == self.sample_num:
            return self.canonicalize_sampled_polygon(coords)
        return self.sample_polygon(polygon)

    def sample_polygon(self, polygon: Polygon) -> NDArray:
        points = np.asarray(polygon.exterior.coords, dtype=np.float32)
        sampled = sample_polygon_preserve_vertices(points, num_points=self.sample_num)
        sampled = sampled[:, : self.coords_dim]
        sampled = self.canonicalize_sampled_polygon(sampled)
        if len(sampled) != self.sample_num:
            raise ValueError(
                f"sample_polygon expected {self.sample_num} points, got {len(sampled)}"
            )
        return sampled

    def canonicalize_sampled_polygon(self, pts: NDArray) -> NDArray:
        """Canonicalize a fixed-length sampled polygon while preserving length.

        Unlike canonicalize_polygon(), this helper must not drop any point,
        because downstream target tensors require an exact fixed sample count.
        """
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) == 0:
            return pts
        pts = ensure_clockwise(pts)
        pts = normalize_polygon_start(pts)
        return pts

    def normalize_pts(self, pts: NDArray) -> NDArray:
        origin = -np.array([self.roi_size[0] / 2, self.roi_size[1] / 2])
        pts = pts.copy()
        pts[:, :2] = pts[:, :2] - origin
        pts[:, :2] = pts[:, :2] / (self.roi_size + 1e-5)
        return pts

    def permute_polygon(self, pts: NDArray, padding=1e5) -> NDArray:
        """Generate cyclic + reversed cyclic polygon candidates.

        Polygon OCC stores a closed shape using `sample_num` unique perimeter
        points without repeating the first point at the end. Therefore the right
        permutation set is all `sample_num` cyclic shifts plus all reversed
        cyclic shifts, i.e. `2 * sample_num` candidates.
        """
        num_points = len(pts)
        permute_num = num_points
        permute_polygons = []

        for shift_i in range(permute_num):
            permute_polygons.append(np.roll(pts, shift_i, axis=0))

        flipped = np.flip(pts, axis=0)
        for shift_i in range(permute_num):
            permute_polygons.append(np.roll(flipped, shift_i, axis=0))

        return np.stack(permute_polygons, axis=0).astype(np.float32)
