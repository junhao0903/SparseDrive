import argparse
import os

import mmcv
import numpy as np
from sklearn.cluster import KMeans


def collect_polygons(results):
    polygons = []
    centers = []
    for sample_annos in results.values():
        for anno_list in sample_annos.values():
            for polygon in anno_list:
                polygon = np.asarray(polygon, dtype=np.float32)
                if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                    continue
                polygons.append(polygon)
                centers.append(polygon.mean(axis=0))
    if not polygons:
        raise ValueError("No valid Polygon OCC polygons found for anchor generation")
    return polygons, np.stack(centers, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Generate Polygon OCC anchors from sidecar polygons")
    parser.add_argument(
        "--input",
        type=str,
        default="data/polygon_occ_infos/polygon_occ_train.pkl",
        help="token-keyed Polygon OCC training sidecar",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/kmeans/kmeans_polygon_occ_100_20.npy",
        help="output .npy anchor file",
    )
    parser.add_argument("--num-anchor", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    data = mmcv.load(args.input)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, dict):
        raise ValueError("Polygon OCC sidecar must be a dict keyed by sample token")

    polygons, centers = collect_polygons(data)
    if len(polygons) < args.num_anchor:
        raise ValueError(
            f"Need at least {args.num_anchor} polygons to build anchors, got {len(polygons)}"
        )

    cluster_centers = KMeans(
        n_clusters=args.num_anchor,
        random_state=args.random_state,
        n_init=10,
    ).fit(centers).cluster_centers_

    anchors = []
    for center in cluster_centers:
        distances = np.linalg.norm(centers - center[None], axis=1)
        anchors.append(polygons[int(np.argmin(distances))])

    anchors = np.stack(anchors, axis=0).astype(np.float32)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        mmcv.mkdir_or_exist(output_dir)
    np.save(args.output, anchors)
    print(f"saved {anchors.shape} anchors to {args.output}")


if __name__ == "__main__":
    main()
