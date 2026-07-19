import argparse
import os
from collections import OrderedDict

import mmcv


def build_occ_index(occ_root):
    """Index Occ3D labels by sample token.

    Expected on-disk layout:
        occ_root/scene-xxxx/<sample_token>/labels.npz

    Returns a flat token -> metadata mapping so later stages can ignore the
    scene folder structure and join directly against info pkls by token.
    """
    index = {}
    for scene_name in sorted(os.listdir(occ_root)):
        scene_dir = os.path.join(occ_root, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        for sample_token in os.listdir(scene_dir):
            sample_dir = os.path.join(scene_dir, sample_token)
            label_path = os.path.join(sample_dir, "labels.npz")
            if os.path.isfile(label_path):
                index[sample_token] = {
                    "scene_name": scene_name,
                    "data_path": label_path,
                }
    return index


def build_manifest(info_path, occ_index, split_name=None):
    """Create a lightweight manifest that points each info sample to its Occ3D file.

    The manifest is intentionally small: only the sample token, Occ3D path,
    scene name, timestamp, and optional split tag are preserved. This is the
    format consumed by occ_to_polygon_occ.py.
    """
    data = mmcv.load(info_path)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data
    samples = []
    missing = []
    for info in infos:
        token = info.get("token")
        if token is None:
            continue
        occ_item = occ_index.get(token)
        if occ_item is None:
            missing.append(token)
            continue
        sample = OrderedDict(
            sample_token=token,
            data_path=occ_item["data_path"],
            scene_name=occ_item["scene_name"],
            timestamp=info.get("timestamp"),
        )
        # Keep the object GT needed for hybrid thing-class polygon generation.
        # gt_boxes is stored as a plain list so the manifest remains easy to
        # inspect and serializable without numpy-specific assumptions.
        if "gt_boxes" in info:
            sample["gt_boxes"] = info["gt_boxes"].tolist()
        if "gt_names" in info:
            sample["gt_names"] = [str(x) for x in info["gt_names"]]
        if "valid_flag" in info:
            sample["valid_flag"] = [bool(x) for x in info["valid_flag"]]
        elif "num_lidar_pts" in info:
            sample["valid_flag"] = [int(x) > 0 for x in info["num_lidar_pts"]]
        if "lidar2ego_translation" in info:
            sample["lidar2ego_translation"] = list(info["lidar2ego_translation"])
        if "lidar2ego_rotation" in info:
            sample["lidar2ego_rotation"] = list(info["lidar2ego_rotation"])
        if split_name is not None:
            sample["split"] = split_name
        samples.append(sample)
    return {
        "samples": samples,
        "num_samples": len(samples),
        "num_missing": len(missing),
        "missing_tokens": missing,
    }


def main():
    """CLI entry point for building train/val-specific Occ3D manifests."""
    parser = argparse.ArgumentParser(description="Build Occ3D manifest from info pkl and occ root")
    parser.add_argument("--info", type=str, required=True, help="input info pkl")
    parser.add_argument("--occ-root", type=str, required=True, help="occ3d root directory")
    parser.add_argument("--output", type=str, required=True, help="output manifest path")
    parser.add_argument("--split", type=str, default=None, help="optional split name stored in manifest")
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="optional summary output path",
    )
    args = parser.parse_args()

    occ_index = build_occ_index(args.occ_root)
    manifest = build_manifest(args.info, occ_index, split_name=args.split)

    # Persist only the samples list for downstream converters. The richer
    # summary is saved separately so the manifest itself stays simple.
    output_dir = os.path.dirname(args.output)
    if output_dir:
        mmcv.mkdir_or_exist(output_dir)
    mmcv.dump({"samples": manifest["samples"]}, args.output)

    summary = {
        "info_path": args.info,
        "occ_root": args.occ_root,
        "num_samples": manifest["num_samples"],
        "num_missing": manifest["num_missing"],
        "missing_tokens": manifest["missing_tokens"],
    }
    if args.summary_path is not None:
        summary_dir = os.path.dirname(args.summary_path)
        if summary_dir:
            mmcv.mkdir_or_exist(summary_dir)
        mmcv.dump(summary, args.summary_path)
    print(summary)


if __name__ == "__main__":
    main()
