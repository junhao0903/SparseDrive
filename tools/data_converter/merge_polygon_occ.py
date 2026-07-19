import argparse
import os
from collections import OrderedDict

import mmcv


def load_results(path):
    """Load token-keyed polygon occ annotations from a pkl/json-style file."""
    data = mmcv.load(path)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unsupported polygon occ file format: {path}")


def main():
    """CLI entry point for merging split polygon occ sidecars.

    This is mainly useful when train/val are generated separately but a single
    token-keyed file is desired for debugging or legacy single-file workflows.
    """
    parser = argparse.ArgumentParser(description="Merge token-keyed polygon occ annotation files")
    parser.add_argument("inputs", nargs="+", help="input polygon occ files")
    parser.add_argument("--output", required=True, help="merged output path")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="allow later inputs to overwrite duplicate sample tokens",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="optional summary output path",
    )
    args = parser.parse_args()

    merged = OrderedDict()
    duplicate_tokens = []
    counts = OrderedDict()
    for path in args.inputs:
        results = load_results(path)
        counts[path] = len(results)
        for token, annos in results.items():
            # Duplicate handling is explicit because token overlap between train
            # and val usually signals a dataset split or generation mistake.
            if token in merged and not args.allow_overwrite:
                duplicate_tokens.append(token)
                continue
            merged[token] = annos

    if duplicate_tokens and not args.allow_overwrite:
        raise ValueError(
            f"Found duplicate tokens across inputs; rerun with --allow-overwrite. "
            f"Examples: {duplicate_tokens[:10]}"
        )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        mmcv.mkdir_or_exist(output_dir)
    mmcv.dump({"results": merged}, args.output)

    summary = {
        "num_inputs": len(args.inputs),
        "input_counts": dict(counts),
        "num_merged_tokens": len(merged),
        "num_duplicate_tokens": len(duplicate_tokens),
        "duplicate_examples": duplicate_tokens[:20],
    }
    if args.summary_path is not None:
        summary_dir = os.path.dirname(args.summary_path)
        if summary_dir:
            mmcv.mkdir_or_exist(summary_dir)
        mmcv.dump(summary, args.summary_path)
    print(summary)


if __name__ == "__main__":
    main()
