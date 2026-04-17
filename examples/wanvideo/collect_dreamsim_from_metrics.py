#!/usr/bin/env python
"""
Collect per-scene DreamSim scores from metrics.txt files written during inference.
Outputs a JSON file in the same format as compute_dreamsim_scores.py, so it can
be used directly with plot_dreamsim_threshold.py.

Supports both flat (base_dir/{scene}/metrics.txt) and sharded
(base_dir/shard_*/{scene}/metrics.txt) layouts.
"""

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path


def parse_metrics_file(metrics_path):
    """Parse a metrics.txt file and return per-frame DreamSim scores.

    Returns:
        list of dicts with 'filename' and 'distance', or None on failure.
    """
    scores = []
    per_frame = False
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("Per-frame metrics:"):
                per_frame = True
                continue
            if per_frame and line.startswith("  "):
                m = re.match(r"\s+(\S+):", line)
                ds = re.search(r"DreamSim=([\d.]+)", line)
                if m and ds:
                    scores.append({
                        "filename": m.group(1),
                        "distance": float(ds.group(1)),
                    })
    return scores if scores else None


def collect_from_directory(base_dir):
    """Walk base_dir for metrics.txt files (flat or shard layout).

    Returns:
        dict matching compute_dreamsim_scores.py JSON structure:
        {group_name: {scene_name: {pairwise_scores, statistics, ...}}}
    """
    all_results = {}

    # Discover scene directories (possibly inside shard_* subdirs)
    scene_dirs = []
    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        metrics = os.path.join(entry_path, "metrics.txt")
        if os.path.isfile(metrics):
            scene_dirs.append(("root", entry, entry_path))
        else:
            # Check for shard or other grouping directories
            for sub in sorted(os.listdir(entry_path)):
                sub_path = os.path.join(entry_path, sub)
                if os.path.isdir(sub_path) and os.path.isfile(
                    os.path.join(sub_path, "metrics.txt")
                ):
                    scene_dirs.append((entry, sub, sub_path))

    print(f"Found {len(scene_dirs)} scene directories with metrics.txt")

    for group, scene_name, scene_path in scene_dirs:
        metrics_path = os.path.join(scene_path, "metrics.txt")
        scores = parse_metrics_file(metrics_path)
        if scores is None:
            continue

        distances = [s["distance"] for s in scores]

        gt_dir = os.path.relpath(
            os.path.join(scene_path, "gt_target"), base_dir
        )
        pred_dir = os.path.relpath(
            os.path.join(scene_path, "predicted_target"), base_dir
        )

        pairwise_scores = [
            {
                "filename": s["filename"],
                "gt_path": os.path.join(gt_dir, s["filename"] + ".png"),
                "pred_path": os.path.join(pred_dir, s["filename"] + ".png"),
                "distance": s["distance"],
            }
            for s in scores
        ]

        entry = {
            "gt_dir": gt_dir,
            "pred_dir": pred_dir,
            "num_pairs": len(pairwise_scores),
            "pairwise_scores": pairwise_scores,
            "statistics": {
                "mean_distance": float(np.mean(distances)),
                "std_distance": float(np.std(distances)),
                "min_distance": float(np.min(distances)),
                "max_distance": float(np.max(distances)),
            },
            "sequence_dir": os.path.relpath(scene_path, base_dir),
        }

        if group not in all_results:
            all_results[group] = {}
        all_results[group][scene_name] = entry

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Collect DreamSim scores from metrics.txt into "
        "compute_dreamsim_scores.py-compatible JSON"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Result directory containing scene folders (or shard_* subdirs)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON path (default: base_dir/dreamsim_scores.json)",
    )
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = os.path.join(args.base_dir, "dreamsim_scores.json")

    print(f"Scanning {args.base_dir} ...")
    results = collect_from_directory(args.base_dir)

    # Summary
    total_scenes = sum(len(v) for v in results.values())
    total_images = sum(
        entry["num_pairs"]
        for group in results.values()
        for entry in group.values()
    )
    all_distances = [
        s["distance"]
        for group in results.values()
        for entry in group.values()
        for s in entry["pairwise_scores"]
    ]

    print(f"\nTotal: {total_scenes} scenes, {total_images} images")
    if all_distances:
        print(
            f"Overall DreamSim: {np.mean(all_distances):.4f} "
            f"± {np.std(all_distances):.4f}"
        )
        print(f"  min={np.min(all_distances):.4f}, max={np.max(all_distances):.4f}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_file}")


if __name__ == "__main__":
    main()
