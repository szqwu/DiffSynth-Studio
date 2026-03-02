"""
Standalone script to (re)generate permutation invariance test summaries
from the per-scene permutation_metrics.json files.

Usage:
    python regenerate_permutation_summary.py /path/to/result_dir [/path/to/another_dir ...]

If no dirs given, processes both default permtest directories.
"""
import os
import sys
import json
import glob
import numpy as np


METRIC_KEYS = ["psnr", "ssim", "lpips", "dreamsim"]


def process_result_dir(result_dir):
    json_files = sorted(glob.glob(os.path.join(result_dir, "*/permutation_metrics.json")))
    if not json_files:
        print(f"  No permutation_metrics.json found in {result_dir}")
        return

    print(f"\nProcessing {result_dir}")
    print(f"  Found {len(json_files)} scene(s)")

    all_scenes = []
    permutations = None

    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)

        scene_hash = data["scene"]
        if permutations is None:
            permutations = data["permutations"]

        available_keys = []
        per_frame = data["per_frame"]
        if per_frame:
            first_frame = next(iter(per_frame.values()))
            available_keys = [k for k in METRIC_KEYS if k in first_frame["mean"]]

        scene_means = {k: [] for k in available_keys}
        scene_stds = {k: [] for k in available_keys}

        for frame_id, frame_data in per_frame.items():
            for k in available_keys:
                scene_means[k].append(frame_data["mean"][k])
                scene_stds[k].append(frame_data["std"][k])

        scene_result = {
            "scene": scene_hash,
            "num_frames": len(per_frame),
            "test_ids": data.get("test_ids_subsampled", []),
        }
        for k in available_keys:
            scene_result[f"{k}_mean"] = np.mean(scene_means[k])
            scene_result[f"{k}_perm_std"] = np.mean(scene_stds[k])

        all_scenes.append(scene_result)

    if not all_scenes:
        return

    available_keys = [k for k in METRIC_KEYS if f"{k}_mean" in all_scenes[0]]

    # Compute overall aggregates
    overall = {}
    for k in available_keys:
        means = [s[f"{k}_mean"] for s in all_scenes]
        stds = [s[f"{k}_perm_std"] for s in all_scenes]
        overall[f"{k}_mean"] = np.mean(means)
        overall[f"{k}_perm_std"] = np.mean(stds)

    # Write summary
    summary_path = os.path.join(result_dir, "permutation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Permutation Invariance Test Summary\n")
        f.write(f"Seed: {all_scenes[0].get('seed', 42)}\n")
        f.write(f"Num permutations: {len(permutations)}\n")
        f.write(f"Scenes: {len(all_scenes)}\n\n")

        f.write("Permutations:\n")
        for pi, p in enumerate(permutations):
            f.write(f"  Perm {pi}: {p}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("Overall (averaged across scenes):\n")
        f.write("=" * 60 + "\n")
        for k in available_keys:
            f.write(f"  {k.upper():>10s}: mean={overall[f'{k}_mean']:.4f}, "
                    f"perm_std={overall[f'{k}_perm_std']:.4f}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("Per-scene:\n")
        f.write("=" * 60 + "\n")
        for s in all_scenes:
            f.write(f"\n  {s['scene'][:8]}... ({s['num_frames']} test frames):\n")
            for k in available_keys:
                f.write(f"    {k.upper():>10s}: mean={s[f'{k}_mean']:.4f}, "
                        f"perm_std={s[f'{k}_perm_std']:.4f}\n")

    print(f"  Summary written to {summary_path}")

    # Print to console
    print(f"\n  {'='*60}")
    print(f"  Overall ({len(all_scenes)} scenes, {len(permutations)} permutations):")
    print(f"  {'='*60}")
    for k in available_keys:
        print(f"    {k.upper():>10s}: mean={overall[f'{k}_mean']:.4f}, "
              f"perm_std={overall[f'{k}_perm_std']:.4f}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        dirs = [
            "/data2/qiwu2/dl3dv_test_results_wan21_6to1_random_79_permtest_permnoise",
            "/data2/qiwu2/dl3dv_test_results_wan21_6to1_zero-temporal-rope_79_permtest_permnoise",
        ]

    for d in dirs:
        if os.path.isdir(d):
            process_result_dir(d)
        else:
            print(f"Directory not found: {d}")
