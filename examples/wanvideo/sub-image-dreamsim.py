#!/usr/bin/env python
# coding=utf-8
"""
Compute sub-image (patch-based) DreamSim scores.
Splits each image into a grid of 224x224 tiles and computes DreamSim per tile,
then aggregates with a weighted combination of max and mean scores.
"""

import os
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import glob
import cv2

try:
    from dreamsim import dreamsim
    DREAMSIM_AVAILABLE = True
except ImportError:
    print("Error: dreamsim not available. Install with: pip install dreamsim")
    DREAMSIM_AVAILABLE = False
    raise


def compute_subimage_dreamsim_distance(model, preprocess, img1_path, img2_path, device,
                                        grid_size=2, tile_size=224, max_score_weight=0.2,
                                        batch_size=8):
    """
    Compute sub-image DreamSim distance between two images.
    Resizes both to (grid_size*tile_size) x (grid_size*tile_size),
    splits into tiles, computes per-tile DreamSim, and returns
    weighted_max = max_score_weight * max + (1-max_score_weight) * mean.
    """
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            print(f"Warning: Could not read {img1_path} or {img2_path}")
            return None

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        new_size = grid_size * tile_size
        img1 = cv2.resize(img1, (new_size, new_size), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (new_size, new_size), interpolation=cv2.INTER_AREA)

        def tile_image(img):
            tiles = []
            h, w = img.shape[:2]
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    tiles.append(img[y:y+tile_size, x:x+tile_size])
            return tiles

        tiles1 = tile_image(img1)
        tiles2 = tile_image(img2)

        scores = []
        for i in range(0, len(tiles1), batch_size):
            b1 = tiles1[i:i+batch_size]
            b2 = tiles2[i:i+batch_size]
            proc1 = torch.cat([preprocess(Image.fromarray(t)).to(device) for t in b1])
            proc2 = torch.cat([preprocess(Image.fromarray(t)).to(device) for t in b2])

            with torch.no_grad():
                batch_scores = model(proc1, proc2).cpu().numpy().flatten().tolist()
            scores.extend(batch_scores)

        max_score = np.max(scores)
        mean_score = np.mean(scores)
        aggregate = max_score_weight * max_score + (1 - max_score_weight) * mean_score
        return float(aggregate)

    except Exception as e:
        print(f"Error computing sub-image DreamSim for {img1_path} and {img2_path}: {e}")
        import traceback; traceback.print_exc()
        return None


def find_matching_images(gt_dir, pred_dir):
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']

    gt_images = []
    for ext in image_extensions:
        gt_images.extend(glob.glob(os.path.join(gt_dir, ext)))
    gt_images = sorted(gt_images)

    pred_images = []
    for ext in image_extensions:
        pred_images.extend(glob.glob(os.path.join(pred_dir, ext)))
    pred_images = sorted(pred_images)

    if len(gt_images) == 0 or len(pred_images) == 0:
        return []

    def get_base_name(path):
        return os.path.splitext(os.path.basename(path))[0]

    gt_dict = {get_base_name(img): img for img in gt_images}
    pred_dict = {get_base_name(img): img for img in pred_images}

    matching_pairs = []
    for filename_base in sorted(gt_dict.keys()):
        if filename_base in pred_dict:
            matching_pairs.append((filename_base, gt_dict[filename_base], pred_dict[filename_base]))
    return matching_pairs


def find_all_result_folders(base_dir):
    result_folders = []
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory {base_dir} does not exist")
        return result_folders

    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        gt_dir = os.path.join(folder_path, 'gt_target')
        pred_dir = os.path.join(folder_path, 'predicted_target')
        if os.path.isdir(gt_dir) and os.path.isdir(pred_dir):
            result_folders.append((folder_name, folder_path, gt_dir, pred_dir))
            continue

        for subdir in sorted(os.listdir(folder_path)):
            subdir_path = os.path.join(folder_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            gt_dir = os.path.join(subdir_path, 'gt_target')
            pred_dir = os.path.join(subdir_path, 'predicted_target')
            if os.path.isdir(gt_dir) and os.path.isdir(pred_dir):
                result_folders.append((folder_name, subdir_path, gt_dir, pred_dir))

    return result_folders


def main():
    parser = argparse.ArgumentParser(
        description="Compute sub-image (patch-based) DreamSim scores"
    )
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing result folders with gt_target/ and predicted_target/")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSON file to save results")
    parser.add_argument("--grid_size", type=int, default=2,
                        help="Grid size (NxN tiles of 224x224). Default: 2 -> 4 tiles")
    parser.add_argument("--max_score_weight", type=float, default=0.2,
                        help="Weight for max score in aggregation (default: 0.2)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading DreamSim model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=device)
    dreamsim_model.eval()
    print("DreamSim model loaded successfully")

    result_folders = find_all_result_folders(args.base_dir)
    if len(result_folders) == 0:
        print(f"Error: No result folders found in {args.base_dir}")
        return

    print(f"Found {len(result_folders)} result folders to process")
    print(f"Grid size: {args.grid_size}x{args.grid_size} = {args.grid_size**2} tiles per image")
    print(f"Aggregation: {args.max_score_weight:.1f}*max + {1-args.max_score_weight:.1f}*mean")

    all_results = {}

    for folder_name, sequence_dir, gt_dir, pred_dir in tqdm(result_folders, desc="Processing folders"):
        sequence_subdir = os.path.basename(sequence_dir)
        matching_pairs = find_matching_images(gt_dir, pred_dir)
        if len(matching_pairs) == 0:
            continue

        pairwise_scores = []
        for filename_base, gt_path, pred_path in matching_pairs:
            distance = compute_subimage_dreamsim_distance(
                dreamsim_model, dreamsim_preprocess,
                gt_path, pred_path, device,
                grid_size=args.grid_size,
                max_score_weight=args.max_score_weight
            )
            if distance is not None:
                pairwise_scores.append({
                    'filename': filename_base,
                    'gt_path': os.path.relpath(gt_path, args.base_dir),
                    'pred_path': os.path.relpath(pred_path, args.base_dir),
                    'distance': float(distance)
                })

        if len(pairwise_scores) == 0:
            continue

        distances = [s['distance'] for s in pairwise_scores]
        results = {
            'gt_dir': os.path.relpath(gt_dir, args.base_dir),
            'pred_dir': os.path.relpath(pred_dir, args.base_dir),
            'sequence_dir': os.path.relpath(sequence_dir, args.base_dir),
            'num_pairs': len(pairwise_scores),
            'pairwise_scores': pairwise_scores,
            'statistics': {
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances))
            }
        }

        if folder_name not in all_results:
            all_results[folder_name] = {}
        all_results[folder_name][sequence_subdir] = results

        if args.verbose:
            print(f"  {folder_name}/{sequence_subdir}: {len(pairwise_scores)} pairs, "
                  f"avg={np.mean(distances):.4f}")

    # Summary
    total_pairs = 0
    all_distances = []
    for folder_results in all_results.values():
        for results in folder_results.values():
            total_pairs += results['num_pairs']
            all_distances.extend([s['distance'] for s in results['pairwise_scores']])

    if len(all_distances) > 0:
        print(f"\nTotal image pairs: {total_pairs}")
        print(f"Overall average sub-image DreamSim: {np.mean(all_distances):.4f} +/- {np.std(all_distances):.4f}")
        print(f"Min: {np.min(all_distances):.4f}, Max: {np.max(all_distances):.4f}")

    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
