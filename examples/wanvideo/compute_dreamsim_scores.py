#!/usr/bin/env python
# coding=utf-8
"""
Script to compute DreamSim scores for pairs of GT and predicted images.
Compares images from two directories and outputs pairwise scores and averages.
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

# Import dreamsim for image similarity
try:
    from dreamsim import dreamsim
    DREAMSIM_AVAILABLE = True
except ImportError:
    print("Error: dreamsim not available. Install with: pip install dreamsim")
    DREAMSIM_AVAILABLE = False
    raise


def compute_dreamsim_distance(model, preprocess, img1_path, img2_path, device,
                              resize_pred_to_gt=False, original_gt_path=None):
    """
    Compute DreamSim distance between two images.

    Args:
        model: DreamSim model
        preprocess: DreamSim preprocessing function
        img1_path: Path to saved GT image (may be downsampled)
        img2_path: Path to predicted image
        device: torch device
        resize_pred_to_gt: If True, resize the predicted image to GT resolution
                           (using LANCZOS) before passing both to DreamSim.
        original_gt_path: When resize_pred_to_gt=True, use this higher-res GT image
                          as the reference for both sizing and DreamSim comparison.
                          Falls back to img1_path if not found.

    Returns:
        distance: DreamSim distance (lower is more similar)
    """
    try:
        # When resizing pred to GT, prefer the original high-res GT if provided
        if resize_pred_to_gt and original_gt_path and os.path.exists(original_gt_path):
            
            img1 = Image.open(original_gt_path).convert('RGB')
            print(f"Using original high-res GT: {original_gt_path}")
        else:
            img1 = Image.open(img1_path).convert('RGB')

        img2 = Image.open(img2_path).convert('RGB')

        if resize_pred_to_gt and img2.size != img1.size:
            img2 = img2.resize(img1.size, Image.LANCZOS)

        img1_tensor = preprocess(img1).to(device)
        img2_tensor = preprocess(img2).to(device)

        with torch.no_grad():
            distance = model(img1_tensor, img2_tensor)
            if isinstance(distance, torch.Tensor):
                distance = distance.item()

        return distance
    except Exception as e:
        print(f"Error computing DreamSim for {img1_path} and {img2_path}: {e}")
        return None


def find_matching_images(gt_dir, pred_dir):
    """
    Find matching image pairs between two directories by filename.
    
    Args:
        gt_dir: Path to GT images directory
        pred_dir: Path to predicted images directory
    
    Returns:
        matching_pairs: List of tuples (filename_base, gt_path, pred_path)
    """
    # Get all image files (support common formats)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    
    gt_images = []
    for ext in image_extensions:
        gt_images.extend(glob.glob(os.path.join(gt_dir, ext)))
    gt_images = sorted(gt_images)
    
    pred_images = []
    for ext in image_extensions:
        pred_images.extend(glob.glob(os.path.join(pred_dir, ext)))
    pred_images = sorted(pred_images)
    
    if len(gt_images) == 0:
        print(f"Warning: No images found in {gt_dir}")
        return []
    
    if len(pred_images) == 0:
        print(f"Warning: No images found in {pred_dir}")
        return []
    
    # Create dictionaries mapping base filename (without extension) to full path
    def get_base_name(path):
        """Get filename without extension."""
        return os.path.splitext(os.path.basename(path))[0]
    
    gt_dict = {get_base_name(img): img for img in gt_images}
    pred_dict = {get_base_name(img): img for img in pred_images}
    
    # Find matching pairs
    matching_pairs = []
    for filename_base in sorted(gt_dict.keys()):
        if filename_base in pred_dict:
            matching_pairs.append((filename_base, gt_dict[filename_base], pred_dict[filename_base]))
    
    return matching_pairs


def compute_scores_for_directories(gt_dir, pred_dir, model, preprocess, device,
                                   resize_pred_to_gt=False,
                                   original_gt_dir=None):
    """
    Compute DreamSim scores for all matching image pairs between two directories.

    Args:
        gt_dir: Path to saved GT images directory (may be downsampled)
        pred_dir: Path to predicted images directory
        model: DreamSim model
        preprocess: DreamSim preprocessing function
        device: torch device
        resize_pred_to_gt: If True, resize each predicted image to GT resolution
                           before computing DreamSim.
        original_gt_dir: When resize_pred_to_gt=True, load GT images from this
                         directory (original high-res) for both sizing and scoring.

    Returns:
        results: Dictionary with pairwise scores and statistics
    """
    # Check if directories exist
    if not os.path.isdir(gt_dir):
        print(f"Error: GT directory {gt_dir} does not exist")
        return None
    
    if not os.path.isdir(pred_dir):
        print(f"Error: Predicted directory {pred_dir} does not exist")
        return None
    
    # Find matching image pairs
    matching_pairs = find_matching_images(gt_dir, pred_dir)
    
    if len(matching_pairs) == 0:
        print(f"Warning: No matching image pairs found between {gt_dir} and {pred_dir}")
        return None
    
    print(f"Found {len(matching_pairs)} matching image pairs")
    print(f"Processing images from:")
    print(f"  GT: {gt_dir}")
    print(f"  Pred: {pred_dir}")
    
    # Compute scores for each pair
    pairwise_scores = []
    for filename_base, gt_path, pred_path in tqdm(matching_pairs, desc="Computing DreamSim scores"):
        # Resolve original high-res GT path if available (try common extensions)
        original_gt_path = None
        if resize_pred_to_gt and original_gt_dir:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                candidate = os.path.join(original_gt_dir, filename_base + ext)
                if os.path.exists(candidate):
                    original_gt_path = candidate
                    break

        distance = compute_dreamsim_distance(model, preprocess, gt_path, pred_path, device,
                                             resize_pred_to_gt=resize_pred_to_gt,
                                             original_gt_path=original_gt_path)
        
        if distance is not None:
            pairwise_scores.append({
                'filename': filename_base,
                'gt_path': gt_path,
                'pred_path': pred_path,
                'distance': float(distance)
            })
    
    if len(pairwise_scores) == 0:
        print(f"Warning: No valid scores computed")
        return None
    
    # Compute statistics
    distances = [s['distance'] for s in pairwise_scores]
    
    results = {
        'gt_dir': gt_dir,
        'pred_dir': pred_dir,
        'num_pairs': len(pairwise_scores),
        'pairwise_scores': pairwise_scores,
        'statistics': {
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances))
        }
    }
    
    return results


def find_all_result_folders(base_dir):
    """
    Find all result folders that contain gt_target/ and predicted_target/.

    Supports two layouts:
      1. Flat:   base_dir/{scene_name}/gt_target/  (WRIVA results)
      2. Nested: base_dir/{folder_name}/N{T_in}M{T_out}/gt_target/
    
    Args:
        base_dir: Base directory containing result folders
    
    Returns:
        List of tuples (folder_name, sequence_dir, gt_dir, pred_dir)
    """
    result_folders = []
    
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory {base_dir} does not exist")
        return result_folders
    
    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Layout 1 (flat): base_dir/{scene}/gt_target/
        gt_dir = os.path.join(folder_path, 'gt_target')
        pred_dir = os.path.join(folder_path, 'predicted_target')
        if os.path.isdir(gt_dir) and os.path.isdir(pred_dir):
            result_folders.append((folder_name, folder_path, gt_dir, pred_dir))
            continue
        
        # Layout 2 (nested): base_dir/{folder}/N{T_in}M{T_out}/gt_target/
        for subdir in sorted(os.listdir(folder_path)):
            subdir_path = os.path.join(folder_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            gt_dir = os.path.join(subdir_path, 'gt_target')
            pred_dir = os.path.join(subdir_path, 'predicted_target')
            if os.path.isdir(gt_dir) and os.path.isdir(pred_dir):
                result_folders.append((folder_name, subdir_path, gt_dir, pred_dir))
    
    return result_folders


def main(args):
    base_dir = args.base_dir
    output_file = args.output_file
    
    # Initialize DreamSim
    print("Loading DreamSim model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=device)
        dreamsim_model.eval()
        print("DreamSim model loaded successfully")
    except Exception as e:
        print(f"Error: Failed to load DreamSim model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if args.resize_pred_to_gt:
        if args.wriva_path:
            print(f"Mode: resize pred → original high-res GT ({args.wriva_path}/references/<scene>/)")
        else:
            print("Mode: resize pred → saved GT resolution before DreamSim")
    else:
        print("Mode: pass images as-is to DreamSim (default)")

    # Find all result folders
    print(f"\n{'='*80}")
    print("Finding all result folders")
    print(f"{'='*80}")
    result_folders = find_all_result_folders(base_dir)
    
    if len(result_folders) == 0:
        print(f"Error: No result folders found in {base_dir}")
        return
    
    print(f"Found {len(result_folders)} result folders to process")
    
    # Compute scores for each folder
    all_results = {}
    
    for folder_name, sequence_dir, gt_dir, pred_dir in tqdm(result_folders, desc="Processing folders"):
        # Extract N{T_in}M{T_out} from sequence_dir
        sequence_subdir = os.path.basename(sequence_dir)
        
        print(f"\n{'='*80}")
        print(f"Processing: {folder_name}/{sequence_subdir}")
        print(f"{'='*80}")
        
        # Derive original high-res GT dir from wriva_path + scene_name
        original_gt_dir = None
        if args.resize_pred_to_gt and args.wriva_path:
            original_gt_dir = os.path.join(args.wriva_path, "references", folder_name)
            if not os.path.isdir(original_gt_dir):
                print(f"  Warning: original GT dir not found at {original_gt_dir}, "
                      f"falling back to saved GT")
                original_gt_dir = None

        results = compute_scores_for_directories(gt_dir, pred_dir, dreamsim_model, dreamsim_preprocess, device,
                                                 resize_pred_to_gt=args.resize_pred_to_gt,
                                                 original_gt_dir=original_gt_dir)
        
        if results is None:
            print(f"Warning: Failed to compute scores for {folder_name}/{sequence_subdir}")
            continue
        
        # Store results in nested structure
        if folder_name not in all_results:
            all_results[folder_name] = {}
        
        # Convert paths to relative paths for storage
        relative_gt_dir = os.path.relpath(gt_dir, base_dir)
        relative_pred_dir = os.path.relpath(pred_dir, base_dir)
        relative_sequence_dir = os.path.relpath(sequence_dir, base_dir)
        
        # Update paths in results to be relative
        results['gt_dir'] = relative_gt_dir
        results['pred_dir'] = relative_pred_dir
        results['sequence_dir'] = relative_sequence_dir
        
        # Update pairwise scores paths to be relative
        for score in results['pairwise_scores']:
            score['gt_path'] = os.path.relpath(score['gt_path'], base_dir)
            score['pred_path'] = os.path.relpath(score['pred_path'], base_dir)
        
        all_results[folder_name][sequence_subdir] = results
        
        # Print summary for this folder
        stats = results['statistics']
        print(f"  Number of image pairs: {results['num_pairs']}")
        print(f"  Average distance: {stats['mean_distance']:.4f} ± {stats['std_distance']:.4f}")
        print(f"  Min distance: {stats['min_distance']:.4f}")
        print(f"  Max distance: {stats['max_distance']:.4f}")
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_pairs = 0
    all_distances = []
    
    for folder_name, folder_results in all_results.items():
        for sequence_subdir, results in folder_results.items():
            total_pairs += results['num_pairs']
            all_distances.extend([s['distance'] for s in results['pairwise_scores']])
    
    if len(all_distances) > 0:
        print(f"Total number of image pairs: {total_pairs}")
        print(f"Overall average distance: {np.mean(all_distances):.4f} ± {np.std(all_distances):.4f}")
        print(f"Overall min distance: {np.min(all_distances):.4f}")
        print(f"Overall max distance: {np.max(all_distances):.4f}")
    
    # Print per-folder summary if verbose
    if args.verbose:
        print(f"\nPer-folder summary:")
        for folder_name, folder_results in all_results.items():
            print(f"  {folder_name}:")
            for sequence_subdir, results in folder_results.items():
                stats = results['statistics']
                print(f"    {sequence_subdir}: {results['num_pairs']} pairs, avg={stats['mean_distance']:.4f}")
    
    # Save results to JSON
    if output_file:
        print(f"\n{'='*80}")
        print(f"Saving results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved successfully")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute DreamSim scores for GT and predicted image pairs in logs_6DoF_all-to-all"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/ocean/projects/cis250177p/qwu6/wriva_test_results_wan21_6to1_BlendedMVS-239",
        help="Base directory containing result folders (logs_6DoF_all-to-all)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/ocean/projects/cis250177p/qwu6/wriva_test_results_wan21_6to1_BlendedMVS-239/dreamsim_scores.json"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all pairwise scores and per-folder summary"
    )
    parser.add_argument(
        "--resize_pred_to_gt",
        action="store_true",
        help="Resize predicted image to GT resolution (LANCZOS) before DreamSim scoring."
    )
    parser.add_argument(
        "--wriva_path",
        type=str,
        default=None,
        help="Path to WRIVA dataset root (e.g. /ocean/projects/cis250200p/mjeon2/datasets/wriva). "
             "When set with --resize_pred_to_gt, loads the original high-res GT from "
             "<wriva_path>/references/<scene>/ instead of the saved (possibly downsampled) GT."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

