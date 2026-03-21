#!/usr/bin/env python
# coding=utf-8
"""
Script to plot DreamSim score threshold analysis.
Plots the ratio of sequences/images with score below threshold vs threshold value.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def load_dreamsim_json(file_path):
    """Load dreamsim scores from JSON file."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist, skipping...")
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def extract_sequence_averages(dreamsim_data):
    """
    Extract average dreamsim score for each sequence.
    
    Args:
        dreamsim_data: Dictionary with structure {folder_name: {sequence_subdir: {pairwise_scores: [...]}}}
    
    Returns:
        List of average dreamsim scores (one per sequence)
    """
    sequence_averages = []
    
    for folder_name, folder_results in dreamsim_data.items():
        for sequence_subdir, results in folder_results.items():
            pairwise_scores = results.get('pairwise_scores', [])
            if len(pairwise_scores) > 0:
                distances = [score['distance'] for score in pairwise_scores]
                avg_distance = np.mean(distances)
                sequence_averages.append(avg_distance)
    
    return sequence_averages


def extract_image_scores(dreamsim_data):
    """
    Extract all individual image dreamsim scores.
    
    Args:
        dreamsim_data: Dictionary with structure {folder_name: {sequence_subdir: {pairwise_scores: [...]}}}
    
    Returns:
        List of all dreamsim scores (one per image)
    """
    image_scores = []
    
    for folder_name, folder_results in dreamsim_data.items():
        for sequence_subdir, results in folder_results.items():
            pairwise_scores = results.get('pairwise_scores', [])
            for score in pairwise_scores:
                image_scores.append(score['distance'])
    
    return image_scores


def compute_threshold_ratios(scores, num_thresholds=1000):
    """
    Compute ratio of scores below threshold for each threshold value.
    
    Args:
        scores: List of dreamsim scores
        num_thresholds: Number of threshold points to evaluate
    
    Returns:
        thresholds: Array of threshold values (0 to 1)
        ratios: Array of ratios (proportion of scores <= threshold)
    """
    if len(scores) == 0:
        return np.array([]), np.array([])
    
    scores = np.array(scores)
    
    # Define threshold range from 0 to 1
    thresholds = np.linspace(0, 1, num_thresholds)
    
    # For each threshold, compute ratio of scores <= threshold
    ratios = []
    for threshold in thresholds:
        ratio = np.mean(scores <= threshold)
        ratios.append(ratio)
    
    ratios = np.array(ratios)
    
    return thresholds, ratios


def plot_threshold_analysis(json_files, mode='images', output_file='dreamsim_results/threshold_plot.png', 
                            num_thresholds=1000, base_dir='dreamsim_results'):
    """
    Plot threshold analysis for multiple JSON files.
    
    Args:
        json_files: List of JSON filenames (relative to base_dir or absolute paths)
        mode: 'sequences' or 'images'
        output_file: Output path for the plot
        num_thresholds: Number of threshold points to evaluate
        base_dir: Base directory for JSON files (if relative paths provided)
    """
    # Load all JSON files
    all_data = {}
    for json_file in json_files:
        # Construct full path
        if os.path.isabs(json_file):
            full_path = json_file
        else:
            full_path = os.path.join(base_dir, json_file)
        
        print(f"Loading {full_path}...")
        data = load_dreamsim_json(full_path)
        if data is not None:
            # Extract name from filename (without extension)
            name = os.path.splitext(os.path.basename(json_file))[0]
            all_data[name] = data
            print(f"  Loaded {name}: {len(data)} folders")
        else:
            print(f"  Failed to load {json_file}")
    
    if len(all_data) == 0:
        print("Error: No valid JSON files loaded")
        return
    
    # Extract scores based on mode
    print(f"\nExtracting scores in '{mode}' mode...")
    all_scores = {}
    
    for name, data in all_data.items():
        if mode == 'sequences':
            scores = extract_sequence_averages(data)
            print(f"  {name}: {len(scores)} sequences")
        else:  # mode == 'images'
            scores = extract_image_scores(data)
            print(f"  {name}: {len(scores)} images")
        
        if len(scores) > 0:
            all_scores[name] = scores
    
    if len(all_scores) == 0:
        print("Error: No scores extracted")
        return
    
    # Compute threshold ratios for each dataset
    print(f"\nComputing threshold ratios...")
    plot_data = {}
    
    # Specific thresholds to report
    report_thresholds = [0.2, 0.4]
    
    for name, scores in all_scores.items():
        thresholds, ratios = compute_threshold_ratios(scores, num_thresholds)
        plot_data[name] = {
            'thresholds': thresholds,
            'ratios': ratios,
            'num_items': len(scores),
            'scores': scores
        }
        print(f"  {name}: {len(scores)} items, min={np.min(scores):.4f}, max={np.max(scores):.4f}, mean={np.mean(scores):.4f}")
    
    # Print scores at specific thresholds
    print(f"\n{'='*80}")
    print("Threshold Analysis Results")
    print(f"{'='*80}")
    for name, data in plot_data.items():
        scores = np.array(data['scores'])
        print(f"\n{name}:")
        for thresh in report_thresholds:
            ratio = np.mean(scores <= thresh)
            print(f"  Threshold {thresh:.1f}: Ratio = {ratio:.4f} ({ratio*100:.2f}%)")
    print(f"{'='*80}")
    
    # Create plot
    print(f"\nGenerating plot...")
    plt.figure(figsize=(10, 6))
    
    for name, data in plot_data.items():
        plt.plot(data['thresholds'], data['ratios'], 
                label=f'{name} (n={data["num_items"]})', 
                linewidth=2)
    
    plt.xlabel('DreamSim Score Threshold', fontsize=12)
    
    if mode == 'sequences':
        plt.ylabel('Ratio of Sequences with Score ≤ Threshold', fontsize=12)
        title = 'CDF of Average DreamSim Scores per Sequence'
    else:
        plt.ylabel('Ratio of Images with Score ≤ Threshold', fontsize=12)
        title = 'CDF of DreamSim Scores per Image'
    
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot DreamSim score threshold analysis (CDF plot)"
    )
    parser.add_argument(
        "--json_files",
        type=str,
        nargs='+',
        required=True,
        help="List of JSON filenames from dreamsim_results directory (e.g., dreamsim_scores_eschernet_colmap.json)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['sequences', 'images'],
        default='sequences',
        help="Mode: 'sequences' for sequence-level averages, 'images' for individual image scores"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="dreamsim_results/threshold_plot_auc2.png",
        help="Output path for the plot"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="dreamsim_results",
        help="Base directory for JSON files (if relative paths provided)"
    )
    parser.add_argument(
        "--num_thresholds",
        type=int,
        default=1000,
        help="Number of threshold points to evaluate (default: 1000)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DreamSim Threshold Analysis Plot")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"JSON files: {args.json_files}")
    print(f"Output: {args.output_file}")
    print("=" * 80)
    
    plot_threshold_analysis(
        json_files=args.json_files,
        mode=args.mode,
        output_file=args.output_file,
        num_thresholds=args.num_thresholds,
        base_dir=args.base_dir
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

