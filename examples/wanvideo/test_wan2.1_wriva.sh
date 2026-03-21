#!/bin/bash

# Script to run Wan2.1 N-to-M NVS evaluation on WRIVA dataset
# 9-to-1 with zero temporal rope

# ── Configuration ──────────────────────────────────────────────────────────────
checkpoint_path="/ocean/projects/cis250177p/qwu6/epoch-79.safetensors"
output_path="/ocean/projects/cis250177p/qwu6/wriva_test_results_wan21_9to1_zero_rope_crop_all"
GPU_ID=0

echo "========================================"
echo "Wan2.1 9-to-1 NVS — WRIVA Evaluation"
echo "========================================"
echo "Checkpoint: $checkpoint_path"
echo "Output path: $output_path"
echo "GPU: $GPU_ID"
echo ""

# Create output directory
mkdir -p "$output_path"

CUDA_VISIBLE_DEVICES=$GPU_ID python test_wan2.1_wriva.py \
    --checkpoint_path $checkpoint_path \
    --output_path $output_path \
    --wriva_path /ocean/projects/cis250200p/mjeon2/datasets/wriva \
    --num_scenes 500 \
    --num_ref 9 \
    --num_targets 1 \
    --height 192 \
    --width 336 \
    --num_inference_steps 50 \
    --use_dreamsim \
    --use_ssim \
    --use_lpips \
    --seed 42 \
    --resize_mode crop \
    --zero_temporal_rope

echo ""
echo "Done!"

