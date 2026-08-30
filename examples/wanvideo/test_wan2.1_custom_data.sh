#!/bin/bash

# Inference on custom test data in /data2/qiwu2/test_data
# Uses first 5 images as input, 6th as target.
# Poses are OpenCV c2w, per-image intrinsics.

MODE="${1:-zero_temporal_rope}"
GPU_ID="${2:-2}"

# ── Configuration ──────────────────────────────────────────────────────────────
if [ "$MODE" = "zero_temporal_rope" ]; then
    checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope-480p/epoch-59.safetensors"
    output_path="/data2/qiwu2/custom_test_results_wan21_6to1_zero-temporal-rope-480p_epoch-59"
    EXTRA_ARGS="--zero_temporal_rope"
elif [ "$MODE" = "standard" ]; then
    checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_sorted_context/epoch-79.safetensors"
    output_path="/data2/qiwu2/custom_test_results_sorted_context_79"
    EXTRA_ARGS=""
else
    echo "Unknown mode: $MODE (use zero_temporal_rope or standard)"
    exit 1
fi

DATA_PATH="/data2/qiwu2/test_data"

echo "========================================"
echo "Custom Data Inference"
echo "========================================"
echo "Mode:       $MODE"
echo "Checkpoint: $checkpoint_path"
echo "Data path:  $DATA_PATH"
echo "Output:     $output_path"
echo "GPU:        $GPU_ID"
echo ""

mkdir -p "$output_path"

CUDA_VISIBLE_DEVICES=$GPU_ID python test_wan2.1_custom_data.py \
    --checkpoint_path "$checkpoint_path" \
    --data_path "$DATA_PATH" \
    --output_path "$output_path" \
    --height 480 \
    --width 832 \
    --num_inference_steps 50 \
    $EXTRA_ARGS

echo ""
echo "Done! Results in $output_path"
