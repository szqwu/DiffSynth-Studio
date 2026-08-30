#!/bin/bash

# Wan2.1 6-to-1 NVS uncertainty sampling on one DL3DV-10K scene.
# For each target frame: K samples × few steps, save per-pixel std map,
# mean image, raw samples, sample grid, and pairwise-LPIPS diversity.
# No GT is involved.

# ── Configuration ───────────────────────────────────────────────────────────
GPU_ID="${1:-6}"
NUM_INPUT_FRAMES="${2:-6}"
NUM_OUTPUT_FRAMES="${3:-1}"
NUM_SAMPLES="${4:-10}"
NUM_STEPS="${5:-5}"
SEED_BASE="${6:-0}"

checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-zero-temporal-rope_T10_480p/epoch-29.safetensors"
output_path="/data2/qiwu2/dl3dv_uncertainty_wan21_${NUM_INPUT_FRAMES}to${NUM_OUTPUT_FRAMES}_zero-temporal-rope_T10_480p_epoch-29_K${NUM_SAMPLES}_T${NUM_STEPS}"

# SCENE="9e9a89ae6fed06d6e2f4749b4b0059f35ca97f848cedc4a14345999e746f7884"
SCENE="cd9c981eeb4a9091547af19181b382698e9d9eee0a838c7c9783a8a268af6aee"

echo "========================================"
echo "Wan2.1 ${NUM_INPUT_FRAMES}-to-${NUM_OUTPUT_FRAMES} Uncertainty Sampling"
echo "========================================"
echo "Checkpoint: $checkpoint_path"
echo "Output:     $output_path"
echo "GPU:        $GPU_ID"
echo "Scene:      $SCENE"
echo "Samples:    K=$NUM_SAMPLES   (seed_base=$SEED_BASE)"
echo "Steps:      $NUM_STEPS"
echo ""

mkdir -p "$output_path"

CUDA_VISIBLE_DEVICES=$GPU_ID python test_wan2.1_6to1_uncertainty.py \
    --checkpoint_path "$checkpoint_path" \
    --output_path "$output_path" \
    --scenes "$SCENE" \
    --num_input_frames $NUM_INPUT_FRAMES \
    --num_output_frames $NUM_OUTPUT_FRAMES \
    --num_samples $NUM_SAMPLES \
    --seed_base $SEED_BASE \
    --num_inference_steps $NUM_STEPS \
    --input_mode crop \
    --height 480 \
    --width 832 \
    --zero_temporal_rope

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo ""
    echo "Run FAILED (exit code $exit_code)"
    exit $exit_code
fi
echo ""
echo "Done."
