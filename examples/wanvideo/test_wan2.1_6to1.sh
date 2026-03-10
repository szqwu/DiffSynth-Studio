#!/bin/bash

# Script to run Wan2.1 M-to-N NVS evaluation on DL3DV-10K scenes
# Processes all scenes sequentially on a single GPU

# ── Mode: "standard", "prope", or "zero_temporal_rope" ────────────────────────
# MODE="${1:-prope}"
MODE="${1:-zero_temporal_rope}"

# ── M-to-N configuration ──────────────────────────────────────────────────────
NUM_INPUT_FRAMES="${2:-6}"
NUM_OUTPUT_FRAMES="${3:-2}"

# ── Configuration ──────────────────────────────────────────────────────────────
if [ "$MODE" = "prope" ]; then
    checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_prope/epoch-39.safetensors"
    output_path="/data2/qiwu2/dl3dv_test_results_wan21_6to1_prope-119"
    EXTRA_ARGS="--use_prope"
elif [ "$MODE" = "zero_temporal_rope" ]; then
    # checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope_2/epoch-79.safetensors"
    checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-zero-temporal-rope_2-T10_2/epoch-79.safetensors"
    output_path="/data2/qiwu2/dl3dv_test_results_wan21_${NUM_INPUT_FRAMES}to${NUM_OUTPUT_FRAMES}_zero-temporal-rope_2-T10_2_79"
    EXTRA_ARGS="--zero_temporal_rope"
else
    # checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_curriculum/epoch-79.safetensors"
    checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_sorted_context/epoch-79.safetensors"
    output_path="/data2/qiwu2/dl3dv_test_results_wan21_${NUM_INPUT_FRAMES}to${NUM_OUTPUT_FRAMES}_sorted_context"
    EXTRA_ARGS=""
fi
GPU_ID=7

# All 10 DL3DV-10K test scenes
SCENES=(
    "165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557"
    "3bb3bb4d3e871d79eb71946cbab1e3afc7a8e33a661153033f32deb3e23d2e52"
    "3bb894d1933f3081134ad2d40e54de5f0636bd8b502b0a8561873bb63b0dce85"
    "9e9a89ae6fed06d6e2f4749b4b0059f35ca97f848cedc4a14345999e746f7884"
    "341b4ff3dfd3d377d7167bd81f443bedafbff003bf04881b99760fc0aeb69510"
    "cd9c981eeb4a9091547af19181b382698e9d9eee0a838c7c9783a8a268af6aee"
    "d4fbeba0168af8fddb2fc695881787aedcd62f477c7dcec9ebca7b8594bbd95b"
    "e78f8cebd2bd93d960bfaeac18fac0bb2524f15c44288903cd20b73e599e8a81"
    "ed16328235c610f15405ff08711eaf15d88a0503884f3a9ccb5a0ee69cb4acb5"
    "f71ac346cd0fc4652a89afb37044887ec3907d37d01d1ceb0ad28e1a780d8e03"
)

echo "========================================"
echo "Wan2.1 ${NUM_INPUT_FRAMES}-to-${NUM_OUTPUT_FRAMES} NVS - DL3DV-10K Evaluation"
echo "========================================"
echo "Mode: $MODE"
echo "Checkpoint: $checkpoint_path"
echo "Output path: $output_path"
echo "GPU: $GPU_ID"
echo "Generation: ${NUM_INPUT_FRAMES}-to-${NUM_OUTPUT_FRAMES}"
echo "Number of scenes: ${#SCENES[@]}"
echo "Processing: sequentially on single GPU"
echo ""

# Create output directory
mkdir -p "$output_path"

# Run all scenes sequentially on one GPU
CUDA_VISIBLE_DEVICES=$GPU_ID python test_wan2.1_6to1.py \
    --checkpoint_path $checkpoint_path \
    --output_path $output_path \
    --scenes ${SCENES[@]} \
    --num_input_frames $NUM_INPUT_FRAMES \
    --num_output_frames $NUM_OUTPUT_FRAMES \
    --use_dreamsim \
    --use_ssim \
    --use_lpips \
    --input_mode "crop" \
    --height 192 \
    --width 336 \
    $EXTRA_ARGS
echo ""
echo "Done!"

