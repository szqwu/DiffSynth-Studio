#!/bin/bash

# Wan2.1 6-to-1 NVS evaluation on WRIVA scenes — 3 GPUs, 2 scenes each.

# ── Configuration ────────────────────────────────────────────────────────────
checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope-480p/epoch-59.safetensors"
output_path="/data2/qiwu2/wriva_eval_results_6to1_zero-temporal-rope-480p_epoch-59"

GPU_IDS=(2 3 4 5 6 7)
SCENES_PER_GPU=1
SEED=42

SCENES=(
    "/data2/qiwu2/wriva_data/t01_v02_s07_r05_ImageDensity_A01_GoPro"
    "/data2/qiwu2/wriva_data/t01_v06_s05_r02_ImageDensity_A05"
    "/data2/qiwu2/wriva_data/t01_v12_s05_r01_ImageDensity_A05_indoor"
    "/data2/qiwu2/wriva_data/t01_v18_s05_r02_ImageDensity_S06_low_contrast"
    "/data2/qiwu2/wriva_data/t01_v19_s05_r01_ImageDensity_S07_Cathedral_Indoor"
    "/data2/qiwu2/wriva_data/t01_v20_s05_r01_ImageDensity_S07_Cathedral_Outdoor"
)

echo "========================================"
echo "Wan2.1 6-to-1 NVS — WRIVA Evaluation"
echo "========================================"
echo "Checkpoint: $checkpoint_path"
echo "Output:     $output_path"
echo "GPUs:       ${GPU_IDS[*]}"
echo "Seed:       $SEED"
echo "Scenes:     ${#SCENES[@]}"
echo ""

mkdir -p "$output_path"

PIDS=()

for i in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$i]}
    start=$((i * SCENES_PER_GPU))
    gpu_scenes=("${SCENES[@]:$start:$SCENES_PER_GPU}")

    if [ ${#gpu_scenes[@]} -eq 0 ]; then
        continue
    fi

    echo "[GPU $gpu_id] Scenes: $(basename ${gpu_scenes[0]}), $(basename ${gpu_scenes[1]:-})"

    CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.1_6to1_wriva.py \
        --checkpoint_path "$checkpoint_path" \
        --output_path "$output_path" \
        --scenes "${gpu_scenes[@]}" \
        --num_input_frames 6 \
        --height 480 \
        --width 832 \
        --seed $SEED \
        --vram_limit 22 \
        --zero_temporal_rope \
        --use_ssim \
        --use_lpips \
        --use_dreamsim \
        --num_inference_steps 50 \
        > >(while read line; do echo "[GPU $gpu_id] $line"; done) \
        2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} parallel jobs. Waiting for completion..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    gpu_id=${GPU_IDS[$i]}
    wait $pid
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "[GPU $gpu_id] FAILED (exit code $exit_code)"
        FAILED=$((FAILED + 1))
    else
        echo "[GPU $gpu_id] Done"
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All jobs completed successfully!"
else
    echo "$FAILED job(s) failed."
    exit 1
fi
