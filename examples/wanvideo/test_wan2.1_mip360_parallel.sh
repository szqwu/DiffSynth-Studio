#!/bin/bash

# Wan2.1 6-to-1 NVS evaluation on mip-NeRF 360 scenes
# 7 scenes distributed across 3 GPUs (round-robin)

# ── Configuration ──────────────────────────────────────────────────────────────
checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope-480p/epoch-59.safetensors"
output_path="/data2/qiwu2/mip360_test_results_wan21_6to1_zero-temporal-rope-480p_epoch-59"
GPU_IDS=( 0 1 2 3 4 5 6)

SCENES=(
    "bicycle"
    "bonsai"
    "counter"
    "garden"
    "kitchen"
    "room"
    "stump"
)

echo "========================================"
echo "Wan2.1 6-to-1 NVS - mip-NeRF 360 Evaluation (Parallel)"
echo "========================================"
echo "Checkpoint: $checkpoint_path"
echo "Output path: $output_path"
echo "GPUs: ${GPU_IDS[*]}"
echo "Scenes: ${SCENES[*]}"
echo ""

mkdir -p "$output_path"

NUM_GPUS=${#GPU_IDS[@]}
PIDS=()

for g in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$g]}

    # Collect scenes assigned to this GPU (round-robin)
    gpu_scenes=()
    for i in "${!SCENES[@]}"; do
        if [ $(( i % NUM_GPUS )) -eq $g ]; then
            gpu_scenes+=("${SCENES[$i]}")
        fi
    done

    echo "[GPU $gpu_id] Assigned scenes: ${gpu_scenes[*]}"

    # Run assigned scenes sequentially on this GPU, in a background subshell
    (
        for scene in "${gpu_scenes[@]}"; do
            echo "[GPU $gpu_id] Starting scene: $scene"
            CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.1_mip360.py \
                --checkpoint_path $checkpoint_path \
                --output_path $output_path \
                --scenes $scene \
                --use_dreamsim \
                --use_ssim \
                --use_lpips \
                --height 480 \
                --width 832 \
                --eval_h 480 \
                --eval_w 480 \
                --zero_temporal_rope \
                2>&1 | while read line; do echo "[GPU $gpu_id $scene] $line"; done
            if [ ${PIPESTATUS[0]} -ne 0 ]; then
                echo "[GPU $gpu_id $scene] FAILED"
                exit 1
            fi
            echo "[GPU $gpu_id $scene] Done"
        done
    ) &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} GPU workers. Waiting for completion..."
echo ""

FAILED=0
for g in "${!PIDS[@]}"; do
    pid=${PIDS[$g]}
    gpu_id=${GPU_IDS[$g]}
    wait $pid
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "[GPU $gpu_id] Worker FAILED (exit code $exit_code)"
        FAILED=$((FAILED + 1))
    else
        echo "[GPU $gpu_id] Worker done"
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All jobs completed successfully!"
else
    echo "$FAILED GPU worker(s) failed."
    exit 1
fi
