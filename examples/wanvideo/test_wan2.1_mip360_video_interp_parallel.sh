#!/bin/bash

# Video interpolation on mip-NeRF 360 scenes (parallel, 1 scene per GPU)
#
# Usage:
#   bash test_wan2.1_mip360_video_interp_parallel.sh          # 41 frames (gap=8,  ~2.5s @ 16fps)
#   bash test_wan2.1_mip360_video_interp_parallel.sh long      # 81 frames (gap=16, ~5s   @ 16fps)

MODE="${1:-short}"

# ── Configuration ──────────────────────────────────────────────────────────────
checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope-480p/epoch-59.safetensors"
GPU_IDS=( 1 2 3 4 5 6 7 )

if [ "$MODE" = "long" ]; then
    KEYFRAME_GAP=16
    TOTAL_FRAMES=81
    output_path="/data2/qiwu2/mip360_video_interp_81f_zero-temporal-rope-480p_epoch-59"
else
    KEYFRAME_GAP=8
    TOTAL_FRAMES=41
    output_path="/data2/qiwu2/mip360_video_interp_41f_zero-temporal-rope-480p_epoch-59"
fi

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
echo "Video Interpolation — mip-NeRF 360 (Parallel)"
echo "========================================"
echo "Mode: $MODE  (${TOTAL_FRAMES} frames, keyframe_gap=${KEYFRAME_GAP})"
echo "Checkpoint: $checkpoint_path"
echo "Output: $output_path"
echo "GPUs: ${GPU_IDS[*]}"
echo "Scenes: ${SCENES[*]}"
echo ""

mkdir -p "$output_path"

NUM_GPUS=${#GPU_IDS[@]}
PIDS=()

for g in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$g]}

    # Round-robin scene assignment
    gpu_scenes=()
    for i in "${!SCENES[@]}"; do
        if [ $(( i % NUM_GPUS )) -eq $g ]; then
            gpu_scenes+=("${SCENES[$i]}")
        fi
    done

    if [ ${#gpu_scenes[@]} -eq 0 ]; then
        continue
    fi

    echo "[GPU $gpu_id] Scenes: ${gpu_scenes[*]}"

    (
        for scene in "${gpu_scenes[@]}"; do
            echo "[GPU $gpu_id] Starting scene: $scene"
            CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.1_mip360_video_interp.py \
                --checkpoint_path $checkpoint_path \
                --output_path $output_path \
                --scenes $scene \
                --height 480 \
                --width 832 \
                --keyframe_gap $KEYFRAME_GAP \
                --fps 16 \
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
echo "Launched ${#PIDS[@]} GPU workers. Waiting..."
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
