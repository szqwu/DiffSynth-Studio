#!/bin/bash

# Generate videos along 4 different camera trajectories for a specific scene.
# Each trajectory runs on a separate GPU in parallel.
#
# Usage: bash test_wan2.1_trajectory_gen_parallel.sh

SCENE="f57351ce9f16ebc7651a1193e9325a59a11e1b6e29b01f12a075facb422046dd"
DATA_PATH="/data2/qiwu2/2K"

checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope-480p/epoch-59.safetensors"
output_path="/data2/qiwu2/trajectory_gen_${SCENE:0:8}_zero-temporal-rope-480p"

GPU_IDS=( 4 5 6 7 )
TRAJECTORIES=( "orbit" "spiral" "arc" "dolly" )
NUM_FRAMES=81
FPS=16

echo "========================================"
echo "Trajectory Generation — 4 trajectories in parallel"
echo "========================================"
echo "Scene:       ${SCENE:0:16}..."
echo "Checkpoint:  $checkpoint_path"
echo "Output:      $output_path"
echo "GPUs:        ${GPU_IDS[*]}"
echo "Trajectories: ${TRAJECTORIES[*]}"
echo "Frames:      $NUM_FRAMES @ ${FPS}fps"
echo ""

mkdir -p "$output_path"

PIDS=()

for i in "${!TRAJECTORIES[@]}"; do
    gpu_id=${GPU_IDS[$i]}
    traj=${TRAJECTORIES[$i]}
    log_file="${output_path}/${traj}.log"

    echo "[GPU $gpu_id] Launching ${traj} -> ${log_file}"

    CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.1_trajectory_gen.py \
        --checkpoint_path "$checkpoint_path" \
        --data_path "$DATA_PATH" \
        --scene "$SCENE" \
        --output_path "$output_path" \
        --trajectories "$traj" \
        --height 480 \
        --width 832 \
        --num_frames "$NUM_FRAMES" \
        --fps "$FPS" \
        --num_inference_steps 50 \
        --zero_temporal_rope \
        > "$log_file" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} parallel jobs. Waiting for completion..."
echo "Monitor with: tail -f ${output_path}/*.log"
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    gpu_id=${GPU_IDS[$i]}
    traj=${TRAJECTORIES[$i]}
    wait $pid
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "[GPU $gpu_id] ${traj} FAILED (exit code $exit_code)"
        FAILED=$((FAILED + 1))
    else
        echo "[GPU $gpu_id] ${traj} Done"
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All 4 trajectories completed successfully!"
    echo "Results in $output_path"
else
    echo "$FAILED trajectory(s) failed."
    exit 1
fi
