#!/bin/bash

# 3-view boomerang video interpolation on DL3DV-10K (parallel across GPUs)
# Trajectory: 1 → 2 → 3 → 2 → 1  (41 frames, 3 input views, 3-to-1 generation)

# ── Configuration ──────────────────────────────────────────────────────────────
checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope-480p/epoch-59.safetensors"
output_path="/data2/qiwu2/dl3dv_video_interp_3view_boomerang_zero-temporal-rope-480p_epoch-59"
GPU_IDS=( 2 3 4 5 6 )
SCENES_PER_GPU=2

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
echo "3-View Boomerang Video Interp — DL3DV-10K"
echo "========================================"
echo "Trajectory: 1→2→3→2→1 (41 frames, 3-to-1)"
echo "Checkpoint: $checkpoint_path"
echo "Output: $output_path"
echo "GPUs: ${GPU_IDS[*]}"
echo "Scenes: ${#SCENES[@]}"
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

    echo "[GPU $gpu_id] Scenes: ${gpu_scenes[*]:0:1}... (${#gpu_scenes[@]} total)"

    CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.1_video_interp_3view.py \
        --checkpoint_path $checkpoint_path \
        --output_path $output_path \
        --scenes ${gpu_scenes[@]} \
        --height 480 \
        --width 832 \
        --fps 16 \
        --zero_temporal_rope \
        > >(while read line; do echo "[GPU $gpu_id] $line"; done) \
        2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} parallel jobs. Waiting..."
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
