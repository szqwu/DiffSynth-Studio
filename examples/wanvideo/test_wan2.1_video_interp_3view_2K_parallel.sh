#!/bin/bash

# 3-view boomerang video interpolation on DL3DV-10K 2K scenes (20 random scenes, 5 GPUs)
# Trajectory: 1 → 2 → 3 → 2 → 1  (41 frames, 3 input views, 3-to-1 generation)

# ── Configuration ──────────────────────────────────────────────────────────────
checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope-480p/epoch-59.safetensors"
output_path="/data2/qiwu2/dl3dv2K_video_interp_3view_boomerang_zero-temporal-rope-480p_epoch-59"
data_path="/data2/qiwu2/2K"
GPU_IDS=( 2 3 4 5 6 )
SCENES_PER_GPU=4

# 20 random scenes from /data2/qiwu2/2K (seed=42, 1000 total)
SCENES=(
    "a4e84d38d0398c40cb809f425febfb1253ecb519d0cfda241afa574f1fe4660f"
    "1608739ca4e4ae95d2433d186ab8f3411bb56df6a61c06a2f601d3d73eac31ca"
    "03f5f648befbfbee436e85e8b43995648a2789f1d5d9e35ea682a53c58f42108"
    "c0adee230189604b746807fa6e73e531534ea7f1a081887df612fa193d77bf91"
    "37b37cd5a820ea577ce7b807db13261b235886c2ff5771b92e793682b53f8ad6"
    "30dc4113621246c20bd0facf2192677e0c0410da1b5240961c0b5b7436e99b0e"
    "2bdf076d2c75b74c6f3fbc501f32dad21ea796ea5196997c95197ba1385faf79"
    "1c172392300e78fcd12a101de13333974abd546706cca05be5fb53120ab0e03f"
    "c02acfbe324525d9c4080a8b4e273f9bb5a471545345a6bba0c464bfd7526acb"
    "14073cdbea2c07a707c904ded4e33d21492682ee9b49125489241b41711cc753"
    "aefd52169b81db31aa726f146a17cc72a07c067ad3b919ee093d885827d3c81f"
    "c09bd1733e0945ed95de509e2f396e21e9acaeb38d42af0706e4854c6d277154"
    "e952fa4663837d85620fb46b8b59954c8e6b332b12cd8c3f5aac55386d894785"
    "88fb23c8d6da1fc80cc307cf984b84e4b18e1c8d15e5b8e0ed6e2964469f2c68"
    "11623db8d456ca3aa4966fe239a2167e0468f7fd2bcafe17b23dabea0a12af85"
    "940ddc0c37df5649c1751549a5d76d2b51ab81f2bcc48fc85414d899494298eb"
    "60edf44d6eaeba1f2219f98f479e88dece591f712f93be7c76defb6478f1a83c"
    "04db5d8c9cbfdc2cd76d318ffd2ce03948f392940533ba3d60edace0b16879f7"
    "04cfe5a3c887e1e27bc2acb8459415ea307f062b2284b7e3527b85c98bb332a6"
    "1288a8707f7cba29932895a046f8e53ee0e4661859091d4c0f601a9dca2a720f"
)

echo "========================================"
echo "3-View Boomerang Video Interp — DL3DV 2K (20 scenes)"
echo "========================================"
echo "Trajectory: 1→2→3→2→1 (41 frames, 3-to-1)"
echo "Checkpoint: $checkpoint_path"
echo "Data: $data_path"
echo "Output: $output_path"
echo "GPUs: ${GPU_IDS[*]}  (${SCENES_PER_GPU} scenes/GPU)"
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

    echo "[GPU $gpu_id] ${#gpu_scenes[@]} scenes: ${gpu_scenes[0]:0:16}..."

    CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.1_video_interp_3view.py \
        --checkpoint_path $checkpoint_path \
        --output_path $output_path \
        --dl3dv_data_path $data_path \
        --scene_subdir "" \
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
