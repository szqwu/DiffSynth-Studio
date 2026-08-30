#!/bin/bash

# RE10K video interpolation (parallel across GPUs)
# 32 scenes: 8 each for K=6,3,4,5 input views
# Supports trajectory: "boomerang" (a→b→c→b→a, 41f) or "forward" (a→b→c, 41f)
# Both produce 41-frame videos. Random seed for scene selection: 789

# ── Configuration ──────────────────────────────────────────────────────────────
checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope-480p/epoch-59.safetensors"
TRAJECTORY="${1:-boomerang}"  # "boomerang" or "forward"
DATA_PATH="/data3/kaihuac/re10k/re10k_reorganize/test"
FRAME_STRIDE=10
GPU_IDS=( 0 1 2 3 4 5 6 7 )

if [ "$TRAJECTORY" = "forward" ]; then
    output_path="/data2/qiwu2/re10k_forward_41f_stride${FRAME_STRIDE}_zero-temporal-rope-480p_epoch-59"
else
    output_path="/data2/qiwu2/re10k_boomerang_41f_stride${FRAME_STRIDE}_zero-temporal-rope-480p_epoch-59"
fi

# Scene lists differ because forward needs longer videos (>=201 frames vs >=101 for boomerang)
if [ "$TRAJECTORY" = "forward" ]; then
# 32 scenes (>=201 frames): scene_id:K  (8 per K=2,3,4,5; seed=789)
SCENE_ASSIGNMENTS=(
    "6de80f26ff52ab0b:2"
    "582d287cce22f586:2"
    "005dd9a58df1ba3c:2"
    "505398c0e31ec5f5:2"
    "1f2153b5fb50d41a:2"
    "1a50fc5440d67fbc:2"
    "f228cfd294bce60a:2"
    "29f52c76f269ae48:2"
    "f088c85322065722:3"
    "550ea130af96ccfa:3"
    "8534404c06a7a475:3"
    "fa60fab979100404:3"
    "7a911883348688e9:3"
    "1cca8650a292e7b0:3"
    "f3c66c5552e28f20:3"
    "fc966f25afa3659f:3"
    "99127d4b1b789247:4"
    "03906f66d3bca71a:4"
    "8000d5e5ca364d8b:4"
    "e7cb1f56ac69308a:4"
    "06058474f164c53d:4"
    "daceae3c5381041b:4"
    "bf800a3d2af8f88c:4"
    "c0611596a79be2b6:4"
    "f8c255e4f6f28bdc:5"
    "61192009ce2e8d25:5"
    "c0c0508813b848ca:5"
    "24668d960406587f:5"
    "599619a790d2b77d:5"
    "90cfef1d9f506bc3:5"
    "06ffaa9ffc2eea95:5"
    "297b57a9296052ce:5"
)
else
# 32 scenes (>=201 frames): scene_id:K  (8 per K=6,3,4,5; seed=789)
SCENE_ASSIGNMENTS=(
    "6de80f26ff52ab0b:6"
    "582d287cce22f586:6"
    "005dd9a58df1ba3c:6"
    "505398c0e31ec5f5:6"
    "1f2153b5fb50d41a:6"
    "1a50fc5440d67fbc:6"
    "f228cfd294bce60a:6"
    "29f52c76f269ae48:6"
    "f088c85322065722:3"
    "550ea130af96ccfa:3"
    "8534404c06a7a475:3"
    "fa60fab979100404:3"
    "7a911883348688e9:3"
    "1cca8650a292e7b0:3"
    "f3c66c5552e28f20:3"
    "fc966f25afa3659f:3"
    "99127d4b1b789247:4"
    "03906f66d3bca71a:4"
    "8000d5e5ca364d8b:4"
    "e7cb1f56ac69308a:4"
    "06058474f164c53d:4"
    "daceae3c5381041b:4"
    "bf800a3d2af8f88c:4"
    "c0611596a79be2b6:4"
    "f8c255e4f6f28bdc:5"
    "61192009ce2e8d25:5"
    "c0c0508813b848ca:5"
    "24668d960406587f:5"
    "599619a790d2b77d:5"
    "90cfef1d9f506bc3:5"
    "06ffaa9ffc2eea95:5"
    "297b57a9296052ce:5"
)
fi

NUM_SCENES=${#SCENE_ASSIGNMENTS[@]}
NUM_GPUS=${#GPU_IDS[@]}
SCENES_PER_GPU=$(( (NUM_SCENES + NUM_GPUS - 1) / NUM_GPUS ))

echo "========================================"
echo "RE10K Video Interp"
echo "========================================"
echo "Trajectory: $TRAJECTORY (stride=$FRAME_STRIDE)"
echo "K values: 2,3,4,5 (8 scenes each)"
echo "Checkpoint: $checkpoint_path"
echo "Data: $DATA_PATH"
echo "Output: $output_path"
echo "GPUs: ${GPU_IDS[*]}"
echo "Total scenes: $NUM_SCENES"
echo "Scenes per GPU: ~$SCENES_PER_GPU"
echo ""

mkdir -p "$output_path"

PIDS=()

for i in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$i]}
    start=$((i * SCENES_PER_GPU))
    gpu_scenes=("${SCENE_ASSIGNMENTS[@]:$start:$SCENES_PER_GPU}")

    if [ ${#gpu_scenes[@]} -eq 0 ]; then
        continue
    fi

    echo "[GPU $gpu_id] Scenes (${#gpu_scenes[@]}): ${gpu_scenes[*]}"

    CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.1_re10k_boomerang.py \
        --checkpoint_path "$checkpoint_path" \
        --data_path "$DATA_PATH" \
        --output_path "$output_path" \
        --scenes ${gpu_scenes[@]} \
        --trajectory "$TRAJECTORY" \
        --frame_stride "$FRAME_STRIDE" \
        --height 480 \
        --width 832 \
        --fps 16 \
        --num_inference_steps 50 \
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
