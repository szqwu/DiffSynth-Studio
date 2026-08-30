#!/bin/bash

# Script to run Wan2.1-T2V-1.3B M-to-N NVS evaluation on DL3DV-10K scenes
# Processes scenes in parallel across 5 GPUs, 2 scenes per GPU.
# Mirrors test_wan2.1_6to1_parallel.sh, but invokes the T2V-1.3B variant of
# the test script (test_wan2.1_t2v_1_3b_6to1.py) and points at the 1.3B
# checkpoint produced by Wan2.1-SE-ZeroTemporalRoPE-T2V-1.3B.sh.

# ── M-to-N configuration ──────────────────────────────────────────────────────
NUM_INPUT_FRAMES="${1:-6}"
NUM_OUTPUT_FRAMES="${2:-1}"

# ── Configuration ─────────────────────────────────────────────────────────────
checkpoint_path="./models/train/Wan2.1-SE-T2V-1.3B-lora512-zero-temporal-rope-480p-T10/epoch-19.safetensors"
output_path="/data2/qiwu2/dl3dv_test_results_wan21_t2v_1_3b_${NUM_INPUT_FRAMES}to${NUM_OUTPUT_FRAMES}_zero-temporal-rope_epoch-480p-T10-19"

GPU_IDS=(0 1 2 3 4)
# GPU_IDS=( 7 )
SCENES_PER_GPU=2

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
echo "Wan2.1-T2V-1.3B ${NUM_INPUT_FRAMES}-to-${NUM_OUTPUT_FRAMES} NVS - DL3DV-10K Evaluation (Parallel)"
echo "========================================"
echo "Checkpoint: $checkpoint_path"
echo "Output path: $output_path"
echo "GPUs: ${GPU_IDS[*]}"
echo "Generation: ${NUM_INPUT_FRAMES}-to-${NUM_OUTPUT_FRAMES}"
echo "Number of scenes: ${#SCENES[@]}"
echo "Scenes per GPU: $SCENES_PER_GPU"
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

    echo "[GPU $gpu_id] Processing scenes: ${gpu_scenes[*]:0:1}... (${#gpu_scenes[@]} total)"

    CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.1_t2v_1_3b_6to1.py \
        --checkpoint_path $checkpoint_path \
        --output_path $output_path \
        --scenes ${gpu_scenes[@]} \
        --num_input_frames $NUM_INPUT_FRAMES \
        --num_output_frames $NUM_OUTPUT_FRAMES \
        --use_dreamsim \
        --use_ssim \
        --use_lpips \
        --input_mode "crop" \
        --height 480 \
        --width 832 \
        --eval_size 480 \
        --zero_temporal_rope \
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
