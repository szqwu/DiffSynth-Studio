#!/bin/bash

# Run Wan2.2-TI2V-5B input-encoder 6-to-1 NVS evaluation on DL3DV-10K scenes.
# Processes scenes in parallel across 5 GPUs, 2 scenes per GPU.
#
# Matches training config in
#   examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B_InputEncoder_6to1.sh
#   - separate (trainable) input VAE + zero-init MLP residual, read via TRUE
#     per-layer cross-attention (per-layer token replacement)
#   - per-token timestep (context=0, target=t), zero-temporal RoPE
#   - raymap at the /16 VAE latent resolution (6 channels * 16^2 = 1536)
#   - new_in_dim = 48 (latent) + 1536 (raymap) = 1584

# ── M-to-N configuration ──────────────────────────────────────────────────────
NUM_INPUT_FRAMES="${1:-6}"
NUM_OUTPUT_FRAMES="${2:-1}"

# ── Configuration ──────────────────────────────────────────────────────────────
EPOCH="${3:-159}"
# Training run to evaluate (subdir under ./models/train). Default: the run that
# ALSO fine-tunes the input VAE encoder (--trainable_input_vae).
RUN_NAME="${4:-Wan2.2-TI2V-5B_lora128_InputEncoder_6to1_TrainableInputVAE}"
checkpoint_path="./models/train/${RUN_NAME}/epoch-${EPOCH}.safetensors"
output_path="/data2/qiwu2/dl3dv_test_results_wan22_5B_${RUN_NAME}_${NUM_INPUT_FRAMES}to${NUM_OUTPUT_FRAMES}_epoch-${EPOCH}"

NEW_IN_DIM=1584
NUM_INFERENCE_STEPS=50
# GPUs to use (override with GPU_IDS env, e.g. GPU_IDS="0 6 7").
read -r -a GPU_IDS <<< "${GPU_IDS:-1 2 3 4 5}"
SCENES_PER_GPU="${SCENES_PER_GPU:-2}"
# Seconds to wait between launching workers. Each worker needs ~30GB CPU RAM
# (the streamed text encoder lives in host RAM); staggering avoids the OS OOM
# killer when many workers load the large model files at once.
STAGGER_SECONDS="${STAGGER_SECONDS:-45}"

# All 10 DL3DV-10K test scenes (override with SCENES env, space-separated hashes).
if [ -n "${SCENES:-}" ]; then
    read -r -a SCENES <<< "${SCENES}"
else
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
fi

echo "========================================"
echo "Wan2.2-TI2V-5B InputEncoder ${NUM_INPUT_FRAMES}-to-${NUM_OUTPUT_FRAMES} NVS - DL3DV-10K Evaluation (Parallel)"
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

    CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.2_6to1_input_encoder.py \
        --checkpoint_path $checkpoint_path \
        --output_path $output_path \
        --scenes ${gpu_scenes[@]} \
        --new_in_dim $NEW_IN_DIM \
        --num_input_frames $NUM_INPUT_FRAMES \
        --num_output_frames $NUM_OUTPUT_FRAMES \
        --use_dreamsim \
        --use_ssim \
        --use_lpips \
        --input_mode "crop" \
        --height 480 \
        --width 832 \
        --eval_size 480 \
        --raymap_downsample_factor 16 \
        --zero_temporal_rope \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        > >(while read line; do echo "[GPU $gpu_id] $line"; done) \
        2>&1 &

    PIDS+=($!)

    # Stagger launches so workers don't all load large models into RAM at once.
    if [ "$i" -lt $((${#GPU_IDS[@]} - 1)) ] && [ "$STAGGER_SECONDS" -gt 0 ]; then
        sleep "$STAGGER_SECONDS"
    fi
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
