#!/bin/bash

# 6-to-1 NVS on 14 randomly selected scenes from /data2/qiwu2/2K
# For each scene: pick 24 random frames, 6 as input, 18 as target
# 6 batches × 3 targets per batch = 18 targets per scene
#
# Usage: bash test_wan2.1_6to1_2K_random_parallel.sh [gpu_ids]
# Default: GPUs 2 3 4 5 6 (5 GPUs, ~3 scenes each)

# ── Configuration ──────────────────────────────────────────────────────────────
checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope-480p/epoch-59.safetensors"
output_path="/data2/qiwu2/dl3dv2K_random_6to1_zero-temporal-rope-480p_epoch-59"
GPU_IDS=( ${@:-1 2 3 4 5 6 7} )

# 14 scenes randomly selected with seed=456 from /data2/qiwu2/2K (1000 scenes)
SCENES=(
    "c2d4dce3ea6cb4611f1119346d929603c43b81fb7de5753771749677419d45c3"
    "67f06eebee5c8ff5049b355e6f289ed217acc7e1389950c205d37d127c742f53"
    "fe39bbbe767bb1fbe52593e44aa5afcbb2ab9389aeb19081d6f4360a78f5470e"
    "6cb22a1620b1a2521876bac307e856b20ed16553af326acce98f017aa763c0e4"
    "64ed92f70e299c54b71fa95c09ff4bb759c1e1489b43f6c477bd9bf88685bfe7"
    "57f3d13ef3be77e32baf6d3c79325c2e29681a2324e2426d8d7a5364ee8e343c"
    "cf1bb2b0419219d6031a1364a35bcd5608002d727339e5c2ea7803f1b5bff44b"
    "e7ffc8e1605737bf210a0287fbcefd0370a4fb42c57f1dbfae86b2540a07d3b5"
    "a621ac6530017745cb9785af80c0a83469239d7ce382e9564363871117d4b51f"
    "06859d23eb82b98b4bebc9080b6ff010e844dc81830abd1a9258800471167d9e"
    "9ff3eca905784e4252b19599ec918a10b4685ec8d4d992f951be9ed4fea4fc16"
    "bdc2100d9152eae21eea9d864ed1ec47c3888457782c4d5e0eaa5fbf8bd38ad5"
    "d698c3ddba21b19f448488ac7b9282d9560d9dce5703b33a3a9705b687f8ccab"
    "ec9effaaa75a8ad5f106905381dc6ddd65c1a6ce15b3b1406ca35610330a175d"
    "2b660a52c63dd131f0a06543203dca9d127d883c49ecc78e6253df0685b46af2"
    "203ef2cbe8cbd106907ddd4184a8be827d77a7ad6beb96f08dbc81a4b36416d8"
    "e924d84b230ac949643490bddb296ebc780f8385e88fc94ddd29658a7bcd79cf"
    "fe39bbbe767bb1fbe52593e44aa5afcbb2ab9389aeb19081d6f4360a78f5470e"
    "ff269a1ed8243ae0df3cc71f184fd7eb06f593cc01ae874b6de6c8af5254cf8e"
    "ffa1e152c40f52a31145940ded98a450e5a81fd76367c0c9793ae787b3b60e1a"
    "f215685055a2a920e83bf219c162711d823bfe9e16ed3439194a43557c3122fd"
    "f57351ce9f16ebc7651a1193e9325a59a11e1b6e29b01f12a075facb422046dd"
    "f9482fae63f142e30818508e0763533e87e31443437457c49d0c329efa84f3a8"
    "f4dcde86abf380b64e9d58f530dbb647a921f7f99b9b4292139c9e142d5cd74d"
    "f0b021144b0545e84ad43a1d704de349923b904bde80cd891957c1eb00e55eec"
    "edf9b3d11167bbe3601670c84eae6c799c2479e6121bd1ed812fa98814a0cc3a"
    "eb6af3becf2b369a5d075dc52ec43a9ce9a6b120e4dbde16d3540f7f44c8edbd"
    "d717a1f9c02e3eeb0df7fcf5c93335742ac612e46f75ffdf2be4d7cae8a55cbb"
)

NUM_SCENES=${#SCENES[@]}
NUM_GPUS=${#GPU_IDS[@]}
SCENES_PER_GPU=$(( (NUM_SCENES + NUM_GPUS - 1) / NUM_GPUS ))

echo "========================================"
echo "6-to-1 NVS — 2K Random Scenes (Parallel)"
echo "========================================"
echo "Checkpoint: $checkpoint_path"
echo "Output:     $output_path"
echo "GPUs:       ${GPU_IDS[*]}"
echo "Scenes:     $NUM_SCENES ($SCENES_PER_GPU per GPU)"
echo "Per scene:  24 frames → 6 input + 18 target → 6 batches × 3"
echo "Seed:       456"
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

    echo "[GPU $gpu_id] Processing ${#gpu_scenes[@]} scenes: ${gpu_scenes[0]:0:8}..."

    CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.1_6to1_2K_random.py \
        --checkpoint_path "$checkpoint_path" \
        --output_path "$output_path" \
        --data_path "/data2/qiwu2/2K" \
        --scenes ${gpu_scenes[@]} \
        --num_input_frames 6 \
        --num_output_frames 1 \
        --num_pick 24 \
        --seed 456 \
        --height 480 \
        --width 832 \
        --eval_size 480 \
        --input_mode "crop" \
        --zero_temporal_rope \
        --use_dreamsim \
        --use_ssim \
        --use_lpips \
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
