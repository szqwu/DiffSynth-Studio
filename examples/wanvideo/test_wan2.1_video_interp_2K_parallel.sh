#!/bin/bash

# 6-view video interpolation on DL3DV 2K scenes (20 random scenes, 5 GPUs)
#
# Usage:
#   bash test_wan2.1_video_interp_2K_parallel.sh          # 41 frames (gap=8,  ~2.5s @ 16fps)
#   bash test_wan2.1_video_interp_2K_parallel.sh long      # 81 frames (gap=16, ~5s   @ 16fps)

MODE="${1:-short}"

# ── Configuration ──────────────────────────────────────────────────────────────
checkpoint_path="./models/train/Wan2.1-SE-14B-lora32-6to1_zero-temporal-rope-480p/epoch-59.safetensors"
data_path="/data2/qiwu2/2K"
GPU_IDS=( 1 2 3 4 5 6 7 )
SCENES_PER_GPU=2

if [ "$MODE" = "long" ]; then
    KEYFRAME_GAP=16
    TOTAL_FRAMES=81
    output_path="/data2/qiwu2/dl3dv2K_2_video_interp_81f_zero-temporal-rope-480p_epoch-59"
else
    KEYFRAME_GAP=8
    TOTAL_FRAMES=41
    output_path="/data2/qiwu2/dl3dv2K_2_video_interp_41f_zero-temporal-rope-480p_epoch-59"
fi

# 20 random scenes from /data2/qiwu2/2K (seed=123, 1000 total)
# SCENES=(
#     "08ba311d70cbb36d4278d5c70c1b7fcc8ff73e5f1bdc168c8ae81d1241a356ab"
#     "363076e01b988452c5de9d09c518f1ded35689ae4be25992f121b6bdf8e3e3f8"
#     "11623db8d456ca3aa4966fe239a2167e0468f7fd2bcafe17b23dabea0a12af85"
#     "c8905ea0f32fb59155a2d08367f3a1e231b044dbc2e6c5a6715ecba5dfede196"
#     "5c21c41499ee41d6da01afd71f725e7f3594357350c60f9f5878dd1baf0a70ee"
#     "35fa9263da4b458a9b0afc87b04c79e9b0dd6cbff3ab2d186f30b40fc8d50d68"
#     "150795c70e2f01671b150998752605a0941cfe96f13f68ca29539108a1a57562"
#     "db516b2fe7b8b73e5fc253a13cb38968dd1b5e5df50e689b22c6cbfed636e416"
#     "ed041cbf3e284af4794176b45b8aae432a59d50cc7aca621eec66849cd96a04e"
#     "e3efc316a6531a6451b024b25d9ef849eed0b802cbd3d141dedd4e1447eeccec"
#     "065f956bc9be8c3b9fff706aa089697a4d933e6a64f872fe4830ff5a9bcadcfc"
#     "54b7714214607122510965bcecb19593d123f1988e987495cb823acadaf1abd6"
#     "84b267fa400ed6ce3f783d2aaa1a72a12664c61f9d6a0f943d150cd04edb75cd"
#     "8c44bd07536bc84a2563488ed2562b5c6e6acd4c65e075e269ce7e05eb297c60"
#     "48353c680324621478cc61fbeff5cf02dcc5d035384042aa7474a3976654ced7"
#     "4a929ece85adb624ac93b057fec8113ee66ab964925a3de935b5fab4538c90e4"
#     "dd5a82e81ce242abcd71ea4b48168010edcb570b3c87ff31c23b1e1bdbe2ff07"
#     "203ef2cbe8cbd106907ddd4184a8be827d77a7ad6beb96f08dbc81a4b36416d8"
#     "1a0b527961d348e6133bebe4d16e8ad7cdae4a56ea8761403880fc38ea719165"
#     "49a22b523a1a6e4adf506997324f2e243e2f7716f874735d1d2487b3a236f9da"
# )
SCENES=(
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
    # "d2f2df370f86aeb1ff6b061236303177a70477d64fc85c52c2b2dd76aa528b1f"
    # "8c44bd07536bc84a2563488ed2562b5c6e6acd4c65e075e269ce7e05eb297c60"
)

echo "========================================"
echo "6-View Video Interpolation — DL3DV 2K (20 scenes)"
echo "========================================"
echo "Mode: $MODE  (${TOTAL_FRAMES} frames, keyframe_gap=${KEYFRAME_GAP})"
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

    CUDA_VISIBLE_DEVICES=$gpu_id python test_wan2.1_video_interp.py \
        --checkpoint_path $checkpoint_path \
        --output_path $output_path \
        --dl3dv_data_path $data_path \
        --scene_subdir "" \
        --scenes ${gpu_scenes[@]} \
        --height 480 \
        --width 832 \
        --keyframe_gap $KEYFRAME_GAP \
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
