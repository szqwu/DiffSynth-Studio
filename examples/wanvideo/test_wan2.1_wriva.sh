#!/bin/bash

# Script to run Wan2.1 N-to-M NVS evaluation on WRIVA dataset
# 6-to-1 with zero temporal rope — multi-GPU parallel evaluation

# ── Configuration ──────────────────────────────────────────────────────────────
# checkpoint_path="/ocean/projects/cis250177p/qwu6/DiffSynth-Studio/examples/wanvideo/models/train/Wan2.1-SE-14B-lora32-zero-temporal-rope_BlendedMVS_re10_stretch/step-50000.safetensors"
# output_path="/ocean/projects/cis250177p/qwu6/wriva_test_results_wan21_6to1_BlendedMVS_re10_6to1_stretch_50k_map_mapa"
checkpoint_path="/ocean/projects/cis250177p/qwu6/DiffSynth-Studio/examples/wanvideo/models/train/Wan2.1-SE-14B-lora32-zero-temporal-rope_BlendedMVS_re10_spatialvid_stretch_rayrope2/step-20000.safetensors"
output_path="/ocean/projects/cis250177p/qwu6/wriva_test_results_wan21_6to1_BlendedMVS_re10_6to1_stretch_rayrope2_20k"
GPU_IDS=(0 1 2 3)

NUM_GPUS=${#GPU_IDS[@]}

echo "========================================"
echo "Wan2.1 6-to-1 NVS — WRIVA Evaluation (Parallel)"
echo "========================================"
echo "Checkpoint: $checkpoint_path"
echo "Output path: $output_path"
echo "GPUs: ${GPU_IDS[*]}"
echo "Number of GPUs: $NUM_GPUS"
echo ""

mkdir -p "$output_path"

# Optional: override COLMAP poses directory (leave empty to use default wriva_path/inputs_colmap)
inputs_colmap_path=""  # use default wriva/inputs_colmap
# inputs_colmap_path="/ocean/projects/cis260088p/shared/wriva/inputs_mapa_colmap/"

INPUTS_COLMAP_ARG=""
if [ -n "$inputs_colmap_path" ]; then
    INPUTS_COLMAP_ARG="--inputs_colmap_path $inputs_colmap_path"
fi

COMMON_ARGS="\
    --checkpoint_path $checkpoint_path \
    --wriva_path /ocean/projects/cis250200p/mjeon2/datasets/wriva \
    $INPUTS_COLMAP_ARG \
    --num_scenes 500 \
    --num_ref 6 \
    --num_targets 1 \
    --height 192 \
    --width 336 \
    --num_inference_steps 50 \
    --use_dreamsim \
    --use_ssim \
    --use_lpips \
    --seed 42 \
    --resize_mode stretch \
    --zero_temporal_rope \
    --use_rayrope \
    --num_shards $NUM_GPUS"

PIDS=()

for i in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$i]}
    shard_output="${output_path}/shard_${i}"
    mkdir -p "$shard_output"

    echo "[GPU $gpu_id] Launching shard $i → $shard_output"

    CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=/ocean/projects/cis250177p/qwu6/DiffSynth-Studio:$PYTHONPATH python test_wan2.1_wriva.py \
        $COMMON_ARGS \
        --shard_id $i \
        --output_path "$shard_output" \
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

# ── Merge per-shard results into a single output directory ─────────────────────
echo ""
echo "Merging shard results into $output_path ..."

for i in "${!GPU_IDS[@]}"; do
    shard_output="${output_path}/shard_${i}"
    for scene_dir in "$shard_output"/*/; do
        scene_name=$(basename "$scene_dir")
        [[ "$scene_name" == shard_* ]] && continue
        mv "$scene_dir" "$output_path/" 2>/dev/null
    done
done

# Aggregate summary from all shard summaries
python -c "
import os, glob, re, numpy as np

output_path = '$output_path'
psnrs, ssims, lpipss, dreamsims = [], [], [], []
scene_lines = []

for summary in sorted(glob.glob(os.path.join(output_path, 'shard_*', 'summary.txt'))):
    for line in open(summary):
        line = line.rstrip()
        if line.startswith('  ') and ':' in line and 'PSNR=' in line:
            scene_lines.append(line)
            m = re.search(r'PSNR=([\d.]+)', line)
            if m: psnrs.append(float(m.group(1)))
            m = re.search(r'SSIM=([\d.]+)', line)
            if m: ssims.append(float(m.group(1)))
            m = re.search(r'LPIPS=([\d.]+)', line)
            if m: lpipss.append(float(m.group(1)))
            m = re.search(r'DreamSim=([\d.]+)', line)
            if m: dreamsims.append(float(m.group(1)))

with open(os.path.join(output_path, 'summary.txt'), 'w') as f:
    f.write(f'WRIVA Evaluation — Overall Results ({len(psnrs)} scenes, merged)\n\n')
    if psnrs:
        f.write(f'Mean PSNR:     {np.mean(psnrs):.2f} +/- {np.std(psnrs):.2f} dB\n')
    if ssims:
        f.write(f'Mean SSIM:     {np.mean(ssims):.4f} +/- {np.std(ssims):.4f}\n')
    if lpipss:
        f.write(f'Mean LPIPS:    {np.mean(lpipss):.4f} +/- {np.std(lpipss):.4f}\n')
    if dreamsims:
        f.write(f'Mean DreamSim: {np.mean(dreamsims):.4f} +/- {np.std(dreamsims):.4f}\n')
    f.write(f'\nPer-scene results:\n')
    for line in scene_lines:
        f.write(line + '\n')

print(f'Merged summary: {len(psnrs)} scenes')
if psnrs:
    print(f'  Mean PSNR:     {np.mean(psnrs):.2f} +/- {np.std(psnrs):.2f} dB')
if ssims:
    print(f'  Mean SSIM:     {np.mean(ssims):.4f} +/- {np.std(ssims):.4f}')
if lpipss:
    print(f'  Mean LPIPS:    {np.mean(lpipss):.4f} +/- {np.std(lpipss):.4f}')
if dreamsims:
    print(f'  Mean DreamSim: {np.mean(dreamsims):.4f} +/- {np.std(dreamsims):.4f}')
"

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All jobs completed successfully!"
else
    echo "$FAILED job(s) failed."
    exit 1
fi
echo "Results: $output_path/summary.txt"
