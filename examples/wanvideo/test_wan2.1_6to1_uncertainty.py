#!/usr/bin/env python3
"""
Wan2.1 6-to-1 NVS uncertainty sampling on one DL3DV-10K scene.

For each target frame:
  - Runs the pipeline K times with seeds seed_base, seed_base+1, ..., seed_base+K-1,
    at --num_inference_steps (default 5 — deliberately small for budget parity).
  - Saves all K raw sample PNGs.
  - Saves the per-sample mean as mean_frame_XXXX.png.
  - Saves per-pixel std map (.npy) and a colormap heatmap (.png).
  - Saves a grid (K samples + mean + std heatmap) as samples_grid_frame_XXXX.png.
  - Computes pairwise LPIPS between samples (K×K matrix) and logs the mean.

No GT is loaded or compared against.

Reuses the model-setup / raymap utilities from test_wan2.1_6to1.py.
"""

import os
import sys
import time
import json
import math
import argparse
import random
from itertools import combinations

import torch
import numpy as np
import cv2
from PIL import Image

# Reuse setup and data utilities from the reference 6to1 script.
# The reference file name contains a dot ("test_wan2.1_6to1.py"), which makes it
# not a valid Python module identifier for a plain import. Load it via
# importlib.util instead.
import importlib.util as _ilu
_ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_wan2.1_6to1.py")
_spec = _ilu.spec_from_file_location("test_wan2_1_6to1_ref", _ref_path)
_ref = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ref)  # type: ignore

load_pipeline = _ref.load_pipeline
prepare_raymap = _ref.prepare_raymap
resize_crop_to_rect = _ref.resize_crop_to_rect


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty helpers
# ─────────────────────────────────────────────────────────────────────────────

def std_heatmap_png(std_scalar_map, target_h=None, target_w=None,
                    colormap=cv2.COLORMAP_INFERNO, add_colorbar=True):
    """
    Turn a 2-D float std map into a BGR→RGB colormapped PNG with an optional
    vertical colorbar strip on the right. Returns uint8 HxW'x3 RGB array.
    """
    if target_h is None:
        target_h = std_scalar_map.shape[0]
    if target_w is None:
        target_w = std_scalar_map.shape[1]

    s_min = float(std_scalar_map.min())
    s_max = float(std_scalar_map.max())
    denom = max(s_max - s_min, 1e-8)
    s_norm = ((std_scalar_map - s_min) / denom * 255.0).clip(0, 255).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(s_norm, colormap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    if (heat_rgb.shape[0], heat_rgb.shape[1]) != (target_h, target_w):
        heat_rgb = cv2.resize(heat_rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    if add_colorbar:
        bar_w = max(16, target_w // 40)
        gradient = np.linspace(255, 0, target_h, dtype=np.uint8).reshape(-1, 1)
        gradient = np.tile(gradient, (1, bar_w))
        bar_bgr = cv2.applyColorMap(gradient, colormap)
        bar_rgb = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2RGB)
        out = np.concatenate([heat_rgb, bar_rgb], axis=1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        lbl_top = f"{s_max:.3f}"
        lbl_bot = f"{s_min:.3f}"
        pad_x = heat_rgb.shape[1] + 2
        cv2.putText(out, lbl_top, (pad_x, 16), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, lbl_bot, (pad_x, target_h - 6), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return out
    return heat_rgb


def make_sample_grid(samples_uint8, mean_uint8, std_heat_uint8, cols=6, label=""):
    """
    samples_uint8: (K, H, W, 3) uint8
    mean_uint8:    (H, W, 3)    uint8
    std_heat_uint8:(H, W', 3)   uint8  (may be wider than W due to colorbar)

    Returns an HxW grid image with K+2 tiles. Missing tiles are white-filled.
    """
    K, H, W, _ = samples_uint8.shape
    tiles = [samples_uint8[i] for i in range(K)] + [mean_uint8]

    # Pad std heatmap (which might be wider) to match W by right-cropping or
    # scaling the heatmap image down to W for the grid tile.
    if std_heat_uint8.shape[1] != W:
        std_tile = cv2.resize(std_heat_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        std_tile = std_heat_uint8
    tiles.append(std_tile)

    n_tiles = len(tiles)
    rows = math.ceil(n_tiles / cols)
    gap = 6
    grid_h = rows * H + (rows - 1) * gap + 30  # + caption strip
    grid_w = cols * W + (cols - 1) * gap
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = r * (H + gap)
        x0 = c * (W + gap)
        grid[y0:y0 + H, x0:x0 + W] = tile
        if i < K:
            cap = f"sample {i}"
        elif i == K:
            cap = "mean"
        else:
            cap = "std"
        cv2.putText(grid, cap, (x0 + 6, y0 + 18), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(grid, cap, (x0 + 6, y0 + 18), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if label:
        cv2.putText(grid, label, (8, grid_h - 10), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(grid, label, (8, grid_h - 10), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return grid


def pairwise_lpips_matrix(samples_uint8, lpips_model):
    """
    Compute a K×K symmetric LPIPS matrix from an (K, H, W, 3) uint8 stack.
    """
    K = samples_uint8.shape[0]
    tensors = []
    for i in range(K):
        t = torch.from_numpy(samples_uint8[i]).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        tensors.append(t.cuda())

    mat = np.zeros((K, K), dtype=np.float32)
    with torch.no_grad():
        for i, j in combinations(range(K), 2):
            d = lpips_model(tensors[i], tensors[j]).item()
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# Per-scene uncertainty processing
# ─────────────────────────────────────────────────────────────────────────────

def process_scene_uncertainty(pipe, args, scene_hash, lpips_model):
    model_h = args.height
    model_w = args.width
    num_input = args.num_input_frames
    num_output = args.num_output_frames
    K = args.num_samples

    scene_meta_path = os.path.join(args.dl3dv_meta_path, scene_hash)
    scene_data_path = os.path.join(args.dl3dv_data_path, scene_hash, "nerfstudio")

    print(f"\n{'='*80}")
    print(f"Processing scene: {scene_hash}")
    print(f"  K={K} samples × steps={args.num_inference_steps} per target frame")
    print(f"{'='*80}")

    split_file = os.path.join(scene_meta_path, f"train_test_split_{num_input}.json")
    if not os.path.exists(split_file):
        print(f"  Warning: {split_file} not found. Skipping.")
        return None
    with open(split_file, 'r') as f:
        split_data = json.load(f)

    train_ids = split_data['train_ids']
    test_ids = split_data['test_ids']
    print(f"  Context frame IDs ({num_input}): {train_ids}")
    print(f"  Target frame IDs ({len(test_ids)}): {test_ids[:5]}{'...' if len(test_ids)>5 else ''}")

    transforms_file = os.path.join(scene_data_path, "transforms.json")
    if not os.path.exists(transforms_file):
        print(f"  Warning: {transforms_file} not found. Skipping.")
        return None
    with open(transforms_file, 'r') as f:
        transforms_data = json.load(f)

    orig_w = transforms_data['w']
    orig_h = transforms_data['h']
    actual_w, actual_h = 960, 540

    scale_w_960 = actual_w / orig_w
    scale_h_960 = actual_h / orig_h

    orig_intrinsic = np.array([
        [transforms_data['fl_x'], 0, transforms_data['cx']],
        [0, transforms_data['fl_y'], transforms_data['cy']],
        [0, 0, 1]
    ], dtype=np.float32)

    scaled_intrinsic_960p = orig_intrinsic.copy()
    scaled_intrinsic_960p[0, 0] *= scale_w_960
    scaled_intrinsic_960p[1, 1] *= scale_h_960
    scaled_intrinsic_960p[0, 2] *= scale_w_960
    scaled_intrinsic_960p[1, 2] *= scale_h_960

    input_mode = args.input_mode
    if input_mode == "crop":
        crop_scale = max(model_h / actual_h, model_w / actual_w)
        resized_h = int(round(actual_h * crop_scale))
        resized_w = int(round(actual_w * crop_scale))
        crop_offset_x = (resized_w - model_w) / 2.0
        crop_offset_y = (resized_h - model_h) / 2.0
        scaled_intrinsic_model = scaled_intrinsic_960p.copy()
        scaled_intrinsic_model[0, 0] *= crop_scale
        scaled_intrinsic_model[1, 1] *= crop_scale
        scaled_intrinsic_model[0, 2] *= crop_scale
        scaled_intrinsic_model[1, 2] *= crop_scale
        scaled_intrinsic_model[0, 2] -= crop_offset_x
        scaled_intrinsic_model[1, 2] -= crop_offset_y
    else:
        scaled_intrinsic_model = scaled_intrinsic_960p.copy()
        scaled_intrinsic_model[0, 0] *= model_w / actual_w
        scaled_intrinsic_model[1, 1] *= model_h / actual_h
        scaled_intrinsic_model[0, 2] *= model_w / actual_w
        scaled_intrinsic_model[1, 2] *= model_h / actual_h

    frames_data = transforms_data['frames']
    all_indices = train_ids + test_ids
    all_images_model = {}
    all_extrinsics = {}

    for idx in all_indices:
        frame_data = frames_data[idx]
        file_path = frame_data['file_path'].replace('images/', 'images_4/')
        img_path = os.path.join(scene_data_path, file_path)
        if not os.path.exists(img_path):
            print(f"  Warning: Image {img_path} not found. Skipping scene.")
            return None
        img_960p = np.array(Image.open(img_path).convert('RGB'))
        if input_mode == "crop":
            img_model, _, _, _ = resize_crop_to_rect(img_960p, model_h, model_w)
        else:
            img_model = cv2.resize(img_960p, (model_w, model_h), interpolation=cv2.INTER_AREA)
        all_images_model[idx] = img_model
        all_extrinsics[idx] = np.array(frame_data['transform_matrix'], dtype=np.float32)

    output_dir = os.path.join(args.output_path, scene_hash)
    os.makedirs(output_dir, exist_ok=True)

    target_batches = [test_ids[i:i + num_output] for i in range(0, len(test_ids), num_output)]
    total_batches = len(target_batches)

    per_frame_records = []

    for batch_pos, target_batch in enumerate(target_batches):
        batch_str = ",".join(str(t) for t in target_batch)
        print(f"\n  [batch {batch_pos+1}/{total_batches}] targets={batch_str}")

        cur_num_output = len(target_batch)
        current_indices = train_ids + target_batch
        context_indices = list(range(num_input))
        target_indices = list(range(num_input, num_input + cur_num_output))
        num_total = num_input + cur_num_output

        context_images = [Image.fromarray(all_images_model[idx]) for idx in train_ids]
        current_extrinsics = np.stack([all_extrinsics[idx] for idx in current_indices], axis=0)
        current_intrinsics = np.stack([scaled_intrinsic_model] * num_total, axis=0)

        raymap, camera_poses_norm, intrinsics_tensor = prepare_raymap(
            current_extrinsics, current_intrinsics,
            context_indices, target_indices,
            model_h, model_w,
            no_pixel_unshuffle=args.no_pixel_unshuffle,
        )
        raymap = raymap.to("cuda", dtype=torch.bfloat16)

        # Collect K samples for each target frame in this batch
        # samples_per_frame[t_off] will be a list of K uint8 (H,W,3) arrays
        samples_per_frame = [[] for _ in range(cur_num_output)]

        for k in range(K):
            seed_k = args.seed_base + k
            print(f"    sample {k+1}/{K}  (seed={seed_k})")
            pipe_kwargs = dict(
                prompt="",
                negative_prompt="",
                input_image=context_images,
                input_video=None,
                raymap=raymap,
                height=model_h,
                width=model_w,
                num_frames=num_total,
                num_latent_frames=num_total,
                cfg_scale=1.0,
                num_inference_steps=args.num_inference_steps,
                seed=seed_k,
                tiled=True,
            )
            if args.zero_temporal_rope:
                pipe_kwargs["zero_temporal_rope"] = True

            video = pipe(**pipe_kwargs)
            for t_off in range(cur_num_output):
                pred_pil = video[num_input + t_off]
                samples_per_frame[t_off].append(np.array(pred_pil))

        # Per target frame in this batch: compute stats & dump outputs
        for t_off in range(cur_num_output):
            frame_idx = target_batch[t_off]
            samples = np.stack(samples_per_frame[t_off], axis=0)  # (K, H, W, 3) uint8

            # Raw samples
            samples_dir = os.path.join(output_dir, f"samples_frame_{frame_idx:04d}")
            os.makedirs(samples_dir, exist_ok=True)
            for k in range(K):
                Image.fromarray(samples[k]).save(
                    os.path.join(samples_dir, f"sample_{k:02d}.png")
                )

            # Mean image
            mean_img = samples.astype(np.float32).mean(axis=0)
            mean_img_u8 = np.clip(mean_img, 0, 255).astype(np.uint8)
            Image.fromarray(mean_img_u8).save(
                os.path.join(output_dir, f"mean_frame_{frame_idx:04d}.png")
            )

            # Per-pixel std map: average over channels to a scalar per pixel
            std_per_channel = samples.astype(np.float32).std(axis=0)   # (H,W,3)
            std_scalar_map = std_per_channel.mean(axis=-1)              # (H,W)
            np.save(os.path.join(output_dir, f"std_map_frame_{frame_idx:04d}.npy"),
                    std_scalar_map.astype(np.float32))

            std_heat = std_heatmap_png(std_scalar_map)
            Image.fromarray(std_heat).save(
                os.path.join(output_dir, f"std_map_frame_{frame_idx:04d}.png")
            )

            # Pairwise LPIPS matrix
            pw_mat = None
            mean_pw = None
            if lpips_model is not None and args.sample_lpips:
                pw_mat = pairwise_lpips_matrix(samples, lpips_model)
                iu = np.triu_indices(K, k=1)
                mean_pw = float(pw_mat[iu].mean()) if iu[0].size else 0.0

            # Scalars for log
            mean_std = float(std_scalar_map.mean())
            max_std = float(std_scalar_map.max())

            msg = f"    frame {frame_idx}: mean_std={mean_std:.3f}, max_std={max_std:.3f}"
            if mean_pw is not None:
                msg += f", mean_pairwise_lpips={mean_pw:.4f}"
            print(msg)

            # Sample grid
            grid_label = f"frame {frame_idx} | K={K} samples | steps={args.num_inference_steps}"
            if mean_pw is not None:
                grid_label += f" | mean_std={mean_std:.3f} | mean_pw_lpips={mean_pw:.4f}"
            else:
                grid_label += f" | mean_std={mean_std:.3f}"
            grid = make_sample_grid(samples, mean_img_u8, std_heat, cols=6, label=grid_label)
            Image.fromarray(grid).save(
                os.path.join(output_dir, f"samples_grid_frame_{frame_idx:04d}.png")
            )

            record = {
                "frame_idx": int(frame_idx),
                "seeds": [int(args.seed_base + k) for k in range(K)],
                "num_samples": int(K),
                "num_inference_steps": int(args.num_inference_steps),
                "mean_per_pixel_std": mean_std,
                "max_per_pixel_std": max_std,
            }
            if pw_mat is not None:
                record["mean_pairwise_lpips"] = mean_pw
                record["pairwise_lpips_matrix"] = pw_mat.tolist()
            per_frame_records.append(record)

    # Persist per-scene metrics
    uncertainty_file = os.path.join(output_dir, "uncertainty_metrics.json")
    with open(uncertainty_file, 'w') as f:
        json.dump(
            {
                "scene": scene_hash,
                "num_input_frames": num_input,
                "num_output_frames": num_output,
                "num_samples": K,
                "seed_base": args.seed_base,
                "num_inference_steps": args.num_inference_steps,
                "model_resolution": f"{model_h}x{model_w}",
                "input_mode": input_mode,
                "frames": per_frame_records,
            },
            f,
            indent=2,
        )

    # Scene summary (mean across frames)
    if per_frame_records:
        mean_stds = [r["mean_per_pixel_std"] for r in per_frame_records]
        max_stds = [r["max_per_pixel_std"] for r in per_frame_records]
        scene_summary = {
            "scene": scene_hash,
            "num_frames": len(per_frame_records),
            "mean_per_pixel_std_avg": float(np.mean(mean_stds)),
            "mean_per_pixel_std_std": float(np.std(mean_stds)),
            "max_per_pixel_std_avg": float(np.mean(max_stds)),
        }
        pw_vals = [r["mean_pairwise_lpips"] for r in per_frame_records if "mean_pairwise_lpips" in r]
        if pw_vals:
            scene_summary["mean_pairwise_lpips_avg"] = float(np.mean(pw_vals))
            scene_summary["mean_pairwise_lpips_std"] = float(np.std(pw_vals))
    else:
        scene_summary = {"scene": scene_hash, "num_frames": 0}

    print(f"\n  Scene summary:")
    for k, v in scene_summary.items():
        print(f"    {k}: {v}")
    print(f"  Uncertainty metrics saved to {uncertainty_file}")

    return scene_summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(args.seed_base)
    torch.manual_seed(args.seed_base)
    np.random.seed(args.seed_base)

    print(f"Model resolution: {args.height}x{args.width}")
    print(f"Generation mode: {args.num_input_frames}-to-{args.num_output_frames}")
    print(f"Sampling: K={args.num_samples}, steps={args.num_inference_steps}, seed_base={args.seed_base}")
    print(f"Number of scenes: {len(args.scenes)}")

    pipe = load_pipeline(args)

    lpips_model = None
    if args.sample_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').cuda()
        except ImportError:
            print("Warning: lpips not available, skipping pairwise LPIPS")
            lpips_model = None

    os.makedirs(args.output_path, exist_ok=True)

    scene_summaries = []
    for scene_hash in args.scenes:
        s = process_scene_uncertainty(pipe, args, scene_hash, lpips_model)
        if s is not None:
            scene_summaries.append(s)

    summary_file = os.path.join(args.output_path, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(
            {
                "config": {
                    "checkpoint": args.checkpoint_path,
                    "num_input_frames": args.num_input_frames,
                    "num_output_frames": args.num_output_frames,
                    "num_samples": args.num_samples,
                    "num_inference_steps": args.num_inference_steps,
                    "seed_base": args.seed_base,
                    "height": args.height,
                    "width": args.width,
                    "input_mode": args.input_mode,
                    "zero_temporal_rope": args.zero_temporal_rope,
                    "no_pixel_unshuffle": args.no_pixel_unshuffle,
                },
                "scenes": scene_summaries,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Wan2.1 6-to-1 NVS uncertainty sampling (K seeds × small steps)."
    )

    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained checkpoint (.safetensors)")
    parser.add_argument("--new_in_dim", type=int, default=420,
                        help="New input dimension for the modified model (must match training)")
    parser.add_argument("--num_input_frames", type=int, default=6,
                        help="Number of context frames M. Default: 6.")
    parser.add_argument("--num_output_frames", type=int, default=1,
                        help="Number of target frames N per inference call. Default: 1.")

    # Data paths
    parser.add_argument("--dl3dv_meta_path", type=str, default="/data2/qiwu2/dl3dv10",
                        help="Path to DL3DV-10K metadata (contains train_test_split_M.json per scene)")
    parser.add_argument("--dl3dv_data_path", type=str, default="/data2/qiwu2/DL3DV-10K-test",
                        help="Path to DL3DV-10K test data (contains nerfstudio/ per scene)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for results")

    # Scenes
    parser.add_argument("--scenes", type=str, nargs='+', required=True,
                        help="Scene hashes to process (normally just one)")

    # Resolution
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--input_mode", type=str, default="crop", choices=["stretch", "crop"])

    # Inference settings
    parser.add_argument("--num_inference_steps", type=int, default=5,
                        help="Diffusion steps per sample. Default kept small for budget parity.")
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False)
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False)

    # Uncertainty sampling
    parser.add_argument("--num_samples", type=int, default=10,
                        help="K — number of samples per target frame.")
    parser.add_argument("--seed_base", type=int, default=0,
                        help="Per-sample seed = seed_base + k.")
    parser.add_argument("--sample_lpips", action="store_true", default=True,
                        help="Compute pairwise LPIPS between samples (default on).")
    parser.add_argument("--no_sample_lpips", dest="sample_lpips", action="store_false",
                        help="Disable pairwise LPIPS between samples.")

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} min")
    if torch.cuda.is_available():
        print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
