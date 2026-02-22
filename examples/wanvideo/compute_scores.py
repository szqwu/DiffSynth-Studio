#!/usr/bin/env python
# coding=utf-8
"""
Compute PSNR, SSIM, LPIPS, and DreamSim scores for Wan2.1 6-to-1 NVS results.

For each scene, loads generated frames and corresponding GT frames from DL3DV-10K,
prepares GT to model resolution using one of two modes:
  - crop:    center-crop original to match model aspect ratio, then resize (default)
  - stretch: directly resize original to model resolution (may distort)
Then center-crops both GT and predicted to CROP_SIZE x CROP_SIZE before computing
metrics.

Usage:
    python compute_scores.py \
        --results_dir /data2/qiwu2/dl3dv_test_results_wan21_6to1_random79 \
        --mode crop

    python compute_scores.py \
        --results_dir /data2/qiwu2/dl3dv_test_results_wan21_6to1_random79 \
        --mode stretch
"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Configuration (must match test_wan2.1_6to1.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
# MODEL_H = 192
# MODEL_W = 336
MODEL_H = 480
MODEL_W = 720
DEFAULT_EVAL_SIZE = 576

# DL3DV-10K data paths (same defaults as test_wan2.1_6to1.py)
DEFAULT_DL3DV_META_PATH = "/data2/qiwu2/dl3dv10"
DEFAULT_DL3DV_DATA_PATH = "/data2/qiwu2/DL3DV-10K-test"

# DL3DV images_4 resolution (960x540)
ACTUAL_W = 960
ACTUAL_H = 540


# ──────────────────────────────────────────────────────────────────────────────
# Image processing (matching test_wan2.1_6to1.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

def center_crop(img, crop_h, crop_w):
    """Center crop an image to crop_h x crop_w."""
    h, w = img.shape[:2]
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]


def resize_and_center_crop(img, target_size):
    """
    Resize image so shorter side = target_size, then center crop to square.
    Matches resize_and_center_crop in test_wan2.1_6to1.py exactly.
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    h, w = img.shape[:2]
    if h < w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return center_crop(img_resized, target_size, target_size)


def load_gt_frame(scene_hash, frame_idx, transforms_data, dl3dv_data_path,
                  mode="crop"):
    """
    Load a GT frame from DL3DV-10K and bring it to model resolution
    (MODEL_H x MODEL_W).

    Two modes:
      - stretch: directly resize original to (MODEL_W, MODEL_H).
                 Fast but may distort if aspect ratios differ.
      - crop:    resize original to cover (MODEL_H, MODEL_W), then center-crop.
                 Preserves aspect ratio. Matches resize_crop_to_rect.
    """
    scene_data_path = os.path.join(dl3dv_data_path, scene_hash, "nerfstudio")
    frame_data = transforms_data["frames"][frame_idx]
    file_path = frame_data["file_path"].replace("images/", "images_4/")
    img_path = os.path.join(scene_data_path, file_path)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"GT image not found: {img_path}")

    # Load as RGB numpy array
    img_orig = np.array(Image.open(img_path).convert("RGB"))

    if mode == "stretch":
        # Directly resize to model resolution (may distort)
        img_model = cv2.resize(img_orig, (MODEL_W, MODEL_H),
                               interpolation=cv2.INTER_AREA)
    elif mode == "crop":
        # Resize to cover model resolution, then center-crop
        # (matches resize_crop_to_rect in test_wan2.1_6to1.py)
        orig_h, orig_w = img_orig.shape[:2]
        scale = max(MODEL_H / orig_h, MODEL_W / orig_w)
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))

        img_resized = cv2.resize(img_orig, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
        crop_offset_y = (new_h - MODEL_H) // 2
        crop_offset_x = (new_w - MODEL_W) // 2
        img_model = img_resized[crop_offset_y:crop_offset_y + MODEL_H,
                                crop_offset_x:crop_offset_x + MODEL_W]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return img_model


def load_pred_frame(scene_dir, frame_idx):
    """Load the generated frame saved by test_wan2.1_6to1.py."""
    fname = f"generated_frame_{frame_idx:04d}_{MODEL_H}x{MODEL_W}.png"
    pred_path = os.path.join(scene_dir, fname)

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Generated frame not found: {pred_path}")

    return np.array(Image.open(pred_path).convert("RGB"))


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred, gt):
    """Compute PSNR between two uint8 numpy arrays."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def init_ssim():
    """Import SSIM function from skimage."""
    from skimage.metrics import structural_similarity as ssim_func
    return ssim_func


def init_lpips():
    """Load LPIPS (AlexNet) model."""
    import lpips
    model = lpips.LPIPS(net="alex").cuda().eval()
    return model


def compute_lpips(lpips_model, pred, gt):
    """Compute LPIPS between two uint8 numpy arrays (H, W, 3)."""
    pred_t = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    gt_t = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    with torch.no_grad():
        return lpips_model(pred_t.cuda(), gt_t.cuda()).item()


def init_dreamsim():
    """Load DreamSim model."""
    from dreamsim import dreamsim
    model, preprocess = dreamsim(pretrained=True, device="cuda")
    return model, preprocess


def compute_dreamsim(dreamsim_model, dreamsim_preprocess, pred, gt):
    """Compute DreamSim distance between two uint8 numpy arrays."""
    pred_pil = Image.fromarray(pred)
    gt_pil = Image.fromarray(gt)

    pred_t = dreamsim_preprocess(pred_pil).to("cuda")
    gt_t = dreamsim_preprocess(gt_pil).to("cuda")

    # Ensure 4D tensors (batch dim)
    if pred_t.dim() == 3:
        pred_t = pred_t.unsqueeze(0)
    if gt_t.dim() == 3:
        gt_t = gt_t.unsqueeze(0)
    while pred_t.dim() > 4:
        pred_t = pred_t.squeeze(0)
    while gt_t.dim() > 4:
        gt_t = gt_t.squeeze(0)

    with torch.no_grad():
        return dreamsim_model(pred_t, gt_t).item()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def process_scene(scene_hash, results_dir, dl3dv_meta_path, dl3dv_data_path,
                  ssim_fn, lpips_model, dreamsim_model, dreamsim_preprocess,
                  mode="crop", eval_size=DEFAULT_EVAL_SIZE):
    """
    Process a single scene: load GT (via crop or stretch) & pred,
    resize_and_center_crop both to eval_size x eval_size, compute all metrics.

    Returns dict with per-frame and mean metrics, or None on failure.
    """
    scene_dir = os.path.join(results_dir, scene_hash)
    if not os.path.isdir(scene_dir):
        print(f"  [SKIP] Scene directory not found: {scene_dir}")
        return None

    # ── Load train/test split ─────────────────────────────────────────────
    split_file = os.path.join(dl3dv_meta_path, scene_hash, "train_test_split_6.json")
    if not os.path.exists(split_file):
        print(f"  [SKIP] Split file not found: {split_file}")
        return None

    with open(split_file, "r") as f:
        split_data = json.load(f)
    test_ids = split_data["test_ids"]

    # ── Load transforms.json ──────────────────────────────────────────────
    transforms_file = os.path.join(dl3dv_data_path, scene_hash, "nerfstudio", "transforms.json")
    if not os.path.exists(transforms_file):
        print(f"  [SKIP] Transforms not found: {transforms_file}")
        return None

    with open(transforms_file, "r") as f:
        transforms_data = json.load(f)

    # ── Compute metrics per frame ─────────────────────────────────────────
    psnrs, ssims, lpipss, dreamsims = [], [], [], []
    per_frame = []

    for frame_idx in tqdm(test_ids, desc=f"  {scene_hash[:12]}...", leave=False):
        try:
            gt_img = load_gt_frame(scene_hash, frame_idx, transforms_data,
                                   dl3dv_data_path, mode=mode)
            pred_img = load_pred_frame(scene_dir, frame_idx)
        except FileNotFoundError as e:
            print(f"    [SKIP] {e}")
            continue

        # Sanity check: both should be (MODEL_H, MODEL_W, 3)
        assert gt_img.shape == (MODEL_H, MODEL_W, 3), f"GT shape mismatch: {gt_img.shape}"
        assert pred_img.shape == (MODEL_H, MODEL_W, 3), f"Pred shape mismatch: {pred_img.shape}"

        # Resize and center crop both to eval_size x eval_size for evaluation
        gt_crop = resize_and_center_crop(gt_img, eval_size)
        pred_crop = resize_and_center_crop(pred_img, eval_size)

        # PSNR
        psnr = compute_psnr(pred_crop, gt_crop)
        psnrs.append(psnr)

        # SSIM
        ssim_val = ssim_fn(gt_crop, pred_crop, multichannel=True, channel_axis=2, data_range=255)
        ssims.append(ssim_val)

        # LPIPS
        lpips_val = compute_lpips(lpips_model, pred_crop, gt_crop)
        lpipss.append(lpips_val)

        # DreamSim
        ds_val = compute_dreamsim(dreamsim_model, dreamsim_preprocess, pred_crop, gt_crop)
        dreamsims.append(ds_val)

        per_frame.append({
            "frame_idx": frame_idx,
            "psnr": psnr,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "dreamsim": ds_val,
        })

    if not psnrs:
        print(f"  [SKIP] No valid frames for scene {scene_hash}")
        return None

    scene_result = {
        "scene": scene_hash,
        "num_frames": len(psnrs),
        "mean_psnr": float(np.mean(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "mean_lpips": float(np.mean(lpipss)),
        "mean_dreamsim": float(np.mean(dreamsims)),
        "per_frame": per_frame,
    }

    print(f"  Scene {scene_hash[:12]}...  "
          f"PSNR={scene_result['mean_psnr']:.2f}  "
          f"SSIM={scene_result['mean_ssim']:.4f}  "
          f"LPIPS={scene_result['mean_lpips']:.4f}  "
          f"DreamSim={scene_result['mean_dreamsim']:.4f}  "
          f"({len(psnrs)} frames)")

    return scene_result


def main():
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, LPIPS, DreamSim on Wan2.1 6-to-1 NVS results"
    )
    parser.add_argument(
        "--results_dir", type=str,
        default="/data2/qiwu2/dl3dv_test_results_10000",
        help="Path to the results directory generated by test_wan2.1_6to1.sh",
    )
    parser.add_argument(
        "--dl3dv_meta_path", type=str,
        default=DEFAULT_DL3DV_META_PATH,
        help="Path to DL3DV-10K metadata (contains train_test_split_6.json per scene)",
    )
    parser.add_argument(
        "--dl3dv_data_path", type=str,
        default=DEFAULT_DL3DV_DATA_PATH,
        help="Path to DL3DV-10K test data (contains nerfstudio/ per scene)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["crop", "stretch"], default="stretch",
        help="How to load GT from original resolution to model resolution: "
             "'crop' = center-crop to match aspect ratio then resize (default), "
             "'stretch' = directly resize (may distort)",
    )
    parser.add_argument(
        "--eval_size", type=int, default=DEFAULT_EVAL_SIZE,
        help="Evaluation size: resize shorter side to this, then center crop to square "
             f"(default: {DEFAULT_EVAL_SIZE}, matches test_wan2.1_6to1.py --eval_size)",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Path to save per-frame JSON results (optional, defaults to "
             "<results_dir>/scores_<mode>_<eval_size>x<eval_size>.json)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    mode = args.mode
    eval_size = args.eval_size
    if args.output_json is None:
        args.output_json = os.path.join(results_dir, f"scores_{mode}_{eval_size}x{eval_size}.json")

    # ── Discover scenes ───────────────────────────────────────────────────
    scene_hashes = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])
    print(f"Found {len(scene_hashes)} scenes in {results_dir}")
    print(f"Evaluation mode: {mode}  {MODEL_H}x{MODEL_W} → {eval_size}x{eval_size}")

    # ── Initialize metric models ──────────────────────────────────────────
    print("\nInitializing metrics...")
    ssim_fn = init_ssim()
    print("  ✓ SSIM (skimage)")

    lpips_model = init_lpips()
    print("  ✓ LPIPS (AlexNet)")

    dreamsim_model, dreamsim_preprocess = init_dreamsim()
    print("  ✓ DreamSim")
    print()

    # ── Process each scene ────────────────────────────────────────────────
    all_results = []
    all_psnr, all_ssim, all_lpips, all_dreamsim = [], [], [], []

    for scene_hash in scene_hashes:
        result = process_scene(
            scene_hash, results_dir,
            args.dl3dv_meta_path, args.dl3dv_data_path,
            ssim_fn, lpips_model, dreamsim_model, dreamsim_preprocess,
            mode=mode, eval_size=eval_size,
        )
        if result is not None:
            all_results.append(result)
            all_psnr.append(result["mean_psnr"])
            all_ssim.append(result["mean_ssim"])
            all_lpips.append(result["mean_lpips"])
            all_dreamsim.append(result["mean_dreamsim"])

    # ── Overall summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"OVERALL RESULTS  ({len(all_results)} scenes, "
          f"{mode} {MODEL_H}x{MODEL_W} → {eval_size}x{eval_size})")
    print(f"{'=' * 80}")

    if all_psnr:
        print(f"  Mean PSNR:     {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
        print(f"  Mean SSIM:     {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
        print(f"  Mean LPIPS:    {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}")
        print(f"  Mean DreamSim: {np.mean(all_dreamsim):.4f} ± {np.std(all_dreamsim):.4f}")
    else:
        print("  No valid results.")
    print(f"{'=' * 80}")

    # ── Per-scene table ───────────────────────────────────────────────────
    if all_results:
        print(f"\nPer-scene breakdown:")
        print(f"  {'Scene':<16s}  {'PSNR':>6s}  {'SSIM':>6s}  {'LPIPS':>6s}  {'DreamSim':>8s}  {'#Frames':>7s}")
        print(f"  {'-'*16}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}")
        for r in all_results:
            print(f"  {r['scene'][:16]}  {r['mean_psnr']:6.2f}  {r['mean_ssim']:6.4f}  "
                  f"{r['mean_lpips']:6.4f}  {r['mean_dreamsim']:8.4f}  {r['num_frames']:>7d}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "config": {
            "results_dir": results_dir,
            "model_resolution": f"{MODEL_H}x{MODEL_W}",
            "eval_mode": mode,
            "eval_size": f"{eval_size}x{eval_size}",
            "dl3dv_meta_path": args.dl3dv_meta_path,
            "dl3dv_data_path": args.dl3dv_data_path,
        },
        "overall": {
            "num_scenes": len(all_results),
            "mean_psnr": float(np.mean(all_psnr)) if all_psnr else None,
            "std_psnr": float(np.std(all_psnr)) if all_psnr else None,
            "mean_ssim": float(np.mean(all_ssim)) if all_ssim else None,
            "std_ssim": float(np.std(all_ssim)) if all_ssim else None,
            "mean_lpips": float(np.mean(all_lpips)) if all_lpips else None,
            "std_lpips": float(np.std(all_lpips)) if all_lpips else None,
            "mean_dreamsim": float(np.mean(all_dreamsim)) if all_dreamsim else None,
            "std_dreamsim": float(np.std(all_dreamsim)) if all_dreamsim else None,
        },
        "per_scene": all_results,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    # ── Also save a concise summary text file ─────────────────────────────
    summary_path = os.path.join(results_dir, f"scores_{mode}_{eval_size}x{eval_size}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Evaluation: {mode} {MODEL_H}x{MODEL_W} → {eval_size}x{eval_size}\n")
        f.write(f"Results dir: {results_dir}\n\n")
        if all_psnr:
            f.write(f"Mean PSNR:     {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB\n")
            f.write(f"Mean SSIM:     {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}\n")
            f.write(f"Mean LPIPS:    {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}\n")
            f.write(f"Mean DreamSim: {np.mean(all_dreamsim):.4f} ± {np.std(all_dreamsim):.4f}\n")
        f.write(f"\nPer-scene:\n")
        for r in all_results:
            f.write(f"  {r['scene'][:16]}...  "
                    f"PSNR={r['mean_psnr']:.2f}  SSIM={r['mean_ssim']:.4f}  "
                    f"LPIPS={r['mean_lpips']:.4f}  DreamSim={r['mean_dreamsim']:.4f}  "
                    f"({r['num_frames']} frames)\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

