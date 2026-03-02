#!/usr/bin/env python
# coding=utf-8
"""
Compute PSNR, SSIM, LPIPS, and DreamSim scores for SEVA 6-to-1 NVS results.

SEVA (with crop mode, H=W=576) internally:
  1. Resizes 960x540 images so the shorter side = 576 → 1024x576
  2. Center-crops to 576x576

To fairly compare with Wan2.1 results (evaluated at 192x192), we:
  - GT : 960x540 → resize+center-crop to 576x576 (matching SEVA's crop) → resize to 192x192
  - Pred: 576x576 SEVA output → resize to 192x192

Usage:
    python compute_scores_seva.py \
        --results_dir /data2/qiwu2/dl3dv_test_results_SEVA_batch
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
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SEVA_SIZE = 576          # SEVA output resolution (crop mode → 576x576)
EVAL_SIZE = 480          # Final evaluation resolution for fair comparison

# DL3DV-10K data paths
DEFAULT_DL3DV_META_PATH = "/data2/qiwu2/dl3dv10"
DEFAULT_DL3DV_DATA_PATH = "/data2/qiwu2/DL3DV-10K-test"


# ──────────────────────────────────────────────────────────────────────────────
# Image processing
# ──────────────────────────────────────────────────────────────────────────────

def center_crop(img, crop_h, crop_w=None):
    """Center crop an image to crop_h x crop_w."""
    if crop_w is None:
        crop_w = crop_h
    h, w = img.shape[:2]
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]


def resize_and_center_crop(img, target_size=576):
    """
    Resize image so shorter side = target_size, then center crop to square.
    This matches SEVA's internal "crop" transform_input behaviour:
      960x540 → scale shorter side to 576 → 1024x576 → center crop → 576x576
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    h, w = img.shape[:2]
    scale = max(target_size / h, target_size / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return center_crop(img_resized, target_size, target_size)


def load_gt_frame_seva(scene_hash, frame_idx, transforms_data, dl3dv_data_path):
    """
    Load a GT frame from DL3DV-10K and apply the same crop SEVA uses:
      1. Load 960p image from images_4/
      2. resize_and_center_crop to 576x576 (matching SEVA's crop transform)
      3. Resize to EVAL_SIZE x EVAL_SIZE
    """
    scene_data_path = os.path.join(dl3dv_data_path, scene_hash, "nerfstudio")
    frame_data = transforms_data["frames"][frame_idx]
    file_path = frame_data["file_path"].replace("images/", "images_4/")
    img_path = os.path.join(scene_data_path, file_path)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"GT image not found: {img_path}")

    img_960p = np.array(Image.open(img_path).convert("RGB"))

    # Step 1: Match SEVA's internal crop → 576x576
    img_576 = resize_and_center_crop(img_960p, SEVA_SIZE)

    # Step 2: Resize to evaluation size 192x192
    img_eval = cv2.resize(img_576, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_AREA)
    return img_eval


def load_pred_frame_seva(scene_dir, target_pos):
    """
    Load SEVA's generated frame and resize to evaluation size.
    SEVA saves outputs in seva_output/samples-rgb/{pos:03d}.png at 576x576.
    """
    pred_path = os.path.join(scene_dir, "seva_output", "samples-rgb", f"{target_pos:03d}.png")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"SEVA output not found: {pred_path}")

    img_576 = np.array(Image.open(pred_path).convert("RGB"))

    # Resize to evaluation size 192x192
    img_eval = cv2.resize(img_576, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_AREA)
    return img_eval


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

def save_comparison_image(gt_img, pred_img, frame_idx, psnr, ssim_val, lpips_val, ds_val,
                          save_path, eval_size):
    """Save a side-by-side GT vs Pred comparison image with metrics at the bottom."""
    bar_h = 40
    gap = 10
    canvas_w = eval_size * 2 + gap
    canvas_h = eval_size + bar_h
    comp = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    comp[:eval_size, :eval_size] = gt_img
    comp[:eval_size, eval_size + gap:] = pred_img

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comp, "GT", (10, 30), font, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(comp, "Pred", (eval_size + gap + 10, 30), font, 0.7, (0, 200, 200), 2, cv2.LINE_AA)

    metrics_txt = (f"Frame {frame_idx}:  PSNR={psnr:.2f}  SSIM={ssim_val:.4f}  "
                   f"LPIPS={lpips_val:.4f}  DreamSim={ds_val:.4f}")
    cv2.putText(comp, metrics_txt, (10, eval_size + 28), font, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)

    Image.fromarray(comp).save(save_path)


def process_scene(scene_hash, results_dir, dl3dv_meta_path, dl3dv_data_path,
                  ssim_fn, lpips_model, dreamsim_model, dreamsim_preprocess,
                  save_comparisons=False):
    """
    Process a single SEVA scene: load GT & pred, apply SEVA crop → resize to
    EVAL_SIZE x EVAL_SIZE, compute all four metrics.

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

    # ── Comparison output dir ─────────────────────────────────────────────
    comp_dir = None
    if save_comparisons:
        comp_dir = os.path.join(scene_dir, f"comparisons_{EVAL_SIZE}x{EVAL_SIZE}")
        os.makedirs(comp_dir, exist_ok=True)

    # ── Compute metrics per frame ─────────────────────────────────────────
    psnrs, ssims, lpipss, dreamsims = [], [], [], []
    per_frame = []

    for target_pos, frame_idx in enumerate(tqdm(test_ids, desc=f"  {scene_hash[:12]}...", leave=False)):
        try:
            gt_img = load_gt_frame_seva(scene_hash, frame_idx, transforms_data, dl3dv_data_path)
            pred_img = load_pred_frame_seva(scene_dir, target_pos)
        except FileNotFoundError as e:
            print(f"    [SKIP] {e}")
            continue

        # Sanity check: both should be (EVAL_SIZE, EVAL_SIZE, 3)
        assert gt_img.shape == (EVAL_SIZE, EVAL_SIZE, 3), f"GT shape mismatch: {gt_img.shape}"
        assert pred_img.shape == (EVAL_SIZE, EVAL_SIZE, 3), f"Pred shape mismatch: {pred_img.shape}"

        # PSNR
        psnr = compute_psnr(pred_img, gt_img)
        psnrs.append(psnr)

        # SSIM
        ssim_val = ssim_fn(gt_img, pred_img, multichannel=True, channel_axis=2, data_range=255)
        ssims.append(ssim_val)

        # LPIPS
        lpips_val = compute_lpips(lpips_model, pred_img, gt_img)
        lpipss.append(lpips_val)

        # DreamSim
        ds_val = compute_dreamsim(dreamsim_model, dreamsim_preprocess, pred_img, gt_img)
        dreamsims.append(ds_val)

        per_frame.append({
            "frame_idx": frame_idx,
            "target_pos": target_pos,
            "psnr": psnr,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "dreamsim": ds_val,
        })

        if comp_dir is not None:
            save_comparison_image(
                gt_img, pred_img, frame_idx,
                psnr, ssim_val, lpips_val, ds_val,
                os.path.join(comp_dir, f"comparison_{frame_idx:04d}.png"),
                EVAL_SIZE,
            )
            Image.fromarray(gt_img).save(
                os.path.join(comp_dir, f"gt_{frame_idx:04d}.png"))
            Image.fromarray(pred_img).save(
                os.path.join(comp_dir, f"pred_{frame_idx:04d}.png"))

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
        description="Compute PSNR, SSIM, LPIPS, DreamSim on SEVA NVS results at 192x192"
    )
    parser.add_argument(
        "--results_dir", type=str,
        default="/data2/qiwu2/dl3dv_test_results_SEVA_576x576",
        help="Path to the SEVA results directory",
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
        "--output_json", type=str, default=None,
        help="Path to save per-frame JSON results (optional, defaults to <results_dir>/scores_480x480.json)",
    )
    parser.add_argument(
        "--save_comparisons", action="store_true",
        help="Save side-by-side GT vs Pred comparison images (EVAL_SIZE x EVAL_SIZE) "
             "with per-frame metrics into a subfolder under each scene directory.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if args.output_json is None:
        args.output_json = os.path.join(results_dir, f"scores_{EVAL_SIZE}x{EVAL_SIZE}.json")

    # ── Discover scenes ───────────────────────────────────────────────────
    scene_hashes = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])
    print(f"Found {len(scene_hashes)} scenes in {results_dir}")
    print(f"Evaluation pipeline:")
    print(f"  GT:   960x540 → resize+center-crop to {SEVA_SIZE}x{SEVA_SIZE} → resize to {EVAL_SIZE}x{EVAL_SIZE}")
    print(f"  Pred: {SEVA_SIZE}x{SEVA_SIZE} (SEVA output) → resize to {EVAL_SIZE}x{EVAL_SIZE}")

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
            save_comparisons=args.save_comparisons,
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
          f"SEVA {SEVA_SIZE}x{SEVA_SIZE} → {EVAL_SIZE}x{EVAL_SIZE})")
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
            "seva_resolution": f"{SEVA_SIZE}x{SEVA_SIZE}",
            "eval_resolution": f"{EVAL_SIZE}x{EVAL_SIZE}",
            "gt_pipeline": f"960x540 → resize+center-crop to {SEVA_SIZE}x{SEVA_SIZE} → resize to {EVAL_SIZE}x{EVAL_SIZE}",
            "pred_pipeline": f"{SEVA_SIZE}x{SEVA_SIZE} → resize to {EVAL_SIZE}x{EVAL_SIZE}",
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
    summary_path = os.path.join(results_dir, f"scores_{EVAL_SIZE}x{EVAL_SIZE}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"SEVA evaluation at {EVAL_SIZE}x{EVAL_SIZE}\n")
        f.write(f"  GT:   960x540 → resize+center-crop to {SEVA_SIZE}x{SEVA_SIZE} → resize to {EVAL_SIZE}x{EVAL_SIZE}\n")
        f.write(f"  Pred: {SEVA_SIZE}x{SEVA_SIZE} → resize to {EVAL_SIZE}x{EVAL_SIZE}\n")
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

