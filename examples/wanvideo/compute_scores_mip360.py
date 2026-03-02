#!/usr/bin/env python
# coding=utf-8
"""
Compute PSNR, SSIM, LPIPS, and DreamSim scores for Wan2.1 6-to-1 NVS results
on the mip-NeRF 360 dataset.

For each scene, loads generated frames and corresponding GT from images_2,
resizes+center-crops GT to model resolution, then center-crops both to
eval_h x eval_w before computing metrics.

Usage:
    python compute_scores_mip360.py \
        --results_dir /data2/qiwu2/mip360_test_results_wan21_6to1

    python compute_scores_mip360.py \
        --results_dir /data2/qiwu2/mip360_test_results_wan21_6to1 \
        --model_h 480 --model_w 832 --eval_h 480 --eval_w 480
"""

import os
import json
import struct
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch


# ──────────────────────────────────────────────────────────────────────────────
# COLMAP reader (only images.bin needed for sorted name list)
# ──────────────────────────────────────────────────────────────────────────────

COLMAP_CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 5, 3: 8, 4: 8, 5: 12, 6: 4, 7: 5, 8: 4,
}


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            f.read(32 + 24)  # skip qvec + tvec
            camera_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)
            images[image_id] = {"name": name}
    return images


def get_sorted_image_names(scene_data_path):
    """Get image filenames sorted by name (same ordering as test_wan2.1_mip360.py)."""
    images_bin = os.path.join(scene_data_path, "sparse", "0", "images.bin")
    imgs = read_images_binary(images_bin)
    return [img["name"] for img in sorted(imgs.values(), key=lambda x: x["name"])]


# ──────────────────────────────────────────────────────────────────────────────
# Image processing
# ──────────────────────────────────────────────────────────────────────────────

def center_crop(img, crop_h, crop_w):
    h, w = img.shape[:2]
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]


def resize_crop_to_rect(img, target_h, target_w):
    """Resize to cover (target_h, target_w), then center crop."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    h, w = img.shape[:2]
    scale = max(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    crop_y = (new_h - target_h) // 2
    crop_x = (new_w - target_w) // 2
    return img_resized[crop_y:crop_y + target_h, crop_x:crop_x + target_w]


def load_gt_frame(scene_data_path, image_name, model_h, model_w):
    """Load GT from images_2 and resize+center-crop to model resolution."""
    img_path = os.path.join(scene_data_path, "images_2", image_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"GT image not found: {img_path}")
    img = np.array(Image.open(img_path).convert("RGB"))
    return resize_crop_to_rect(img, model_h, model_w)


def load_pred_frame(scene_dir, frame_idx, model_h, model_w):
    """Load the generated frame saved by test_wan2.1_mip360.py."""
    fname = f"generated_frame_{frame_idx:04d}_{model_h}x{model_w}.png"
    pred_path = os.path.join(scene_dir, fname)
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Generated frame not found: {pred_path}")
    return np.array(Image.open(pred_path).convert("RGB"))


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred, gt):
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def init_ssim():
    from skimage.metrics import structural_similarity as ssim_func
    return ssim_func


def init_lpips():
    import lpips
    return lpips.LPIPS(net="alex").cuda().eval()


def compute_lpips(lpips_model, pred, gt):
    pred_t = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    gt_t = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    with torch.no_grad():
        return lpips_model(pred_t.cuda(), gt_t.cuda()).item()


def init_dreamsim():
    from dreamsim import dreamsim
    model, preprocess = dreamsim(pretrained=True, device="cuda")
    return model, preprocess


def compute_dreamsim(dreamsim_model, dreamsim_preprocess, pred, gt):
    pred_t = dreamsim_preprocess(Image.fromarray(pred)).to("cuda")
    gt_t = dreamsim_preprocess(Image.fromarray(gt)).to("cuda")
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
# Scene processing
# ──────────────────────────────────────────────────────────────────────────────

def save_comparison_image(gt_crop, pred_crop, frame_idx, image_name,
                          psnr, ssim_val, lpips_val, ds_val, save_path):
    img_h, img_w = gt_crop.shape[:2]
    bar_h = 40
    gap = 10
    canvas_w = img_w * 2 + gap
    canvas_h = img_h + bar_h
    comp = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    comp[:img_h, :img_w] = gt_crop
    comp[:img_h, img_w + gap:] = pred_crop

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comp, "GT", (10, 30), font, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(comp, "Pred", (img_w + gap + 10, 30), font, 0.7, (0, 200, 200), 2, cv2.LINE_AA)

    metrics_txt = (f"{image_name} (idx {frame_idx}):  PSNR={psnr:.2f}  SSIM={ssim_val:.4f}  "
                   f"LPIPS={lpips_val:.4f}  DS={ds_val:.4f}")
    cv2.putText(comp, metrics_txt, (10, img_h + 28), font, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA)

    Image.fromarray(comp).save(save_path)


def process_scene(scene_name, results_dir, mip360_data_path, mip360_split_path,
                  ssim_fn, lpips_model, dreamsim_model, dreamsim_preprocess,
                  model_h, model_w, eval_h, eval_w,
                  save_comparisons=False, no_center_crop=False):
    scene_dir = os.path.join(results_dir, scene_name)
    if not os.path.isdir(scene_dir):
        print(f"  [SKIP] Scene directory not found: {scene_dir}")
        return None

    split_file = os.path.join(mip360_split_path, scene_name, "train_test_split_6.json")
    if not os.path.exists(split_file):
        print(f"  [SKIP] Split file not found: {split_file}")
        return None

    with open(split_file, "r") as f:
        split_data = json.load(f)
    test_ids = split_data["test_ids"]

    scene_data_path = os.path.join(mip360_data_path, scene_name)
    image_names = get_sorted_image_names(scene_data_path)

    comp_dir = None
    if save_comparisons:
        tag = f"full_{model_h}x{model_w}" if no_center_crop else f"{eval_h}x{eval_w}"
        comp_dir = os.path.join(scene_dir, f"comparisons_{tag}")
        os.makedirs(comp_dir, exist_ok=True)

    psnrs, ssims, lpipss, dreamsims = [], [], [], []
    per_frame = []

    for frame_idx in tqdm(test_ids, desc=f"  {scene_name}", leave=False):
        if frame_idx >= len(image_names):
            print(f"    [SKIP] Index {frame_idx} out of range ({len(image_names)} images)")
            continue

        img_name = image_names[frame_idx]

        try:
            gt_img = load_gt_frame(scene_data_path, img_name, model_h, model_w)
            pred_img = load_pred_frame(scene_dir, frame_idx, model_h, model_w)
        except FileNotFoundError as e:
            print(f"    [SKIP] {e}")
            continue

        assert gt_img.shape == (model_h, model_w, 3), f"GT shape mismatch: {gt_img.shape}"
        assert pred_img.shape == (model_h, model_w, 3), f"Pred shape mismatch: {pred_img.shape}"

        if no_center_crop:
            gt_crop = gt_img
            pred_crop = pred_img
        else:
            gt_crop = center_crop(gt_img, eval_h, eval_w)
            pred_crop = center_crop(pred_img, eval_h, eval_w)

        psnr = compute_psnr(pred_crop, gt_crop)
        psnrs.append(psnr)

        ssim_val = ssim_fn(gt_crop, pred_crop, multichannel=True, channel_axis=2, data_range=255)
        ssims.append(ssim_val)

        lpips_val = compute_lpips(lpips_model, pred_crop, gt_crop)
        lpipss.append(lpips_val)

        ds_val = compute_dreamsim(dreamsim_model, dreamsim_preprocess, pred_crop, gt_crop)
        dreamsims.append(ds_val)

        per_frame.append({
            "frame_idx": frame_idx,
            "image_name": img_name,
            "psnr": psnr,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "dreamsim": ds_val,
        })

        if comp_dir is not None:
            save_comparison_image(
                gt_crop, pred_crop, frame_idx, img_name,
                psnr, ssim_val, lpips_val, ds_val,
                os.path.join(comp_dir, f"comparison_{frame_idx:04d}.png"),
            )

    if not psnrs:
        print(f"  [SKIP] No valid frames for scene {scene_name}")
        return None

    scene_result = {
        "scene": scene_name,
        "num_frames": len(psnrs),
        "mean_psnr": float(np.mean(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "mean_lpips": float(np.mean(lpipss)),
        "mean_dreamsim": float(np.mean(dreamsims)),
        "per_frame": per_frame,
    }

    print(f"  {scene_name:<12s}  "
          f"PSNR={scene_result['mean_psnr']:.2f}  "
          f"SSIM={scene_result['mean_ssim']:.4f}  "
          f"LPIPS={scene_result['mean_lpips']:.4f}  "
          f"DreamSim={scene_result['mean_dreamsim']:.4f}  "
          f"({len(psnrs)} frames)")

    return scene_result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, LPIPS, DreamSim on Wan2.1 6-to-1 NVS results (mip-NeRF 360)"
    )
    parser.add_argument("--results_dir", type=str,
                        default="/data2/qiwu2/mip360_test_results_wan21_6to1")
    parser.add_argument("--mip360_data_path", type=str, default="/data2/qiwu2/mip360",
                        help="Path to mip-NeRF 360 data (scene dirs with images_2/ and sparse/)")
    parser.add_argument("--mip360_split_path", type=str, default="/data2/qiwu2/mipnerf360",
                        help="Path to train/test split files")
    parser.add_argument("--model_h", type=int, default=480)
    parser.add_argument("--model_w", type=int, default=832)
    parser.add_argument("--eval_h", type=int, default=480)
    parser.add_argument("--eval_w", type=int, default=480)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--no_center_crop", action="store_true",
                        help="Evaluate at full model resolution without center crop")
    parser.add_argument("--save_comparisons", action="store_true",
                        help="Save side-by-side GT vs Pred comparison images")
    args = parser.parse_args()

    model_h = args.model_h
    model_w = args.model_w
    eval_h = args.eval_h
    eval_w = args.eval_w
    no_center_crop = args.no_center_crop

    if no_center_crop:
        eval_tag = f"full_{model_h}x{model_w}"
    else:
        eval_tag = f"{eval_h}x{eval_w}"

    if args.output_json is None:
        args.output_json = os.path.join(args.results_dir, f"scores_crop_{eval_tag}.json")

    # ── Discover scenes ──────────────────────────────────────────────────
    scene_names = sorted([
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    ])
    print(f"Found {len(scene_names)} scenes in {args.results_dir}")
    if no_center_crop:
        print(f"Evaluation: {model_h}x{model_w} (full resolution, no center crop)")
    else:
        print(f"Evaluation: {model_h}x{model_w} → center crop {eval_h}x{eval_w}")

    # ── Initialize metric models ─────────────────────────────────────────
    print("\nInitializing metrics...")
    ssim_fn = init_ssim()
    print("  SSIM (skimage) ready")
    lpips_model = init_lpips()
    print("  LPIPS (AlexNet) ready")
    dreamsim_model, dreamsim_preprocess = init_dreamsim()
    print("  DreamSim ready")
    print()

    # ── Process each scene ───────────────────────────────────────────────
    all_results = []
    all_psnr, all_ssim, all_lpips, all_dreamsim = [], [], [], []

    for scene_name in scene_names:
        result = process_scene(
            scene_name, args.results_dir,
            args.mip360_data_path, args.mip360_split_path,
            ssim_fn, lpips_model, dreamsim_model, dreamsim_preprocess,
            model_h, model_w, eval_h, eval_w,
            save_comparisons=args.save_comparisons,
            no_center_crop=no_center_crop,
        )
        if result is not None:
            all_results.append(result)
            all_psnr.append(result["mean_psnr"])
            all_ssim.append(result["mean_ssim"])
            all_lpips.append(result["mean_lpips"])
            all_dreamsim.append(result["mean_dreamsim"])

    # ── Overall summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"OVERALL RESULTS  ({len(all_results)} scenes, {model_h}x{model_w} → {eval_tag})")
    print(f"{'=' * 80}")

    if all_psnr:
        print(f"  Mean PSNR:     {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB")
        print(f"  Mean SSIM:     {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f}")
        print(f"  Mean LPIPS:    {np.mean(all_lpips):.4f} +/- {np.std(all_lpips):.4f}")
        print(f"  Mean DreamSim: {np.mean(all_dreamsim):.4f} +/- {np.std(all_dreamsim):.4f}")
    else:
        print("  No valid results.")
    print(f"{'=' * 80}")

    if all_results:
        print(f"\nPer-scene breakdown:")
        print(f"  {'Scene':<12s}  {'PSNR':>6s}  {'SSIM':>6s}  {'LPIPS':>6s}  {'DreamSim':>8s}  {'#Frames':>7s}")
        print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}")
        for r in all_results:
            print(f"  {r['scene']:<12s}  {r['mean_psnr']:6.2f}  {r['mean_ssim']:6.4f}  "
                  f"{r['mean_lpips']:6.4f}  {r['mean_dreamsim']:8.4f}  {r['num_frames']:>7d}")

    # ── Save JSON ────────────────────────────────────────────────────────
    output = {
        "config": {
            "results_dir": args.results_dir,
            "model_resolution": f"{model_h}x{model_w}",
            "eval_size": eval_tag,
            "no_center_crop": no_center_crop,
            "mip360_data_path": args.mip360_data_path,
            "mip360_split_path": args.mip360_split_path,
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

    # ── Summary text ─────────────────────────────────────────────────────
    summary_path = os.path.join(args.results_dir, f"scores_crop_{eval_tag}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Evaluation: {model_h}x{model_w} → {eval_tag}\n")
        f.write(f"Results dir: {args.results_dir}\n\n")
        if all_psnr:
            f.write(f"Mean PSNR:     {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB\n")
            f.write(f"Mean SSIM:     {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f}\n")
            f.write(f"Mean LPIPS:    {np.mean(all_lpips):.4f} +/- {np.std(all_lpips):.4f}\n")
            f.write(f"Mean DreamSim: {np.mean(all_dreamsim):.4f} +/- {np.std(all_dreamsim):.4f}\n")
        f.write(f"\nPer-scene:\n")
        for r in all_results:
            f.write(f"  {r['scene']:<12s}  "
                    f"PSNR={r['mean_psnr']:.2f}  SSIM={r['mean_ssim']:.4f}  "
                    f"LPIPS={r['mean_lpips']:.4f}  DreamSim={r['mean_dreamsim']:.4f}  "
                    f"({r['num_frames']} frames)\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
