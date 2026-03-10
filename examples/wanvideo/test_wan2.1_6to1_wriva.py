#!/usr/bin/env python3
"""
Wan2.1 6-to-1 NVS evaluation on WRIVA scenes.

For each WRIVA scene:
  - Loads GT camera metadata from reference/metadata for all reference images
  - Randomly picks 6 reference images as context frames
  - For each remaining reference image, runs 6-to-1 NVS inference
  - Saves GT, inputs, predictions, side-by-side comparisons at model resolution
"""

import os
import sys
import time
import json
import math
import argparse
import random

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.core.data.my_v2v_dataset_images_in_plucker_SE import (
    get_plucker_rays,
    normalize_w2c_make_cam_last_origin,
)

R_EARTH = 6_371_000.0


# ── Model setup (same as test_wan2.1_6to1.py) ───────────────────────────────

def modify_model_channels(pipe, model_attr, new_in_dim, device):
    model = getattr(pipe, model_attr)
    if model is None:
        return

    old_in_dim = model.in_dim
    old_out_dim = model.out_dim

    print(f"Modifying {model_attr} input channels: in_dim {old_in_dim}->{new_in_dim}")

    new_model = WanModel(
        dim=model.dim,
        in_dim=new_in_dim,
        ffn_dim=model.ffn_dim,
        out_dim=old_out_dim,
        text_dim=model.text_embedding[0].in_features,
        freq_dim=model.freq_dim,
        eps=1e-6,
        patch_size=model.patch_size,
        num_heads=model.num_heads,
        num_layers=model.num_layers,
        has_image_input=model.has_image_input,
        has_image_pos_emb=model.has_image_pos_emb,
        has_ref_conv=model.has_ref_conv,
        add_control_adapter=model.control_adapter is not None,
        in_dim_control_adapter=24 if model.control_adapter is not None else 24,
        seperated_timestep=model.seperated_timestep,
        require_vae_embedding=model.require_vae_embedding,
        require_clip_embedding=model.require_clip_embedding,
        fuse_vae_embedding_in_latents=model.fuse_vae_embedding_in_latents,
        fuse_vae_embedding_in_latents_multiple=False,
        seperated_encoding=True,
    )

    pretrained_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    new_state_dict = new_model.state_dict()

    for key, value in pretrained_state_dict.items():
        if key.startswith("patch_embedding"):
            print(f"  Skipping {key} — will be loaded from checkpoint")
            continue
        if key in new_state_dict and value.shape == new_state_dict[key].shape:
            new_state_dict[key] = value

    new_model.load_state_dict(new_state_dict, strict=False)

    model.cpu()
    del model
    del pretrained_state_dict
    torch.cuda.empty_cache()
    import gc; gc.collect()

    new_model = new_model.to(device=device, dtype=torch.bfloat16)
    setattr(pipe, model_attr, new_model)
    print(f"Model {model_attr} channels modified successfully")


def load_pipeline(args):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        vram_limit=args.vram_limit,
    )

    modify_model_channels(pipe, "dit", args.new_in_dim, "cuda")

    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = load_state_dict(args.checkpoint_path, torch_dtype=torch.bfloat16, device="cuda")

    has_lora_keys = any(
        "lora_A" in k or "lora_B" in k or "lora_up" in k or "lora_down" in k
        for k in checkpoint.keys()
    )

    if has_lora_keys:
        pipe.load_lora(pipe.dit, state_dict=checkpoint, alpha=1.0)
        print("LoRA weights loaded")
        patch_emb_state = {k: v for k, v in checkpoint.items() if "patch_embedding" in k}
        if patch_emb_state:
            pipe.dit.load_state_dict(patch_emb_state, strict=False)
            print(f"Loaded {len(patch_emb_state)} patch_embedding parameters")
        else:
            print("Warning: No patch_embedding weights found in checkpoint!")
    else:
        load_result = pipe.dit.load_state_dict(checkpoint, strict=False)
        missing = [k for k in load_result.missing_keys if k not in checkpoint]
        unexpected = load_result.unexpected_keys
        print(f"Full checkpoint loaded — {len(checkpoint)} keys")
        if missing:
            print(f"  Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    return pipe


# ── WRIVA coordinate conversion ─────────────────────────────────────────────

def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Convert geodetic coords (degrees, meters) to local ENU (meters)."""
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    lat0_rad = math.radians(lat0)
    east  = dlon * R_EARTH * math.cos(lat0_rad)
    north = dlat * R_EARTH
    up    = alt - alt0
    return np.array([east, north, up], dtype=np.float32)


def opk_to_w2c_rotation(omega_deg, phi_deg, kappa_deg):
    """
    Pix4D omega/phi/kappa → 3×3 w2c rotation matrix (OpenCV convention).
    M = Rz(kappa) @ Ry(phi) @ Rx(omega)
    Transforms world coords (ENU) to camera coords (X-right, Y-down, Z-forward).
    """
    o = math.radians(omega_deg)
    p = math.radians(phi_deg)
    k = math.radians(kappa_deg)

    Rx = np.array([
        [1, 0, 0],
        [0,  math.cos(o), -math.sin(o)],
        [0,  math.sin(o),  math.cos(o)]
    ], dtype=np.float32)
    Ry = np.array([
        [ math.cos(p), 0, math.sin(p)],
        [0, 1, 0],
        [-math.sin(p), 0, math.cos(p)]
    ], dtype=np.float32)
    Rz = np.array([
        [math.cos(k), -math.sin(k), 0],
        [math.sin(k),  math.cos(k), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)


def build_c2w_opengl(lat, lon, alt, omega, phi, kappa, lat0, lon0, alt0):
    """
    Build a 4×4 c2w matrix in OpenGL convention (Y-up, Z-backward).

    Pix4D: M = Rz(κ)Ry(φ)Rx(ω) is the w2c rotation in OpenCV convention
    (camera: X-right, Y-down, Z-forward).
    To get c2w in OpenGL (X-right, Y-up, Z-backward):
        c2w_gl[:3,:3] = M^T @ diag(1, -1, -1)
    """
    pos_enu = geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0)
    M = opk_to_w2c_rotation(omega, phi, kappa)
    # M is w2c (OpenCV). c2w_cv = M^T. Flip Y,Z columns for OpenGL.
    FLIP_YZ = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_c2w_gl = M.T @ FLIP_YZ

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w_gl
    c2w[:3, 3]  = pos_enu
    return c2w


# ── Image and intrinsics helpers ─────────────────────────────────────────────

def get_fx_fy(intrinsics_dict, projection):
    """Extract pinhole-equivalent fx, fy from WRIVA intrinsics metadata."""
    if projection == "fisheye_pix4d":
        # For fisheye_pix4d with D=E=0, C and F are focal lengths in the
        # equidistant fisheye domain (r' = f*θ) not the pinhole domain (r = f*tan(θ)).
        # If fx/fy exist in the dict, prefer those (some scenes carry both).
        fx = intrinsics_dict.get("fx", 0)
        fy = intrinsics_dict.get("fy", 0)
        if fx == 0 or fy == 0:
            # Fall back to C/F as best available approximation
            fx = float(intrinsics_dict.get("C", 1000))
            fy = float(intrinsics_dict.get("F", 1000))
    else:
        fx = float(intrinsics_dict.get("fx", intrinsics_dict.get("C", 1000)))
        fy = float(intrinsics_dict.get("fy", intrinsics_dict.get("F", 1000)))
    return float(fx), float(fy)


def scale_intrinsic_for_crop(fx, fy, cx, cy, orig_h, orig_w, target_h, target_w):
    """Adjust intrinsics for a resize-to-cover then center-crop operation."""
    scale = max(target_h / orig_h, target_w / orig_w)
    resized_h = int(round(orig_h * scale))
    resized_w = int(round(orig_w * scale))
    crop_x = (resized_w - target_w) / 2.0
    crop_y = (resized_h - target_h) / 2.0

    return np.array([
        [fx * scale, 0,          cx * scale - crop_x],
        [0,          fy * scale, cy * scale - crop_y],
        [0,          0,          1],
    ], dtype=np.float32)


def resize_crop_to_rect(img, target_h, target_w):
    """Resize image to cover target, then center crop."""
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


# ── Raymap ───────────────────────────────────────────────────────────────────

def prepare_raymap(extrinsics, intrinsics, context_indices, target_indices, height, width,
                   no_pixel_unshuffle=False):
    """
    Prepare plucker ray features from c2w (OpenGL) extrinsics and intrinsics.
    Replicates the logic from test_wan2.1_6to1.py.
    """
    context_camera_poses = extrinsics[context_indices]
    target_camera_poses  = extrinsics[target_indices]
    camera_poses = np.concatenate([context_camera_poses, target_camera_poses], axis=0)
    camera_poses = torch.from_numpy(camera_poses).float()

    # c2w → w2c, OpenGL → OpenCV
    w2cs = torch.linalg.inv(camera_poses)
    w2cs[:, [1, 2], :] *= -1

    context_intr = intrinsics[context_indices]
    target_intr  = intrinsics[target_indices]
    intr_cat = np.concatenate([context_intr, target_intr], axis=0)
    intr_tensor = torch.from_numpy(intr_cat).float()

    _, camera_poses_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)

    raymap = get_plucker_rays(
        camera_poses_norm, intr_tensor,
        height=height, width=width,
        no_pixel_unshuffle=no_pixel_unshuffle,
    )
    if isinstance(raymap, np.ndarray):
        raymap = torch.from_numpy(raymap).float()

    return raymap, camera_poses_norm, intr_tensor


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_psnr(pred, gt):
    mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


# ── Load WRIVA metadata ─────────────────────────────────────────────────────

def load_wriva_metadata(scene_path):
    """
    Return dict mapping image filename → GT metadata for all reference images.
    Both inputs and targets are drawn from reference/images/ with GT poses
    from reference/metadata/ to ensure a consistent coordinate frame.
    """
    ref_dir     = os.path.join(scene_path, "reference", "metadata")
    ref_img_dir = os.path.join(scene_path, "reference", "images")

    ref_meta = {}
    for fname in sorted(os.listdir(ref_img_dir)):
        meta_file = os.path.join(ref_dir, os.path.splitext(fname)[0] + ".json")
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                ref_meta[fname] = json.load(f)

    return ref_meta


# ── Side-by-side comparison ──────────────────────────────────────────────────

def make_comparison(gt_img, pred_img, label="", gap=10):
    """Create a side-by-side GT | Pred image with labels."""
    h, w = gt_img.shape[:2]
    comp = np.ones((h, w * 2 + gap, 3), dtype=np.uint8) * 255
    comp[:, :w] = gt_img
    comp[:, w + gap:] = pred_img

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comp, "Ground Truth", (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(comp, "Ground Truth", (10, 30), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(comp, "Prediction", (w + gap + 10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(comp, "Prediction", (w + gap + 10, 30), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(comp, label, (10, h - 15), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comp, label, (10, h - 15), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return comp


def make_input_grid(input_images, model_h, model_w, cols=3):
    """Tile the N input images into a grid at model resolution."""
    n = len(input_images)
    rows = math.ceil(n / cols)
    grid = np.ones((rows * model_h, cols * model_w, 3), dtype=np.uint8) * 255
    for i, img in enumerate(input_images):
        r, c = divmod(i, cols)
        grid[r * model_h:(r + 1) * model_h, c * model_w:(c + 1) * model_w] = img
    return grid


# ── Process one WRIVA scene ──────────────────────────────────────────────────

def process_scene(pipe, args, scene_path, scene_idx, total_scenes,
                  dreamsim_model=None, dreamsim_preprocess=None,
                  ssim_fn=None, lpips_model=None):

    scene_name = os.path.basename(scene_path)
    model_h = args.height
    model_w = args.width
    num_input = args.num_input_frames
    seed = args.seed

    print(f"\n{'='*80}")
    print(f"[{scene_idx+1}/{total_scenes}] Scene: {scene_name}")
    print(f"{'='*80}")

    # ── Load metadata ────────────────────────────────────────────────────
    ref_meta = load_wriva_metadata(scene_path)
    if len(ref_meta) < num_input + 1:
        print(f"  Warning: only {len(ref_meta)} reference images, need at least {num_input + 1}. Skipping.")
        return None

    # ── Split reference images into inputs and targets ────────────────────
    rng = random.Random(seed)
    all_ref_fnames = sorted(ref_meta.keys())
    chosen_inputs = sorted(rng.sample(all_ref_fnames, num_input))
    ref_fnames = sorted(f for f in all_ref_fnames if f not in chosen_inputs)

    print(f"  Total reference images: {len(all_ref_fnames)}")
    print(f"  Chosen {num_input} as inputs (seed={seed}): {chosen_inputs}")
    print(f"  Remaining targets: {len(ref_fnames)} images")

    # ── Compute reference point for ENU conversion ───────────────────────
    all_extr = [ref_meta[f]["extrinsics"] for f in all_ref_fnames]
    lats = [e["lat"] for e in all_extr]
    lons = [e["lon"] for e in all_extr]
    alts = [e["alt"] for e in all_extr]
    lat0, lon0, alt0 = np.mean(lats), np.mean(lons), np.mean(alts)
    print(f"  ENU reference: lat={lat0:.6f}, lon={lon0:.6f}, alt={alt0:.2f}")

    # ── Load input images and build their camera parameters ──────────────
    ref_img_dir = os.path.join(scene_path, "reference", "images")
    input_images_model = []
    input_c2ws = []
    input_intrinsics = []

    for fname in chosen_inputs:
        meta = ref_meta[fname]
        img = np.array(Image.open(os.path.join(ref_img_dir, fname)).convert("RGB"))
        img_model = resize_crop_to_rect(img, model_h, model_w)
        input_images_model.append(img_model)

        ext = meta["extrinsics"]
        c2w = build_c2w_opengl(
            ext["lat"], ext["lon"], ext["alt"],
            ext["omega"], ext["phi"], ext["kappa"],
            lat0, lon0, alt0,
        )
        input_c2ws.append(c2w)

        intr = meta["intrinsics"]
        proj = meta["projection"]
        fx, fy = get_fx_fy(intr, proj)
        cx, cy = float(intr["cx"]), float(intr["cy"])
        orig_h, orig_w = int(intr["rows"]), int(intr["columns"])
        K_model = scale_intrinsic_for_crop(fx, fy, cx, cy, orig_h, orig_w, model_h, model_w)
        input_intrinsics.append(K_model)

    # ── Setup output directory ───────────────────────────────────────────
    output_dir = os.path.join(args.output_path, scene_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save input images at model resolution
    for i, (fname, img) in enumerate(zip(chosen_inputs, input_images_model)):
        save_path = os.path.join(output_dir, f"input_{i:02d}_{fname}")
        Image.fromarray(img).save(save_path)
    input_grid = make_input_grid(input_images_model, model_h, model_w)
    Image.fromarray(input_grid).save(os.path.join(output_dir, "inputs_grid.png"))

    # ── Process each remaining reference image as target ─────────────────
    psnrs, ssims, lpips_scores, dreamsim_scores = [], [], [], []

    for t_idx, ref_fname in enumerate(ref_fnames):
        print(f"\n  [{t_idx+1}/{len(ref_fnames)}] Target: {ref_fname}")

        meta_t = ref_meta[ref_fname]
        ext_t = meta_t["extrinsics"]
        c2w_t = build_c2w_opengl(
            ext_t["lat"], ext_t["lon"], ext_t["alt"],
            ext_t["omega"], ext_t["phi"], ext_t["kappa"],
            lat0, lon0, alt0,
        )

        intr_t = meta_t["intrinsics"]
        proj_t = meta_t["projection"]
        fx_t, fy_t = get_fx_fy(intr_t, proj_t)
        cx_t, cy_t = float(intr_t["cx"]), float(intr_t["cy"])
        orig_h_t, orig_w_t = int(intr_t["rows"]), int(intr_t["columns"])
        K_model_t = scale_intrinsic_for_crop(fx_t, fy_t, cx_t, cy_t, orig_h_t, orig_w_t, model_h, model_w)

        # Stack all cameras: 6 context + 1 target
        all_c2ws = np.stack(input_c2ws + [c2w_t], axis=0)           # (7, 4, 4)
        all_Ks   = np.stack(input_intrinsics + [K_model_t], axis=0) # (7, 3, 3)
        context_indices = list(range(num_input))
        target_indices  = [num_input]
        num_total = num_input + 1

        # Prepare context images (PIL) at model resolution
        context_images = [Image.fromarray(img) for img in input_images_model]

        # Compute plucker raymap
        raymap, camera_poses_norm, intr_tensor = prepare_raymap(
            all_c2ws, all_Ks,
            context_indices, target_indices,
            model_h, model_w,
            no_pixel_unshuffle=args.no_pixel_unshuffle,
        )
        raymap = raymap.to("cuda", dtype=torch.bfloat16)

        # Run inference
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
            seed=seed,
            tiled=True,
        )

        if args.zero_temporal_rope:
            pipe_kwargs["zero_temporal_rope"] = True

        video = pipe(**pipe_kwargs)

        # Extract prediction (last frame = target)
        pred_pil = video[num_input]
        pred_np  = np.array(pred_pil)

        # Load GT at model resolution
        gt_img_orig = np.array(Image.open(os.path.join(ref_img_dir, ref_fname)).convert("RGB"))
        gt_model = resize_crop_to_rect(gt_img_orig, model_h, model_w)

        # ── Save outputs ─────────────────────────────────────────────────
        ref_stem = os.path.splitext(ref_fname)[0]
        Image.fromarray(gt_model).save(os.path.join(output_dir, f"gt_{ref_stem}.png"))
        pred_pil.save(os.path.join(output_dir, f"pred_{ref_stem}.png"))

        # ── Metrics (at model resolution) ────────────────────────────────
        psnr = compute_psnr(pred_np, gt_model)
        psnrs.append(psnr)

        ssim_val = None
        if ssim_fn is not None:
            ssim_val = ssim_fn(gt_model, pred_np, multichannel=True, channel_axis=2, data_range=255)
            ssims.append(ssim_val)

        lpips_val = None
        if lpips_model is not None:
            p_t = torch.from_numpy(pred_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            g_t = torch.from_numpy(gt_model).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            with torch.no_grad():
                lpips_val = lpips_model(p_t.cuda(), g_t.cuda()).item()
            lpips_scores.append(lpips_val)

        ds_val = None
        if dreamsim_model is not None:
            pred_ds = dreamsim_preprocess(Image.fromarray(pred_np)).to("cuda")
            gt_ds   = dreamsim_preprocess(Image.fromarray(gt_model)).to("cuda")
            if pred_ds.dim() == 3: pred_ds = pred_ds.unsqueeze(0)
            if gt_ds.dim()   == 3: gt_ds   = gt_ds.unsqueeze(0)
            while pred_ds.dim() > 4: pred_ds = pred_ds.squeeze(0)
            while gt_ds.dim()   > 4: gt_ds   = gt_ds.squeeze(0)
            with torch.no_grad():
                ds_val = dreamsim_model(pred_ds, gt_ds).item()
            dreamsim_scores.append(ds_val)

        msg = f"    PSNR={psnr:.2f}"
        if ssim_val   is not None: msg += f", SSIM={ssim_val:.4f}"
        if lpips_val  is not None: msg += f", LPIPS={lpips_val:.4f}"
        if ds_val     is not None: msg += f", DreamSim={ds_val:.4f}"
        print(msg)

        # Side-by-side comparison
        comp = make_comparison(gt_model, pred_np, label=msg.strip())
        Image.fromarray(comp).save(os.path.join(output_dir, f"comparison_{ref_stem}.png"))

    # ── Scene summary ────────────────────────────────────────────────────
    scene_metrics = {
        "psnr": float(np.mean(psnrs)) if psnrs else 0,
        "ssim": float(np.mean(ssims)) if ssims else 0,
        "lpips": float(np.mean(lpips_scores)) if lpips_scores else 0,
        "dreamsim": float(np.mean(dreamsim_scores)) if dreamsim_scores else 0,
    }

    print(f"\n  Scene summary ({scene_name}):")
    print(f"    Mean PSNR: {scene_metrics['psnr']:.2f} dB")
    if ssims:         print(f"    Mean SSIM: {scene_metrics['ssim']:.4f}")
    if lpips_scores:  print(f"    Mean LPIPS: {scene_metrics['lpips']:.4f}")
    if dreamsim_scores: print(f"    Mean DreamSim: {scene_metrics['dreamsim']:.4f}")

    per_frame = []
    for i, ref_fname in enumerate(ref_fnames):
        entry = {"frame": ref_fname, "psnr": float(psnrs[i])}
        if i < len(ssims):         entry["ssim"]     = float(ssims[i])
        if i < len(lpips_scores):  entry["lpips"]    = float(lpips_scores[i])
        if i < len(dreamsim_scores): entry["dreamsim"] = float(dreamsim_scores[i])
        per_frame.append(entry)

    results = {
        "scene": scene_name,
        "model_resolution": f"{model_h}x{model_w}",
        "num_input_frames": num_input,
        "chosen_inputs": chosen_inputs,
        "seed": seed,
        "mean_metrics": scene_metrics,
        "per_frame": per_frame,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to {output_dir}")
    return scene_metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Model resolution: {args.height}x{args.width}")
    print(f"Seed: {args.seed}")
    print(f"Scenes: {len(args.scenes)}")
    for s in args.scenes:
        print(f"  {s}")

    pipe = load_pipeline(args)

    dreamsim_model = dreamsim_preprocess = None
    if args.use_dreamsim:
        try:
            from dreamsim import dreamsim
            dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device="cuda")
        except ImportError:
            print("Warning: dreamsim not available")

    ssim_fn = None
    if args.use_ssim:
        try:
            from skimage.metrics import structural_similarity as ssim_func
            ssim_fn = ssim_func
        except ImportError:
            print("Warning: skimage not available")

    lpips_model = None
    if args.use_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').cuda()
        except ImportError:
            print("Warning: lpips not available")

    os.makedirs(args.output_path, exist_ok=True)

    all_psnr, all_ssim, all_lpips, all_dreamsim = [], [], [], []

    for scene_idx, scene_path in enumerate(args.scenes):
        result = process_scene(
            pipe, args, scene_path, scene_idx, len(args.scenes),
            dreamsim_model=dreamsim_model,
            dreamsim_preprocess=dreamsim_preprocess,
            ssim_fn=ssim_fn,
            lpips_model=lpips_model,
        )
        if result is not None:
            all_psnr.append(result["psnr"])
            if result["ssim"] > 0:     all_ssim.append(result["ssim"])
            if result["lpips"] > 0:    all_lpips.append(result["lpips"])
            if result["dreamsim"] > 0: all_dreamsim.append(result["dreamsim"])

    # ── Overall summary ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS ({len(all_psnr)} scenes)")
    print(f"{'='*80}")
    if all_psnr:     print(f"  Mean PSNR:     {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    if all_ssim:     print(f"  Mean SSIM:     {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    if all_lpips:    print(f"  Mean LPIPS:    {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}")
    if all_dreamsim: print(f"  Mean DreamSim: {np.mean(all_dreamsim):.4f} ± {np.std(all_dreamsim):.4f}")

    summary = {
        "num_scenes": len(all_psnr),
        "seed": args.seed,
        "model_resolution": f"{args.height}x{args.width}",
        "checkpoint": args.checkpoint_path,
        "mean_psnr": float(np.mean(all_psnr)) if all_psnr else None,
        "mean_ssim": float(np.mean(all_ssim)) if all_ssim else None,
        "mean_lpips": float(np.mean(all_lpips)) if all_lpips else None,
        "mean_dreamsim": float(np.mean(all_dreamsim)) if all_dreamsim else None,
        "per_scene": {os.path.basename(s): all_psnr[i] if i < len(all_psnr) else None
                      for i, s in enumerate(args.scenes)},
    }
    with open(os.path.join(args.output_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {os.path.join(args.output_path, 'summary.json')}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Wan2.1 6-to-1 NVS on WRIVA scenes")

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--new_in_dim", type=int, default=420)
    parser.add_argument("--num_input_frames", type=int, default=6)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--scenes", type=str, nargs="+", required=True,
                        help="Full paths to WRIVA scene directories")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vram_limit", type=int, default=22)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False)
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False)

    parser.add_argument("--use_dreamsim", action="store_true")
    parser.add_argument("--use_ssim", action="store_true")
    parser.add_argument("--use_lpips", action="store_true")

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
