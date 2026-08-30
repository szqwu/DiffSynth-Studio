"""
6-to-1 NVS evaluation on randomly selected frames from DL3DV 2K scenes.

For each scene:
  - Randomly pick 24 frames from transforms.json
  - 6 random frames as input (context), 18 as targets
  - Generate targets in batches of num_output_frames (default 3 → 6 batches)

Data layout:
  /data2/qiwu2/2K/{scene_hash}/transforms.json     (OpenGL c2w)
  /data2/qiwu2/2K/{scene_hash}/images_4/frame_*.png (960x540)
"""

import os
import sys
import time
import json
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


# ──────────────────────────────────────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────────────────────────────────────

def modify_model_channels(pipe, model_attr, new_in_dim, device):
    model = getattr(pipe, model_attr)
    if model is None:
        return
    old_in_dim = model.in_dim
    old_out_dim = model.out_dim
    print(f"Modifying {model_attr} input channels: in_dim {old_in_dim}->{new_in_dim}")

    new_model = WanModel(
        dim=model.dim, in_dim=new_in_dim, ffn_dim=model.ffn_dim,
        out_dim=old_out_dim, text_dim=model.text_embedding[0].in_features,
        freq_dim=model.freq_dim, eps=1e-6, patch_size=model.patch_size,
        num_heads=model.num_heads, num_layers=model.num_layers,
        has_image_input=model.has_image_input, has_image_pos_emb=model.has_image_pos_emb,
        has_ref_conv=model.has_ref_conv,
        add_control_adapter=model.control_adapter is not None,
        in_dim_control_adapter=24, seperated_timestep=model.seperated_timestep,
        require_vae_embedding=model.require_vae_embedding,
        require_clip_embedding=model.require_clip_embedding,
        fuse_vae_embedding_in_latents=model.fuse_vae_embedding_in_latents,
        fuse_vae_embedding_in_latents_multiple=False, seperated_encoding=True,
    )

    pretrained_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    new_state_dict = new_model.state_dict()
    for key, value in pretrained_state_dict.items():
        if key.startswith("patch_embedding"):
            continue
        if key in new_state_dict and value.shape == new_state_dict[key].shape:
            new_state_dict[key] = value
    new_model.load_state_dict(new_state_dict, strict=False)

    model.cpu()
    del model, pretrained_state_dict
    torch.cuda.empty_cache()
    import gc; gc.collect()

    new_model = new_model.to(device=device, dtype=torch.bfloat16)
    setattr(pipe, model_attr, new_model)
    print(f"Model {model_attr} channels modified successfully")


def load_pipeline(args):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        vram_limit=46,
    )
    modify_model_channels(pipe, "dit", args.new_in_dim, "cuda")

    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = load_state_dict(args.checkpoint_path, torch_dtype=torch.bfloat16, device="cuda")

    has_lora_keys = any("lora_A" in k or "lora_B" in k or "lora_up" in k or "lora_down" in k for k in checkpoint.keys())
    if has_lora_keys:
        pipe.load_lora(pipe.dit, state_dict=checkpoint, alpha=1.0)
        print("LoRA weights loaded")
        patch_emb_state = {k: v for k, v in checkpoint.items() if "patch_embedding" in k}
        if patch_emb_state:
            pipe.dit.load_state_dict(patch_emb_state, strict=False)
            print(f"Loaded {len(patch_emb_state)} patch_embedding parameters")
    else:
        pipe.dit.load_state_dict(checkpoint, strict=False)
        print(f"Full checkpoint loaded — {len(checkpoint)} keys")

    return pipe


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def resize_crop_to_rect(img, target_h, target_w):
    if isinstance(img, Image.Image):
        img = np.array(img)
    h, w = img.shape[:2]
    scale = max(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    crop_offset_y = (new_h - target_h) // 2
    crop_offset_x = (new_w - target_w) // 2
    cropped = img_resized[crop_offset_y:crop_offset_y + target_h,
                          crop_offset_x:crop_offset_x + target_w]
    return cropped, scale, crop_offset_x, crop_offset_y


def center_crop(img, crop_size):
    h, w = img.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h:start_h + crop_size, start_w:start_w + crop_size]


def resize_and_center_crop(img, target_size=576):
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
    return center_crop(img_resized, target_size)


def prepare_raymap(extrinsics, intrinsics, context_indices, target_indices, height, width,
                   no_pixel_unshuffle=False):
    """OpenGL c2w → w2c → flip y,z → normalize → plucker rays."""
    context_camera_poses = extrinsics[context_indices]
    target_camera_poses = extrinsics[target_indices]
    camera_poses = np.concatenate([context_camera_poses, target_camera_poses], axis=0)
    camera_poses = torch.from_numpy(camera_poses).float()

    w2cs = torch.linalg.inv(camera_poses)
    w2cs[:, [1, 2], :] *= -1

    context_intrinsics = intrinsics[context_indices]
    target_intrinsics = intrinsics[target_indices]
    intrinsics_cat = np.concatenate([context_intrinsics, target_intrinsics], axis=0)
    intrinsics_tensor = torch.from_numpy(intrinsics_cat).float()

    _, camera_poses_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)

    raymap = get_plucker_rays(
        camera_poses_norm, intrinsics_tensor,
        height=height, width=width,
        no_pixel_unshuffle=no_pixel_unshuffle,
    )
    if isinstance(raymap, np.ndarray):
        raymap = torch.from_numpy(raymap).float()

    return raymap, camera_poses_norm, intrinsics_tensor


def compute_psnr(pred, gt):
    mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


# ──────────────────────────────────────────────────────────────────────────────
# Process a single scene
# ──────────────────────────────────────────────────────────────────────────────

def process_scene(pipe, args, scene_hash, scene_idx, total_scenes,
                  dreamsim_model=None, dreamsim_preprocess=None,
                  ssim_fn=None, lpips_model=None):
    model_h = args.height
    model_w = args.width
    num_input = args.num_input_frames
    num_output = args.num_output_frames

    scene_dir = os.path.join(args.data_path, scene_hash)

    print(f"\n{'=' * 80}")
    print(f"[{scene_idx + 1}/{total_scenes}] Processing scene: {scene_hash}")
    print(f"{'=' * 80}")

    # ── Load transforms.json ──────────────────────────────────────────
    transforms_file = os.path.join(scene_dir, "transforms.json")
    if not os.path.exists(transforms_file):
        print(f"  Warning: {transforms_file} not found. Skipping.")
        return None

    with open(transforms_file, 'r') as f:
        transforms_data = json.load(f)

    frames_data = transforms_data['frames']
    total_available = len(frames_data)
    print(f"  Total frames available: {total_available}")

    if total_available < args.num_pick:
        print(f"  Warning: only {total_available} frames, need {args.num_pick}. Skipping.")
        return None

    # ── Randomly pick frames ──────────────────────────────────────────
    rng = random.Random(args.seed + scene_idx)
    picked_indices = sorted(rng.sample(range(total_available), args.num_pick))
    input_pick = sorted(rng.sample(picked_indices, num_input))
    target_pick = sorted([i for i in picked_indices if i not in input_pick])

    print(f"  Picked {args.num_pick} frames: {picked_indices[:6]}...{picked_indices[-3:]}")
    print(f"  Input ({num_input}): {input_pick}")
    print(f"  Target ({len(target_pick)}): {target_pick[:6]}...{target_pick[-3:] if len(target_pick) > 6 else ''}")
    print(f"  Generation mode: {num_input}-to-{num_output}, {len(target_pick)} targets in "
          f"{(len(target_pick) + num_output - 1) // num_output} batches")

    # ── Intrinsics ────────────────────────────────────────────────────
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
        print(f"  Input mode: crop (scale={crop_scale:.4f}, crop_offset=({crop_offset_x},{crop_offset_y}))")
    else:
        model_scale_w = model_w / actual_w
        model_scale_h = model_h / actual_h

        scaled_intrinsic_model = scaled_intrinsic_960p.copy()
        scaled_intrinsic_model[0, 0] *= model_scale_w
        scaled_intrinsic_model[1, 1] *= model_scale_h
        scaled_intrinsic_model[0, 2] *= model_scale_w
        scaled_intrinsic_model[1, 2] *= model_scale_h
        print(f"  Input mode: stretch")

    print(f"  Original: {orig_w}x{orig_h}, 960p: {actual_w}x{actual_h}, model: {model_w}x{model_h}")

    # ── Load images and poses ─────────────────────────────────────────
    images_dir = os.path.join(scene_dir, "images_4")
    all_indices = input_pick + target_pick

    all_images_960p = {}
    all_images_model = {}
    all_extrinsics = {}

    for idx in all_indices:
        frame_data = frames_data[idx]
        file_path = frame_data['file_path'].replace('images/', 'images_4/')
        img_path = os.path.join(scene_dir, file_path)

        if not os.path.exists(img_path):
            print(f"  Warning: {img_path} not found. Skipping scene.")
            return None

        img_960p = np.array(Image.open(img_path).convert('RGB'))
        all_images_960p[idx] = img_960p

        if input_mode == "crop":
            img_model, _, _, _ = resize_crop_to_rect(img_960p, model_h, model_w)
        else:
            img_model = cv2.resize(img_960p, (model_w, model_h), interpolation=cv2.INTER_AREA)
        all_images_model[idx] = img_model

        c2w = np.array(frame_data['transform_matrix'], dtype=np.float32)
        all_extrinsics[idx] = c2w

    # ── Output directory ──────────────────────────────────────────────
    output_dir = os.path.join(args.output_path, scene_hash)
    os.makedirs(output_dir, exist_ok=True)

    eval_size = args.eval_size

    # Save selection info
    selection_file = os.path.join(output_dir, "selection.json")
    with open(selection_file, 'w') as f:
        json.dump({
            "seed": args.seed,
            "scene_seed": args.seed + scene_idx,
            "num_pick": args.num_pick,
            "num_input": num_input,
            "picked_indices": picked_indices,
            "input_indices": input_pick,
            "target_indices": target_pick,
        }, f, indent=2)

    # ── Generate targets in batches ───────────────────────────────────
    psnrs = []
    ssims = []
    lpips_scores = []
    dreamsim_scores = []

    target_batches = [target_pick[i:i + num_output] for i in range(0, len(target_pick), num_output)]
    total_batches = len(target_batches)

    for batch_pos, target_batch in enumerate(target_batches):
        batch_str = ",".join(str(t) for t in target_batch)
        print(f"\n  [batch {batch_pos + 1}/{total_batches}] Generating frame_idx={batch_str}...")

        cur_num_output = len(target_batch)
        current_indices = input_pick + target_batch
        context_indices = list(range(num_input))
        target_indices = list(range(num_input, num_input + cur_num_output))
        num_total = num_input + cur_num_output

        context_images = [Image.fromarray(all_images_model[idx]) for idx in input_pick]

        current_extrinsics = np.stack([all_extrinsics[idx] for idx in current_indices], axis=0)
        current_intrinsics = np.stack([scaled_intrinsic_model] * num_total, axis=0)

        raymap, camera_poses_norm, intrinsics_tensor = prepare_raymap(
            current_extrinsics, current_intrinsics,
            context_indices, target_indices,
            model_h, model_w,
            no_pixel_unshuffle=args.no_pixel_unshuffle,
        )
        raymap = raymap.to("cuda", dtype=torch.bfloat16)

        pipe_kwargs = dict(
            prompt="", negative_prompt="",
            input_image=context_images, input_video=None,
            raymap=raymap,
            height=model_h, width=model_w,
            num_frames=num_total, num_latent_frames=num_total,
            cfg_scale=1.0,
            num_inference_steps=args.num_inference_steps,
            seed=42, tiled=True,
        )

        if args.zero_temporal_rope:
            pipe_kwargs["zero_temporal_rope"] = True

        video = pipe(**pipe_kwargs)

        for out_i in range(cur_num_output):
            target_frame_idx = target_batch[out_i]
            pred_pil = video[num_input + out_i]
            pred_frame = np.array(pred_pil)

            generated_frame_path = os.path.join(
                output_dir, f"generated_frame_{target_frame_idx:04d}_{model_h}x{model_w}.png"
            )
            pred_pil.save(generated_frame_path)

            gt_frame_960p = all_images_960p[target_frame_idx]
            if input_mode == "crop":
                gt_frame_model_res, _, _, _ = resize_crop_to_rect(gt_frame_960p, model_h, model_w)
            else:
                gt_frame_model_res = cv2.resize(gt_frame_960p, (model_w, model_h), interpolation=cv2.INTER_AREA)

            pred_frame_eval = resize_and_center_crop(pred_frame, eval_size)
            gt_frame_eval = resize_and_center_crop(gt_frame_model_res, eval_size)

            psnr = compute_psnr(pred_frame_eval, gt_frame_eval)
            psnrs.append(psnr)

            ssim_value = None
            if ssim_fn is not None:
                ssim_value = ssim_fn(gt_frame_eval, pred_frame_eval, multichannel=True,
                                     channel_axis=2, data_range=255)
                ssims.append(ssim_value)

            lpips_value = None
            if lpips_model is not None:
                pred_lpips = torch.from_numpy(pred_frame_eval).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                gt_lpips = torch.from_numpy(gt_frame_eval).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                with torch.no_grad():
                    lpips_value = lpips_model(pred_lpips.cuda(), gt_lpips.cuda()).item()
                lpips_scores.append(lpips_value)

            ds_score = None
            if dreamsim_model is not None:
                pred_ds_pil = Image.fromarray(pred_frame_eval)
                gt_ds_pil = Image.fromarray(gt_frame_eval)
                pred_t = dreamsim_preprocess(pred_ds_pil).to("cuda")
                gt_t = dreamsim_preprocess(gt_ds_pil).to("cuda")
                if pred_t.dim() == 3: pred_t = pred_t.unsqueeze(0)
                if gt_t.dim() == 3: gt_t = gt_t.unsqueeze(0)
                while pred_t.dim() > 4: pred_t = pred_t.squeeze(0)
                while gt_t.dim() > 4: gt_t = gt_t.squeeze(0)
                with torch.no_grad():
                    ds_score = dreamsim_model(pred_t, gt_t).item()
                dreamsim_scores.append(ds_score)

            msg = f"    Frame {target_frame_idx}: PSNR={psnr:.2f} dB"
            if ssim_value is not None: msg += f", SSIM={ssim_value:.4f}"
            if lpips_value is not None: msg += f", LPIPS={lpips_value:.4f}"
            if ds_score is not None: msg += f", DreamSim={ds_score:.4f}"
            print(msg)

            comparison = np.zeros((eval_size, eval_size * 2 + 20, 3), dtype=np.uint8)
            comparison[:, :eval_size] = gt_frame_eval
            comparison[:, eval_size + 20:] = pred_frame_eval
            comparison[:, eval_size:eval_size + 20] = 255

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Ground Truth", (20, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(comparison, "Ground Truth", (20, 40), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(comparison, "Prediction", (eval_size + 40, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(comparison, "Prediction", (eval_size + 40, 40), font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

            metrics_txt = f"Frame {target_frame_idx}: PSNR={psnr:.2f}"
            if ssim_value is not None: metrics_txt += f", SSIM={ssim_value:.3f}"
            if lpips_value is not None: metrics_txt += f", LPIPS={lpips_value:.3f}"
            if ds_score is not None: metrics_txt += f", DS={ds_score:.3f}"
            cv2.putText(comparison, metrics_txt, (20, eval_size - 20), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(comparison, metrics_txt, (20, eval_size - 20), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            comp_path = os.path.join(output_dir, f"comparison_frame_{target_frame_idx:04d}.png")
            Image.fromarray(comparison).save(comp_path)

    # ── Scene summary ─────────────────────────────────────────────────
    scene_metrics = {
        'psnr': np.mean(psnrs) if psnrs else 0,
        'ssim': np.mean(ssims) if ssims else 0,
        'lpips': np.mean(lpips_scores) if lpips_scores else 0,
        'dreamsim': np.mean(dreamsim_scores) if dreamsim_scores else 0,
    }

    print(f"\n  Scene {scene_hash[:8]}... summary:")
    print(f"    Mean PSNR:     {scene_metrics['psnr']:.2f} dB")
    if ssims:
        print(f"    Mean SSIM:     {scene_metrics['ssim']:.4f}")
    if lpips_scores:
        print(f"    Mean LPIPS:    {scene_metrics['lpips']:.4f}")
    if dreamsim_scores:
        print(f"    Mean DreamSim: {scene_metrics['dreamsim']:.4f}")

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Scene: {scene_hash}\n")
        f.write(f"Method: {num_input}-to-{num_output} NVS, {len(target_pick)} targets in {total_batches} batches\n")
        f.write(f"Model resolution: {model_h}x{model_w}\n")
        f.write(f"Evaluation resolution: {eval_size}x{eval_size} (center crop)\n")
        f.write(f"Seed: {args.seed}, scene_seed: {args.seed + scene_idx}\n\n")
        f.write(f"Input frames: {input_pick}\n")
        f.write(f"Target frames: {target_pick}\n\n")
        f.write(f"Mean PSNR: {scene_metrics['psnr']:.2f} dB\n")
        if ssims: f.write(f"Mean SSIM: {scene_metrics['ssim']:.4f}\n")
        if lpips_scores: f.write(f"Mean LPIPS: {scene_metrics['lpips']:.4f}\n")
        if dreamsim_scores: f.write(f"Mean DreamSim: {scene_metrics['dreamsim']:.4f}\n")
        f.write(f"\nPer-frame metrics:\n")
        for i, target_idx in enumerate(target_pick):
            line = f"Frame {target_idx}: PSNR={psnrs[i]:.2f} dB"
            if i < len(ssims): line += f", SSIM={ssims[i]:.4f}"
            if i < len(lpips_scores): line += f", LPIPS={lpips_scores[i]:.4f}"
            if i < len(dreamsim_scores): line += f", DreamSim={dreamsim_scores[i]:.4f}"
            f.write(line + "\n")

    print(f"  Metrics saved to {metrics_file}")
    return scene_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"6-to-1 NVS on 2K random frames")
    print(f"  Data path: {args.data_path}")
    print(f"  Scenes: {len(args.scenes)}")
    print(f"  Frames per scene: {args.num_pick} ({args.num_input_frames} input + {args.num_pick - args.num_input_frames} target)")
    print(f"  Model resolution: {args.height}x{args.width}")
    print(f"  Eval resolution: {args.eval_size}x{args.eval_size}")
    print(f"  Seed: {args.seed}")

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

    for scene_idx, scene_hash in enumerate(args.scenes):
        result = process_scene(
            pipe, args, scene_hash, scene_idx, len(args.scenes),
            dreamsim_model=dreamsim_model, dreamsim_preprocess=dreamsim_preprocess,
            ssim_fn=ssim_fn, lpips_model=lpips_model,
        )
        if result is not None:
            all_psnr.append(result['psnr'])
            if result['ssim'] > 0: all_ssim.append(result['ssim'])
            if result['lpips'] > 0: all_lpips.append(result['lpips'])
            if result['dreamsim'] > 0: all_dreamsim.append(result['dreamsim'])

    print(f"\n{'=' * 80}")
    print(f"OVERALL RESULTS ({len(all_psnr)} scenes)")
    print(f"{'=' * 80}")
    if all_psnr: print(f"  Mean PSNR:     {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    if all_ssim: print(f"  Mean SSIM:     {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    if all_lpips: print(f"  Mean LPIPS:    {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}")
    if all_dreamsim: print(f"  Mean DreamSim: {np.mean(all_dreamsim):.4f} ± {np.std(all_dreamsim):.4f}")

    summary_file = os.path.join(args.output_path, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Overall Results ({len(all_psnr)} scenes)\n")
        f.write(f"Seed: {args.seed}, {args.num_pick} frames/scene, "
                f"{args.num_input_frames} input + {args.num_pick - args.num_input_frames} target\n\n")
        if all_psnr: f.write(f"Mean PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB\n")
        if all_ssim: f.write(f"Mean SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}\n")
        if all_lpips: f.write(f"Mean LPIPS: {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}\n")
        if all_dreamsim: f.write(f"Mean DreamSim: {np.mean(all_dreamsim):.4f} ± {np.std(all_dreamsim):.4f}\n")
        f.write(f"\nPer-scene results:\n")
        for i, scene_hash in enumerate(args.scenes):
            if i < len(all_psnr):
                line = f"  {scene_hash[:8]}...: PSNR={all_psnr[i]:.2f}"
                if i < len(all_ssim): line += f", SSIM={all_ssim[i]:.4f}"
                if i < len(all_lpips): line += f", LPIPS={all_lpips[i]:.4f}"
                if i < len(all_dreamsim): line += f", DreamSim={all_dreamsim[i]:.4f}"
                f.write(line + "\n")
            else:
                f.write(f"  {scene_hash[:8]}...: FAILED\n")

    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="6-to-1 NVS on random 2K frames")

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--new_in_dim", type=int, default=420)
    parser.add_argument("--num_input_frames", type=int, default=6)
    parser.add_argument("--num_output_frames", type=int, default=1,
                        help="Number of output (target) frames N per inference call. Default: 1.")
    parser.add_argument("--data_path", type=str, default="/data2/qiwu2/2K")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, nargs='+', required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--eval_size", type=int, default=480)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--input_mode", type=str, default="crop", choices=["stretch", "crop"],
                        help="How to fit images to model resolution: "
                             "'crop' resizes to cover then center-crops (preserves aspect ratio); "
                             "'stretch' distorts to exact (h,w).")
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False)
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False)
    parser.add_argument("--num_pick", type=int, default=24,
                        help="Total frames to pick per scene (default 24: 6 input + 18 target)")
    parser.add_argument("--seed", type=int, default=456)

    parser.add_argument("--use_dreamsim", action="store_true")
    parser.add_argument("--use_ssim", action="store_true")
    parser.add_argument("--use_lpips", action="store_true")

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
