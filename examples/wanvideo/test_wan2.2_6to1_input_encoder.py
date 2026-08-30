"""
Wan2.2-TI2V-5B novel view synthesis inference (6-to-1) for the INPUT-ENCODER model.

This matches the training config in
  examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B_InputEncoder_6to1.sh
i.e. the 6 conditioning frames are encoded by a SEPARATE VAE copy (+ zero-init MLP
residual) and read via TRUE per-layer cross-attention (per-layer token replacement),
with per-token timestep (context=0, target=t) and zero-temporal RoPE.

Data / scene handling and metrics mirror test_wan2.1_6to1.py so it can be driven by a
parallel launcher (see test_wan2.2_6to1_input_encoder_parallel.sh).
"""
import os
import sys
import time
import json
import glob
import copy
import argparse
import random

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from einops import rearrange

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.core.data.my_v2v_dataset_images_in_plucker_SE import (
    get_plucker_rays,
    normalize_w2c_make_cam_last_origin,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model setup utilities
# ──────────────────────────────────────────────────────────────────────────────

def modify_model_channels(pipe, model_attr, new_in_dim, device):
    """
    Recreate the DiT with the training input dimension and the input-encoder
    modules (zero-init residual MLP), matching train_SE.py's modify_model_channels
    with --use_input_encoder --fuse_vae_embedding_in_latents_multiple --seperated_encoding.
    """
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
        fuse_vae_embedding_in_latents_multiple=True,
        seperated_encoding=True,
        use_input_encoder=True,
    )

    # Copy pretrained weights on CPU (except patch_embedding which has different dims,
    # and input_latent_mlp which is new — both come from the checkpoint).
    pretrained_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    new_state_dict = new_model.state_dict()

    for key, value in pretrained_state_dict.items():
        if key.startswith("patch_embedding"):
            print(f"  Skipping {key} — will be loaded from checkpoint")
            continue
        if key in new_state_dict and value.shape == new_state_dict[key].shape:
            new_state_dict[key] = value

    new_model.load_state_dict(new_state_dict, strict=False)

    # Free the old model from GPU before moving the new one there
    model.cpu()
    del model
    del pretrained_state_dict
    torch.cuda.empty_cache()
    import gc; gc.collect()

    new_model = new_model.to(device=device, dtype=torch.bfloat16)

    setattr(pipe, model_attr, new_model)
    print(f"Model {model_attr} channels modified successfully (use_input_encoder=True)")


def load_pipeline(args):
    """Load the Wan2.2-TI2V-5B pipeline, set up the input encoder, and load the checkpoint."""
    # On 24GB GPUs the DiT (~11GB) + umt5-xxl text encoder (~11GB) don't co-fit.
    # The text encoder is only needed once (empty prompt), so stream it from CPU
    # (offload/onload on CPU, compute on CUDA) to free ~11GB for the DiT + VAEs.
    # The DiT and VAE stay resident on CUDA for speed.
    text_encoder_offload = {}
    if not args.no_offload_text_encoder:
        # Keep the text encoder on CPU (offload/onload/preparing all CPU) and only
        # stream each submodule to CUDA transiently for computation. With vram_limit
        # unset, a "cuda" preparing_device would persistently onload it (~11GB) and
        # OOM alongside the resident DiT; "cpu" forces true per-module streaming.
        text_encoder_offload = dict(
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cpu",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **text_encoder_offload),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )

    # Recreate the DiT with the trained input dim + input-encoder modules.
    modify_model_channels(pipe, "dit", args.new_in_dim, "cuda")
    pipe.dit.use_input_encoder = True

    # Separate VAE encoder for the 6 conditioning frames (copy of the main VAE).
    pipe.input_vae = copy.deepcopy(pipe.vae)
    print("Created separate input_vae (copy of vae) for input-frame conditioning")

    # ── Load checkpoint ────────────────────────────────────────────────────
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = load_state_dict(args.checkpoint_path, torch_dtype=torch.bfloat16, device="cuda")

    has_lora_keys = any(
        "lora_A" in k or "lora_B" in k or "lora_up" in k or "lora_down" in k
        for k in checkpoint.keys()
    )

    if has_lora_keys:
        pipe.load_lora(pipe.dit, state_dict=checkpoint, alpha=1.0)
        print("LoRA weights loaded")
    else:
        # Full-DiT checkpoint: load the dit-prefixed weights directly.
        dit_full = {k: v for k, v in checkpoint.items() if not k.startswith("pipe.input_vae.")}
        load_result = pipe.dit.load_state_dict(dit_full, strict=False)
        print(f"Full DiT checkpoint loaded — {len(dit_full)} keys; "
              f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")

    # patch_embedding + input_latent_mlp are trained as full weights under pipe.dit.*
    # (saved with the 'pipe.dit.' prefix stripped -> 'patch_embedding.*'/'input_latent_mlp.*').
    dit_extra = {k: v for k, v in checkpoint.items()
                 if k.startswith("patch_embedding") or k.startswith("input_latent_mlp")}
    if dit_extra:
        pipe.dit.load_state_dict(dit_extra, strict=False)
        n_pe = sum(1 for k in dit_extra if k.startswith("patch_embedding"))
        n_mlp = sum(1 for k in dit_extra if k.startswith("input_latent_mlp"))
        print(f"Loaded {n_pe} patch_embedding + {n_mlp} input_latent_mlp parameters")
    else:
        print("Warning: no patch_embedding/input_latent_mlp weights found in checkpoint!")

    # Trainable input_vae was saved under 'pipe.input_vae.*' (prefix not stripped).
    ivae_state = {k[len("pipe.input_vae."):]: v for k, v in checkpoint.items()
                  if k.startswith("pipe.input_vae.")}
    if ivae_state:
        load_result = pipe.input_vae.load_state_dict(ivae_state, strict=False)
        print(f"Loaded {len(ivae_state)} input_vae parameters (trainable VAE) — "
              f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
    else:
        print("No input_vae keys in checkpoint (frozen input VAE) — using the pretrained VAE copy")

    pipe.input_vae = pipe.input_vae.to(device="cuda", dtype=torch.bfloat16)
    pipe.dit.eval()
    pipe.input_vae.eval()
    return pipe


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers (matching cog-nvs DL3DV test pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def scale_intrinsics(intrinsics, orig_height, orig_width, target_height, target_width):
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    scaled = intrinsics.copy()
    scaled[:, 0, 0] *= scale_x
    scaled[:, 0, 2] *= scale_x
    scaled[:, 1, 1] *= scale_y
    scaled[:, 1, 2] *= scale_y
    return scaled


def prepare_raymap(extrinsics, intrinsics, context_indices, target_indices, height, width,
                   no_pixel_unshuffle=False, downsample_factor=16):
    """
    Normalize camera poses and compute plucker rays at the Wan2.2 VAE /16 resolution.
    NOTE: intrinsics must already be scaled to (height, width) before calling.
    Returns raymap [N, C, H/16, W/16], normalized c2w poses, and intrinsics tensor.
    """
    context_camera_poses = extrinsics[context_indices]
    target_camera_poses = extrinsics[target_indices]
    camera_poses = np.concatenate([context_camera_poses, target_camera_poses], axis=0)
    camera_poses = torch.from_numpy(camera_poses).float()

    # c2w -> w2c, OpenGL -> OpenCV
    w2cs = torch.linalg.inv(camera_poses)
    w2cs[:, [1, 2], :] *= -1

    context_intrinsics = intrinsics[context_indices]
    target_intrinsics = intrinsics[target_indices]
    intrinsics_cat = np.concatenate([context_intrinsics, target_intrinsics], axis=0)
    intrinsics_tensor = torch.from_numpy(intrinsics_cat).float()

    # Normalize so last camera (target) is at origin
    _, camera_poses_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)

    raymap = get_plucker_rays(
        camera_poses_norm,
        intrinsics_tensor,
        height=height,
        width=width,
        no_pixel_unshuffle=no_pixel_unshuffle,
        downsample_factor=downsample_factor,
    )
    if isinstance(raymap, np.ndarray):
        raymap = torch.from_numpy(raymap).float()

    return raymap, camera_poses_norm, intrinsics_tensor


def center_crop(img, crop_size):
    h, w = img.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h:start_h+crop_size, start_w:start_w+crop_size]


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


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred, gt):
    mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


# ──────────────────────────────────────────────────────────────────────────────
# Process a single DL3DV scene
# ──────────────────────────────────────────────────────────────────────────────

def process_scene(pipe, args, scene_hash, scene_idx, total_scenes,
                  dreamsim_model=None, dreamsim_preprocess=None,
                  ssim_fn=None, lpips_model=None):
    model_h = args.height
    model_w = args.width
    num_input = args.num_input_frames
    num_output = args.num_output_frames

    scene_meta_path = os.path.join(args.dl3dv_meta_path, scene_hash)
    scene_data_path = os.path.join(args.dl3dv_data_path, scene_hash, "nerfstudio")

    print(f"\n{'='*80}")
    print(f"[{scene_idx+1}/{total_scenes}] Processing scene: {scene_hash}")
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
    print(f"  Target frame IDs ({len(test_ids)} frames): {test_ids[:5]}{'...' if len(test_ids)>5 else ''}")
    print(f"  Generation mode: {num_input}-to-{num_output}")

    transforms_file = os.path.join(scene_data_path, "transforms.json")
    if not os.path.exists(transforms_file):
        print(f"  Warning: {transforms_file} not found. Skipping.")
        return None

    with open(transforms_file, 'r') as f:
        transforms_data = json.load(f)

    orig_w = transforms_data['w']
    orig_h = transforms_data['h']
    actual_w = 960
    actual_h = 540
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

    print(f"  Original res: {orig_w}x{orig_h}, actual 960p: {actual_w}x{actual_h}, model: {model_w}x{model_h}")

    frames_data = transforms_data['frames']

    all_indices = train_ids + test_ids
    all_images_960p = {}
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
        all_images_960p[idx] = img_960p
        if input_mode == "crop":
            img_model, _, _, _ = resize_crop_to_rect(img_960p, model_h, model_w)
        else:
            img_model = cv2.resize(img_960p, (model_w, model_h), interpolation=cv2.INTER_AREA)
        all_images_model[idx] = img_model
        c2w = np.array(frame_data['transform_matrix'], dtype=np.float32)
        all_extrinsics[idx] = c2w

    output_dir = os.path.join(args.output_path, scene_hash)
    os.makedirs(output_dir, exist_ok=True)

    eval_size = args.eval_size

    psnrs = []
    ssims = []
    lpips_scores = []
    dreamsim_scores = []
    scene_inference_time_s = 0.0
    scene_views = 0

    target_batches = [test_ids[i:i+num_output] for i in range(0, len(test_ids), num_output)]
    total_batches = len(target_batches)

    for batch_pos, target_batch in enumerate(target_batches):
        batch_str = ",".join(str(t) for t in target_batch)
        print(f"\n  [batch {batch_pos+1}/{total_batches}] Generating target frame_idx={batch_str}...")

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
            downsample_factor=args.raymap_downsample_factor,
        )
        raymap = raymap.to("cuda", dtype=torch.bfloat16)

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
            seed=42,
            tiled=True,
        )

        if args.zero_temporal_rope:
            pipe_kwargs["zero_temporal_rope"] = True
        if args.zero_xy_rope:
            pipe_kwargs["zero_xy_rope"] = True

        if args.time_inference:
            torch.cuda.synchronize()
            _t_inf_start = time.perf_counter()

        video = pipe(**pipe_kwargs)

        if args.time_inference:
            torch.cuda.synchronize()
            _t_inf = time.perf_counter() - _t_inf_start
            scene_inference_time_s += _t_inf
            scene_views += cur_num_output
            print(f"    Inference: {_t_inf:.2f}s "
                  f"({_t_inf / cur_num_output:.2f}s/view, {cur_num_output} view(s))")

        for out_i in range(cur_num_output):
            target_frame_idx = target_batch[out_i]
            pred_pil = video[num_input + out_i]
            pred_frame = np.array(pred_pil)

            generated_frame_path = os.path.join(output_dir, f"generated_frame_{target_frame_idx:04d}_{model_h}x{model_w}.png")
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
                ssim_value = ssim_fn(gt_frame_eval, pred_frame_eval, multichannel=True, channel_axis=2, data_range=255)
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
                if pred_t.dim() == 3:
                    pred_t = pred_t.unsqueeze(0)
                if gt_t.dim() == 3:
                    gt_t = gt_t.unsqueeze(0)
                while pred_t.dim() > 4:
                    pred_t = pred_t.squeeze(0)
                while gt_t.dim() > 4:
                    gt_t = gt_t.squeeze(0)
                with torch.no_grad():
                    ds_score = dreamsim_model(pred_t, gt_t).item()
                dreamsim_scores.append(ds_score)

            msg = f"    Frame {target_frame_idx}: PSNR={psnr:.2f} dB"
            if ssim_value is not None:
                msg += f", SSIM={ssim_value:.4f}"
            if lpips_value is not None:
                msg += f", LPIPS={lpips_value:.4f}"
            if ds_score is not None:
                msg += f", DreamSim={ds_score:.4f}"
            print(msg)

            comparison = np.zeros((eval_size, eval_size * 2 + 20, 3), dtype=np.uint8)
            comparison[:, :eval_size] = gt_frame_eval
            comparison[:, eval_size+20:] = pred_frame_eval
            comparison[:, eval_size:eval_size+20] = 255

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Ground Truth", (20, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(comparison, "Ground Truth", (20, 40), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(comparison, "Prediction", (eval_size + 40, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(comparison, "Prediction", (eval_size + 40, 40), font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

            metrics_txt = f"Frame {target_frame_idx}: PSNR={psnr:.2f}"
            if ssim_value is not None:
                metrics_txt += f", SSIM={ssim_value:.3f}"
            if lpips_value is not None:
                metrics_txt += f", LPIPS={lpips_value:.3f}"
            if ds_score is not None:
                metrics_txt += f", DS={ds_score:.3f}"
            cv2.putText(comparison, metrics_txt, (20, eval_size - 20), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(comparison, metrics_txt, (20, eval_size - 20), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            comp_path = os.path.join(output_dir, f"comparison_frame_{target_frame_idx:04d}.png")
            Image.fromarray(comparison).save(comp_path)

    scene_metrics = {}
    scene_metrics['psnr'] = np.mean(psnrs) if psnrs else 0
    scene_metrics['ssim'] = np.mean(ssims) if ssims else 0
    scene_metrics['lpips'] = np.mean(lpips_scores) if lpips_scores else 0
    scene_metrics['dreamsim'] = np.mean(dreamsim_scores) if dreamsim_scores else 0
    scene_metrics['inference_time_total_s'] = scene_inference_time_s
    scene_metrics['inference_views'] = scene_views
    scene_metrics['inference_time_per_view_s'] = (
        scene_inference_time_s / scene_views if scene_views > 0 else 0
    )

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
        f.write(f"Method: {num_input}-to-{num_output} generation (input-encoder / per-layer cross-attn)\n")
        f.write(f"Model resolution: {model_h}x{model_w}\n")
        f.write(f"Evaluation resolution: {eval_size}x{eval_size} (center crop)\n\n")
        f.write(f"Mean PSNR: {scene_metrics['psnr']:.2f} dB\n")
        if ssims:
            f.write(f"Mean SSIM: {scene_metrics['ssim']:.4f}\n")
        if lpips_scores:
            f.write(f"Mean LPIPS: {scene_metrics['lpips']:.4f}\n")
        if dreamsim_scores:
            f.write(f"Mean DreamSim: {scene_metrics['dreamsim']:.4f}\n")
        f.write(f"\nPer-frame metrics:\n")
        for i, target_idx in enumerate(test_ids):
            line = f"Frame {target_idx}: PSNR={psnrs[i]:.2f} dB"
            if i < len(ssims):
                line += f", SSIM={ssims[i]:.4f}"
            if i < len(lpips_scores):
                line += f", LPIPS={lpips_scores[i]:.4f}"
            if i < len(dreamsim_scores):
                line += f", DreamSim={dreamsim_scores[i]:.4f}"
            f.write(line + "\n")

    print(f"  Metrics saved to {metrics_file}")
    return scene_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)
    torch.manual_seed(42)

    print(f"Model resolution: {args.height}x{args.width}")
    print(f"Evaluation resolution: {args.eval_size}x{args.eval_size} (center crop)")
    print(f"Generation mode: {args.num_input_frames}-to-{args.num_output_frames}")
    print(f"Number of scenes: {len(args.scenes)}")

    pipe = load_pipeline(args)

    dreamsim_model = None
    dreamsim_preprocess = None
    if args.use_dreamsim:
        try:
            from dreamsim import dreamsim
            dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device="cuda")
        except ImportError:
            print("Warning: dreamsim not available, skipping DreamSim metric")

    ssim_fn = None
    if args.use_ssim:
        try:
            from skimage.metrics import structural_similarity as ssim_func
            ssim_fn = ssim_func
        except ImportError:
            print("Warning: skimage not available, skipping SSIM metric")

    lpips_model = None
    if args.use_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').cuda()
        except ImportError:
            print("Warning: lpips not available, skipping LPIPS metric")

    os.makedirs(args.output_path, exist_ok=True)

    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_dreamsim = []

    for scene_idx, scene_hash in enumerate(args.scenes):
        result = process_scene(
            pipe, args, scene_hash, scene_idx, len(args.scenes),
            dreamsim_model=dreamsim_model,
            dreamsim_preprocess=dreamsim_preprocess,
            ssim_fn=ssim_fn,
            lpips_model=lpips_model,
        )
        if result is not None:
            all_psnr.append(result['psnr'])
            if result['ssim'] > 0:
                all_ssim.append(result['ssim'])
            if result['lpips'] > 0:
                all_lpips.append(result['lpips'])
            if result['dreamsim'] > 0:
                all_dreamsim.append(result['dreamsim'])

    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS ({len(all_psnr)} scenes)")
    print(f"{'='*80}")
    if all_psnr:
        print(f"  Mean PSNR:     {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    if all_ssim:
        print(f"  Mean SSIM:     {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    if all_lpips:
        print(f"  Mean LPIPS:    {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}")
    if all_dreamsim:
        print(f"  Mean DreamSim: {np.mean(all_dreamsim):.4f} ± {np.std(all_dreamsim):.4f}")
    print(f"{'='*80}")

    summary_file = os.path.join(args.output_path, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Overall Results ({len(all_psnr)} scenes)\n")
        if all_psnr:
            f.write(f"Mean PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB\n")
        if all_ssim:
            f.write(f"Mean SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}\n")
        if all_lpips:
            f.write(f"Mean LPIPS: {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}\n")
        if all_dreamsim:
            f.write(f"Mean DreamSim: {np.mean(all_dreamsim):.4f} ± {np.std(all_dreamsim):.4f}\n")
        f.write(f"\nPer-scene results:\n")
        for i, scene_hash in enumerate(args.scenes):
            if i < len(all_psnr):
                line = f"  {scene_hash[:8]}...: PSNR={all_psnr[i]:.2f}"
                if i < len(all_ssim):
                    line += f", SSIM={all_ssim[i]:.4f}"
                if i < len(all_lpips):
                    line += f", LPIPS={all_lpips[i]:.4f}"
                if i < len(all_dreamsim):
                    line += f", DreamSim={all_dreamsim[i]:.4f}"
                f.write(line + "\n")
            else:
                f.write(f"  {scene_hash[:8]}...: FAILED\n")

    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B input-encoder 6-to-1 NVS evaluation on DL3DV-10K")

    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained checkpoint (.safetensors)")
    parser.add_argument("--new_in_dim", type=int, default=1584,
                        help="Input dim of the modified model (48 latent + 1536 raymap = 1584 for the 5B).")
    parser.add_argument("--num_input_frames", type=int, default=6,
                        help="Number of input (context) frames M. Default: 6.")
    parser.add_argument("--num_output_frames", type=int, default=1,
                        help="Number of output (target) frames N per inference call. Default: 1.")

    # Data paths
    parser.add_argument("--dl3dv_meta_path", type=str, default="/data2/qiwu2/dl3dv10",
                        help="Path to DL3DV-10K metadata (contains train_test_split_6.json per scene)")
    parser.add_argument("--dl3dv_data_path", type=str, default="/data2/qiwu2/DL3DV-10K-test",
                        help="Path to DL3DV-10K test data (contains nerfstudio/ per scene)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for results")

    # Scenes
    parser.add_argument("--scenes", type=str, nargs='+', required=True,
                        help="Scene hashes to process")

    # Resolution
    parser.add_argument("--height", type=int, default=480, help="Model input height (training resolution)")
    parser.add_argument("--width", type=int, default=832, help="Model input width (training resolution)")
    parser.add_argument("--eval_size", type=int, default=480,
                        help="Evaluation size for center-crop metrics (default: 480)")
    parser.add_argument("--input_mode", type=str, default="crop", choices=["stretch", "crop"],
                        help="How to fit images to model resolution: 'stretch' or 'crop' (preserves aspect ratio).")

    # Raymap / RoPE (must match training)
    parser.add_argument("--raymap_downsample_factor", type=int, default=16,
                        help="PixelUnshuffle downscale for raymaps (16 for Wan2.2-TI2V-5B -> 1536 ch).")
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False,
                        help="Zero out temporal RoPE. Must match training --zero_temporal_rope.")
    parser.add_argument("--zero_xy_rope", action="store_true", default=False,
                        help="Zero out spatial (H/W) RoPE. Must match training --zero_xy_rope.")
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False,
                        help="Use bilinear downsampling instead of PixelUnshuffle for raymaps.")

    # Inference settings
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--no_offload_text_encoder", action="store_true",
                        help="Keep the text encoder resident on GPU (default: stream it from "
                             "CPU to fit the 5B DiT + text encoder on 24GB GPUs).")

    # Metrics
    parser.add_argument("--use_dreamsim", action="store_true", help="Compute DreamSim metric")
    parser.add_argument("--use_ssim", action="store_true", help="Compute SSIM metric")
    parser.add_argument("--use_lpips", action="store_true", help="Compute LPIPS metric")

    # Timing
    parser.add_argument("--time_inference", action="store_true",
                        help="Measure per-view inference time (pipe forward only, with cuda.synchronize).")

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
