import os
import sys
import time
import json
import glob
import argparse
import random
import itertools
import math

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
# Permutation generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_permutations(num_context, num_permutations, seed=42):
    """
    Generate fixed random permutations of context view indices.
    Always includes identity as the first permutation.
    """
    rng = random.Random(seed)
    identity = list(range(num_context))
    perms = [identity]

    all_perms = list(itertools.permutations(range(num_context)))
    all_perms.remove(tuple(identity))
    rng.shuffle(all_perms)

    for p in all_perms[:num_permutations - 1]:
        perms.append(list(p))

    return perms


def subsample_test_ids(test_ids, fraction, seed=42):
    """Randomly subsample test_ids to the given fraction, minimum 1."""
    rng = random.Random(seed)
    k = max(1, math.ceil(len(test_ids) * fraction))
    return sorted(rng.sample(test_ids, k))


# ──────────────────────────────────────────────────────────────────────────────
# Model setup utilities (same as test_wan2.1_6to1.py)
# ──────────────────────────────────────────────────────────────────────────────

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
        vram_limit=46,
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
            print(f"  Missing keys (not in ckpt): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    return pipe


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
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
                   no_pixel_unshuffle=False):
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
        camera_poses_norm,
        intrinsics_tensor,
        height=height,
        width=width,
        no_pixel_unshuffle=no_pixel_unshuffle,
    )
    if isinstance(raymap, np.ndarray):
        raymap = torch.from_numpy(raymap).float()

    return raymap, camera_poses_norm, intrinsics_tensor


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
# Process a single DL3DV scene with permutation testing
# ──────────────────────────────────────────────────────────────────────────────

def process_scene(pipe, args, scene_hash, scene_idx, total_scenes, permutations,
                  dreamsim_model=None, dreamsim_preprocess=None,
                  ssim_fn=None, lpips_model=None):
    model_h = args.height
    model_w = args.width
    num_perms = len(permutations)

    scene_meta_path = os.path.join(args.dl3dv_meta_path, scene_hash)
    scene_data_path = os.path.join(args.dl3dv_data_path, scene_hash, "nerfstudio")

    print(f"\n{'='*80}")
    print(f"[{scene_idx+1}/{total_scenes}] Processing scene: {scene_hash}")
    print(f"{'='*80}")

    # ── Load train/test split ──────────────────────────────────────────
    split_file = os.path.join(scene_meta_path, "train_test_split_6.json")
    if not os.path.exists(split_file):
        print(f"  Warning: {split_file} not found. Skipping.")
        return None

    with open(split_file, 'r') as f:
        split_data = json.load(f)

    train_ids = split_data['train_ids']
    test_ids = split_data['test_ids']

    # Subsample test_ids to 1/10 with fixed seed
    test_ids_sub = subsample_test_ids(test_ids, args.test_fraction, seed=args.seed)

    print(f"  Context frame IDs: {train_ids}")
    print(f"  Full test IDs: {len(test_ids)} frames")
    print(f"  Subsampled test IDs ({len(test_ids_sub)} frames): {test_ids_sub}")
    print(f"  Number of permutations: {num_perms}")
    print(f"  Total inferences for this scene: {len(test_ids_sub) * num_perms}")

    # ── Load transforms.json ───────────────────────────────────────────
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

    # ── Load frames data ───────────────────────────────────────────────
    frames_data = transforms_data['frames']
    all_indices = train_ids + test_ids_sub
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

    # ── Setup output directory ─────────────────────────────────────────
    output_dir = os.path.join(args.output_path, scene_hash)
    os.makedirs(output_dir, exist_ok=True)

    # ── Pre-generate base noise for --permute_noise mode ─────────────
    base_noise = None
    if args.permute_noise:
        num_total = len(train_ids) + 1
        z_dim = pipe.vae.model.z_dim
        uf = pipe.vae.upsampling_factor
        noise_shape = (1, z_dim, num_total, model_h // uf, model_w // uf)
        base_noise = pipe.generate_noise(noise_shape, seed=42, rand_device="cpu")
        print(f"  Pre-generated base noise: {base_noise.shape} (will permute context frames per perm)")

    # ── Process each target frame with all permutations ────────────────
    # per_frame_results[target_idx] = list of metric dicts, one per permutation
    per_frame_results = {}

    for target_pos, target_frame_idx in enumerate(test_ids_sub):
        print(f"\n  [{target_pos+1}/{len(test_ids_sub)}] Target frame_idx={target_frame_idx}")
        per_frame_results[target_frame_idx] = []

        for perm_idx, perm in enumerate(permutations):
            permuted_train_ids = [train_ids[i] for i in perm]

            print(f"    Perm {perm_idx}/{num_perms}: context order = {perm} "
                  f"(frame IDs: {permuted_train_ids})")

            current_indices = permuted_train_ids + [target_frame_idx]
            context_indices = list(range(len(train_ids)))
            target_indices = [len(train_ids)]
            num_total = len(train_ids) + 1

            context_images = [Image.fromarray(all_images_model[idx]) for idx in permuted_train_ids]

            current_extrinsics = np.stack([all_extrinsics[idx] for idx in current_indices], axis=0)
            current_intrinsics = np.stack([scaled_intrinsic_model] * num_total, axis=0)

            raymap, camera_poses_norm, intrinsics_tensor = prepare_raymap(
                current_extrinsics, current_intrinsics,
                context_indices, target_indices,
                model_h, model_w,
                no_pixel_unshuffle=args.no_pixel_unshuffle,
            )
            raymap = raymap.to("cuda", dtype=torch.bfloat16)

            # If --permute_noise, monkey-patch pipe.generate_noise to return
            # noise with context frames permuted to match image ordering.
            # This ensures each image always pairs with the same noise pattern.
            original_generate_noise = None
            if args.permute_noise and base_noise is not None:
                permuted_noise = base_noise.clone()
                num_ctx = len(train_ids)
                perm_tensor = torch.tensor(perm, dtype=torch.long)
                permuted_noise[:, :, :num_ctx] = base_noise[:, :, perm_tensor]
                permuted_noise = permuted_noise.to(dtype=pipe.torch_dtype, device=pipe.device)
                original_generate_noise = pipe.generate_noise
                pipe.generate_noise = lambda *a, _pn=permuted_noise, **k: _pn

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

            if args.use_prope:
                pipe_kwargs["use_prope"] = True
                pipe_kwargs["camera_poses_norm"] = camera_poses_norm.to("cuda", dtype=torch.bfloat16)
                pipe_kwargs["intrinsics"] = intrinsics_tensor.to("cuda", dtype=torch.bfloat16)

            if args.zero_temporal_rope:
                pipe_kwargs["zero_temporal_rope"] = True

            video = pipe(**pipe_kwargs)

            if original_generate_noise is not None:
                pipe.generate_noise = original_generate_noise

            pred_pil = video[-1]
            pred_frame = np.array(pred_pil)

            generated_path = os.path.join(
                output_dir,
                f"generated_frame_{target_frame_idx:04d}_perm{perm_idx:02d}_{model_h}x{model_w}.png"
            )
            pred_pil.save(generated_path)

            gt_frame_960p = all_images_960p[target_frame_idx]
            if input_mode == "crop":
                gt_frame, _, _, _ = resize_crop_to_rect(gt_frame_960p, model_h, model_w)
            else:
                gt_frame = cv2.resize(gt_frame_960p, (model_w, model_h), interpolation=cv2.INTER_AREA)

            metrics = {}
            metrics['psnr'] = compute_psnr(pred_frame, gt_frame)

            if ssim_fn is not None:
                metrics['ssim'] = ssim_fn(gt_frame, pred_frame,
                                          multichannel=True, channel_axis=2, data_range=255)

            if lpips_model is not None:
                pred_lpips = torch.from_numpy(pred_frame).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                gt_lpips = torch.from_numpy(gt_frame).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                with torch.no_grad():
                    metrics['lpips'] = lpips_model(pred_lpips.cuda(), gt_lpips.cuda()).item()

            if dreamsim_model is not None:
                pred_ds_pil = Image.fromarray(pred_frame)
                gt_ds_pil = Image.fromarray(gt_frame)
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
                    metrics['dreamsim'] = dreamsim_model(pred_t, gt_t).item()

            per_frame_results[target_frame_idx].append(metrics)

            msg = f"      PSNR={metrics['psnr']:.2f}"
            if 'ssim' in metrics:
                msg += f", SSIM={metrics['ssim']:.4f}"
            if 'lpips' in metrics:
                msg += f", LPIPS={metrics['lpips']:.4f}"
            if 'dreamsim' in metrics:
                msg += f", DreamSim={metrics['dreamsim']:.4f}"
            print(msg)

    # ── Compute per-frame mean/std across permutations ─────────────────
    metric_keys = []
    if per_frame_results:
        first_frame = next(iter(per_frame_results.values()))
        if first_frame:
            metric_keys = list(first_frame[0].keys())

    frame_means = {}
    frame_stds = {}
    for target_idx, perm_metrics_list in per_frame_results.items():
        frame_means[target_idx] = {}
        frame_stds[target_idx] = {}
        for mk in metric_keys:
            vals = [m[mk] for m in perm_metrics_list if mk in m]
            frame_means[target_idx][mk] = np.mean(vals)
            frame_stds[target_idx][mk] = np.std(vals)

    # ── Scene-level aggregation ────────────────────────────────────────
    scene_metrics = {}
    for mk in metric_keys:
        all_means = [frame_means[t][mk] for t in per_frame_results]
        all_stds = [frame_stds[t][mk] for t in per_frame_results]
        scene_metrics[f'{mk}_mean'] = np.mean(all_means)
        scene_metrics[f'{mk}_std_mean'] = np.mean(all_stds)

    print(f"\n  Scene {scene_hash[:8]}... permutation invariance summary:")
    for mk in metric_keys:
        print(f"    {mk.upper():>10s}: mean={scene_metrics[f'{mk}_mean']:.4f}, "
              f"avg_std_across_perms={scene_metrics[f'{mk}_std_mean']:.4f}")

    # ── Save detailed results ──────────────────────────────────────────
    metrics_file = os.path.join(output_dir, "permutation_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Scene: {scene_hash}\n")
        f.write(f"Permutation invariance test\n")
        f.write(f"Number of permutations: {num_perms}\n")
        f.write(f"Test frames (subsampled {args.test_fraction}): {test_ids_sub}\n")
        f.write(f"Model resolution: {model_h}x{model_w}\n")
        f.write(f"Evaluation resolution: {model_h}x{model_w} (native pred resolution)\n")
        f.write(f"Permute noise: {args.permute_noise}\n")
        f.write(f"Seed: {args.seed}\n\n")

        f.write("Permutations used:\n")
        for pi, p in enumerate(permutations):
            f.write(f"  Perm {pi}: {p}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("Scene-level summary (averaged over all test frames):\n")
        f.write("=" * 70 + "\n")
        for mk in metric_keys:
            f.write(f"  {mk.upper():>10s}: mean={scene_metrics[f'{mk}_mean']:.4f}, "
                    f"avg_std_across_perms={scene_metrics[f'{mk}_std_mean']:.4f}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("Per-frame breakdown:\n")
        f.write("=" * 70 + "\n")
        for target_idx in per_frame_results:
            f.write(f"\nFrame {target_idx}:\n")
            f.write(f"  Mean across perms: ")
            parts = [f"{mk}={frame_means[target_idx][mk]:.4f}" for mk in metric_keys]
            f.write(", ".join(parts) + "\n")
            f.write(f"  Std  across perms: ")
            parts = [f"{mk}={frame_stds[target_idx][mk]:.4f}" for mk in metric_keys]
            f.write(", ".join(parts) + "\n")

            f.write(f"  Per-permutation values:\n")
            for pi, m in enumerate(per_frame_results[target_idx]):
                parts = [f"{mk}={m[mk]:.4f}" for mk in metric_keys if mk in m]
                f.write(f"    Perm {pi} {permutations[pi]}: {', '.join(parts)}\n")

    # Save as JSON for easy parsing
    json_file = os.path.join(output_dir, "permutation_metrics.json")
    json_data = {
        "scene": scene_hash,
        "num_permutations": num_perms,
        "permutations": permutations,
        "test_ids_subsampled": test_ids_sub,
        "seed": args.seed,
        "per_frame": {},
        "scene_summary": scene_metrics,
    }
    for target_idx in per_frame_results:
        json_data["per_frame"][str(target_idx)] = {
            "mean": frame_means[target_idx],
            "std": frame_stds[target_idx],
            "per_permutation": per_frame_results[target_idx],
        }
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"  Results saved to {metrics_file}")
    print(f"  JSON saved to {json_file}")

    return scene_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_context = 6
    permutations = generate_permutations(num_context, args.num_permutations, seed=args.seed)

    print(f"Permutation invariance test")
    print(f"  Seed: {args.seed}")
    print(f"  Number of permutations: {len(permutations)}")
    print(f"  Test frame fraction: {args.test_fraction}")
    print(f"  Permute noise: {args.permute_noise}")
    for pi, p in enumerate(permutations):
        print(f"    Perm {pi}: {p}")
    print(f"  Model resolution: {args.height}x{args.width}")
    print(f"  Number of scenes: {len(args.scenes)}")

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

    all_scene_metrics = []

    for scene_idx, scene_hash in enumerate(args.scenes):
        result = process_scene(
            pipe, args, scene_hash, scene_idx, len(args.scenes), permutations,
            dreamsim_model=dreamsim_model,
            dreamsim_preprocess=dreamsim_preprocess,
            ssim_fn=ssim_fn,
            lpips_model=lpips_model,
        )
        if result is not None:
            all_scene_metrics.append((scene_hash, result))

    # ── Overall summary ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"OVERALL PERMUTATION INVARIANCE RESULTS ({len(all_scene_metrics)} scenes)")
    print(f"{'='*80}")

    if all_scene_metrics:
        metric_keys_full = list(all_scene_metrics[0][1].keys())
        base_keys = sorted(set(k.replace('_mean', '').replace('_std_mean', '') for k in metric_keys_full))

        for mk in base_keys:
            means = [s[1][f'{mk}_mean'] for s in all_scene_metrics if f'{mk}_mean' in s[1]]
            stds = [s[1][f'{mk}_std_mean'] for s in all_scene_metrics if f'{mk}_std_mean' in s[1]]
            if means:
                print(f"  {mk.upper():>10s}: overall_mean={np.mean(means):.4f}, "
                      f"overall_perm_std={np.mean(stds):.4f}")

    summary_file = os.path.join(args.output_path, "permutation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Permutation Invariance Test Summary\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Num permutations: {args.num_permutations}\n")
        f.write(f"Test fraction: {args.test_fraction}\n")
        f.write(f"Permute noise: {args.permute_noise}\n")
        f.write(f"Scenes: {len(all_scene_metrics)}\n\n")

        f.write("Permutations:\n")
        for pi, p in enumerate(permutations):
            f.write(f"  Perm {pi}: {p}\n")
        f.write("\n")

        if all_scene_metrics:
            base_keys = sorted(set(k.replace('_mean', '').replace('_std_mean', '')
                                   for k in all_scene_metrics[0][1].keys()))
            for mk in base_keys:
                means = [s[1][f'{mk}_mean'] for s in all_scene_metrics if f'{mk}_mean' in s[1]]
                stds = [s[1][f'{mk}_std_mean'] for s in all_scene_metrics if f'{mk}_std_mean' in s[1]]
                if means:
                    f.write(f"{mk.upper():>10s}: overall_mean={np.mean(means):.4f}, "
                            f"overall_perm_std={np.mean(stds):.4f}\n")

            f.write(f"\nPer-scene:\n")
            for scene_hash, metrics in all_scene_metrics:
                f.write(f"  {scene_hash[:8]}...:\n")
                for mk in base_keys:
                    if f'{mk}_mean' in metrics:
                        f.write(f"    {mk}: mean={metrics[f'{mk}_mean']:.4f}, "
                                f"perm_std={metrics[f'{mk}_std_mean']:.4f}\n")

    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Permutation invariance test for Wan2.1 6-to-1 NVS on DL3DV-10K"
    )

    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--new_in_dim", type=int, default=420)

    # Data paths
    parser.add_argument("--dl3dv_meta_path", type=str, default="/data2/qiwu2/dl3dv10")
    parser.add_argument("--dl3dv_data_path", type=str, default="/data2/qiwu2/DL3DV-10K-test")
    parser.add_argument("--output_path", type=str, required=True)

    # Scenes
    parser.add_argument("--scenes", type=str, nargs='+', required=True)

    # Resolution
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--input_mode", type=str, default="crop", choices=["stretch", "crop"])

    # Inference settings
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--use_prope", action="store_true", default=False)
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False)
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False)

    # Metrics
    parser.add_argument("--use_dreamsim", action="store_true")
    parser.add_argument("--use_ssim", action="store_true")
    parser.add_argument("--use_lpips", action="store_true")

    # Permutation test settings
    parser.add_argument("--num_permutations", type=int, default=10,
                        help="Number of random permutations of context views to test")
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="Fraction of test frames to evaluate (default: 0.1 = 1/10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for permutation generation and test frame subsampling")
    parser.add_argument("--permute_noise", action="store_true", default=False,
                        help="Also permute the initial noise to match context view permutation. "
                             "If enabled, each image always gets paired with the same noise "
                             "pattern regardless of permutation order.")

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
