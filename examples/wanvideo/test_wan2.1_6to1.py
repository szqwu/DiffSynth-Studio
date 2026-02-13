import os
import sys
import time
import json
import glob
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
    Modify the model's input dimension to match training configuration.
    Recreates the logic from train_SE.py.
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
        fuse_vae_embedding_in_latents_multiple=False,
        seperated_encoding=True,
     )

    # Copy pretrained weights on CPU (except patch_embedding which has different dims)
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
    print(f"Model {model_attr} channels modified successfully")


def load_pipeline(args):
    """Load pipeline, modify channels, and load checkpoint."""
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
        vram_limit=46,  # Enable VRAM management: offload models to CPU when not in use
    )

    # Modify model input channels to match training configuration
    modify_model_channels(pipe, "dit", args.new_in_dim, "cuda")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = load_state_dict(args.checkpoint_path, torch_dtype=torch.bfloat16, device="cuda")

    # Detect whether this is a LoRA checkpoint or a full checkpoint.
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
# Data loading helpers (matching cog-nvs DL3DV test pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def scale_intrinsics(intrinsics, orig_height, orig_width, target_height, target_width):
    """
    Scale camera intrinsics from original resolution to target resolution.

    Args:
        intrinsics: (N, 3, 3) numpy array of intrinsic matrices
        orig_height, orig_width: resolution the intrinsics correspond to
        target_height, target_width: resolution to scale to

    Returns:
        (N, 3, 3) scaled intrinsic matrices
    """
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height

    scaled = intrinsics.copy()
    scaled[:, 0, 0] *= scale_x   # fx
    scaled[:, 0, 2] *= scale_x   # cx
    scaled[:, 1, 1] *= scale_y   # fy
    scaled[:, 1, 2] *= scale_y   # cy
    return scaled


def prepare_raymap(extrinsics, intrinsics, context_indices, target_indices, height, width):
    """
    Normalize camera poses and compute plucker rays.
    Follows the exact same logic as cog-nvs test and the training dataset.

    NOTE: intrinsics must already be scaled to (height, width) before calling.
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

    # Normalize so last camera (target) is at origin
    _, camera_poses_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)

    # Compute plucker rays → [N, C, H/16, W/16]
    raymap = get_plucker_rays(
        camera_poses_norm,
        torch.from_numpy(intrinsics_cat).float(),
        height=height,
        width=width,
    )
    if isinstance(raymap, np.ndarray):
        raymap = torch.from_numpy(raymap).float()

    return raymap


def center_crop(img, crop_size):
    """Center crop an image to crop_size x crop_size."""
    h, w = img.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h:start_h+crop_size, start_w:start_w+crop_size]


def resize_and_center_crop(img, target_size=576):
    """
    Resize image so shorter side = target_size, then center crop to square.
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
    return center_crop(img_resized, target_size)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred, gt):
    """Compute PSNR between two uint8 numpy arrays."""
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
    """
    Process a single DL3DV-10K scene: load data, run 6-to-1 NVS for each
    target frame, compute metrics, save results.

    Returns:
        dict with per-scene mean metrics, or None on failure.
    """
    model_h = args.height
    model_w = args.width

    # Paths
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

    train_ids = split_data['train_ids']   # 6 context frames
    test_ids = split_data['test_ids']     # target frames

    print(f"  Context frame IDs: {train_ids}")
    print(f"  Target frame IDs ({len(test_ids)} frames): {test_ids[:5]}{'...' if len(test_ids)>5 else ''}")

    # ── Load transforms.json ───────────────────────────────────────────
    transforms_file = os.path.join(scene_data_path, "transforms.json")
    if not os.path.exists(transforms_file):
        print(f"  Warning: {transforms_file} not found. Skipping.")
        return None

    with open(transforms_file, 'r') as f:
        transforms_data = json.load(f)

    # Original resolution from transforms.json (4K)
    orig_w = transforms_data['w']
    orig_h = transforms_data['h']

    # Actual image resolution (960p)
    actual_w = 960
    actual_h = 540

    # Scale factors for intrinsics: 4K -> 960p
    scale_w_960 = actual_w / orig_w
    scale_h_960 = actual_h / orig_h

    # Extract intrinsics and scale to 960p
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

    # Further scale intrinsics to model input resolution
    model_scale_w = model_w / actual_w
    model_scale_h = model_h / actual_h

    scaled_intrinsic_model = scaled_intrinsic_960p.copy()
    scaled_intrinsic_model[0, 0] *= model_scale_w
    scaled_intrinsic_model[1, 1] *= model_scale_h
    scaled_intrinsic_model[0, 2] *= model_scale_w
    scaled_intrinsic_model[1, 2] *= model_scale_h

    print(f"  Original res: {orig_w}x{orig_h}, actual 960p: {actual_w}x{actual_h}, model: {model_w}x{model_h}")

    # ── Load frames data ───────────────────────────────────────────────
    frames_data = transforms_data['frames']
    images_dir = os.path.join(scene_data_path, "images_4")

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

        img_model = cv2.resize(img_960p, (model_w, model_h), interpolation=cv2.INTER_AREA)
        all_images_model[idx] = img_model

        c2w = np.array(frame_data['transform_matrix'], dtype=np.float32)
        all_extrinsics[idx] = c2w

    # ── Setup output directory ─────────────────────────────────────────
    output_dir = os.path.join(args.output_path, scene_hash)
    os.makedirs(output_dir, exist_ok=True)

    eval_size = args.eval_size

    # ── Process each target frame (6-to-1 generation) ──────────────────
    psnrs = []
    ssims = []
    lpips_scores = []
    dreamsim_scores = []

    for target_pos, target_frame_idx in enumerate(test_ids):
        print(f"\n  [{target_pos+1}/{len(test_ids)}] Generating target frame_idx={target_frame_idx}...")

        # 6 context + 1 target = 7 total
        current_indices = train_ids + [target_frame_idx]
        context_indices = list(range(len(train_ids)))           # [0,1,2,3,4,5]
        target_indices = [len(train_ids)]                       # [6]
        num_total = len(train_ids) + 1                          # 7

        # Prepare context images at model resolution (PIL)
        context_images = [Image.fromarray(all_images_model[idx]) for idx in train_ids]

        # Prepare extrinsics and intrinsics arrays
        current_extrinsics = np.stack([all_extrinsics[idx] for idx in current_indices], axis=0)  # (7,4,4)
        current_intrinsics = np.stack([scaled_intrinsic_model] * num_total, axis=0)               # (7,3,3)

        # Compute plucker raymap
        raymap = prepare_raymap(
            current_extrinsics, current_intrinsics,
            context_indices, target_indices,
            model_h, model_w,
        )
        raymap = raymap.to("cuda", dtype=torch.bfloat16)

        # Run inference
        video = pipe(
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

        # Extract generated target frame (last frame)
        pred_pil = video[-1]
        pred_frame = np.array(pred_pil)

        # Save generated frame at model resolution
        generated_frame_path = os.path.join(output_dir, f"generated_frame_{target_frame_idx:04d}_{model_h}x{model_w}.png")
        pred_pil.save(generated_frame_path)

        # Resize GT to generation (model) resolution first, then resize+crop to eval_size
        gt_frame_960p = all_images_960p[target_frame_idx]
        gt_frame_model_res = cv2.resize(gt_frame_960p, (model_w, model_h), interpolation=cv2.INTER_AREA)

        # Resize and center crop both to eval_size for metrics
        pred_frame_eval = resize_and_center_crop(pred_frame, eval_size)
        gt_frame_eval = resize_and_center_crop(gt_frame_model_res, eval_size)

        # PSNR
        psnr = compute_psnr(pred_frame_eval, gt_frame_eval)
        psnrs.append(psnr)

        # SSIM
        ssim_value = None
        if ssim_fn is not None:
            ssim_value = ssim_fn(gt_frame_eval, pred_frame_eval, multichannel=True, channel_axis=2, data_range=255)
            ssims.append(ssim_value)

        # LPIPS
        lpips_value = None
        if lpips_model is not None:
            pred_lpips = torch.from_numpy(pred_frame_eval).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            gt_lpips = torch.from_numpy(gt_frame_eval).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            with torch.no_grad():
                lpips_value = lpips_model(pred_lpips.cuda(), gt_lpips.cuda()).item()
            lpips_scores.append(lpips_value)

        # DreamSim
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

        # Print metrics
        msg = f"    PSNR={psnr:.2f} dB"
        if ssim_value is not None:
            msg += f", SSIM={ssim_value:.4f}"
        if lpips_value is not None:
            msg += f", LPIPS={lpips_value:.4f}"
        if ds_score is not None:
            msg += f", DreamSim={ds_score:.4f}"
        print(msg)

        # Save comparison image
        comparison = np.zeros((eval_size, eval_size * 2 + 20, 3), dtype=np.uint8)
        comparison[:, :eval_size] = gt_frame_eval
        comparison[:, eval_size+20:] = pred_frame_eval
        comparison[:, eval_size:eval_size+20] = 255  # White separator

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

    # ── Scene summary ──────────────────────────────────────────────────
    scene_metrics = {}
    scene_metrics['psnr'] = np.mean(psnrs) if psnrs else 0
    scene_metrics['ssim'] = np.mean(ssims) if ssims else 0
    scene_metrics['lpips'] = np.mean(lpips_scores) if lpips_scores else 0
    scene_metrics['dreamsim'] = np.mean(dreamsim_scores) if dreamsim_scores else 0

    print(f"\n  Scene {scene_hash[:8]}... summary:")
    print(f"    Mean PSNR:     {scene_metrics['psnr']:.2f} dB")
    if ssims:
        print(f"    Mean SSIM:     {scene_metrics['ssim']:.4f}")
    if lpips_scores:
        print(f"    Mean LPIPS:    {scene_metrics['lpips']:.4f}")
    if dreamsim_scores:
        print(f"    Mean DreamSim: {scene_metrics['dreamsim']:.4f}")

    # Save per-scene metrics
    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Scene: {scene_hash}\n")
        f.write(f"Method: 6-to-1 generation (6 context frames + 1 target frame)\n")
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
    print(f"Number of scenes: {len(args.scenes)}")

    # ── Load pipeline ─────────────────────────────────────────────────────
    pipe = load_pipeline(args)

    # ── Optional metrics models ────────────────────────────────────────────
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

    # ── Process scenes sequentially ────────────────────────────────────────
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

    # ── Overall summary ────────────────────────────────────────────────────
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

    # Save aggregate results
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

    parser = argparse.ArgumentParser(description="Wan2.1 6-to-1 NVS evaluation on DL3DV-10K scenes")

    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained checkpoint (.safetensors)")
    parser.add_argument("--new_in_dim", type=int, default=420,
                        help="New input dimension for the modified model (must match training --new_in_dim)")

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
    parser.add_argument("--height", type=int, default=192, help="Model input height (training resolution)")
    parser.add_argument("--width", type=int, default=336, help="Model input width (training resolution)")
    parser.add_argument("--eval_size", type=int, default=576,
                        help="Evaluation size for center-crop metrics (default: 576)")

    # Inference settings
    parser.add_argument("--num_inference_steps", type=int, default=50)

    # Metrics
    parser.add_argument("--use_dreamsim", action="store_true", help="Compute DreamSim metric")
    parser.add_argument("--use_ssim", action="store_true", help="Compute SSIM metric")
    parser.add_argument("--use_lpips", action="store_true", help="Compute LPIPS metric")

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

