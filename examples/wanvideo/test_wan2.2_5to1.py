import os
import sys
import time
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
        fuse_vae_embedding_in_latents_multiple=True,
     )

    # Load pretrained weights (except patch_embedding which has different dims)
    pretrained_state_dict = model.state_dict()
    new_state_dict = new_model.state_dict()

    for key, value in pretrained_state_dict.items():
        if key.startswith("patch_embedding"):
            print(f"  Skipping {key} — will be loaded from checkpoint")
            continue
        if key in new_state_dict and value.shape == new_state_dict[key].shape:
            new_state_dict[key] = value

    new_model.load_state_dict(new_state_dict, strict=False)
    new_model = new_model.to(device=device, dtype=torch.bfloat16)

    setattr(pipe, model_attr, new_model)
    print(f"Model {model_attr} channels modified successfully")


def load_pipeline(args):
    """Load pipeline, modify channels, and load checkpoint."""
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )

    # Modify model input channels to match training configuration
    modify_model_channels(pipe, "dit", args.new_in_dim, "cuda")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = load_state_dict(args.checkpoint_path, torch_dtype=torch.bfloat16, device="cuda")

    # Detect whether this is a LoRA checkpoint or a full checkpoint.
    # LoRA checkpoints contain keys with "lora_A"/"lora_B" or "lora_up"/"lora_down".
    has_lora_keys = any(
        "lora_A" in k or "lora_B" in k or "lora_up" in k or "lora_down" in k
        for k in checkpoint.keys()
    )

    if has_lora_keys:
        # LoRA checkpoint: fuse LoRA weights into the base model
        pipe.load_lora(pipe.dit, state_dict=checkpoint, alpha=1.0)
        print("LoRA weights loaded")

        # Load patch_embedding weights (full weights trained alongside LoRA)
        patch_emb_state = {k: v for k, v in checkpoint.items() if "patch_embedding" in k}
        if patch_emb_state:
            pipe.dit.load_state_dict(patch_emb_state, strict=False)
            print(f"Loaded {len(patch_emb_state)} patch_embedding parameters")
        else:
            print("Warning: No patch_embedding weights found in checkpoint!")
    else:
        # Full checkpoint: load all weights directly into the model
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
# Data loading helpers (matching cog-nvs test pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def load_scene_images(scene_dir):
    """Load all images from a scene directory at their original resolution."""
    image_paths = sorted(glob.glob(os.path.join(scene_dir, "frame_*.png")))
    if len(image_paths) == 0:
        image_paths = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
    if len(image_paths) == 0:
        image_paths = sorted(glob.glob(os.path.join(scene_dir, "*.png")))

    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        images.append(img)
    return images


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
        downsample_factor=16,
    )
    if isinstance(raymap, np.ndarray):
        raymap = torch.from_numpy(raymap).float()

    return raymap


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
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)
    torch.manual_seed(42)

    # Model resolution (what the model was trained on)
    model_h = args.height
    model_w = args.width

    context_indices = list(range(args.num_context))          # [0,1,2,3,4]
    target_indices = list(range(args.num_context, args.num_context + args.num_target))  # [5]
    num_total = args.num_context + args.num_target           # 6

    print(f"Model resolution: {model_h}x{model_w}")

    # ── Load pipeline ─────────────────────────────────────────────────────
    pipe = load_pipeline(args)

    # ── Optional: DreamSim ────────────────────────────────────────────────
    dreamsim_model = None
    dreamsim_preprocess = None
    if args.use_dreamsim:
        try:
            from dreamsim import dreamsim
            dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device="cuda")
        except ImportError:
            print("Warning: dreamsim not available, skipping DreamSim metric")

    # ── Discover test scenes ──────────────────────────────────────────────
    scene_dirs = sorted([
        d for d in glob.glob(os.path.join(args.data_path, "*"))
        if os.path.isdir(d)
    ])
    print(f"Found {len(scene_dirs)} test scenes")

    psnr_all = []
    dreamsim_all = []

    for scene_idx, scene_dir in enumerate(scene_dirs):
        scene_name = os.path.basename(scene_dir)
        print(f"\n{'='*60}")
        print(f"[{scene_idx+1}/{len(scene_dirs)}] Processing scene: {scene_name}")
        print(f"{'='*60}")

        # ── Load data ────────────────────────────────────────────────
        intrinsics = np.load(os.path.join(scene_dir, "intrinsics.npy"))  # at GT resolution
        extrinsics = np.load(os.path.join(scene_dir, "extrinsics.npy"))
        images = load_scene_images(scene_dir)  # load at original (GT) resolution

        if len(images) < num_total:
            print(f"  Warning: expected {num_total} images, found {len(images)}. Skipping.")
            continue

        # GT resolution is the loaded image size
        gt_w, gt_h = images[0].size  # PIL size is (width, height)
        print(f"  GT resolution (from loaded images): {gt_h}x{gt_w}")

        # ── Keep GT copies at loaded size, resize for model input ────
        gt_images = images  # already at GT resolution
        model_images = [img.resize((model_w, model_h), Image.BILINEAR) for img in images]

        context_images = [model_images[i] for i in context_indices]    # 5 PIL @ model res
        target_gt_images = [gt_images[i] for i in target_indices]      # 1 PIL @ GT res

        # ── Scale intrinsics from GT resolution to model resolution ──
        intrinsics_model = scale_intrinsics(intrinsics, gt_h, gt_w, model_h, model_w)

        # ── Compute plucker raymap at model resolution ───────────────
        raymap = prepare_raymap(
            extrinsics, intrinsics_model,
            context_indices, target_indices,
            model_h, model_w,
        )
        raymap = raymap.to("cuda", dtype=torch.bfloat16)
        print(f"  Raymap shape: {raymap.shape}")

        # ── Run inference at model resolution ────────────────────────
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

        # ── Save outputs (at model resolution) ───────────────────────
        output_dir = os.path.join(scene_dir, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        save_video(video, os.path.join(output_dir, f"scene_{scene_name}_out.mp4"), fps=10, quality=5)

        # for i, frame in enumerate(video):
        #     frame.save(os.path.join(output_dir, f"pred_frame_{i:02d}.png"))

        # ── Evaluate target frame(s) — resize predictions to GT res ──
        scene_psnrs = []
        scene_dreamsim = []

        for i, target_idx in enumerate(target_indices):
            # Resize prediction from model resolution to GT resolution
            pred_pil = video[target_idx].resize((gt_w, gt_h), Image.BILINEAR)
            pred_frame = np.array(pred_pil)

            gt_frame = np.array(target_gt_images[i])
            # Safety: ensure GT is exactly at eval resolution
            if gt_frame.shape[:2] != (gt_h, gt_w):
                gt_frame = cv2.resize(gt_frame, (gt_w, gt_h))

            # PSNR
            psnr = compute_psnr(pred_frame, gt_frame)
            scene_psnrs.append(psnr)

            # DreamSim
            ds_score = None
            if dreamsim_model is not None:
                pred_t = dreamsim_preprocess(pred_pil).to("cuda")
                gt_pil = Image.fromarray(gt_frame)
                gt_t = dreamsim_preprocess(gt_pil).to("cuda")
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
                scene_dreamsim.append(ds_score)

            msg = f"  Target frame {target_idx}: PSNR = {psnr:.2f} dB"
            if ds_score is not None:
                msg += f", DreamSim = {ds_score:.4f}"
            print(msg)

            # Save comparison image at GT resolution (GT | Prediction)
            h, w = gt_frame.shape[:2]
            sep = 10
            comparison = np.ones((h, w * 2 + sep, 3), dtype=np.uint8) * 255
            comparison[:, :w] = gt_frame
            comparison[:, w + sep:] = pred_frame

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Ground Truth", (20, 40), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(comparison, "Prediction", (w + sep + 20, 40), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            metrics_txt = f"PSNR: {psnr:.2f}"
            if ds_score is not None:
                metrics_txt += f"  DreamSim: {ds_score:.4f}"
            cv2.putText(comparison, metrics_txt, (20, h - 20), font, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

            comp_path = os.path.join(output_dir, f"comparison_target_{target_idx}.png")
            Image.fromarray(comparison).save(comp_path)

        mean_psnr = np.mean(scene_psnrs)
        psnr_all.append(mean_psnr)
        if scene_dreamsim:
            mean_ds = np.mean(scene_dreamsim)
            dreamsim_all.append(mean_ds)
            print(f"  Scene mean — PSNR: {mean_psnr:.2f} dB, DreamSim: {mean_ds:.4f}")
        else:
            print(f"  Scene mean — PSNR: {mean_psnr:.2f} dB")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    if psnr_all:
        print(f"  PSNR  — mean: {np.mean(psnr_all):.2f}, std: {np.std(psnr_all):.2f}, "
              f"min: {np.min(psnr_all):.2f}, max: {np.max(psnr_all):.2f}")
    if dreamsim_all:
        print(f"  DreamSim — mean: {np.mean(dreamsim_all):.4f}, std: {np.std(dreamsim_all):.4f}, "
              f"min: {np.min(dreamsim_all):.4f}, max: {np.max(dreamsim_all):.4f}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B SE inference (matching cog-nvs test pipeline)")
    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained checkpoint (.safetensors)")
    parser.add_argument("--new_in_dim", type=int, default=1584,
                        help="New input dimension for the modified model (must match training --new_in_dim)")
    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test data directory containing scene subdirectories")
    parser.add_argument("--output_dir", type=str, default="wan_se_output",
                        help="Output subdirectory name (created inside each scene dir)")
    # Resolution
    parser.add_argument("--height", type=int, default=480, help="Model input height (training resolution)")
    parser.add_argument("--width", type=int, default=832, help="Model input width (training resolution)")
    # Inference settings
    parser.add_argument("--num_context", type=int, default=5, help="Number of context frames")
    parser.add_argument("--num_target", type=int, default=1, help="Number of target frames to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    # Metrics
    parser.add_argument("--use_dreamsim", action="store_true", help="Compute DreamSim metric")

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
