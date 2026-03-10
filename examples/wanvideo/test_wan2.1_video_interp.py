"""
Video interpolation experiment on DL3DV-10K scenes.

For each scene:
  1. Randomly pick a contiguous 41-frame window (seed=42).
  2. Use 6 keyframes at positions 0, 8, 16, 24, 32, 40 as context.
  3. Generate the 35 intermediate frames with the 6-to-1 model.
  4. Save GT frames, generated frames, GT video, generated video,
     and a side-by-side comparison video (with labels) — all at model resolution.
"""

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
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange

import imageio.v2 as imageio

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.core.data.my_v2v_dataset_images_in_plucker_SE import (
    get_plucker_rays,
    normalize_w2c_make_cam_last_origin,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model setup (from test_wan2.1_6to1.py)
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
# Data helpers (from test_wan2.1_6to1.py)
# ──────────────────────────────────────────────────────────────────────────────

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
        camera_poses_norm, intrinsics_tensor,
        height=height, width=width,
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


def compute_psnr(pred, gt):
    mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


SEED = 42


def load_scene_data(scene_hash, dl3dv_data_path, model_h, model_w, scene_subdir="nerfstudio"):
    """Load transforms.json, return frames metadata + intrinsics at model res."""
    if scene_subdir:
        scene_data_path = os.path.join(dl3dv_data_path, scene_hash, scene_subdir)
    else:
        scene_data_path = os.path.join(dl3dv_data_path, scene_hash)
    transforms_file = os.path.join(scene_data_path, "transforms.json")

    with open(transforms_file, "r") as f:
        transforms_data = json.load(f)

    orig_w = transforms_data["w"]
    orig_h = transforms_data["h"]
    actual_w, actual_h = 960, 540

    scale_w_960 = actual_w / orig_w
    scale_h_960 = actual_h / orig_h

    orig_intrinsic = np.array(
        [
            [transforms_data["fl_x"], 0, transforms_data["cx"]],
            [0, transforms_data["fl_y"], transforms_data["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    scaled_intrinsic_960p = orig_intrinsic.copy()
    scaled_intrinsic_960p[0, 0] *= scale_w_960
    scaled_intrinsic_960p[1, 1] *= scale_h_960
    scaled_intrinsic_960p[0, 2] *= scale_w_960
    scaled_intrinsic_960p[1, 2] *= scale_h_960

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

    return transforms_data, scene_data_path, scaled_intrinsic_model


def load_frame(scene_data_path, frames_data, idx, model_h, model_w):
    """Load a single frame, return (img_model_res, c2w)."""
    frame_data = frames_data[idx]
    file_path = frame_data["file_path"].replace("images/", "images_4/")
    img_path = os.path.join(scene_data_path, file_path)
    img_960p = np.array(Image.open(img_path).convert("RGB"))
    img_model, _, _, _ = resize_crop_to_rect(img_960p, model_h, model_w)
    c2w = np.array(frame_data["transform_matrix"], dtype=np.float32)
    return img_model, c2w


def add_label(img_np, text, position="top_left"):
    """Add a text label on top-left of an image. Returns np array."""
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 24)
    except (IOError, OSError):
        font = ImageFont.load_default()

    x, y = 10, 10
    draw.rectangle([x - 2, y - 2, x + len(text) * 15 + 4, y + 30], fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(img_pil)


def make_comparison_frame(gt_np, gen_np, label_gt="Ground Truth", label_gen="Generated"):
    """Create a side-by-side frame with labels."""
    h, w = gt_np.shape[:2]
    sep = 4
    canvas = np.ones((h, w * 2 + sep, 3), dtype=np.uint8) * 255
    canvas[:, :w] = gt_np
    canvas[:, w + sep :] = gen_np

    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 22)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # GT label
    draw.rectangle([8, 8, 200, 38], fill=(0, 0, 0))
    draw.text((12, 10), label_gt, fill=(0, 255, 0), font=font)
    # Gen label
    draw.rectangle([w + sep + 8, 8, w + sep + 200, 38], fill=(0, 0, 0))
    draw.text((w + sep + 12, 10), label_gen, fill=(0, 200, 255), font=font)

    return np.array(canvas_pil)


def process_scene(pipe, args, scene_hash, scene_idx, total_scenes):
    """Process one scene: pick window, generate frames, save outputs."""
    model_h = args.height
    model_w = args.width
    keyframe_gap = args.keyframe_gap
    total_frames = keyframe_gap * 5 + 1
    keyframe_positions = [keyframe_gap * i for i in range(6)]
    fps = args.fps

    print(f"\n{'=' * 80}")
    print(f"[{scene_idx + 1}/{total_scenes}] Scene: {scene_hash}")
    print(f"{'=' * 80}")

    transforms_data, scene_data_path, scaled_intrinsic_model = load_scene_data(
        scene_hash, args.dl3dv_data_path, model_h, model_w,
        scene_subdir=args.scene_subdir,
    )
    frames_data = transforms_data["frames"]
    num_total_frames = len(frames_data)

    rng = random.Random(SEED + hash(scene_hash) % 10000)
    window_start = rng.randint(0, num_total_frames - total_frames)
    window_indices = list(range(window_start, window_start + total_frames))
    keyframe_global_indices = [window_indices[k] for k in keyframe_positions]
    target_local_positions = [i for i in range(total_frames) if i not in keyframe_positions]

    print(f"  Total scene frames: {num_total_frames}")
    print(f"  Window: [{window_start}, {window_start + total_frames - 1}] ({total_frames} frames)")
    print(f"  Keyframe gap: {keyframe_gap}, Keyframes (local): {keyframe_positions}")
    print(f"  Keyframes (global): {keyframe_global_indices}")
    print(f"  Targets to generate: {len(target_local_positions)} frames")

    # Load all frames in the window
    all_images = {}
    all_extrinsics = {}
    for local_pos, global_idx in enumerate(window_indices):
        img, c2w = load_frame(scene_data_path, frames_data, global_idx, model_h, model_w)
        all_images[local_pos] = img
        all_extrinsics[local_pos] = c2w

    # Output directory
    output_dir = os.path.join(args.output_path, scene_hash)
    os.makedirs(output_dir, exist_ok=True)

    # Save GT frames
    gt_frames_dir = os.path.join(output_dir, "gt_frames")
    os.makedirs(gt_frames_dir, exist_ok=True)
    for local_pos in range(total_frames):
        path = os.path.join(gt_frames_dir, f"frame_{local_pos:04d}.png")
        Image.fromarray(all_images[local_pos]).save(path)

    # Generate target frames one by one (6-to-1)
    generated_frames = {}
    for kp in keyframe_positions:
        generated_frames[kp] = all_images[kp]

    num_input = 6
    for ti, target_local in enumerate(target_local_positions):
        print(f"  [{ti + 1}/{len(target_local_positions)}] Generating local frame {target_local} "
              f"(global {window_indices[target_local]})...")

        context_indices_local = list(range(num_input))
        target_indices_local = [num_input]
        num_total = num_input + 1

        current_extrinsics = np.stack(
            [all_extrinsics[keyframe_positions[i]] for i in range(num_input)]
            + [all_extrinsics[target_local]],
            axis=0,
        )
        current_intrinsics = np.stack([scaled_intrinsic_model] * num_total, axis=0)

        raymap, camera_poses_norm, intrinsics_tensor = prepare_raymap(
            current_extrinsics,
            current_intrinsics,
            context_indices_local,
            target_indices_local,
            model_h,
            model_w,
            no_pixel_unshuffle=args.no_pixel_unshuffle,
        )
        raymap = raymap.to("cuda", dtype=torch.bfloat16)

        context_images = [Image.fromarray(all_images[kp]) for kp in keyframe_positions]

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

        video = pipe(**pipe_kwargs)
        pred_frame = np.array(video[num_input])
        generated_frames[target_local] = pred_frame

        gt_frame = all_images[target_local]
        psnr = compute_psnr(pred_frame, gt_frame)
        print(f"    PSNR={psnr:.2f} dB")

    # Save generated frames
    gen_frames_dir = os.path.join(output_dir, "gen_frames")
    os.makedirs(gen_frames_dir, exist_ok=True)
    for local_pos in range(total_frames):
        path = os.path.join(gen_frames_dir, f"frame_{local_pos:04d}.png")
        Image.fromarray(generated_frames[local_pos]).save(path)

    # Build ordered frame lists for videos
    gt_frame_list = [all_images[i] for i in range(total_frames)]
    gen_frame_list = [generated_frames[i] for i in range(total_frames)]

    # GT video
    gt_video_path = os.path.join(output_dir, "gt_video.mp4")
    writer = imageio.get_writer(gt_video_path, fps=fps, codec="libx264", quality=8)
    for f in gt_frame_list:
        writer.append_data(f)
    writer.close()
    print(f"  Saved GT video: {gt_video_path}")

    # Generated video
    gen_video_path = os.path.join(output_dir, "gen_video.mp4")
    writer = imageio.get_writer(gen_video_path, fps=fps, codec="libx264", quality=8)
    for f in gen_frame_list:
        writer.append_data(f)
    writer.close()
    print(f"  Saved generated video: {gen_video_path}")

    # Comparison video (side-by-side with labels)
    comp_video_path = os.path.join(output_dir, "comparison_video.mp4")
    writer = imageio.get_writer(comp_video_path, fps=fps, codec="libx264", quality=8)
    for i in range(total_frames):
        comp = make_comparison_frame(gt_frame_list[i], gen_frame_list[i])
        writer.append_data(comp)
    writer.close()
    print(f"  Saved comparison video: {comp_video_path}")

    # Compute and save metrics for non-keyframe positions
    psnrs = []
    for local_pos in target_local_positions:
        psnr = compute_psnr(generated_frames[local_pos], all_images[local_pos])
        psnrs.append(psnr)

    mean_psnr = np.mean(psnrs)
    print(f"  Mean PSNR (non-keyframes): {mean_psnr:.2f} dB")

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Scene: {scene_hash}\n")
        f.write(f"Window: [{window_start}, {window_start + total_frames - 1}]\n")
        f.write(f"Keyframes (local): {keyframe_positions}\n")
        f.write(f"Keyframes (global): {keyframe_global_indices}\n")
        f.write(f"Model resolution: {model_h}x{model_w}\n")
        f.write(f"Keyframe gap: {keyframe_gap}, Total frames: {total_frames}\n")
        f.write(f"FPS: {fps}\n")
        f.write(f"Seed: {SEED}\n\n")
        f.write(f"Mean PSNR: {mean_psnr:.2f} dB\n\n")
        f.write("Per-frame PSNR:\n")
        for local_pos, psnr in zip(target_local_positions, psnrs):
            f.write(f"  Frame {local_pos} (global {window_indices[local_pos]}): {psnr:.2f} dB\n")

    return {"psnr": mean_psnr}


def main(args):
    random.seed(SEED)
    torch.manual_seed(SEED)

    total_frames = args.keyframe_gap * 5 + 1
    keyframe_positions = [args.keyframe_gap * i for i in range(6)]

    print(f"Video Interpolation Experiment")
    print(f"  Model resolution: {args.height}x{args.width}")
    print(f"  Keyframe gap: {args.keyframe_gap}")
    print(f"  Total frames per scene: {total_frames}")
    print(f"  Keyframe positions: {keyframe_positions}")
    print(f"  FPS: {args.fps}, Seed: {SEED}")
    print(f"  Scenes: {len(args.scenes)}")

    pipe = load_pipeline(args)

    os.makedirs(args.output_path, exist_ok=True)

    all_psnr = []
    for scene_idx, scene_hash in enumerate(args.scenes):
        result = process_scene(pipe, args, scene_hash, scene_idx, len(args.scenes))
        if result is not None:
            all_psnr.append(result["psnr"])

    print(f"\n{'=' * 80}")
    print(f"OVERALL ({len(all_psnr)} scenes)")
    print(f"{'=' * 80}")
    if all_psnr:
        print(f"  Mean PSNR: {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB")

    summary_file = os.path.join(args.output_path, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Video Interpolation — {len(all_psnr)} scenes\n")
        f.write(f"Resolution: {args.height}x{args.width}, FPS: {args.fps}, Seed: {SEED}\n")
        f.write(f"Keyframe gap: {args.keyframe_gap}, Keyframes: {keyframe_positions}, Total frames: {total_frames}\n\n")
        if all_psnr:
            f.write(f"Mean PSNR: {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB\n\n")
        for i, scene_hash in enumerate(args.scenes):
            if i < len(all_psnr):
                f.write(f"  {scene_hash[:16]}...: PSNR={all_psnr[i]:.2f}\n")
            else:
                f.write(f"  {scene_hash[:16]}...: FAILED\n")

    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Video interpolation experiment on DL3DV-10K")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--new_in_dim", type=int, default=420)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dl3dv_data_path", type=str, default="/data2/qiwu2/DL3DV-10K-test")
    parser.add_argument("--scene_subdir", type=str, default="nerfstudio",
                        help="Subdirectory within each scene folder containing transforms.json "
                             "and images_4/. Use empty string for scenes with no subdirectory.")
    parser.add_argument("--scenes", type=str, nargs="+", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--keyframe_gap", type=int, default=8,
                        help="Gap between keyframes. 6 keyframes at 0, gap, 2*gap, ..., 5*gap. "
                             "Total frames = 5*gap + 1. E.g. gap=8 -> 41 frames, gap=16 -> 81 frames.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False)
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False)

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
