"""
Video interpolation experiment on mip-NeRF 360 scenes.

For each scene:
  1. Randomly pick a contiguous N-frame window (seed=42, deterministic per scene).
  2. Use 6 keyframes evenly spaced as context.
  3. Generate the intermediate frames with the 6-to-1 model.
  4. Save GT frames, generated frames, GT video, generated video,
     and a side-by-side comparison video (with labels) — all at model resolution.
"""

import os
import sys
import time
import json
import struct
import argparse
import random

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import imageio.v2 as imageio

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.core.data.my_v2v_dataset_images_in_plucker_SE import (
    get_plucker_rays,
    normalize_w2c_make_cam_last_origin,
)


SEED = 42


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
# COLMAP readers (from test_wan2.1_mip360.py)
# ──────────────────────────────────────────────────────────────────────────────

COLMAP_CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12,
}


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = COLMAP_CAMERA_MODEL_NUM_PARAMS[model_id]
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[camera_id] = {"model_id": model_id, "width": width, "height": height, "params": params}
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
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
            images[image_id] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": camera_id,
                "name": name,
            }
    return images


def colmap_to_intrinsic_matrix(cam):
    model_id = cam["model_id"]
    params = cam["params"]
    if model_id == 0:
        f, cx, cy = params[0], params[1], params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    elif model_id == 1:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    elif model_id in (2, 3, 8, 9):
        f, cx, cy = params[0], params[1], params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    elif model_id in (4, 5, 6, 7, 10):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown COLMAP camera model id: {model_id}")


def colmap_qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


def load_colmap_scene(scene_path):
    sparse_dir = os.path.join(scene_path, "sparse", "0")
    cams = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    imgs = read_images_binary(os.path.join(sparse_dir, "images.bin"))

    cam = list(cams.values())[0]
    intrinsic_full = colmap_to_intrinsic_matrix(cam)
    full_width = cam["width"]
    full_height = cam["height"]

    sorted_imgs = sorted(imgs.values(), key=lambda x: x["name"])

    image_names = []
    w2c_list = []

    for img_data in sorted_imgs:
        image_names.append(img_data["name"])
        R_w2c = colmap_qvec_to_rotmat(img_data["qvec"])
        t_w2c = img_data["tvec"]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        w2c_list.append(w2c.astype(np.float32))

    return intrinsic_full, image_names, w2c_list, full_width, full_height


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def prepare_raymap(w2c_matrices, intrinsics, context_indices, target_indices, height, width,
                   no_pixel_unshuffle=False):
    context_w2c = w2c_matrices[context_indices]
    target_w2c = w2c_matrices[target_indices]
    w2cs = np.concatenate([context_w2c, target_w2c], axis=0)
    w2cs = torch.from_numpy(w2cs).float()

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


def make_comparison_frame(gt_np, gen_np, label_gt="Ground Truth", label_gen="Generated"):
    h, w = gt_np.shape[:2]
    sep = 4
    canvas = np.ones((h, w * 2 + sep, 3), dtype=np.uint8) * 255
    canvas[:, :w] = gt_np
    canvas[:, w + sep:] = gen_np

    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 22)
    except (IOError, OSError):
        font = ImageFont.load_default()

    draw.rectangle([8, 8, 200, 38], fill=(0, 0, 0))
    draw.text((12, 10), label_gt, fill=(0, 255, 0), font=font)
    draw.rectangle([w + sep + 8, 8, w + sep + 200, 38], fill=(0, 0, 0))
    draw.text((w + sep + 12, 10), label_gen, fill=(0, 200, 255), font=font)

    return np.array(canvas_pil)


# ──────────────────────────────────────────────────────────────────────────────
# Process a single scene
# ──────────────────────────────────────────────────────────────────────────────

def process_scene(pipe, args, scene_name, scene_idx, total_scenes):
    model_h = args.height
    model_w = args.width
    keyframe_gap = args.keyframe_gap
    total_frames = keyframe_gap * 5 + 1
    keyframe_positions = [keyframe_gap * i for i in range(6)]
    fps = args.fps

    scene_data_path = os.path.join(args.mip360_data_path, scene_name)

    print(f"\n{'=' * 80}")
    print(f"[{scene_idx + 1}/{total_scenes}] Scene: {scene_name}")
    print(f"{'=' * 80}")

    # Load COLMAP data
    intrinsic_full, image_names, w2c_list, full_w, full_h = load_colmap_scene(scene_data_path)
    num_total_images = len(image_names)

    # Read images_2 resolution
    images_dir = os.path.join(scene_data_path, "images_2")
    first_img_name = sorted(os.listdir(images_dir))[0]
    first_img = Image.open(os.path.join(images_dir, first_img_name))
    img2_w, img2_h = first_img.size

    # Scale intrinsics: full-res -> images_2
    intrinsic_img2 = intrinsic_full.copy()
    intrinsic_img2[0, :] *= (img2_w / full_w)
    intrinsic_img2[1, :] *= (img2_h / full_h)

    # Scale intrinsics: images_2 -> model res (resize + center crop)
    crop_scale = max(model_h / img2_h, model_w / img2_w)
    resized_h = int(round(img2_h * crop_scale))
    resized_w = int(round(img2_w * crop_scale))
    crop_offset_x = (resized_w - model_w) / 2.0
    crop_offset_y = (resized_h - model_h) / 2.0

    intrinsic_model = intrinsic_img2.copy()
    intrinsic_model[0, 0] *= crop_scale
    intrinsic_model[1, 1] *= crop_scale
    intrinsic_model[0, 2] *= crop_scale
    intrinsic_model[1, 2] *= crop_scale
    intrinsic_model[0, 2] -= crop_offset_x
    intrinsic_model[1, 2] -= crop_offset_y

    # Pick random window
    rng = random.Random(SEED + hash(scene_name) % 10000)
    window_start = rng.randint(0, num_total_images - total_frames)
    window_indices = list(range(window_start, window_start + total_frames))
    keyframe_global_indices = [window_indices[k] for k in keyframe_positions]
    target_local_positions = [i for i in range(total_frames) if i not in keyframe_positions]

    print(f"  Total images: {num_total_images}, images_2 res: {img2_w}x{img2_h}")
    print(f"  Window: [{window_start}, {window_start + total_frames - 1}] ({total_frames} frames)")
    print(f"  Keyframe gap: {keyframe_gap}, Keyframes (local): {keyframe_positions}")
    print(f"  Targets to generate: {len(target_local_positions)} frames")

    # Load all frames in window
    all_images = {}
    all_w2c = {}
    for local_pos, global_idx in enumerate(window_indices):
        img_name = image_names[global_idx]
        img_path = os.path.join(images_dir, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))
        img_model, _, _, _ = resize_crop_to_rect(img, model_h, model_w)
        all_images[local_pos] = img_model
        all_w2c[local_pos] = w2c_list[global_idx]

    # Output directory
    output_dir = os.path.join(args.output_path, scene_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save GT frames
    gt_frames_dir = os.path.join(output_dir, "gt_frames")
    os.makedirs(gt_frames_dir, exist_ok=True)
    for local_pos in range(total_frames):
        Image.fromarray(all_images[local_pos]).save(
            os.path.join(gt_frames_dir, f"frame_{local_pos:04d}.png"))

    # Generate target frames (6-to-1)
    generated_frames = {}
    for kp in keyframe_positions:
        generated_frames[kp] = all_images[kp]

    num_input = 6
    for ti, target_local in enumerate(target_local_positions):
        print(f"  [{ti + 1}/{len(target_local_positions)}] Generating local frame {target_local} "
              f"(global {window_indices[target_local]}, {image_names[window_indices[target_local]]})...")

        context_indices_local = list(range(num_input))
        target_indices_local = [num_input]
        num_total = num_input + 1

        current_w2c = np.stack(
            [all_w2c[keyframe_positions[i]] for i in range(num_input)]
            + [all_w2c[target_local]],
            axis=0,
        )
        current_intrinsics = np.stack([intrinsic_model] * num_total, axis=0)

        raymap, camera_poses_norm, intrinsics_tensor = prepare_raymap(
            current_w2c, current_intrinsics,
            context_indices_local, target_indices_local,
            model_h, model_w,
            no_pixel_unshuffle=args.no_pixel_unshuffle,
        )
        raymap = raymap.to("cuda", dtype=torch.bfloat16)

        context_images = [Image.fromarray(all_images[kp]) for kp in keyframe_positions]

        pipe_kwargs = dict(
            prompt="", negative_prompt="",
            input_image=context_images, input_video=None,
            raymap=raymap,
            height=model_h, width=model_w,
            num_frames=num_total, num_latent_frames=num_total,
            cfg_scale=1.0, num_inference_steps=args.num_inference_steps,
            seed=42, tiled=True,
        )

        if args.zero_temporal_rope:
            pipe_kwargs["zero_temporal_rope"] = True

        video = pipe(**pipe_kwargs)
        pred_frame = np.array(video[num_input])
        generated_frames[target_local] = pred_frame

        psnr = compute_psnr(pred_frame, all_images[target_local])
        print(f"    PSNR={psnr:.2f} dB")

    # Save generated frames
    gen_frames_dir = os.path.join(output_dir, "gen_frames")
    os.makedirs(gen_frames_dir, exist_ok=True)
    for local_pos in range(total_frames):
        Image.fromarray(generated_frames[local_pos]).save(
            os.path.join(gen_frames_dir, f"frame_{local_pos:04d}.png"))

    # Build frame lists
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

    # Comparison video
    comp_video_path = os.path.join(output_dir, "comparison_video.mp4")
    writer = imageio.get_writer(comp_video_path, fps=fps, codec="libx264", quality=8)
    for i in range(total_frames):
        comp = make_comparison_frame(gt_frame_list[i], gen_frame_list[i])
        writer.append_data(comp)
    writer.close()
    print(f"  Saved comparison video: {comp_video_path}")

    # Metrics
    psnrs = []
    for local_pos in target_local_positions:
        psnrs.append(compute_psnr(generated_frames[local_pos], all_images[local_pos]))

    mean_psnr = np.mean(psnrs)
    print(f"  Mean PSNR (non-keyframes): {mean_psnr:.2f} dB")

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Scene: {scene_name}\n")
        f.write(f"Window: [{window_start}, {window_start + total_frames - 1}]\n")
        f.write(f"Keyframes (local): {keyframe_positions}\n")
        f.write(f"Keyframes (global): {keyframe_global_indices}\n")
        f.write(f"Keyframe gap: {keyframe_gap}, Total frames: {total_frames}\n")
        f.write(f"Model resolution: {model_h}x{model_w}\n")
        f.write(f"FPS: {fps}, Seed: {SEED}\n\n")
        f.write(f"Mean PSNR: {mean_psnr:.2f} dB\n\n")
        f.write("Per-frame PSNR:\n")
        for local_pos, psnr in zip(target_local_positions, psnrs):
            f.write(f"  Frame {local_pos} (global {window_indices[local_pos]}, "
                    f"{image_names[window_indices[local_pos]]}): {psnr:.2f} dB\n")

    return {"psnr": mean_psnr}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(SEED)
    torch.manual_seed(SEED)

    total_frames = args.keyframe_gap * 5 + 1
    keyframe_positions = [args.keyframe_gap * i for i in range(6)]

    print(f"Video Interpolation — mip-NeRF 360")
    print(f"  Model resolution: {args.height}x{args.width}")
    print(f"  Keyframe gap: {args.keyframe_gap}")
    print(f"  Total frames per scene: {total_frames}")
    print(f"  Keyframe positions: {keyframe_positions}")
    print(f"  FPS: {args.fps}, Seed: {SEED}")
    print(f"  Scenes: {len(args.scenes)}")

    pipe = load_pipeline(args)

    os.makedirs(args.output_path, exist_ok=True)

    all_psnr = []
    for scene_idx, scene_name in enumerate(args.scenes):
        result = process_scene(pipe, args, scene_name, scene_idx, len(args.scenes))
        if result is not None:
            all_psnr.append(result["psnr"])

    print(f"\n{'=' * 80}")
    print(f"OVERALL ({len(all_psnr)} scenes)")
    print(f"{'=' * 80}")
    if all_psnr:
        print(f"  Mean PSNR: {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB")

    summary_file = os.path.join(args.output_path, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Video Interpolation (mip-NeRF 360) — {len(all_psnr)} scenes\n")
        f.write(f"Resolution: {args.height}x{args.width}, FPS: {args.fps}, Seed: {SEED}\n")
        f.write(f"Keyframe gap: {args.keyframe_gap}, Keyframes: {keyframe_positions}, Total frames: {total_frames}\n\n")
        if all_psnr:
            f.write(f"Mean PSNR: {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB\n\n")
        for i, scene_name in enumerate(args.scenes):
            if i < len(all_psnr):
                f.write(f"  {scene_name}: PSNR={all_psnr[i]:.2f}\n")
            else:
                f.write(f"  {scene_name}: FAILED\n")

    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Video interpolation on mip-NeRF 360")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--new_in_dim", type=int, default=420)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mip360_data_path", type=str, default="/data2/qiwu2/mip360")
    parser.add_argument("--scenes", type=str, nargs="+", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--keyframe_gap", type=int, default=8,
                        help="Gap between keyframes. Total frames = 5*gap + 1. "
                             "gap=8 -> 41 frames, gap=16 -> 81 frames.")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False)
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False)

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
