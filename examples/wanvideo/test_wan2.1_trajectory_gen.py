"""
Generate videos along synthetic camera trajectories for a specific DL3DV scene.

Picks 6 reference views from the scene, then generates frames along a specified
camera trajectory (orbit, spiral, arc, dolly) using 6-to-1 NVS.

Trajectory types:
  orbit  — 360° horizontal orbit around scene center
  spiral — orbit with vertical oscillation
  arc    — 120° arc sweep (like a crane shot)
  dolly  — forward/backward dolly along the median viewing direction
"""

import os
import sys
import time
import json
import argparse
import random
import math

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

    has_lora_keys = any("lora_A" in k or "lora_B" in k or "lora_up" in k or "lora_down" in k
                        for k in checkpoint.keys())
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
# Data / geometry helpers
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


def look_at_opengl(position, target, world_up):
    """Construct an OpenGL-convention c2w matrix looking from position at target."""
    forward = target - position
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward  # OpenGL: camera Z points backward
    c2w[:3, 3] = position
    return c2w


def analyze_scene_cameras(all_c2w):
    """Compute scene center, mean radius, up vector, and forward direction from poses."""
    positions = np.array([c[:3, 3] for c in all_c2w])
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    mean_radius = radii.mean()

    # Estimate "up" from camera up vectors (column 1 of c2w in OpenGL)
    up_vectors = np.array([c[:3, 1] for c in all_c2w])
    avg_up = up_vectors.mean(axis=0)
    avg_up = avg_up / (np.linalg.norm(avg_up) + 1e-8)

    # Median viewing direction (negative of column 2 = forward)
    forwards = np.array([-c[:3, 2] for c in all_c2w])
    avg_forward = forwards.mean(axis=0)
    avg_forward = avg_forward / (np.linalg.norm(avg_forward) + 1e-8)

    return center, mean_radius, avg_up, avg_forward, positions


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory generators
# ──────────────────────────────────────────────────────────────────────────────

def generate_orbit_trajectory(center, radius, up, num_frames, start_angle=0.0):
    """Full 360° orbit around center in the plane perpendicular to `up`."""
    # Build a coordinate frame: up, and two orthogonal directions in the orbit plane
    up = up / (np.linalg.norm(up) + 1e-8)
    # Pick an arbitrary perpendicular
    arbitrary = np.array([1, 0, 0], dtype=np.float32)
    if abs(np.dot(up, arbitrary)) > 0.9:
        arbitrary = np.array([0, 1, 0], dtype=np.float32)
    u = np.cross(up, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(up, u)

    poses = []
    for i in range(num_frames):
        angle = start_angle + 2 * math.pi * i / num_frames
        pos = center + radius * (math.cos(angle) * u + math.sin(angle) * v)
        c2w = look_at_opengl(pos, center, up)
        poses.append(c2w)
    return poses


def generate_spiral_trajectory(center, radius, up, num_frames, height_amp=None,
                               num_loops=2, start_angle=0.0):
    """Orbit with vertical oscillation (spiral/helix)."""
    if height_amp is None:
        height_amp = radius * 0.3
    up = up / (np.linalg.norm(up) + 1e-8)
    arbitrary = np.array([1, 0, 0], dtype=np.float32)
    if abs(np.dot(up, arbitrary)) > 0.9:
        arbitrary = np.array([0, 1, 0], dtype=np.float32)
    u = np.cross(up, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(up, u)

    poses = []
    for i in range(num_frames):
        t = i / num_frames
        angle = start_angle + 2 * math.pi * num_loops * t
        h = height_amp * math.sin(2 * math.pi * t)
        pos = center + radius * (math.cos(angle) * u + math.sin(angle) * v) + h * up
        c2w = look_at_opengl(pos, center, up)
        poses.append(c2w)
    return poses


def generate_arc_trajectory(center, radius, up, num_frames, arc_degrees=120,
                            start_angle=None):
    """Partial arc sweep (like a crane shot)."""
    up = up / (np.linalg.norm(up) + 1e-8)
    arbitrary = np.array([1, 0, 0], dtype=np.float32)
    if abs(np.dot(up, arbitrary)) > 0.9:
        arbitrary = np.array([0, 1, 0], dtype=np.float32)
    u = np.cross(up, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(up, u)

    arc_rad = math.radians(arc_degrees)
    if start_angle is None:
        start_angle = -arc_rad / 2

    poses = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        angle = start_angle + arc_rad * t
        pos = center + radius * (math.cos(angle) * u + math.sin(angle) * v)
        c2w = look_at_opengl(pos, center, up)
        poses.append(c2w)
    return poses


def generate_dolly_trajectory(center, radius, up, avg_forward, num_frames,
                              dolly_range=None):
    """
    Dolly in/out: camera moves along the viewing direction toward/away from center.
    Goes from far → close → far (boomerang).
    """
    if dolly_range is None:
        dolly_range = radius * 0.8

    up = up / (np.linalg.norm(up) + 1e-8)
    fwd = avg_forward / (np.linalg.norm(avg_forward) + 1e-8)

    start_pos = center - fwd * radius

    poses = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        # Boomerang: 0 → 1 → 0
        if t <= 0.5:
            progress = 2 * t
        else:
            progress = 2 * (1 - t)
        offset = dolly_range * progress
        pos = start_pos + fwd * offset
        c2w = look_at_opengl(pos, center, up)
        poses.append(c2w)
    return poses


TRAJECTORY_GENERATORS = {
    "orbit": generate_orbit_trajectory,
    "spiral": generate_spiral_trajectory,
    "arc": generate_arc_trajectory,
    "dolly": generate_dolly_trajectory,
}


def generate_trajectory(traj_type, center, radius, up, avg_forward, num_frames):
    if traj_type == "orbit":
        return generate_orbit_trajectory(center, radius, up, num_frames)
    elif traj_type == "spiral":
        return generate_spiral_trajectory(center, radius, up, num_frames)
    elif traj_type == "arc":
        return generate_arc_trajectory(center, radius, up, num_frames)
    elif traj_type == "dolly":
        return generate_dolly_trajectory(center, radius, up, avg_forward, num_frames)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")


# ──────────────────────────────────────────────────────────────────────────────
# Reference view selection
# ──────────────────────────────────────────────────────────────────────────────

def select_reference_views(all_c2w, num_refs=6, seed=42):
    """
    Select reference views that maximize coverage:
    pick evenly spaced indices from the sorted camera positions (by angle around center).
    """
    positions = np.array([c[:3, 3] for c in all_c2w])
    center = positions.mean(axis=0)

    # Compute the "up" direction
    up_vectors = np.array([c[:3, 1] for c in all_c2w])
    avg_up = up_vectors.mean(axis=0)
    avg_up = avg_up / (np.linalg.norm(avg_up) + 1e-8)

    # Project positions onto the plane perpendicular to up
    offsets = positions - center
    # Remove component along up
    proj = offsets - np.outer(offsets @ avg_up, avg_up)
    # Compute angles
    arbitrary = np.array([1, 0, 0], dtype=np.float32)
    if abs(np.dot(avg_up, arbitrary)) > 0.9:
        arbitrary = np.array([0, 1, 0], dtype=np.float32)
    u = np.cross(avg_up, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(avg_up, u)

    angles = np.arctan2(proj @ v, proj @ u)
    sorted_by_angle = np.argsort(angles)

    step = len(sorted_by_angle) / num_refs
    ref_indices = [sorted_by_angle[int(i * step)] for i in range(num_refs)]
    return sorted(ref_indices)


# ──────────────────────────────────────────────────────────────────────────────
# Main generation
# ──────────────────────────────────────────────────────────────────────────────

def add_label(img_np, text):
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


def run_trajectory(pipe, args, traj_type, ref_indices, all_c2w_list, frames_data,
                   scene_data_path, scaled_intrinsic_model, center, mean_radius,
                   avg_up, avg_forward):
    model_h = args.height
    model_w = args.width
    num_frames = args.num_frames
    fps = args.fps
    num_input = len(ref_indices)

    print(f"\n{'=' * 80}")
    print(f"Trajectory: {traj_type} ({num_frames} frames)")
    print(f"{'=' * 80}")

    # Load reference images
    ref_images = []
    ref_c2ws = []
    for idx in ref_indices:
        frame_data = frames_data[idx]
        file_path = frame_data['file_path'].replace('images/', 'images_4/')
        img_path = os.path.join(scene_data_path, file_path)
        img_960p = np.array(Image.open(img_path).convert('RGB'))
        img_model, _, _, _ = resize_crop_to_rect(img_960p, model_h, model_w)
        ref_images.append(img_model)
        c2w = np.array(frame_data['transform_matrix'], dtype=np.float32)
        ref_c2ws.append(c2w)

    # Generate trajectory poses
    traj_poses = generate_trajectory(traj_type, center, mean_radius, avg_up,
                                     avg_forward, num_frames)

    # Output directory
    output_dir = os.path.join(args.output_path, traj_type)
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Save reference views
    refs_dir = os.path.join(output_dir, "ref_views")
    os.makedirs(refs_dir, exist_ok=True)
    for i, (idx, img) in enumerate(zip(ref_indices, ref_images)):
        Image.fromarray(img).save(os.path.join(refs_dir, f"ref_{i:02d}_idx{idx:04d}.png"))
    print(f"  Saved {num_input} reference views")

    # Generate frames one by one (6-to-1)
    generated_frames = []
    context_images_pil = [Image.fromarray(img) for img in ref_images]

    for fi in range(num_frames):
        print(f"  [{fi + 1}/{num_frames}] Generating frame {fi}...")

        target_c2w = traj_poses[fi]

        context_indices_local = list(range(num_input))
        target_indices_local = [num_input]
        num_total = num_input + 1

        current_extrinsics = np.stack(ref_c2ws + [target_c2w], axis=0)
        current_intrinsics = np.stack([scaled_intrinsic_model] * num_total, axis=0)

        raymap, _, _ = prepare_raymap(
            current_extrinsics, current_intrinsics,
            context_indices_local, target_indices_local,
            model_h, model_w,
            no_pixel_unshuffle=args.no_pixel_unshuffle,
        )
        raymap = raymap.to("cuda", dtype=torch.bfloat16)

        pipe_kwargs = dict(
            prompt="", negative_prompt="",
            input_image=context_images_pil, input_video=None,
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
        pred_frame = np.array(video[num_input])
        generated_frames.append(pred_frame)

        frame_path = os.path.join(frames_dir, f"frame_{fi:04d}.png")
        Image.fromarray(pred_frame).save(frame_path)

    # Save video
    video_path = os.path.join(output_dir, f"{traj_type}_video.mp4")
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)
    for f in generated_frames:
        writer.append_data(f)
    writer.close()
    print(f"  Saved video: {video_path}")

    # Save labeled video
    labeled_path = os.path.join(output_dir, f"{traj_type}_labeled.mp4")
    writer = imageio.get_writer(labeled_path, fps=fps, codec="libx264", quality=8)
    for i, f in enumerate(generated_frames):
        labeled = add_label(f, f"{traj_type} | frame {i}/{num_frames}")
        writer.append_data(labeled)
    writer.close()
    print(f"  Saved labeled video: {labeled_path}")

    # Save trajectory info
    traj_info = {
        "trajectory_type": traj_type,
        "num_frames": num_frames,
        "fps": fps,
        "ref_indices": ref_indices,
        "scene_center": center.tolist(),
        "mean_radius": float(mean_radius),
    }
    with open(os.path.join(output_dir, "traj_info.json"), 'w') as f:
        json.dump(traj_info, f, indent=2)

    print(f"  Done: {traj_type}")


def main(args):
    random.seed(42)
    torch.manual_seed(42)

    scene_hash = args.scene
    data_path = args.data_path
    model_h = args.height
    model_w = args.width

    # ── Load scene data ───────────────────────────────────────────────
    scene_dir = os.path.join(data_path, scene_hash)
    transforms_file = os.path.join(scene_dir, "transforms.json")

    with open(transforms_file, 'r') as f:
        transforms_data = json.load(f)

    frames_data = transforms_data['frames']
    scene_data_path = scene_dir

    # Intrinsics: 4K → 960p → model resolution (crop mode)
    orig_w = transforms_data['w']
    orig_h = transforms_data['h']
    actual_w, actual_h = 960, 540

    orig_intrinsic = np.array([
        [transforms_data['fl_x'], 0, transforms_data['cx']],
        [0, transforms_data['fl_y'], transforms_data['cy']],
        [0, 0, 1]
    ], dtype=np.float32)

    scale_w = actual_w / orig_w
    scale_h = actual_h / orig_h
    scaled_960p = orig_intrinsic.copy()
    scaled_960p[0, 0] *= scale_w; scaled_960p[1, 1] *= scale_h
    scaled_960p[0, 2] *= scale_w; scaled_960p[1, 2] *= scale_h

    crop_scale = max(model_h / actual_h, model_w / actual_w)
    resized_w = int(round(actual_w * crop_scale))
    resized_h = int(round(actual_h * crop_scale))
    off_x = (resized_w - model_w) / 2.0
    off_y = (resized_h - model_h) / 2.0

    scaled_intrinsic_model = scaled_960p.copy()
    scaled_intrinsic_model[0, 0] *= crop_scale; scaled_intrinsic_model[1, 1] *= crop_scale
    scaled_intrinsic_model[0, 2] *= crop_scale; scaled_intrinsic_model[1, 2] *= crop_scale
    scaled_intrinsic_model[0, 2] -= off_x; scaled_intrinsic_model[1, 2] -= off_y

    # All c2w poses
    all_c2w = []
    for fr in frames_data:
        all_c2w.append(np.array(fr['transform_matrix'], dtype=np.float32))

    # Analyze scene geometry
    center, mean_radius, avg_up, avg_forward, positions = analyze_scene_cameras(all_c2w)
    print(f"Scene: {scene_hash}")
    print(f"  Total frames: {len(frames_data)}")
    print(f"  Center: {center}")
    print(f"  Mean radius: {mean_radius:.4f}")
    print(f"  Avg up: {avg_up}")
    print(f"  Avg forward: {avg_forward}")

    # Select reference views
    ref_indices = select_reference_views(all_c2w, num_refs=6, seed=42)
    print(f"  Reference view indices: {ref_indices}")

    # Load pipeline
    pipe = load_pipeline(args)
    os.makedirs(args.output_path, exist_ok=True)

    # Run requested trajectories
    for traj_type in args.trajectories:
        run_trajectory(
            pipe, args, traj_type, ref_indices, all_c2w, frames_data,
            scene_data_path, scaled_intrinsic_model,
            center, mean_radius, avg_up, avg_forward,
        )

    print(f"\nAll trajectories done. Results in {args.output_path}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Generate NVS videos along synthetic trajectories")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--new_in_dim", type=int, default=420)
    parser.add_argument("--data_path", type=str, default="/data2/qiwu2/2K")
    parser.add_argument("--scene", type=str, required=True, help="Scene hash")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--trajectories", type=str, nargs="+",
                        choices=["orbit", "spiral", "arc", "dolly"],
                        required=True, help="Trajectory types to generate")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames per trajectory video")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False)
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False)

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
