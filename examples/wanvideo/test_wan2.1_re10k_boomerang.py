"""
K-view video interpolation on RealEstate10K scenes.

Supports two trajectory modes:
  - "boomerang": a → b → c → b → a  (41 frames)
  - "forward":   a → b → c           (21 frames)

For each scene:
  1. Randomly pick 21 frames with configurable stride (deterministic per scene).
  2. Build trajectory and place K reference views evenly spaced.
  3. Generate intermediate frames with K-to-1 NVS.
  4. Save input views, GT frames, generated frames, GT video, generated video,
     and a side-by-side comparison video — all at model resolution.

RE10K data format:
  - video/{scene_id}.mp4  (640x360)
  - pose/{scene_id}.json  → {"poses": [[fx,fy,cx,cy,near,far, w2c_3x4_flat], ...]}
    - Intrinsics are normalized by image dimensions
    - Extrinsics: 3x4 w2c matrix in OpenCV convention
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
TOTAL_VIDEO_FRAMES = 41


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
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def prepare_raymap(extrinsics, intrinsics, context_indices, target_indices, height, width,
                   no_pixel_unshuffle=False):
    """Expects OpenGL c2w poses. Converts: c2w(GL) → w2c(GL) → flip y,z → w2c(CV) → normalize."""
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
# RE10K data loading
# ──────────────────────────────────────────────────────────────────────────────

def parse_re10k_pose(pose_18):
    """
    Parse one RE10K pose (18 values) → intrinsics (normalized), c2w (OpenGL 4x4).

    RE10K format: [fx_n, fy_n, cx_n, cy_n, near, far, w2c_3x4_row_major]
    - Intrinsics are normalized by image width/height
    - Extrinsics: 3x4 w2c in OpenCV convention

    Returns: (fx_n, fy_n, cx_n, cy_n), c2w_opengl (4x4 float32)
    """
    fx_n, fy_n, cx_n, cy_n = pose_18[0], pose_18[1], pose_18[2], pose_18[3]

    w2c_34 = np.array(pose_18[6:], dtype=np.float32).reshape(3, 4)
    w2c_cv = np.eye(4, dtype=np.float32)
    w2c_cv[:3, :] = w2c_34

    c2w_cv = np.linalg.inv(w2c_cv)

    # OpenCV c2w → OpenGL c2w: flip Y and Z columns
    c2w_gl = c2w_cv.copy()
    c2w_gl[:3, [1, 2]] *= -1

    return (fx_n, fy_n, cx_n, cy_n), c2w_gl


def load_re10k_scene(data_path, scene_id, model_h, model_w):
    """
    Load RE10K scene: read video frames and poses.

    Returns:
        all_frames: list of np.ndarray (model_h x model_w x 3)
        all_c2w_gl: list of 4x4 OpenGL c2w matrices
        scaled_intrinsic_model: 3x3 intrinsic at model resolution
    """
    video_path = os.path.join(data_path, "video", f"{scene_id}.mp4")
    pose_path = os.path.join(data_path, "pose", f"{scene_id}.json")

    with open(pose_path, 'r') as f:
        pose_data = json.load(f)
    poses = pose_data['poses']

    # Parse first pose for intrinsics (constant across frames)
    (fx_n, fy_n, cx_n, cy_n), _ = parse_re10k_pose(poses[0])

    # Read video to get frame dimensions
    cap = cv2.VideoCapture(video_path)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Build pixel-space intrinsics
    K_orig = np.array([
        [fx_n * img_w, 0, cx_n * img_w],
        [0, fy_n * img_h, cy_n * img_h],
        [0, 0, 1]
    ], dtype=np.float32)

    # Scale intrinsics: original → model resolution (crop mode)
    crop_scale = max(model_h / img_h, model_w / img_w)
    resized_h = int(round(img_h * crop_scale))
    resized_w = int(round(img_w * crop_scale))
    crop_offset_x = (resized_w - model_w) / 2.0
    crop_offset_y = (resized_h - model_h) / 2.0

    K_model = K_orig.copy()
    K_model[0, 0] *= crop_scale
    K_model[1, 1] *= crop_scale
    K_model[0, 2] *= crop_scale
    K_model[1, 2] *= crop_scale
    K_model[0, 2] -= crop_offset_x
    K_model[1, 2] -= crop_offset_y

    # Parse all c2w poses
    all_c2w_gl = []
    for p in poses:
        _, c2w_gl = parse_re10k_pose(p)
        all_c2w_gl.append(c2w_gl)

    return num_frames_video, all_c2w_gl, K_model, video_path, img_h, img_w


def read_video_frames(video_path, frame_indices, model_h, model_w):
    """Read specific frames from video and resize+crop to model resolution."""
    cap = cv2.VideoCapture(video_path)
    frames = {}
    max_idx = max(frame_indices)

    idx = 0
    needed = set(frame_indices)
    while cap.isOpened() and idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in needed:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_model, _, _, _ = resize_crop_to_rect(frame_rgb, model_h, model_w)
            frames[idx] = frame_model
        idx += 1
    cap.release()

    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_unique_frame_count(trajectory):
    """Number of unique frames to sample from the video."""
    if trajectory == "boomerang":
        return (TOTAL_VIDEO_FRAMES + 1) // 2  # 21
    return TOTAL_VIDEO_FRAMES  # 41


def video_pos_to_forward(video_pos, trajectory):
    """Map video position (0..40) to forward index."""
    n_unique = get_unique_frame_count(trajectory)
    if trajectory == "forward" or video_pos < n_unique:
        return video_pos
    return 2 * (n_unique - 1) - video_pos


def compute_ref_positions(num_input, trajectory):
    """Compute evenly spaced reference positions in forward range [0, n_unique-1]."""
    n_unique = get_unique_frame_count(trajectory)
    if num_input == 1:
        return [n_unique // 2]
    step = (n_unique - 1) / (num_input - 1)
    return [round(i * step) for i in range(num_input)]


def compute_keyframe_video_positions(ref_forward_positions, trajectory):
    """
    Given reference forward positions, compute all video positions where they appear.
    For boomerang: forward pos p appears at video_pos p and its mirror.
    For forward: video_pos == forward_pos directly.
    """
    n_unique = get_unique_frame_count(trajectory)
    kf_set = set()
    for p in ref_forward_positions:
        kf_set.add(p)
        if trajectory == "boomerang":
            mirror = 2 * (n_unique - 1) - p
            if mirror != p and mirror < TOTAL_VIDEO_FRAMES:
                kf_set.add(mirror)
    return sorted(kf_set)


# ──────────────────────────────────────────────────────────────────────────────
# Process a single scene
# ──────────────────────────────────────────────────────────────────────────────

def process_scene(pipe, args, scene_id, scene_idx, total_scenes, num_input):
    model_h = args.height
    model_w = args.width
    fps = args.fps
    trajectory = args.trajectory
    n_unique = get_unique_frame_count(trajectory)

    ref_forward_pos = compute_ref_positions(num_input, trajectory)
    keyframe_video_positions = compute_keyframe_video_positions(ref_forward_pos, trajectory)

    print(f"\n{'=' * 80}")
    print(f"[{scene_idx + 1}/{total_scenes}] Scene: {scene_id} (K={num_input})")
    print(f"{'=' * 80}")

    # Load scene metadata
    num_frames_video, all_c2w_gl, K_model, video_path, img_h, img_w = \
        load_re10k_scene(args.data_path, scene_id, model_h, model_w)

    stride = args.frame_stride
    span_needed = (n_unique - 1) * stride + 1

    if num_frames_video < span_needed:
        print(f"  Warning: only {num_frames_video} frames, need {span_needed} (stride={stride}). Skipping.")
        return None

    # Pick random forward window with stride
    rng = random.Random(SEED + hash(scene_id) % 10000)
    window_start = rng.randint(0, num_frames_video - span_needed)
    forward_global_indices = [window_start + i * stride for i in range(n_unique)]

    # Build trajectory mapping: video_pos → global frame index
    traj_global = []
    for vp in range(TOTAL_VIDEO_FRAMES):
        fwd = video_pos_to_forward(vp, trajectory)
        traj_global.append(forward_global_indices[fwd])

    ref_global = [forward_global_indices[p] for p in ref_forward_pos]

    traj_label = "boomerang" if trajectory == "boomerang" else "forward"
    print(f"  Video frames: {num_frames_video}, Image res: {img_w}x{img_h}")
    print(f"  Stride: {stride}, Unique frames: {n_unique}, Span: {span_needed}")
    print(f"  Forward window: [{forward_global_indices[0]}, {forward_global_indices[-1]}]")
    print(f"  K={num_input}, Ref forward pos: {ref_forward_pos}")
    print(f"  Ref global indices: {ref_global}")
    print(f"  Keyframe video positions: {keyframe_video_positions}")
    print(f"  Trajectory: {traj_label} ({TOTAL_VIDEO_FRAMES} frames)")

    # Read needed video frames (unique global indices)
    unique_global = sorted(set(traj_global))
    video_frames = read_video_frames(video_path, unique_global, model_h, model_w)

    if len(video_frames) < len(unique_global):
        print(f"  Warning: could only read {len(video_frames)}/{len(unique_global)} frames. Skipping.")
        return None

    # Map forward index → image and c2w
    forward_images = {}
    forward_extrinsics = {}
    for fwd_idx in range(n_unique):
        g_idx = forward_global_indices[fwd_idx]
        forward_images[fwd_idx] = video_frames[g_idx]
        forward_extrinsics[fwd_idx] = all_c2w_gl[g_idx]

    # Output directory
    output_dir = os.path.join(args.output_path, f"K{num_input}", scene_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save input (reference) views
    input_views_dir = os.path.join(output_dir, "input_views")
    os.makedirs(input_views_dir, exist_ok=True)
    for i, rfp in enumerate(ref_forward_pos):
        path = os.path.join(input_views_dir, f"view_{i}_fwd{rfp:02d}_global{forward_global_indices[rfp]:04d}.png")
        Image.fromarray(forward_images[rfp]).save(path)
    print(f"  Saved {num_input} input views")

    # Save GT frames
    gt_frames_dir = os.path.join(output_dir, "gt_frames")
    os.makedirs(gt_frames_dir, exist_ok=True)
    for vp in range(TOTAL_VIDEO_FRAMES):
        fwd = video_pos_to_forward(vp, trajectory)
        path = os.path.join(gt_frames_dir, f"frame_{vp:04d}.png")
        Image.fromarray(forward_images[fwd]).save(path)

    # Generate target frames (K-to-1)
    generated_frames = {}
    for vp in keyframe_video_positions:
        fwd = video_pos_to_forward(vp, trajectory)
        generated_frames[vp] = forward_images[fwd]

    target_video_positions = [vp for vp in range(TOTAL_VIDEO_FRAMES) if vp not in keyframe_video_positions]
    print(f"  Targets to generate: {len(target_video_positions)} frames")

    for ti, vp in enumerate(target_video_positions):
        fwd = video_pos_to_forward(vp, trajectory)
        print(f"  [{ti + 1}/{len(target_video_positions)}] Video pos {vp} "
              f"(fwd {fwd}, global {forward_global_indices[fwd]})...")

        context_indices_local = list(range(num_input))
        target_indices_local = [num_input]
        num_total = num_input + 1

        current_extrinsics = np.stack(
            [forward_extrinsics[p] for p in ref_forward_pos]
            + [forward_extrinsics[fwd]],
            axis=0,
        )
        current_intrinsics = np.stack([K_model] * num_total, axis=0)

        raymap, _, _ = prepare_raymap(
            current_extrinsics, current_intrinsics,
            context_indices_local, target_indices_local,
            model_h, model_w,
            no_pixel_unshuffle=args.no_pixel_unshuffle,
        )
        raymap = raymap.to("cuda", dtype=torch.bfloat16)

        context_images = [Image.fromarray(forward_images[p]) for p in ref_forward_pos]

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

        video_out = pipe(**pipe_kwargs)
        pred_frame = np.array(video_out[num_input])
        generated_frames[vp] = pred_frame

        gt_frame = forward_images[fwd]
        psnr = compute_psnr(pred_frame, gt_frame)
        print(f"    PSNR={psnr:.2f} dB")

    # Save generated frames
    gen_frames_dir = os.path.join(output_dir, "gen_frames")
    os.makedirs(gen_frames_dir, exist_ok=True)
    for vp in range(TOTAL_VIDEO_FRAMES):
        path = os.path.join(gen_frames_dir, f"frame_{vp:04d}.png")
        Image.fromarray(generated_frames[vp]).save(path)

    # Build frame lists
    gt_frame_list = [forward_images[video_pos_to_forward(vp, trajectory)] for vp in range(TOTAL_VIDEO_FRAMES)]
    gen_frame_list = [generated_frames[vp] for vp in range(TOTAL_VIDEO_FRAMES)]

    # GT video
    gt_video_path = os.path.join(output_dir, "gt_video.mp4")
    writer = imageio.get_writer(gt_video_path, fps=fps, codec="libx264", quality=8)
    for f in gt_frame_list:
        writer.append_data(f)
    writer.close()

    # Generated video
    gen_video_path = os.path.join(output_dir, "gen_video.mp4")
    writer = imageio.get_writer(gen_video_path, fps=fps, codec="libx264", quality=8)
    for f in gen_frame_list:
        writer.append_data(f)
    writer.close()

    # Comparison video
    comp_video_path = os.path.join(output_dir, "comparison_video.mp4")
    writer = imageio.get_writer(comp_video_path, fps=fps, codec="libx264", quality=8)
    for vp in range(TOTAL_VIDEO_FRAMES):
        comp = make_comparison_frame(gt_frame_list[vp], gen_frame_list[vp])
        writer.append_data(comp)
    writer.close()
    print(f"  Saved videos to {output_dir}")

    # Metrics
    psnrs = []
    for vp in target_video_positions:
        fwd = video_pos_to_forward(vp, trajectory)
        psnrs.append(compute_psnr(generated_frames[vp], forward_images[fwd]))

    mean_psnr = np.mean(psnrs)
    print(f"  Mean PSNR (non-keyframes): {mean_psnr:.2f} dB")

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Scene: {scene_id}\n")
        f.write(f"K={num_input} input views, {traj_label} trajectory ({TOTAL_VIDEO_FRAMES} frames)\n")
        f.write(f"Stride: {stride}, Forward indices: [{forward_global_indices[0]}, {forward_global_indices[-1]}]\n")
        f.write(f"Reference forward pos: {ref_forward_pos}\n")
        f.write(f"Reference global indices: {ref_global}\n")
        f.write(f"Model resolution: {model_h}x{model_w}\n")
        f.write(f"FPS: {fps}, Seed: {SEED}\n\n")
        f.write(f"Mean PSNR: {mean_psnr:.2f} dB\n\n")
        f.write("Per-frame PSNR:\n")
        for vp, psnr_val in zip(target_video_positions, psnrs):
            fwd = video_pos_to_forward(vp, trajectory)
            f.write(f"  Video pos {vp:2d} (fwd {fwd:2d}, global {forward_global_indices[fwd]}): {psnr_val:.2f} dB\n")

    return {"psnr": mean_psnr}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Parse scene assignments: "scene_id:K" pairs
    scene_k_pairs = []
    for item in args.scenes:
        if ':' in item:
            sid, k = item.rsplit(':', 1)
            scene_k_pairs.append((sid, int(k)))
        else:
            scene_k_pairs.append((item, args.default_k))

    n_unique = get_unique_frame_count(args.trajectory)
    print(f"RE10K Video Interpolation")
    print(f"  Data path: {args.data_path}")
    print(f"  Scenes: {len(scene_k_pairs)}")
    print(f"  Trajectory: {args.trajectory} ({TOTAL_VIDEO_FRAMES} frames, {n_unique} unique)")
    print(f"  Frame stride: {args.frame_stride}")
    print(f"  Model resolution: {args.height}x{args.width}")
    print(f"  FPS: {args.fps}, Seed: {SEED}")
    for sid, k in scene_k_pairs:
        print(f"    {sid} → K={k}")

    pipe = load_pipeline(args)
    os.makedirs(args.output_path, exist_ok=True)

    all_psnr = []
    all_labels = []
    for scene_idx, (scene_id, num_input) in enumerate(scene_k_pairs):
        result = process_scene(pipe, args, scene_id, scene_idx, len(scene_k_pairs), num_input)
        if result is not None:
            all_psnr.append(result["psnr"])
            all_labels.append(f"{scene_id} (K={num_input})")

    print(f"\n{'=' * 80}")
    print(f"OVERALL ({len(all_psnr)} scenes)")
    print(f"{'=' * 80}")
    if all_psnr:
        print(f"  Mean PSNR: {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB")

    summary_file = os.path.join(args.output_path, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"RE10K Video Interpolation — {len(all_psnr)} scenes\n")
        f.write(f"Trajectory: {args.trajectory} ({TOTAL_VIDEO_FRAMES} frames, {n_unique} unique)\n")
        f.write(f"Resolution: {args.height}x{args.width}, FPS: {args.fps}, Seed: {SEED}\n\n")
        if all_psnr:
            f.write(f"Mean PSNR: {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB\n\n")
        for i, label in enumerate(all_labels):
            f.write(f"  {label}: PSNR={all_psnr[i]:.2f}\n")

    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="RE10K boomerang video interpolation")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--new_in_dim", type=int, default=420)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to re10k split dir (e.g. .../re10k_reorganize/test)")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--scenes", type=str, nargs="+", required=True,
                        help="Scene IDs with K assignment, format: scene_id:K (e.g. abc123:3)")
    parser.add_argument("--default_k", type=int, default=3,
                        help="Default K if not specified per scene")
    parser.add_argument("--trajectory", type=str, default="boomerang",
                        choices=["boomerang", "forward"],
                        help="'boomerang' = a→b→c→b→a (41 frames), 'forward' = a→b→c (21 frames)")
    parser.add_argument("--frame_stride", type=int, default=5,
                        help="Stride between sampled video frames (default 5, i.e. every 5th frame)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False)
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False)

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
