#!/usr/bin/env python
"""
Evaluation script for WRIVA dataset using Wan2.1 model.
Performs N-to-M novel view synthesis with pose-based reference selection.
Supports both N-to-1 (default) and N-to-M (e.g. 5-to-5) via --num_targets.

Data flow:
  - Context/input images: wriva/inputs/{scene_name}/*.jpg
  - Context poses & intrinsics: wriva/inputs_colmap/{scene_name}/images.txt, cameras.txt
  - Target GT images: wriva/references/{scene_name}/*.jpg
  - Target poses & intrinsics: wriva/references_colmap/{scene_name}.json

Camera processing follows test_wan2.1_6to1.py (plucker rays + normalization).
Data loading & output format follows eval_nto1_wriva.py.
"""

import os
import time
import json
import argparse
import random
import gc

import torch
import numpy as np
import cv2
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict
from diffsynth.core.data.my_v2v_dataset_images_in_plucker_SE import (
    get_plucker_rays,
    normalize_w2c_make_cam_last_origin,
)


# ══════════════════════════════════════════════════════════════════════════════
# Model setup (from test_wan2.1_6to1.py)
# ══════════════════════════════════════════════════════════════════════════════

def modify_model_channels(pipe, model_attr, new_in_dim, device):
    """
    Modify the model's input dimension to match training configuration.

    Memory-efficient version: only the patch_embedding Conv3d layer differs
    when in_dim changes, so we replace it in-place instead of constructing
    an entire second 14B-parameter model.  The checkpoint loaded afterwards
    will supply the correct patch_embedding weights.
    """
    import torch.nn as nn

    model = getattr(pipe, model_attr)
    if model is None:
        return

    old_in_dim = model.in_dim
    print(f"Modifying {model_attr} input channels: in_dim {old_in_dim}->{new_in_dim}")

    # 1. Replace patch_embedding with the correct in_dim
    old_pe = model.patch_embedding
    pe_device = next(old_pe.parameters()).device
    pe_dtype = next(old_pe.parameters()).dtype
    new_pe = nn.Conv3d(new_in_dim, model.dim,
                       kernel_size=model.patch_size, stride=model.patch_size)
    new_pe = new_pe.to(device=pe_device, dtype=pe_dtype)
    model.patch_embedding = new_pe
    del old_pe

    # 2. Update bookkeeping attributes
    model.in_dim = new_in_dim
    model.seperated_encoding = True
    model.fuse_vae_embedding_in_latents_multiple = False

    torch.cuda.empty_cache()
    gc.collect()

    # 3. Ensure model lives on the target device in bfloat16
    model = model.to(device=device, dtype=torch.bfloat16)
    setattr(pipe, model_attr, model)
    print(f"Model {model_attr} channels modified successfully (in-place)")


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
        vram_limit=46,
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


# ══════════════════════════════════════════════════════════════════════════════
# COLMAP file reading utilities (from eval_nto1_wriva.py)
# ══════════════════════════════════════════════════════════════════════════════

def qvec2rotmat(qvec):
    """Convert quaternion [QW, QX, QY, QZ] to rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz,   2*qz*qx + 2*qw*qy],
            [2*qx*qy + 2*qw*qz,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
            [2*qz*qx - 2*qw*qy,     2*qy*qz + 2*qw*qx,   1 - 2*qx**2 - 2*qy**2],
        ]
    )


def create_pose_matrix(qvec, tvec):
    """Create 4x4 w2c pose matrix from COLMAP quaternion and translation."""
    R = qvec2rotmat(qvec)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = tvec
    return pose


def read_cameras_text(path_to_cameras_txt):
    """Read COLMAP cameras.txt file and return dictionary of camera intrinsics."""
    cameras = {}
    with open(path_to_cameras_txt, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                camera_id = int(elements[0])
                model = elements[1]
                width = int(elements[2])
                height = int(elements[3])
                params = np.array(tuple(map(float, elements[4:])))
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params,
                }
    return cameras


def read_images_text(path_to_images_txt):
    """Read COLMAP images.txt file and return dictionary of image poses."""
    images = {}
    with open(path_to_images_txt, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                image_id = int(elements[0])
                qvec = np.array(tuple(map(float, elements[1:5])))
                tvec = np.array(tuple(map(float, elements[5:8])))
                camera_id = int(elements[8])
                image_name = elements[9]
                fid.readline()  # skip POINTS2D line
                images[image_name] = {
                    'id': image_id,
                    'qvec': qvec,
                    'tvec': tvec,
                    'camera_id': camera_id,
                }
    return images


def create_intrinsic_matrix(camera_data):
    """Create 3x3 intrinsic matrix from COLMAP camera data."""
    model = camera_data['model']
    params = camera_data['params']
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f = params[0]
        cx, cy = params[1], params[2]
        fx = fy = f
    elif model == "OPENCV":
        fx, fy, cx, cy = params[:4]
    else:
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def load_input_poses_and_intrinsics(colmap_dir):
    """
    Load w2c poses and intrinsics from COLMAP input files.

    Returns:
        poses_dict: image_name -> (4, 4) w2c pose
        intrinsics_dict: image_name -> (3, 3) intrinsic matrix
        cam_sizes_dict: image_name -> (width, height) from cameras.txt
    """
    images_data = read_images_text(os.path.join(colmap_dir, "images.txt"))
    cameras_data = read_cameras_text(os.path.join(colmap_dir, "cameras.txt"))

    poses_dict = {}
    intrinsics_dict = {}
    cam_sizes_dict = {}

    for image_name, info in images_data.items():
        pose = create_pose_matrix(info['qvec'], info['tvec'])
        poses_dict[image_name] = pose

        cam = cameras_data.get(info['camera_id'])
        if cam is not None:
            intrinsics_dict[image_name] = create_intrinsic_matrix(cam)
            cam_sizes_dict[image_name] = (cam['width'], cam['height'])

    return poses_dict, intrinsics_dict, cam_sizes_dict


def load_target_poses_and_intrinsics(json_path):
    """
    Load w2c poses and intrinsics from references_colmap JSON.

    Returns:
        poses_dict: image_name -> (4, 4) w2c pose
        intrinsics_dict: image_name -> (3, 3) intrinsic matrix
        orig_sizes_dict: image_name -> (width, height) from JSON rows/columns
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    poses_dict = {}
    intrinsics_dict = {}
    orig_sizes_dict = {}

    for frame_key, frame_data in data.items():
        image_name = frame_data.get('Image Name', frame_key)

        ext = frame_data.get('Extrinsics', {})
        if not ext:
            continue
        qvec = np.array([ext['QW'], ext['QX'], ext['QY'], ext['QZ']])
        tvec = np.array([ext['TX'], ext['TY'], ext['TZ']])
        poses_dict[image_name] = create_pose_matrix(qvec, tvec)

        intr = frame_data.get('Intrinsics', {}).get('Params', {})
        if intr:
            fx = intr.get('fx', 0.0)
            fy = intr.get('fy', 0.0)
            cx = intr.get('cx', 0.0)
            cy = intr.get('cy', 0.0)
            intrinsics_dict[image_name] = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )
            orig_sizes_dict[image_name] = (
                int(intr.get('columns', 0)),
                int(intr.get('rows', 0)),
            )

    return poses_dict, intrinsics_dict, orig_sizes_dict


# ══════════════════════════════════════════════════════════════════════════════
# Reference selection (from eval_nto1_wriva.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_pose_distance(pose1, pose2):
    """
    Combined translation + rotation distance between two w2c poses.
    """
    t1, t2 = pose1[:3, 3], pose2[:3, 3]
    translation_dist = np.linalg.norm(t1 - t2)

    R1, R2 = pose1[:3, :3], pose2[:3, :3]
    R_diff = R1.T @ R2
    trace = np.clip(np.trace(R_diff), -1, 3)
    rotation_dist = np.arccos((trace - 1) / 2)

    return translation_dist + 0.5 * rotation_dist


def select_closest_refs(ref_poses, ref_names, tgt_pose, num_ref):
    """Select the num_ref closest reference images to tgt_pose."""
    if len(ref_names) <= num_ref:
        return list(ref_names)
    dists = [(compute_pose_distance(ref_poses[n], tgt_pose), n) for n in ref_names]
    dists.sort(key=lambda x: x[0])
    return [n for _, n in dists[:num_ref]]


def select_closest_refs_for_group(ref_poses, ref_names, tgt_poses_list, num_ref):
    """Select the num_ref closest reference images to a *group* of targets.

    For each candidate context view, the score is its minimum distance to any
    target in the group.  The top-num_ref candidates are returned.
    """
    if len(ref_names) <= num_ref:
        return list(ref_names)
    scored = []
    for n in ref_names:
        min_dist = min(compute_pose_distance(ref_poses[n], tp) for tp in tgt_poses_list)
        scored.append((min_dist, n))
    scored.sort(key=lambda x: x[0])
    return [n for _, n in scored[:num_ref]]


# ══════════════════════════════════════════════════════════════════════════════
# Camera processing  (follows test_wan2.1_6to1.py conventions)
# ══════════════════════════════════════════════════════════════════════════════

def resize_stretch(img, tgt_h, tgt_w):
    """
    Simply resize (stretch) the image to (tgt_h, tgt_w) without preserving
    aspect ratio.

    Returns:
        resized: (tgt_h, tgt_w, 3) uint8 numpy array
        crop_params: dict (offset_x/y are 0; resize_w/h equal tgt_w/h)
    """
    h, w = img.shape[:2]
    resized = cv2.resize(img, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
    crop_params = {
        'resize_w': tgt_w,
        'resize_h': tgt_h,
        'offset_x': 0,
        'offset_y': 0,
    }
    return resized, crop_params


def resize_and_center_crop(img, tgt_h, tgt_w):
    """
    Resize so the image covers (tgt_h, tgt_w), then center-crop.

    The shorter side (relative to the target aspect ratio) is matched exactly;
    the longer side is cropped symmetrically.

    Args:
        img: (H, W, 3) uint8 numpy array
        tgt_h, tgt_w: desired output size

    Returns:
        cropped: (tgt_h, tgt_w, 3) uint8 numpy array
        crop_params: dict with 'resize_h', 'resize_w', 'offset_x', 'offset_y'
                     (needed for intrinsic adjustment)
    """
    h, w = img.shape[:2]
    tgt_aspect = tgt_w / tgt_h
    src_aspect = w / h

    if src_aspect > tgt_aspect:
        # Source is wider → match height, crop width
        resize_h = tgt_h
        resize_w = int(round(w * tgt_h / h))
    else:
        # Source is taller → match width, crop height
        resize_w = tgt_w
        resize_h = int(round(h * tgt_w / w))

    resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

    offset_x = (resize_w - tgt_w) // 2
    offset_y = (resize_h - tgt_h) // 2
    cropped = resized[offset_y : offset_y + tgt_h, offset_x : offset_x + tgt_w]

    crop_params = {
        'resize_w': resize_w,
        'resize_h': resize_h,
        'offset_x': offset_x,
        'offset_y': offset_y,
    }
    return cropped, crop_params


def resize_image(img, tgt_h, tgt_w, mode="crop"):
    """Dispatch to the appropriate resize function based on *mode*.

    Args:
        mode: "crop" → resize-and-center-crop  |  "stretch" → plain resize
    """
    if mode == "stretch":
        return resize_stretch(img, tgt_h, tgt_w)
    else:
        return resize_and_center_crop(img, tgt_h, tgt_w)


def scale_intrinsic_with_crop(K, orig_w, orig_h, crop_params):
    """
    Adjust a 3x3 intrinsic matrix for resize-then-center-crop.

    Steps:
      1. Scale fx, fy, cx, cy for the resize from (orig_w, orig_h)
         to (crop_params['resize_w'], crop_params['resize_h']).
      2. Shift cx, cy by the crop offset.
    """
    resize_w = crop_params['resize_w']
    resize_h = crop_params['resize_h']
    offset_x = crop_params['offset_x']
    offset_y = crop_params['offset_y']

    sx = resize_w / orig_w
    sy = resize_h / orig_h

    K_out = K.copy()
    K_out[0, 0] *= sx          # fx
    K_out[0, 2] = K[0, 2] * sx - offset_x   # cx
    K_out[1, 1] *= sy          # fy
    K_out[1, 2] = K[1, 2] * sy - offset_y   # cy
    return K_out


def prepare_raymap_wriva(w2c_poses_np, intrinsics_np, height, width):
    """
    Compute plucker ray map for WRIVA cameras.

    Unlike DL3DV (c2w in OpenGL convention), WRIVA poses are already
    w2c in OpenCV/COLMAP convention.  We skip the c2w->w2c inversion and
    the OpenGL->OpenCV sign flip.

    Args:
        w2c_poses_np: (N, 4, 4) numpy w2c poses — context first, target last
        intrinsics_np: (N, 3, 3) numpy intrinsics at model resolution
        height, width: model resolution

    Returns:
        raymap: (N, C, H/8, W/8) tensor, bfloat16 on cuda
    """
    w2cs = torch.from_numpy(w2c_poses_np).float()

    # Normalize so last camera (target) is at origin
    _, c2w_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)

    # Compute plucker rays  →  [N, 6*64, H/8, W/8]
    raymap = get_plucker_rays(
        c2w_norm,
        torch.from_numpy(intrinsics_np).float(),
        height=height,
        width=width,
    )
    return raymap


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(pred, gt):
    """PSNR between two uint8 numpy arrays."""
    mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


# ══════════════════════════════════════════════════════════════════════════════
# Process a single WRIVA scene
# ══════════════════════════════════════════════════════════════════════════════

def process_scene(
    pipe, args, scene_name, scene_idx, total_scenes,
    dreamsim_model=None, dreamsim_preprocess=None,
    ssim_fn=None, lpips_model=None,
):
    """
    Process a single WRIVA scene: load data, run N-to-1 NVS for each
    target frame, compute metrics, save results.

    Returns:
        dict with per-scene mean metrics, or None on failure.
    """
    model_h = args.height
    model_w = args.width

    # ── Paths ─────────────────────────────────────────────────────────────
    input_dir = os.path.join(args.wriva_path, "inputs", scene_name)
    ref_dir = os.path.join(args.wriva_path, "references", scene_name)
    colmap_dir = os.path.join(args.wriva_path, "inputs_colmap", scene_name)
    ref_json = os.path.join(args.wriva_path, "references_colmap", f"{scene_name}.json")

    print(f"\n{'='*80}")
    print(f"[{scene_idx+1}/{total_scenes}] Processing scene: {scene_name}")
    print(f"{'='*80}")

    # ── Validate paths ────────────────────────────────────────────────────
    for p, desc in [(input_dir, "inputs"), (ref_dir, "references"),
                     (colmap_dir, "inputs_colmap"), (ref_json, "references_colmap JSON")]:
        if not os.path.exists(p):
            print(f"  Warning: {desc} not found at {p}. Skipping scene.")
            return None

    # ── Load input (context) poses & intrinsics ───────────────────────────
    inp_poses, inp_intrinsics, inp_cam_sizes = load_input_poses_and_intrinsics(colmap_dir)
    print(f"  Loaded {len(inp_poses)} input poses from COLMAP")

    # ── Load target poses & intrinsics ────────────────────────────────────
    tgt_poses, tgt_intrinsics, tgt_orig_sizes = load_target_poses_and_intrinsics(ref_json)
    tgt_names = sorted(tgt_poses.keys())
    print(f"  Loaded {len(tgt_names)} target poses from JSON")

    if len(tgt_names) == 0:
        print("  Warning: No target frames found. Skipping scene.")
        return None

    # ── Load & resize input images to model resolution ────────────────────
    # Only load images that exist on disk and have COLMAP data
    inp_images = {}   # image_name -> (H, W, 3) uint8 at model resolution
    inp_scaled_K = {}  # image_name -> (3, 3) intrinsic at model resolution
    available_inp_names = []

    for img_name in sorted(inp_poses.keys()):
        img_path = os.path.join(input_dir, img_name)
        if not os.path.exists(img_path):
            continue
        if img_name not in inp_intrinsics:
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = img.shape[:2]
        img_cropped, crop_params = resize_image(img, model_h, model_w, mode=args.resize_mode)
        inp_images[img_name] = img_cropped

        # Scale intrinsics: resize + center-crop adjustment
        K_scaled = scale_intrinsic_with_crop(
            inp_intrinsics[img_name], orig_w, orig_h, crop_params
        )
        inp_scaled_K[img_name] = K_scaled

        available_inp_names.append(img_name)

    print(f"  Loaded {len(available_inp_names)} input images at {model_h}x{model_w}")

    if len(available_inp_names) == 0:
        print("  Warning: No usable input images. Skipping scene.")
        return None

    # ── Load target GT images ─────────────────────────────────────────────
    tgt_images_model = {}   # image_name -> (H, W, 3) uint8 at model resolution
    tgt_crop_params = {}    # image_name -> crop_params dict
    for tgt_name in tgt_names:
        tgt_path = os.path.join(ref_dir, tgt_name)
        if os.path.exists(tgt_path):
            img = np.array(Image.open(tgt_path).convert("RGB"))
            img_cropped, crop_p = resize_image(img, model_h, model_w, mode=args.resize_mode)
            tgt_images_model[tgt_name] = img_cropped
            tgt_crop_params[tgt_name] = crop_p

    # ── Setup output directory ────────────────────────────────────────────
    output_dir = os.path.join(args.output_path, scene_name)
    side_by_side_dir = os.path.join(output_dir, "side_by_side")
    gt_target_dir = os.path.join(output_dir, "gt_target")
    predicted_dir = os.path.join(output_dir, "predicted_target")
    for d in [output_dir, side_by_side_dir, gt_target_dir, predicted_dir]:
        os.makedirs(d, exist_ok=True)

    # Determine effective num_ref and num_targets
    num_ref = min(args.num_ref, len(available_inp_names))
    num_targets = args.num_targets
    num_total = num_ref + num_targets
    print(f"  Using {num_ref} context views + {num_targets} target(s) = {num_total} total frames")

    # Limit targets per scene if requested
    target_names_to_process = tgt_names
    if args.max_targets_per_scene is not None and args.max_targets_per_scene < len(tgt_names):
        target_names_to_process = random.sample(tgt_names, args.max_targets_per_scene)
        target_names_to_process.sort()
        print(f"  Processing {len(target_names_to_process)}/{len(tgt_names)} target frames (random subset)")

    # ── Process target frames in batches of num_targets ─────────────────
    psnrs, ssims, lpips_scores, dreamsim_scores = [], [], [], []

    # Split targets into batches
    batches = [
        target_names_to_process[i : i + num_targets]
        for i in range(0, len(target_names_to_process), num_targets)
    ]
    total_batches = len(batches)

    for batch_idx, batch_tgt_names in enumerate(batches):
        batch_size = len(batch_tgt_names)
        global_offset = batch_idx * num_targets
        print(f"\n  [Batch {batch_idx+1}/{total_batches}] "
              f"Targets {global_offset+1}-{global_offset+batch_size}/{len(target_names_to_process)}: "
              f"{[os.path.splitext(n)[0] for n in batch_tgt_names]}")

        batch_tgt_poses = [tgt_poses[n] for n in batch_tgt_names]

        # Select context views closest to this group of targets
        if batch_size == 1:
            selected = select_closest_refs(
                inp_poses, available_inp_names, batch_tgt_poses[0], num_ref
            )
        else:
            selected = select_closest_refs_for_group(
                inp_poses, available_inp_names, batch_tgt_poses, num_ref
            )
        actual_num_ctx = len(selected)
        actual_num_total = actual_num_ctx + batch_size
        print(f"    Selected {actual_num_ctx} context views: {[s[:20] for s in selected]}")

        # Prepare context images as PIL (for pipeline input)
        context_pil = [Image.fromarray(inp_images[n]) for n in selected]

        # Build w2c pose array: [ctx_0, ..., ctx_{N-1}, tgt_0, ..., tgt_{K-1}]
        all_w2c = np.stack(
            [inp_poses[n] for n in selected] + batch_tgt_poses, axis=0
        )

        # Build intrinsic array at model resolution
        batch_tgt_K_list = []
        for tgt_name in batch_tgt_names:
            if tgt_name in tgt_intrinsics and tgt_name in tgt_orig_sizes:
                tgt_orig_w, tgt_orig_h = tgt_orig_sizes[tgt_name]
                if tgt_name in tgt_crop_params:
                    tgt_cp = tgt_crop_params[tgt_name]
                else:
                    _, tgt_cp = resize_image(
                        np.zeros((tgt_orig_h, tgt_orig_w, 3), dtype=np.uint8),
                        model_h, model_w, mode=args.resize_mode,
                    )
                batch_tgt_K_list.append(
                    scale_intrinsic_with_crop(tgt_intrinsics[tgt_name], tgt_orig_w, tgt_orig_h, tgt_cp)
                )
            else:
                batch_tgt_K_list.append(inp_scaled_K[selected[0]])
                print(f"    Warning: No intrinsics for {tgt_name}, using context[0]")

        all_K = np.stack(
            [inp_scaled_K[n] for n in selected] + batch_tgt_K_list, axis=0
        )

        # Compute plucker raymap
        raymap = prepare_raymap_wriva(all_w2c, all_K, model_h, model_w)
        raymap = raymap.to("cuda", dtype=torch.bfloat16)

        # Run inference
        pipe_kwargs = dict(
            prompt="",
            negative_prompt="",
            input_image=context_pil,
            input_video=None,
            raymap=raymap,
            height=model_h,
            width=model_w,
            num_frames=actual_num_total,
            num_latent_frames=actual_num_total,
            cfg_scale=1.0,
            num_inference_steps=args.num_inference_steps,
            seed=42,
            tiled=True,
        )
        if args.zero_temporal_rope:
            pipe_kwargs["zero_temporal_rope"] = True

        video = pipe(**pipe_kwargs)

        # Extract generated target frames (last batch_size frames)
        pred_frames = video[-batch_size:]

        for k, tgt_name in enumerate(batch_tgt_names):
            tgt_base = os.path.splitext(tgt_name)[0]
            pred_pil = pred_frames[k]
            pred_np = np.array(pred_pil)

            pred_path = os.path.join(predicted_dir, f"{tgt_base}.png")
            pred_pil.save(pred_path)

            # ── Metrics (compare at model resolution) ────────────────
            gt_np = tgt_images_model.get(tgt_name)

            if gt_np is not None:
                gt_path = os.path.join(gt_target_dir, f"{tgt_base}.png")
                Image.fromarray(gt_np).save(gt_path)

                separator = np.full((model_h, 10, 3), 255, dtype=np.uint8)
                sbs = np.concatenate([gt_np, separator, pred_np], axis=1)
                sbs_path = os.path.join(side_by_side_dir, f"{tgt_base}_gt_pred.png")
                Image.fromarray(sbs).save(sbs_path)

                psnr = compute_psnr(pred_np, gt_np)
                psnrs.append(psnr)

                ssim_value = None
                if ssim_fn is not None:
                    ssim_value = ssim_fn(
                        gt_np, pred_np, multichannel=True, channel_axis=2, data_range=255
                    )
                    ssims.append(ssim_value)

                lpips_value = None
                if lpips_model is not None:
                    pred_t = torch.from_numpy(pred_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                    gt_t = torch.from_numpy(gt_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                    with torch.no_grad():
                        lpips_value = lpips_model(pred_t.cuda(), gt_t.cuda()).item()
                    lpips_scores.append(lpips_value)

                ds_score = None
                if dreamsim_model is not None:
                    pred_ds = dreamsim_preprocess(Image.fromarray(pred_np)).to("cuda")
                    gt_ds = dreamsim_preprocess(Image.fromarray(gt_np)).to("cuda")
                    if pred_ds.dim() == 3:
                        pred_ds = pred_ds.unsqueeze(0)
                    if gt_ds.dim() == 3:
                        gt_ds = gt_ds.unsqueeze(0)
                    while pred_ds.dim() > 4:
                        pred_ds = pred_ds.squeeze(0)
                    while gt_ds.dim() > 4:
                        gt_ds = gt_ds.squeeze(0)
                    with torch.no_grad():
                        ds_score = dreamsim_model(pred_ds, gt_ds).item()
                    dreamsim_scores.append(ds_score)

                msg = f"    [{tgt_base}] PSNR={psnr:.2f} dB"
                if ssim_value is not None:
                    msg += f", SSIM={ssim_value:.4f}"
                if lpips_value is not None:
                    msg += f", LPIPS={lpips_value:.4f}"
                if ds_score is not None:
                    msg += f", DreamSim={ds_score:.4f}"
                print(msg)
            else:
                print(f"    No GT image available for {tgt_name}, skipping metrics")

    # ── Scene summary ─────────────────────────────────────────────────────
    scene_metrics = {
        'psnr': float(np.mean(psnrs)) if psnrs else 0,
        'ssim': float(np.mean(ssims)) if ssims else 0,
        'lpips': float(np.mean(lpips_scores)) if lpips_scores else 0,
        'dreamsim': float(np.mean(dreamsim_scores)) if dreamsim_scores else 0,
    }

    print(f"\n  Scene {scene_name} summary:")
    print(f"    Mean PSNR:     {scene_metrics['psnr']:.2f} dB  ({len(psnrs)} frames)")
    if ssims:
        print(f"    Mean SSIM:     {scene_metrics['ssim']:.4f}")
    if lpips_scores:
        print(f"    Mean LPIPS:    {scene_metrics['lpips']:.4f}")
    if dreamsim_scores:
        print(f"    Mean DreamSim: {scene_metrics['dreamsim']:.4f}")

    # Save per-scene metrics
    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Scene: {scene_name}\n")
        f.write(f"Method: {num_ref}-to-{num_targets} generation\n")
        f.write(f"Model resolution: {model_h}x{model_w}\n")
        f.write(f"Targets processed: {len(target_names_to_process)}\n\n")
        f.write(f"Mean PSNR: {scene_metrics['psnr']:.2f} dB\n")
        if ssims:
            f.write(f"Mean SSIM: {scene_metrics['ssim']:.4f}\n")
        if lpips_scores:
            f.write(f"Mean LPIPS: {scene_metrics['lpips']:.4f}\n")
        if dreamsim_scores:
            f.write(f"Mean DreamSim: {scene_metrics['dreamsim']:.4f}\n")
        f.write(f"\nPer-frame metrics:\n")
        for i, tgt_name in enumerate(target_names_to_process):
            tgt_base = os.path.splitext(tgt_name)[0]
            line = f"  {tgt_base}:"
            if i < len(psnrs):
                line += f" PSNR={psnrs[i]:.2f} dB"
            if i < len(ssims):
                line += f", SSIM={ssims[i]:.4f}"
            if i < len(lpips_scores):
                line += f", LPIPS={lpips_scores[i]:.4f}"
            if i < len(dreamsim_scores):
                line += f", DreamSim={dreamsim_scores[i]:.4f}"
            f.write(line + "\n")

    print(f"  Metrics saved to {metrics_file}")
    return scene_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Scene discovery
# ══════════════════════════════════════════════════════════════════════════════

def discover_valid_scenes(wriva_path):
    """
    Return sorted list of scene names that have all four required sub-paths:
    inputs/, references/, inputs_colmap/, references_colmap/*.json
    """
    inputs = set(os.listdir(os.path.join(wriva_path, "inputs")))
    refs = set(os.listdir(os.path.join(wriva_path, "references")))
    ic = set(os.listdir(os.path.join(wriva_path, "inputs_colmap")))
    rc = set(
        f.replace(".json", "")
        for f in os.listdir(os.path.join(wriva_path, "references_colmap"))
        if f.endswith(".json")
    )
    valid = sorted(inputs & refs & ic & rc)
    return valid


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Discover & select scenes ──────────────────────────────────────────
    all_scenes = discover_valid_scenes(args.wriva_path)
    print(f"Found {len(all_scenes)} valid scenes in {args.wriva_path}")

    if args.scenes:
        # Use explicitly provided scenes
        scenes = args.scenes
    else:
        # Randomly sample num_scenes
        if args.num_scenes >= len(all_scenes):
            scenes = all_scenes
        else:
            scenes = sorted(random.sample(all_scenes, args.num_scenes))
    print(f"Will evaluate on {len(scenes)} scenes")

    print(f"Model resolution: {args.height}x{args.width}")
    print(f"Resize mode: {args.resize_mode}")
    print(f"Num context views (max): {args.num_ref}")
    print(f"Num targets per pass: {args.num_targets}")

    # ── Load pipeline ─────────────────────────────────────────────────────
    pipe = load_pipeline(args)

    # ── Optional metrics models ───────────────────────────────────────────
    dreamsim_model, dreamsim_preprocess = None, None
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

    # ── Process scenes ────────────────────────────────────────────────────
    os.makedirs(args.output_path, exist_ok=True)

    all_psnr, all_ssim, all_lpips, all_dreamsim = [], [], [], []

    for scene_idx, scene_name in enumerate(scenes):
        result = process_scene(
            pipe, args, scene_name, scene_idx, len(scenes),
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

    # ── Overall summary ───────────────────────────────────────────────────
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
        f.write(f"WRIVA Evaluation — Overall Results ({len(all_psnr)} scenes)\n")
        f.write(f"Model resolution: {args.height}x{args.width}\n")
        f.write(f"Num context views (max): {args.num_ref}\n")
        f.write(f"Num targets per pass: {args.num_targets}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n\n")
        if all_psnr:
            f.write(f"Mean PSNR:     {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB\n")
        if all_ssim:
            f.write(f"Mean SSIM:     {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}\n")
        if all_lpips:
            f.write(f"Mean LPIPS:    {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}\n")
        if all_dreamsim:
            f.write(f"Mean DreamSim: {np.mean(all_dreamsim):.4f} ± {np.std(all_dreamsim):.4f}\n")
        f.write(f"\nPer-scene results:\n")
        for i, scene_name in enumerate(scenes):
            if i < len(all_psnr):
                line = f"  {scene_name}: PSNR={all_psnr[i]:.2f}"
                if i < len(all_ssim):
                    line += f", SSIM={all_ssim[i]:.4f}"
                if i < len(all_lpips):
                    line += f", LPIPS={all_lpips[i]:.4f}"
                if i < len(all_dreamsim):
                    line += f", DreamSim={all_dreamsim[i]:.4f}"
                f.write(line + "\n")
            else:
                f.write(f"  {scene_name}: FAILED\n")

    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Wan2.1 N-to-M NVS evaluation on WRIVA dataset"
    )

    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained checkpoint (.safetensors)")
    parser.add_argument("--new_in_dim", type=int, default=420,
                        help="Input dimension for the modified model (must match training)")

    # Data paths
    parser.add_argument("--wriva_path", type=str,
                        default="/ocean/projects/cis250200p/mjeon2/datasets/wriva",
                        help="Base path to the WRIVA dataset")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for results")

    # Scenes
    parser.add_argument("--scenes", type=str, nargs='+', default=None,
                        help="Specific scene names to process (overrides --num_scenes)")
    parser.add_argument("--num_scenes", type=int, default=50,
                        help="Number of random scenes to evaluate (default: 50)")

    # Resolution
    parser.add_argument("--height", type=int, default=192,
                        help="Model input height")
    parser.add_argument("--width", type=int, default=336,
                        help="Model input width")

    # Resize mode
    parser.add_argument("--resize_mode", type=str, default="crop",
                        choices=["crop", "stretch"],
                        help="How to fit images to model resolution: "
                             "'crop' = resize + center-crop (default), "
                             "'stretch' = plain resize (may distort)")

    # Generation
    parser.add_argument("--num_ref", type=int, default=6,
                        help="Max number of context/reference views (default: 6)")
    parser.add_argument("--num_targets", type=int, default=1,
                        help="Number of target views to generate per forward pass (default: 1)")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--max_targets_per_scene", type=int, default=None,
                        help="Limit number of targets per scene (default: all)")
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False,
                        help="Zero out temporal RoPE (identity rotation for frame axis). "
                             "Requires checkpoint trained with --zero_temporal_rope.")

    # Metrics
    parser.add_argument("--use_dreamsim", action="store_true")
    parser.add_argument("--use_ssim", action="store_true")
    parser.add_argument("--use_lpips", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

