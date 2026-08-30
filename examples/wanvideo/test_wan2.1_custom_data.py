"""
Inference on custom test data with per-image intrinsics and OpenCV c2w poses.

Each example folder has:
  - N .jpg images (sorted alphabetically; first N-1 = input, last = target)
  - extrinsics.npy: (N, 4, 4) OpenCV c2w matrices
  - intrinsics.npy: (N, 3, 3) per-image intrinsic matrices

Saves: generated image, GT image, comparison image, and input views
(all at model resolution).
"""

import os
import sys
import time
import argparse
import random

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

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
    """Resize to cover target, then center crop. Returns (cropped, scale, ox, oy)."""
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


def adjust_intrinsic_for_crop(intrinsic, orig_h, orig_w, target_h, target_w):
    """Adjust intrinsic for resize-to-cover + center-crop."""
    scale = max(target_h / orig_h, target_w / orig_w)
    resized_h = int(round(orig_h * scale))
    resized_w = int(round(orig_w * scale))
    crop_offset_x = (resized_w - target_w) / 2.0
    crop_offset_y = (resized_h - target_h) / 2.0

    K = intrinsic.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale
    K[0, 2] -= crop_offset_x
    K[1, 2] -= crop_offset_y
    return K


def prepare_raymap_opencv_c2w(c2w_matrices, intrinsics, context_indices, target_indices,
                               height, width, no_pixel_unshuffle=False):
    """
    Prepare raymap from OpenCV c2w poses.
    c2w (OpenCV) -> w2c (OpenCV) -> normalize -> plucker rays.
    """
    context_c2w = c2w_matrices[context_indices]
    target_c2w = c2w_matrices[target_indices]
    all_c2w = np.concatenate([context_c2w, target_c2w], axis=0)
    all_c2w = torch.from_numpy(all_c2w).float()

    w2cs = torch.linalg.inv(all_c2w)

    context_K = intrinsics[context_indices]
    target_K = intrinsics[target_indices]
    all_K = np.concatenate([context_K, target_K], axis=0)
    K_tensor = torch.from_numpy(all_K).float()

    _, camera_poses_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)

    raymap = get_plucker_rays(
        camera_poses_norm, K_tensor,
        height=height, width=width,
        no_pixel_unshuffle=no_pixel_unshuffle,
    )
    if isinstance(raymap, np.ndarray):
        raymap = torch.from_numpy(raymap).float()

    return raymap, camera_poses_norm, K_tensor


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
# Process one example
# ──────────────────────────────────────────────────────────────────────────────

def process_example(pipe, args, example_dir, example_name, idx, total):
    model_h = args.height
    model_w = args.width

    print(f"\n{'=' * 80}")
    print(f"[{idx + 1}/{total}] Example: {example_name}")
    print(f"{'=' * 80}")

    extrinsics = np.load(os.path.join(example_dir, "extrinsics.npy"))  # (N, 4, 4) OpenCV c2w
    intrinsics = np.load(os.path.join(example_dir, "intrinsics.npy"))  # (N, 3, 3) per-image

    image_files = sorted([f for f in os.listdir(example_dir) if f.endswith(".jpg") or f.endswith(".png")])
    assert len(image_files) == extrinsics.shape[0], \
        f"Mismatch: {len(image_files)} images vs {extrinsics.shape[0]} poses"

    num_input = len(image_files) - 1
    input_files = image_files[:num_input]
    target_file = image_files[num_input]

    print(f"  Images: {len(image_files)}, Input: {num_input}, Target: {target_file}")

    # Load and resize all images, adjust intrinsics per image
    all_images_model = []
    all_intrinsics_model = []
    for i, fname in enumerate(image_files):
        img = np.array(Image.open(os.path.join(example_dir, fname)).convert("RGB"))
        orig_h, orig_w = img.shape[:2]
        img_model, _, _, _ = resize_crop_to_rect(img, model_h, model_w)
        K_model = adjust_intrinsic_for_crop(intrinsics[i], orig_h, orig_w, model_h, model_w)
        all_images_model.append(img_model)
        all_intrinsics_model.append(K_model)

    all_intrinsics_model = np.stack(all_intrinsics_model, axis=0)

    # Output directory
    output_dir = os.path.join(args.output_path, example_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save input views
    for i in range(num_input):
        path = os.path.join(output_dir, f"input_{i}_{input_files[i].replace('.jpg','.png').replace('.jpeg','.png')}")
        if not path.endswith('.png'):
            path = os.path.splitext(path)[0] + '.png'
        Image.fromarray(all_images_model[i]).save(path)
    print(f"  Saved {num_input} input views")

    # Save GT
    gt_frame = all_images_model[num_input]
    gt_path = os.path.join(output_dir, f"gt_{target_file.replace('.jpg','.png').replace('.jpeg','.png')}")
    if not gt_path.endswith('.png'):
        gt_path = os.path.splitext(gt_path)[0] + '.png'
    Image.fromarray(gt_frame).save(gt_path)
    print(f"  Saved GT: {os.path.basename(gt_path)}")

    # Prepare inference
    context_indices = list(range(num_input))
    target_indices = [num_input]
    num_total = num_input + 1

    raymap, camera_poses_norm, K_tensor = prepare_raymap_opencv_c2w(
        extrinsics, all_intrinsics_model,
        context_indices, target_indices,
        model_h, model_w,
        no_pixel_unshuffle=args.no_pixel_unshuffle,
    )
    raymap = raymap.to("cuda", dtype=torch.bfloat16)

    context_images = [Image.fromarray(all_images_model[i]) for i in range(num_input)]

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

    # Save generated
    gen_path = os.path.join(output_dir, f"generated_{target_file.replace('.jpg','.png').replace('.jpeg','.png')}")
    if not gen_path.endswith('.png'):
        gen_path = os.path.splitext(gen_path)[0] + '.png'
    Image.fromarray(pred_frame).save(gen_path)
    print(f"  Saved generated: {os.path.basename(gen_path)}")

    # Save comparison
    comp = make_comparison_frame(gt_frame, pred_frame)
    comp_path = os.path.join(output_dir, "comparison.png")
    Image.fromarray(comp).save(comp_path)

    psnr = compute_psnr(pred_frame, gt_frame)
    print(f"  PSNR={psnr:.2f} dB")

    return {"psnr": psnr}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)
    torch.manual_seed(42)

    data_dir = args.data_path
    examples = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))
                       and os.path.exists(os.path.join(data_dir, d, "extrinsics.npy"))])

    print(f"Custom Data Inference")
    print(f"  Data path: {data_dir}")
    print(f"  Examples: {examples}")
    print(f"  Model resolution: {args.height}x{args.width}")

    pipe = load_pipeline(args)

    os.makedirs(args.output_path, exist_ok=True)

    all_psnr = []
    for idx, example_name in enumerate(examples):
        example_dir = os.path.join(data_dir, example_name)
        result = process_example(pipe, args, example_dir, example_name, idx, len(examples))
        if result is not None:
            all_psnr.append(result["psnr"])

    print(f"\n{'=' * 80}")
    print(f"OVERALL ({len(all_psnr)} examples)")
    print(f"{'=' * 80}")
    if all_psnr:
        print(f"  Mean PSNR: {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB")

    summary_file = os.path.join(args.output_path, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Custom Data Inference — {len(all_psnr)} examples\n")
        f.write(f"Resolution: {args.height}x{args.width}\n\n")
        if all_psnr:
            f.write(f"Mean PSNR: {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB\n\n")
        for i, name in enumerate(examples):
            if i < len(all_psnr):
                f.write(f"  {name}: PSNR={all_psnr[i]:.2f}\n")

    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Inference on custom test data")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--new_in_dim", type=int, default=420)
    parser.add_argument("--data_path", type=str, default="/data2/qiwu2/test_data")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False)
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False)

    args = parser.parse_args()
    main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
