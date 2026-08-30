"""
Wan2.1-T2V-1.3B M-to-N NVS evaluation on DL3DV-10K scenes.

Thin wrapper around test_wan2.1_6to1.py that swaps the pretrained backbone from
Wan2.1-I2V-14B-480P to Wan2.1-T2V-1.3B (no CLIP image encoder). All argument
names, data loading, raymap construction, and metrics computation are identical
to the 14B test script — only `load_pipeline` is overridden.

Channel arithmetic is unchanged (--new_in_dim 420):
    16 (latents)
  + 20 (y from VAE-encoded context: 4 mask + 16 vae)
  + 384 (raymap, 6 ch * 8x8 pixel-unshuffle)
  = 420 channels into the modified patch_embedding.

The base test file's name has a dot ("test_wan2.1_6to1.py"), so we load it via
importlib.util rather than `import`.
"""
import argparse
import importlib.util
import os
import random
import time

import numpy as np
import torch

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


# ── Load the 14B test module as `base` (filename has a dot, so use importlib) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_FILE = os.path.join(_HERE, "test_wan2.1_6to1.py")
_spec = importlib.util.spec_from_file_location("_test_wan21_6to1_base", _BASE_FILE)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def load_pipeline(args):
    """
    Load the Wan2.1-T2V-1.3B pipeline, modify channels for SE+raymap, and load
    the trained checkpoint. Mirrors base.load_pipeline but:
      - uses Wan-AI/Wan2.1-T2V-1.3B (4 -> 3 model files; no CLIP)
      - drops the explicit vram_limit (1.3B fits comfortably on a single GPU)
    """
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )

    base.modify_model_channels(pipe, "dit", args.new_in_dim, "cuda")

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


# Monkey-patch the base module so its main() picks up the 1.3B loader.
base.load_pipeline = load_pipeline


def build_parser():
    """Mirror of the argparse block in test_wan2.1_6to1.py's __main__."""
    parser = argparse.ArgumentParser(
        description="Wan2.1-T2V-1.3B M-to-N NVS evaluation on DL3DV-10K scenes"
    )

    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained checkpoint (.safetensors)")
    parser.add_argument("--new_in_dim", type=int, default=420,
                        help="New input dimension for the modified model (must match training --new_in_dim)")
    parser.add_argument("--num_input_frames", type=int, default=6,
                        help="Number of input (context) frames M. Default: 6.")
    parser.add_argument("--num_output_frames", type=int, default=1,
                        help="Number of output (target) frames N per inference call. Default: 1.")

    # Data paths
    parser.add_argument("--dl3dv_meta_path", type=str, default="/data2/qiwu2/dl3dv10")
    parser.add_argument("--dl3dv_data_path", type=str, default="/data2/qiwu2/DL3DV-10K-test")
    parser.add_argument("--output_path", type=str, required=True)

    # Scenes
    parser.add_argument("--scenes", type=str, nargs="+", required=True)

    # Resolution
    parser.add_argument("--height", type=int, default=192,
                        help="Model input height. The training run uses 192.")
    parser.add_argument("--width", type=int, default=336,
                        help="Model input width. The training run uses 336.")
    parser.add_argument("--eval_size", type=int, default=576,
                        help="Evaluation size for center-crop metrics (default: 576).")
    parser.add_argument("--input_mode", type=str, default="crop", choices=["stretch", "crop"])

    # Inference settings
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--use_prope", action="store_true", default=False)
    parser.add_argument("--zero_temporal_rope", action="store_true", default=False)
    parser.add_argument("--zero_xy_rope", action="store_true", default=False)
    parser.add_argument("--aat_frame_attention", action="store_true", default=False)
    parser.add_argument("--no_pixel_unshuffle", action="store_true", default=False)

    # Metrics
    parser.add_argument("--use_dreamsim", action="store_true")
    parser.add_argument("--use_ssim", action="store_true")
    parser.add_argument("--use_lpips", action="store_true")

    # Timing
    parser.add_argument("--time_inference", action="store_true")

    return parser


if __name__ == "__main__":
    start_time = time.time()
    args = build_parser().parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    base.main(args)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60.0:.1f} minutes")
    print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
