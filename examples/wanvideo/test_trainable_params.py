#!/usr/bin/env python
"""
Inspect trainable / frozen parameters of the Wan2.1-SE-14B LoRA32 training setup.

Replicates the exact training configuration from Wan2.1-SE-14B.sh:
  - Wan2.1-I2V-14B-480P base model
  - DiT channels modified (in_dim -> 420) with separated encoding
  - LoRA rank=32 on q,k,v,o,ffn.0,ffn.2
  - patch_embedding unfrozen for full training

Components:
  - DiT (WanModel):   LoRA adapters + patch_embedding trainable, rest frozen
  - VAE:              frozen
  - Text Encoder (T5): frozen
  - Image Encoder (CLIP): frozen
"""

import sys
import os
import torch
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from peft import LoraConfig, inject_adapter_in_model
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import WanModel


def fmt(n):
    if n >= 1e9:
        return f"{n:>14,d}  ({n/1e9:.2f}B)"
    return f"{n:>14,d}  ({n/1e6:.2f}M)"


def analyse_model(name, model, role_desc):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - train
    dtype = next(model.parameters()).dtype if total > 0 else "N/A"
    size_mb = sum(p.numel() * p.element_size()
                  for p in model.parameters()) / (1024 ** 2)

    print(f"\n{'─' * 70}")
    print(f"  {name}")
    print(f"{'─' * 70}")
    print(f"  Role in training:  {role_desc}")
    print(f"  Total params:      {fmt(total)}")
    print(f"  Trainable params:  {fmt(train)}")
    print(f"  Frozen params:     {fmt(frozen)}")
    print(f"  Model size:        {size_mb:.1f} MB  (dtype={dtype})")

    children = list(model.named_children())
    if children:
        print(f"\n  {'Submodule':<40s} {'Params':>14s}  {'Trainable':>14s}")
        print(f"  {'─'*40}  {'─'*14}  {'─'*14}")
        for child_name, child in children:
            cp = sum(p.numel() for p in child.parameters())
            ct = sum(p.numel() for p in child.parameters() if p.requires_grad)
            if ct == 0:
                flag = "no"
            elif ct == cp:
                flag = "all"
            else:
                flag = f"{ct:,d}"
            print(f"  {child_name:<40s} {cp:>14,d}  {flag:>14s}")

    return total, train, frozen, size_mb


def modify_model_channels(old_model, new_in_dim):
    """Recreate model with modified input channels (same as train_SE.py)."""
    new_model = WanModel(
        dim=old_model.dim,
        in_dim=new_in_dim,
        ffn_dim=old_model.ffn_dim,
        out_dim=old_model.out_dim,
        text_dim=old_model.text_embedding[0].in_features,
        freq_dim=old_model.freq_dim,
        eps=1e-6,
        patch_size=old_model.patch_size,
        num_heads=old_model.num_heads,
        num_layers=old_model.num_layers,
        has_image_input=old_model.has_image_input,
        has_image_pos_emb=old_model.has_image_pos_emb,
        has_ref_conv=old_model.has_ref_conv,
        add_control_adapter=old_model.control_adapter is not None,
        in_dim_control_adapter=24 if old_model.control_adapter is not None else 24,
        seperated_timestep=old_model.seperated_timestep,
        require_vae_embedding=old_model.require_vae_embedding,
        require_clip_embedding=old_model.require_clip_embedding,
        fuse_vae_embedding_in_latents=old_model.fuse_vae_embedding_in_latents,
        fuse_vae_embedding_in_latents_multiple=False,
        seperated_encoding=True,
    )

    pretrained_sd = old_model.state_dict()
    new_sd = new_model.state_dict()
    for key, value in pretrained_sd.items():
        if key.startswith("patch_embedding"):
            continue
        if key in new_sd and value.shape == new_sd[key].shape:
            new_sd[key] = value
    new_model.load_state_dict(new_sd, strict=False)
    return new_model.to(dtype=torch.bfloat16)


def main():
    print("=" * 70)
    print("  Wan2.1-SE-14B LoRA32 Parameter Analysis")
    print("=" * 70)
    print("  Config: Wan2.1-I2V-14B-480P + LoRA32 + modified patchify (in_dim=420)")
    print("  LoRA targets: q, k, v, o, ffn.0, ffn.2")

    # ── Load pipeline (all on CPU to avoid GPU memory) ───────────────────
    print("\nLoading pipeline (this may take a while)...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                        origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                        origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/"),
        vram_limit=None,
    )
    print("Pipeline loaded.")

    # ── Step 1: Modify DiT channels (in_dim -> 420) ─────────────────────
    print("\nModifying DiT input channels: in_dim -> 420...")
    old_in_dim = pipe.dit.in_dim
    pipe.dit = modify_model_channels(pipe.dit, new_in_dim=420)
    print(f"  DiT in_dim: {old_in_dim} -> {pipe.dit.in_dim}")

    # ── Step 2: Freeze everything ────────────────────────────────────────
    print("Freezing entire pipeline...")
    pipe.eval()
    pipe.requires_grad_(False)

    # ── Step 3: Apply LoRA to DiT ────────────────────────────────────────
    lora_target_modules = ["q", "k", "v", "o", "ffn.0", "ffn.2"]
    lora_rank = 32
    lora_alpha = lora_rank
    print(f"Injecting LoRA (rank={lora_rank}) on: {lora_target_modules}")

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha,
        target_modules=lora_target_modules)
    pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)

    # Upcast LoRA params to training dtype
    for param in pipe.dit.parameters():
        if param.requires_grad:
            param.data = param.to(torch.bfloat16)

    # ── Step 4: Unfreeze patch_embedding ─────────────────────────────────
    print("Unfreezing patch_embedding for full training...")
    for param in pipe.dit.patch_embedding.parameters():
        param.requires_grad = True

    # ── Analyse each component ───────────────────────────────────────────
    stats = OrderedDict()

    stats["DiT (WanModel 14B)"] = analyse_model(
        "DiT (WanModel 14B + LoRA32 + patch_embedding)",
        pipe.dit,
        "LoRA adapters + patch_embedding TRAINABLE, base weights FROZEN")

    stats["VAE (WanVideoVAE)"] = analyse_model(
        "VAE (WanVideoVAE)", pipe.vae, "FROZEN")

    stats["Text Encoder (T5-XXL)"] = analyse_model(
        "Text Encoder (T5-XXL / umt5-xxl-enc)",
        pipe.text_encoder, "FROZEN")

    stats["Image Encoder (CLIP)"] = analyse_model(
        "Image Encoder (CLIP / open-clip-xlm-roberta-large-vit-huge-14)",
        pipe.image_encoder, "FROZEN")

    # ── DiT LoRA detail breakdown ────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  DiT LoRA Adapter Breakdown")
    print(f"{'─' * 70}")

    lora_params = 0
    patch_emb_params = 0
    base_frozen_params = 0

    for name, param in pipe.dit.named_parameters():
        if "lora_" in name:
            lora_params += param.numel()
        elif "patch_embedding" in name:
            patch_emb_params += param.numel()
        else:
            base_frozen_params += param.numel()

    print(f"  LoRA adapter params:      {fmt(lora_params)}")
    print(f"  patch_embedding params:   {fmt(patch_emb_params)}")
    print(f"  Base frozen params:       {fmt(base_frozen_params)}")

    # Count LoRA modules
    lora_a_count = sum(1 for n, _ in pipe.dit.named_parameters() if "lora_A" in n)
    print(f"  Number of LoRA pairs:     {lora_a_count}")

    # ── Grand totals ─────────────────────────────────────────────────────
    grand_total = sum(s[0] for s in stats.values())
    grand_train = sum(s[1] for s in stats.values())
    grand_frozen = sum(s[2] for s in stats.values())
    grand_size = sum(s[3] for s in stats.values())

    print(f"\n{'═' * 70}")
    print(f"  GRAND TOTALS")
    print(f"{'═' * 70}")
    print(f"  Total params:      {fmt(grand_total)}")
    print(f"  Trainable params:  {fmt(grand_train)}")
    print(f"  Frozen params:     {fmt(grand_frozen)}")
    if grand_total > 0:
        print(f"  Trainable ratio:   {grand_train/grand_total*100:.2f}%")
    print(f"  Total model size:  {grand_size:.1f} MB")
    print(f"{'═' * 70}")

    # ── Optimizer summary ────────────────────────────────────────────────
    print(f"\n  Optimizer trainable params (what gets updated):")
    print(f"    LoRA adapters (rank 32):  {fmt(lora_params)}")
    print(f"    patch_embedding (full):   {fmt(patch_emb_params)}")
    print(f"    TOTAL trainable:          {fmt(lora_params + patch_emb_params)}")
    print()


if __name__ == "__main__":
    main()
