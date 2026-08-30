# Wan2.1-T2V-14B novel view synthesis, 6-to-1, input-encoder with BOTH AAT-style
# alternating attention AND the prefix attention mask (mask applied on the GLOBAL
# layers only).
#
# This is the combined sibling of the two input-encoder variants:
#   - Wan2.1-T2V-14B_InputEncoder_6to1.sh       (prefix mask, no AAT)
#   - Wan2.1-T2V-14B_InputEncoder_AAT_6to1.sh   (AAT, per-layer replacement)
#
# Attention structure over the 7-frame token grid (6 input + 1 target):
#   - Even-indexed DiT blocks (0, 2, ...): within-frame self-attention with 2D xy
#     RoPE. Each frame attends only to itself, so context frames cannot see the
#     target regardless of the denoising step -- NO mask is needed (and axis='frame'
#     does not support one).
#   - Odd-indexed  DiT blocks (1, 3, ...): full 3D global attention with NO RoPE,
#     BUT with the prefix attention mask -> context tokens attend only to context
#     across frames; the target attends to everything.
#
# Why this is the "best of both": AAT gives the per-frame / global inductive bias,
# while masking the global passes keeps the context a clean, STEP-INVARIANT prefix
# all the way through the network (even blocks are already step-invariant; masked
# odd blocks stay step-invariant too). This is enabled purely by passing BOTH
# --aat_frame_attention and --input_prefix_attention; model_fn_wan_video routes the
# mask to the global blocks and passes None to the within-frame blocks.
#
# Notes:
#   - --zero_temporal_rope is intentionally NOT passed: AAT uses its own per-block
#     freqs (2D xy on even blocks, none on odd), so zero_temporal_rope would be a
#     no-op here and only invites confusion.
#   - The transform-once inference cache IS supported with AAT (pass --transform_once
#     at eval): context K/V are cached at the global (odd) blocks and even blocks run
#     target-only within-frame attention on decode. Verified bit-identical to full
#     recompute in _smoke_test_input_encoder_prefix.py (test [7]).
#   - Frozen input VAE (no --trainable_input_vae) and NO residual MLP
#     (--no_input_latent_mlp): raw frozen VAE latents used directly.
#   - Wan2.1 VAE is 16-ch / 8x spatial, so --raymap_downsample_factor 8
#     (6 Plucker channels * 8^2 = 384); new_in_dim = 16 (latent) + 384 = 400.
#   - Resolution 192x336 (VAE /8 -> 24x42; patch 1x2x2 -> 12x21 = 252 tokens/frame;
#     7 frames -> 1764 tokens). LoRA rank 32.
#
# NOTE: the 14B bf16 trunk is ~28 GB; even with LoRA + gradient checkpointing this
# needs >=48 GB GPUs (or FSDP / parameter offload).
accelerate launch model_training/train_SE.py \
  --dataset_base_path /data2/qiwu2/DL3DV-10K_960P/1K \
  --dataset_metadata_path /data2/qiwu2/DL3DV-10K_960P/1K \
  --height 192 \
  --width 336 \
  --num_frames 7 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 160 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-14B_lora32_InputEncoder_6to1_AAT_PrefixAttn" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --modify_channels \
  --new_in_dim 400 \
  --gradient_accumulation_steps 1 \
  --use_gradient_checkpointing \
  --initialize_model_on_cpu \
  --seperated_encoding \
  --fuse_vae_embedding_in_latents_multiple \
  --raymap_downsample_factor 8 \
  --use_input_encoder \
  --aat_frame_attention \
  --input_prefix_attention \
  --no_input_latent_mlp \
  --num_input_frames 6 \
  --num_output_frames 1 \
