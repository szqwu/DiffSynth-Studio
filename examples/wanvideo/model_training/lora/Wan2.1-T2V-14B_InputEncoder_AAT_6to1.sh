# Wan2.1-T2V-14B novel view synthesis, 6-to-1, input-encoder with AAT-style
# ALTERNATING ATTENTION (instead of the prefix attention mask).
#
# This is the AAT sibling of Wan2.1-T2V-14B_InputEncoder_6to1.sh. Same input
# encoder (frozen VAE, no residual MLP, raymap concat, 6->1, LoRA rank 32), but
# the attention structure over the 7-frame token grid is AAT's, not a prefix mask:
#   - Even-indexed DiT blocks (0, 2, ...): within-frame self-attention with 2D xy
#     RoPE -> each of the 6 input frames and the target frame attend only to
#     themselves.
#   - Odd-indexed  DiT blocks (1, 3, ...): full 3D global attention with NO RoPE ->
#     the target frame reads all input frames (and vice versa).
# The input encoder already lays the 6 conditioning frames out as real frames in
# the (f, h, w) grid, so AAT's per-frame / global alternation is a natural fit.
#
# WHY NOT just add --aat_frame_attention to the prefix script: the two are
# mutually exclusive.
#   1. Mechanically, AAT's even (within-frame) blocks reshape tokens per-frame and
#      REJECT any attn_mask (axis='frame' raises on attn_mask), so AAT + a prefix
#      mask crashes on block 0.
#   2. Conceptually, AAT's odd blocks are fully global, so context attends to the
#      target -> the context is NOT a step-invariant prefix, and the transform-once
#      inference cache does not apply (it is gated off whenever AAT is on). AAT
#      instead uses per-layer token replacement (the default when prefix is off):
#      the clean input tokens are re-injected before every block.
#
# Notes vs the prefix script:
#   - --input_prefix_attention removed  (=> per_layer_input_replacement is on).
#   - --aat_frame_attention added.
#   - --zero_temporal_rope removed: AAT is self-contained and ignores it (it only
#     affects the legacy `freqs` consumed by VAP / vace, neither used here), so
#     passing it would be misleading.
#   - Everything else (frozen VAE, --no_input_latent_mlp, raymap /8 -> new_in_dim
#     400, 192x336, 7 frames, LoRA rank 32) is unchanged.
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
  --output_path "./models/train/Wan2.1-T2V-14B_lora32_InputEncoder_6to1_AAT" \
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
  --no_input_latent_mlp \
  --num_input_frames 6 \
  --num_output_frames 1 \
