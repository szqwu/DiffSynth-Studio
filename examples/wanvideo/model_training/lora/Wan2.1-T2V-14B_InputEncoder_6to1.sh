# Wan2.1-T2V-14B novel view synthesis, 6-to-1, input-encoder with
# TRANSFORM-THROUGH-LAYERS + a PREFIX ATTENTION MASK (no per-layer replacement).
#
# Difference vs. the 5B run (Wan2.2-TI2V-5B_InputEncoder_6to1.sh):
#   - The 6 clean conditioning frames now transform through all DiT layers like
#     normal tokens (--input_prefix_attention), instead of being re-injected
#     unchanged before every block. A prefix attention mask makes context tokens
#     attend ONLY to context, while the target attends to everything. With the
#     per-token timestep (context=0, target=t) this makes the context a clean,
#     step-invariant prefix -> at inference it can be transformed/cached ONCE.
#   - Frozen input VAE (no --trainable_input_vae) and NO residual MLP
#     (--no_input_latent_mlp): use the raw frozen VAE latents directly.
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
  --output_path "./models/train/Wan2.1-T2V-14B_lora32_InputEncoder_6to1_PrefixAttn" \
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
  --zero_temporal_rope \
  --raymap_downsample_factor 8 \
  --use_input_encoder \
  --input_prefix_attention \
  --no_input_latent_mlp \
  --num_input_frames 6 \
  --num_output_frames 1 \
