# Wan2.2-TI2V-5B novel view synthesis, 6-to-1, with a separate trainable input
# encoder read via TRUE per-layer cross-attention (per-layer token replacement).
#
# Key ideas (see plan "Clean input-encoder cross-conditioning"):
#   - The 6 conditioning frames are encoded ONCE by a separate VAE copy
#     (--trainable_input_vae) + a zero-init MLP residual, then re-injected
#     unchanged before every DiT block (--use_input_encoder). They are clean
#     (timestep=0), so the target frame reads them via self-attention as a fixed
#     cross-attention condition and a strong gradient flows into the encoder.
#   - --seperated_encoding: per-frame VAE encoding so each of the 7 frames gets
#     its own latent (the loss target for the last frame is a per-frame latent).
#   - --fuse_vae_embedding_in_latents_multiple: fill the 6 context latent slots
#     with the clean input-encoder latents and drive the per-token timestep
#     (context=0, target=t).
#   - --raymap_downsample_factor 16: match the Wan2.2 VAE /16 latent resolution
#     (6 Plucker channels * 16^2 = 1536); new_in_dim = 48 (latent) + 1536 = 1584.
#   - --zero_temporal_rope: keep the zero-temporal-RoPE behavior.
accelerate launch model_training/train_SE.py \
  --dataset_base_path /data2/qiwu2/DL3DV-10K_960P/1K \
  --dataset_metadata_path /data2/qiwu2/DL3DV-10K_960P/1K \
  --height 480 \
  --width 832 \
  --num_frames 7 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 160 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_lora128_InputEncoder_6to1_TrainableInputVAE" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 128 \
  --extra_inputs "input_image" \
  --modify_channels \
  --new_in_dim 1584 \
  --gradient_accumulation_steps 1 \
  --initialize_model_on_cpu \
  --seperated_encoding \
  --fuse_vae_embedding_in_latents_multiple \
  --zero_temporal_rope \
  --raymap_downsample_factor 16 \
  --use_input_encoder \
  --num_input_frames 6 \
  --num_output_frames 1 \
  --trainable_input_vae \
  --find_unused_parameters \
