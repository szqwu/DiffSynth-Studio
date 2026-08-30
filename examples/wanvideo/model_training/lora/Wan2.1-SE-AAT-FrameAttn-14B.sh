# AAT-style alternating attention (no new parameters):
#   - Even-indexed DiT blocks (0, 2, 4, ..., 38): within-frame self-attention only,
#     with 2D RoPE on (h, w). Each frame attends only to itself.
#   - Odd-indexed DiT blocks (1, 3, 5, ..., 39): full 3D global attention with NO RoPE
#     (frames AND spatial positions are permutation-equivariant in the global pass).
# This matches the MapAnything/VGGT AAT recipe. The flag is self-contained: it does
# NOT need --zero_temporal_rope or --zero_xy_rope (which still control the legacy
# `freqs` consumed by VAP / vace, with their original semantics).
# All weights load from the original pretrained checkpoint (strict=True compatible).
accelerate launch --config_file my_config.yaml model_training/train_SE.py \
  --dataset_base_path ../../../DL3DV-10K_960P/1K \
  --dataset_metadata_path ../../../DL3DV-10K_960P/1K \
  --height 192 \
  --width 336 \
  --num_frames 7 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 97 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-SE-14B-lora32-aat-frame-attn2" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --modify_channels \
  --new_in_dim 420 \
  --gradient_accumulation_steps 1 \
  --initialize_model_on_cpu \
  --seperated_encoding \
  --aat_frame_attention \
  --sampling_strategy "prob_random" \
  --wandb_project "wan-se-14b" \
  --wandb_run_name "lora32-aat-frame-attn" \
  --num_input_frames 6 \
  --resume_checkpoint "./models/train/Wan2.1-SE-14B-lora32-aat-frame-attn/epoch-62.safetensors" \
