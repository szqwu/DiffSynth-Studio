# M-to-N generation with zero-XY (spatial) RoPE:
# Spatial H/W rope components are replaced with identity so all tokens
# within a frame share the same spatial position. Temporal RoPE is kept.
accelerate launch --config_file my_config.yaml model_training/train_SE.py \
  --dataset_base_path ../../../DL3DV-10K_960P/1K \
  --dataset_metadata_path ../../../DL3DV-10K_960P/1K \
  --height 192 \
  --width 336 \
  --num_frames 7 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 160 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-SE-14B-lora32-zero-xy-rope" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --modify_channels \
  --new_in_dim 420 \
  --gradient_accumulation_steps 1 \
  --initialize_model_on_cpu \
  --seperated_encoding \
  --zero_temporal_rope \
  --zero_xy_rope \
  --sampling_strategy "prob_random" \
  --wandb_project "wan-se-14b" \
  --wandb_run_name "lora32-zero-xy-rope" \
  --num_input_frames 6 \
