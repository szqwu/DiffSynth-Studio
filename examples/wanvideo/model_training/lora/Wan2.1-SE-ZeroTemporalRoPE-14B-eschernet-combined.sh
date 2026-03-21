# Train on EscherNet combined data: BlendedMVS + RealEstate10K + SpatialVid (4:3:3)
# M-to-N generation: num_frames = num_input_frames + num_output_frames
# Fixed split:  --num_input_frames 6 --num_output_frames 1  (6-to-1)
# Random split: omit both --num_input_frames and --num_output_frames
#               each sample randomly picks M in [min_input, num_frames-min_output]
#               use --min_input_frames 3 --min_output_frames 1 to control bounds
accelerate launch model_training/train_SE.py \
  --dataset_type eschernet_combined \
  --combined_dataset_names blendedmvs realestate10k spatialvid \
  --combined_dataset_ratios 4 3 3 \
  --height 192 \
  --width 336 \
  --num_frames 7 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 160 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-SE-14B-lora32-zero-temporal-rope_eschernet-combined" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --modify_channels \
  --new_in_dim 420 \
  --gradient_accumulation_steps 2 \
  --initialize_model_on_cpu \
  --seperated_encoding \
  --zero_temporal_rope \
  --sampling_strategy "prob_random" \
  --wandb_project "wan-se-14b" \
  --wandb_run_name "lora32-zero-temporal-rope-eschernet-combined" \
  --num_input_frames 6 \
  --num_output_frames 1 \
  # --resume_checkpoint "./models/train/Wan2.1-SE-14B-lora32-zero-temporal-rope_eschernet-combined/epoch-29.safetensors" \
