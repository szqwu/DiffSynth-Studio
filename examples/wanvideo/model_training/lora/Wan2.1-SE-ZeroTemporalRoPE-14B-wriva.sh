# M-to-N generation: num_frames = num_input_frames + num_output_frames
# Fixed split:  --num_input_frames 6 --num_output_frames 1  (6-to-1)
# Random split: omit both --num_input_frames and --num_output_framesume
#               each sample randomly picks M in [min_input, num_frames-min_output]
#               use --min_input_frames 3 --min_output_frames 1 to control bounds
accelerate launch model_training/train_SE.py \
  --dataset_base_path ../../../DL3DV-10K_960P/1K \
  --dataset_metadata_path ../../../DL3DV-10K_960P/1K \
  --height 192 \
  --width 336 \
  --num_frames 7 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 5e-5 \
  --num_epochs 30 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-SE-14B-lora32-zero-temporal-rope_wriva" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --modify_channels \
  --new_in_dim 420 \
  --gradient_accumulation_steps 4 \
  --initialize_model_on_cpu \
  --seperated_encoding \
  --zero_temporal_rope \
  --sampling_strategy "prob_random" \
  --wandb_project "wan-se-14b" \
  --wandb_run_name "lora32-zero-temporal-rope-wriva" \
  --num_input_frames 6 \
  --num_output_frames 1 \
  # --resume_checkpoint "./models/train/Wan2.1-SE-14B-lora32-zero-temporal-rope_480p_T10/epoch-29.safetensors" \
  # --num_output_frames None \
