accelerate launch --config_file my_config.yaml model_training/train_SE.py \
  --dataset_base_path /data2/qiwu2/DL3DV-10K_960P/1K \
  --dataset_metadata_path /data2/qiwu2/DL3DV-10K_960P/1K \
  --height 480 \
  --width 832 \
  --num_frames 6 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 40 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_full_SE" \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --modify_channels \
  --new_in_dim 1584 \
  --gradient_accumulation_steps 1 \
  --initialize_model_on_cpu \
  --save_steps 1250 \

