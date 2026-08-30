# Wan2.1-T2V-1.3B variant of the SE + zero-temporal-RoPE 6-to-1 NVS training run.
#
# Channel arithmetic for --new_in_dim 420 on the 1.3B (in_dim=16, has_image_input=False):
#   16 (latents) + 20 (y: 4 mask + 16 vae) + 384 (raymap, 6 ch * 8x8 unshuffle) = 420
#
# LoRA rank 512: the 14B run uses rank 32 on a model with dim=5120; the 1.3B has
# dim=1536 and ~10x fewer parameters, so rank 512 keeps the LoRA capacity
# proportional (and is the safer of the two ranks the user offered, vs 1024 which
# would be more prone to overfitting on the ~1K-scene DL3DV split).
accelerate launch model_training/train_SE_T2V_1_3B.py \
  --dataset_base_path ../../../DL3DV-10K_960P/1K \
  --dataset_metadata_path ../../../DL3DV-10K_960P/1K \
  --height 480 \
  --width 832 \
  --num_frames 10 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 5e-5 \
  --num_epochs 50 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-SE-T2V-1.3B-lora512-zero-temporal-rope-480p-T10" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 512 \
  --extra_inputs "input_image" \
  --modify_channels \
  --new_in_dim 420 \
  --gradient_accumulation_steps 2 \
  --initialize_model_on_cpu \
  --seperated_encoding \
  --zero_temporal_rope \
  --sampling_strategy "prob_random" \
  --wandb_project "wan-se-t2v-1.3b" \
  --wandb_run_name "lora512-zero-temporal-rope" \
  --resume_checkpoint "./models/train/Wan2.1-SE-T2V-1.3B-lora512-zero-temporal-rope-480p-3/epoch-49.safetensors" \
    # --num_input_frames 6 \
  # --num_output_frames None \
