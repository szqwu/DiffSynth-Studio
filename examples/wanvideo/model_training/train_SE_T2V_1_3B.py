"""
Training script for Wan2.1-T2V-1.3B with Separated Encoding (SE),
raymap-based camera conditioning, and (optionally) zero-temporal-RoPE.

Differences vs. train_SE.py (which targets Wan2.1-I2V-14B-480P):
  - Base model is Wan2.1-T2V-1.3B (text-to-video, no CLIP image encoder).
      * has_image_input = False, dim = 1536, num_layers = 30, in_dim = 16
      * No image_encoder is loaded.
  - Larger LoRA rank is recommended (the 1.3B has ~10x fewer parameters
    than the 14B, so rank ~16x of the 14B run keeps the LoRA capacity
    proportionally similar).

Channel arithmetic for --new_in_dim 420 (same as the 14B SE run):
    16 (latents)
  + 20 (y from VAE-encoded context frames: 4 mask + 16 vae)
  + 384 (raymap with 8x8 pixel-unshuffle: 6 channels * 64)
  = 420 channels into the modified patch_embedding.

The SE pipeline unit (WanVideoUnit_ImageEmbedderVAE_SeparatedEncoding) fires
when input_image is provided and pipe.dit.{require_vae_embedding,
seperated_encoding} are both True, regardless of has_image_input. The default
WanModel ctor sets require_vae_embedding=True for T2V-1.3B as well.

This file reuses WanTrainingModule from train_SE.py to avoid duplicating
the (~250-line) SE-specific training-module logic.
"""
import os
import accelerate

from diffsynth.core.data.my_v2v_dataset_images_in_plucker_SE import my_cognvs_dataset
from diffsynth.diffusion import (
    ModelLogger,
    launch_data_process_task,
    launch_training_task,
)

from train_SE import WanTrainingModule, wan_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )

    dataset = my_cognvs_dataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        height_division_factor=8,
        width_division_factor=8,
        time_division_factor=4,
        time_division_remainder=1,
        sampling_strategy=args.sampling_strategy,
        reverse_pred_order=args.reverse_pred_order,
        num_dataset_samples=args.num_dataset_samples,
        no_pixel_unshuffle=args.no_pixel_unshuffle,
        num_input_frames=args.num_input_frames,
        num_output_frames=args.num_output_frames,
        min_input_frames=args.min_input_frames,
        min_output_frames=args.min_output_frames,
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        modify_channels=args.modify_channels,
        new_in_dim=args.new_in_dim,
        seperated_encoding=args.seperated_encoding,
        fuse_vae_embedding_in_latents_multiple=args.fuse_vae_embedding_in_latents_multiple,
        resume_checkpoint=args.resume_checkpoint,
        use_camera_adapter=args.use_camera_adapter,
        reverse_pred_order=args.reverse_pred_order,
        use_prope=args.use_prope,
        zero_temporal_rope=args.zero_temporal_rope,
        zero_xy_rope=args.zero_xy_rope,
        aat_frame_attention=args.aat_frame_attention,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
