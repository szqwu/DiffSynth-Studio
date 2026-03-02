import torch, os, argparse, accelerate, warnings
from diffsynth.core.data.my_v2v_dataset_images_in_plucker_SE import my_cognvs_dataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        modify_channels=False,
        new_in_dim=None,
        no_SE=False,
        resume_checkpoint=None,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        self.no_SE = no_SE

        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)

        # no-SE: standard VAE encoding (not separated), with channel modification
        self.seperated_encoding = False
        self.fuse_vae_embedding_in_latents_multiple = False

        if modify_channels and new_in_dim is not None:
            self.modify_model_channels(self.pipe.dit, new_in_dim, device)
            if self.pipe.dit2 is not None:
                self.modify_model_channels(self.pipe.dit2, new_in_dim, device)

        if no_SE:
            self.pipe.dit.no_SE = True
            if self.pipe.dit2 is not None:
                self.pipe.dit2.no_SE = True

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        effective_lora_checkpoint = resume_checkpoint if resume_checkpoint is not None else lora_checkpoint

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, effective_lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        if modify_channels and new_in_dim is not None and lora_base_model is not None:
            self.unfreeze_patch_embedding(self.pipe, lora_base_model)

        if resume_checkpoint is not None and lora_base_model is not None:
            from diffsynth.core import load_state_dict as _load_sd
            ckpt_sd = _load_sd(resume_checkpoint)
            extra_state = {k: v for k, v in ckpt_sd.items()
                          if "patch_embedding" in k}
            if extra_state:
                load_result = getattr(self.pipe, lora_base_model).load_state_dict(extra_state, strict=False)
                print(f"Resume: loaded {len(extra_state)} extra keys (patch_embedding) from {resume_checkpoint}")
            else:
                print(f"Resume warning: no patch_embedding keys found in {resume_checkpoint}")
            del ckpt_sd

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def modify_model_channels(self, model, new_in_dim, device):
        if model is None:
            return

        old_model = model
        old_in_dim = model.in_dim
        old_out_dim = model.out_dim
        new_out_dim = old_out_dim

        from diffsynth.models.wan_video_dit import WanModel

        new_model = WanModel(
            dim=model.dim,
            in_dim=new_in_dim,
            ffn_dim=model.ffn_dim,
            out_dim=new_out_dim,
            text_dim=model.text_embedding[0].in_features,
            freq_dim=model.freq_dim,
            eps=1e-6,
            patch_size=model.patch_size,
            num_heads=model.num_heads,
            num_layers=model.num_layers,
            has_image_input=model.has_image_input,
            has_image_pos_emb=model.has_image_pos_emb,
            has_ref_conv=model.has_ref_conv,
            seperated_timestep=model.seperated_timestep,
            require_vae_embedding=model.require_vae_embedding,
            require_clip_embedding=model.require_clip_embedding,
            fuse_vae_embedding_in_latents=model.fuse_vae_embedding_in_latents,
            fuse_vae_embedding_in_latents_multiple=self.fuse_vae_embedding_in_latents_multiple,
            seperated_encoding=self.seperated_encoding,
        )

        pretrained_state_dict = old_model.state_dict()
        new_state_dict = new_model.state_dict()

        for key, value in pretrained_state_dict.items():
            if key.startswith("patch_embedding"):
                print(f"Skipping {key} due to in_dim change (old_in_dim={old_in_dim}, new_in_dim={new_in_dim})")
                continue
            if key in new_state_dict and value.shape == new_state_dict[key].shape:
                new_state_dict[key] = value
            else:
                print(f"Skipping {key} due to shape mismatch or absence in new model")

        new_model.load_state_dict(new_state_dict, strict=False)
        new_model = new_model.to(device=device, dtype=torch.bfloat16)

        if hasattr(self.pipe, 'dit') and self.pipe.dit is old_model:
            self.pipe.dit = new_model
        if hasattr(self.pipe, 'dit2') and self.pipe.dit2 is old_model:
            self.pipe.dit2 = new_model

        print(f"Model input channels modified: in_dim {old_in_dim}->{new_in_dim} (out_dim unchanged: {old_out_dim})")

    def unfreeze_patch_embedding(self, pipe, lora_base_model):
        model = getattr(pipe, lora_base_model, None)
        if model is None:
            return
        if hasattr(model, 'patch_embedding'):
            for param in model.patch_embedding.parameters():
                param.requires_grad = True
            print(f"Unfroze patch_embedding layer in {lora_base_model} for full training")
        if lora_base_model == "dit" and hasattr(pipe, 'dit2') and pipe.dit2 is not None:
            if hasattr(pipe.dit2, 'patch_embedding'):
                for param in pipe.dit2.patch_embedding.parameters():
                    param.requires_grad = True
                print(f"Unfroze patch_embedding layer in dit2 for full training")

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["input_images"]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def _group_raymap_for_tokens(self, raymap):
        """
        Group per-frame raymap [7, C, H, W] into per-token raymap [3, 4*C, H, W]
        after VAE temporal compression of 9 zero-padded frames into 3 tokens.
        Token 1: frame 1 (+ 3 zero padding)
        Token 2: frames 2-5
        Token 3: frames 6-7 (+ 2 zero padding)
        """
        T, C, h, w = raymap.shape
        zeros = torch.zeros(1, C, h, w, dtype=raymap.dtype, device=raymap.device)

        # Token 1: frame 1 + 3 zero cameras
        t1 = torch.cat([raymap[0:1], zeros, zeros, zeros], dim=0).reshape(4 * C, h, w)
        # Token 2: frames 2-5
        t2 = raymap[1:5].reshape(4 * C, h, w)
        # Token 3: frame 6 + frame 7 (target camera) + 2 zero cameras
        t3 = torch.cat([raymap[5:7], zeros, zeros], dim=0).reshape(4 * C, h, w)

        return torch.stack([t1, t2, t3], dim=0)  # [3, 4*C, H, W]

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        height = data["input_images"][0].size[1]
        width = data["input_images"][0].size[0]

        if self.no_SE:
            # Pad target_images from 7 to 9 with black frames
            black_img = Image.new('RGB', (width, height), (0, 0, 0))
            padded_target = list(data["target_images"]) + [black_img, black_img]

            # Group raymap from per-frame [7, 384, h, w] to per-token [3, 1536, h, w]
            grouped_raymap = self._group_raymap_for_tokens(data["raymap"])

            inputs_shared = {
                "input_image": data["input_images"],
                "input_video": padded_target,
                "raymap": grouped_raymap,
                "height": height,
                "width": width,
                "num_frames": 9,
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
            }
        else:
            inputs_shared = {
                "input_image": data["input_images"],
                "input_video": data["target_images"],
                "raymap": data["raymap"],
                "height": height,
                "width": width,
                "num_frames": len(data["target_images"]),
                "num_latent_frames": len(data["target_images"]),
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
            }

        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Training script for no-SE ablation.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--modify_channels", default=False, action="store_true")
    parser.add_argument("--new_in_dim", type=int, default=None)
    parser.add_argument("--no_SE", default=False, action="store_true",
                        help="No Separated Encoding ablation: encode frames jointly through VAE "
                             "temporal compression instead of separately. Pads 7 frames to 9, "
                             "resulting in 3 latent tokens. Raymap channels become 384*4=1536 per "
                             "token (4 cameras concatenated). new_in_dim should be 1572.")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--sampling_strategy", type=str, default="prob_random",
                        choices=["all_random", "prob_random", "all_window", "curriculum"])
    parser.add_argument("--num_dataset_samples", type=int, default=1000)
    parser.add_argument("--no_pixel_unshuffle", default=False, action="store_true")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
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
        num_dataset_samples=args.num_dataset_samples,
        no_pixel_unshuffle=args.no_pixel_unshuffle,
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
        no_SE=args.no_SE,
        resume_checkpoint=args.resume_checkpoint,
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
