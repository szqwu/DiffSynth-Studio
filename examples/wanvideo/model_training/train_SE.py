import torch, os, argparse, accelerate, warnings
# from diffsynth.core import UnifiedDataset
from diffsynth.core.data.my_v2v_dataset_images_in_plucker_SE import my_cognvs_dataset
from diffsynth.core.data.eschernet_combined_dataset import EscherNetCombinedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
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
        seperated_encoding=False,
        fuse_vae_embedding_in_latents_multiple=False,
        resume_checkpoint=None,
        use_camera_adapter=False,
        reverse_pred_order=False,
        use_prope=False,
        zero_temporal_rope=False,
        use_rayrope=False,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        self.use_camera_adapter = use_camera_adapter
        self.reverse_pred_order = reverse_pred_order
        self.use_prope = use_prope
        self.zero_temporal_rope = zero_temporal_rope
        self.use_rayrope = use_rayrope
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        
        # Store these before modify_model_channels which needs them
        self.seperated_encoding = seperated_encoding
        self.fuse_vae_embedding_in_latents_multiple = fuse_vae_embedding_in_latents_multiple
        
        # Modify channels if requested (similar to CogVideoX approach)
        if modify_channels and new_in_dim is not None:
            self.modify_model_channels(self.pipe.dit, new_in_dim, device)
            if self.pipe.dit2 is not None:
                self.modify_model_channels(self.pipe.dit2, new_in_dim, device)
        
        # Camera adapter mode: use the pretrained SimpleAdapter from the checkpoint.
        # If the checkpoint doesn't have one, create a new (randomly initialized) adapter.
        # Must happen BEFORE split_pipeline_units so the adapter is on the model.
        if use_camera_adapter:
            if self.pipe.dit.control_adapter is not None:
                print("Using pretrained control_adapter from checkpoint (not re-initializing)")
            else:
                self.add_camera_adapter(self.pipe.dit, device)
            if self.pipe.dit2 is not None:
                if self.pipe.dit2.control_adapter is not None:
                    print("Using pretrained control_adapter from dit2 checkpoint")
                else:
                    self.add_camera_adapter(self.pipe.dit2, device)
            # Set seperated_encoding flag on the pretrained model
            # (not done by modify_model_channels since we don't modify channels)
            if seperated_encoding:
                self.pipe.dit.seperated_encoding = True
                if self.pipe.dit2 is not None:
                    self.pipe.dit2.seperated_encoding = True
        
        # Set reverse_pred_order flag on dit model (independent of camera adapter mode)
        if reverse_pred_order:
            self.pipe.dit.reverse_pred_order = True
            if self.pipe.dit2 is not None:
                self.pipe.dit2.reverse_pred_order = True

        # RayRoPE: add d_proj layers to each SelfAttention block
        if use_rayrope:
            self._add_rayrope_layers(self.pipe.dit, device)
            if self.pipe.dit2 is not None:
                self._add_rayrope_layers(self.pipe.dit2, device)
        
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # If resume_checkpoint is given, use it as the lora_checkpoint source too
        effective_lora_checkpoint = resume_checkpoint if resume_checkpoint is not None else lora_checkpoint

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, effective_lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # If channels were modified, unfreeze patch_embedding for training
        if modify_channels and new_in_dim is not None and lora_base_model is not None:
            self.unfreeze_patch_embedding(self.pipe, lora_base_model)

        # If camera adapter mode, unfreeze the control_adapter for training
        # if use_camera_adapter and lora_base_model is not None:
        #     self.unfreeze_control_adapter(self.pipe, lora_base_model)

        # If camera adapter is frozen, offload it to CPU to save GPU memory.
        # The patchify method will move it to GPU on-the-fly for each forward pass.
        if use_camera_adapter and lora_base_model is not None:
            model = getattr(self.pipe, lora_base_model, None)
            if model is not None and hasattr(model, 'control_adapter') and model.control_adapter is not None:
                adapter_trainable = any(p.requires_grad for p in model.control_adapter.parameters())
                if not adapter_trainable:
                    model.control_adapter.to("cpu")
                    print(f"Offloaded frozen control_adapter to CPU to save GPU memory")

        # Resume: load patch_embedding / control_adapter weights from checkpoint
        # (mapping_lora_state_dict only keeps lora_A/lora_B keys, so these
        #  weights are dropped during the LoRA loading above — restore them here)
        if resume_checkpoint is not None and lora_base_model is not None:
            from diffsynth.core import load_state_dict as _load_sd
            ckpt_sd = _load_sd(resume_checkpoint)
            extra_state = {k: v for k, v in ckpt_sd.items()
                          if "patch_embedding" in k or "control_adapter" in k or "d_proj" in k}
            if extra_state:
                load_result = getattr(self.pipe, lora_base_model).load_state_dict(extra_state, strict=False)
                print(f"Resume: loaded {len(extra_state)} extra keys (patch_embedding/control_adapter/d_proj) from {resume_checkpoint}")
            else:
                print(f"Resume warning: no patch_embedding/control_adapter/d_proj keys found in {resume_checkpoint}")
            del ckpt_sd
        
        # Store other configs
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
        """
        Modify the model's input dimension, loading pretrained weights.
        Similar to the approach in CogVideoX separate encoding training.
        Only the input dimension is modified; output dimension remains unchanged.
        """
        if model is None:
            return
        
        # Store the old model to extract pretrained weights
        old_model = model
        old_in_dim = model.in_dim
        
        old_out_dim = model.out_dim
        new_out_dim = old_out_dim
        
        # Preserve control_adapter if the pretrained model has one
        has_adapter = model.control_adapter is not None
        
        # Get the old configuration
        from diffsynth.models.wan_video_dit import WanModel
        
        # Create new model with modified channels
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
            add_control_adapter=has_adapter,
            in_dim_control_adapter=24 if has_adapter else None,
            seperated_timestep=model.seperated_timestep,
            require_vae_embedding=model.require_vae_embedding,
            require_clip_embedding=model.require_clip_embedding,
            fuse_vae_embedding_in_latents=model.fuse_vae_embedding_in_latents,
            fuse_vae_embedding_in_latents_multiple = self.fuse_vae_embedding_in_latents_multiple,
            seperated_encoding=self.seperated_encoding,
            reverse_pred_order=self.reverse_pred_order,
        )
        
        # Load all pretrained weights EXCEPT layers with modified dimensions
        pretrained_state_dict = old_model.state_dict()
        new_state_dict = new_model.state_dict()
        
        # Copy weights that have matching shapes
        for key, value in pretrained_state_dict.items():
            # Skip patch_embedding since in_dim changed
            if key.startswith("patch_embedding"):
                print(f"Skipping {key} due to in_dim change (old_in_dim={old_in_dim}, new_in_dim={new_in_dim})")
                continue
            
            # Copy matching weights
            if key in new_state_dict and value.shape == new_state_dict[key].shape:
                new_state_dict[key] = value
            else:
                print(f"Skipping {key} due to shape mismatch or absence in new model")
        
        # Load the modified state dict
        new_model.load_state_dict(new_state_dict, strict=False)
        new_model = new_model.to(device=device, dtype=torch.bfloat16)
        
        # Replace the model in the pipeline
        if hasattr(self.pipe, 'dit') and self.pipe.dit is old_model:
            self.pipe.dit = new_model
        if hasattr(self.pipe, 'dit2') and self.pipe.dit2 is old_model:
            self.pipe.dit2 = new_model
        
        print(f"Model input channels modified: in_dim {old_in_dim}->{new_in_dim} (out_dim unchanged: {old_out_dim})")
    
    def add_camera_adapter(self, model, device):
        """
        Add a randomly-initialized SimpleAdapter (camera control adapter) to the model.
        The patchify layer is NOT modified — camera conditioning enters via additive
        injection after patch embedding, just like the Fun Camera Control pipeline.
        """
        if model is None:
            return
        from diffsynth.models.wan_video_camera_controller import SimpleAdapter
        # 24 = 6 Plucker channels × 4 temporal packing (SE repeat-4 convention)
        adapter = SimpleAdapter(
            in_dim=24,
            out_dim=model.dim,
            kernel_size=model.patch_size[1:],
            stride=model.patch_size[1:],
        )
        adapter = adapter.to(device=device, dtype=torch.bfloat16)
        model.control_adapter = adapter
        print(f"Added SimpleAdapter camera control adapter to model "
              f"(in_dim=24, out_dim={model.dim}, kernel={model.patch_size[1:]})")

    def unfreeze_patch_embedding(self, pipe, lora_base_model):
        """
        Unfreeze the patch_embedding layer for training since it was randomly initialized
        when in_dim was modified. This allows training the patch_embedding with full weights
        while using LoRA for other layers.
        """
        model = getattr(pipe, lora_base_model, None)
        if model is None:
            return
        
        # Unfreeze patch_embedding parameters
        if hasattr(model, 'patch_embedding'):
            for param in model.patch_embedding.parameters():
                param.requires_grad = True
            print(f"Unfroze patch_embedding layer in {lora_base_model} for full training")
        
        # If there's a second model (dit2), handle it too
        if lora_base_model == "dit" and hasattr(pipe, 'dit2') and pipe.dit2 is not None:
            if hasattr(pipe.dit2, 'patch_embedding'):
                for param in pipe.dit2.patch_embedding.parameters():
                    param.requires_grad = True
                print(f"Unfroze patch_embedding layer in dit2 for full training")

    def unfreeze_control_adapter(self, pipe, lora_base_model):
        """
        Unfreeze the control_adapter (SimpleAdapter) for training.
        Called after switch_pipe_to_training_mode which freezes everything except LoRA params.
        """
        model = getattr(pipe, lora_base_model, None)
        if model is None:
            return
        if hasattr(model, 'control_adapter') and model.control_adapter is not None:
            model.control_adapter.train()
            for param in model.control_adapter.parameters():
                param.requires_grad = True
            print(f"Unfroze control_adapter in {lora_base_model} for full training")
        # Handle dit2 as well
        if lora_base_model == "dit" and hasattr(pipe, 'dit2') and pipe.dit2 is not None:
            if hasattr(pipe.dit2, 'control_adapter') and pipe.dit2.control_adapter is not None:
                pipe.dit2.control_adapter.train()
                for param in pipe.dit2.control_adapter.parameters():
                    param.requires_grad = True
                print(f"Unfroze control_adapter in dit2 for full training")
        
    def _add_rayrope_layers(self, model, device):
        """Add d_proj layers to each SelfAttention block for RayRoPE depth prediction."""
        if model is None:
            return
        import torch.nn as nn
        model.use_rayrope = True
        for block in model.blocks:
            sa = block.self_attn
            sa.use_rayrope = True
            sa.d_proj = nn.Linear(sa.dim, 2)
            nn.init.zeros_(sa.d_proj.weight)
            sa.d_proj.bias = nn.Parameter(torch.tensor([0.0, 3.0]))
            sa.d_proj = sa.d_proj.to(device=device, dtype=torch.bfloat16)
        print(f"Added d_proj layers to {len(model.blocks)} SelfAttention blocks for RayRoPE")

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
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        # Determine whether camera_poses_norm and intrinsics should be passed through
        # They are needed for: camera_adapter (SimpleAdapter) and/or prope (attention-level encoding)
        need_camera_params = self.use_camera_adapter or self.use_prope or self.use_rayrope
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["input_images"],
            "input_video": data["target_images"],
            # Camera conditioning: either raymap (channel concat) or poses+intrinsics (SimpleAdapter/PRoPE)
            "raymap": None if self.use_camera_adapter else data["raymap"],
            "camera_poses_norm": data.get("camera_poses_norm", None) if need_camera_params else None,
            "intrinsics": data.get("intrinsics", None) if need_camera_params else None,
            "use_prope": self.use_prope,
            "zero_temporal_rope": self.zero_temporal_rope,
            "use_rayrope": self.use_rayrope,
            "height": data["input_images"][0].size[1],
            "width": data["input_images"][0].size[0],
            "num_frames": len(data["target_images"]),
            "num_output_frames": len(data["target_images"]) - len(data["input_images"]),
            # For separate encoding: directly specify latent temporal dimension
            "num_latent_frames": len(data["target_images"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
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
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--modify_channels", default=False, action="store_true", help="Whether to modify the model's input channels.")
    parser.add_argument("--new_in_dim", type=int, default=None, help="New input dimension for the model (required if modify_channels is True).")
    parser.add_argument("--seperated_encoding", default=False, action="store_true", help="Whether to use separated encoding.")
    parser.add_argument("--fuse_vae_embedding_in_latents_multiple", default=False, action="store_true", help="Whether to fuse vae embedding in latents multiple times.")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a previously saved checkpoint (.safetensors) for resuming training. "
                             "This loads BOTH LoRA weights AND patch_embedding weights. "
                             "Use this INSTEAD of --lora_checkpoint for proper resume.")
    parser.add_argument("--sampling_strategy", type=str, default="prob_random",
                        choices=["all_random", "prob_random", "all_window", "curriculum"],
                        help="Strategy for sampling seperate_encoding_num_samples: "
                             "all_random (always full range), "
                             "prob_random (80%% full / 20%% window [24,48]), "
                             "all_window (always [24,48] window), "
                             "curriculum (first half epochs window, second half random).")
    parser.add_argument("--use_camera_adapter", default=False, action="store_true",
                        help="Use SimpleAdapter (Fun Camera Control style) for camera conditioning "
                             "instead of channel concatenation. When set, the patchify layer is NOT "
                             "modified; the pretrained SimpleAdapter from the checkpoint is finetuned "
                             "alongside LoRA. If the checkpoint has no adapter, a new one is created. "
                             "Mutually exclusive with --modify_channels.")
    parser.add_argument("--reverse_pred_order", default=False, action="store_true",
                        help="Reverse the frame order so the FIRST frame is the prediction target "
                             "and the remaining frames are context. By default (False), the LAST "
                             "frame is the prediction target. Affects dataset ordering, condition "
                             "embedding mask, pose normalization, and loss masking.")
    parser.add_argument("--use_prope", default=False, action="store_true",
                        help="Use PRoPE (Projective Positional Encoding) for camera-geometry-aware "
                             "attention. Replaces 3D grid RoPE with camera-relative positional "
                             "encoding in self-attention. Requires camera_poses_norm and intrinsics "
                             "from the dataset. Can be combined with raymap channel concat or "
                             "SimpleAdapter for token-level camera conditioning.")
    parser.add_argument("--zero_temporal_rope", default=False, action="store_true",
                        help="Zero out the temporal (frame) component of 3D RoPE by replacing "
                             "temporal frequencies with identity (1+0j). Spatial (height/width) "
                             "RoPE remains unchanged. This removes temporal position information "
                             "so the model treats all frames as having the same temporal position.")
    parser.add_argument("--use_rayrope", default=False, action="store_true",
                        help="Use hybrid RayRoPE encoding. Within each (t,h,w) axis, replaces "
                             "the low-frequency 6 complex dims with geometry-aware RayRoPE "
                             "(camera center + ray direction + predicted depth). Adds per-layer "
                             "d_proj heads for depth prediction. Requires camera_poses_norm and "
                             "intrinsics from the dataset. Mutually exclusive with --use_prope.")
    parser.add_argument("--num_dataset_samples", type=int, default=1000, help="Number of dataset samples to use for training.")
    parser.add_argument("--no_pixel_unshuffle", default=False, action="store_true",
                        help="Do not use pixel unshuffle to downscale the raymap to 1/8 resolution.")
    parser.add_argument("--num_input_frames", type=int, default=None,
                        help="Number of input (context) frames M. "
                             "If neither this nor --num_output_frames is set, M is randomly sampled "
                             "each iteration (random M-to-N mode).")
    parser.add_argument("--num_output_frames", type=int, default=None,
                        help="Number of output (target) frames N. "
                             "If neither this nor --num_input_frames is set, N is randomly sampled "
                             "each iteration (random M-to-N mode). Default when M is fixed: 1.")
    parser.add_argument("--min_input_frames", type=int, default=3,
                        help="Minimum input frames in random M-to-N mode. Default: 3.")
    parser.add_argument("--min_output_frames", type=int, default=1,
                        help="Minimum output frames in random M-to-N mode. Default: 1.")
    parser.add_argument("--dataset_type", type=str, default="dl3dv",
                        choices=["dl3dv", "eschernet_combined"],
                        help="Which dataset to use. 'dl3dv' uses the DL3DV dataset via "
                             "my_cognvs_dataset. 'eschernet_combined' uses BlendedMVS + "
                             "RealEstate10K + SpatialVid via EscherNetCombinedDataset.")
    parser.add_argument("--combined_dataset_names", type=str, nargs="+",
                        default=["blendedmvs", "realestate10k", "spatialvid"],
                        help="Dataset names for eschernet_combined mode.")
    parser.add_argument("--combined_dataset_ratios", type=float, nargs="+",
                        default=[4, 3, 3],
                        help="Sampling ratios for eschernet_combined mode.")
    parser.add_argument("--resize_mode", type=str, default="crop",
                        choices=["crop", "stretch"],
                        help="Image resize mode for training. 'crop' = uniform-scale + "
                             "center-crop (default). 'stretch' = resize to exact target "
                             "dimensions (matches inference stretch mode).")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    if args.dataset_type == "eschernet_combined":
        dataset = EscherNetCombinedDataset(
            dataset_names=args.combined_dataset_names,
            dataset_ratios=args.combined_dataset_ratios,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            repeat=args.dataset_repeat,
            sampling_strategy=args.sampling_strategy,
            reverse_pred_order=args.reverse_pred_order,
            no_pixel_unshuffle=args.no_pixel_unshuffle,
            num_input_frames=args.num_input_frames,
            num_output_frames=args.num_output_frames,
            min_input_frames=args.min_input_frames,
            min_output_frames=args.min_output_frames,
            resize_mode=args.resize_mode,
        )
    else:
        assert args.dataset_base_path, (
            "--dataset_base_path is required when --dataset_type is 'dl3dv'"
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
        use_rayrope=args.use_rayrope,
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
