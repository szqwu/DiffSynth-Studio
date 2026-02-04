import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override


import json
import os
from PIL import Image
import random
# import cv2
import numpy as np
import glob
import struct
from collections import defaultdict
from einops import rearrange

from io import BytesIO
import imageio.v3 as iio
import torch.nn.functional as F
from torch import Tensor
random.seed(1234)
import torchvision
import imageio


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
# import decord  # isort:skip

# decord.bridge.set_bridge("torch")

logger = get_logger("trainer", "INFO")


def camera_to_raymap(
    Ks: Tensor,
    camtoworlds: Tensor,
    height: int,
    width: int,
    downscale: float = 1.0,
    include_ups: bool = False,
):
    """Construct the raymap from the camera intrinsics and extrinsics.

    Note: This function expects OpenCV camera coordinates.

    Args:
        Ks: The camera intrinsics tensor with shape (..., 3, 3).
        camtoworlds: The camera extrinsics tensor with shape (..., 4, 4).
        height: The height of original image corresponding to intrinsics.
        width: The width of original image corresponding to intrinsics.
        downscale: Downscale factor for the raymap.
        include_ups: Whether to include the up direction in the raymap.

    Returns:
        The raymap tensor with shape (..., H, W, 6).
    """
    assert Ks.shape[-2:] == (3, 3), "Expected Ks to have shape (..., 3, 3)."
    assert camtoworlds.shape[-2:] == (
        4,
        4,
    ), "Expected camtoworlds to have shape (..., 4, 4)."
    # assert width % downscale == 0, "Expected width to be divisible by downscale."
    # assert height % downscale == 0, "Expected height to be divisible by downscale."

    # Downscale the intrinsics.
    intrinsic = Ks.clone()
    dtype = Ks.dtype
    Ks = torch.stack(
        [
            Ks[..., 0, :] * downscale,
            Ks[..., 1, :] * downscale,
            Ks[..., 2, :],
        ],
        dim=-2,
    )  # [..., 3, 3]
    width = int(width * downscale)
    height = int(height * downscale)

    # Construct pixel coordinates
    x, y = torch.meshgrid(
        torch.arange(width, device=Ks.device),
        torch.arange(height, device=Ks.device),
        indexing="xy",
    )  # [H, W]
    coords = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1).to(dtype)  # [H, W, 3]

    # To camera coordinates [..., H, W, 3]
    dirs = torch.einsum("...ij,...hwj->...hwi", Ks.float().inverse().to(dtype), coords)

    # To world coordinates [..., H, W, 3]
    dirs = torch.einsum("...ij,...hwj->...hwi", camtoworlds[..., :3, :3], dirs)
    dirs = F.normalize(dirs, p=2, dim=-1)

    # Camera origin in world coordinates [..., H, W, 3]
    origins = torch.broadcast_to(camtoworlds[..., None, None, :3, -1], dirs.shape)

    if include_ups:
        # Extract the up direction (second column)
        ups = torch.broadcast_to(camtoworlds[..., None, None, :3, 1], dirs.shape)
        ups = F.normalize(ups, p=2, dim=-1)
        return torch.cat([origins, dirs, ups], dim=-1)
    else:
        return torch.cat([origins, dirs], dim=-1)  # [..., H, W, 6]

def raymap_to_plucker(raymap: Tensor) -> Tensor:
    """Convert raymap to Plücker coordinates.

    Args:
        raymap: The raymap tensor with shape (..., H, W, 6).

    Returns:
        The Plücker coordinates tensor with shape (..., H, W, 3).
    """
    assert raymap.shape[-1] == 6, "Expected raymap to have shape (..., H, W, 6)."
    ray_origins, ray_directions = torch.split(raymap, [3, 3], dim=-1)
    # Normalize ray directions to unit vectors
    ray_directions = F.normalize(ray_directions, p=2, dim=-1)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    return torch.cat([ray_directions, plucker_normal], dim=-1)

def get_plucker_rays(pose, intrinsic, height, width):
    """
    Compute Plucker ray maps for pose_in relative to pose_out (query pose).
    
    Args:
        pose_in: (B, T_in, 4, 4) - input camera poses (w2c)
        intrinsic: (B, T_in, 3, 3) - input camera intrinsic matrix
        image_size: int - image size (assumed square)
        patch_size: int - patch size for Vision Transformer (32 for 224x224 -> 7x7 patches)
    
    Returns:
        plucker_ray_map: (B*T_in, num_patches, 6) - Plucker coordinates for each patch
    """
    downscale_factor = 1
    raymap = camera_to_raymap(intrinsic, pose, height=height, width=width, downscale=downscale_factor)
    plucker_ray_map = raymap_to_plucker(raymap)
    plucker_ray_map_permuted = plucker_ray_map.permute(0, 3, 1, 2)
    pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor=16)
    plucker_ray_map_permuted_unshuffled = pixel_unshuffle(plucker_ray_map_permuted)
    # interpolate to 1/8 resolution
    # plucker_ray_map_permuted_unshuffled = F.interpolate(plucker_ray_map_permuted, scale_factor=1/8, mode='bilinear', align_corners=False)

    return plucker_ray_map_permuted_unshuffled

def get_raymap_from_camera_parameters(
    intrinsic,
    camera_pose,
    H,
    W,
    vae_downsample=8,
    align_corners=True,
):
    def get_raymap_from_trans2d(intrinsic, H, W):
        fu = intrinsic[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
        fv = intrinsic[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
        cu = intrinsic[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
        cv = intrinsic[:, 1, 2].unsqueeze(-1).unsqueeze(-1)

        u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        u = u.unsqueeze(0).repeat(intrinsic.shape[0], 1, 1).to(intrinsic.device)
        v = v.unsqueeze(0).repeat(intrinsic.shape[0], 1, 1).to(intrinsic.device)

        z_cam = torch.ones_like(u).to(intrinsic.device)
        x_cam = (u - cu) / fu
        y_cam = (v - cv) / fv
        addition_dim = torch.ones_like(u).to(intrinsic.device)
        return torch.stack((x_cam, y_cam, z_cam, addition_dim), dim=-1)

    raymap_cam = get_raymap_from_trans2d(intrinsic, H, W).to(camera_pose.device)

    T, raymap_cam_h, raymap_cam_w, _ = raymap_cam.shape
    raymap_cam = rearrange(raymap_cam, "t h w c -> t c (h w)")

    _camera_pose = camera_pose.clone()
    _camera_pose[:, :3, 3] = 0.0
    raymap_world = torch.bmm(_camera_pose, raymap_cam)
    raymap_world = rearrange(
        raymap_world, "t c (h w) -> t c h w", h=raymap_cam_h, w=raymap_cam_w
    )

    if vae_downsample != 1:
        raymap_world = F.interpolate(
            raymap_world,
            scale_factor=1 / vae_downsample,
            mode="bilinear",
            align_corners=align_corners,
        )
    raymap_world = raymap_world[:, :3]
    ray_o = torch.ones_like(raymap_world).to(raymap_world.device) * camera_pose[
        :, :3, 3
    ].unsqueeze(-1).unsqueeze(-1)

    raymap_world = torch.cat([raymap_world, ray_o], dim=1)
    return raymap_world

def signed_log1p(x):
    """
    Computes log(1 + abs(x)) while keeping the original sign of x.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transformed tensor with the same sign as x.
    """
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    elif isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x))
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

def camera_pose_to_raymap(
    camera_pose,
    intrinsic,
    ray_o_scale_factor: float = 10.0,
    dmax: float = 1.0,
    H: int = 480,
    W: int = 720,
    vae_downsample: int = 8,
    align_corners: bool = False,
) -> np.ndarray:
    """
    Convert camera pose to raymap.

    Args:
        camera_pose: (N, 4, 4) camera poses
        intrinsic: (N, 3, 3) intrinsics
        ray_o_scale_factor: A constant scale factor for ray_o to avoid too large translation values.
            Default to 10.0. If you use pre-trained AetherV1 model, you should always set it to 10.0.
        dmax: A constant scale factor for ray_d to avoid too large translation values.
            It should be equal to the maximum disparity value (before sqrt) of the sequence
            if you have ground truth disparity. Default to 1.0.
    Returns:
        (N, 6, H, W) raymap
    """
    is_numpy = isinstance(camera_pose, np.ndarray)
    if is_numpy:
        camera_pose = torch.from_numpy(camera_pose).float()
        intrinsic = torch.from_numpy(intrinsic).float()
    scale_factor = 1.0 / dmax
    camera_pose[:, :3, 3] = signed_log1p(
        camera_pose[:, :3, 3] / scale_factor * ray_o_scale_factor
    )
    raymap = get_raymap_from_camera_parameters(
        intrinsic,
        camera_pose,
        H,
        W,
        vae_downsample,
        align_corners,
    )
    if is_numpy:
        raymap = raymap.cpu().numpy()
    return raymap

@torch.no_grad()
def normalize_w2c_make_cam0_origin(w2c: torch.Tensor):
    """
    Args:
        w2c: (N,4,4) world-to-camera extrinsics (R|t; 0 0 0 1)
    Returns:
        w2c_norm: (N,4,4) normalized world-to-camera
        c2w_norm: (N,4,4) inverse of w2c_norm (camera-to-world)
        scale:    scalar used to make avg camera-center distance = 1
    """
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4), "w2c must be (N,4,4)"

    device, dtype = w2c.device, w2c.dtype
    N = w2c.shape[0]

    # Convert to camera-to-world for easier operations
    c2w = torch.linalg.inv(w2c)                      # (N,4,4)
    R = c2w[:, :3, :3]                               # (N,3,3)
    t = c2w[:, :3,  3]                               # (N,3) camera centers in world

    # Reference (camera 0)
    R0 = R[0]                                        # (3,3)
    t0 = t[0]                                        # (3,)

    # Align such that cam0 -> identity pose at origin
    R_align = R0.transpose(0, 1)                     # R0^T
    t_shift = t - t0                                 # translate centers so cam0 at origin
    t_rot   = (R_align @ t_shift.unsqueeze(-1)).squeeze(-1)  # rotate centers
    R_rot   = R_align @ R                            # rotate orientations

    # Uniform scale so avg ||center|| = 1
    dists = t_rot.norm(dim=-1)                       # (N,)
    scale = dists.mean().clamp_min(1e-12)            # avoid div-by-zero
    t_norm = t_rot / scale

    # Reassemble normalized c2w
    c2w_norm = torch.zeros_like(c2w)
    c2w_norm[:, :3, :3] = R_rot
    c2w_norm[:, :3,  3] = t_norm
    c2w_norm[:,  3, :]  = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)

    # Back to w2c
    w2c_norm = torch.linalg.inv(c2w_norm)

    return w2c_norm, c2w_norm, scale

@torch.no_grad()
def normalize_w2c_make_cam_last_origin(w2c: torch.Tensor):
    """
    Args:
        w2c: (N,4,4) world-to-camera extrinsics (R|t; 0 0 0 1)
    Returns:
        w2c_norm: (N,4,4) normalized world-to-camera
        c2w_norm: (N,4,4) inverse of w2c_norm (camera-to-world)
        scale:    scalar used to make avg camera-center distance = 1
    """
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4), "w2c must be (N,4,4)"

    device, dtype = w2c.device, w2c.dtype
    N = w2c.shape[0]

    # Convert to camera-to-world for easier operations
    c2w = torch.linalg.inv(w2c)                      # (N,4,4)
    R = c2w[:, :3, :3]                               # (N,3,3)
    t = c2w[:, :3,  3]                               # (N,3) camera centers in world

    # Reference (camera 0)
    R0 = R[-1]                                        # (3,3)
    t0 = t[-1]                                        # (3,)

    # Align such that cam0 -> identity pose at origin
    R_align = R0.transpose(0, 1)                     # R0^T
    t_shift = t - t0                                 # translate centers so cam0 at origin
    t_rot   = (R_align @ t_shift.unsqueeze(-1)).squeeze(-1)  # rotate centers
    R_rot   = R_align @ R                            # rotate orientations

    # Uniform scale so avg ||center|| = 1
    dists = t_rot.norm(dim=-1)                       # (N,)
    scale = dists.mean().clamp_min(1e-12)            # avoid div-by-zero
    t_norm = t_rot / scale

    # Reassemble normalized c2w
    c2w_norm = torch.zeros_like(c2w)
    c2w_norm[:, :3, :3] = R_rot
    c2w_norm[:, :3,  3] = t_norm
    c2w_norm[:,  3, :]  = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)

    # Back to w2c
    w2c_norm = torch.linalg.inv(c2w_norm)

    return w2c_norm, c2w_norm, scale

class my_cognvs_dataset(Dataset):
    def __init__(self, 
            base_path, 
            metadata_path,
            repeat, 
            num_frames, 
            height, 
            width, 
            height_division_factor, 
            width_division_factor, 
            time_division_factor, 
            time_division_remainder, 
        ):

        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.load_from_cache = False  # This dataset doesn't use cached data

        # get full path of videos
        # self.video_list_dl3dv = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.mp4')]
        self.video_list_dl3dv = [os.path.join(base_path, f, "images_4") for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        self.video_list = self.video_list_dl3dv
        self.video_list.sort()
        print(f"Total number of videos: {len(self.video_list)}")

    
    def scale_intrinsics(self, fx, fy, cx, cy, orig_width, orig_height, target_width, target_height):
        """
        Scale camera intrinsics to target resolution
        Args:
            fx, fy: focal lengths
            cx, cy: principal point
            orig_width, orig_height: original image resolution
            target_width, target_height: target resolution
        Returns:
            3x3 intrinsic matrix scaled to target resolution
        """
        # Scale factors
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        
        # Scale intrinsics
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y
        
        # Build intrinsic matrix
        K = np.array([
            [fx_scaled, 0, cx_scaled],
            [0, fy_scaled, cy_scaled],
            [0, 0, 1]
        ])
        return K
    
    def load_camera_parameters(self, video_name, frame_indices):
        """
        Load camera intrinsics and extrinsics for given frame indices from transforms.json
        Args:
            video_name: name of the video/scene
            frame_indices: list of frame indices to load
        Returns:
            tuple of (intrinsics, camera_poses) or (None, None) if not available
            - intrinsics: (N, 3, 3) numpy array
            - camera_poses: (N, 4, 4) numpy array
        """
        transforms_path = os.path.join(self.metadata_path, video_name.replace("images_4", "transforms.json"))
        
        if not os.path.exists(transforms_path):
            logger.warning(f"transforms.json not found: {transforms_path}")
            return None, None
        
        try:
            # Load transforms.json
            with open(transforms_path, 'r') as f:
                transforms_data = json.load(f)
            
            # Get camera intrinsics (shared across all frames)
            orig_width = transforms_data['w']
            orig_height = transforms_data['h']
            fl_x = transforms_data['fl_x']
            fl_y = transforms_data['fl_y']
            cx = transforms_data['cx']
            cy = transforms_data['cy']
            
            # Scale intrinsics to target resolution (width=720, height=480)
            K = self.scale_intrinsics(fl_x, fl_y, cx, cy, orig_width, orig_height, self.width, self.height)
            
            # Get frames data
            frames_data = transforms_data['frames']
            
            # Load camera parameters for each frame index
            intrinsics_list = []
            camera_poses_list = []
            
            for frame_idx in frame_indices:
                if frame_idx >= len(frames_data):
                    logger.warning(f"Frame index {frame_idx} out of range for {video_name} (has {len(frames_data)} frames)")
                    return None, None
                
                frame_data = frames_data[frame_idx]
                
                # Get transform matrix (4x4 camera-to-world transformation)
                transform_matrix = np.array(frame_data['transform_matrix'], dtype=np.float32)
                
                # The intrinsics are the same for all frames
                intrinsics_list.append(K)
                camera_poses_list.append(transform_matrix)
            
            intrinsics = np.stack(intrinsics_list, axis=0)  # (N, 3, 3)
            camera_poses = np.stack(camera_poses_list, axis=0)  # (N, 4, 4)
            
            return intrinsics, camera_poses
            
        except Exception as e:
            logger.warning(f"Error loading camera parameters from transforms.json: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def __len__(self):
        """Return the total number of sequences."""
        return len(self.video_list)

    def __getitem__(self, idx):
        input_video_path = self.video_list[idx]
        target_video_path = self.video_list[idx]

        input_video_frames = LoadVideo(num_frames=self.num_frames, 
            time_division_factor=self.time_division_factor, 
            time_division_remainder=self.time_division_remainder
        )(input_video_path)
        # print(f"input_video_frames: {len(input_video_frames)}")
        # target_video_frames = self.read_video_frames(target_video_path)
        target_video_frames = input_video_frames.copy()

        if len(input_video_frames) == 0:
            logger.error(f"Empty input video for idx {idx}: {input_video_path}")
            # Return a different sample or skip
            return self.__getitem__((idx + 1) % len(self))
    
        if len(target_video_frames) == 0:
            logger.error(f"Empty target video for idx {idx}: {target_video_path}")
            return self.__getitem__((idx + 1) % len(self))

        num_frames = min(len(input_video_frames), len(target_video_frames))
        first_frame = input_video_frames[0]
        W_orig, H_orig = first_frame.size
        # print(f"W_orig: {W_orig}, H_orig: {H_orig}")

        seperate_encoding_num_samples = random.randint(24, 48)
        start_idx = random.randint(0, num_frames - seperate_encoding_num_samples)
        sampled_indices = list(range(start_idx, start_idx + seperate_encoding_num_samples))

        input_images = []
        target_images = []
        metadata = {}
        metadata["frame_indices"] = sampled_indices
        metadata["H_orig"] = H_orig
        metadata["W_orig"] = W_orig

        process = ImageCropAndResize(height=self.height, width=self.width, max_pixels=1920*1080, height_division_factor=self.height_division_factor, width_division_factor=self.width_division_factor)
        input_images = [process(frame) for frame in input_video_frames]
        target_images = input_images.copy()
        
        # Sample the indices from the list
        input_images = [input_images[i] for i in sampled_indices]
        target_images = [target_images[i] for i in sampled_indices]
        # print(f"input_images: {len(input_images)}, target_images: {len(target_images)}")

        # Separate encoding
        input_target_indices = np.random.choice(seperate_encoding_num_samples, 6, replace=False)
        context_indices = input_target_indices[:5]
        target_idx = input_target_indices[5:]

        context_frames = [input_images[i] for i in context_indices]
        target_frame = [target_images[i] for i in target_idx]
        
        target_images = context_frames + target_frame
        input_images = context_frames
        # print(f"target_images: {len(target_images)}, input_images: {len(input_images)}")

        # Load camera parameters and convert to raymaps
        intrinsics, camera_poses = self.load_camera_parameters(input_video_path, sampled_indices)
        # print(intrinsics[0])

        if intrinsics is not None and camera_poses is not None:
            camera_poses = torch.from_numpy(camera_poses).float()
            w2cs = torch.linalg.inv(camera_poses)
            w2cs[:, [1, 2], :] *= -1  # OpenGL -> OpenCV

            context_poses = w2cs[context_indices]
            target_pose = w2cs[target_idx]
            context_intrinsics = intrinsics[context_indices]
            target_intrinsics = intrinsics[target_idx]
            w2cs = torch.cat([context_poses, target_pose], dim=0)
            intrinsics = np.concatenate([context_intrinsics, target_intrinsics], axis=0)

            _, camera_poses_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)

            raymaps = get_plucker_rays(camera_poses_norm, torch.from_numpy(intrinsics).float(), height=self.height, width=self.width)
            print(f"raymaps shape: {raymaps.shape}")
            # Convert to torch tensor if it's numpy
            if isinstance(raymaps, np.ndarray):
                raymaps = torch.from_numpy(raymaps).float()
            
            
            metadata["raymaps"] = raymaps
            metadata["intrinsics"] = torch.from_numpy(intrinsics).float()
            metadata["camera_poses"] = camera_poses
            metadata["has_camera_params"] = True
            camera_conditions = raymaps

        else:
            logger.warning(f"Could not load camera parameters for {input_video_path}, raymaps not available")
            metadata["has_camera_params"] = False
            # Create dummy camera_conditions with proper shape
            camera_conditions = torch.zeros(self.num_frames, 
                6 * self.height_division_factor * self.width_division_factor, 
                self.height // self.height_division_factor, 
                self.width // self.width_division_factor
            )

        data = {
            "input_images": input_images,
            "target_images": target_images,
            "metadata": metadata,
            "raymap": camera_conditions,
            "prompt": "",  # Empty prompt for unconditional generation
            # "input_indices": list(input_target_indices) if self.data_strategy == "seperate_encoding" else None
        }
        return data

    # def read_video_frames(self, video_path):
    #     """Reads all frames from a video file using OpenCV and converts them to PIL Images."""
    #     cap = cv2.VideoCapture(video_path)

    #     if not cap.isOpened():
    #         logger.error(f"Failed to open video file: {video_path}")
    #         cap.release()
    #         return []

    #     frames = []
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         # Convert BGR (OpenCV) to RGB (PIL)
    #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         pil_img = Image.fromarray(frame_rgb)
    #         frames.append(pil_img)
    #     cap.release()

    #     if len(frames) == 0:
    #         logger.error(f"Video file returned 0 frames: {video_path}")

    #     return frames


    @staticmethod
    def load_image(image_path):
        """Load an image from the given path."""
        return Image.open(image_path).convert("RGB")

    def read_images_binary(self, path_to_model_file):
        """
        Read COLMAP images.bin file
        Returns: dict mapping image_id to (image_name, camera_id, qvec, tvec, point3D_ids)
        """
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_reg_images):
                binary_image_properties = struct.unpack("<I", fid.read(4))
                image_id = binary_image_properties[0]
                qvec = struct.unpack("<dddd", fid.read(32))
                tvec = struct.unpack("<ddd", fid.read(24))
                camera_id = struct.unpack("<I", fid.read(4))[0]
                image_name_bytes = b""
                current_char = struct.unpack("<c", fid.read(1))[0]
                while current_char != b"\x00":
                    image_name_bytes += current_char
                    current_char = struct.unpack("<c", fid.read(1))[0]
                image_name = image_name_bytes.decode("utf-8")
                num_points2D = struct.unpack("<Q", fid.read(8))[0]
                x_y_id_s = struct.unpack("<" + "ddq" * num_points2D, fid.read(24 * num_points2D))
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                images[image_id] = (image_name, camera_id, qvec, tvec, point3D_ids)
        return images

class LoadVideo:
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def load_images_from_folder(self, folder_path):
        """Load all images from folder in sorted order."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        image_files = sorted(image_files)
        
        frames = []
        for img_path in image_files:
            try:
                frame = Image.open(img_path).convert('RGB')
                frame = self.frame_processor(frame)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue
        return frames
        
    def __call__(self, data: str):
        # Check if data is a directory (image folder) or a file (video)
        if os.path.isdir(data):
            # Load images from folder
            frames = self.load_images_from_folder(data)
        else:
            # Load video file
            reader = imageio.get_reader(data)
            num_frames = self.get_num_frames(reader)
            frames = []
            for frame_id in range(num_frames):
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                frames.append(frame)
            reader.close()
        return frames

class ImageCropAndResize:
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image
