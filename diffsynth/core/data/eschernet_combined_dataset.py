"""
Combined dataset loading BlendedMVS, RealEstate10K, and SpatialVid data.
Outputs match the format expected by DiffSynth-Studio's train_SE.py.

The data sources and camera conventions follow the EscherNet codebase
(train_eschernet_rayrope_multigpu_combined.sh), while image processing,
intrinsics scaling, and output format follow DiffSynth-Studio's pipeline.
"""

import glob
import json
import os
import os.path as osp
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .my_v2v_dataset_images_in_plucker_SE import (
    ImageCropAndResize,
    get_plucker_rays,
    normalize_w2c_make_cam0_origin,
    normalize_w2c_make_cam_last_origin,
)

logger = get_logger("trainer", "INFO")

# ── Dataset paths (matching EscherNet module-level constants) ──────────────────
BLENDEDMVS_DIR = "/ocean/projects/cis240058p/qitaoz/BlendedMVS_processed"
BLENDEDMVS_SPLITS_DIR = (
    "/ocean/projects/cis240037p/qitaoz/ray_diffusion/diffusion/dataset/blendedmvs_splits"
)
REAL_ESTATE_10K_DIR = "/ocean/projects/cis240058p/ywu15/re10k"
SPATIALVID_PATH = (
    "/ocean/projects/cis250200p/mjeon2/datasets/SpatialVID_annotations/processed"
)
SPATIALVID_HQ_NAMES_PATH = (
    "/ocean/projects/cis250200p/mjeon2/EscherNet_legacy/process_spatialvid/HQ_names_3.txt"
)


def _convert_pytorch3d_to_opencv_w2c(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Build a 4x4 OpenCV w2c from PyTorch3D-convention R (3x3) and T (3,).

    PyTorch3D stores R in row-major (effectively transposed) and uses a
    different axis convention (x-right, y-up, z-toward-viewer).  The
    conversion matches EscherNet's ``convert_to_opencv`` exactly:
      R_cv = flip_rows_01( R^T )
      T_cv = flip_elems_01( T )
    """
    R_cv = R.T.copy()
    R_cv[[0, 1], :] *= -1
    T_cv = T.copy()
    T_cv[[0, 1]] *= -1
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_cv
    w2c[:3, 3] = T_cv
    return w2c


class EscherNetCombinedDataset(Dataset):
    """Combined multi-view dataset for DiffSynth-Studio training.

    Loads sequences from BlendedMVS, RealEstate10K, and/or SpatialVid.
    At each iteration a dataset is chosen by weighted probability, a random
    sequence is selected, frames are sampled, and data is returned in the
    exact format expected by ``train_SE.py``'s training loop.
    """

    # ── construction ───────────────────────────────────────────────────────
    def __init__(
        self,
        dataset_names: List[str],
        dataset_ratios: List[float],
        height: int,
        width: int,
        num_frames: int = 7,
        repeat: int = 1,
        sampling_strategy: str = "prob_random",
        reverse_pred_order: bool = False,
        no_pixel_unshuffle: bool = False,
        num_input_frames: Optional[int] = None,
        num_output_frames: Optional[int] = None,
        min_input_frames: int = 3,
        min_output_frames: int = 1,
        height_division_factor: int = 8,
        width_division_factor: int = 8,
    ):
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.repeat = repeat
        self.no_pixel_unshuffle = no_pixel_unshuffle
        self.reverse_pred_order = reverse_pred_order
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.load_from_cache = False
        self.current_epoch = 0
        self.num_epochs = 1

        # M-to-N frame split (mirrors my_cognvs_dataset logic)
        self.random_split = num_input_frames is None and num_output_frames is None
        if self.random_split:
            self.num_input_frames = None
            self.num_output_frames = None
            self.min_input_frames = min_input_frames
            self.min_output_frames = min_output_frames
            assert min_input_frames + min_output_frames <= num_frames, (
                f"min_input ({min_input_frames}) + min_output ({min_output_frames}) "
                f"> num_frames ({num_frames})"
            )
        elif num_input_frames is not None:
            self.num_input_frames = num_input_frames
            self.num_output_frames = num_output_frames if num_output_frames is not None else 1
            assert self.num_input_frames + self.num_output_frames == self.num_frames
        else:
            num_output_frames = num_output_frames if num_output_frames is not None else 1
            self.num_input_frames = num_frames - num_output_frames
            self.num_output_frames = num_output_frames

        assert sampling_strategy in (
            "all_random", "prob_random", "all_window", "curriculum",
        ), f"Unknown sampling_strategy: {sampling_strategy}"
        self.sampling_strategy = sampling_strategy

        # ── load per-dataset sequences ─────────────────────────────────────
        _loaders = {
            "blendedmvs": self._load_blendedmvs,
            "realestate10k": self._load_realestate10k,
            "spatialvid": self._load_spatialvid,
        }
        self.dataset_sequences: dict[str, list] = {}
        loaded_names: list[str] = []
        loaded_ratios: list[float] = []
        for name, ratio in zip(dataset_names, dataset_ratios):
            if name not in _loaders:
                raise ValueError(
                    f"Unknown dataset '{name}'. Choose from {list(_loaders)}"
                )
            try:
                seqs = _loaders[name]()
            except (PermissionError, FileNotFoundError, OSError) as e:
                print(
                    f"[EscherNetCombinedDataset] WARNING: skipping '{name}' "
                    f"(inaccessible): {e}"
                )
                continue
            if len(seqs) == 0:
                print(
                    f"[EscherNetCombinedDataset] WARNING: skipping '{name}' "
                    f"(0 sequences loaded)"
                )
                continue
            self.dataset_sequences[name] = seqs
            loaded_names.append(name)
            loaded_ratios.append(ratio)
            print(
                f"[EscherNetCombinedDataset] {name}: {len(seqs)} sequences"
            )

        if not loaded_names:
            raise RuntimeError(
                "No datasets could be loaded. Check file permissions and paths."
            )

        ratios = np.array(loaded_ratios, dtype=np.float64)
        self.dataset_names = loaded_names
        self.dataset_probs = ratios / ratios.sum()

        self._total_len = (
            max(len(s) for s in self.dataset_sequences.values()) * self.repeat
        )
        print(f"[EscherNetCombinedDataset] Virtual length: {self._total_len}")

    # ── metadata loaders ───────────────────────────────────────────────────
    def _load_blendedmvs(self) -> list:
        train_list = osp.join(BLENDEDMVS_SPLITS_DIR, "BlendedMVG_training.txt")
        with open(train_list, "r") as f:
            sequence_names = [ln.strip() for ln in f if ln.strip()]

        sequences = []
        for seq_name in tqdm(sequence_names, desc="BlendedMVS"):
            anno_path = osp.join(BLENDEDMVS_DIR, "annotations", f"{seq_name}.json")
            if not osp.exists(anno_path):
                continue
            with open(anno_path, "r") as f:
                cam = json.load(f)
            image_names = cam["image_names"]

            frames = []
            for img_name in image_names:
                info = cam[img_name]
                w_anno, h_anno = int(info["image_size_wh"][0]), int(info["image_size_wh"][1])
                l = min(w_anno, h_anno)
                fl = info["focal_length"]
                pp = info["principal_point"]
                frames.append({
                    "image_path": osp.join(
                        BLENDEDMVS_DIR, "images", seq_name, img_name + ".jpg"
                    ),
                    "fx": fl[0] * l,
                    "fy": fl[1] * l,
                    "cx": (w_anno - l * pp[0]) / 2.0,
                    "cy": (h_anno - l * pp[1]) / 2.0,
                    "R": np.array(info["R"], dtype=np.float32),
                    "T": np.array(info["T"], dtype=np.float32),
                    "convention": "pytorch3d",
                })
            if len(frames) >= 12:
                sequences.append(frames)
        return sequences

    def _load_realestate10k(self) -> list:
        dataset_path = osp.join(REAL_ESTATE_10K_DIR, "train")
        full_list_path = osp.join(dataset_path, "full_list.txt")
        seq_names: list[str] = []
        if osp.exists(full_list_path):
            with open(full_list_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and "metadata/" in line:
                        seq_names.append(line.split("metadata/")[-1].split(".")[0])
        else:
            for seq in glob.glob(osp.join(dataset_path, "*/")):
                seq_names.append(seq.rstrip("/").split("/")[-1])

        sequences = []
        for seq_name in tqdm(seq_names[:0], desc="RealEstate10K"):
            meta_path = osp.join(dataset_path, "metadata", seq_name + ".json")
            if not osp.exists(meta_path):
                continue
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            if len(metadata["frames"]) < 100:
                continue

            frames = []
            for fr in metadata["frames"]:
                fxfycxcy = fr["fxfycxcy"]
                w2c = np.array(fr["w2c"], dtype=np.float32)
                frames.append({
                    "image_path": fr["image_path"],
                    "fx": float(fxfycxcy[0]),
                    "fy": float(fxfycxcy[1]),
                    "cx": float(fxfycxcy[2]),
                    "cy": float(fxfycxcy[3]),
                    "w2c": w2c,
                    "convention": "opencv",
                })
            sequences.append(frames)
        return sequences

    def _load_spatialvid(self) -> list:
        with open(SPATIALVID_HQ_NAMES_PATH, "r") as f:
            seq_names = [ln.strip() for ln in f if ln.strip()]

        sequences = []
        for seq_name in tqdm(seq_names[:1000], desc="SpatialVid"):
            anno = osp.join(SPATIALVID_PATH, seq_name, f"{seq_name}.json")
            if not osp.exists(anno):
                continue
            with open(anno, "r") as f:
                data = json.load(f)

            frames = []
            for fr in data["frames"]:
                img_h, img_w = fr["image_hw"]
                fxfy = fr["fxfy"]
                cxcy = fr["cxcy"]
                w2c_raw = np.array(fr["w2c"], dtype=np.float32)
                frames.append({
                    "image_path": fr["image_path"],
                    "fx": fxfy[0] * img_w,
                    "fy": fxfy[1] * img_h,
                    "cx": cxcy[0] * img_w,
                    "cy": cxcy[1] * img_h,
                    "R": w2c_raw[:3, :3],
                    "T": w2c_raw[:3, 3],
                    "convention": "pytorch3d",
                })
            if len(frames) >= 48:
                sequences.append(frames)
        return sequences

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _scale_intrinsics(
        fx, fy, cx, cy, orig_w, orig_h, tgt_w, tgt_h,
    ) -> np.ndarray:
        """Adjust intrinsics for uniform-scale-then-center-crop (ImageCropAndResize)."""
        scale = max(tgt_w / orig_w, tgt_h / orig_h)
        resized_w = round(orig_w * scale)
        resized_h = round(orig_h * scale)
        crop_x = (resized_w - tgt_w) / 2.0
        crop_y = (resized_h - tgt_h) / 2.0
        return np.array([
            [fx * scale, 0, cx * scale - crop_x],
            [0, fy * scale, cy * scale - crop_y],
            [0, 0, 1],
        ], dtype=np.float64)

    def _determine_window_size(self, num_available: int) -> int:
        if self.sampling_strategy == "all_random":
            window = num_available
        elif self.sampling_strategy == "prob_random":
            window = num_available if random.random() < 0.8 else random.randint(24, 48)
        elif self.sampling_strategy == "all_window":
            window = random.randint(24, 48)
        elif self.sampling_strategy == "curriculum":
            if self.current_epoch < self.num_epochs // 2:
                window = random.randint(24, 48)
            else:
                window = num_available
        else:
            window = num_available
        return max(self.num_frames, min(window, num_available))

    # ── Dataset protocol ───────────────────────────────────────────────────
    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        # Pick a dataset according to ratios
        ds_idx = np.random.choice(len(self.dataset_names), p=self.dataset_probs)
        ds_name = self.dataset_names[ds_idx]
        sequences = self.dataset_sequences[ds_name]
        frames = sequences[random.randrange(len(sequences))]
        num_available = len(frames)

        if num_available < self.num_frames:
            return self.__getitem__((idx + 1) % len(self))

        # Sample frame indices using DiffSynth-style windowing
        window = self._determine_window_size(num_available)
        print(f"window: {window}")
        start = random.randint(0, num_available - window)
        pool = list(range(start, start + window))
        chosen_in_pool = np.random.choice(len(pool), self.num_frames, replace=False)
        sampled_indices = [pool[i] for i in chosen_in_pool]

        # M-to-N split
        if self.random_split:
            max_input = self.num_frames - self.min_output_frames
            cur_num_input = random.randint(self.min_input_frames, max_input)
            cur_num_output = self.num_frames - cur_num_input
        else:
            cur_num_input = self.num_input_frames
            cur_num_output = self.num_output_frames

        context_indices = sampled_indices[:cur_num_input]
        target_indices = sampled_indices[cur_num_input:]

        # Determine ordered indices (affects image list & camera arrays)
        if self.reverse_pred_order:
            ordered_indices = target_indices + context_indices
        else:
            ordered_indices = context_indices + target_indices

        # Image resizer
        process = ImageCropAndResize(
            height=self.height,
            width=self.width,
            max_pixels=1920 * 1080,
            height_division_factor=self.height_division_factor,
            width_division_factor=self.width_division_factor,
        )

        try:
            all_images: list[Image.Image] = []
            intrinsics_list: list[np.ndarray] = []
            w2cs_list: list[np.ndarray] = []

            for frame_idx in ordered_indices:
                frame = frames[frame_idx]

                # Load and resize image (returns PIL)
                img = Image.open(frame["image_path"]).convert("RGB")
                orig_w, orig_h = img.width, img.height
                img = process(img)
                all_images.append(img)

                # Intrinsics scaled for the target resolution
                K = self._scale_intrinsics(
                    frame["fx"], frame["fy"], frame["cx"], frame["cy"],
                    orig_w, orig_h, self.width, self.height,
                )
                intrinsics_list.append(K)

                # Extrinsics -> OpenCV w2c
                if frame["convention"] == "opencv":
                    w2c = frame["w2c"].copy()
                else:
                    w2c = _convert_pytorch3d_to_opencv_w2c(frame["R"], frame["T"])
                w2cs_list.append(w2c)

            w2cs = torch.from_numpy(np.stack(w2cs_list)).float()
            intrinsics = np.stack(intrinsics_list)

            # Normalize poses (reference camera at origin)
            if self.reverse_pred_order:
                _, camera_poses_norm, _ = normalize_w2c_make_cam0_origin(w2cs)
            else:
                _, camera_poses_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)

            # Plucker raymaps
            raymaps = get_plucker_rays(
                camera_poses_norm,
                torch.from_numpy(intrinsics).float(),
                height=self.height,
                width=self.width,
                no_pixel_unshuffle=self.no_pixel_unshuffle,
            )
            if isinstance(raymaps, np.ndarray):
                raymaps = torch.from_numpy(raymaps).float()

            # Split into context / target image lists
            if self.reverse_pred_order:
                input_images = all_images[cur_num_output:]
            else:
                input_images = all_images[:cur_num_input]
            target_images = all_images

            intrinsics_out = torch.from_numpy(intrinsics).float()

        except Exception as e:
            logger.warning(
                f"[EscherNetCombinedDataset] Error loading {ds_name} sample: {e}"
            )
            return self.__getitem__((idx + 1) % len(self))

        metadata = {
            "frame_indices": sampled_indices,
            "H_orig": orig_h,
            "W_orig": orig_w,
            "raymaps": raymaps,
            "intrinsics": intrinsics_out,
            "camera_poses": camera_poses_norm,
            "has_camera_params": True,
        }

        return {
            "input_images": input_images,
            "target_images": target_images,
            "metadata": metadata,
            "raymap": raymaps,
            "camera_poses_norm": camera_poses_norm,
            "intrinsics": intrinsics_out,
            "prompt": "",
        }
