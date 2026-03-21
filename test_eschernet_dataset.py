"""Quick smoke test: load the EscherNetCombinedDataset and fetch one sample."""
import time
from diffsynth.core.data.eschernet_combined_dataset import EscherNetCombinedDataset

print("=== Constructing dataset (realestate10k + spatialvid) ===")
t0 = time.time()
ds = EscherNetCombinedDataset(
    dataset_names=["realestate10k", "spatialvid"],
    dataset_ratios=[3, 3],
    height=192,
    width=336,
    num_frames=7,
    num_input_frames=6,
    num_output_frames=1,
    sampling_strategy="prob_random",
)
print(f"Dataset built in {time.time() - t0:.1f}s  |  len={len(ds)}")

print("\n=== Fetching sample 0 ===")
t0 = time.time()
sample = ds[0]
print(f"Sample fetched in {time.time() - t0:.2f}s")

print(f"  input_images : {len(sample['input_images'])} x {sample['input_images'][0].size}")
print(f"  target_images: {len(sample['target_images'])} x {sample['target_images'][0].size}")
print(f"  raymap       : {sample['raymap'].shape}  dtype={sample['raymap'].dtype}")
print(f"  cam_poses    : {sample['camera_poses_norm'].shape}  dtype={sample['camera_poses_norm'].dtype}")
print(f"  intrinsics   : {sample['intrinsics'].shape}  dtype={sample['intrinsics'].dtype}")
print(f"  prompt       : {repr(sample['prompt'])}")
print(f"  metadata keys: {list(sample['metadata'].keys())}")
print(f"  has_camera   : {sample['metadata']['has_camera_params']}")

print("\n=== Fetching 4 more random samples ===")
import random
for i in range(4):
    idx = random.randrange(len(ds))
    t0 = time.time()
    s = ds[idx]
    dt = time.time() - t0
    print(f"  sample[{idx}]: input={len(s['input_images'])} target={len(s['target_images'])} "
          f"raymap={s['raymap'].shape} ({dt:.2f}s)")

print("\n=== All OK ===")
