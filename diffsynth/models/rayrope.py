"""
Hybrid RayRoPE module for Wan2.1 video DiT.

Copies core utility functions from the official RayRoPE implementation
(/ocean/projects/cis250177p/qwu6/RayRoPE/pos_enc/rayrope.py) and wraps
them in a HybridRayRoPE class that produces complex-valued frequency
tensors compatible with Wan2.1's rope_apply().

Layout:  [grid_t(16) | rr_0(6) | grid_h(15) | rr_1(6) | grid_w(15) | rr_2(6)]
         = 64 complex dims = 128 real dims = head_dim
"""

import math
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

MAX_DEPTH = 100.0
MAX_LOG_DEPTH = 3.0
MAX_D_F = 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions (copied from official RayRoPE)
# ═══════════════════════════════════════════════════════════════════════════════

def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def normalize_K(Ks: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
    Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
    Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
    Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
    Ks_norm[..., 2, 2] = 1.0
    return Ks_norm


def _get_cam_centers(
    c2ws: torch.Tensor,  # (batch, num_cameras, 4, 4)
    num_patches: int,
) -> torch.Tensor:
    batches = c2ws.shape[0]
    num_cameras = c2ws.shape[1]
    cam_centers = c2ws[:, :, :, 3]  # (batch, num_cameras, 4)
    cam_centers = cam_centers.view(batches, num_cameras, 1, 4).expand(
        batches, num_cameras, num_patches, 4
    )
    return cam_centers


def _get_point_coords(
    P_inv: torch.Tensor,  # (batch, num_cameras, 4, 4)
    patches_x: int,
    patches_y: int,
    offsets: List[Tuple[float, float]],
    depths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = P_inv.device
    batches = P_inv.shape[0]
    num_cameras = P_inv.shape[1]
    num_patches = patches_x * patches_y
    num_rays_per_patch = len(offsets)
    u_base, v_base = torch.meshgrid(
        torch.arange(patches_x, device=device),
        torch.arange(patches_y, device=device),
        indexing="xy",
    )
    coords = []
    for offset in offsets:
        u = ((u_base + offset[0]) / patches_x) - 0.5
        v = ((v_base + offset[1]) / patches_y) - 0.5
        coords.append(torch.stack([u, v], dim=-1).reshape(-1, 2))
    coords = torch.stack(coords, dim=1)
    coords = coords.view(1, 1, num_patches, num_rays_per_patch, 2).expand(
        batches, num_cameras, -1, -1, -1
    )

    if depths is not None:
        depths = torch.clamp(depths, min=1e-2, max=MAX_DEPTH)
        disparity = 1 / depths
        if depths.ndim == 5:
            coords_4d = torch.cat(
                [coords, torch.ones_like(disparity), disparity], dim=-1
            )
            coords_4d = torch.einsum("bcij,bcprj->bcpri", P_inv, coords_4d)
        elif depths.ndim == 6:
            coords = coords.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
            coords_4d = torch.cat(
                [coords, torch.ones_like(disparity), disparity], dim=-1
            )
            coords_4d = torch.einsum("bcij,ebcprj->ebcpri", P_inv, coords_4d)
    else:
        depths_ones = torch.ones_like(coords[..., :1])
        scales = torch.zeros_like(coords[..., :1])
        coords_4d = torch.cat([coords, depths_ones, scales], dim=-1)
        coords_4d = torch.einsum("bcij,bcprj->bcpri", P_inv, coords_4d)

    return coords_4d


def _transform_to_query_frame(
    points_world: torch.Tensor,
    P: torch.Tensor,
    w2c: torch.Tensor,
    transform_type: str = "pj",
    denc_type: str = "d",
    norm_by: str = "w",
):
    if transform_type == "3d":
        points_cam = torch.einsum("bij,...bcprj->...bcpri", w2c, points_world)
        if norm_by == "w":
            points_cam = points_cam / torch.clamp(points_cam[..., -1:], min=1e-4)
        elif norm_by == "length":
            points_cam = points_cam / (
                torch.norm(points_cam[..., :3], dim=-1, keepdim=True) + 1e-6
            )
        return points_cam[..., :3]
    elif transform_type == "pj":
        points_cam = torch.einsum("bij,...bcprj->...bcpri", P, points_world)
        safe_abs = torch.sqrt(points_cam[..., 2:3].pow(2) + 1e-9)
        z = torch.clamp(safe_abs, min=1e-4)
        w = torch.clamp(points_cam[..., -1:], min=1e-4)
        pd_dir = points_cam[..., :3] / (
            torch.norm(points_cam[..., :3], dim=-1, keepdim=True) + 1e-6
        )
        if denc_type == "d":
            pd_depth = z / w
            return pd_dir, pd_depth
        elif denc_type == "inv_d":
            pd_disparity = w / z
            return pd_dir, pd_disparity
        elif denc_type == "asinh_d":
            pd_depth = z / w
            pd_depth = torch.asinh(pd_depth)
            return pd_dir, pd_depth


def _prepare_depths(
    predicted_d: torch.Tensor,
    batch: int,
    num_cameras: int,
    num_patches: int,
    num_rays_per_patch: int = 1,
):
    predicted_d = predicted_d.reshape(batch, num_cameras, num_patches, 1, 2)
    predicted_logd = predicted_d[..., 0:1]
    predicted_sigma = predicted_d[..., 1:2]
    predicted_d1 = torch.exp(
        torch.clamp(predicted_logd - predicted_sigma, max=MAX_LOG_DEPTH)
    )
    predicted_d2 = torch.exp(
        torch.clamp(predicted_logd + predicted_sigma, max=MAX_LOG_DEPTH)
    )
    depths = torch.stack([predicted_d1, predicted_d2], dim=0)
    depths = depths.expand(-1, -1, -1, -1, num_rays_per_patch, -1)
    return depths


def _get_frequency(num_freqs: int, max_period: float, min_period: float):
    log_min_frequency = torch.log(torch.tensor(2 * torch.pi / max_period))
    log_max_frequency = torch.log(torch.tensor(2 * torch.pi / min_period))
    log_freqs = torch.linspace(log_min_frequency, log_max_frequency, num_freqs)
    freqs = torch.exp(log_freqs)
    return freqs


def _prepare_rope_coeff_uniformd(
    positions: dict,
    num_freqs: int,
    freq_base: float,
    batch: int,
    num_cameras: int,
    num_patches: int,
):
    coord_dim = 0
    device = None
    for key, value in positions.items():
        coord_dim += value.shape[-1]
        device = value.device

    cosine_list = []
    sine_list = []
    for pos_name, pos in positions.items():
        if pos_name in ["p0", "pd_3d"]:
            max_period = 1.0 * 4
        elif pos_name in ["pinf_dir", "pd_dir", "p0_dir"]:
            max_period = 2.0 * 4
        elif pos_name in ["pd_depth", "p0_depth"]:
            max_period = MAX_D_F * 2 * 4
            pos = torch.clamp(pos, min=-MAX_D_F, max=MAX_D_F)
        elif pos_name in ["pd_disparity", "p0_disparity"]:
            max_period = 20.0 * 4
            pos = torch.clamp(pos, min=0.0, max=20.0)
        elif pos_name in ["pd_asinh_depth", "p0_asinh_depth"]:
            max_period = math.asinh(MAX_D_F) * 2 * 4
            pos = torch.clamp(pos, min=-math.asinh(MAX_D_F), max=math.asinh(MAX_D_F))
        else:
            raise ValueError(f"Unknown position name: {pos_name}")

        min_period = max_period / (freq_base ** (num_freqs - 1))
        freqs = _get_frequency(num_freqs, max_period=max_period, min_period=min_period).to(
            device
        )
        rope_angle = torch.einsum("f,bcpd->bcpfd", freqs, pos)

        if rope_angle.shape[0] == batch * 2:
            rope_angles1 = rope_angle[:batch]
            rope_angles2 = rope_angle[batch:]
            same_mask = torch.isclose(rope_angles1, rope_angles2, atol=1e-2, rtol=0)

            cosine1 = torch.cos(rope_angles1)
            cosine2 = torch.cos(rope_angles2)
            sine1 = torch.sin(rope_angles1)
            sine2 = torch.sin(rope_angles2)
            delta = rope_angles2 - rope_angles1
            delta_safe = torch.where(same_mask, torch.ones_like(delta), delta)
            E_cosine = (sine2 - sine1) / delta_safe
            E_sine = (cosine1 - cosine2) / delta_safe

            cosine_final = torch.where(same_mask, cosine1, E_cosine)
            sine_final = torch.where(same_mask, sine1, E_sine)
            cosine_list.append(cosine_final)
            sine_list.append(sine_final)
        elif rope_angle.shape[0] == batch:
            cosine_list.append(torch.cos(rope_angle))
            sine_list.append(torch.sin(rope_angle))
        else:
            raise ValueError(f"Unexpected rope_angle batch size: {rope_angle.shape}")

    cosine_out = (
        torch.cat(cosine_list, dim=-1)
        .reshape(batch, num_cameras * num_patches, num_freqs * coord_dim)
        .contiguous()
    )
    sine_out = (
        torch.cat(sine_list, dim=-1)
        .reshape(batch, num_cameras * num_patches, num_freqs * coord_dim)
        .contiguous()
    )
    return cosine_out, sine_out


# ═══════════════════════════════════════════════════════════════════════════════
# HybridRayRoPE
# ═══════════════════════════════════════════════════════════════════════════════

class HybridRayRoPE:
    """
    Computes the 18 complex-dim RayRoPE portion of the hybrid frequency tensor.

    Config used (matches paper default for one center ray):
        pos_enc_type = 'd_pj+0_3d'   ->  6 coordinate dims
        num_rays_per_patch = 1
        depth_type = 'predict_dsig'
        denc_type = 'd'

    With num_rayrope_freqs=3, this gives 6 coords * 3 freqs = 18 complex dims.
    These are returned as complex tensors for concatenation with grid RoPE.
    """

    def __init__(
        self,
        w2cs: torch.Tensor,      # (B, C, 4, 4)
        Ks: torch.Tensor,         # (B, C, 3, 3)
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        num_rayrope_freqs: int = 3,
        freq_base: float = 3.0,
    ):
        self.batch = w2cs.shape[0]
        self.num_cameras = w2cs.shape[1]
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.num_patches = patches_x * patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.num_rayrope_freqs = num_rayrope_freqs
        self.freq_base = freq_base
        self.offsets = [[0.5, 0.5]]  # center ray only
        self.num_rays_per_patch = 1

        # pos_enc_type = 'd_pj+0_3d' => use_pd=True, pd_type='pj', use_p0=True, p0_type='3d'
        self.use_pd = True
        self.pd_type = "pj"
        self.use_p0 = True
        self.p0_type = "3d"
        self.denc_type = "d"

        # rope_coord_dim: 3 (p0_3d xyz) + 1*3 (pd_pj: dir_x, dir_y, depth) = 6
        self.rope_coord_dim = 6

        self._precompute(w2cs, Ks)

    def _precompute(self, w2cs: torch.Tensor, Ks: torch.Tensor):
        self.w2cs = w2cs
        self.c2ws = _invert_SE3(w2cs)
        Ks_norm = normalize_K(Ks, self.image_width, self.image_height)

        self.P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), w2cs)
        self.P_inv = torch.einsum(
            "...ij,...jk->...ik",
            self.c2ws,
            _lift_K(_invert_K(Ks_norm)),
        )

        self.p0_world = _get_cam_centers(self.c2ws, self.num_patches).unsqueeze(-2)
        self.pinf_world = _get_point_coords(
            self.P_inv, self.patches_x, self.patches_y, self.offsets
        )

    def build_rayrope_freqs(
        self, predicted_d: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            predicted_d: (B, S, 2) raw depth predictions (log-depth, log-sigma)
                         S = num_cameras * num_patches

        Returns:
            freqs_Q_complex: (B, S, 18) complex tensor for Q
            all_freqs_K_complex: list of C tensors, each (B, S, 18) complex, one per query camera
        """
        batch = self.batch
        num_cameras = self.num_cameras
        num_patches = self.num_patches

        depths = _prepare_depths(
            predicted_d,
            batch=batch,
            num_cameras=num_cameras,
            num_patches=num_patches,
            num_rays_per_patch=self.num_rays_per_patch,
        )

        pd_world = _get_point_coords(
            self.P_inv,
            self.patches_x,
            self.patches_y,
            self.offsets,
            depths,
        )

        positions_Q = defaultdict(list)
        all_cos_K = []
        all_sin_K = []

        for cam_idx in range(num_cameras):
            positions_KV = {}
            P_q = self.P[:, cam_idx, :, :]
            w2c_q = self.w2cs[:, cam_idx, :, :]

            # p0: camera center in query frame (3D transform)
            p0_3d = _transform_to_query_frame(
                self.p0_world, P_q, w2c_q, "3d", self.denc_type
            )
            p0_3d = p0_3d.flatten(-2, -1)
            positions_KV["p0"] = p0_3d
            positions_Q["p0"].append(p0_3d[:, cam_idx])

            # pd: point at depth in query frame (projective transform)
            pd_dir, pd_d = _transform_to_query_frame(
                pd_world, P_q, w2c_q, "pj", self.denc_type
            )
            pd_dir = pd_dir.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
            pd_d = pd_d.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
            positions_KV["pd_dir"] = pd_dir
            positions_Q["pd_dir"].append(pd_dir[:, cam_idx])
            positions_KV["pd_depth"] = pd_d
            positions_Q["pd_depth"].append(pd_d[:, cam_idx])

            cos_KV, sin_KV = _prepare_rope_coeff_uniformd(
                positions_KV,
                self.num_rayrope_freqs,
                self.freq_base,
                batch,
                num_cameras,
                num_patches,
            )
            all_cos_K.append(cos_KV)
            all_sin_K.append(sin_KV)

        for key, val in positions_Q.items():
            positions_Q[key] = torch.stack(val, dim=1)

        cos_Q, sin_Q = _prepare_rope_coeff_uniformd(
            positions_Q,
            self.num_rayrope_freqs,
            self.freq_base,
            batch,
            num_cameras,
            num_patches,
        )

        # Convert (cos, sin) with inverse=True convention to complex:
        # RayRoPE applies rotation with inverse=True on both Q and K, which means:
        #   x1_out = x1 * cos + x2 * sin
        #   x2_out = -x1 * sin + x2 * cos
        # This is equivalent to complex multiplication by (cos - i*sin) = conj(e^{iθ})
        # In Wan2.1's rope_apply: x_out = x_complex * freqs_complex
        # So freqs_complex = cos - i*sin = complex(cos, -sin)
        freqs_Q_complex = torch.complex(cos_Q.float(), -sin_Q.float())
        all_freqs_K_complex = [
            torch.complex(c.float(), -s.float()) for c, s in zip(all_cos_K, all_sin_K)
        ]

        return freqs_Q_complex, all_freqs_K_complex


def interleave_grid_rayrope(
    grid_t: torch.Tensor,
    grid_h: torch.Tensor,
    grid_w: torch.Tensor,
    rr_freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Interleave factored grid freqs with RayRoPE freqs into the per-axis layout.

    Args:
        grid_t: (S, 1, 16) or broadcastable — temporal grid freqs (complex)
        grid_h: (S, 1, 15) or broadcastable — height grid freqs (complex)
        grid_w: (S, 1, 15) or broadcastable — width grid freqs (complex)
        rr_freqs: (B, S, 18) complex — RayRoPE freqs to split into 3 groups of 6

    Returns:
        (B, S, 1, 64) complex — interleaved freqs ready for rope_apply
    """
    rr_0, rr_1, rr_2 = rr_freqs.chunk(3, dim=-1)  # each (B, S, 6)

    B = rr_freqs.shape[0]
    S = rr_freqs.shape[1]

    # Expand grid freqs to (B, S, N) — they're factored so same across batch
    if grid_t.dim() == 3 and grid_t.shape[0] == S:
        # grid_t is (S, 1, 16) from the factored construction
        gt = grid_t.squeeze(1).unsqueeze(0).expand(B, -1, -1)  # (B, S, 16)
    else:
        gt = grid_t.expand(B, S, -1)

    if grid_h.dim() == 3 and grid_h.shape[0] == S:
        gh = grid_h.squeeze(1).unsqueeze(0).expand(B, -1, -1)
    else:
        gh = grid_h.expand(B, S, -1)

    if grid_w.dim() == 3 and grid_w.shape[0] == S:
        gw = grid_w.squeeze(1).unsqueeze(0).expand(B, -1, -1)
    else:
        gw = grid_w.expand(B, S, -1)

    # Cast grid to same dtype as rr
    gt = gt.to(rr_freqs.dtype)
    gh = gh.to(rr_freqs.dtype)
    gw = gw.to(rr_freqs.dtype)

    # Interleave: [g_t(16) | rr_0(6) | g_h(15) | rr_1(6) | g_w(15) | rr_2(6)]
    combined = torch.cat([gt, rr_0, gh, rr_1, gw, rr_2], dim=-1)  # (B, S, 64)
    return combined.unsqueeze(2)  # (B, S, 1, 64) for rope_apply broadcast over heads
