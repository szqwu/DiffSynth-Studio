import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attn_mask=None):
    # An explicit attention mask (e.g. the input-encoder "prefix" mask) is only
    # supported by the SDPA path; FA2/FA3/Sage take no arbitrary mask, so force
    # the compatibility (SDPA) branch whenever a mask is provided.
    if attn_mask is not None:
        compatibility_mode = True
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attn_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attn_mask=attn_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, axis="global", fhw=None, prope_attn=None, attn_mask=None,
                ctx_cache=None, cache_key=None, n_in=None, cache_mode=None):
        if axis == "frame":
            if fhw is None:
                raise ValueError("axis='frame' requires fhw=(f, h, w) to reshape tokens.")
            if prope_attn is not None:
                raise NotImplementedError(
                    "PRoPE is not supported with axis='frame'. PRoPE expects the joint (B, num_heads, S, head_dim) layout."
                )
            if attn_mask is not None:
                raise NotImplementedError(
                    "attn_mask is not supported with axis='frame' (tokens are reshaped per-frame)."
                )
            f, _h, _w = fhw
            # Reshape (B, f*h*w, C) -> (B*f, h*w, C) so each frame attends only to itself
            x_in = rearrange(x, "b (f n) c -> (b f) n c", f=f)
        else:
            x_in = x

        # Transform-once context K/V cache (only used at inference with the prefix
        # attention mask; global axis, no PRoPE). Because the context tokens never
        # attend to the target and use a fixed t=0, their per-layer K/V are identical
        # across all denoising steps -> compute once ("write") and reuse ("read").
        if cache_mode is not None:
            if axis != "global" or prope_attn is not None:
                raise NotImplementedError("context K/V cache only supports global axis without PRoPE.")
            q = self.norm_q(self.q(x_in))
            k = self.norm_k(self.k(x_in))
            v = self.v(x_in)
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
            if cache_mode == "write":
                # freqs spans the full sequence; stash the (post-RoPE) context K/V.
                ctx_cache[("k", cache_key)] = k[:, :n_in].detach()
                ctx_cache[("v", cache_key)] = v[:, :n_in].detach()
                out = self.attn(q, k, v, attn_mask=attn_mask)
            elif cache_mode == "read":
                # x/freqs here are TARGET-only; prepend the cached context K/V so the
                # target attends to [context ; target] (no mask needed).
                ck = ctx_cache[("k", cache_key)]
                cv = ctx_cache[("v", cache_key)]
                k = torch.cat([ck, k], dim=1)
                v = torch.cat([cv, v], dim=1)
                out = self.attn(q, k, v, attn_mask=None)
            else:
                raise ValueError(f"unknown cache_mode {cache_mode!r}")
            return self.o(out)

        q = self.norm_q(self.q(x_in))
        k = self.norm_k(self.k(x_in))
        v = self.v(x_in)
        if prope_attn is not None:
            # PRoPE path: camera-geometry-aware positional encoding
            # Reshape to (B, num_heads, S, head_dim) for PRoPE
            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
            q = prope_attn._apply_to_q(q)
            k = prope_attn._apply_to_kv(k)
            v = prope_attn._apply_to_kv(v)
            # PRoPE's projection matrices are not norm-preserving (unlike RoPE rotations),
            # so Q/K magnitudes can grow unboundedly, causing attention logit explosion.
            # Normalize Q and K to restore the expected scale for dot-product attention.
            q = F.normalize(q, dim=-1) * (q.shape[-1] ** 0.5)
            k = F.normalize(k, dim=-1) * (k.shape[-1] ** 0.5)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            out = prope_attn._apply_to_o(out)
            out = rearrange(out, "b n s d -> b s (n d)")
        else:
            # Original path: 3D grid RoPE (also used for axis='frame' with 2D-only RoPE supplied via freqs)
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
            out = self.attn(q, k, v, attn_mask=attn_mask)
        out = self.o(out)

        if axis == "frame":
            f, _h, _w = fhw
            out = rearrange(out, "(b f) n c -> b (f n) c", f=f)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, axis="global", fhw=None, prope_attn=None, attn_mask=None,
                ctx_cache=None, cache_key=None, n_in=None, cache_mode=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, axis=axis, fhw=fhw, prope_attn=prope_attn,
                                                  attn_mask=attn_mask, ctx_cache=ctx_cache, cache_key=cache_key,
                                                  n_in=n_in, cache_mode=cache_mode))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class InputLatentResidualMLP(nn.Module):
    """Zero-init residual adaptor applied to the clean input-frame VAE latents.

    Returns ``z + MLP(z)``. The second linear layer is zero-initialized so the
    module starts as an exact identity (``z``) and only learns to add "visual
    juice" to the conditioning representation as training progresses.
    Operates on channel-first latents of shape ``(B, C, T, H, W)``.
    """
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor):
        x_perm = x.movedim(1, -1)  # (B, C, T, H, W) -> (B, T, H, W, C)
        residual = self.fc2(self.act(self.fc1(self.norm(x_perm))))
        out = x_perm + residual
        return out.movedim(-1, 1)  # back to (B, C, T, H, W)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        fuse_vae_embedding_in_latents_multiple: bool = False,
        seperated_encoding: bool = False,
        reverse_pred_order: bool = False,
        use_input_encoder: bool = False,
        use_input_latent_mlp: bool = True,
        per_layer_input_replacement: bool = True,
        input_prefix_attention: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.fuse_vae_embedding_in_latents_multiple = fuse_vae_embedding_in_latents_multiple
        self.seperated_encoding = seperated_encoding
        self.reverse_pred_order = reverse_pred_order
        self.use_input_encoder = use_input_encoder
        # Input-conditioning behavior flags:
        #   per_layer_input_replacement: re-inject the fixed clean input-frame tokens
        #     before every DiT block (true per-layer cross-attention; the inputs are
        #     never transformed). Default True for back-compat with the 5B run.
        #   input_prefix_attention: instead of replacing, let the input tokens
        #     transform through the layers but mask attention so context tokens
        #     attend only to each other (a clean, step-invariant prefix). Mutually
        #     exclusive in spirit with per_layer_input_replacement.
        self.per_layer_input_replacement = per_layer_input_replacement
        self.input_prefix_attention = input_prefix_attention
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        # Zero-init residual adaptor for the clean input-frame conditioning latents.
        # Applied (as z + MLP(z)) to the VAE latents of the input frames before
        # patch embedding, so the conditioning encoder can extract extra "visual
        # juice" once, before iterative denoising (true per-layer cross-attention).
        if use_input_encoder and use_input_latent_mlp:
            self.input_latent_mlp = InputLatentResidualMLP(out_dim)
        else:
            self.input_latent_mlp = None
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            # If adapter is frozen, skip storing activations to save GPU memory
            # and offload weights to CPU between forward passes
            adapter_trainable = any(p.requires_grad for p in self.control_adapter.parameters())
            if adapter_trainable:
                y_camera = self.control_adapter(control_camera_latents_input)
            else:
                # Move frozen adapter to GPU, run forward, move back to CPU
                device = control_camera_latents_input.device
                self.control_adapter.to(device)
                with torch.no_grad():
                    y_camera = self.control_adapter(control_camera_latents_input)
                y_camera = y_camera.detach()
                self.control_adapter.to("cpu")
                torch.cuda.empty_cache()
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                viewmats: Optional[torch.Tensor] = None,
                Ks: Optional[torch.Tensor] = None,
                image_hw: Optional[Tuple[int, int]] = None,
                zero_temporal_rope: bool = False,
                zero_xy_rope: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        f_freqs = self.freqs[0][:f]
        h_freqs = self.freqs[1][:h]
        w_freqs = self.freqs[2][:w]
        if zero_temporal_rope:
            f_freqs = torch.ones_like(f_freqs)
        if zero_xy_rope:
            h_freqs = torch.ones_like(h_freqs)
            w_freqs = torch.ones_like(w_freqs)
        freqs = torch.cat([
            f_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
            h_freqs.view(1, h, 1, -1).expand(f, h, w, -1),
            w_freqs.view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        # Prepare PRoPE if camera parameters are provided
        prope_attn = None
        if viewmats is not None and Ks is not None and image_hw is not None:
            from .prope import PropeDotProductAttention
            head_dim = self.dim // self.num_heads
            prope_attn = PropeDotProductAttention(
                head_dim=head_dim,
                patches_x=w, patches_y=h,
                image_width=image_hw[1], image_height=image_hw[0],
            ).to(x.device)
            prope_attn._precompute_and_cache_apply_fns(
                viewmats.to(dtype=x.dtype, device=x.device),
                Ks.to(dtype=x.dtype, device=x.device),
            )
        
        def create_custom_forward(module, _prope_attn=None):
            def custom_forward(*inputs):
                return module(*inputs, prope_attn=_prope_attn)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block, prope_attn),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block, prope_attn),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, prope_attn=prope_attn)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
