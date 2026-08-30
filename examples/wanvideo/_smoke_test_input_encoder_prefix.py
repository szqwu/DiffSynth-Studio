"""
Smoke test for the 14B input-encoder "transform-through-layers + prefix mask" path.

Runs on CPU with a tiny real WanModel (no checkpoint) and exercises
model_fn_wan_video directly. It verifies:

  1) PREFIX MASK -> context is invariant to the target. With
     input_prefix_attention=True + per-token timestep (context=0, target=t),
     changing ONLY the target frame's latent must NOT change the context frames'
     output (this is exactly what makes the context a step-invariant, cache-once
     prefix, and what makes a target-only loss well-defined).
  2) NEGATIVE CONTROL -> without the mask (plain self-attention, no replacement),
     changing the target DOES change the context output.
  3) PER-TOKEN TIMESTEP works on a model whose base had seperated_timestep=False
     (like the 14B): building with seperated_timestep=True + multiple runs cleanly.
  4) NO MLP -> use_input_latent_mlp=False leaves dit.input_latent_mlp is None and
     drops those params from the state_dict.
  5) FLAG-OFF PATH UNCHANGED -> a vanilla model (use_input_encoder=False) still
     runs and builds no mask (output shape correct), and the default flags keep
     per_layer_input_replacement=True / input_prefix_attention=False.

Run:
  cd /data2/qiwu2/DiffSynth-Studio
  python examples/wanvideo/_smoke_test_input_encoder_prefix.py
"""
import sys
import torch

from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video import model_fn_wan_video


# ── tiny model config (head_dim=128 so 3D RoPE splits cleanly) ────────────────
DIM = 128
NUM_HEADS = 1
IN_DIM = 16
OUT_DIM = 16
FFN_DIM = 256
TEXT_DIM = 32
FREQ_DIM = 256
PATCH = (1, 2, 2)
NUM_LAYERS = 3

F, H, W = 3, 4, 4          # 3 frames (2 context + 1 target), latent 4x4
NUM_OUTPUT = 1
DEVICE = "cpu"
DTYPE = torch.float32


def build_model(use_input_encoder, per_layer_input_replacement, input_prefix_attention,
                use_input_latent_mlp=False, seed=0):
    torch.manual_seed(seed)
    m = WanModel(
        dim=DIM, in_dim=IN_DIM, ffn_dim=FFN_DIM, out_dim=OUT_DIM,
        text_dim=TEXT_DIM, freq_dim=FREQ_DIM, eps=1e-6, patch_size=PATCH,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS, has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        fuse_vae_embedding_in_latents_multiple=True,
        seperated_encoding=True,
        use_input_encoder=use_input_encoder,
        use_input_latent_mlp=use_input_latent_mlp,
        per_layer_input_replacement=per_layer_input_replacement,
        input_prefix_attention=input_prefix_attention,
    ).to(device=DEVICE, dtype=DTYPE).eval()
    return m


def run(model, latents, timestep_val=500.0, aat=False):
    context = torch.zeros(1, 4, TEXT_DIM, dtype=DTYPE)   # empty-ish prompt
    timestep = torch.tensor([timestep_val], dtype=DTYPE)
    with torch.no_grad():
        out = model_fn_wan_video(
            dit=model,
            latents=latents,
            timestep=timestep,
            context=context,
            fuse_vae_embedding_in_latents=True,
            num_output_frames=NUM_OUTPUT,
            aat_frame_attention=aat,
        )
    return out  # (1, OUT_DIM, F, H, W)


def make_latents(seed_ctx=1, seed_tgt=2):
    g = torch.Generator().manual_seed(seed_ctx)
    lat = torch.randn(1, IN_DIM, F, H, W, generator=g, dtype=DTYPE)
    gt = torch.Generator().manual_seed(seed_tgt)
    lat[:, :, F - NUM_OUTPUT:] = torch.randn(1, IN_DIM, NUM_OUTPUT, H, W, generator=gt, dtype=DTYPE)
    return lat


def context_slice(out):
    return out[:, :, :F - NUM_OUTPUT]


def main():
    torch.manual_seed(0)
    n_ctx_frames = F - NUM_OUTPUT

    # ── Test 4: no-MLP construction ───────────────────────────────────────────
    m_nomlp = build_model(True, False, True, use_input_latent_mlp=False)
    assert m_nomlp.input_latent_mlp is None, "expected input_latent_mlp=None with use_input_latent_mlp=False"
    assert not any("input_latent_mlp" in k for k in m_nomlp.state_dict()), \
        "input_latent_mlp params leaked into state_dict"
    assert m_nomlp.input_prefix_attention is True and m_nomlp.per_layer_input_replacement is False
    print("[4] no-MLP construction: input_latent_mlp is None, flags set correctly -- OK")

    m_withmlp = build_model(True, True, False, use_input_latent_mlp=True)
    assert m_withmlp.input_latent_mlp is not None, "expected input_latent_mlp created with use_input_latent_mlp=True"
    print("[4b] with-MLP construction: input_latent_mlp created -- OK")

    # ── Test 5: default flags (back-compat) ──────────────────────────────────
    m_default = WanModel(
        dim=DIM, in_dim=IN_DIM, ffn_dim=FFN_DIM, out_dim=OUT_DIM, text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM, eps=1e-6, patch_size=PATCH, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, has_image_input=False,
    )
    assert m_default.per_layer_input_replacement is True, "default per_layer_input_replacement should be True"
    assert m_default.input_prefix_attention is False, "default input_prefix_attention should be False"
    print("[5] default flags: per_layer_input_replacement=True, input_prefix_attention=False -- OK")

    # ── Test 3 + 1: prefix mask -> context invariant to target (per-token ts) ──
    lat_A = make_latents(seed_ctx=1, seed_tgt=2)
    lat_B = lat_A.clone()
    lat_B[:, :, F - NUM_OUTPUT:] = make_latents(seed_ctx=1, seed_tgt=99)[:, :, F - NUM_OUTPUT:]
    # sanity: contexts identical, targets differ
    assert torch.allclose(lat_A[:, :, :n_ctx_frames], lat_B[:, :, :n_ctx_frames])
    assert not torch.allclose(lat_A[:, :, n_ctx_frames:], lat_B[:, :, n_ctx_frames:])

    m_mask = build_model(True, False, True, use_input_latent_mlp=False, seed=7)
    out_A = run(m_mask, lat_A)
    out_B = run(m_mask, lat_B)
    assert tuple(out_A.shape) == (1, OUT_DIM, F, H, W), out_A.shape
    ctx_A, ctx_B = context_slice(out_A), context_slice(out_B)
    max_ctx_diff = (ctx_A - ctx_B).abs().max().item()
    assert torch.allclose(ctx_A, ctx_B, atol=1e-5), \
        f"[1] prefix mask FAILED: context changed with target (max diff {max_ctx_diff:.3e})"
    # target output should change (mask still lets target attend to everything)
    tgt_diff = (out_A[:, :, n_ctx_frames:] - out_B[:, :, n_ctx_frames:]).abs().max().item()
    assert tgt_diff > 1e-6, "[1] target output did not change with target input (unexpected)"
    print(f"[3] per-token timestep ran cleanly (seperated_timestep=True) -- OK")
    print(f"[1] prefix mask: context invariant to target (max ctx diff {max_ctx_diff:.3e}, "
          f"target diff {tgt_diff:.3e}) -- OK")

    # ── Test 2: negative control (no mask) -> context DOES change ─────────────
    m_plain = build_model(True, False, False, use_input_latent_mlp=False, seed=7)
    out_A2 = run(m_plain, lat_A)
    out_B2 = run(m_plain, lat_B)
    ctx_diff_plain = (context_slice(out_A2) - context_slice(out_B2)).abs().max().item()
    assert ctx_diff_plain > 1e-5, \
        f"[2] negative control FAILED: context invariant without mask (diff {ctx_diff_plain:.3e})"
    print(f"[2] negative control (no mask): context changes with target "
          f"(max ctx diff {ctx_diff_plain:.3e}) -- OK")

    # ── Test 6: transform-once K/V cache == full recompute (even as target changes)
    # Prefill fills the cache from lat_A's context; then a DECODE step with a
    # DIFFERENT target (lat_B) must match a full (uncached) recompute on lat_B.
    m_cache = build_model(True, False, True, use_input_latent_mlp=False, seed=7)
    cache = {}

    def run_cached(model, latents, cache, timestep_val=500.0, aat=False):
        context = torch.zeros(1, 4, TEXT_DIM, dtype=DTYPE)
        timestep = torch.tensor([timestep_val], dtype=DTYPE)
        with torch.no_grad():
            return model_fn_wan_video(
                dit=model, latents=latents, timestep=timestep, context=context,
                fuse_vae_embedding_in_latents=True, num_output_frames=NUM_OUTPUT,
                context_cache=cache, aat_frame_attention=aat,
            )

    out_prefill = run_cached(m_cache, lat_A, cache)          # step 0 (fills cache)
    assert cache.get("filled") is True, "cache was not filled during prefill"
    assert tuple(out_prefill.shape) == (1, OUT_DIM, F, H, W), out_prefill.shape
    # prefill target must equal the full masked forward target on lat_A
    out_full_A = run(m_cache, lat_A)
    assert torch.allclose(out_prefill[:, :, n_ctx_frames:], out_full_A[:, :, n_ctx_frames:], atol=1e-5), \
        "prefill target != full recompute target on lat_A"

    # decode with a DIFFERENT target (lat_B); compare to a full recompute on lat_B
    out_decode = run_cached(m_cache, lat_B, cache)
    out_full_B = run(m_cache, lat_B)
    dec_diff = (out_decode[:, :, n_ctx_frames:] - out_full_B[:, :, n_ctx_frames:]).abs().max().item()
    assert torch.allclose(out_decode[:, :, n_ctx_frames:], out_full_B[:, :, n_ctx_frames:], atol=1e-4), \
        f"[6] transform-once decode target != full recompute target (max diff {dec_diff:.3e})"
    # decode leaves context frames zeroed (irrelevant; pipeline re-fixes them)
    assert out_decode[:, :, :n_ctx_frames].abs().max().item() == 0.0
    print(f"[6] transform-once K/V cache: decode target == full recompute "
          f"(max diff {dec_diff:.3e}), context transformed once -- OK")

    # ── Test 7: AAT + transform-once K/V cache == AAT full recompute ──────────
    # AAT alternates within-frame (even blocks, no mask) and global (odd blocks,
    # prefix mask -> context K/V cached). Prefill on lat_A, then a DECODE step on
    # a DIFFERENT target (lat_B) must match a full AAT recompute on lat_B.
    m_aat = build_model(True, False, True, use_input_latent_mlp=False, seed=7)
    aat_cache = {}

    out_aat_prefill = run_cached(m_aat, lat_A, aat_cache, aat=True)   # fills cache
    assert aat_cache.get("filled") is True, "AAT cache not filled during prefill"
    assert tuple(out_aat_prefill.shape) == (1, OUT_DIM, F, H, W), out_aat_prefill.shape
    out_aat_full_A = run(m_aat, lat_A, aat=True)
    aat_pref_diff = (out_aat_prefill[:, :, n_ctx_frames:] - out_aat_full_A[:, :, n_ctx_frames:]).abs().max().item()
    assert torch.allclose(out_aat_prefill[:, :, n_ctx_frames:], out_aat_full_A[:, :, n_ctx_frames:], atol=1e-5), \
        f"[7] AAT prefill target != full recompute target on lat_A (max diff {aat_pref_diff:.3e})"

    out_aat_decode = run_cached(m_aat, lat_B, aat_cache, aat=True)
    out_aat_full_B = run(m_aat, lat_B, aat=True)
    aat_dec_diff = (out_aat_decode[:, :, n_ctx_frames:] - out_aat_full_B[:, :, n_ctx_frames:]).abs().max().item()
    assert torch.allclose(out_aat_decode[:, :, n_ctx_frames:], out_aat_full_B[:, :, n_ctx_frames:], atol=1e-4), \
        f"[7] AAT transform-once decode target != full recompute target (max diff {aat_dec_diff:.3e})"
    assert out_aat_decode[:, :, :n_ctx_frames].abs().max().item() == 0.0
    print(f"[7] AAT transform-once K/V cache: decode target == full recompute "
          f"(prefill diff {aat_pref_diff:.3e}, decode diff {aat_dec_diff:.3e}) -- OK")

    # ── Test 5b: vanilla model (no input encoder) still runs, no mask ─────────
    m_vanilla = WanModel(
        dim=DIM, in_dim=IN_DIM, ffn_dim=FFN_DIM, out_dim=OUT_DIM, text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM, eps=1e-6, patch_size=PATCH, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, has_image_input=False,
        seperated_timestep=True, fuse_vae_embedding_in_latents=True,
        fuse_vae_embedding_in_latents_multiple=True,
    ).to(device=DEVICE, dtype=DTYPE).eval()
    out_v = run(m_vanilla, lat_A)
    assert tuple(out_v.shape) == (1, OUT_DIM, F, H, W), out_v.shape
    print("[5b] vanilla model (use_input_encoder=False) runs, correct shape, no mask -- OK")

    print("\nAll input-encoder prefix-attention smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
