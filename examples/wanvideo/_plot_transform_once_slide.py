"""Client slide: prefix-mask 'transform-once' speedup vs the original model.

Left-top : frame-level prefix attention mask (why context is step-invariant).
Left-bot : end-to-end DiT compute, baseline vs transform-once, over denoising steps.
Right    : the two denoising pipelines drawn side by side.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

# ---- setup numbers (192x336, Wan2.1 VAE /8, patch 1x2x2) ----
TOK_PER_FRAME = 12 * 21          # 252
N_CTX = 6
N_TGT = 1
CTX_TOK = N_CTX * TOK_PER_FRAME  # 1512
TGT_TOK = N_TGT * TOK_PER_FRAME  # 252
S = CTX_TOK + TGT_TOK            # 1764

C_CTX = "#9aa7b4"   # context grey-blue
C_TGT = "#e8743b"   # target orange
C_CACHE = "#d8e2ec"  # cached (light)
C_BASE = "#c0392b"
C_OURS = "#2e86c1"

fig = plt.figure(figsize=(17, 8.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], height_ratios=[1, 1],
                      hspace=0.42, wspace=0.22)
ax_mask = fig.add_subplot(gs[0, 0])
ax_bar = fig.add_subplot(gs[1, 0])
ax_pipe = fig.add_subplot(gs[:, 1])

fig.suptitle("Encode the input views ONCE, not at every denoising step",
             fontsize=20, fontweight="bold", y=0.99)

# ================= Panel A: frame-level prefix mask =================
F = N_CTX + N_TGT
mask = np.ones((F, F))            # 1 = can attend
mask[:N_CTX, N_CTX:] = 0          # context rows cannot see the target col
ax_mask.imshow(mask, cmap="Greens", vmin=0, vmax=1.6, aspect="equal")
for i in range(F):
    for j in range(F):
        ok = mask[i, j] > 0
        ax_mask.text(j, i, "\u2713" if ok else "\u2717",
                     ha="center", va="center",
                     color=("#1e7d34" if ok else "#c0392b"),
                     fontsize=11, fontweight="bold")
ax_mask.set_xticks(range(F))
ax_mask.set_yticks(range(F))
labels = [f"ctx{i+1}" for i in range(N_CTX)] + ["TGT"]
ax_mask.set_xticklabels(labels, fontsize=9)
ax_mask.set_yticklabels(labels, fontsize=9)
ax_mask.set_xlabel("key (attended to)", fontsize=11)
ax_mask.set_ylabel("query", fontsize=11)
ax_mask.set_title("Prefix attention mask\ncontext never looks at the target "
                  "\u2192 context is identical every step",
                  fontsize=12, fontweight="bold")
ax_mask.tick_params(length=0)

# ================= Panel B: end-to-end compute vs steps =================
steps = np.arange(2, 51)
base = steps * S
ours = S + (steps - 1) * TGT_TOK       # 1 prefill + cheap decodes
speedup = base / ours
ax_bar.plot(steps, speedup, color=C_OURS, lw=3)
for n in (20, 30, 50):
    sp = (n * S) / (S + (n - 1) * TGT_TOK)
    ax_bar.plot(n, sp, "o", color=C_OURS, ms=7)
    ax_bar.annotate(f"{sp:.1f}\u00d7", (n, sp), textcoords="offset points",
                    xytext=(4, -14), fontsize=11, fontweight="bold", color=C_OURS)
ax_bar.axhline(1.0, color=C_BASE, lw=2, ls="--")
ax_bar.text(50, 1.0, " baseline (1\u00d7)", color=C_BASE, va="center",
            ha="right", fontsize=10, fontweight="bold")
ax_bar.set_xlabel("number of denoising steps", fontsize=11)
ax_bar.set_ylabel("DiT speedup", fontsize=11)
ax_bar.set_title("End-to-end DiT compute: up to ~7\u00d7 faster\n"
                 "(asymptote = 1764 / 252 tokens = 7\u00d7)",
                 fontsize=12, fontweight="bold")
ax_bar.grid(True, alpha=0.3)
ax_bar.set_ylim(0, 7.5)

# ================= Panel C: the two pipelines =================
ax_pipe.set_xlim(0, 10)
ax_pipe.set_ylim(0, 10)
ax_pipe.axis("off")

CELL_H = 0.22
CELL_GAP = 0.04
COL_H = (N_CTX + 1) * (CELL_H + CELL_GAP)   # full stack height ~1.82

def frame_col(ax, x, y0, states, w=0.5, h=CELL_H, gap=CELL_GAP):
    """Draw a vertical stack of frame cells; states: list of colors bottom->top."""
    for k, c in enumerate(states):
        ax.add_patch(Rectangle((x, y0 + k * (h + gap)), w, h,
                     facecolor=c, edgecolor="#33404d", lw=0.8))

def step_label(ax, x, y, txt, color, fs=9.5, weight="normal"):
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, color=color,
            fontweight=weight)

# ---- Baseline track (top) ----
yb = 7.7
ax_pipe.text(0.1, yb + COL_H + 0.5,
             "BASELINE  \u2014  recompute all 7 frames every step",
             fontsize=13, fontweight="bold", color=C_BASE)
xs = [0.9, 2.6, 4.3, 6.0]
for si, x in enumerate(xs):
    states = [C_CTX] * N_CTX + [C_TGT]
    frame_col(ax_pipe, x, yb, states)
    step_label(ax_pipe, x + 0.25, yb - 0.35, f"step {si+1}", "#33404d")
    step_label(ax_pipe, x + 0.25, yb - 0.68, "1764 tok", C_BASE, weight="bold")
    if si < len(xs) - 1:
        ax_pipe.add_patch(FancyArrowPatch((x + 0.55, yb + COL_H / 2),
                          (x + 1.35, yb + COL_H / 2), arrowstyle="->",
                          mutation_scale=13, color="#7f8c8d"))
ax_pipe.text(7.9, yb + COL_H / 2, ". . .", fontsize=20, va="center",
             color="#7f8c8d")
ax_pipe.text(9.25, yb + COL_H / 2, "\u00d7 N steps\n= N \u00d7 1764",
             fontsize=10.5, va="center", ha="center", color=C_BASE,
             fontweight="bold")

# ---- Transform-once track (bottom) ----
yo = 2.4
ax_pipe.text(0.1, yo + COL_H + 0.5,
             "TRANSFORM-ONCE  \u2014  encode context once, cache it, denoise target",
             fontsize=13, fontweight="bold", color=C_OURS)
# prefill (full)
x0 = 0.9
frame_col(ax_pipe, x0, yo, [C_CTX] * N_CTX + [C_TGT])
step_label(ax_pipe, x0 + 0.25, yo - 0.35, "step 1 (prefill)", "#33404d")
step_label(ax_pipe, x0 + 0.25, yo - 0.68, "1764 tok", "#33404d", weight="bold")
# cache box
cache_x = x0 + 0.95
ax_pipe.add_patch(Rectangle((cache_x, yo), 0.72, N_CTX * (CELL_H + CELL_GAP),
                  facecolor=C_CACHE, edgecolor=C_OURS, lw=1.4, ls="--"))
ax_pipe.text(cache_x + 0.36, yo + COL_H / 2, "cached\nctx K/V", ha="center",
             va="center", fontsize=8.5, color=C_OURS, fontweight="bold")
# decode steps: target only + dashed link to cache
xs2 = [3.4, 4.9, 6.4]
for si, x in enumerate(xs2):
    frame_col(ax_pipe, x, yo, [C_TGT])   # only the target frame
    step_label(ax_pipe, x + 0.25, yo - 0.35, f"step {si+2}", "#33404d")
    step_label(ax_pipe, x + 0.25, yo - 0.68, "252 tok", C_OURS, weight="bold")
    ax_pipe.add_patch(FancyArrowPatch((cache_x + 0.72, yo + COL_H / 2),
                      (x + 0.02, yo + CELL_H / 2), arrowstyle="->",
                      mutation_scale=10, color=C_OURS, ls="--", alpha=0.6))
ax_pipe.text(7.9, yo + COL_H / 2, ". . .", fontsize=20, va="center",
             color="#7f8c8d")
ax_pipe.text(9.25, yo + COL_H / 2, "\u00d7 N steps\n= 1764 + (N-1)\u00d7252",
             fontsize=10.0, va="center", ha="center", color=C_OURS,
             fontweight="bold")

# legend
ly = 0.35
ax_pipe.add_patch(Rectangle((0.9, ly), 0.32, 0.26, facecolor=C_CTX,
                  edgecolor="#33404d"))
ax_pipe.text(1.32, ly + 0.13, "context frame (6)", fontsize=9.5, va="center")
ax_pipe.add_patch(Rectangle((3.5, ly), 0.32, 0.26, facecolor=C_TGT,
                  edgecolor="#33404d"))
ax_pipe.text(3.92, ly + 0.13, "target frame (1, noisy)", fontsize=9.5, va="center")
ax_pipe.add_patch(Rectangle((6.5, ly), 0.32, 0.26, facecolor=C_CACHE,
                  edgecolor=C_OURS, ls="--"))
ax_pipe.text(6.92, ly + 0.13, "cached (computed once)", fontsize=9.5, va="center")

# exactness banner
fig.text(0.5, 0.015,
         "Exact, not an approximation: cached output is bit-identical to recompute "
         "(verified max error = 0.0)  \u2192  speedup with ZERO quality loss",
         ha="center", fontsize=13, fontweight="bold", color="#1e7d34",
         bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf0", ec="#1e7d34"))

fig.savefig("/data2/qiwu2/transform_once_slide.png", dpi=130,
            bbox_inches="tight")
print("wrote /data2/qiwu2/transform_once_slide.png")
