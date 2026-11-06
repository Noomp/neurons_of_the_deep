"""Architecture diagram of P, the action head on the frozen LLM (models/action_head.py).

Writes action_head_architecture.svg and .png next to this file. Parameter
counts are read from the models (the LLM is built on the meta device from its
config, so no weights are loaded), so re-run this after changing the
architecture, config.HEAD or config.LLM:

    python report/figures/action_head_architecture.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from config import HEAD, LLM
from models.action_head import ActionHead

OUT = Path(__file__).with_suffix("")

INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#ffffff"
# Same palette as symbolic_agent_architecture.py: one hue per part (border) with a light tint (fill).
INPUT = ("#2a78d6", "#e8f1fc")
FROZEN = ("#52514e", "#f1f0ed")
POOL = ("#1baf7a", "#e5f6ef")
TRAINED = ("#eb6834", "#fdeee7")
SKIPPED = ("#a8a7a3", "#ffffff")


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def llm_counts() -> tuple[int, int, int, int]:
    """(total, embedding, one decoder layer, number of layers) of config.LLM, built without weights."""
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(LLM.name)
    with torch.device("meta"):
        lm = AutoModel.from_config(cfg)
    return count(lm), count(lm.embed_tokens), count(lm.layers[0]), cfg.num_hidden_layers


def main():
    total, emb, per_layer, n_layers = llm_counts()
    d = 1024
    heads = {
        (p, h): ActionHead(replace(HEAD, pooling=p, hidden_layers=h), d).num_parameters()
        for p in ("last", "attention") for h in (1, 2)
    }
    attn = ActionHead(replace(HEAD, pooling="attention"), d)
    p_attn = count(attn.key) + attn.query.numel()
    p_norm = count(attn.norm)
    mlp = {h: count(ActionHead(replace(HEAD, hidden_layers=h), d).mlp) for h in (1, 2)}
    hd = HEAD.hidden_dim

    fig = Figure(figsize=(13, 9.6), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 130)
    ax.set_ylim(-4, 92)
    ax.axis("off")

    def box(cx, cy, w, h, title, sub="", colors=FROZEN, bold=False, dashed=False):
        edge, fill = colors
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                facecolor=fill, edgecolor=edge, linewidth=1.4, linestyle=(0, (4, 3)) if dashed else "solid",
            )
        )
        color = INK_2 if dashed else INK
        if sub:
            ax.text(cx, cy + 1.3, title, ha="center", va="center", fontsize=10.5, color=color, fontweight="bold" if bold else "normal")
            ax.text(cx, cy - 1.9, sub, ha="center", va="center", fontsize=8.8, color=INK_2)
        else:
            ax.text(cx, cy, title, ha="center", va="center", fontsize=10.5, color=color, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color=INK_2, label="", label_side="right", dashed=False):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3, shrinkA=0, shrinkB=0, mutation_scale=12,
                            linestyle=(0, (4, 3)) if dashed else "solid"),
        )
        if label:
            dx = 1.2 if label_side == "right" else -1.2
            ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, label, fontsize=8.3, color=INK_2, va="center",
                    ha="left" if label_side == "right" else "right", style="italic")

    ix, lx, rx = 18, 55, 103  # column centres: input, frozen LLM, head
    iw, lw, rw = 30, 36, 46

    for x, name, (edge, _) in ((ix, "Input", INPUT), (lx, f"Frozen LLM: Qwen3-0.6B ({n_layers} layers)", FROZEN),
                               (rx, "Hidden states → action head P", TRAINED)):
        ax.text(x, 89.5, name, ha="center", va="center", fontsize=11.5, color=edge, fontweight="bold")

    # Input: the GLAM prompt, shared with B1 and B2.
    box(ix, 83, iw, 6, "GLAM prompt (text)", "goal + last 3 observations + 2 actions", INPUT, bold=True)
    ax.add_patch(FancyBboxPatch((ix - iw / 2, 61.5), iw, 14.5, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=SURFACE, edgecolor=INPUT[0], linewidth=1.0, linestyle=(0, (2, 2))))
    excerpt = ("Possible action of the agent:\n turn left, turn right, …\n Goal of the agent: go to a grey key\n"
               " Observation 0: You see a grey key\n  1 step left and 5 steps forward, …\n Action 0: turn left\n"
               " …\n Observation 2: …\n Action 2: ▌")
    ax.text(ix - iw / 2 + 1.5, 75, excerpt, ha="left", va="top", fontsize=7.2, color=INK_2, family="monospace",
            linespacing=1.35)
    box(ix, 53, iw, 6, "Tokenizer", "→ (B, L) token IDs   ·   L ≈ 270-300", INPUT)
    arrow(ix, 80, ix, 76, INPUT[0])
    arrow(ix, 61.5, ix, 56, INPUT[0])
    arrow(ix + iw / 2, 53, lx - lw / 2, 81, INPUT[0])
    ax.text(ix, 46, "the same prompt that B1 and B2 read", ha="center", fontsize=8.6, color=INK_2, style="italic")

    # Frozen LLM, truncated at the layer the head reads.
    box(lx, 83, lw, 6, f"Token embedding ({emb // d:,} × {d})", f"→ (B, L, {d})   ·   {emb / 1e6:.1f}M params", FROZEN)
    box(lx, 72, lw, 6, "Decoder layer 1", f"self-attention + MLP   ·   {per_layer / 1e6:.1f}M params each", FROZEN)
    ax.text(lx, 65, "⋮", ha="center", va="center", fontsize=16, color=INK_2)
    box(lx, 58, lw, 7, "Decoder layer l", f"l ∈ {{{', '.join(map(str, HEAD.sweep_layers))}}}  (RQ2)", FROZEN, bold=True)
    box(lx, 45, lw, 7, f"Layers l+1 … {n_layers}  (+ final norm)", "not run: the LLM is truncated at l", SKIPPED,
        dashed=True)
    arrow(lx, 80, lx, 75, FROZEN[0])
    arrow(lx, 69, lx, 67.5, FROZEN[0])
    arrow(lx, 63, lx, 61.5, FROZEN[0])
    arrow(lx, 54.5, lx, 48.5, SKIPPED[0], dashed=True)
    ax.text(lx, 37.5, f"{total / 1e6:.0f}M parameters, none trained;\nlayer {n_layers}'s output includes the final norm",
            ha="center", va="center", fontsize=8.6, color=INK_2, style="italic")

    # Hidden states: live when acting, from the cache when training.
    box(rx, 58, rw, 7, f"Token vectors at layer l   (B, L, {d})", "acting: one live forward pass per decision", FROZEN,
        bold=True)
    arrow(lx + lw / 2, 58, rx - rw / 2, 58, FROZEN[0])
    box(rx, 76, rw, 7, "Hidden-state cache (training only)", "computed once for every demo step · data/hidden/",
        FROZEN, dashed=True)
    arrow(lx + lw / 2, 59.5, rx - rw / 2, 74.5, INK_2, dashed=True)
    ax.text(76.2, 70.5, "once", fontsize=8.3, color=INK_2, style="italic", ha="right")
    arrow(rx, 72.5, rx, 61.5, INK_2, label="training reads it", dashed=True)

    # Pooling: one of three.
    px = (rx - 15.5, rx, rx + 15.5)
    ax.text(rx, 50.2, f"pooling (one of three)  →  (B, {d})", ha="center", va="center", fontsize=8.8, color=POOL[0],
            fontweight="bold", bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.5), zorder=5)
    box(px[0], 42, 14.5, 7, "Last token", "0 params", POOL)
    box(px[1], 42, 14.5, 7, "Mean", "0 params", POOL)
    box(px[2], 42, 14.5, 7, "Attention", f"{p_attn:,} params", POOL)
    for x in px:
        arrow(rx + (x - rx) * 0.45, 54.5, x, 45.5, POOL[0])

    # The trained head.
    box(rx, 29, rw, 6, f"LayerNorm ({d})", f"{p_norm:,} params", TRAINED)
    for x in px:
        arrow(x, 38.5, rx + (x - rx) * 0.45, 32, POOL[0])
    box(rx, 17, rw, 7, f"Linear {d}→{hd} · ReLU · Linear {hd}→6", f"h2 adds Linear {hd}→{hd} · ReLU   ·   "
        f"{mlp[1]:,} / {mlp[2]:,} params", TRAINED, bold=True)
    arrow(rx, 26, rx, 20.5, TRAINED[0])
    box(rx, 4.5, 24, 6, "Action logits  (B, 6)", "argmax → action", TRAINED, bold=True)
    arrow(rx, 13.5, rx, 7.5, TRAINED[0])

    # Notes on the left.
    notes = (
        "Attention pooling: a key Linear 1024→64 scored\n"
        "against one learned query, softmax over tokens.\n"
        "Its LayerNorm is applied to each token first;\n"
        "for last / mean, to the pooled vector.\n\n"
        "Swept (val success): layer l × pooling × h1/h2;\n"
        "attention only at the best layers (12, 16, 20)."
    )
    for i, (label, (edge, fill)) in enumerate((("input", INPUT), ("frozen (never trained)", FROZEN),
                                                ("pooling", POOL), ("trained", TRAINED))):
        y = 33 - i * 3.2
        ax.add_patch(FancyBboxPatch((3, y - 0.9), 3.2, 1.8, boxstyle="round,pad=0,rounding_size=0.4",
                                    facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.text(7.5, y, label, va="center", fontsize=8.6, color=INK_2)
    ax.text(3, 18.5, notes, ha="left", va="top", fontsize=8.6, color=INK_2, linespacing=1.45)

    ax.text(1.5, -2.6,
            f"Trainable: {heads[('last', 1)]:,} (last / mean, h1) to {heads[('attention', 2)]:,} (attention, h2) parameters;"
            f" frozen LLM: {total / 1e6:.0f}M.   B = batch size, L = prompt length in tokens, l = layer read.",
            fontsize=8.6, color=INK_2, va="bottom")

    for ext in ("svg", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
        print(f"wrote {OUT.with_suffix(f'.{ext}')}")


if __name__ == "__main__":
    main()
