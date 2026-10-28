"""Architecture diagram of the B3 symbolic agent (models/symbolic_agent.py).

Writes symbolic_agent_architecture.svg and .png next to this file. Parameter
counts are read from the model, so re-run this after changing the architecture
or config.SYMBOLIC:

    python report/figures/symbolic_agent_architecture.py
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from models.symbolic_agent import SymbolicAgent

OUT = Path(__file__).with_suffix("")

INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#ffffff"
# One hue per branch (border) with a light tint (fill); the head is neutral.
VIEW = ("#2a78d6", "#e8f1fc")
MISSION = ("#eb6834", "#fdeee7")
ACTION = ("#1baf7a", "#e5f6ef")
HEAD = ("#52514e", "#f1f0ed")


def params(module) -> str:
    return f"{sum(p.numel() for p in module.parameters()):,} params"


def main():
    m = SymbolicAgent()
    p_cells = sum(sum(p.numel() for p in e.parameters()) for e in (m.obj_emb, m.color_emb, m.state_emb))
    p_block = [
        sum(p.numel() for p in m.convs[i].parameters()) + sum(p.numel() for p in m.norms[i].parameters())
        for i in range(len(m.convs))
    ]

    fig = Figure(figsize=(13, 9.6), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 130)
    ax.set_ylim(-4, 92)
    ax.axis("off")

    def box(cx, cy, w, h, title, sub="", colors=HEAD, bold=False):
        edge, fill = colors
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                facecolor=fill, edgecolor=edge, linewidth=1.4,
            )
        )
        if sub:
            ax.text(cx, cy + 1.3, title, ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold" if bold else "normal")
            ax.text(cx, cy - 1.9, sub, ha="center", va="center", fontsize=8.8, color=INK_2)
        else:
            ax.text(cx, cy, title, ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color=INK_2, label="", label_side="right"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3, shrinkA=0, shrinkB=0, mutation_scale=12),
        )
        if label:
            dx = 1.2 if label_side == "right" else -1.2
            ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, label, fontsize=8.3, color=INK_2, va="center",
                    ha="left" if label_side == "right" else "right", style="italic")

    vx, mx, ax_ = 25, 88, 118  # branch centre columns
    bw = 40  # main box width

    # Branch headings.
    for x, name, (edge, _) in ((vx, "View branch", VIEW), (mx, "Mission branch", MISSION), (ax_, "Action branch", ACTION)):
        ax.text(x, 89.5, name, ha="center", va="center", fontsize=11.5, color=edge, fontweight="bold")

    # Inputs.
    box(vx, 83, bw, 6, "Last 3 views  (B, 3, 7, 7, 3)", "cell = (object, colour, state)  +  view mask (B, 3)", VIEW, bold=True)
    box(mx, 83, 30, 6, "Mission  (B, L)", "word IDs", MISSION, bold=True)
    box(ax_, 83, 22, 6, "Previous actions  (B, 2)", "6 = none yet", ACTION, bold=True)

    # View branch.
    box(vx, 72, bw, 6, "Cell embedding (object + colour + state, summed)", f"→ (B, 3, 7, 7, 16)   ·   {p_cells:,} params", VIEW)
    box(vx, 61, bw, 6, "Zero missing views, stack along channels", "+ 3 “view present” channels  → (B, 51, 7, 7)", VIEW)
    box(vx, 48, bw, 7.5, "Conv 3×3 (51→64) · BatchNorm · FiLM · ReLU", f"→ (B, 64, 7, 7)   ·   {p_block[0]:,} params", VIEW, bold=True)
    box(vx, 35, bw, 7.5, "Conv 3×3 (64→64) · BatchNorm · FiLM · ReLU", f"→ (B, 64, 7, 7)   ·   {p_block[1]:,} params", VIEW, bold=True)
    box(vx, 24, bw, 6, "Flatten (keeps positions)", "→ (B, 3136)", VIEW)
    for y1, y2 in ((80, 75), (69, 64), (58, 51.75), (44.25, 38.75), (31.25, 27)):
        arrow(vx, y1, vx, y2, VIEW[0])

    # Mission branch.
    box(mx, 72, 30, 6, "Word embedding (20 × 32)", f"→ (B, L, 32)   ·   {params(m.word_emb)}", MISSION)
    box(mx, 61, 30, 6, "GRU (32 → 128)", f"final hidden state   ·   {params(m.mission_rnn)}", MISSION)
    box(mx, 41.5, 26, 6, "Mission vector m", "(B, 128)", MISSION, bold=True)
    arrow(mx, 80, mx, 75, MISSION[0])
    arrow(mx, 69, mx, 64, MISSION[0])
    arrow(mx, 58, mx, 44.5, MISSION[0])

    # FiLM generators: m -> (gamma, beta) for each conv block.
    fx = 59
    for i, y in enumerate((48, 35)):
        box(fx, y, 24, 6.5, f"FiLM {i + 1}: Linear 128 → 2×64", f"γ, β   ·   {params(m.films[i])}", MISSION)
        arrow(fx - 12, y, vx + bw / 2, y, MISSION[0])
        arrow(mx - 13, 41.5 + (1.5 if i == 0 else -1.5), fx + 12, y, MISSION[0])
    ax.text(fx, 53.8, "scale & shift each channel:  x · (1 + γ) + β", ha="center", fontsize=8.6, color=INK_2, style="italic")

    # Action branch.
    box(ax_, 72, 22, 6, "Action embedding (7 × 8)", f"→ (B, 2, 8)   ·   {params(m.action_emb)}", ACTION)
    box(ax_, 61, 22, 6, "Flatten", "→ (B, 16)", ACTION)
    arrow(ax_, 80, ax_, 75, ACTION[0])
    arrow(ax_, 69, ax_, 64, ACTION[0])

    # Head.
    hx = 72
    box(hx, 14, 56, 6, "Concatenate  →  (B, 3280)", "3136 (views)  +  128 (mission)  +  16 (actions)", HEAD, bold=True)
    arrow(vx, 21, hx - 22, 17, VIEW[0])
    arrow(mx, 38.5, mx, 17, MISSION[0])
    arrow(ax_, 58, hx + 26, 17, ACTION[0])
    box(hx, 4.5, 56, 6, "Linear 3280 → 128 · ReLU · Linear 128 → 6", f"{params(m.head)}", HEAD, bold=True)
    arrow(hx, 11, hx, 7.5, HEAD[0])
    box(118, 4.5, 20, 6, "Action logits  (B, 6)", "argmax → action", HEAD, bold=True)
    arrow(hx + 28, 4.5, 108, 4.5, HEAD[0])

    ax.text(1.5, -2.6, f"Total: {m.num_parameters():,} trainable parameters.   B = batch size, L = mission length.",
            fontsize=8.6, color=INK_2, va="bottom")

    for ext in ("svg", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
        print(f"wrote {OUT.with_suffix(f'.{ext}')}")


if __name__ == "__main__":
    main()
