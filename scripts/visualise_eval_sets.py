"""Side-by-side initial states of every eval set (one figure per task).

Shows what each distribution shift changes: the room layout, the goal and
the text the model sees. ``test``, ``test_nonce`` and ``test_heldout`` share
the test seeds; ``test_nonce`` has exactly the same layout as ``test`` with
renamed colours and objects.

Usage:
    python scripts/visualise_eval_sets.py --seed-index 0
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from babytext import EVAL_SETS, TASKS, get_seeds, make_env  # noqa: E402
from babytext.viz import INK, INK_SECONDARY, SURFACE, TILE, capture  # noqa: E402

CAPTIONS = {
    "train": "train (8 distractors)",
    "val": "val (8 distractors)",
    "test": "test (8 distractors)",
    "test_dist16": "shift: 16 distractors",
    "test_nonce": "shift: invented names",
    "test_heldout": "shift: held-out goal objects",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        fig, axes = plt.subplots(2, 6, figsize=(21, 8.4), facecolor=SURFACE, gridspec_kw={"height_ratios": [1, 0.95]})
        for col, eval_set in enumerate(EVAL_SETS):
            env = make_env(task, eval_set)
            seed = get_seeds(task, eval_set, args.seed_index + 1)[-1]
            obs, info = env.reset(seed=seed)
            frame = capture(env, obs, info)

            ax_img, ax_txt = axes[0, col], axes[1, col]
            ax_img.imshow(frame.full)
            for x, y in frame.goal_positions:
                ax_img.add_patch(Rectangle((x * TILE + 1.5, y * TILE + 1.5), TILE - 3, TILE - 3, fill=False, edgecolor="white", linewidth=2))
            ax_img.set_title(f"{CAPTIONS[eval_set]}\nseed {seed}", fontsize=10, color=INK)
            ax_img.axis("off")

            ax_txt.axis("off")
            ax_txt.text(0, 1, textwrap.fill(f"Goal: {frame.mission}", 34), fontsize=9.5, fontweight="bold", color=INK, va="top")
            lines = [textwrap.fill(d.replace("You see a ", ""), 34, subsequent_indent="  ") for d in frame.descriptions]
            ax_txt.text(0, 0.9, "\n".join(lines), fontsize=8, color=INK_SECONDARY, va="top", family="monospace")
            env.close()

        fig.suptitle(
            f"Eval sets for '{task}' (initial state; white outline = goal; text = descriptions without 'You see a')",
            fontsize=12, color=INK, x=0.01, ha="left",
        )
        fig.tight_layout()
        path = args.out / f"eval_sets_{task}.png"
        fig.savefig(path, dpi=110, facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
