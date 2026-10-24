"""Summary charts of the demonstrations and of the eval sets.

Writes to ``--out``:

* ``demo_stats.png``: expert episode lengths, expert action mix and the
  goal-object frequency in the training demos (held-out goals must be 0);
* ``eval_set_stats.png``: observation complexity per eval set (visible
  objects per observation, prompt length in words), from expert rollouts;
* ``demo_stats.json`` / ``eval_set_stats.csv``: the numbers behind the charts.

Usage:
    python scripts/collect_demos.py          # first
    python scripts/dataset_stats.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from babytext import ACTION_NAMES, EVAL_SETS, HELD_OUT_GOALS, TASKS, ExpertPolicy, evaluate  # noqa: E402
from babytext.constants import COLORS, OBJ_TYPES  # noqa: E402
from babytext.data import load_episodes  # noqa: E402
from babytext.viz import INK, INK_MUTED, INK_SECONDARY, SURFACE  # noqa: E402

# Categorical slots 1-2 of the project palette (one colour per task, fixed).
TASK_COLORS = {"goto": "#2a78d6", "pickup": "#eb6834"}
GRID = "#e4e3df"
# Sequential blue ramp (light -> dark) for counts.
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def style_axis(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_axisbelow(True)


def grouped_bars(ax, categories, series: dict[str, list[float]], horizontal=False):
    """Bars per category, one bar per task, with a surface-coloured gap."""
    n = len(series)
    width = 0.8 / n
    pos = np.arange(len(categories))
    for k, (task, values) in enumerate(series.items()):
        offset = pos - 0.4 + width * (k + 0.5)
        kwargs = dict(color=TASK_COLORS[task], edgecolor=SURFACE, linewidth=1.5, label=task)
        if horizontal:
            ax.barh(offset, values, height=width, **kwargs)
        else:
            ax.bar(offset, values, width=width, **kwargs)
    if horizontal:
        ax.set_yticks(pos, categories)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    else:
        ax.set_xticks(pos, categories)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)


def demo_stats(demos_dir: Path, out: Path):
    stats = {}
    for task in TASKS:
        eps = load_episodes(demos_dir / f"{task}_train")
        lengths = collections.Counter(e["length"] for e in eps)
        actions = collections.Counter(s["action"] for e in eps for s in e["steps"])
        goals = collections.Counter(tuple(e["goal"]) for e in eps)
        stats[task] = {
            "episodes": len(eps),
            "steps": sum(e["length"] for e in eps),
            "success_rate": float(np.mean([e["success"] for e in eps])),
            "length_counts": dict(sorted(lengths.items())),
            "action_counts": {ACTION_NAMES[a]: actions.get(a, 0) for a in range(len(ACTION_NAMES))},
            "goal_counts": {f"{c} {t}": goals.get((t, c), 0) for t in OBJ_TYPES for c in COLORS},
        }

    fig = plt.figure(figsize=(15, 8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.3, wspace=0.3)

    # Episode length distribution.
    ax = fig.add_subplot(gs[0, :2])
    # Fold the rare long tail (> 99.5th percentile over both tasks) into one "N+" bar.
    all_lengths = [L for s in stats.values() for L, n in s["length_counts"].items() for _ in range(n)]
    cap = int(np.percentile(all_lengths, 99.5)) + 1
    lengths = list(range(1, cap + 1))

    def pct(task, L):
        counts = stats[task]["length_counts"]
        n = counts.get(L, 0) if L < cap else sum(v for k, v in counts.items() if k >= cap)
        return 100 * n / stats[task]["episodes"]

    grouped_bars(ax, [str(L) for L in lengths[:-1]] + [f"{cap}+"], {t: [pct(t, L) for L in lengths] for t in TASKS})
    style_axis(ax, "Expert episode length (training demos)", "steps to success", "% of episodes")

    # Action mix.
    ax = fig.add_subplot(gs[0, 2])
    grouped_bars(
        ax,
        list(ACTION_NAMES),
        {t: [100 * stats[t]["action_counts"][a] / stats[t]["steps"] for a in ACTION_NAMES] for t in TASKS},
        horizontal=True,
    )
    style_axis(ax, "Expert action mix", "% of steps")

    # Goal frequency heatmaps, one per task.
    for k, task in enumerate(TASKS):
        ax = fig.add_subplot(gs[1, k])
        counts = np.array([[stats[task]["goal_counts"][f"{c} {t}"] for c in COLORS] for t in OBJ_TYPES])
        ax.imshow(counts, cmap=SEQ_BLUE, vmin=0)
        for i, t in enumerate(OBJ_TYPES):
            for j, c in enumerate(COLORS):
                held = (t, c) in HELD_OUT_GOALS
                v = counts[i, j]
                label = "held\nout" if held and v == 0 else str(v)
                dark = v > 0.6 * counts.max()
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if dark else (INK_MUTED if held else INK))
        ax.set_xticks(range(len(COLORS)), COLORS)
        ax.set_yticks(range(len(OBJ_TYPES)), OBJ_TYPES)
        style_axis(ax, f"Goal objects in '{task}' training demos")
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    lines = [f"{t}: {stats[t]['episodes']} episodes, {stats[t]['steps']} steps, expert success {stats[t]['success_rate']:.1%}" for t in TASKS]
    ax.text(0, 0.9, "Training demonstrations\n\n" + "\n".join(lines), fontsize=9.5, color=INK_SECONDARY, va="top", wrap=True)

    fig.savefig(out / "demo_stats.png", dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    with open(out / "demo_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {out / 'demo_stats.png'}")


def eval_set_stats(n_episodes: int, out: Path):
    rows = []
    for task in TASKS:
        for eval_set in EVAL_SETS:
            eps = evaluate(ExpertPolicy(), task, eval_set, n_episodes)["episodes"]
            steps = [s for e in eps for s in e.steps]
            rows.append(
                {
                    "task": task,
                    "eval_set": eval_set,
                    "visible_objects": float(np.mean([sum("wall" not in d and "carry" not in d for d in s.descriptions) for s in steps])),
                    "prompt_words": float(np.mean([len(s.prompt.split()) for s in steps])),
                    "expert_length": float(np.mean([e.length for e in eps])),
                }
            )

    with open(out / "eval_set_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sets = list(EVAL_SETS)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), facecolor=SURFACE)
    for ax, key, title, ylabel in (
        (axes[0], "visible_objects", "Objects described per observation", "mean objects in view"),
        (axes[1], "prompt_words", "GLAM prompt length", "mean words per prompt"),
    ):
        grouped_bars(ax, sets, {t: [r[key] for r in rows if r["task"] == t] for t in TASKS})
        style_axis(ax, title, ylabel=ylabel)
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle(f"Observation complexity by eval set (expert rollouts, {n_episodes} episodes each)", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "eval_set_stats.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / 'eval_set_stats.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos", type=Path, default=Path("data/demos"))
    parser.add_argument("--n-eval", type=int, default=200, help="expert episodes per eval set for complexity stats")
    parser.add_argument("--out", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    demo_stats(args.demos, args.out)
    eval_set_stats(args.n_eval, args.out)


if __name__ == "__main__":
    main()
