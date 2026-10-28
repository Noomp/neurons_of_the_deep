"""Summary charts of the demonstrations and of the eval sets.

Writes to ``--out``:

* ``demo_stats.png``: expert episode lengths, expert action mix and how
  often each object is named in training instructions (held-out must be 0);
* ``eval_set_stats.png``: observation complexity per eval set (visible
  objects per observation, prompt length in words), from expert rollouts;
* ``demo_stats.json`` / ``eval_set_stats.csv``: the numbers behind the charts.

Usage:
    python -m babytext.tools.collect_demos          # first
    python -m babytext.tools.dataset_stats
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

from babytext import ACTION_NAMES, EVAL_SETS, TASKS, ExpertPolicy, evaluate  # noqa: E402
from babytext.constants import COLORS, OBJ_TYPES  # noqa: E402
from babytext.data import load_episodes  # noqa: E402

import config  # noqa: E402
from babytext.viz import INK, INK_MUTED, INK_SECONDARY, SURFACE  # noqa: E402

# Categorical slots 1-4 of the project palette, one per task in TASKS order
# (fixed: a task keeps its colour in every chart).
TASK_COLORS = {"goto": "#2a78d6", "pickup": "#eb6834", "putnext": "#1baf7a", "pickup_then_goto": "#eda100"}
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
        named = collections.Counter(tuple(obj) for e in eps for obj in e["goal"])
        stats[task] = {
            "episodes": len(eps),
            "steps": sum(e["length"] for e in eps),
            "success_rate": float(np.mean([e["success"] for e in eps])),
            "length_counts": dict(sorted(lengths.items())),
            "action_counts": {ACTION_NAMES[a]: actions.get(a, 0) for a in range(len(ACTION_NAMES))},
            "instruction_object_counts": {f"{c} {t}": named.get((t, c), 0) for t in OBJ_TYPES for c in COLORS},
        }

    fig = plt.figure(figsize=(18, 8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.75], hspace=0.28, wspace=0.3)

    # Episode length distribution: one line per task (bars would need ~100).
    ax = fig.add_subplot(gs[0, :2])
    for task in TASKS:
        counts = stats[task]["length_counts"]
        xs = np.arange(1, max(counts) + 1)
        ys = [100 * counts.get(int(x), 0) / stats[task]["episodes"] for x in xs]
        ax.plot(xs, ys, color=TASK_COLORS[task], linewidth=2, marker="o", markersize=4, label=task)
    # Clip the rare long tail (> 99.5th percentile over all tasks); numbers in demo_stats.json.
    all_lengths = [L for st in stats.values() for L, n in st["length_counts"].items() for _ in range(n)]
    ax.set_xlim(0.5, np.percentile(all_lengths, 99.5) + 0.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    style_axis(ax, "Expert episode length (training demos)", "steps to success", "% of episodes")

    # Action mix.
    ax = fig.add_subplot(gs[0, 2:])
    grouped_bars(
        ax,
        list(ACTION_NAMES),
        {t: [100 * stats[t]["action_counts"][a] / stats[t]["steps"] for a in ACTION_NAMES] for t in TASKS},
        horizontal=True,
    )
    style_axis(ax, "Expert action mix", "% of steps")

    # Instruction-object frequency heatmaps, one per task.
    for k, task in enumerate(TASKS):
        ax = fig.add_subplot(gs[1, k])
        counts = np.array([[stats[task]["instruction_object_counts"][f"{c} {t}"] for c in COLORS] for t in OBJ_TYPES])
        ax.imshow(counts, cmap=SEQ_BLUE, vmin=0)
        for i, t in enumerate(OBJ_TYPES):
            for j, c in enumerate(COLORS):
                held = (t, c) in config.ENV.held_out_goals
                v = counts[i, j]
                label = "held\nout" if held and v == 0 else str(v)
                dark = v > 0.6 * counts.max()
                ax.text(j, i, label, ha="center", va="center", fontsize=7.5, color="white" if dark else (INK_MUTED if held else INK))
        ax.set_xticks(range(len(COLORS)), COLORS, rotation=30)
        ax.set_yticks(range(len(OBJ_TYPES)), OBJ_TYPES)
        style_axis(ax, f"Objects named in '{task}'")
        for spine in ax.spines.values():
            spine.set_visible(False)

    summary = "   |   ".join(
        f"{t}: {stats[t]['episodes']} episodes, {stats[t]['steps']} steps, expert success {stats[t]['success_rate']:.0%}" for t in TASKS
    )
    fig.text(0.01, 0.005, "Training demonstrations - " + summary, fontsize=8.5, color=INK_SECONDARY)

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
    fig, axes = plt.subplots(1, 2, figsize=(18, 4.8), facecolor=SURFACE)
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
    parser.add_argument("--demos", type=Path, default=config.DATA.demos_dir)
    parser.add_argument("--n-eval", type=int, default=config.EVAL.n_stats_episodes, help="expert episodes per eval set for complexity stats")
    parser.add_argument("--out", type=Path, default=config.DATA.outputs_dir / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    demo_stats(args.demos, args.out)
    eval_set_stats(args.n_eval, args.out)


if __name__ == "__main__":
    main()
