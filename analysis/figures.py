"""The report's result figures, from the evaluation cells, the failure analysis and the training runs.

    python -m analysis.figures       # -> report/figures/results_*.png and .svg (after analysis.failures)

Figures (each a PNG at 300 dpi and an SVG with live text, sized for an A4
single-column page, Arial):

* ``results_success``: success on the four test sets per condition, both tiers;
* ``results_gaps``: generalisation gap (shifted minus ``test``) of the trained conditions;
* ``results_demo_curve``: success on ``test`` against demonstrations per task (B3 vs P);
* ``results_cost``: success on ``test`` against latency per decision;
* ``results_layers``: val success of P by layer and pooling (easy tier; selection data, not test);
* ``results_failures``: failure categories per condition and test set (``analysis.failures``);
* ``results_confusion``: action confusion against the expert on ``test``;
* ``results_drift``: agreement with the expert by step, successful vs failed episodes;
* ``results_examples``: one failed episode per row, a few frames each.

Every condition keeps one colour in every figure (``COLOURS``); intervals are
the evaluation's 95% intervals (``evaluation.summarise``).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from babytext import ACTION_NAMES, make_env  # noqa: E402
from babytext.constants import TASK_TIERS  # noqa: E402
from babytext.viz import outline_goals, replay  # noqa: E402
from evaluation.record import read_episodes  # noqa: E402
from evaluation.summarise import gap_interval, group_runs, load_cells, mean_interval, success_interval  # noqa: E402

from .failures import CATEGORIES, category_shares, cell_prefix  # noqa: E402

__all__ = ["COLOURS", "LABELS", "main"]

# Chart tokens (light theme of the project's palette; see training/tools/dataset_stats.py).
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#ffffff"
# One colour per condition. Neighbouring bars are neighbouring slots of the validated categorical order
# (magenta, yellow, aqua, orange, blue = slots 5..1), so adjacent marks stay distinguishable under colour blindness.
COLOURS = {"random": "#a8a7a3", "B1": "#e87ba4", "B2": "#eda100", "B3": "#1baf7a", "P-L12-attn-h1": "#eb6834",
           "P-L28-attn-h1": "#2a78d6"}
LABELS = {"random": "random", "B1": "B1 generate", "B2": "B2 score", "B3": "B3 symbolic", "P-L12-attn-h1": "P (L12, attn)",
          "P-L28-attn-h1": "P (L28, attn)"}
SET_LABELS = {"test": "test", "test_dist16": "16 distractors", "test_nonce": "invented words",
              "test_heldout": "held-out goals"}
CATEGORY_LABELS = {"wrong_object": "picked up a wrong object", "first_step_only": "first step only",
                   "reached_no_pickup": "reached, never picked up", "stuck": "stuck in place",
                   "never_saw_goal": "never saw the goal", "wandered": "wandered"}
# Stacked segments. The three large categories take the first three slots (blue, aqua, orange), which the palette
# validates for every pair; the rare ones take violet and magenta, and the residual "wandered" a neutral grey, so that
# no segment that can touch another is the yellow-orange pair.
CATEGORY_COLOURS = dict(zip(CATEGORIES, ["#2a78d6", "#1baf7a", "#6250d6", "#eb6834", "#e87ba4", "#8a8985"]))
SEQ_BLUE = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
WIDTH = 6.3  # inches: A4 text width with 2.5 cm margins
TRAINED = ("B3", "P-L12-attn-h1", "P-L28-attn-h1")


def style() -> None:
    """Arial at print sizes, recessive axes."""
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": INK_3, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlecolor": INK,
        "axes.titlelocation": "left", "legend.frameon": False, "svg.fonttype": "none", "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })


def grid_y(ax) -> None:
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def add_legend(fig, axes, ncol: int | None, y: float) -> None:
    """One legend below the panels, each label once (in first-drawn order); ``ncol=None``: one row."""
    handles = {}
    for ax in np.atleast_1d(axes):
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=ncol or len(handles),
                   bbox_to_anchor=(0.5, y))


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png/.svg")


def models_in(groups: dict, tier: str, wanted) -> list[str]:
    return [m for m in wanted if (m, tier) in groups]


def fig_success(groups: dict, out: Path) -> None:
    """Grouped bars: success per eval set, one bar per condition, a panel per tier."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (tier, tasks) in zip(axes, TASK_TIERS.items()):
        models = models_in(groups, tier, COLOURS)
        if not models:  # tier not evaluated
            ax.set_visible(False)
            continue
        width = 0.8 / len(models)
        for k, m in enumerate(models):
            for j, s in enumerate(config.EVAL.test_sets):
                iv = success_interval(groups[(m, tier)], tasks, s)
                if iv is None:
                    continue
                x = j - 0.4 + width * (k + 0.5)
                ax.bar(x, iv.value, width * 0.86, color=COLOURS[m], label=LABELS[m])
                if iv.low is not None:
                    ax.errorbar(x, iv.value, [[iv.value - max(iv.low, 0)], [iv.high - iv.value]], color=INK, lw=0.7,
                                capsize=0)
        ax.set_xticks(range(len(config.EVAL.test_sets)), [SET_LABELS[s].replace(" ", "\n", 1) if s != "test" else s
                                                          for s in config.EVAL.test_sets])
        ax.set_title(f"{tier} tier ({', '.join(tasks)})")
        ax.set_ylim(0, 1)
        grid_y(ax)
    axes[0].set_ylabel("success rate (test)")
    add_legend(fig, axes, None, -0.13)
    save(fig, out, "results_success")


def fig_gaps(groups: dict, out: Path) -> None:
    """Horizontal bars from zero: generalisation gap with 95% interval, one labelled row per trained condition,
    grouped under a heading per shifted set (the layout of ``results_failures``), a panel per tier."""
    shifted = [s for s in config.EVAL.test_sets if s != "test"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharex=True, gridspec_kw={"wspace": 0.42})
    for ax, (tier, tasks) in zip(axes, TASK_TIERS.items()):
        models = models_in(groups, tier, TRAINED)
        labels, y = [], 0
        for s in shifted:
            rows = [(m, gap_interval(groups[(m, tier)], tasks, s)) for m in models]
            rows = [(m, iv) for m, iv in rows if iv is not None]
            if not rows:
                continue
            # Set heading in the gap above its block of rows (left of zero, where the bars are).
            ax.text(-0.69, y - 0.75, SET_LABELS[s], fontsize=7.5, color=INK, fontweight="bold", va="center")
            for m, iv in rows:
                xerr = [[iv.value - iv.low], [iv.high - iv.value]] if iv.low is not None else None  # one seed: none
                ax.barh(y, iv.value, height=0.72, color=COLOURS[m])
                if xerr is not None:
                    ax.errorbar(iv.value, y, xerr=xerr, fmt="none", ecolor=INK, elinewidth=0.7)
                labels.append((y, LABELS[m]))
                y += 1
            y += 1.0
        ax.axvline(0, color=INK_3, lw=0.8)
        ax.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=7)
        ax.set_ylim(y - 1.4, -1.3)  # inverted: first block on top, room for its heading
        ax.set_xlim(-0.7, 0.1)
        ax.set_title(f"{tier} tier", pad=6)
        ax.set_xlabel("change in success from test (left = drop)")
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    save(fig, out, "results_gaps")


def fig_demo_curve(groups: dict, out: Path, random_floor: float | None) -> None:
    """Success on ``test`` against demonstrations per task, with 95% bands (easy tier)."""
    tasks = TASK_TIERS["easy"]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.48, 2.2))
    for base in ("B3", "P-L12-attn-h1"):
        counts, ivs = [], []
        for n in config.TRAIN.demo_counts:
            m = base if n == config.DATA.n_train_demos else f"{base}-n{n}"
            iv = success_interval(groups.get((m, "easy"), {}), tasks, "test")
            if iv is not None and iv.low is not None:
                counts.append(n)
                ivs.append(iv)
        if not counts:
            continue
        ax.fill_between(counts, [i.low for i in ivs], [i.high for i in ivs], color=COLOURS[base], alpha=0.18, lw=0)
        ax.plot(counts, [i.value for i in ivs], color=COLOURS[base], lw=2, marker="o", ms=4, mec=SURFACE, mew=0.8)
        ax.annotate(LABELS[base], (counts[0], ivs[0].value), xytext=(-4, 6), textcoords="offset points", ha="left",
                    color=INK, fontsize=7.5)
    if random_floor is not None:
        ax.axhline(random_floor, color=INK_3, lw=0.8, ls=(0, (3, 2)))
        ax.annotate("random", (config.TRAIN.demo_counts[-1], random_floor), xytext=(0, 3), textcoords="offset points",
                    ha="right", color=INK_2, fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks(config.TRAIN.demo_counts, [str(n) for n in config.TRAIN.demo_counts])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("demonstrations per task")
    ax.set_ylabel("success rate (easy, test)")
    grid_y(ax)
    save(fig, out, "results_demo_curve")


def fig_cost(groups: dict, root: Path, out: Path) -> None:
    """Success on ``test`` against median latency per decision (log scale), a panel per tier."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (tier, tasks) in zip(axes, TASK_TIERS.items()):
        for m in models_in(groups, tier, [m for m in COLOURS if m != "random"]):
            runs = groups[(m, tier)]
            iv = success_interval(runs, tasks, "test")
            ms = np.concatenate([e["ms"] for v in runs.values() for (t, s), c in v.items() if s == "test"
                                 for e in read_episodes(cell_prefix(root, c["identity"]))])
            x = float(np.median(ms))
            yerr = [[iv.value - max(iv.low, 0)], [iv.high - iv.value]] if iv.low is not None else None
            ax.errorbar(x, iv.value, yerr=yerr, fmt="o", ms=6, color=COLOURS[m], mec=SURFACE, mew=0.8, lw=1)
            # Label beside the point; B1/B2 sit on the x axis, so above it; P (L12) left of P (L28) on the hard tier.
            dx, dy, ha = {"B1": (2, 8, "left"), "B2": (-2, 8, "right")}.get(m, (7, 0, "left"))
            if m == "P-L12-attn-h1" and tier == "hard":
                dx, ha = -7, "right"
            ax.annotate(f"{LABELS[m]}\n{x:.1f} ms", (x, iv.value), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7, color=INK, va="bottom" if dy else "center", ha=ha)
        ax.set_xscale("log")
        ax.set_xlim(0.6, 900)
        ax.set_xticks([1, 10, 100], ["1", "10", "100"])
        ax.set_ylim(-0.03, 1)
        ax.set_title(f"{tier} tier")
        ax.set_xlabel("median latency per decision (ms, log scale)")
        grid_y(ax)
    axes[0].set_ylabel("success rate (test)")
    save(fig, out, "results_cost")


def layer_val(runs_dir: Path, tier: str = "easy") -> dict[tuple[str, int], object]:
    """Val success over seeds of every finished ``P-L<l>-<pooling>-h1`` run: ``{(pooling, layer): Interval}``."""
    by = defaultdict(list)
    for path in sorted(runs_dir.glob(f"P-L*-*-h1/{tier}/seed*/run.json")):
        match = re.fullmatch(r"P-L(\d+)-(last|mean|attn)-h1", path.parts[-4])
        info = json.loads(path.read_text(encoding="utf-8"))
        if match and info.get("finished"):
            by[(match[2], int(match[1]))].append(info["selected"]["val_success"])
    return {k: mean_interval(v) for k, v in by.items()}


def fig_layers(runs_dir: Path, out: Path, random_val: float | None) -> None:
    """Val success of P by layer for each pooling (MLP with one hidden layer), easy tier."""
    data = layer_val(runs_dir)
    colours = {"last": "#2a78d6", "attn": "#eb6834", "mean": "#1baf7a"}  # slots 1, 2, 3
    names = {"last": "last token", "attn": "learned attention", "mean": "mean over tokens"}
    fig, ax = plt.subplots(figsize=(WIDTH * 0.48, 2.2))
    for pooling in ("last", "attn", "mean"):
        layers = sorted(l for p, l in data if p == pooling)
        if not layers:
            continue
        ivs = [data[(pooling, l)] for l in layers]
        ax.errorbar(layers, [i.value for i in ivs], yerr=[[i.value - i.low for i in ivs], [i.high - i.value for i in ivs]],
                    color=colours[pooling], lw=2 if len(layers) > 2 else 0, elinewidth=0.8, marker="o", ms=4,
                    mec=SURFACE, mew=0.8)
        # Direct label at the line's right end, in place of a legend.
        ax.annotate(names[pooling], (layers[-1], ivs[-1].value), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    if random_val is not None:
        ax.hlines(random_val, 2, 29, color=INK_3, lw=0.8, ls=(0, (3, 2)))
        ax.annotate("random", (28, random_val), xytext=(7, 0), textcoords="offset points", va="center", color=INK_2,
                    fontsize=7)
    ax.set_xticks(config.HEAD.sweep_layers)
    ax.set_xlim(2, 40)  # room for the direct labels right of layer 28
    ax.spines["bottom"].set_bounds(2, 30)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Qwen3 layer read by the head")
    ax.set_ylabel("val success (easy)")
    grid_y(ax)
    save(fig, out, "results_layers")


def fig_failures(rows: list[dict], cells: list[dict], out: Path) -> None:
    """Stacked horizontal bars: share of all episodes failing in each category, per condition and test set."""
    shares = category_shares(rows, cells)
    used = [c for c in CATEGORIES if any(d["counts"][c] for d in shares.values())]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.9), sharex=True, gridspec_kw={"wspace": 0.42})
    for ax, tier in zip(axes, TASK_TIERS):
        labels, y = [], 0
        for s in config.EVAL.test_sets:
            if any((m, tier, s) in shares for m in config.ANALYSIS.models):
                # Eval-set heading in the gap above its block of rows.
                ax.text(0.005, y - 0.75, SET_LABELS[s], fontsize=7.5, color=INK, fontweight="bold", va="center")
            for m in config.ANALYSIS.models:
                d = shares.get((m, tier, s))
                if d is None:
                    continue
                left = 0.0
                for c in used:
                    w = d["shares"][c]
                    ax.barh(y, w, left=left, height=0.78, color=CATEGORY_COLOURS[c], edgecolor=SURFACE, lw=0.6,
                            label=CATEGORY_LABELS[c] if y == 0 else None)
                    left += w
                labels.append((y, f"{LABELS[m]}"))
                y += 1
            y += 1.0
        ax.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=7)
        ax.set_ylim(y - 1.4, -1.3)  # inverted: first block on top, room for its heading
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of episodes (bar length = failure rate)")
        ax.set_title(f"{tier} tier", pad=6)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    add_legend(fig, axes, 3, -0.1)
    save(fig, out, "results_failures")


def fig_confusion(tables: dict, out: Path) -> None:
    """Row-normalised action confusion (expert vs policy) on ``test`` for B3 and P, per tier."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    short = ["left", "right", "fwd", "pick", "drop", "toggle"]
    panels = [(m, t) for t in TASK_TIERS for m in ("B3", "P-L12-attn-h1") if f"{m}|{t}|test" in tables]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 1.9), gridspec_kw={"wspace": 0.12})
    for ax, (m, tier) in zip(np.atleast_1d(axes), panels):
        counts = np.array(tables[f"{m}|{tier}|test"]["confusion"], dtype=float)
        keep = [i for i in range(len(ACTION_NAMES)) if counts[i].sum() > 0]  # rows the expert uses (never toggle)
        mat = counts[keep] / counts[keep].sum(axis=1, keepdims=True)
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] >= 0.005:
                    ax.text(j, i, f"{mat[i, j]:.2f}"[1:] if mat[i, j] < 1 else "1", ha="center", va="center",
                            fontsize=6, color=SURFACE if mat[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(short)), short, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(keep)), [short[i] for i in keep], fontsize=6.5)
        if ax is not np.atleast_1d(axes)[0]:
            ax.set_yticklabels([])
        ax.set_title(f"{LABELS[m]}, {tier}", fontsize=7.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    np.atleast_1d(axes)[0].set_ylabel("expert's action")
    fig.text(0.5, -0.12, "policy's action in the same state (row-normalised)", ha="center", color=INK_2, fontsize=8)
    save(fig, out, "results_confusion")


def fig_drift(tables: dict, out: Path, max_t: int = 32) -> None:
    """Agreement with the expert at each step on ``test``: failed episodes (solid) vs successful ones (dotted)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, tier in zip(axes, TASK_TIERS):
        for m in ("B3", "P-L12-attn-h1"):
            a = tables.get(f"{m}|{tier}|test", {}).get("agreement_by_step")
            if not a:
                continue
            for kind, ls in (("failure", "-"), ("success", (0, (1, 1.5)))):
                # Only steps resting on at least 50 decisions, so the tail of rare long episodes is not plotted.
                pts = [(t, v) for t, (v, n) in enumerate(zip(a[kind], a[f"n_{kind}"])) if v is not None and n >= 50
                       and t < max_t]
                if pts:
                    t, v = zip(*pts)
                    ax.plot(t, v, color=COLOURS[m], lw=1.8 if kind == "failure" else 1.4, ls=ls,
                            label=f"{LABELS[m]}, {'failed' if kind == 'failure' else 'succeeded'}")
        ax.set_title(f"{tier} tier, test")
        ax.set_xlabel("step")
        ax.set_ylim(0, 1)
        grid_y(ax)
    axes[0].set_ylabel("agreement with the expert")
    add_legend(fig, axes, 2, -0.27)
    save(fig, out, "results_drift")


# Representative failures: (model, tier, eval set, task, category, wrong-pickup kind or stuck kind, caption).
EXAMPLES = [
    ("B3", "easy", "test", "pickup", "wrong_object", "lookalike", "B3 picks up a look-alike"),
    ("P-L12-attn-h1", "hard", "test", "putnext", "wrong_object", "other_instruction_object",
     "P (L12) picks up the target, not the object to move"),
    ("P-L12-attn-h1", "easy", "test_dist16", "goto", "stuck", "blocked", "P (L12), 16 distractors, walks into an object"),
    ("P-L28-attn-h1", "hard", "test_nonce", "pickup_then_goto", "stuck", "spinning",
     "P (L28), invented words, turns on the spot"),
]


def pick_example(rows: list[dict], model, tier, eval_set, task, category, kind) -> dict | None:
    """The first failed episode (lowest seed of the first training seed) matching the description."""
    for r in sorted(rows, key=lambda r: (r["run"], int(r["seed"]))):
        if (r["model"], r["tier"], r["eval_set"], r["task"], r["category"]) == (model, tier, eval_set, task, category) \
                and kind in (r["wrong_pickup_kind"], r["stuck_kind"]):
            return r
    return None


def picked_object(row: dict, episode: dict) -> str:
    """'<colour> <type>' of the object carried after the row's first wrong pickup (real names, also under nonce)."""
    env = make_env(row["task"], row["eval_set"])
    env.reset(seed=episode["seed"])
    for a in episode["actions"][: int(row["wrong_pickup_t"]) + 1]:
        env.step(a)
    obj = env.unwrapped.carrying
    env.close()
    return f"{obj.color} {obj.type}"


def fig_examples(rows: list[dict], root: Path, out: Path, n_frames: int = 6) -> None:
    """One failed episode per row: frames around its decisive moment, room view with goal objects outlined."""
    picked = [(e, pick_example(rows, *e[:6])) for e in EXAMPLES]
    picked = [(e, r) for e, r in picked if r is not None]
    if not picked:
        print("no failed episode matches EXAMPLES: results_examples not drawn")
        return
    fig, axes = plt.subplots(len(picked), n_frames, figsize=(WIDTH, 1.32 * len(picked)),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.55})
    for row, (e, r) in zip(np.atleast_2d(axes), picked):
        prefix = root / r["model"] / r["tier"] / r["run"] / f"{r['task']}.{r['eval_set']}"
        episode = next(x for x in read_episodes(prefix) if x["seed"] == int(r["seed"]))
        frames = replay(r["task"], r["eval_set"], episode["seed"], episode["actions"])
        t_key = int(r["wrong_pickup_t"]) if int(r["wrong_pickup_t"]) >= 0 else n_frames * 2
        caption = e[6]
        if int(r["wrong_pickup_t"]) >= 0:
            caption += f" (the {picked_object(r, episode)} at t={t_key})"
        idx = list(np.unique(np.linspace(0, min(t_key + 1, len(frames) - 2), n_frames - 1).round().astype(int)))
        idx += [len(frames) - 1]
        for ax, i in zip(row, idx + [None] * (n_frames - len(idx))):
            ax.axis("off")
            if i is None:
                continue
            f = frames[i]
            ax.imshow(f.full)
            outline_goals(ax, f.goal_positions, linewidth=1.0)
            label = f"t={f.t}: {ACTION_NAMES[f.action]}" if f.action is not None else f"t={f.t}: time limit"
            ax.set_title(label, fontsize=6.5, color=INK if f.action is not None else "#e34948", loc="center", pad=2)
        row[0].annotate(f"{caption}:  “{frames[0].mission}”", (0, 1), xycoords="axes fraction",
                        xytext=(0, 13), textcoords="offset points", fontsize=7.5, color=INK, fontweight="bold")
    save(fig, out, "results_examples")


def random_success(path: Path, tier: str, eval_set: str) -> float | None:
    """Random policy's mean success over the tier's tasks from ``babytext.tools.reference_baselines``' CSV."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        got = [float(r["success_rate"]) for r in csv.DictReader(f)
               if r["policy"] == "random" and r["eval_set"] == eval_set and r["task"] in TASK_TIERS[tier]]
    return float(np.mean(got)) if len(got) == len(TASK_TIERS[tier]) else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, default=config.DATA.outputs_dir / "eval", help="evaluation cells")
    parser.add_argument("--runs", type=Path, default=config.DATA.outputs_dir / "runs", help="training runs")
    parser.add_argument("--results", type=Path, default=config.DATA.outputs_dir / "results",
                        help="analysis.failures output and reference_baselines.csv")
    parser.add_argument("--out", type=Path, default=config.ANALYSIS.figures_dir)
    args = parser.parse_args(argv)
    style()
    cells = load_cells(args.eval)
    if not cells:
        parser.error(f"no evaluation cells under {args.eval}")
    groups = group_runs(cells)
    baselines = args.results / "reference_baselines.csv"
    fig_success(groups, args.out)
    fig_gaps(groups, args.out)
    fig_demo_curve(groups, args.out, success_interval(groups[("random", "easy")], TASK_TIERS["easy"], "test").value
                   if ("random", "easy") in groups else None)
    fig_cost(groups, args.eval, args.out)
    fig_layers(args.runs, args.out, random_success(baselines, "easy", "val"))

    failures = args.results / "failures.csv"
    if not failures.exists():
        print(f"{failures} not found: run `python -m analysis.failures` for the failure figures")
        return
    with open(failures, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    analysed = [c for c in cells if c["identity"]["model"] in config.ANALYSIS.models]
    fig_failures(rows, analysed, args.out)
    tables = json.loads((args.results / "failures.json").read_text(encoding="utf-8"))
    fig_confusion(tables, args.out)
    fig_drift(tables, args.out)
    fig_examples(rows, args.eval, args.out)


if __name__ == "__main__":
    main()
