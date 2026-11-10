"""Failure analysis (proposal 6.6), from the cells written by `evaluation.evaluate`.

    `python -m evaluation.failures --models B3 P-L12-attn-h1 B2 B1`

Every failed episode gets exactly one category: the first of these rules that holds. The agent's state is
(position, direction, carried object), from replaying `(task, eval set, seed, actions)`.

1. **wrong object**: the episode ends with the agent carrying an object the
   mission does not name. It is checked first because it is usually the
   cause of the other symptoms: the agent keeps pressing `pick up` with its
   hands full and spins in place.
2. **loop**: in its last `LOOP_WINDOW` states the agent is in at most
   `LOOP_MAX_STATES` distinct states: it spins in place, steps back and
   forth or circles a small square instead of making progress.
3. **drift**: the first disagreement with the expert falls in the first
   `DRIFT_EARLY` of the episode, and the agent agrees with the expert on
   less than `DRIFT_MAX_AGREEMENT` of the labelled steps from then on.
4. **other**: none of these, e.g. the right object dropped in the wrong
   place, a late mistake, or wandering that is neither confined nor early.

A rule must be common in failed episodes and rare in successful ones; otherwise it describes a habit,
not why the episode failed. (Since the behaviour also shows up in games that were won, doing it doesn't mean you lose)
When checked on B3 (50 episodes per cell), examples of some rules in contention for being chosen:

- wrong object: true for 72% of failures and 8% of successes (easy tier),
  60% and 0.4% (hard tier). Kept.
- "picked up, dropped or toggled when the expert would not": true for 71% of
  successful hard episodes. Dropped.
- drift: true for 65% of successful hard episodes, because agreeing with the
  expert does not predict success on the hard tier. Kept, but only meaningful
  on the easy tier.

Steps without an expert label (after a label error) count for no rule that
needs the expert.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

import config
from babytext import ACTION_NAMES, NUM_ACTIONS, make_env
from babytext.constants import TASK_TIERS
from babytext.viz import INK, INK_SECONDARY, SURFACE, replay, save_strip

from .evaluate import plan_runs
from .record import read_episodes

__all__ = ["CATEGORIES", "categorise", "confusion", "main", "replay_states"]

LOOP_WINDOW = 32  # states at the end of the episode
LOOP_MAX_STATES = 8  # distinct states within them, e.g. 4 cells x 2 directions
DRIFT_EARLY = 0.25  # share of the episode's steps
DRIFT_MAX_AGREEMENT = 0.5  # share of labelled steps after the first disagreement
EXAMPLES = 2  # filmstrips per model and category

CATEGORIES = ("wrong object", "loop", "drift", "other")
CONFUSION_CMAP = LinearSegmentedColormap.from_list("confusion", [SURFACE, "#cde2fb", "#3987e5", "#1c5cab"])


def replay_states(task: str, eval_set: str, episode: dict) -> list[tuple]:
    """The agent's (position, direction, carried (type, colour) or None) before each step and after the last."""
    env = make_env(task, eval_set)
    env.reset(seed=episode["seed"])
    level = env.unwrapped

    def state():
        carried = None if level.carrying is None else (level.carrying.type, level.carrying.color)
        return tuple(level.agent_pos), level.agent_dir, carried

    states = [state()]
    for action in episode["actions"]:
        env.step(action)
        states.append(state())
    env.close()
    return states


def categorise(task: str, eval_set: str, episode: dict) -> str:
    """The category of a failed episode (module docstring)."""
    states = replay_states(task, eval_set, episode)
    carried = states[-1][2]
    if carried is not None and list(carried) not in episode["goal"]:
        return "wrong object"
    if len(set(states[-LOOP_WINDOW:])) <= LOOP_MAX_STATES:
        return "loop"
    steps = list(zip(episode["actions"], episode["expert"]))
    first = next((i for i, (a, t) in enumerate(steps) if t is not None and a != t), None)
    if first is not None and first < DRIFT_EARLY * len(steps):
        if np.mean([a == t for a, t in steps[first:] if t is not None]) < DRIFT_MAX_AGREEMENT:
            return "drift"
    return "other"


def confusion(episodes: list[dict]) -> np.ndarray:
    """Counts of (expert action, policy action) over the expert-labelled steps."""
    counts = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=int)
    for e in episodes:
        for a, t in zip(e["actions"], e["expert"]):
            if t is not None:
                counts[t, a] += 1
    return counts


def load(models: list[str], tiers: list[str], root: Path) -> tuple[list[tuple[tuple, list[dict]]], list[Path]]:
    """The recorded cells as ((model, tier, run, task, eval set), episodes), and the prefixes of missing cells."""
    cells, missing = [], []
    for run in plan_runs(models, tiers, list(config.TRAIN.seeds), config.DATA.outputs_dir / "runs"):
        for task in TASK_TIERS[run.tier]:
            for eval_set in config.EVAL.test_sets:
                prefix = run.out_dir(root) / f"{task}.{eval_set}"
                if Path(f"{prefix}.json").exists():
                    cells.append(((run.model, run.tier, run.run, task, eval_set), read_episodes(prefix)))
                else:
                    missing.append(prefix)
    return cells, missing


def category_table(cells: list, failed: list) -> list[str]:
    """Markdown rows: each category's count and share of the failed episodes, per model, tier, task and eval set."""
    episodes, counts = Counter(), defaultdict(Counter)
    for (model, tier, _, task, eval_set), eps in cells:
        episodes[(model, tier, task, eval_set)] += len(eps)
    for (model, tier, _, task, eval_set), _, category in failed:
        counts[(model, tier, task, eval_set)][category] += 1
    lines = ["| model | tier | task | eval set | episodes | failed | " + " | ".join(CATEGORIES) + " |",
             "|---" * (6 + len(CATEGORIES)) + "|"]
    for key, n in episodes.items():
        c = counts[key]
        n_failed = sum(c.values())
        shares = [f"{c[k]} ({c[k] / n_failed:.0%})" if n_failed else "-" for k in CATEGORIES]
        lines.append(f"| {' | '.join(key)} | {n} | {n_failed} | " + " | ".join(shares) + " |")
    return lines


def confusion_table(counts: np.ndarray) -> list[str]:
    """Markdown rows of a row-normalised confusion matrix, with each expert action's number of steps."""
    lines = ["| expert \\ policy | steps | " + " | ".join(ACTION_NAMES) + " |", "|---" * (NUM_ACTIONS + 2) + "|"]
    for name, row in zip(ACTION_NAMES, counts):
        shares = [f"{v / row.sum():.2f}" if row.sum() else "-" for v in row]
        lines.append(f"| {name} | {row.sum():,} | " + " | ".join(shares) + " |")
    return lines


def save_confusion(counts: np.ndarray, path: Path, title: str) -> None:
    """Save a row-normalised confusion matrix as a PNG heatmap, each cell labelled with its share."""
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    fig = Figure(figsize=(6.4, 5.4), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.imshow(shares, cmap=CONFUSION_CMAP, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(shares):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(NUM_ACTIONS), ACTION_NAMES, rotation=30, ha="right", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(range(NUM_ACTIONS), [f"{n} ({t:,})" for n, t in zip(ACTION_NAMES, totals[:, 0])], fontsize=9,
                  color=INK_SECONDARY)
    ax.set_xlabel("policy action", color=INK_SECONDARY)
    ax.set_ylabel("expert action (labelled steps)", color=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="random, B1, B2 and/or training run IDs")
    parser.add_argument("--tiers", nargs="+", choices=sorted(TASK_TIERS), default=list(TASK_TIERS))
    parser.add_argument("--eval", type=Path, default=config.DATA.outputs_dir / "eval", help="evaluation cells")
    parser.add_argument("--out", type=Path, default=config.DATA.outputs_dir / "results" / "failures")
    args = parser.parse_args(argv)
    cells, missing = load(args.models, args.tiers, args.eval)
    for prefix in missing:
        print(f"skipped, not evaluated yet: {prefix}")
    if not cells:
        parser.error(f"no evaluation cells of {args.models} under {args.eval}")
    args.out.mkdir(parents=True, exist_ok=True)
    failed = [(key, e, categorise(key[3], key[4], e)) for key, eps in cells for e in eps if not e["success"]]

    out = ["# Failure analysis", "", "Every failed episode in exactly one category, the first rule that holds "
           "(`evaluation/failures.py`): **wrong object** (ends carrying an object the mission does not name), "
           f"**loop** (at most {LOOP_MAX_STATES} distinct states in the last {LOOP_WINDOW}), **drift** (first "
           f"disagreement with the expert in the first {DRIFT_EARLY:.0%} of the episode, then agreement below "
           f"{DRIFT_MAX_AGREEMENT:.0%}; meaningful on the easy tier only), **other**. Pooled over seeds.", "",
           "## Failure categories", ""]
    out += category_table(cells, failed)

    out += ["", "## Action confusion matrices", "",
            "Expert action (rows) against the policy's action (columns) over the expert-labelled steps, pooled over "
            "seeds, tasks and eval sets; each row sums to 1.", ""]
    for model, tier in dict.fromkeys(key[:2] for key, _ in cells):
        counts = confusion([e for key, eps in cells if key[:2] == (model, tier) for e in eps])
        png = f"confusion_{model}_{tier}.png"
        save_confusion(counts, args.out / png, f"{model}, {tier} tier")
        out += [f"### {model}, {tier} tier", ""] + confusion_table(counts) + ["", f"![]({png})", ""]

    out += ["## Example failures", ""]
    for model in dict.fromkeys(key[0] for key, _ in cells):
        for category in CATEGORIES:
            examples = [(key, e) for key, e, c in failed if key[0] == model and c == category][:EXAMPLES]
            for k, ((_, tier, run, task, eval_set), e) in enumerate(examples):
                title = f"{model} {tier} {run}, {task}/{eval_set} seed {e['seed']}: {category}"
                png = f"{model}_{category.replace(' ', '_')}_{k}.png"
                save_strip(replay(task, eval_set, e["seed"], e["actions"]), args.out / png, title)
                out += [f"**{title}** ({e['mission']})", "", f"![]({png})", ""]

    (args.out / "failures.md").write_text("\n".join(out), encoding="utf-8")
    print(f"{len(failed)} failed episodes in {len(cells)} cells; wrote {args.out / 'failures.md'}")


if __name__ == "__main__":
    main()
