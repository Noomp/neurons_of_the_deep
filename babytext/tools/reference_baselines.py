"""Success of the expert (ceiling) and a uniform random agent (floor).

Evaluates both reference policies on every task and eval set with the fixed
seed lists and writes ``<out>/reference_baselines.csv`` and ``.md``.

Usage:
    python -m babytext.tools.reference_baselines --n-episodes 500
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from babytext import EVAL_SETS, TASKS, ExpertPolicy, RandomPolicy, evaluate

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-episodes", type=int, default=config.EVAL.n_episodes)
    parser.add_argument("--out", type=Path, default=config.DATA.outputs_dir / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in TASKS:
        for eval_set in EVAL_SETS:
            for name, policy in (("expert", ExpertPolicy()), ("random", RandomPolicy(seed=0))):
                r = evaluate(policy, task, eval_set, args.n_episodes)
                rows.append({k: r[k] for k in ("task", "eval_set", "n_episodes", "success_rate", "mean_reward", "mean_length")} | {"policy": name})
                print(f"{task:7s} {eval_set:13s} {name:7s} success={r['success_rate']:.3f} length={r['mean_length']:.1f}")

    fields = ["task", "eval_set", "policy", "n_episodes", "success_rate", "mean_reward", "mean_length"]
    with open(args.out / "reference_baselines.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(args.out / "reference_baselines.md", "w") as f:
        f.write(f"Reference policies, {args.n_episodes} episodes per cell (success rate / mean episode length)\n\n")
        f.write("| task | eval set | expert | random |\n|---|---|---|---|\n")
        for task in TASKS:
            for eval_set in EVAL_SETS:
                cells = {r["policy"]: r for r in rows if r["task"] == task and r["eval_set"] == eval_set}
                fmt = lambda r: f"{r['success_rate']:.3f} / {r['mean_length']:.1f}"  # noqa: E731
                f.write(f"| {task} | {eval_set} | {fmt(cells['expert'])} | {fmt(cells['random'])} |\n")
    print(f"wrote {args.out / 'reference_baselines.md'}")


if __name__ == "__main__":
    main()
