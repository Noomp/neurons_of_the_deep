"""Collect expert demonstrations for behaviour cloning.

Writes, for each task and split, ``<out>/<task>_<split>.jsonl`` and ``.npz``
(see ``babytext.data``) plus ``<out>/summary.json``.

Episodes use the first N seeds of each split, so a smaller training set
(for demonstration-count learning curves) is just a prefix of a larger one:
take the first k episodes of ``goto_train.jsonl``.

Usage:
    python scripts/collect_demos.py                      # 2000 train / 200 val per task
    python scripts/collect_demos.py --n-train 500 --tasks goto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from babytext import TASKS, ExpertPolicy, get_seeds, make_env, run_episode
from babytext.data import save_demos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--n-train", type=int, default=2000, help="training episodes per task")
    parser.add_argument("--n-val", type=int, default=200, help="validation episodes per task")
    parser.add_argument("--out", type=Path, default=Path("data/demos"))
    args = parser.parse_args()

    summary = {}
    for task in args.tasks:
        for split, n in (("train", args.n_train), ("val", args.n_val)):
            env = make_env(task, split)
            policy = ExpertPolicy()
            episodes = [
                run_episode(env, policy, seed, task, split)
                for seed in tqdm(get_seeds(task, split, n), desc=f"{task}/{split}")
            ]
            failed = [e.seed for e in episodes if not e.success]
            if failed:
                print(f"warning: expert failed on {len(failed)} {task}/{split} seeds, e.g. {failed[:5]}")
            prefix = args.out / f"{task}_{split}"
            save_demos(episodes, prefix)
            summary[f"{task}_{split}"] = {
                "episodes": len(episodes),
                "steps": sum(e.length for e in episodes),
                "success_rate": sum(e.success for e in episodes) / len(episodes),
                "mean_length": sum(e.length for e in episodes) / len(episodes),
                "first_seed": episodes[0].seed,
                "last_seed": episodes[-1].seed,
            }
            print(f"{prefix}: {summary[f'{task}_{split}']}")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
