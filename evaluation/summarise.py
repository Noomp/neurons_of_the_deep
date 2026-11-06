"""Tables of the final evaluation, from the cells written by ``evaluation.evaluate``.

    python -m evaluation.summarise          # outputs/eval -> outputs/results/evaluation.md, evaluation_cells.csv

Per tier: success per condition and eval set (mean over the tier's tasks),
per task, the generalisation gap of each shifted set (shifted minus ``test``),
latency and training cost, B1's invalid outputs, agreement with the expert,
and the demonstration-count curve of any ``<ID>-n<k>`` runs evaluated.

95% intervals: for trained conditions, a t-interval over training seeds (each
seed's success is the mean over the tier's tasks); for the zero-shot
conditions (one run), a Wilson interval over episodes, and for their gaps a
normal interval for a difference of proportions. A tier's success is only
reported for runs with every task of the tier evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
from babytext.constants import TASK_TIERS

from .record import read_episodes

__all__ = ["Interval", "load_cells", "main", "mean_interval", "tables", "wilson"]

# Two-sided 95% Student t quantiles by degrees of freedom (scipy is not a dependency).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
Z95 = 1.96


class Interval:
    """A point estimate with a 95% interval, and how many units (seeds or episodes) it rests on."""

    def __init__(self, value: float, low: float | None, high: float | None, n: int, unit: str):
        self.value, self.low, self.high, self.n, self.unit = value, low, high, n, unit

    def __str__(self) -> str:
        if self.low is None:
            return f"{self.value:.3f} (n={self.n})"
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}]"


def mean_interval(values: list[float]) -> Interval:
    """Mean of per-seed values with a t-interval; no interval for a single seed."""
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return Interval(mean, None, None, n, "seeds")
    half = T95.get(n - 1, Z95) * float(np.std(values, ddof=1)) / math.sqrt(n)
    return Interval(mean, mean - half, mean + half, n, "seeds")


def wilson(successes: float, n: int) -> Interval:
    """Success proportion over ``n`` episodes with a Wilson interval."""
    p = successes / n
    centre = (p + Z95**2 / (2 * n)) / (1 + Z95**2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / (1 + Z95**2 / n)
    return Interval(p, centre - half, centre + half, n, "episodes")


def load_cells(root: Path) -> list[dict]:
    """Every finished cell summary under ``root``."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/*/*/*.json"))]


def model_order(model: str) -> tuple:
    curve = re.fullmatch(r"(.+)-n(\d+)", model)
    rank = {"random": 0, "B1": 1, "B2": 2, "B3": 3}
    if curve:
        return (6, curve[1], int(curve[2]))
    return (rank.get(model, 4 if model.startswith("P-") else 5), model, 0)


def tables(cells: list[dict], root: Path) -> str:
    """The markdown report of all tiers (module docstring)."""
    by = defaultdict(dict)  # (model, tier, run) -> {(task, eval_set): cell}
    for c in cells:
        i = c["identity"]
        by[(i["model"], i["tier"], i["run"])][(i["task"], i["eval_set"])] = c
    out = ["# Final evaluation", "",
           "95% intervals: trained conditions over training seeds (t), zero-shot conditions over episodes (Wilson). "
           "Success of a tier is the mean over its tasks. Test sets were used for nothing else.", ""]

    for tier, tasks in TASK_TIERS.items():
        runs = {k: v for k, v in by.items() if k[1] == tier}
        if not runs:
            continue
        models = sorted({k[0] for k in runs}, key=model_order)
        sets = [s for s in config.EVAL.test_sets if any(s == es for v in runs.values() for _, es in v)]
        sets += sorted({es for v in runs.values() for _, es in v} - set(sets))

        def model_runs(model):
            return {k[2]: v for k, v in runs.items() if k[0] == model}

        def success(model, eval_set, task=None) -> Interval | None:
            """Success of ``model`` on ``eval_set`` (one task, or the tier's mean over tasks)."""
            wanted = [task] if task else list(tasks)
            per_run = {}
            for run, run_cells in model_runs(model).items():
                got = [run_cells.get((t, eval_set)) for t in wanted]
                if all(got):
                    per_run[run] = got
            if not per_run:
                return None
            if "zero-shot" in per_run:
                got = per_run["zero-shot"]
                return wilson(sum(c["success_rate"] * c["n_episodes"] for c in got), sum(c["n_episodes"] for c in got))
            return mean_interval([float(np.mean([c["success_rate"] for c in got])) for got in per_run.values()])

        def gap(model, eval_set) -> Interval | None:
            """Success on ``eval_set`` minus success on ``test``, paired per run."""
            diffs, zero_shot = [], None
            for run, run_cells in model_runs(model).items():
                shifted = [run_cells.get((t, eval_set)) for t in tasks]
                base = [run_cells.get((t, "test")) for t in tasks]
                if not (all(shifted) and all(base)):
                    continue
                p1 = float(np.mean([c["success_rate"] for c in shifted]))
                p0 = float(np.mean([c["success_rate"] for c in base]))
                if run == "zero-shot":
                    n1, n0 = sum(c["n_episodes"] for c in shifted), sum(c["n_episodes"] for c in base)
                    half = Z95 * math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
                    zero_shot = Interval(p1 - p0, p1 - p0 - half, p1 - p0 + half, n1, "episodes")
                diffs.append(p1 - p0)
            if zero_shot is not None:
                return zero_shot
            return mean_interval(diffs) if diffs else None

        out += [f"## {tier} tier ({', '.join(tasks)})", "", "### Success rate", ""]
        out += ["| condition | " + " | ".join(sets) + " |", "|---" * (len(sets) + 1) + "|"]
        for m in models:
            out.append(f"| {m} | " + " | ".join(str(success(m, s) or "-") for s in sets) + " |")

        out += ["", "### Success rate per task", ""]
        out += ["| condition | task | " + " | ".join(sets) + " |", "|---" * (len(sets) + 2) + "|"]
        for m in models:
            for t in tasks:
                out.append(f"| {m} | {t} | " + " | ".join(str(success(m, s, t) or "-") for s in sets) + " |")

        shifted_sets = [s for s in sets if s != "test"]
        if "test" in sets and shifted_sets:
            out += ["", "### Generalisation gap (shifted minus `test`; relative drop in brackets)", ""]
            out += ["| condition | " + " | ".join(shifted_sets) + " |", "|---" * (len(shifted_sets) + 1) + "|"]
            for m in models:
                base = success(m, "test")
                row = []
                for s in shifted_sets:
                    g = gap(m, s)
                    rel = f" ({g.value / base.value:+.0%})" if g and base and base.value > 0 else ""
                    row.append(f"{g}{rel}" if g else "-")
                out.append(f"| {m} | " + " | ".join(row) + " |")

        out += ["", "### Cost", "",
                "| condition | trainable parameters | training wall time (s, per seed) | latency median (ms) | latency p90 (ms) "
                "| decisions |", "|---|---|---|---|---|---|"]
        for m in models:
            all_cells = [c for v in model_runs(m).values() for c in v.values()]
            ms = np.concatenate([np.concatenate([e["ms"] for e in read_episodes(cell_prefix(root, c))])
                                 for c in all_cells])
            params = ", ".join(f"{p:,}" for p in sorted({c.get("trainable_parameters") for c in all_cells} - {None}))
            train = [c["train_wall_time_s"] for c in all_cells if c.get("train_wall_time_s") is not None]
            train_s = f"{np.mean(train):.0f}" if train else "-"
            out.append(f"| {m} | {params or '0'} | {train_s} | {np.median(ms):.1f} | {np.percentile(ms, 90):.1f} "
                       f"| {len(ms):,} |")

        if any(m == "B1" for m in models):
            out += ["", "### B1 invalid outputs (generated text that is not exactly an action name)", ""]
            out += ["| eval set | invalid rate | decisions |", "|---|---|---|"]
            for s in sets:
                got = [c for c in model_runs("B1").get("zero-shot", {}).values() if c["eval_set"] == s]
                if got:
                    n = sum(c["decisions"] for c in got)
                    out.append(f"| {s} | {sum(c['invalid_rate'] * c['decisions'] for c in got) / n:.3f} | {n:,} |")

        out += ["", "### Agreement with the expert (share of decisions where the expert, in the same state, "
                "would act the same)", ""]
        out += ["| condition | " + " | ".join(sets) + " |", "|---" * (len(sets) + 1) + "|"]
        for m in models:
            row = []
            for s in sets:
                got = [c for v in model_runs(m).values() for c in v.values() if c["eval_set"] == s and c["expert_agreement"] is not None]
                n = sum(c["expert_labelled_steps"] for c in got)
                row.append(f"{sum(c['expert_agreement'] * c['expert_labelled_steps'] for c in got) / n:.3f}" if n else "-")
            out.append(f"| {m} | " + " | ".join(row) + " |")

        curves = defaultdict(dict)
        for m in models:
            match = re.fullmatch(r"(.+)-n(\d+)", m)
            if match:
                curves[match[1]][int(match[2])] = m
        if curves and "test" in sets:
            counts = sorted({k for c in curves.values() for k in c} | {config.DATA.n_train_demos})
            out += ["", "### Demonstration-count curve (`test`)", ""]
            out += ["| condition | " + " | ".join(map(str, counts)) + " |", "|---" * (len(counts) + 1) + "|"]
            for base, points in curves.items():
                points = {**points, config.DATA.n_train_demos: base}
                out.append(f"| {base} | " + " | ".join(str(success(points[k], "test") or "-") if k in points else "-"
                                                        for k in counts) + " |")
        out.append("")
    return "\n".join(out)


def cell_prefix(root: Path, cell: dict) -> Path:
    i = cell["identity"]
    return root / i["model"] / i["tier"] / i["run"] / f"{i['task']}.{i['eval_set']}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, default=config.DATA.outputs_dir / "eval", help="evaluation cells")
    parser.add_argument("--out", type=Path, default=config.DATA.outputs_dir / "results")
    args = parser.parse_args(argv)
    cells = load_cells(args.eval)
    if not cells:
        parser.error(f"no evaluation cells under {args.eval}")
    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["model", "tier", "run", "task", "eval_set", "n_episodes", "success_rate", "mean_reward", "mean_length",
              "truncated_rate", "latency_median_ms", "invalid_rate", "expert_agreement", "trainable_parameters",
              "train_wall_time_s"]
    with open(args.out / "evaluation_cells.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for c in cells:
            writer.writerow({**c["identity"], **c, "latency_median_ms": c["latency_ms"]["median"]})
    report = tables(cells, args.eval)
    (args.out / "evaluation.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {args.out / 'evaluation.md'} and {args.out / 'evaluation_cells.csv'}")


if __name__ == "__main__":
    main()
