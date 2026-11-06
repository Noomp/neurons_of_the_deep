"""Final evaluation: every condition on the test sets, once (proposal 6.5).

    python -m evaluation.evaluate --models random B3 P-L12-attn-h1 B2 B1 --dry-run
    python -m evaluation.evaluate --models random B3 P-L12-attn-h1 B2 B1
    python -m evaluation.evaluate --models B3-n100 P-L12-attn-h1-n100 --sets test

Models are ``random``, ``B1`` and ``B2`` (no training; one run each) or a
training run ID (``B3``, ``P-L12-attn-h1``, ``B3-n100``, ...), evaluated for
every seed from ``<runs>/<ID>/<tier>/seed<k>/best.pt``. Each model is
evaluated on every task of each ``--tiers`` tier and every ``--sets`` eval set,
with ``--n-episodes`` fixed seeds per (task, eval set): the same episodes for
every condition. Results go to ``<out>/<ID>/<tier>/<seed<k> | zero-shot>/``,
one cell per ``<task>.<eval set>`` (``evaluation.record``).

Everything is checked before the first episode, so a long run does not fail
hours in: a trained run must be finished and trained under the current
``ENV`` / ``EVAL_SETS``, and every head must read the same LLM as B1 and B2
(``config.LLM``). A run whose checkpoint is missing (not trained yet) is
listed and skipped. A cell is done when its summary exists with the same
identity (model, checkpoint hash, LLM and prompt settings, episodes, eval set
and environment config); done cells are skipped, so an interrupted
evaluation resumes with the same command. A summary with a different
identity stops the evaluation unless ``--overwrite`` is given.

Latency is the time of each policy call at batch size 1. Run nothing else on
the GPU during an evaluation, or the latencies are not comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import config
from babytext import RandomPolicy, get_seeds, make_env
from babytext.constants import TASK_TIERS

from .record import evaluate_cell

__all__ = ["ZERO_SHOT", "PlannedRun", "cell_identity", "check_run", "main", "plan_runs"]

ZERO_SHOT = ("random", "B1", "B2")


class EvaluationError(Exception):
    """A planned run that cannot be evaluated as asked."""


@dataclass
class PlannedRun:
    """One policy to evaluate: a zero-shot condition, or one seed of a trained run."""

    model: str  # "random", "B1", "B2" or a training run ID
    tier: str
    run: str  # "zero-shot" or "seed<k>"
    run_dir: Path | None = None  # the training run (trained models only)

    @property
    def trained(self) -> bool:
        return self.run_dir is not None

    @property
    def checkpoint(self) -> Path | None:
        return self.run_dir / "best.pt" if self.trained else None

    def out_dir(self, root: Path) -> Path:
        return root / self.model / self.tier / self.run


def plan_runs(models: list[str], tiers: list[str], seeds: list[int], runs_root: Path) -> list[PlannedRun]:
    """Every (model, tier, run) to evaluate, in the order given."""
    plan = []
    for model in models:
        for tier in tiers:
            if model in ZERO_SHOT:
                plan.append(PlannedRun(model, tier, "zero-shot"))
            else:
                plan += [PlannedRun(model, tier, f"seed{s}", runs_root / model / tier / f"seed{s}") for s in seeds]
    return plan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_info(run: PlannedRun) -> dict | None:
    """The finished training run's ``run.json``; None if the run is not (fully) trained."""
    run_json = run.run_dir / "run.json"
    if not run_json.exists() or not run.checkpoint.exists():
        return None
    info = json.loads(run_json.read_text(encoding="utf-8"))
    return info if info.get("finished") else None


def check_run(run: PlannedRun, info: dict) -> None:
    """Raise ``EvaluationError`` if a trained run does not belong to the current setup."""
    current = config.as_dict()
    for section in ("env", "eval_sets"):
        if info["file_config"].get(section) != current[section]:
            raise EvaluationError(f"{run.run_dir}: trained with a different config.{section.upper()} than config.py now")
    if info["model"]["type"] not in ("head", "b3"):
        raise EvaluationError(f"{run.run_dir}: unknown model type {info['model']['type']!r}")
    if info["model"]["type"] == "head":
        llm = info["model"]["llm"]
        if (llm["name"], llm["dtype"]) != (config.LLM.name, config.LLM.dtype):
            raise EvaluationError(f"{run.run_dir}: the head reads {llm['name']} ({llm['dtype']}), but B1 and B2 use "
                                  f"{config.LLM.name} ({config.LLM.dtype}); every LLM condition must share the backbone")


def cell_identity(run: PlannedRun, task: str, eval_set: str, n_episodes: int, label: bool,
                  checkpoint_sha256: str | None) -> dict:
    """What makes two results of a cell the same measurement; a done cell must match it exactly."""
    seeds = get_seeds(task, eval_set, n_episodes)
    identity = {
        "model": run.model, "tier": run.tier, "run": run.run, "task": task, "eval_set": eval_set,
        "n_episodes": n_episodes, "first_seed": int(seeds[0]), "last_seed": int(seeds[-1]), "expert_labels": label,
        "env": config.as_dict()["env"], "eval_set_config": config.as_dict()["eval_sets"][eval_set],
    }
    if run.trained:
        identity["checkpoint_sha256"] = checkpoint_sha256
    elif run.model in ("B1", "B2"):
        identity.update(llm=asdict(config.LLM), prompted=asdict(config.PROMPTED))
    else:
        identity["random_seed"] = 0
    return json.loads(json.dumps(identity))  # compared with the stored JSON, so tuples must already be lists


def cell_status(prefix: Path, identity: dict) -> str:
    """"todo", "done", or "conflict" (a summary with a different identity)."""
    summary = Path(f"{prefix}.json")
    if not summary.exists():
        return "todo"
    return "done" if json.loads(summary.read_text(encoding="utf-8")).get("identity") == identity else "conflict"


def load_policy(run: PlannedRun, info: dict | None, device: str):
    """The policy of a planned run (``random`` is built per cell, so each cell is reproducible on its own)."""
    if run.model == "B1":
        from models.prompted import GenerationPolicy

        return GenerationPolicy.load(config.LLM, config.PROMPTED, device)
    if run.model == "B2":
        from models.prompted import ScoringPolicy

        return ScoringPolicy.load(config.LLM, config.PROMPTED, device)
    if info["model"]["type"] == "head":
        from models.action_head import load_policy as load_head

        return load_head(run.checkpoint, device)
    from models.symbolic_agent import SymbolicPolicy, load_model

    return SymbolicPolicy(load_model(run.checkpoint), device)


def warm_up(policy, task: str, n: int) -> None:
    """``n`` untimed decisions on a val observation, so the first timed ones do not pay for GPU start-up."""
    env = make_env(task, "val")
    obs, _ = env.reset(seed=get_seeds(task, "val", 1)[0])
    for _ in range(n):
        if hasattr(policy, "reset"):
            policy.reset(env)
        policy(obs)
    env.close()


def environment_info(device: str) -> dict:
    import torch
    import transformers

    from training.common import git_state

    gpu = torch.cuda.get_device_name() if device.startswith("cuda") and torch.cuda.is_available() else None
    return {"git": git_state(), "device": device, "gpu": gpu, "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="random, B1, B2 and/or training run IDs")
    parser.add_argument("--tiers", nargs="+", choices=sorted(TASK_TIERS), default=list(TASK_TIERS))
    parser.add_argument("--sets", nargs="+", choices=sorted(config.EVAL_SETS), default=list(config.EVAL.test_sets))
    parser.add_argument("--seeds", nargs="+", type=int, help="training seeds (default: config.TRAIN.seeds)")
    parser.add_argument("--n-episodes", type=int, default=config.EVAL.n_episodes, help="per (task, eval set)")
    parser.add_argument("--runs", type=Path, default=config.DATA.outputs_dir / "runs", help="training runs")
    parser.add_argument("--out", type=Path, default=config.DATA.outputs_dir / "eval")
    parser.add_argument("--device", default=None, help="default: cuda if available")
    parser.add_argument("--no-expert-labels", action="store_true", help="do not record the expert's actions")
    parser.add_argument("--overwrite", action="store_true", help="replace cells whose summary has a different identity")
    parser.add_argument("--dry-run", action="store_true", help="check and list the plan without evaluating")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    unknown_sets = [s for s in args.sets if not s.startswith("test")]
    if unknown_sets:
        print(f"warning: {unknown_sets} are not test sets; they are for training and selection")
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = args.seeds if args.seeds is not None else list(config.TRAIN.seeds)
    label = not args.no_expert_labels

    # Check everything first: a missing checkpoint is skipped, anything inconsistent stops before the first episode.
    jobs, missing, conflicts = [], [], []
    for run in plan_runs(args.models, args.tiers, seeds, args.runs):
        info, digest = None, None
        if run.trained:
            info = run_info(run)
            if info is None:
                missing.append(run)
                continue
            try:
                check_run(run, info)
            except EvaluationError as e:
                parser.error(str(e))
            digest = sha256(run.checkpoint)
        cells = []
        for task in TASK_TIERS[run.tier]:
            for eval_set in args.sets:
                prefix = run.out_dir(args.out) / f"{task}.{eval_set}"
                identity = cell_identity(run, task, eval_set, args.n_episodes, label, digest)
                status = cell_status(prefix, identity)
                if status == "conflict" and not args.overwrite:
                    conflicts.append(prefix)
                cells.append((task, eval_set, prefix, identity, status))
        jobs.append((run, info, cells))
    if conflicts:
        listed = "\n  ".join(f"{p}.json" for p in conflicts)
        parser.error(f"these cells hold results of a different measurement (pass --overwrite to replace them):\n  {listed}")

    todo = sum(status != "done" for _, _, cells in jobs for *_, status in cells)
    for run, _, cells in jobs:
        done = sum(status == "done" for *_, status in cells)
        print(f"{run.model:22s} {run.tier:4s} {run.run:9s} {done}/{len(cells)} cells done")
    for run in missing:
        print(f"{run.model:22s} {run.tier:4s} {run.run:9s} skipped: no finished training run at {run.run_dir}")
    print(f"{todo} cells to evaluate, {args.n_episodes} episodes each")
    if args.dry_run or not todo:
        return

    env_info = environment_info(args.device)
    for run, info, cells in jobs:
        if all(status == "done" for *_, status in cells):
            continue
        print(f"# {run.model} {run.tier} {run.run}", flush=True)
        policy = None if run.model == "random" else load_policy(run, info, args.device)
        if policy is not None:
            warm_up(policy, TASK_TIERS[run.tier][0], config.EVAL.warmup_decisions)
        for task, eval_set, prefix, identity, status in cells:
            if status == "done":
                continue
            meta = {"identity": identity, **env_info, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
            if run.trained:
                meta.update(checkpoint=run.checkpoint.as_posix(), model_type=info["model"]["type"],
                            trainable_parameters=info["model"]["trainable_parameters"],
                            train_wall_time_s=info.get("wall_time_s"), val=info.get("selected"))
            cell_policy = RandomPolicy(seed=0) if run.model == "random" else policy
            s = evaluate_cell(cell_policy, task, eval_set, args.n_episodes, prefix, meta, label, not args.no_progress)
            extra = f" invalid={s['invalid_rate']:.3f}" if "invalid_rate" in s else ""
            print(f"  {task}/{eval_set}: success={s['success_rate']:.3f} length={s['mean_length']:.1f} "
                  f"latency={s['latency_ms']['median']:.1f}ms{extra} ({s['wall_time_s']:.0f}s)", flush=True)
        del policy
        if args.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
