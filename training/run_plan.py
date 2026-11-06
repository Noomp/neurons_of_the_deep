"""Every trained condition as ``python -m training.train`` commands, and a summary of the results.

One command per model ID x training seed (``config.TRAIN.seeds``) x tier
(easy = goto pickup, hard = putnext pickup_then_goto), into
``outputs/runs/<id>/<tier>/seed<k>/``, all under the same protocol (same
demos, seeds, batches, epochs and selection rule):

* ``B3``: the symbolic agent.
* ``P-L<l>-<last|mean>-h<n>``: the action head at layer ``l`` of
  ``config.HEAD.sweep_layers``, last-token or mean pooling, ``n`` = 1 or 2 MLP
  hidden layers (step 1 of the sweep; trains from the cache in seconds).
* ``P-L<l>-attn-h<n>``: learned attention pooling at the layers given with
  ``--attention-layers`` (step 2: the layers tied for best in step 1 by
  val success, see ``--summary``); these need cached token sequences.
* ``--curve ID ...``: the demo-count curve of those IDs, ``<ID>-n<k>`` for every
  k in ``config.TRAIN.demo_counts`` below the full demo count (which is the
  ID's own run).

B1 and B2 are not trained, so they are not here. The hidden-state caches the
head runs need are printed first.

Usage:
    python -m training.run_plan --dry-run                                  # every command
    python -m training.run_plan --dry-run --ids "P-L*-last-h1" --tiers easy --seeds 0
    python -m training.run_plan --ids B3 "P-L*" --tiers easy               # run them, in order
    python -m training.run_plan --summary                                  # val success per ID, best first
    python -m training.run_plan --dry-run --attention-layers 12 16 --ids "P-*-attn-*"

Without ``--dry-run`` the commands run one after another; runs whose
``run.json`` says ``"finished": true`` are skipped, and a missing cache is
built first. ``--set`` and ``--n-demos`` are passed to every run.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config

from .hidden_cache import CacheError, cache_dir, check_cache
from .overrides import OverrideError, effective_configs, parse_overrides

TIERS: dict[str, tuple[str, ...]] = {"easy": ("goto", "pickup"), "hard": ("putnext", "pickup_then_goto")}


@dataclass(frozen=True)
class PlannedModel:
    """One model ID: how it is trained.

    Attributes:
        id: model ID (also the folder name).
        model: "head" or "b3".
        sets: ``--set`` items that define it.
        n_demos: training demos per task (None = all).
        layer, pooling: the head's layer and pooling (for the cache it needs).
    """

    id: str
    model: str = "head"
    sets: tuple[str, ...] = ()
    n_demos: int | None = None
    layer: int | None = None
    pooling: str | None = None


def head_model(layer: int, pooling: str, hidden_layers: int) -> PlannedModel:
    short = "attn" if pooling == "attention" else pooling
    return PlannedModel(f"P-L{layer}-{short}-h{hidden_layers}", sets=(
        f"head.layer={layer}", f"head.pooling={pooling}", f"head.hidden_layers={hidden_layers}"),
        layer=layer, pooling=pooling)


def models(sweep_layers: tuple[int, ...], attention_layers: list[int]) -> list[PlannedModel]:
    """B3, the step-1 sweep (last / mean pooling at every sweep layer) and the step-2 attention heads."""
    return [
        PlannedModel("B3", model="b3"),
        *[head_model(l, p, h) for l in sweep_layers for p in ("last", "mean") for h in (1, 2)],
        *[head_model(l, "attention", h) for l in attention_layers for h in (1, 2)],
    ]


def curve(m: PlannedModel, counts: tuple[int, ...], full: int) -> list[PlannedModel]:
    return [PlannedModel(f"{m.id}-n{k}", m.model, m.sets, k, m.layer, m.pooling) for k in counts if k < full]


def run_dir(root: Path, model_id: str, tier: str, seed: int) -> Path:
    return root / model_id / tier / f"seed{seed}"


def command(m: PlannedModel, tier: str, seed: int, root: Path, global_sets: list[str], n_demos: int | None) -> list[str]:
    """The ``python -m training.train`` arguments of one run; a global ``--set`` of a field ``m`` sets is dropped."""
    own = {s.partition("=")[0] for s in m.sets} | {"head.sweep_layers"}  # the sweep only shapes the plan and cache
    sets = [s for s in global_sets if s.partition("=")[0].strip() not in own] + list(m.sets)
    cmd = ["python", "-m", "training.train", "--model", m.model, "--tasks", *TIERS[tier], "--seed", str(seed), "--id", m.id]
    n = m.n_demos if m.n_demos is not None else n_demos
    if n is not None:
        cmd += ["--n-demos", str(n)]
    for s in sets:
        cmd += ["--set", s]
    return cmd + ["--out", run_dir(root, m.id, tier, seed).as_posix()]


def cache_commands(tiers: list[str], layers: list[int], sequence_layers: list[int],
                   global_sets: list[str]) -> list[list[str]]:
    llm_sets = [a for s in global_sets if s.startswith("llm.") for a in ("--set", s)]
    cmds = []
    for tier in tiers:
        cmd = ["python", "-m", "training.tools.cache_hidden", "--tasks", *TIERS[tier], *llm_sets]
        if layers:
            cmd += ["--layers", *map(str, layers)]
        if sequence_layers:
            cmd += ["--sequence-layers", *map(str, sequence_layers)]
        cmds.append(cmd)
    return cmds


def cache_ready(tier: str, layers: list[int], sequence_layers: list[int], llm) -> bool:
    directory = cache_dir(config.DATA.hidden_dir, llm)
    try:
        for task in TIERS[tier]:
            for split in ("train", "val"):
                for layer in layers:
                    check_cache(directory, f"{task}_{split}", config.DATA.demos_dir, llm, layer, False)
                for layer in sequence_layers:
                    check_cache(directory, f"{task}_{split}", config.DATA.demos_dir, llm, layer, True)
    except CacheError:
        return False
    return True


def finished(out: Path) -> dict | None:
    """The ``run.json`` of a finished run, else None."""
    try:
        run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return run if run.get("finished") else None


def summary(root: Path, plan: list[PlannedModel], tiers: list[str], seeds: list[int]) -> None:
    """Print mean val success over the finished seeds of every ID, best first, per tier."""
    for tier in tiers:
        rows = []
        for m in plan:
            runs = [finished(run_dir(root, m.id, tier, s)) for s in seeds]
            scores = [r["selected"]["val_success"] for r in runs if r and "val_success" in r.get("selected", {})]
            if scores:
                rows.append((float(np.mean(scores)), float(np.std(scores)), len(scores), m.id))
        print(f"# {tier}: mean val success over seeds (selection only; never test)")
        for mean, std, n, model_id in sorted(rows, reverse=True):
            print(f"{model_id:24s} {mean:.3f} +- {std:.3f}  (n={n})")
        print()


def layer_number(text: str) -> int:
    """A layer given as ``20`` or as in the run IDs, ``L20``."""
    try:
        return int(text.removeprefix("L"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a layer number: {text!r} (e.g. 20 or L20)") from None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    parser.add_argument("--summary", action="store_true", help="print val success of finished runs instead")
    parser.add_argument("--ids", nargs="+", help="model IDs or glob patterns to include (default: all)")
    parser.add_argument("--attention-layers", nargs="*", type=layer_number, default=[], metavar="LAYER",
                        help="layers for the attention-pooling heads (step 2), e.g. 20 16 or L20 L16")
    parser.add_argument("--curve", nargs="*", default=[], metavar="ID", help="add demo-count curves of these IDs")
    parser.add_argument("--tiers", nargs="+", choices=sorted(TIERS), default=list(TIERS))
    parser.add_argument("--seeds", nargs="+", type=int, help="training seeds (default: config.TRAIN.seeds)")
    parser.add_argument("--root", type=Path, default=config.DATA.outputs_dir / "runs")
    parser.add_argument("--n-demos", type=int, help="passed to every run")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE", help="passed to every run")
    args = parser.parse_args(argv)

    try:
        cfgs = effective_configs(parse_overrides(args.set))
    except OverrideError as e:
        parser.error(str(e))
    unswept = sorted(set(args.attention_layers) - set(cfgs["head"].sweep_layers))
    if unswept:
        parser.error(f"--attention-layers {unswept}: pick from the step-1 layers {list(cfgs['head'].sweep_layers)}")
    plan = models(cfgs["head"].sweep_layers, args.attention_layers)
    by_id = {m.id: m for m in plan}
    unknown = [c for c in args.curve if c not in by_id]
    if unknown:
        parser.error(f"--curve: unknown IDs {unknown}")
    plan += [c for i in args.curve for c in curve(by_id[i], cfgs["train"].demo_counts, config.DATA.n_train_demos)]
    if args.ids is not None:
        plan = [m for m in plan if any(fnmatch.fnmatchcase(m.id, pattern) for pattern in args.ids)]
        if not plan:
            parser.error(f"--ids {args.ids} match no model ID")
    seeds = args.seeds if args.seeds is not None else list(cfgs["train"].seeds)
    if args.summary:
        summary(args.root, plan, args.tiers, seeds)
        return

    heads = [m for m in plan if m.model == "head"]
    pooled_layers = sorted({m.layer for m in heads if m.pooling != "attention"})
    sequence_layers = sorted({m.layer for m in heads if m.pooling == "attention"})
    if heads:
        print("# hidden-state caches (once per tier; seconds to minutes)")
        for tier, cmd in zip(args.tiers, cache_commands(args.tiers, pooled_layers, sequence_layers, args.set)):
            if args.dry_run:
                print(shlex.join(cmd))
            elif cache_ready(tier, pooled_layers, sequence_layers, cfgs["llm"]):
                print(f"skip (cached): {tier}")
            else:
                print(shlex.join(cmd), flush=True)
                subprocess.run([sys.executable, *cmd[1:]], check=True)
        print()

    for m in plan:
        print(f"# {m.id}")
        for tier in args.tiers:
            for seed in seeds:
                cmd = command(m, tier, seed, args.root, args.set, args.n_demos)
                out = run_dir(args.root, m.id, tier, seed)
                if args.dry_run:
                    print(shlex.join(cmd))
                elif finished(out):
                    print(f"skip (finished): {out.as_posix()}")
                else:
                    print(shlex.join(cmd), flush=True)
                    subprocess.run([sys.executable, *cmd[1:]], check=True)
        print()


if __name__ == "__main__":
    main()
