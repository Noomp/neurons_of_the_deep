"""Every trained model of the comparison as ``python -m training.train`` commands.

One command per model ID x training seed (``config.TRAIN.seeds``) x tier
(easy = goto pickup, hard = putnext pickup_then_goto), all under the same
protocol: same demos, seeds, epochs and selection rule; each control differs
from its reference by one ``--set``. Runs go to
``outputs/runs/<id>/<tier>/seed<k>/``, and a later stage's ``--init`` is the
``best.pt`` of the matching earlier run (same tier and seed).

Usage:
    python -m training.run_plan --dry-run                       # print every command
    python -m training.run_plan --dry-run --ids M1 M2 --tiers easy --seeds 0
    python -m training.run_plan --ids B3 M1 --tiers easy        # run them, in order

Without ``--dry-run`` the commands run one after another; runs whose
``run.json`` says ``"finished": true`` are skipped, so an interrupted plan can
be restarted. ``--set`` and ``--n-demos`` are passed to every run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import config

from .overrides import OverrideError, effective_configs, parse_overrides

TIERS: dict[str, tuple[str, ...]] = {"easy": ("goto", "pickup"), "hard": ("putnext", "pickup_then_goto")}


@dataclass(frozen=True)
class PlannedModel:
    """One model ID: how it is trained, and from which earlier run.

    Attributes:
        id: model ID (also the folder name).
        question: what it is compared for (printed as a comment).
        model: "slm" or "b3".
        stage: SLM policy stage, or None for B3.
        init: ID of the run whose ``best.pt`` is ``--init``.
        sets: ``--set`` items that make this ID differ from its reference.
    """

    id: str
    question: str
    model: str = "slm"
    stage: int | None = None
    init: str | None = None
    sets: tuple[str, ...] = ()


def models(bc_epochs: int) -> list[PlannedModel]:
    """The comparison and its controls, every model after the run it starts from.

    Behaviour cloning only: M4 (stage 4, PPO) is a stretch goal outside the
    core results, so it is not planned; run it by hand with ``--stage 4``.

    Args:
        bc_epochs: epochs of one behaviour-cloning stage; M1-long trains for
            two (stage 1 + stage 2).
    """
    return [
        PlannedModel("B3", "standard BabyAI-style reference", model="b3"),
        PlannedModel("M1", "base of the M chain", stage=1),
        PlannedModel("M2", "does a pretrained LM help?", stage=2, init="M1"),
        PlannedModel("M3", "does tuning the LM's input layers help?", stage=3, init="M2"),
        PlannedModel("M1-nomem", "does the memory help?", stage=1, sets=("slm.memory_slots=0",)),
        PlannedModel("M1-long", "M1 with as many epochs as M1 + M2 (fair M1 vs M2)", stage=1,
                     sets=(f"bc.epochs={2 * bc_epochs}",)),
        PlannedModel("M2-randlm", "the LM's knowledge, or just its size?", stage=2, init="M1",
                     sets=("slm.slm_pretrained=false",)),
        *[
            PlannedModel(f"M2-L{n}", "where the decision-relevant information sits", stage=2, init="M1",
                         sets=(f"slm.readout_layer={n}",))
            for n in (10, 18)  # layer 28 is the whole of Qwen3-0.6B, i.e. M2 itself
        ],
    ]


def run_dir(root: Path, model_id: str, tier: str, seed: int) -> Path:
    return root / model_id / tier / f"seed{seed}"


def command(m: PlannedModel, tier: str, seed: int, root: Path, global_sets: list[str], n_demos: int | None) -> list[str]:
    """The ``python -m training.train`` arguments of one run.

    A global ``--set`` of a field that ``m`` sets itself is dropped for this run.
    """
    own = {s.partition("=")[0] for s in m.sets}
    sets = [s for s in global_sets if s.partition("=")[0].strip() not in own] + list(m.sets)
    cmd = ["python", "-m", "training.train"]
    cmd += ["--model", "b3"] if m.model == "b3" else ["--stage", str(m.stage)]
    cmd += ["--tasks", *TIERS[tier], "--seed", str(seed), "--id", m.id]
    if m.init is not None:
        cmd += ["--init", (run_dir(root, m.init, tier, seed) / "best.pt").as_posix()]
    if n_demos is not None:
        cmd += ["--n-demos", str(n_demos)]
    for s in sets:
        cmd += ["--set", s]
    return cmd + ["--out", run_dir(root, m.id, tier, seed).as_posix()]


def finished(out: Path) -> bool:
    try:
        return json.loads((out / "run.json").read_text(encoding="utf-8")).get("finished", False)
    except (OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    parser.add_argument("--ids", nargs="+", help="model IDs to include (default: all)")
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
    plan = models(cfgs["bc"].epochs)
    known = [m.id for m in plan]
    unknown = set(args.ids or []) - set(known)
    if unknown:
        parser.error(f"unknown model IDs {sorted(unknown)}; known: {', '.join(known)}")
    selected = [m for m in plan if args.ids is None or m.id in args.ids]
    seeds = args.seeds if args.seeds is not None else list(cfgs["train"].seeds)

    for m in selected:
        source = "B3" if m.model == "b3" else f"stage {m.stage}" + (f" from {m.init}" if m.init else "")
        print(f"# {m.id}: {source}; {m.question}")
        for tier in args.tiers:
            for seed in seeds:
                cmd = command(m, tier, seed, args.root, args.set, args.n_demos)
                out = run_dir(args.root, m.id, tier, seed)
                if args.dry_run:
                    print(shlex.join(cmd))
                    continue
                if finished(out):
                    print(f"skip (finished): {out.as_posix()}")
                    continue
                if m.init is not None and not (run_dir(args.root, m.init, tier, seed) / "best.pt").exists():
                    raise SystemExit(f"{m.id} {tier} seed {seed} needs {m.init} first (include it in --ids)")
                print(shlex.join(cmd), flush=True)
                subprocess.run([sys.executable, *cmd[1:]], check=True)
        print()


if __name__ == "__main__":
    main()
