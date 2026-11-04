"""Train one model: the SLM policy (one stage at a time) or B3, the symbolic agent.

SLM policy stages start from the previous stage's ``best.pt`` (``--init``);
stage 1 and B3 start from scratch. What each stage trains is in
``training/stages.py``, sizes in ``config.SLM`` / ``config.SYMBOLIC``,
optimiser settings in ``config.BC`` / ``config.PPO``. Change any of these for
one run with ``--set section.field=value`` (repeatable; sections slm, bc,
ppo, symbolic, train) instead of editing config.py.

With ``--init``, the model's config is the checkpoint's saved config, then
the stage's backbone, then the ``slm`` overrides; ``config.SLM`` is not used
(a warning names the fields where it differs). ``run.json`` records what was
actually used.

Usage (from the repo root, after ``python -m training.tools.collect_demos``):
    python -m training.train --stage 1 --tasks goto pickup --out outputs/runs/m1
    python -m training.train --stage 2 --tasks goto pickup --init outputs/runs/m1/best.pt --out outputs/runs/m2
    python -m training.train --stage 3 --tasks goto pickup --init outputs/runs/m2/best.pt --out outputs/runs/m3
    python -m training.train --stage 4 --tasks goto pickup --init outputs/runs/m3/best.pt --out outputs/runs/m4
    python -m training.train --model b3 --tasks goto pickup --out outputs/runs/b3
    python -m training.train --stage 1 --tasks goto pickup --set slm.memory_slots=0 --out outputs/runs/m1-nomem
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch

import config
from babytext import TASKS
from config import SLMPolicyConfig
from models.slm_policy import SLMPolicy, load_model
from models.symbolic_agent import SymbolicAgent

from .bc import train_bc
from .bc_symbolic import train_symbolic_bc
from .common import git_state, set_seed
from .demos import DemoConfigError, check_demo_config, load_demo_episodes
from .overrides import OverrideError, effective_configs, parse_overrides
from .ppo import train_ppo
from .stages import STAGES, Stage

FIXED = {"slm.backbone": "the backbone follows --stage (readout in stage 1, slm in stages 2-4)"}


def build_slm(
    stage: Stage, init: Path | None, slm: SLMPolicyConfig, overrides: dict
) -> tuple[SLMPolicy, list[str]]:
    """Build the SLM policy for ``stage``.

    Without ``init`` (stage 1 only), a fresh model from ``slm`` (``config.SLM``
    plus ``overrides``). With ``init``, the checkpoint's saved config plus the
    stage's backbone plus ``overrides``, so later stages keep the architecture
    their checkpoint was trained with.

    Returns:
        The model and warnings: with ``init``, one naming the fields where
        ``config.SLM`` differs from the config actually used without being overridden.
    """
    if init is None:
        if stage.number != 1:
            raise SystemExit(f"stage {stage.number} needs --init (the previous stage's best.pt)")
        return SLMPolicy(replace(slm, backbone=stage.backbone)), []
    model = load_model(init, backbone=stage.backbone, **overrides)
    differ = [
        f"{f.name} (config.py: {getattr(config.SLM, f.name)!r}, used: {getattr(model.cfg, f.name)!r})"
        for f in fields(SLMPolicyConfig)
        if f.name != "backbone" and f.name not in overrides
        and getattr(model.cfg, f.name) != getattr(config.SLM, f.name)
    ]
    warnings = []
    if differ:
        warnings.append(
            f"config.SLM differs from the config saved in {init} on: {'; '.join(differ)}. "
            "The checkpoint's values are used; pass --set slm.<field>=<value> to change one."
        )
    return model, warnings


def unused_sections(model: str, stage: Stage | None, overrides: dict) -> list[str]:
    """Overridden sections that this run never reads."""
    used = {"train"}
    if model == "b3":
        used |= {"symbolic", "bc"}
    else:
        used |= {"slm", "bc" if stage.algorithm == "bc" else "ppo"}
    return sorted(set(overrides) - used)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["slm", "b3"], default="slm", help="SLM policy (default) or B3")
    parser.add_argument("--stage", type=int, choices=sorted(STAGES), help="SLM policy stage (required for --model slm)")
    parser.add_argument("--tasks", nargs="+", default=["goto", "pickup"], choices=TASKS)
    parser.add_argument("--init", type=Path, help="checkpoint of the previous stage")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demos", type=Path, default=config.DATA.demos_dir, help="folder written by collect_demos")
    parser.add_argument("--n-demos", type=int, help="first N training demos per task (demo-count curves)")
    parser.add_argument("--seed", type=int, help="training seed (default: first of config.TRAIN.seeds)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                        help="change one setting for this run, e.g. bc.epochs=40 (repeatable)")
    parser.add_argument("--id", help="model ID recorded in run.json, e.g. M2-randlm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if args.model == "slm" and args.stage is None:
        parser.error("--model slm needs --stage")
    if args.model == "b3" and (args.stage is not None or args.init is not None):
        parser.error("--model b3 trains from scratch by behaviour cloning: no --stage or --init")
    try:
        overrides = parse_overrides(args.set, FIXED)
    except OverrideError as e:
        parser.error(str(e))
    cfgs = effective_configs(overrides)
    seed = cfgs["train"].seeds[0] if args.seed is None else args.seed
    stage = STAGES[args.stage] if args.model == "slm" else None

    try:
        demos_info = check_demo_config(
            args.demos, [f"{t}_{s}" for t in args.tasks for s in ("train", "val")], config.as_dict()
        )
    except DemoConfigError as e:
        raise SystemExit(f"error: {e}") from None
    if args.n_demos is not None:
        short = {t: demos_info["datasets"][f"{t}_train"]["episodes"] for t in args.tasks}
        short = {t: n for t, n in short.items() if n < args.n_demos}
        if short:
            raise SystemExit(f"error: --n-demos {args.n_demos}, but only {short} training demos were collected")

    warnings = [
        f"--set {section}.* has no effect on this run" for section in unused_sections(args.model, stage, overrides)
    ]
    set_seed(seed)
    if args.model == "b3":
        model = SymbolicAgent(cfgs["symbolic"])
        model_info = {"type": "b3", "cfg": asdict(model.cfg), "num_frames": model.num_frames, "view_size": model.view_size}
        description = "B3 symbolic agent, behaviour cloning"
    else:
        model, more = build_slm(stage, args.init, cfgs["slm"], overrides.get("slm", {}))
        warnings += more
        model_info = {"type": "slm", "stage": stage.number, "init": args.init, "cfg": asdict(model.cfg)}
        description = f"stage {stage.number}: {stage.description}"
    for w in warnings:
        print(f"warning: {w}")

    info = {
        "model_id": args.id,
        "command": shlex.join(["python", "-m", "training.train", *argv]),
        "args": {**vars(args), "seed": seed},
        "git": git_state(),
        "device": args.device,
        "overrides": overrides,
        "model": {**model_info, "trainable_parameters": model.num_parameters()},
        **{section: asdict(cfgs[section]) for section in ("bc", "ppo", "symbolic", "train")},
        "demos": demos_info,
        "file_config": config.as_dict(),
        "warnings": warnings,
        "finished": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    run_json = args.out / "run.json"
    run_json.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"{description}; {model.num_parameters():,} trainable parameters before freezing")

    start = time.time()
    if args.model == "b3" or stage.algorithm == "bc":
        train = [ep for t in args.tasks for ep in load_demo_episodes(args.demos / f"{t}_train", args.n_demos)]
        val = [ep for t in args.tasks for ep in load_demo_episodes(args.demos / f"{t}_val")]
        if args.model == "b3":
            train_symbolic_bc(model, train, val, args.tasks, args.out, cfg=cfgs["bc"], device=args.device, seed=seed)
        else:
            train_bc(model, stage, train, val, args.tasks, args.out, cfg=cfgs["bc"], device=args.device, seed=seed)
    else:
        train_ppo(model, stage, args.tasks, args.out, cfg=cfgs["ppo"], device=args.device)

    info.update(finished=True, wall_time_s=round(time.time() - start, 1))
    run_json.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
