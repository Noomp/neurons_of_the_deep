"""Train the SLM policy, one stage at a time.

Each stage starts from the previous stage's ``best.pt`` (``--init``); stage 1
starts from scratch. What each stage trains is in ``training/stages.py``,
sizes in ``config.SLM``, optimiser settings in ``config.BC`` / ``config.PPO``.

Usage (from the repo root, after ``python -m training.tools.collect_demos``):
    python -m training.train --stage 1 --tasks goto pickup --out outputs/runs/m1
    python -m training.train --stage 2 --tasks goto pickup --init outputs/runs/m1/best.pt --out outputs/runs/m2
    python -m training.train --stage 3 --tasks goto pickup --init outputs/runs/m2/best.pt --out outputs/runs/m3
    python -m training.train --stage 4 --tasks goto pickup --init outputs/runs/m3/best.pt --out outputs/runs/m4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

import config
from babytext import TASKS
from models.slm_policy import SLMPolicy, load_model

from .bc import train_bc
from .common import set_seed
from .demos import load_demo_episodes
from .ppo import train_ppo
from .stages import STAGES


def build_model(stage_number: int, init: Path | None) -> SLMPolicy:
    """Stage 1 builds a fresh model; later stages load ``init`` and switch to the stage's backbone."""
    stage = STAGES[stage_number]
    if init is None:
        if stage_number != 1:
            raise SystemExit(f"stage {stage_number} needs --init (the previous stage's best.pt)")
        return SLMPolicy(replace(config.SLM, backbone=stage.backbone))
    return load_model(init, backbone=stage.backbone)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, required=True, choices=sorted(STAGES))
    parser.add_argument("--tasks", nargs="+", default=["goto", "pickup"], choices=TASKS)
    parser.add_argument("--init", type=Path, help="checkpoint of the previous stage")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demos", type=Path, default=config.DATA.demos_dir, help="folder written by collect_demos")
    parser.add_argument("--n-demos", type=int, help="first N training demos per task (demo-count curves)")
    parser.add_argument("--seed", type=int, default=config.TRAIN.seeds[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    stage = STAGES[args.stage]
    model = build_model(args.stage, args.init)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "config": config.as_dict()}, indent=2)
    )
    print(f"stage {stage.number}: {stage.description}; {model.num_parameters():,} trainable parameters before freezing")

    if stage.algorithm == "bc":
        demos = args.demos
        train = [ep for t in args.tasks for ep in load_demo_episodes(demos / f"{t}_train", args.n_demos)]
        val = [ep for t in args.tasks for ep in load_demo_episodes(demos / f"{t}_val")]
        train_bc(model, stage, train, val, args.tasks, args.out, device=args.device, seed=args.seed)
    else:
        train_ppo(model, stage, args.tasks, args.out, device=args.device)


if __name__ == "__main__":
    main()
