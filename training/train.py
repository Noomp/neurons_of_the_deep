"""Train one model by behaviour cloning: the action head on the frozen LLM (P), or B3.

Both use the same demos, seeds, batches, optimiser settings and selection rule
(``training.bc``). The head trains on cached hidden states, so run
``python -m training.tools.cache_hidden`` once first. Sizes are in
``config.HEAD`` / ``config.SYMBOLIC``, optimiser settings in ``config.BC``;
change any of them for one run with ``--set section.field=value`` (repeatable;
sections head, llm, bc, symbolic, train) instead of editing config.py.
``run.json`` records what was actually used.

Usage (from the repo root, after ``collect_demos`` and ``cache_hidden``):
    python -m training.train --tasks goto pickup --set head.layer=16 --set head.pooling=mean --out outputs/runs/p
    python -m training.train --model b3 --tasks goto pickup --out outputs/runs/b3
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

import config
from babytext import TASKS
from models.action_head import ActionHead
from models.symbolic_agent import SymbolicAgent

from .bc_head import head_demos, train_head_bc
from .bc_symbolic import train_symbolic_bc
from .common import git_state, set_seed
from .demos import DemoConfigError, check_demo_config, load_demo_episodes
from .hidden_cache import CacheError, cache_dir
from .overrides import OverrideError, effective_configs, parse_overrides

FIXED = {"head.sweep_layers": "it only decides which layers training.tools.cache_hidden caches"}
USED = {"head": {"head", "llm", "bc", "train"}, "b3": {"symbolic", "bc", "train"}}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["head", "b3"], default="head", help="action head on the LLM (default) or B3")
    parser.add_argument("--tasks", nargs="+", default=["goto", "pickup"], choices=TASKS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demos", type=Path, default=config.DATA.demos_dir, help="folder written by collect_demos")
    parser.add_argument("--hidden", type=Path, default=config.DATA.hidden_dir, help="folder written by cache_hidden")
    parser.add_argument("--n-demos", type=int, help="first N training demos per task (demo-count curves)")
    parser.add_argument("--seed", type=int, help="training seed (default: first of config.TRAIN.seeds)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                        help="change one setting for this run, e.g. head.layer=16 (repeatable)")
    parser.add_argument("--id", help="model ID recorded in run.json, e.g. P-L16-mean-h1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="replace a finished run in --out")
    args = parser.parse_args(argv)

    previous = args.out / "run.json"
    if previous.exists() and not args.overwrite:
        try:
            done = json.loads(previous.read_text(encoding="utf-8")).get("finished", False)
        except ValueError:
            done = False
        if done:
            raise SystemExit(f"error: {args.out} holds a finished run; pass --overwrite to replace it")

    try:
        overrides = parse_overrides(args.set, FIXED)
    except OverrideError as e:
        parser.error(str(e))
    cfgs = effective_configs(overrides)
    seed = cfgs["train"].seeds[0] if args.seed is None else args.seed

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
    warnings = [f"--set {s}.* has no effect on --model {args.model}" for s in sorted(set(overrides) - USED[args.model])]
    for w in warnings:
        print(f"warning: {w}")

    train_eps = {t: load_demo_episodes(args.demos / f"{t}_train", args.n_demos) for t in args.tasks}
    val_eps = {t: load_demo_episodes(args.demos / f"{t}_val") for t in args.tasks}
    set_seed(seed)
    if args.model == "head":
        head_cfg, llm = cfgs["head"], cfgs["llm"]
        directory = cache_dir(args.hidden, llm)
        try:
            train, cache_meta = head_demos(train_eps, "train", directory, args.demos, llm, head_cfg.layer, head_cfg.pooling)
            val, val_meta = head_demos(val_eps, "val", directory, args.demos, llm, head_cfg.layer, head_cfg.pooling)
        except CacheError as e:
            raise SystemExit(f"error: {e}") from None
        model = ActionHead(head_cfg, input_dim=next(iter(cache_meta.values()))["hidden_size"])
        model_info = {"type": "head", "cfg": asdict(head_cfg), "llm": asdict(llm), "cache": {**cache_meta, **val_meta}}
        description = f"action head: layer {head_cfg.layer}, {head_cfg.pooling} pooling, {head_cfg.hidden_layers} hidden layer(s)"
    else:
        model = SymbolicAgent(cfgs["symbolic"])
        model_info = {"type": "b3", "cfg": asdict(model.cfg), "num_frames": model.num_frames, "view_size": model.view_size}
        description = "B3 symbolic agent"

    info = {
        "model_id": args.id,
        "command": shlex.join(["python", "-m", "training.train", *argv]),
        "args": {**vars(args), "seed": seed},
        "git": git_state(),
        "device": args.device,
        "overrides": overrides,
        "model": {**model_info, "trainable_parameters": model.num_parameters()},
        **{section: asdict(cfgs[section]) for section in ("bc", "train")},
        "demos": demos_info,
        "file_config": config.as_dict(),
        "warnings": warnings,
        "finished": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.jsonl").unlink(missing_ok=True)  # one run per folder: no lines from an earlier attempt
    run_json = args.out / "run.json"
    run_json.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"{description}; {model.num_parameters():,} trainable parameters")

    start = time.time()
    common = dict(cfg=cfgs["bc"], device=args.device, seed=seed)
    if args.model == "head":
        _, summary = train_head_bc(model, train, val, args.tasks, args.out, llm=cfgs["llm"], **common)
    else:
        flat = lambda eps: [e for t in args.tasks for e in eps[t]]  # noqa: E731
        _, summary = train_symbolic_bc(model, flat(train_eps), flat(val_eps), args.tasks, args.out, **common)

    info.update(finished=True, wall_time_s=round(time.time() - start, 1), selected=summary)
    run_json.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
