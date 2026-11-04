"""Cache the frozen LLM's hidden states on every demonstration step (see ``training.hidden_cache``).

Run once after ``collect_demos``, and again after changing the demos, ``config.LLM``
or the layers. Last-token and mean vectors are cached at ``--layers``; full token
sequences (for attention pooling) only at ``--sequence-layers``, as they take
about 0.6 MB per step per layer.

Usage:
    python -m training.tools.cache_hidden                                   # all tasks, train + val, HEAD.sweep_layers
    python -m training.tools.cache_hidden --tasks goto pickup --sequence-layers 12 16
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from babytext import TASKS
from babytext.data import load_steps
from models.frozen_lm import FrozenLM
from training.demos import DemoConfigError, check_demo_config
from training.hidden_cache import FILES, cache_dir, file_sha256, to_bits
from training.overrides import OverrideError, effective_configs, parse_overrides


def cache_dataset(lm: FrozenLM, llm: config.LLMConfig, demos: Path, out: Path, prefix: str, layers: list[int],
                  sequence_layers: list[int], batch: int) -> dict:
    """Write the cache files of one demo dataset and return its meta data."""
    prompts = [s["prompt"] for s in load_steps(demos / prefix)]
    lengths = np.array([len(lm.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts])
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    d = lm.hidden_size
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{prefix}.meta.json").unlink(missing_ok=True)  # written last: a partial cache is never used
    pooled = {name: np.lib.format.open_memmap(out / f"{prefix}.{name}.npy", "w+", np.uint16, (len(prompts), len(layers), d))
              for name in FILES}
    tokens = {l: np.lib.format.open_memmap(out / f"{prefix}.seq{l}.npy", "w+", np.uint16, (int(offsets[-1]), d))
              for l in sequence_layers}
    if sequence_layers:
        np.save(out / f"{prefix}.offsets.npy", offsets)

    order = np.argsort(lengths, kind="stable")  # similar lengths together: less padding
    for start in tqdm(range(0, len(order), batch), desc=prefix):
        rows = order[start:start + batch]
        states = lm.run([prompts[r] for r in rows], sorted(set(layers) | set(sequence_layers)))
        for j, l in enumerate(layers):
            pooled["last"][rows, j] = to_bits(states.last(l))
            pooled["mean"][rows, j] = to_bits(states.mean(l))
        for l in sequence_layers:
            for r, seq in zip(rows, states.sequences(l)):
                tokens[l][offsets[r]:offsets[r + 1]] = to_bits(seq)
    for array in [*pooled.values(), *tokens.values()]:
        array.flush()

    meta = {
        "llm": json.loads(json.dumps(asdict(llm))),
        "layers": layers,
        "sequence_layers": sequence_layers,
        "steps": len(prompts),
        "hidden_size": d,
        "demos_sha256": file_sha256(demos / f"{prefix}.jsonl"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / f"{prefix}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--layers", nargs="+", type=int, default=list(config.HEAD.sweep_layers),
                        help="layers whose last-token and mean vectors are cached (default: config.HEAD.sweep_layers)")
    parser.add_argument("--sequence-layers", nargs="*", type=int, default=[],
                        help="layers whose full token sequences are cached, for attention pooling")
    parser.add_argument("--demos", type=Path, default=config.DATA.demos_dir)
    parser.add_argument("--out", type=Path, default=config.DATA.hidden_dir)
    parser.add_argument("--set", action="append", default=[], metavar="llm.FIELD=VALUE", help="change config.LLM")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    try:
        overrides = parse_overrides(args.set)
    except OverrideError as e:
        parser.error(str(e))
    if set(overrides) - {"llm"}:
        parser.error("cache_hidden only takes --set llm.<field>=<value>")
    llm = effective_configs(overrides)["llm"]
    prefixes = [f"{t}_{s}" for t in args.tasks for s in args.splits]
    try:
        check_demo_config(args.demos, prefixes, config.as_dict())
    except DemoConfigError as e:
        raise SystemExit(f"error: {e}") from None

    lm = FrozenLM.load(llm, num_layers=max(args.layers + args.sequence_layers), device=args.device)
    out = cache_dir(args.out, llm)
    for prefix in prefixes:
        meta = cache_dataset(lm, llm, args.demos, out, prefix, sorted(args.layers), sorted(args.sequence_layers),
                             llm.cache_batch)
        print(f"{out / prefix}: {meta['steps']} steps, layers {meta['layers']}, sequences {meta['sequence_layers']}")


if __name__ == "__main__":
    main()
