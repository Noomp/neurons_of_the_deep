"""Saving and loading demonstration datasets.

A dataset ``<prefix>`` is two files written side by side:

* ``<prefix>.jsonl``: one JSON object per episode (seed, mission, outcome and
  per-step descriptions, prompt and action), human-readable;
* ``<prefix>.npz``: flat per-step arrays for the symbolic baseline:
  ``image`` (N, 7, 7, 3) uint8, ``direction`` (N,), ``action`` (N,),
  ``episode`` (N,) episode index, ``t`` (N,) step within episode.

Row ``i`` of the npz arrays is the ``i``-th step when iterating episodes in
file order and steps in time order, which is also the order of ``load_steps``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rollout import Episode


def save_demos(episodes: list[Episode], prefix: str | Path) -> None:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(prefix.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for ep in episodes:
            record = {
                "task": ep.task,
                "eval_set": ep.eval_set,
                "seed": ep.seed,
                "mission": ep.mission,
                "goal": list(ep.goal),
                "success": ep.success,
                "reward": ep.reward,
                "length": ep.length,
                "steps": [
                    {"descriptions": s.descriptions, "prompt": s.prompt, "action": s.action}
                    for s in ep.steps
                ],
            }
            f.write(json.dumps(record) + "\n")

    steps = [(i, s) for i, ep in enumerate(episodes) for s in ep.steps]
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        image=np.stack([s.image for _, s in steps]).astype(np.uint8),
        direction=np.array([s.direction for _, s in steps], dtype=np.int64),
        action=np.array([s.action for _, s in steps], dtype=np.int64),
        episode=np.array([i for i, _ in steps], dtype=np.int64),
        t=np.array([s.t for _, s in steps], dtype=np.int64),
    )


def load_episodes(prefix: str | Path) -> list[dict]:
    """Episode records from ``<prefix>.jsonl`` (dicts, as written by save_demos)."""
    with open(Path(prefix).with_suffix(".jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_steps(prefix: str | Path) -> list[dict]:
    """Flat list of steps: ``{episode, t, seed, mission, prompt, descriptions, action}``.

    Aligned row-for-row with the arrays in ``<prefix>.npz``.
    """
    rows = []
    for i, ep in enumerate(load_episodes(prefix)):
        for t, s in enumerate(ep["steps"]):
            rows.append({"episode": i, "t": t, "seed": ep["seed"], "mission": ep["mission"], **s})
    return rows


def load_symbolic(prefix: str | Path) -> dict[str, np.ndarray]:
    """Per-step arrays from ``<prefix>.npz``."""
    with np.load(Path(prefix).with_suffix(".npz")) as data:
        return {k: data[k] for k in data.files}
