"""Demonstration episodes for behaviour cloning, and the checks that they are usable.

``load_demo_episodes`` reads a demo dataset as episodes; ``batch_indices``
gives every trainer the same batches of whole episodes for the same seed;
``check_demo_config`` refuses demos collected with other environment settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from babytext.data import load_episodes, load_symbolic


@dataclass
class DemoEpisode:
    """One expert episode of T steps: views, actions, prompts and its mission."""

    image: np.ndarray  # (T, V, V, 3) uint8, the symbolic view (B3's input)
    action: np.ndarray  # (T,) the expert's action
    prompts: list[str]  # T GLAM prompts (the LLM's input)
    mission: str

    def __len__(self) -> int:
        return len(self.action)


def load_demo_episodes(prefix: str | Path, limit: int | None = None) -> list[DemoEpisode]:
    """Episodes of a demo dataset written by ``training.tools.collect_demos``.

    Args:
        prefix: e.g. ``data/demos/goto_train`` (without extension).
        limit: keep only the first ``limit`` episodes (demonstration-count curves).
    """
    arrays = load_symbolic(prefix)
    records = load_episodes(prefix)
    if limit is not None:
        records = records[:limit]
    starts = np.flatnonzero(arrays["t"] == 0)
    ends = np.append(starts[1:], len(arrays["t"]))
    return [
        DemoEpisode(
            image=arrays["image"][s:e],
            action=arrays["action"][s:e],
            prompts=[step["prompt"] for step in rec["steps"]],
            mission=rec["mission"],
        )
        for rec, s, e in zip(records, starts, ends)
    ]


def batch_indices(lengths: list[int], batch_size: int, rng: np.random.Generator | None = None):
    """Yield lists of episode indices, one per batch; shuffled if ``rng`` is given.

    Episodes of similar length are grouped (after shuffling, within chunks of
    50 batches) to waste less compute on padding. The order depends only on
    ``lengths`` and ``rng``, so every trainer given the same episodes and seed
    (the action head's and B3's) sees the same batches.
    """
    order = rng.permutation(len(lengths)) if rng is not None else np.arange(len(lengths))
    chunk = batch_size * 50
    for c in range(0, len(order), chunk):
        part = sorted(order[c : c + chunk], key=lambda i: lengths[i])
        batches = [part[i : i + batch_size] for i in range(0, len(part), batch_size)]
        if rng is not None:
            rng.shuffle(batches)
        yield from batches


class DemoConfigError(ValueError):
    """The demonstrations on disk were not generated with the current ``config``."""


def check_demo_config(demos_dir: str | Path, datasets: list[str], current: dict) -> dict:
    """Check that ``<demos_dir>/summary.json`` matches the current environment settings.

    The ``env`` and ``eval_sets`` sections decide which episodes exist, so
    training on demos collected with other values would mix two setups.

    Args:
        demos_dir: folder written by ``training.tools.collect_demos``.
        datasets: names that must be listed in the summary, e.g. ``["goto_train", "goto_val"]``.
        current: ``config.as_dict()``.

    Returns:
        The summary's ``datasets`` and ``config`` (for ``run.json``).

    Raises:
        DemoConfigError: no summary, a listed dataset is missing, or a setting differs.
    """
    path = Path(demos_dir) / "summary.json"
    fix = "re-collect them with `python -m training.tools.collect_demos`"
    if not path.exists():
        raise DemoConfigError(f"{path} not found, so the demos' settings are unknown; {fix}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    missing = [d for d in datasets if d not in summary["datasets"]]
    if missing:
        raise DemoConfigError(f"{path} does not list {missing} (collected separately or not at all); {fix}")
    current = json.loads(json.dumps(current))  # tuples -> lists, as stored
    diffs = [
        f"{section}.{key}: demos {summary['config'][section].get(key)!r}, config.py {current[section].get(key)!r}"
        for section in ("env", "eval_sets")
        for key in sorted(set(summary["config"][section]) | set(current[section]))
        if summary["config"][section].get(key) != current[section].get(key)
    ]
    if diffs:
        raise DemoConfigError(
            f"the demos in {demos_dir} were collected with other environment settings:\n  "
            + "\n  ".join(diffs)
            + f"\nEither restore those values in config.py or {fix}."
        )
    return {"datasets": summary["datasets"], "config": summary["config"]}
