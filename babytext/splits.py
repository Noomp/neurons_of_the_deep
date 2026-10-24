"""Evaluation sets, seed splits and the environment factory.

Every episode is identified by (task, eval_set, seed). Seeds come from
disjoint ranges per split and task, so train / val / test layouts never
overlap. The three shifted test sets reuse the test seeds with a modified
generator:

==============  ===========  ===========  =========  ===========
eval set        seed split   distractors  goal pool  names
==============  ===========  ===========  =========  ===========
train           train        8            train      real
val             val          8            train      real
test            test         8            train      real
test_dist16     test         16           train      real
test_nonce      test         8            train      invented
test_heldout    test         8            heldout    real
==============  ===========  ===========  =========  ===========
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import NUM_DISTRACTORS, TASKS
from .levels import LocalLevel
from .text import Renamer
from .wrapper import TextEnv


@dataclass(frozen=True)
class EvalSet:
    seed_split: str
    num_dists: int = NUM_DISTRACTORS
    goal_pool: str = "train"
    nonce: bool = False


EVAL_SETS = {
    "train": EvalSet("train"),
    "val": EvalSet("val"),
    "test": EvalSet("test"),
    "test_dist16": EvalSet("test", num_dists=16),
    "test_nonce": EvalSet("test", nonce=True),
    "test_heldout": EvalSet("test", goal_pool="heldout"),
}

# Each (split, task) owns a block of SPLIT_SIZE consecutive seeds.
SPLIT_SIZE = 1_000_000
_SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


def get_seeds(task: str, eval_set: str, n: int, start: int = 0) -> list[int]:
    """Seeds ``start .. start + n - 1`` of the given task and eval set."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    if start + n > SPLIT_SIZE:
        raise ValueError(f"at most {SPLIT_SIZE} seeds per split")
    split = EVAL_SETS[eval_set].seed_split
    base = (_SPLIT_INDEX[split] * len(TASKS) + TASKS.index(task)) * SPLIT_SIZE
    return list(range(base + start, base + start + n))


def make_env(task: str, eval_set: str = "train", render_mode: str | None = None, **level_kwargs) -> TextEnv:
    """Create a text environment for ``task`` under ``eval_set``'s generator.

    Extra ``level_kwargs`` (e.g. ``num_dists=4``) override the eval set's
    settings, which is handy for quick experiments.
    """
    cfg = EVAL_SETS[eval_set]
    kwargs = {"num_dists": cfg.num_dists, "goal_pool": cfg.goal_pool, **level_kwargs}
    level = LocalLevel(task=task, render_mode=render_mode, **kwargs)
    return TextEnv(level, renamer=Renamer() if cfg.nonce else None)
