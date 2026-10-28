"""Seed splits and the environment factory.

Every episode is identified by (task, eval_set, seed). Seeds come from
disjoint ranges per split and task, so train / val / test layouts never
overlap. The eval sets themselves are defined in ``config.EVAL_SETS``; with
the default config the three shifted test sets reuse the test seeds with a
modified generator:

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

from config import EVAL_SETS
from .constants import TASKS
from .levels import LocalLevel
from .text import Renamer
from .wrapper import TextEnv

# Each (split, task) owns a block of SPLIT_SIZE consecutive seeds:
# seed = split_index * MAX_TASKS * SPLIT_SIZE + task_index * SPLIT_SIZE + i.
# Appending a task to TASKS therefore never moves another task's seeds.
SPLIT_SIZE = 1_000_000
MAX_TASKS = 10
assert len(TASKS) <= MAX_TASKS, "raising MAX_TASKS changes every seed"
_SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


def get_seeds(task: str, eval_set: str, n: int, start: int = 0) -> list[int]:
    """Seeds ``start .. start + n - 1`` of the given task and eval set."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    if start + n > SPLIT_SIZE:
        raise ValueError(f"at most {SPLIT_SIZE} seeds per split")
    split = EVAL_SETS[eval_set].seed_split
    base = (_SPLIT_INDEX[split] * MAX_TASKS + TASKS.index(task)) * SPLIT_SIZE
    return list(range(base + start, base + start + n))


def make_env(task: str, eval_set: str = "train", render_mode: str | None = None, **level_kwargs) -> TextEnv:
    """Create a text environment for ``task`` under ``eval_set``'s generator.

    Settings come from ``config.ENV``, then the eval set's overrides, then any
    extra ``level_kwargs`` (e.g. ``num_dists=4``) for quick experiments.
    """
    cfg = EVAL_SETS[eval_set]
    level = LocalLevel(task=task, render_mode=render_mode, **{**cfg.level_kwargs(), **level_kwargs})
    return TextEnv(level, renamer=Renamer() if cfg.nonce else None)
