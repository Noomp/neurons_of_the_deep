"""Small helpers shared by the trainers: seeding, logging, validation, git state."""

from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from babytext import evaluate


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every GPU).

    GPU runs are still not bit-for-bit reproducible: some CUDA kernels
    (e.g. atomic adds in backward passes) are non-deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RunLogger:
    """Prints metrics and appends them as JSON lines to ``<out_dir>/metrics.jsonl``."""

    def __init__(self, out_dir: str | Path):
        self.path = Path(out_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

    def log(self, **metrics) -> None:
        metrics = {"time": round(time.time() - self.start, 1), **metrics}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))


def validate(policy, tasks: list[str], n_episodes: int) -> dict[str, float]:
    """Greedy success rate on the first ``n_episodes`` val seeds of each task.

    Args:
        policy: a ready-made babytext policy (``obs -> action``, with
            ``reset``), e.g. ``SLMAgentPolicy(model, greedy=True)`` or
            ``SymbolicPolicy(model)``. Both put the model in eval mode; the
            caller switches it back to train mode afterwards.

    Returns:
        ``val_<task>`` per task, plus their mean under ``val_success``.
    """
    scores = {f"val_{task}": evaluate(policy, task, "val", n_episodes)["success_rate"] for task in tasks}
    scores["val_success"] = float(np.mean(list(scores.values())))
    return scores


def git_state() -> dict:
    """``{"commit": hash or None, "dirty": bool or None}`` of the working tree."""

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}
