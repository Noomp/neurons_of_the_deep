"""Small helpers shared by the trainers: seeding, logging, validation."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from babytext import evaluate
from models.slm_policy import SLMAgentPolicy, SLMPolicy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def validate(model: SLMPolicy, tasks: list[str], n_episodes: int, device) -> dict[str, float]:
    """Greedy success rate on the first ``n_episodes`` val seeds of each task,
    plus their mean under "val_success". Leaves the model in train mode."""
    policy = SLMAgentPolicy(model, device=device, greedy=True)
    scores = {f"val_{task}": evaluate(policy, task, "val", n_episodes)["success_rate"] for task in tasks}
    scores["val_success"] = float(np.mean(list(scores.values())))
    model.train()
    return scores
