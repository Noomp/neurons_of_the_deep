"""Recorded rollouts: one evaluation cell, with everything kept per decision.

A cell is one policy on the first ``n_episodes`` seeds of one (task, eval set).
For every decision it keeps:

* the action taken and the time the policy took to choose it (ms; the policy
  call only, not the environment step);
* the expert's action in the same state (``babytext.ExpertPolicy`` relabelling
  the policy's own trajectory), for action confusion matrices;
* what the policy exposes about its choice: ``last_logits`` (P, B3),
  ``last_scores`` (B2), ``last_text`` and ``last_valid`` (B1).

Episodes are written to ``<prefix>.episodes.jsonl.gz`` (one JSON object per
line) and the summary to ``<prefix>.json``, last and atomically, so a cell
whose summary exists is complete. Observations are not stored: an episode is
replayed exactly from its ``(task, eval set, seed)`` and ``actions``.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from babytext import ExpertPolicy, get_seeds, make_env

__all__ = ["decision_details", "evaluate_cell", "read_episodes", "run_recorded_episode", "summarise_episodes"]


def decision_details(policy) -> dict:
    """What ``policy`` exposes about its latest decision, as JSON-ready values."""
    details = {}
    for attribute, key in (("last_logits", "logits"), ("last_scores", "scores")):
        values = getattr(policy, attribute, None)
        if values is not None:
            details[key] = [round(float(v), 4) for v in values.flatten().tolist()]
    if hasattr(policy, "last_text"):
        details["text"] = policy.last_text
        details["valid"] = bool(policy.last_valid)
    return details


def run_recorded_episode(env, policy, seed: int, label: bool = True) -> dict:
    """Roll out ``policy`` from ``env.reset(seed=seed)``, recording every decision (module docstring).

    The expert is queried before the policy acts, with the action executed on
    the previous step (``ExpertPolicy``'s relabelling protocol); it only reads
    the environment, so the episode is the same with or without labels. If the
    bot fails (e.g. the policy opened a box it needed), the remaining labels of
    the episode are ``None`` and ``label_error`` says why.
    """
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(env)
    expert, label_error = None, None
    if label:
        expert = ExpertPolicy()
        expert.reset(env)
    record = {"seed": int(seed), "mission": obs["mission"], "goal": [list(g) for g in info["goal"]],
              "actions": [], "expert": [], "ms": []}
    details = defaultdict(list)
    previous = None
    while True:
        target = None
        if expert is not None:
            try:
                target = expert(obs, action_taken=previous)
            except Exception as e:  # noqa: BLE001 - any bot failure ends labelling, never the evaluation
                expert, label_error = None, f"step {len(record['actions'])}: {e!r}"
        start = time.perf_counter()
        action = int(policy(obs))  # int() waits for the GPU, so the time includes the whole decision
        record["ms"].append(round((time.perf_counter() - start) * 1000, 3))
        record["actions"].append(action)
        record["expert"].append(target)
        for key, value in decision_details(policy).items():
            details[key].append(value)
        previous = action
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    record.update(details)
    record.update(success=bool(terminated and reward > 0), reward=float(reward), length=len(record["actions"]),
                  truncated=bool(truncated and not terminated), label_error=label_error)
    return record


def summarise_episodes(episodes: list[dict]) -> dict:
    """Success, reward, length, latency, expert agreement and (B1) invalid outputs over a cell's episodes."""
    ms = np.concatenate([e["ms"] for e in episodes])
    pairs = [(a, t) for e in episodes for a, t in zip(e["actions"], e["expert"]) if t is not None]
    summary = {
        "n_episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_reward": float(np.mean([e["reward"] for e in episodes])),
        "mean_length": float(np.mean([e["length"] for e in episodes])),
        "truncated_rate": float(np.mean([e["truncated"] for e in episodes])),
        "decisions": int(len(ms)),
        "latency_ms": {"median": float(np.median(ms)), "p90": float(np.percentile(ms, 90)), "mean": float(ms.mean())},
        "expert_labelled_steps": len(pairs),
        "expert_agreement": float(np.mean([a == t for a, t in pairs])) if pairs else None,
        "label_errors": sum(e["label_error"] is not None for e in episodes),
    }
    if episodes and "valid" in episodes[0]:
        valid = np.concatenate([e["valid"] for e in episodes])
        summary["invalid_rate"] = float(1 - valid.mean())
    return summary


def read_episodes(prefix: str | Path) -> list[dict]:
    """The episodes of a cell written by ``evaluate_cell``."""
    with gzip.open(f"{prefix}.episodes.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write_atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def evaluate_cell(policy, task: str, eval_set: str, n_episodes: int, prefix: str | Path, meta: dict,
                  label: bool = True, progress: bool = False) -> dict:
    """Evaluate ``policy`` on one cell and write its episodes and summary (module docstring).

    Args:
        prefix: output path without suffix, e.g. ``outputs/eval/B3/easy/seed0/goto.test``.
        meta: stored in the summary next to the results (what was evaluated, and how).

    Returns:
        The summary that was written.
    """
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    seeds = get_seeds(task, eval_set, n_episodes)
    env = make_env(task, eval_set)
    if progress:
        from tqdm import tqdm

        seeds = tqdm(seeds, desc=f"{prefix.parent.as_posix()} {task}/{eval_set}", leave=False)
    start = time.time()
    episodes = [run_recorded_episode(env, policy, s, label) for s in seeds]
    env.close()

    def write_episodes(tmp: Path) -> None:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for e in episodes:
                f.write(json.dumps(e) + "\n")

    _write_atomic(Path(f"{prefix}.episodes.jsonl.gz"), write_episodes)
    summary = {**meta, "task": task, "eval_set": eval_set, **summarise_episodes(episodes),
               "wall_time_s": round(time.time() - start, 1)}
    _write_atomic(Path(f"{prefix}.json"), lambda tmp: tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8"))
    return summary
