"""Running policies in the environment and summarising the results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import ACTION_NAMES
from .splits import get_seeds, make_env


@dataclass
class Step:
    t: int
    descriptions: list[str]
    prompt: str
    image: np.ndarray  # (7, 7, 3) uint8 symbolic view
    direction: int
    action: int

    @property
    def action_name(self) -> str:
        return ACTION_NAMES[self.action]


@dataclass
class Episode:
    task: str
    eval_set: str
    seed: int
    mission: str
    goal: tuple[tuple[str, str], ...]  # (type, colour) per instruction object, before renaming
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0
    success: bool = False
    truncated: bool = False

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def actions(self) -> list[int]:
        return [s.action for s in self.steps]


def run_episode(env, policy, seed: int, task: str = "", eval_set: str = "") -> Episode:
    """Roll out ``policy`` for one episode from ``env.reset(seed=seed)``."""
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(env)
    ep = Episode(task=task, eval_set=eval_set, seed=seed, mission=obs["mission"], goal=info["goal"])
    while True:
        action = int(policy(obs))
        ep.steps.append(
            Step(
                t=len(ep.steps),
                descriptions=obs["descriptions"],
                prompt=obs["prompt"],
                image=obs["image"].copy(),
                direction=obs["direction"],
                action=action,
            )
        )
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            ep.reward = reward
            ep.success = terminated and reward > 0
            ep.truncated = truncated and not terminated
            return ep


def evaluate(policy, task: str, eval_set: str, n_episodes: int, start: int = 0, progress: bool = False) -> dict:
    """Evaluate ``policy`` on ``n_episodes`` fixed seeds of an eval set.

    Returns a dict with ``success_rate``, ``mean_reward``, ``mean_length``
    and the list of ``episodes`` (for failure analysis / visualisation).
    """
    env = make_env(task, eval_set)
    seeds = get_seeds(task, eval_set, n_episodes, start)
    if progress:
        from tqdm import tqdm

        seeds = tqdm(seeds, desc=f"{task}/{eval_set}")
    episodes = [run_episode(env, policy, s, task, eval_set) for s in seeds]
    env.close()
    return {
        "task": task,
        "eval_set": eval_set,
        "n_episodes": n_episodes,
        "success_rate": float(np.mean([e.success for e in episodes])),
        "mean_reward": float(np.mean([e.reward for e in episodes])),
        "mean_length": float(np.mean([e.length for e in episodes])),
        "episodes": episodes,
    }
