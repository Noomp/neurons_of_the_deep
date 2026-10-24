"""Reference policies: the scripted BabyAI expert and a uniform random agent.

A policy is any callable ``policy(obs) -> int`` with an optional
``reset(env)`` method that is called after every ``env.reset``. Learned
policies (action heads, baselines) only need to implement ``__call__``; most
will read ``obs["prompt"]`` or ``obs["image"]`` / ``obs["direction"]``.
"""

from __future__ import annotations

import numpy as np
from minigrid.utils.baby_ai_bot import BabyAIBot

from .constants import NUM_ACTIONS


class ExpertPolicy:
    """BabyAI's procedural expert (``BabyAIBot``), as used for demonstrations.

    The bot plans from the full environment state, not from the observation,
    so it must be ``reset`` with the environment after each ``env.reset``.
    """

    def __init__(self):
        self.bot = None

    def reset(self, env):
        self.bot = BabyAIBot(env.unwrapped)

    def __call__(self, obs) -> int:
        action = int(self.bot.replan())
        if not 0 <= action < NUM_ACTIONS:
            raise RuntimeError(f"expert produced unsupported action {action}")
        return action


class RandomPolicy:
    """Uniform random actions (reference floor for every condition)."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs) -> int:
        return int(self.rng.integers(NUM_ACTIONS))
