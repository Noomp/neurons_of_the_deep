"""Reference policies: the scripted BabyAI expert and a uniform random agent.

A policy is any callable ``policy(obs) -> int`` with an optional
``reset(env)`` method that is called after every ``env.reset``. Learned
policies (action heads, baselines) only need to implement ``__call__``; most
will read ``obs["prompt"]`` or ``obs["image"]`` / ``obs["direction"]``.
"""

from __future__ import annotations

import numpy as np
from minigrid.utils.baby_ai_bot import BabyAIBot, DisappearedBoxError, DropSubgoal

from .constants import NUM_ACTIONS


class ExpertPolicy:
    """BabyAI's procedural expert (``BabyAIBot``), as used for demonstrations.

    The bot reads the environment directly and remembers every cell it has
    seen this episode (more than the 3 observations in the prompt), so it must
    be ``reset`` with the environment after each ``env.reset``.

    One change from the stock bot: in ``pickup_then_goto`` the bot plans
    "pick up X, drop X, go to Y", dropping X back where it was. The drop is not
    needed for success, and two steps later nothing in the prompt says X was
    ever picked up, so the expert's next action would depend on information
    the model cannot see. We remove that drop: the expert carries X to Y and
    every observation says "You carry a ...".

    Relabelling another policy's states (DAgger): pass the action that was
    actually executed on the previous step as ``action_taken``. Without it the
    bot assumes its own suggestion was followed; if the policy did something
    else with pick up / drop, the bot's plan goes out of sync and it crashes
    or claims the mission is done::

        expert.reset(env)
        prev = None
        while not done:
            label = expert(obs, action_taken=prev)   # training target
            prev = head(obs)                         # what is executed
            obs, reward, term, trunc, info = env.step(prev)
            done = term or trunc

    If the executed action opened a box (``toggle`` on a box makes it vanish),
    the bot raises ``DisappearedBoxError``: the box may have been needed, so
    stop relabelling that episode.
    """

    def __init__(self):
        self.bot = None

    def reset(self, env):
        level = env.unwrapped
        self.bot = BabyAIBot(level)
        if getattr(level, "task", None) == "pickup_then_goto":
            # Initial stack, top last: [GoNextTo(Y), Drop, Pickup, GoNextTo(X)].
            self.bot.stack = [g for g in self.bot.stack if not isinstance(g, DropSubgoal)]

    def __call__(self, obs, action_taken: int | None = None) -> int:
        """Next expert action.

        Args:
            obs: unused (the bot reads the environment), kept for the policy interface.
            action_taken: action executed on the previous step. ``None`` on the first
                step, or when the expert's own previous suggestion was executed.
        """
        if action_taken is not None:
            action_taken = self.bot.mission.unwrapped.actions(int(action_taken))
        action = int(self.bot.replan(action_taken))
        if not 0 <= action < NUM_ACTIONS:
            raise RuntimeError(f"expert produced unsupported action {action}")
        return action


class RandomPolicy:
    """Uniform random actions (reference floor for every condition)."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs) -> int:
        return int(self.rng.integers(NUM_ACTIONS))
