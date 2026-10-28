"""Gymnasium wrapper that exposes the BabyAI-Text interface.

Observations are dicts with both views of the same state:

* ``image`` (7, 7, 3) uint8 and ``direction`` int: MiniGrid's symbolic
  egocentric view (input for the non-LLM baseline B3);
* ``mission``: goal string, e.g. "go to the red ball";
* ``descriptions``: list of text descriptions of the current view;
* ``prompt``: GLAM prompt built from the last 3 observations and 2 actions.

The action space is ``Discrete(6)`` (see ``constants.ACTION_NAMES``).

``info`` adds ``goal``, the (type, colour) of each object named in the
instruction in mission order (one for goto/pickup, two for putnext and
pickup_then_goto), and ``goal_positions``, one list per instruction object of
the matching objects' current grid positions (carried objects are not listed).
"""

from __future__ import annotations

import gymnasium as gym

from .constants import ACTION_NAMES, NUM_ACTIONS
from .prompt import PromptHistory
from .text import Renamer, describe


def matching_positions(level, kind: str, color: str) -> list[tuple[int, int]]:
    """Grid positions of all objects currently on the floor matching (kind, color)."""
    return [
        (i, j)
        for j in range(level.grid.height)
        for i in range(level.grid.width)
        if (cell := level.grid.get(i, j)) is not None and cell.type == kind and cell.color == color
    ]


class TextEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, renamer: Renamer | None = None):
        super().__init__(env)
        self.renamer = renamer
        self.history = PromptHistory()
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        # Note: observation_space is left as MiniGrid's (image/direction/mission);
        # the text fields are free-form strings and are not range-checked.

    def _rename(self, text: str) -> str:
        return self.renamer(text) if self.renamer else text

    def _text_obs(self, obs: dict) -> dict:
        mission = self._rename(obs["mission"])
        descriptions = [self._rename(d) for d in describe(obs["image"])]
        self.history.add_observation(descriptions)
        return {
            "image": obs["image"],
            "direction": int(obs["direction"]),
            "mission": mission,
            "descriptions": descriptions,
            "prompt": self.history.prompt(mission),
        }

    def _info(self, info: dict) -> dict:
        level = self.env.unwrapped
        info = dict(info)
        info["goal"] = level.instr_objects
        info["goal_positions"] = [matching_positions(level, kind, color) for kind, color in level.instr_objects]
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.history.reset()
        obs, info = self.env.reset(seed=seed, options=options)
        return self._text_obs(obs), self._info(info)

    def step(self, action):
        action = int(action)
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError(f"action must be in [0, {NUM_ACTIONS}), got {action}")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.add_action(ACTION_NAMES[action])
        return self._text_obs(obs), float(reward), terminated, truncated, self._info(info)
