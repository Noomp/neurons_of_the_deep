"""GLAM prompt template, shared by every LLM-based condition.

Port of ``BaseAgent.generate_prompt`` from flowersteam/Grounding_LLMs_with_online_RL
(experiments/agents/base_agent.py), kept character-for-character identical,
including its spacing quirks. Example with three observations::

    Possible action of the agent: turn left, turn right, go forward, pick up, drop, toggle
     Goal of the agent: go to the red ball
     Observation 0: You see a wall 2 steps forward, You see a ...,
     Action 0: turn left
     Observation 1: ...,
     Action 1: go forward
     Observation 2: ...,
     Action 2:

(Each line after the first starts with a space; the prompt ends in "Action N: ".)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from .constants import ACTION_NAMES, PROMPT_NUM_OBS


def glam_prompt(
    goal: str,
    obs_history: Sequence[Sequence[str]],
    action_history: Sequence[str],
    action_names: Sequence[str] = ACTION_NAMES,
) -> str:
    """Build the GLAM prompt.

    Args:
        goal: mission string, e.g. "go to the red ball".
        obs_history: the last (up to 3) observations, oldest first; each is a
            list of description strings.
        action_history: names of the actions taken between those observations
            (one fewer than ``obs_history``).
    """
    head = "Possible action of the agent:" + ",".join(f" {a}" for a in action_names)
    prompt = head + f" \n Goal of the agent: {goal}"
    for i, descriptions in enumerate(obs_history):
        prompt += f" \n Observation {i}: " + "".join(f"{d}, " for d in descriptions)
        prompt += f"\n Action {i}: "
        if i < len(action_history):
            prompt += action_history[i]
    return prompt


class PromptHistory:
    """Rolling window of observations and actions, as kept by GLAM's agent."""

    def __init__(self, num_obs: int = PROMPT_NUM_OBS):
        self.num_obs = num_obs
        self.reset()

    def reset(self):
        self.observations: deque[list[str]] = deque(maxlen=self.num_obs)
        self.actions: deque[str] = deque(maxlen=self.num_obs - 1)

    def add_observation(self, descriptions: Sequence[str]):
        self.observations.append(list(descriptions))

    def add_action(self, action_name: str):
        self.actions.append(action_name)

    def prompt(self, goal: str) -> str:
        return glam_prompt(goal, list(self.observations), list(self.actions))
