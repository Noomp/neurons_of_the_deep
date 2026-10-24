"""babytext: a minimal BabyAI-Text re-implementation on Farama MiniGrid.

Quick start::

    from babytext import make_env, get_seeds, ExpertPolicy, run_episode

    env = make_env("goto", "train")
    obs, info = env.reset(seed=get_seeds("goto", "train", 1)[0])
    print(obs["prompt"])
"""

from .constants import ACTION_NAMES, ACTION_TO_IDX, HELD_OUT_GOALS, NUM_ACTIONS, TASKS
from .policies import ExpertPolicy, RandomPolicy
from .prompt import PromptHistory, glam_prompt
from .rollout import Episode, Step, evaluate, run_episode
from .splits import EVAL_SETS, get_seeds, make_env
from .text import Renamer, describe

__all__ = [
    "ACTION_NAMES",
    "ACTION_TO_IDX",
    "EVAL_SETS",
    "Episode",
    "ExpertPolicy",
    "HELD_OUT_GOALS",
    "NUM_ACTIONS",
    "PromptHistory",
    "RandomPolicy",
    "Renamer",
    "Step",
    "TASKS",
    "describe",
    "evaluate",
    "get_seeds",
    "glam_prompt",
    "make_env",
    "run_episode",
]
