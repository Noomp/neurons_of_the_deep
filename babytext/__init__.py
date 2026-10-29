"""babytext: a minimal BabyAI-Text re-implementation on Farama MiniGrid.

This is the environment section of the project and imports nothing from
``models`` or ``training``. Other code may use:

* the names in ``__all__`` below (environments, seeds, actions, policies,
  rollout / evaluation, text helpers);
* ``babytext.data``: ``save_demos``, ``load_episodes``, ``load_steps``,
  ``load_symbolic`` (the demonstration file format);
* ``babytext.viz``: ``replay``, ``capture``, ``draw_frame``, ``save_gif``,
  ``save_strip``, ``outline_goals``, ``Frame`` and the colour tokens
  (imported separately because it switches matplotlib to the Agg backend);
* ``babytext.constants``: the fixed vocabulary (``TASK_TIERS``,
  ``TASK_NUM_OBJECTS``, ``COLORS``, ``OBJ_TYPES``).

Everything else (``levels``, ``wrapper``, ``text``, ``splits`` internals,
names starting with ``_``) is an implementation detail.

Quick start::

    from babytext import make_env, get_seeds, ExpertPolicy, run_episode

    env = make_env("goto", "train")
    obs, info = env.reset(seed=get_seeds("goto", "train", 1)[0])
    print(obs["prompt"])
"""

import config
from config import EVAL_SETS
from .constants import ACTION_NAMES, ACTION_TO_IDX, NUM_ACTIONS, TASKS
from .policies import DisappearedBoxError, ExpertPolicy, RandomPolicy
from .prompt import PromptHistory, glam_prompt
from .rollout import Episode, Step, evaluate, run_episode
from .splits import get_seeds, make_env
from .text import Renamer, describe

__all__ = [
    "ACTION_NAMES",
    "ACTION_TO_IDX",
    "DisappearedBoxError",
    "EVAL_SETS",
    "Episode",
    "ExpertPolicy",
    "NUM_ACTIONS",
    "PromptHistory",
    "RandomPolicy",
    "Renamer",
    "Step",
    "TASKS",
    "config",
    "describe",
    "evaluate",
    "get_seeds",
    "glam_prompt",
    "make_env",
    "run_episode",
]
