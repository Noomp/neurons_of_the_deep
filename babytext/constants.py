"""Shared constants for the BabyAI-Text-style environment.

Values mirror the GLAM / BabyAI-Text setup (Carta et al., 2023) so that our
re-implementation stays comparable with the published environment.
"""

# The six actions exposed to every agent, in MiniGrid's index order
# (0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle). MiniGrid's 7th
# action ("done") is never used by the expert or by GLAM, so we drop it.
ACTION_NAMES = ("turn left", "turn right", "go forward", "pick up", "drop", "toggle")
NUM_ACTIONS = len(ACTION_NAMES)
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}

TASKS = ("goto", "pickup")

COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
OBJ_TYPES = ("key", "ball", "box")

# Goal objects GLAM excluded from training instructions (doors are not used in
# our single-room tasks, so "red door" / "grey door" are dropped). These may
# still appear as distractors during training; only the *goal* is held out.
HELD_OUT_GOALS = frozenset({("box", "yellow"), ("key", "red"), ("ball", "green")})

# Invented-word substitution for the name-shift test set. Words are short,
# pronounceable non-words chosen so none is a substring of another.
NONCE_WORDS = {
    # colours
    "red": "dax",
    "green": "wug",
    "blue": "fep",
    "purple": "blick",
    "yellow": "toma",
    "grey": "zup",
    # object types
    "key": "glorp",
    "ball": "tiv",
    "box": "marl",
}

# Default room layout (GLAM / BabyAI GoToLocal): one 8x8 room (6x6 interior),
# 8 distractors + 1 goal object, 7x7 egocentric view.
ROOM_SIZE = 8
NUM_DISTRACTORS = 8
AGENT_VIEW_SIZE = 7

# GLAM prompt history: last 3 observations and the 2 actions between them.
PROMPT_NUM_OBS = 3
