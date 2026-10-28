"""Fixed definitions the code depends on.

These are not experimental choices: they mirror MiniGrid (action indices,
object vocabulary) or define the task set. Settings you may want to change
(room size, distractors, held-out goals, invented words, prompt window,
eval sets, data and training settings) live in ``config.py``.
"""

# The six actions exposed to every agent, in MiniGrid's index order
# (0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle). MiniGrid's 7th
# action ("done") is never used by the expert or by GLAM, so we drop it.
ACTION_NAMES = ("turn left", "turn right", "go forward", "pick up", "drop", "toggle")
NUM_ACTIONS = len(ACTION_NAMES)
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}

# Task order is append-only: a task's position fixes its seed range (splits.py).
TASKS = ("goto", "pickup", "putnext", "pickup_then_goto")

# Easy tier: GLAM-comparable single-object tasks. Hard tier: two objects,
# needs "drop" (putnext) or carrying an object while navigating, longer episodes.
TASK_TIERS = {
    "easy": ("goto", "pickup"),
    "hard": ("putnext", "pickup_then_goto"),
}

# Number of objects named in each task's instruction, e.g.
# "put the red ball next to the blue key" names two.
TASK_NUM_OBJECTS = {"goto": 1, "pickup": 1, "putnext": 2, "pickup_then_goto": 2}

# MiniGrid's object vocabulary for the objects we place.
COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
OBJ_TYPES = ("key", "ball", "box")
