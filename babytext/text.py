"""Text descriptions of the agent's partial view, in BabyAI-Text format.

``describe`` is a port of ``MiniGridEnv.gen_graph`` from the BabyAI-Text fork
of gym-minigrid (flowersteam/Grounding_LLMs_with_online_RL), English branch,
producing the same strings in the same order, e.g.::

    You carry a red ball
    You see a wall 2 steps forward
    You see a wall 3 steps left
    You see a blue key 1 step left and 2 steps forward
    You see a green box 2 steps right

Coordinates are egocentric: "forward" is the direction the agent faces and
left/right are relative to it. Only cells visible to the agent are described
(walls block sight). A wall is mentioned only when it is the first non-empty
cell straight ahead, left or right of the agent.

The text is a deterministic function of MiniGrid's symbolic observation
(``obs["image"]``), so the text and symbolic views hold the same information.
"""

from __future__ import annotations

import re

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from config import ENV

IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
IDX_TO_STATE = {0: "open", 1: "closed", 2: "locked"}

_UNSEEN, _EMPTY, _WALL, _DOOR = (OBJECT_TO_IDX[k] for k in ("unseen", "empty", "wall", "door"))


def _steps(n: int, direction: str) -> str:
    return f"{n} step{'s' if n > 1 else ''} {direction}"


def describe(image: np.ndarray) -> list[str]:
    """Return the text descriptions for a symbolic egocentric view.

    ``image`` is MiniGrid's ``obs["image"]``: shape (view, view, 3) holding
    (object, colour, state) indices, indexed ``image[x][y]``, with the agent
    at the bottom centre facing up (towards y = 0) and any carried object
    encoded in the agent's own cell.
    """
    size = image.shape[0]
    ax, ay = size // 2, size - 1

    descriptions = []
    carried = image[ax][ay]
    if carried[0] not in (_UNSEEN, _EMPTY):
        descriptions.append(f"You carry a {IDX_TO_COLOR[carried[1]]} {IDX_TO_OBJECT[carried[0]]}")

    # Walls: walk outwards from the agent and stop at the first non-empty cell.
    for direction, cells in (
        ("forward", [(ax, y) for y in range(ay - 1, -1, -1)]),
        ("left", [(x, ay) for x in range(ax - 1, -1, -1)]),
        ("right", [(x, ay) for x in range(ax + 1, size)]),
    ):
        for x, y in cells:
            obj = image[x][y][0]
            if obj in (_UNSEEN, _EMPTY):
                continue
            if obj == _WALL:
                dist = abs(ax - x) + abs(ay - y)
                descriptions.append(f"You see a wall {_steps(dist, direction)}")
            break

    # Objects: column by column (left to right), then far to near, as in GLAM.
    for x in range(size):
        for y in range(size):
            obj, color, state = image[x][y]
            if obj in (_UNSEEN, _EMPTY, _WALL) or (x == ax and y == ay):
                continue  # the agent's own cell holds the carried object
            name = f"{IDX_TO_COLOR[color]} {IDX_TO_OBJECT[obj]}"
            if obj == _DOOR:
                state_word = IDX_TO_STATE[state]
                name = f"{state_word} {name}"
                prefix = "You see an " if state_word == "open" else "You see a "
            else:
                prefix = "You see a "

            parts = []
            if x != ax:
                parts.append(_steps(abs(x - ax), "right" if x > ax else "left"))
            if y != ay:
                parts.append(_steps(ay - y, "forward"))
            descriptions.append(prefix + name + " " + " and ".join(parts))

    return descriptions


class Renamer:
    """Word-level substitution of colour and object names (name-shift test).

    Applied to both the mission and the descriptions. Matching is on whole
    words, so "wall" and the action names are untouched.
    """

    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping = dict(ENV.nonce_words if mapping is None else mapping)
        words = sorted(self.mapping, key=len, reverse=True)
        self._pattern = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")

    def __call__(self, text: str) -> str:
        return self._pattern.sub(lambda m: self.mapping[m.group(1)], text)
