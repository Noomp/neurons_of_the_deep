"""Single-room BabyAI levels for the "go to" and "pick up" tasks.

Built on MiniGrid's BabyAI ``RoomGridLevel`` so we inherit the official
instruction verifier, reward (1 - 0.9 * steps / max_steps) and time limit
(64 steps for an 8x8 room). Compared with ``BabyAI-GoToLocal-v0`` we add:

* a ``pickup`` variant with the same layout (GLAM's single-room pick-up task);
* a configurable number of distractors (for the 16-distractor shift);
* control over which (type, colour) pairs may be chosen as the goal
  (for the held-out-goal shift).
"""

from __future__ import annotations

from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc, PickupInstr

from .constants import COLORS, HELD_OUT_GOALS, NUM_DISTRACTORS, OBJ_TYPES, ROOM_SIZE, TASKS

# Which goals each pool may sample from.
GOAL_POOLS = ("train", "heldout", "any")


def allowed_goals(goal_pool: str) -> list[tuple[str, str]]:
    """Return the (type, colour) pairs that may be used as goals in a pool."""
    all_goals = [(t, c) for t in OBJ_TYPES for c in COLORS]
    if goal_pool == "train":
        return [g for g in all_goals if g not in HELD_OUT_GOALS]
    if goal_pool == "heldout":
        return [g for g in all_goals if g in HELD_OUT_GOALS]
    if goal_pool == "any":
        return all_goals
    raise ValueError(f"unknown goal_pool {goal_pool!r}, expected one of {GOAL_POOLS}")


class LocalLevel(RoomGridLevel):
    """One room, one goal object plus ``num_dists`` random distractors.

    The goal (type, colour) is drawn uniformly from ``goal_pool`` and placed
    first; distractors are then drawn uniformly over all 18 (type, colour)
    pairs and may duplicate the goal, as in BabyAI. Any object matching the
    instruction counts as success. Every object is reachable without moving
    another one.
    """

    def __init__(
        self,
        task: str = "goto",
        num_dists: int = NUM_DISTRACTORS,
        goal_pool: str = "train",
        room_size: int = ROOM_SIZE,
        **kwargs,
    ):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
        self.task = task
        self.num_dists = num_dists
        self.goal_pool = goal_pool
        self._goal_choices = allowed_goals(goal_pool)
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        goal = self._add_reachable_object(*self._rand_elem(self._goal_choices))
        for _ in range(self.num_dists):
            self._add_reachable_object(self._rand_elem(OBJ_TYPES), self._rand_elem(COLORS))

        desc = ObjDesc(goal.type, goal.color)
        self.instrs = GoToInstr(desc) if self.task == "goto" else PickupInstr(desc)

    def _add_reachable_object(self, kind: str, color: str, max_tries: int = 100):
        """Place an object at a random free cell such that every object stays
        reachable from the agent without moving anything.

        BabyAI instead places all objects and rejects the whole layout if any is
        unreachable. That is equivalent in practice for 8 distractors, but with
        16 distractors in a 6x6 interior ~1300 layouts are rejected per accepted
        one (~0.5 s per reset), so we reject per object instead.
        """
        room = self.room_grid[0][0]
        for _ in range(max_tries):
            obj, pos = self.add_object(0, 0, kind, color)
            if self.check_objs_reachable(raise_exc=False) is not False:
                return obj
            self.grid.set(*pos, None)
            room.objs = [o for o in room.objs if o is not obj]
        raise RejectSampling("could not place a reachable object")

    def _gen_grid(self, width, height):
        # Same rejection loop as RoomGridLevel._gen_grid, minus the print on
        # every rejected layout (which floods the console during collection).
        while True:
            try:
                super(RoomGridLevel, self)._gen_grid(width, height)
                self.gen_mission()
                self.validate_instrs(self.instrs)
            except (RecursionError, RejectSampling):
                continue
            break

        self.surface = self.instrs.surface(self)
        self.mission = self.surface
