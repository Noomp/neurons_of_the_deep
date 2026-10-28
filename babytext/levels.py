"""Single-room BabyAI levels for the four tasks.

=================  ==============================================  ========
task               example mission                                 max steps
=================  ==============================================  ========
goto               go to the red ball                              64
pickup             pick up the red ball                            64
putnext            put the red ball next to the blue key           128
pickup_then_goto   pick up the red ball, then go to the blue key   128
=================  ==============================================  ========

The last two are GLAM's single-room "put next" and "pick up then go to"
tasks (Level_MixedTrainLocal / Level_PickUpThenGoToLocal). Built on
MiniGrid's BabyAI ``RoomGridLevel`` so we inherit the official instruction
verifier, reward (1 - 0.9 * steps / max_steps) and time limit
(room_size**2 per navigation the instruction needs). On top of that we add a
configurable number of distractors (16-distractor shift) and control over
which (type, colour) pairs may appear in instructions (held-out shift).
"""

from __future__ import annotations

from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
from minigrid.envs.babyai.core.verifier import BeforeInstr, GoToInstr, ObjDesc, PickupInstr, PutNextInstr

from config import ENV
from .constants import COLORS, OBJ_TYPES, TASK_NUM_OBJECTS, TASKS

# Which (type, colour) pairs instructions may name:
#   train   - never a held-out pair;
#   heldout - exactly one instruction object is a held-out pair, any others
#             are drawn from the train pool;
#   any     - all 18 pairs.
GOAL_POOLS = ("train", "heldout", "any")


def allowed_goals(goal_pool: str) -> list[tuple[str, str]]:
    """Return the (type, colour) pairs that may be used as goals in a pool."""
    all_goals = [(t, c) for t in OBJ_TYPES for c in COLORS]
    if goal_pool == "train":
        return [g for g in all_goals if g not in ENV.held_out_goals]
    if goal_pool == "heldout":
        return [g for g in all_goals if g in ENV.held_out_goals]
    if goal_pool == "any":
        return all_goals
    raise ValueError(f"unknown goal_pool {goal_pool!r}, expected one of {GOAL_POOLS}")


class LocalLevel(RoomGridLevel):
    """One room: the instruction objects plus ``num_dists`` random distractors.

    The instruction objects' (type, colour) pairs are drawn from ``goal_pool``
    and placed first; distractors are then drawn uniformly over all 18 pairs
    and may duplicate them, as in BabyAI. Any object matching a description
    counts. Every object is reachable without moving another one.

    After reset, ``instr_objects`` holds the (type, colour) pair of each object
    named in the instruction, in mission order.
    """

    def __init__(
        self,
        task: str = "goto",
        num_dists: int | None = None,
        goal_pool: str = "train",
        room_size: int | None = None,
        **kwargs,
    ):
        # Unset settings come from config.ENV; extra kwargs (agent_view_size,
        # max_steps, render_mode, ...) go to MiniGrid.
        num_dists = ENV.num_dists if num_dists is None else num_dists
        room_size = ENV.room_size if room_size is None else room_size
        kwargs.setdefault("agent_view_size", ENV.agent_view_size)
        if ENV.max_steps is not None:
            kwargs.setdefault("max_steps", ENV.max_steps)
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
        self.task = task
        self.num_dists = num_dists
        self.goal_pool = goal_pool
        allowed_goals(goal_pool)  # validate early
        self.instr_objects: tuple[tuple[str, str], ...] = ()
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def _sample_instr_objects(self) -> list[tuple[str, str]]:
        n = TASK_NUM_OBJECTS[self.task]
        if self.goal_pool == "heldout":
            # Draw the held-out slot only for 2-object tasks, so single-object
            # tasks consume the RNG exactly like the other pools and a test
            # seed gives the same object positions in test and test_heldout.
            held_slot = self._rand_int(0, n) if n > 1 else 0
            pools = [allowed_goals("heldout" if i == held_slot else "train") for i in range(n)]
        else:
            pools = [allowed_goals(self.goal_pool)] * n
        while True:
            objs = [self._rand_elem(pool) for pool in pools]
            # Two-object instructions name two different (type, colour) pairs.
            # MiniGrid rejects "put the X next to the X", and "pick up a X, then
            # go to a X" degenerates: the go-to verifier keeps the positions
            # recorded at reset, so picking up X also counts as facing X.
            if len(set(objs)) == n:
                return objs

    def gen_mission(self):
        self.place_agent()
        self.instr_objects = tuple(self._sample_instr_objects())
        for kind, color in self.instr_objects:
            self._add_reachable_object(kind, color)
        for _ in range(self.num_dists):
            self._add_reachable_object(self._rand_elem(OBJ_TYPES), self._rand_elem(COLORS))

        descs = [ObjDesc(kind, color) for kind, color in self.instr_objects]
        if self.task == "goto":
            self.instrs = GoToInstr(descs[0])
        elif self.task == "pickup":
            self.instrs = PickupInstr(descs[0])
        elif self.task == "putnext":
            self.instrs = PutNextInstr(descs[0], descs[1])
        else:  # pickup_then_goto
            self.instrs = BeforeInstr(PickupInstr(descs[0]), GoToInstr(descs[1]))

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
