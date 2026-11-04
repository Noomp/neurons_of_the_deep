"""Environment adapter for reinforcement learning.

Everything that touches babytext / MiniGrid internals during RL lives here, so
the PPO code only sees arrays: ``VecTextEnv`` steps N environments in lockstep,
resets finished ones on fresh seeds, tracks each episode's step counter, and
adds the shaped reward. Single-process: MiniGrid steps take well under a
millisecond, far less than one SLM forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from babytext import get_seeds, make_env
from config import DATA, ENV

from .rewards import shaped_reward


def goal_potential(env, info: dict) -> float:
    """Phi(s) = -(Manhattan distance to the nearest current goal object) / room size.

    The current goal is the first object named in the mission, or the second
    once the agent carries something in a two-object task (putnext,
    pickup_then_goto). Returns 0 when no matching object is on the floor.
    Reads ``agent_pos`` and ``carrying`` from the MiniGrid level, which
    babytext does not expose in ``info``.
    """
    level = env.unwrapped
    goals = info["goal_positions"]
    target = 1 if level.carrying is not None and len(goals) > 1 else 0
    positions = goals[target]
    if not positions:
        return 0.0
    ax, ay = level.agent_pos
    distance = min(abs(ax - x) + abs(ay - y) for x, y in positions)
    return -distance / ENV.room_size


@dataclass
class StepResult:
    """What ``VecTextEnv.step`` returns, one entry per environment.

    ``obs`` and ``t`` describe the state the *next* action is taken in: for an
    environment whose episode just ended, that is the first observation of its
    new episode (``reset`` is then True and ``t`` is 0).
    """

    obs: list[dict]
    t: np.ndarray  # (N,) step index of obs within its episode
    reset: np.ndarray  # (N,) True if obs starts a new episode
    reward: np.ndarray  # (N,) shaped reward for the action just taken
    env_reward: np.ndarray  # (N,) environment reward alone
    done: np.ndarray  # (N,) the episode ended with the action just taken
    success: np.ndarray  # (N,) ... and was solved


class VecTextEnv:
    """N babytext environments with automatic reset.

    Args:
        tasks: tasks to train on; environment i runs ``tasks[i % len(tasks)]``.
        num_envs: N.
        eval_set: seed split and generator ("train" for RL).
        first_seed_index: index of the first seed used per task. Defaults to
            ``config.DATA.n_train_demos`` so RL rooms differ from the demos.
        gamma, shaping: discount and weight of the potential-based shaping.
    """

    def __init__(
        self,
        tasks: list[str],
        num_envs: int,
        eval_set: str = "train",
        first_seed_index: int | None = None,
        gamma: float = 0.99,
        shaping: float = 0.0,
    ):
        self.tasks = [tasks[i % len(tasks)] for i in range(num_envs)]
        self.eval_set = eval_set
        self.envs = [make_env(task, eval_set) for task in self.tasks]
        self.gamma, self.shaping = gamma, shaping
        self._next_seed = {task: DATA.n_train_demos if first_seed_index is None else first_seed_index for task in tasks}
        self._obs: list[dict] = []
        self._potential = np.zeros(num_envs)
        self._t = np.zeros(num_envs, dtype=np.int64)

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def _reset_one(self, i: int) -> dict:
        task = self.tasks[i]
        seed = get_seeds(task, self.eval_set, 1, start=self._next_seed[task])[0]
        self._next_seed[task] += 1
        obs, info = self.envs[i].reset(seed=seed)
        self._potential[i] = goal_potential(self.envs[i], info)
        self._t[i] = 0
        return obs

    def reset(self) -> StepResult:
        """Start a new episode in every environment."""
        self._obs = [self._reset_one(i) for i in range(self.num_envs)]
        n = self.num_envs
        zeros = np.zeros(n)
        return StepResult(self._obs, self._t.copy(), np.ones(n, bool), zeros, zeros, np.zeros(n, bool), np.zeros(n, bool))

    def step(self, actions: np.ndarray) -> StepResult:
        """Apply one action per environment; finished episodes restart on a new seed."""
        n = self.num_envs
        reward, env_reward = np.zeros(n), np.zeros(n)
        done, success, reset = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, r, terminated, truncated, info = env.step(int(action))
            ended = terminated or truncated
            next_potential = goal_potential(env, info)
            reward[i] = shaped_reward(r, self._potential[i], next_potential, ended, self.gamma, self.shaping)
            env_reward[i], done[i], success[i] = r, ended, terminated and r > 0
            if ended:
                obs, reset[i] = self._reset_one(i), True
            else:
                self._potential[i] = next_potential
                self._t[i] += 1
            self._obs[i] = obs
        return StepResult(list(self._obs), self._t.copy(), reset, reward, env_reward, done, success)

    def close(self) -> None:
        for env in self.envs:
            env.close()
