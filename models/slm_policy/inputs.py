"""Model inputs: babytext observations as (batch, time) tensors.

Every forward pass of the SLM policy takes an ``Observations`` block of shape
(B, T): B independent sequences (episodes or rollout chunks) of T steps each.
A rollout step is the special case T = 1, so acting and training share one
code path (see ``SLMPolicy.forward``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass
class Observations:
    """A (B, T) block of observations.

    Attributes:
        image: (B, T, V, V, 3) long, MiniGrid's symbolic view (``obs["image"]``).
        direction: (B, T) long, absolute heading 0-3 (``obs["direction"]``).
        t: (B, T) long, step index within the episode (0 at reset).
        reset: (B, T) bool, True where a new episode starts; the memory is
            cleared before that step is read.
        missions: B lists of T mission strings (``obs["mission"]``).
    """

    image: torch.Tensor
    direction: torch.Tensor
    t: torch.Tensor
    reset: torch.Tensor
    missions: list[list[str]]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.direction.shape)

    def to(self, device: str | torch.device) -> Observations:
        return replace(
            self,
            image=self.image.to(device),
            direction=self.direction.to(device),
            t=self.t.to(device),
            reset=self.reset.to(device),
        )

    def select(self, rows: torch.Tensor | list[int]) -> Observations:
        """The sub-block with only the given sequences (rows of B)."""
        rows = torch.as_tensor(rows, dtype=torch.long)
        return Observations(
            image=self.image[rows],
            direction=self.direction[rows],
            t=self.t[rows],
            reset=self.reset[rows],
            missions=[self.missions[i] for i in rows.tolist()],
        )


def observations_from_env(obs_list: list[dict], t: list[int] | np.ndarray, reset: list[bool] | np.ndarray) -> Observations:
    """One time step (T = 1) for B environments, from babytext observation dicts.

    Args:
        obs_list: B observation dicts as returned by ``env.reset`` / ``env.step``.
        t: B step indices within each environment's current episode.
        reset: B flags, True if that observation is the first of an episode.
    """
    return Observations(
        image=torch.from_numpy(np.stack([o["image"] for o in obs_list])).long()[:, None],
        direction=torch.tensor([int(o["direction"]) for o in obs_list], dtype=torch.long)[:, None],
        t=torch.as_tensor(np.asarray(t), dtype=torch.long)[:, None],
        reset=torch.as_tensor(np.asarray(reset), dtype=torch.bool)[:, None],
        missions=[[o["mission"]] for o in obs_list],
    )


@dataclass
class MissionTable:
    """The distinct missions of a block and, per step, which one applies.

    Missions repeat across every step of an episode, so each distinct string is
    encoded once (by the GRU and by the SLM tokenizer) and looked up by index.
    """

    texts: list[str]  # U distinct missions
    index: torch.Tensor  # (B, T) long, into ``texts``

    @classmethod
    def from_missions(cls, missions: list[list[str]], device: str | torch.device = "cpu") -> MissionTable:
        lookup: dict[str, int] = {}
        index = [[lookup.setdefault(m, len(lookup)) for m in row] for row in missions]
        return cls(texts=list(lookup), index=torch.tensor(index, dtype=torch.long, device=device))
