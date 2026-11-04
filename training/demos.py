"""Demonstration episodes as padded (B, T) training batches for behaviour cloning.

The policy carries memory across time, so batches hold whole episodes in time
order (full back-propagation through time; expert episodes are short).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from babytext.data import load_episodes, load_symbolic
from models.slm_policy import Observations


@dataclass
class DemoEpisode:
    """One expert episode: per-step arrays of length T and its mission."""

    image: np.ndarray  # (T, V, V, 3) uint8
    direction: np.ndarray  # (T,)
    action: np.ndarray  # (T,)
    mission: str

    def __len__(self) -> int:
        return len(self.action)


@dataclass
class DemoBatch:
    """Padded episodes. ``mask`` is False on padding after an episode ends."""

    obs: Observations
    actions: torch.Tensor  # (B, T) long
    mask: torch.Tensor  # (B, T) bool

    def to(self, device) -> DemoBatch:
        return DemoBatch(self.obs.to(device), self.actions.to(device), self.mask.to(device))


def load_demo_episodes(prefix: str | Path, limit: int | None = None) -> list[DemoEpisode]:
    """Episodes of a demo dataset written by ``training.tools.collect_demos``.

    Args:
        prefix: e.g. ``data/demos/goto_train`` (without extension).
        limit: keep only the first ``limit`` episodes (demonstration-count curves).
    """
    arrays = load_symbolic(prefix)
    records = load_episodes(prefix)
    if limit is not None:
        records = records[:limit]
    starts = np.flatnonzero(arrays["t"] == 0)
    ends = np.append(starts[1:], len(arrays["t"]))
    return [
        DemoEpisode(
            image=arrays["image"][s:e],
            direction=arrays["direction"][s:e],
            action=arrays["action"][s:e],
            mission=rec["mission"],
        )
        for rec, s, e in zip(records, starts, ends)
    ]


def collate(episodes: list[DemoEpisode]) -> DemoBatch:
    """Pad episodes with zeros to the longest one; padded steps are masked out."""
    b, t_len = len(episodes), max(len(e) for e in episodes)
    view = episodes[0].image.shape[1:]
    image = np.zeros((b, t_len, *view), dtype=np.int64)
    direction = np.zeros((b, t_len), dtype=np.int64)
    actions = np.zeros((b, t_len), dtype=np.int64)
    mask = np.zeros((b, t_len), dtype=bool)
    for i, ep in enumerate(episodes):
        n = len(ep)
        image[i, :n], direction[i, :n], actions[i, :n], mask[i, :n] = ep.image, ep.direction, ep.action, True
    reset = np.zeros((b, t_len), dtype=bool)
    reset[:, 0] = True
    obs = Observations(
        image=torch.from_numpy(image),
        direction=torch.from_numpy(direction),
        t=torch.arange(t_len).expand(b, t_len).clone(),
        reset=torch.from_numpy(reset),
        missions=[[ep.mission] * t_len for ep in episodes],
    )
    return DemoBatch(obs, torch.from_numpy(actions), torch.from_numpy(mask))


def iterate_batches(episodes: list[DemoEpisode], batch_size: int, rng: np.random.Generator | None = None):
    """Yield ``DemoBatch``es; shuffled if ``rng`` is given.

    Episodes of similar length are grouped (after shuffling, within chunks of
    50 batches) to waste less compute on padding.
    """
    order = rng.permutation(len(episodes)) if rng is not None else np.arange(len(episodes))
    chunk = batch_size * 50
    for c in range(0, len(order), chunk):
        part = sorted(order[c : c + chunk], key=lambda i: len(episodes[i]))
        batches = [part[i : i + batch_size] for i in range(0, len(part), batch_size)]
        if rng is not None:
            rng.shuffle(batches)
        for idx in batches:
            yield collate([episodes[i] for i in idx])
