"""Space-time attention: reads H vectors from the current view and the memory.

The current view and the K memory slots form a grid of (K + 1) x P
positions (time slot x cell). A small convolutional scorer, shared over slots
and conditioned on the mission, gives every position a score per read head;
a softmax over all (K + 1) * P positions turns them into the 2-D space-time
weight matrix A^h, and each head returns a weighted sum of the features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import TimeEncoding
from .memory import MemoryState


class SpaceTimeAttention(nn.Module):
    """H read heads over the (K + 1) x P space-time grid.

    Args:
        dim: feature width d.
        mission_dim: width of the mission vector that conditions the scorer.
        heads: H, the number of read-outs (and of soft tokens for the SLM).
        view_size: V (the grid is V x V cells, P = V * V).
        time: the shared time encoding, applied to each slot's age.
    """

    def __init__(self, dim: int, mission_dim: int, heads: int, view_size: int, time: TimeEncoding):
        super().__init__()
        self.view_size = view_size
        self.time = time
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.film = nn.Linear(mission_dim, 2 * dim)  # mission -> per-channel scale and shift
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.score = nn.Conv2d(dim, heads, 1)
        self.value = nn.Linear(dim, dim)

    def forward(
        self, view: torch.Tensor, memory: MemoryState, mission: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read from [current view; memory].

        Args:
            view: (B, P, d) current view features.
            memory: memory before this step (K slots, possibly K = 0).
            mission: (B, mission_dim).

        Returns:
            readouts: (B, H, d).
            weights: (B, H, K + 1, P), the space-time weight matrix per head;
                row 0 is the current view, row k the memory slot k - 1.
                Sums to 1 over the last two axes.
        """
        b, p, d = view.shape
        grid = torch.cat([view[:, None], memory.slots], dim=1)  # (B, K+1, P, d)
        k1 = grid.shape[1]
        ages = torch.cat([view.new_zeros(b, 1), memory.age], dim=1)  # (B, K+1)
        grid = grid + self.time(ages)[:, :, None]

        x = grid.reshape(b * k1, p, d).transpose(1, 2).reshape(b * k1, d, self.view_size, self.view_size)
        gamma, beta = self.film(mission).repeat_interleave(k1, dim=0)[:, :, None, None].chunk(2, dim=1)
        x = F.relu(self.conv(x) * (1 + gamma) + beta)
        scores = self.score(x).flatten(2).reshape(b, k1, -1, p).transpose(1, 2)  # (B, H, K+1, P)

        # Slots never written get (almost) no weight.
        fill = torch.cat([view.new_ones(b, 1), memory.fill], dim=1)
        scores = scores + torch.log(fill.clamp_min(1e-6))[:, None, :, None]

        h = scores.shape[1]
        weights = torch.softmax(scores.reshape(b, h, -1), dim=-1).reshape(b, h, k1, p)
        readouts = torch.einsum("bhkp,bkpd->bhd", weights, self.value(grid))
        return readouts, weights
