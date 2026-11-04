"""Encoders: time, the 7x7 view, and the mission.

All three run independently of the memory and the SLM, and the view encoder
has no recurrence, so every step of a (B, T) block is encoded in one batch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from torch.nn.utils.rnn import pack_padded_sequence

from models.symbolic_agent import MissionTokenizer


class TimeEncoding(nn.Module):
    """Sinusoidal features of a step count, followed by a learned linear map.

    Accepts fractional counts, because a slot's age is a soft (gated) quantity.
    Used for the step index t of the current view and the age of each slot.
    """

    def __init__(self, dim: int, num_freqs: int = 16, max_period: float = 256.0):
        super().__init__()
        freqs = torch.exp(-math.log(max_period) * torch.arange(num_freqs) / num_freqs)
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(2 * num_freqs, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """(...,) step counts -> (..., dim) encodings."""
        angles = t.float()[..., None] * self.freqs
        return self.proj(torch.cat([angles.sin(), angles.cos()], dim=-1))


class ViewEncoder(nn.Module):
    """The spatial encoder: a 7x7 symbolic view -> one feature vector per cell.

    Each cell's (object, colour, state) codes are embedded and summed with a
    learned embedding of the cell's position (the view is egocentric, so
    "2 cells in front of me" is a fixed position). Residual 3x3 convolutions
    then mix neighbouring cells. The time encoding of t, and the heading if
    ``use_direction`` is True (off by default, as B3 gets no heading), are
    added to every cell.
    """

    def __init__(self, dim: int, num_layers: int, view_size: int, time: TimeEncoding, use_direction: bool = False):
        super().__init__()
        self.view_size = view_size
        self.obj_emb = nn.Embedding(len(OBJECT_TO_IDX), dim)
        self.color_emb = nn.Embedding(len(COLOR_TO_IDX), dim)
        self.state_emb = nn.Embedding(3, dim)
        self.cell_pos = nn.Parameter(torch.zeros(view_size * view_size, dim))
        self.direction_emb = nn.Embedding(4, dim) if use_direction else None
        self.convs = nn.ModuleList(nn.Conv2d(dim, dim, 3, padding=1) for _ in range(num_layers))
        self.norm = nn.LayerNorm(dim)
        self.time = time

    def forward(self, image: torch.Tensor, direction: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Encode N views.

        Args:
            image: (N, V, V, 3) long.
            direction: (N,) long; ignored when ``use_direction`` is False.
            t: (N,) long step index.

        Returns:
            (N, V*V, dim) features, cells in row-major order of MiniGrid's
            ``image[x, y]`` indexing (x across, y forward-to-back).
        """
        image = image.long()
        cells = self.obj_emb(image[..., 0]) + self.color_emb(image[..., 1]) + self.state_emb(image[..., 2])
        x = cells.permute(0, 3, 1, 2)  # (N, dim, V, V)
        for conv in self.convs:
            x = x + conv(F.relu(x))
        x = x.flatten(2).transpose(1, 2)  # (N, P, dim)
        per_view = self.time(t)
        if self.direction_emb is not None:
            per_view = per_view + self.direction_emb(direction)
        x = x + self.cell_pos + per_view[:, None]
        return self.norm(x)


class MissionEncoder(nn.Module):
    """Mission string -> vector, with a word GRU over a fixed vocabulary.

    It is deliberately independent of the SLM: the memory gate and the
    attention scorer need the mission at every step, and keeping the SLM out
    of that recurrent path is what lets the SLM run on all T steps in parallel.
    Invented words (``test_nonce``) become ``<unk>``.
    """

    def __init__(self, word_dim: int, out_dim: int):
        super().__init__()
        self.tokenizer = MissionTokenizer()
        self.word_emb = nn.Embedding(len(self.tokenizer), word_dim, padding_idx=MissionTokenizer.PAD)
        self.rnn = nn.GRU(word_dim, out_dim, batch_first=True)

    def forward(self, texts: list[str]) -> torch.Tensor:
        """U mission strings -> (U, out_dim)."""
        tokens, lengths = self.tokenizer.encode(texts)
        tokens = tokens.to(self.word_emb.weight.device)
        packed = pack_padded_sequence(self.word_emb(tokens), lengths, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return h[-1]
