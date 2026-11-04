"""The action head's demonstration data (from the hidden-state cache), and its trainer.

Training reads cached hidden states only (``training.hidden_cache``); the LLM
runs just once at the end, truncated to the head's layer, for the val
episodes. The loop, batches and selection rule are ``training.bc``'s, shared
with B3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config import BC, LLM, BCConfig, LLMConfig
from models.action_head import ActionHead, HeadPolicy, save_model
from models.frozen_lm import FrozenLM

from .bc import train_bc
from .demos import DemoEpisode
from .hidden_cache import check_cache, from_bits, load_pooled, open_sequences


@dataclass
class HeadDemos:
    """Every step of E demo episodes (S steps) as the head's cached inputs, on the CPU.

    Exactly one of ``vectors`` (last / mean pooling) and ``sequences``
    (attention pooling) is set.
    """

    actions: torch.Tensor  # (S,) long, the expert's actions
    starts: np.ndarray  # (E + 1,) first step of each episode, then S
    vectors: torch.Tensor | None = None  # (S, D) bfloat16
    # Per dataset: (memory-mapped token bits, step offsets, first step of the dataset in this object).
    sequences: list[tuple[np.ndarray, np.ndarray, int]] | None = None

    def lengths(self) -> list[int]:
        return np.diff(self.starts).tolist()

    def batch(self, episodes, device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """All steps of ``episodes``: ``{"features": (S, D)}`` or, for attention,
        ``{"features": (S, L, D), "mask": (S, L)}`` (right-padded), and the (S,) actions."""
        rows = np.concatenate([np.arange(self.starts[e], self.starts[e + 1]) for e in episodes])
        actions = self.actions[rows].to(device)
        if self.vectors is not None:
            return {"features": self.vectors[rows].to(device)}, actions
        firsts = [first for _, _, first in self.sequences]
        blocks = []
        for row in rows:
            k = int(np.searchsorted(firsts, row, side="right")) - 1
            tokens, offsets, first = self.sequences[k]
            local = row - first
            blocks.append(from_bits(tokens[offsets[local]:offsets[local + 1]]))
        length = max(len(b) for b in blocks)
        features = torch.zeros(len(blocks), length, blocks[0].shape[1], dtype=torch.bfloat16)
        mask = torch.zeros(len(blocks), length, dtype=torch.bool)
        for i, b in enumerate(blocks):
            features[i, : len(b)], mask[i, : len(b)] = b, True
        return {"features": features.to(device), "mask": mask.to(device)}, actions


def head_demos(
    episodes: dict[str, list[DemoEpisode]],
    split: str,
    directory: Path,
    demos_dir: str | Path,
    llm: LLMConfig,
    layer: int,
    pooling: str,
) -> tuple[HeadDemos, dict[str, dict]]:
    """The cached inputs of ``episodes`` (per task, the first episodes of ``<task>_<split>``).

    Returns:
        The data, and each dataset's cache meta data (for ``run.json``).

    Raises:
        hidden_cache.CacheError: a cache is missing or does not match.
    """
    sequences = pooling == "attention"
    vectors, seqs, metas, actions, lengths, first = [], [], {}, [], [], 0
    for task, eps in episodes.items():
        prefix = f"{task}_{split}"
        meta = check_cache(directory, prefix, demos_dir, llm, layer, sequences)
        metas[prefix] = meta
        rows = sum(len(e) for e in eps)
        if sequences:
            tokens, offsets = open_sequences(directory, prefix, layer)
            seqs.append((tokens, offsets, first))
        else:
            vectors.append(load_pooled(directory, prefix, layer, pooling, meta, rows))
        actions += [torch.from_numpy(e.action.astype(np.int64)) for e in eps]
        lengths += [len(e) for e in eps]
        first += rows
    data = HeadDemos(
        actions=torch.cat(actions),
        starts=np.concatenate([[0], np.cumsum(lengths)]),
        vectors=None if sequences else torch.cat(vectors),
        sequences=seqs if sequences else None,
    )
    return data, metas


def train_head_bc(
    head: ActionHead,
    train: HeadDemos,
    val: HeadDemos,
    tasks: list[str],
    out_dir: str | Path,
    llm: LLMConfig = LLM,
    cfg: BCConfig = BC,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[ActionHead, dict]:
    """Train the head with ``training.bc.train_bc``; checkpoints are ``models.action_head.save_model`` files.

    The val episodes run the frozen LLM ``llm`` truncated to ``head.cfg.layer``.

    Returns:
        The head with the selected epoch's weights, and the selection summary.
    """
    return train_bc(
        head,
        train,
        val,
        tasks,
        out_dir,
        make_policy=lambda h: HeadPolicy(h, FrozenLM.load(llm, num_layers=h.cfg.layer, device=device), device),
        save=lambda h, path, **extra: save_model(h, path, llm, model_type="head", **extra),
        cfg=cfg,
        device=device,
        seed=seed,
        log_extra={"model_type": "head"},
    )
