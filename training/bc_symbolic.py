"""B3's demonstration data, and its trainer: the shared loop of ``training.bc``.

Each step is one independent sample: the frame-stacked inputs of
``stack_demo_steps`` (the last 3 views and 2 actions, as in the GLAM prompt)
and the mission. A batch is every step of ``batch_episodes`` episodes, the
same batches as the action head's for the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config import BC, BCConfig
from models.symbolic_agent import SymbolicAgent, SymbolicPolicy, save_model, stack_demo_steps

from .bc import train_bc
from .demos import DemoEpisode


@dataclass
class SymbolicDemos:
    """Every step of E demo episodes as B3 inputs (S steps in total), on the CPU."""

    frames: torch.Tensor  # (S, N, V, V, 3) uint8, oldest view first
    frame_mask: torch.Tensor  # (S, N) bool
    prev_actions: torch.Tensor  # (S, N - 1) long
    actions: torch.Tensor  # (S,) long, the expert's action
    mission_tokens: torch.Tensor  # (E, L) long, one mission per episode
    mission_lengths: torch.Tensor  # (E,) long
    starts: np.ndarray  # (E + 1,) first step of each episode, then S

    def __len__(self) -> int:
        return len(self.starts) - 1

    def lengths(self) -> list[int]:
        return np.diff(self.starts).tolist()

    def batch(self, episodes: list[int] | np.ndarray, device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """All steps of the given episodes: ``SymbolicAgent.forward`` keyword
        arguments, each with a leading dim of their total step count, and the (steps,) expert actions."""
        episodes = np.asarray(episodes, dtype=np.int64)
        lengths = np.diff(self.starts)[episodes]
        rows = torch.from_numpy(np.concatenate([np.arange(self.starts[e], self.starts[e + 1]) for e in episodes]))
        owner = torch.from_numpy(np.repeat(episodes, lengths))
        inputs = {
            "frames": self.frames[rows],
            "frame_mask": self.frame_mask[rows],
            "prev_actions": self.prev_actions[rows],
            "mission_tokens": self.mission_tokens[owner],
            "mission_lengths": self.mission_lengths[owner],
        }
        return {k: v.to(device) for k, v in inputs.items()}, self.actions[rows].to(device)


def symbolic_demos(episodes: list[DemoEpisode], model: SymbolicAgent) -> SymbolicDemos:
    """Build B3's frame-stacked inputs (``stack_demo_steps``) for ``episodes``,
    the same ``DemoEpisode`` list the action head's data comes from."""
    starts = np.concatenate([[0], np.cumsum([len(e) for e in episodes])])
    stacked = stack_demo_steps(
        np.concatenate([e.image for e in episodes]),
        np.concatenate([e.action for e in episodes]),
        np.concatenate([np.arange(len(e)) for e in episodes]),
        model.num_frames,
    )
    tokens, lengths = model.tokenizer.encode([e.mission for e in episodes])
    return SymbolicDemos(
        frames=torch.from_numpy(stacked["frames"]),
        frame_mask=torch.from_numpy(stacked["frame_mask"]),
        prev_actions=torch.from_numpy(stacked["prev_actions"]),
        actions=torch.from_numpy(np.concatenate([e.action for e in episodes]).astype(np.int64)),
        mission_tokens=tokens,
        mission_lengths=lengths,
        starts=starts,
    )


def train_symbolic_bc(
    model: SymbolicAgent,
    train_episodes: list[DemoEpisode],
    val_episodes: list[DemoEpisode],
    tasks: list[str],
    out_dir: str | Path,
    cfg: BCConfig = BC,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[SymbolicAgent, dict]:
    """Train B3 with ``training.bc.train_bc``; checkpoints are ``models.symbolic_agent.save_model`` files.

    Returns:
        The model with the selected epoch's weights, and the selection summary.
    """
    return train_bc(
        model,
        symbolic_demos(train_episodes, model),
        symbolic_demos(val_episodes, model),
        tasks,
        out_dir,
        make_policy=lambda m: SymbolicPolicy(m, device),
        save=lambda m, path, **extra: save_model(m, path, model_type="b3", **extra),
        cfg=cfg,
        device=device,
        seed=seed,
        log_extra={"model_type": "b3"},
    )
