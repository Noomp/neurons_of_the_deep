"""Behaviour cloning for B3, the symbolic agent, under the SLM policy's protocol.

Same as ``training.bc`` wherever B3 allows it: the same demo episodes, the
same batches of whole episodes for the same seed (``demos.batch_indices``),
the same ``config.BC`` optimiser settings (epochs, lr, weight decay, grad
clip; AdamW), and the same selection rule and metric names. B3 has no
memory, so there is no write penalty, and each step is an independent
sample: a batch is every step of ``batch_episodes`` episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import BC, BCConfig
from models.symbolic_agent import SymbolicAgent, SymbolicPolicy, save_model, stack_demo_steps

from .common import RunLogger, validate
from .demos import DemoEpisode, batch_indices


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
    the same ``DemoEpisode`` list the SLM trainer uses."""
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


def symbolic_bc_loss(
    model: SymbolicAgent, inputs: dict[str, torch.Tensor], actions: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean cross-entropy over steps, and metrics (``ce``, ``accuracy``) for logging.

    Args:
        inputs: from ``SymbolicDemos.batch``, S steps.
        actions: (S,) expert actions.
    """
    logits = model(**inputs)  # (S, 6)
    ce = F.cross_entropy(logits, actions)
    accuracy = (logits.argmax(-1) == actions).float().mean()
    return ce, {"ce": ce.item(), "accuracy": accuracy.item()}


@torch.no_grad()
def evaluate_symbolic_bc(model: SymbolicAgent, demos: SymbolicDemos, cfg: BCConfig, device) -> dict[str, float]:
    """Mean ``symbolic_bc_loss`` metrics over held-out demonstrations, in eval mode."""
    model.eval()
    totals, steps = {}, 0
    for idx in batch_indices(demos.lengths(), cfg.batch_episodes):
        inputs, actions = demos.batch(idx, device)
        _, metrics = symbolic_bc_loss(model, inputs, actions)
        n = len(actions)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v * n
        steps += n
    model.train()
    return {f"val_{k}": v / steps for k, v in totals.items()}


def train_symbolic_bc(
    model: SymbolicAgent,
    train_episodes: list[DemoEpisode],
    val_episodes: list[DemoEpisode],
    tasks: list[str],
    out_dir: str | Path,
    cfg: BCConfig = BC,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> SymbolicAgent:
    """Train B3 by behaviour cloning and keep the best checkpoint.

    After every epoch: demo loss on ``val_episodes``, and (if
    ``cfg.val_episodes`` > 0) greedy success on that many val seeds per task,
    both in eval mode. The checkpoint with the best val success (or lowest
    val loss when success is not measured) is saved as ``<out_dir>/best.pt``;
    ``last.pt`` is saved every epoch. Checkpoints are
    ``models.symbolic_agent.save_model`` files.

    Returns:
        The model as trained after the last epoch (load ``best.pt`` for the selected one).
    """
    out_dir = Path(out_dir)
    model.to(device).train()
    train, val = symbolic_demos(train_episodes, model), symbolic_demos(val_episodes, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger = RunLogger(out_dir)
    rng = np.random.default_rng(seed)
    best = -float("inf")

    for epoch in range(cfg.epochs):
        running = []
        for idx in batch_indices(train.lengths(), cfg.batch_episodes, rng):
            loss, metrics = symbolic_bc_loss(model, *train.batch(idx, device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running.append(metrics)

        scores = {k: float(np.mean([m[k] for m in running])) for k in running[0]}
        scores.update(evaluate_symbolic_bc(model, val, cfg, device))
        if cfg.val_episodes > 0:
            scores.update(validate(SymbolicPolicy(model, device), tasks, cfg.val_episodes))
            model.train()
        logger.log(model_type="b3", epoch=epoch, **scores)

        save_model(model, out_dir / "last.pt", model_type="b3", epoch=epoch, **scores)
        score = scores.get("val_success", -scores["val_ce"])
        if score > best:
            best = score
            save_model(model, out_dir / "best.pt", model_type="b3", epoch=epoch, **scores)
    return model
