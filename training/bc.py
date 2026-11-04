"""Behaviour cloning, one loop for every trained condition (B3 and the action head).

Sharing the loop makes the protocol identical by construction: the same
batches of whole episodes for the same seed (``demos.batch_indices``), AdamW
with ``config.BC``'s epochs, lr, weight decay and gradient clipping, the same
metrics, and the same selection rule:

1. after every epoch, cross-entropy and accuracy on the val demos (eval mode);
2. the epoch with the lowest val cross-entropy is kept as ``best.pt``
   (``last.pt`` is the latest epoch);
3. the learning rate decays when val cross-entropy stalls (``cfg.lr_decay``),
   and training stops after ``cfg.patience`` epochs without improvement, so
   every model trains until it stops improving rather than for a fixed count;
4. greedy success on ``cfg.val_episodes`` val episodes per task is measured
   once, on the kept epoch, and stored with it. Configurations (e.g. layers
   and poolings of the head) are compared by that success.

Each condition supplies its data as a ``BCData`` and its own policy and
checkpoint functions.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BC, BCConfig

from .common import RunLogger, validate
from .demos import batch_indices


class BCData(Protocol):
    """Demonstration steps grouped by episode, as a model's inputs."""

    def lengths(self) -> list[int]:
        """Steps per episode."""

    def batch(self, episodes, device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Every step of the given episodes: the model's keyword inputs (leading dim S) and the (S,) expert actions."""


def bc_loss(model: nn.Module, inputs: dict[str, torch.Tensor], actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean cross-entropy over the S steps of a batch, and metrics (``ce``, ``accuracy``).

    Args:
        inputs: keyword arguments of ``model``, which returns (S, 6) logits.
        actions: (S,) expert actions.
    """
    logits = model(**inputs)
    ce = F.cross_entropy(logits, actions)
    accuracy = (logits.argmax(-1) == actions).float().mean()
    return ce, {"ce": ce.item(), "accuracy": accuracy.item()}


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: BCConfig):
    """The learning-rate schedule named by ``cfg.lr_decay``; None for "none".

    "plateau" is stepped with the epoch's val cross-entropy, "cosine" once per epoch.
    """
    if cfg.lr_decay == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, threshold=0.0, min_lr=cfg.min_lr)
    if cfg.lr_decay == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.min_lr)
    if cfg.lr_decay == "none":
        return None
    raise ValueError(f"lr_decay must be 'plateau', 'cosine' or 'none', got {cfg.lr_decay!r}")


@torch.no_grad()
def evaluate_bc(model: nn.Module, data: BCData, cfg: BCConfig, device) -> dict[str, float]:
    """``val_ce`` and ``val_accuracy`` over all steps of ``data``, in eval mode."""
    model.eval()
    totals, steps = {}, 0
    for idx in batch_indices(data.lengths(), cfg.batch_episodes):
        inputs, actions = data.batch(idx, device)
        _, metrics = bc_loss(model, inputs, actions)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v * len(actions)
        steps += len(actions)
    model.train()
    return {f"val_{k}": v / steps for k, v in totals.items()}


def train_bc(
    model: nn.Module,
    train: BCData,
    val: BCData,
    tasks: list[str],
    out_dir: str | Path,
    make_policy: Callable[[nn.Module], object],
    save: Callable[..., None],
    cfg: BCConfig = BC,
    device: str | torch.device = "cpu",
    seed: int = 0,
    log_extra: dict | None = None,
) -> tuple[nn.Module, dict]:
    """Train ``model`` by behaviour cloning under the shared protocol (module docstring).

    Args:
        make_policy: the trained model -> a babytext policy, for the val episodes.
        save: ``save(model, path, **extra)`` writes a checkpoint.
        log_extra: constant fields added to every metrics line (e.g. the model type).

    Returns:
        The model with the kept epoch's weights, and a summary: ``selected_epoch``,
        that epoch's metrics, ``epochs_run``, ``stopped_early``, and (if measured)
        ``val_success`` and ``val_<task>``.
    """
    out_dir = Path(out_dir)
    log_extra = log_extra or {}
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg)
    logger = RunLogger(out_dir)
    rng = np.random.default_rng(seed)
    best, best_state, summary, stale = float("inf"), None, {}, 0

    for epoch in range(cfg.epochs):
        lr = optimizer.param_groups[0]["lr"]
        running = []
        for idx in batch_indices(train.lengths(), cfg.batch_episodes, rng):
            loss, metrics = bc_loss(model, *train.batch(idx, device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            running.append(metrics)

        scores = {k: float(np.mean([m[k] for m in running])) for k in running[0]}
        scores.update(evaluate_bc(model, val, cfg, device), lr=lr)
        logger.log(**log_extra, epoch=epoch, **scores)
        save(model, out_dir / "last.pt", epoch=epoch, **scores)
        if scores["val_ce"] < best:
            best, best_state, stale = scores["val_ce"], copy.deepcopy(model.state_dict()), 0
            summary = {"selected_epoch": epoch, **scores}
        else:
            stale += 1
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(scores["val_ce"])
        elif scheduler is not None:
            scheduler.step()
        if cfg.patience > 0 and stale >= cfg.patience:
            break

    summary.update(epochs_run=epoch + 1, stopped_early=epoch + 1 < cfg.epochs)
    model.load_state_dict(best_state)
    if cfg.val_episodes > 0:
        summary.update(validate(make_policy(model), tasks, cfg.val_episodes))
        model.train()
    logger.log(**log_extra, **summary)
    save(model, out_dir / "best.pt", epoch=summary["selected_epoch"], **summary)
    return model, summary
