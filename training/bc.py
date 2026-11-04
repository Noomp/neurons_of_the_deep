"""Behaviour cloning on whole demonstration episodes (training stages 1-3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import BC, BCConfig
from models.slm_policy import SLMPolicy, save_model

from .common import RunLogger, validate
from .demos import DemoBatch, DemoEpisode, iterate_batches
from .stages import Stage, make_optimizer


def bc_loss(model: SLMPolicy, batch: DemoBatch, write_penalty: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy on expert actions plus a penalty on the memory write rate.

    Padded steps are excluded from both terms. Returns the loss and metrics
    (``ce``, ``accuracy``, ``write_rate``) for logging.
    """
    out = model(batch.obs, model.initial_memory(batch.actions.shape[0]))
    mask = batch.mask
    ce = F.cross_entropy(out.logits[mask], batch.actions[mask])
    write_rate = out.write_gate[mask].mean()
    loss = ce + write_penalty * write_rate
    accuracy = (out.logits[mask].argmax(-1) == batch.actions[mask]).float().mean()
    return loss, {"ce": ce.item(), "accuracy": accuracy.item(), "write_rate": write_rate.item()}


@torch.no_grad()
def evaluate_bc(model: SLMPolicy, episodes: list[DemoEpisode], cfg: BCConfig, device) -> dict[str, float]:
    """Mean ``bc_loss`` metrics over held-out demonstrations."""
    model.eval()
    totals, steps = {}, 0
    for batch in iterate_batches(episodes, cfg.batch_episodes):
        batch = batch.to(device)
        _, metrics = bc_loss(model, batch, cfg.write_penalty)
        n = int(batch.mask.sum())
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v * n
        steps += n
    model.train()
    return {f"val_{k}": v / steps for k, v in totals.items()}


def train_bc(
    model: SLMPolicy,
    stage: Stage,
    train_episodes: list[DemoEpisode],
    val_episodes: list[DemoEpisode],
    tasks: list[str],
    out_dir: str | Path,
    cfg: BCConfig = BC,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> SLMPolicy:
    """Train ``model`` by behaviour cloning and keep the best checkpoint.

    After every epoch: demo loss on ``val_episodes``, and (if
    ``cfg.val_episodes`` > 0) greedy success on that many val seeds per task.
    The checkpoint with the best val success (or lowest val loss when success
    is not measured) is saved as ``<out_dir>/best.pt``; ``last.pt`` is saved
    every epoch.

    Returns:
        The model as trained after the last epoch (load ``best.pt`` for the selected one).
    """
    out_dir = Path(out_dir)
    model.to(device).train()
    optimizer = make_optimizer(model, stage, cfg.lr, cfg.weight_decay)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    logger = RunLogger(out_dir)
    rng = np.random.default_rng(seed)
    best = -float("inf")

    for epoch in range(cfg.epochs):
        running = []
        for batch in iterate_batches(train_episodes, cfg.batch_episodes, rng):
            loss, metrics = bc_loss(model, batch.to(device), cfg.write_penalty)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            running.append(metrics)

        scores = {k: float(np.mean([m[k] for m in running])) for k in running[0]}
        scores.update(evaluate_bc(model, val_episodes, cfg, device))
        if cfg.val_episodes > 0:
            scores.update(validate(model, tasks, cfg.val_episodes, device))
        logger.log(stage=stage.number, epoch=epoch, **scores)

        save_model(model, out_dir / "last.pt", stage=stage.number, epoch=epoch, **scores)
        score = scores.get("val_success", -scores["val_ce"])
        if score > best:
            best = score
            save_model(model, out_dir / "best.pt", stage=stage.number, epoch=epoch, **scores)
    return model
