"""Recurrent PPO in the environment (training stage 4).

Each update collects ``rollout_len`` steps from ``num_envs`` environments.
Every environment's rollout is one recurrent sequence: the memory at its
start is stored, and during the update the whole sequence is replayed
through the model from that memory (episodes that end inside it are cut by
``Observations.reset``). Gradients from the delayed (GAE) return therefore
reach the memory gate and the space-time attention through time, within one
rollout (truncated back-propagation through time).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config import PPO, PPOConfig
from models.slm_policy import MemoryState, Observations, SLMAgentPolicy, SLMPolicy, observations_from_env, save_model

from .common import RunLogger, validate
from .rewards import compute_gae
from .stages import Stage, make_optimizer
from .vec_env import StepResult, VecTextEnv


@dataclass
class Rollout:
    """``num_envs`` sequences of ``rollout_len`` steps, all tensors (N, T)."""

    obs: Observations
    memory0: MemoryState  # memory before the first step of each sequence
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    last_value: torch.Tensor  # (N,) value after the last step


@torch.no_grad()
def collect_rollout(
    model: SLMPolicy, env: VecTextEnv, current: StepResult, memory: MemoryState, length: int, device
) -> tuple[Rollout, StepResult, MemoryState, dict[str, float]]:
    """Run the current policy (sampling actions) for ``length`` steps.

    Args:
        current: the environments' latest ``StepResult`` (from ``reset`` or the previous rollout).
        memory: the model's memory for those observations.

    Returns:
        The rollout, the new current step and memory (to continue from), and
        statistics of the episodes that finished during it.
    """
    model.eval()
    memory0 = memory
    steps = {k: [] for k in ("image", "direction", "t", "reset", "missions", "action", "log_prob", "value", "reward", "done")}
    finished, successes, env_returns = 0, 0, []
    for _ in range(length):
        block = observations_from_env(current.obs, current.t, current.reset).to(device)
        out = model(block, memory)
        dist = out.distribution()
        action = dist.sample()  # (N, 1)
        memory = out.memory

        steps["image"].append(block.image[:, 0])
        steps["direction"].append(block.direction[:, 0])
        steps["t"].append(block.t[:, 0])
        steps["reset"].append(block.reset[:, 0])
        steps["missions"].append([m[0] for m in block.missions])
        steps["action"].append(action[:, 0])
        steps["log_prob"].append(dist.log_prob(action)[:, 0])
        steps["value"].append(out.value[:, 0])

        current = env.step(action[:, 0].cpu().numpy())
        steps["reward"].append(torch.as_tensor(current.reward, dtype=torch.float32, device=device))
        steps["done"].append(torch.as_tensor(current.done, device=device))
        finished += int(current.done.sum())
        successes += int(current.success.sum())
        env_returns.extend(current.env_reward[current.done].tolist())

    block = observations_from_env(current.obs, current.t, current.reset).to(device)
    last_value = model(block, memory).value[:, 0]  # the memory this writes is discarded

    stack = lambda key: torch.stack(steps[key], dim=1)  # noqa: E731
    obs = Observations(
        image=stack("image"),
        direction=stack("direction"),
        t=stack("t"),
        reset=stack("reset"),
        missions=[list(row) for row in zip(*steps["missions"])],
    )
    rollout = Rollout(
        obs=obs,
        memory0=memory0,
        actions=stack("action"),
        log_probs=stack("log_prob"),
        values=stack("value"),
        rewards=stack("reward"),
        dones=stack("done"),
        last_value=last_value,
    )
    stats = {
        "episodes": finished,
        "train_success": successes / max(finished, 1),
        "train_return": float(np.mean(env_returns)) if env_returns else 0.0,
    }
    model.train()
    return rollout, current, memory, stats


def ppo_update(
    model: SLMPolicy, optimizer: torch.optim.Optimizer, rollout: Rollout, cfg: PPOConfig
) -> dict[str, float]:
    """Several epochs of clipped PPO over the rollout, in minibatches of whole sequences.

    Loss = -clipped surrogate + value_coef * value MSE - entropy_coef * entropy
    + write_penalty * mean write gate.
    """
    advantages, returns = compute_gae(
        rollout.rewards, rollout.values, rollout.dones, rollout.last_value, cfg.gamma, cfg.gae_lambda
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    n = rollout.actions.shape[0]
    history = []
    for _ in range(cfg.epochs):
        order = torch.randperm(n, device=rollout.actions.device)
        for start in range(0, n, cfg.minibatch_envs):
            idx = order[start : start + cfg.minibatch_envs]
            out = model(rollout.obs.select(idx), rollout.memory0.select(idx))
            dist = out.distribution()
            log_probs = dist.log_prob(rollout.actions[idx])
            ratio = torch.exp(log_probs - rollout.log_probs[idx])
            adv = advantages[idx]
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
            value_loss = (out.value - returns[idx]).pow(2).mean()
            entropy = dist.entropy().mean()
            write_rate = out.write_gate.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy + cfg.write_penalty * write_rate

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            history.append(
                {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                    "write_rate": write_rate.item(),
                    "approx_kl": (rollout.log_probs[idx] - log_probs).mean().item(),
                }
            )
    return {k: float(np.mean([h[k] for h in history])) for k in history[0]}


def train_ppo(
    model: SLMPolicy,
    stage: Stage,
    tasks: list[str],
    out_dir: str | Path,
    cfg: PPOConfig = PPO,
    device: str | torch.device = "cpu",
) -> SLMPolicy:
    """Fine-tune ``model`` with recurrent PPO on the train seeds of ``tasks``.

    Every ``cfg.val_every`` updates, greedy success on ``cfg.val_episodes`` val
    seeds per task is measured; the best model is saved as ``<out_dir>/best.pt``
    and the latest as ``last.pt``.

    Returns:
        The model after the last update (load ``best.pt`` for the selected one).
    """
    out_dir = Path(out_dir)
    model.to(device).train()
    optimizer = make_optimizer(model, stage, cfg.lr)
    logger = RunLogger(out_dir)
    env = VecTextEnv(tasks, cfg.num_envs, gamma=cfg.gamma, shaping=cfg.shaping)
    current = env.reset()
    memory = model.initial_memory(env.num_envs)
    num_updates = max(1, cfg.total_steps // (cfg.num_envs * cfg.rollout_len))
    best = -float("inf")

    for update in range(1, num_updates + 1):
        rollout, current, memory, stats = collect_rollout(model, env, current, memory, cfg.rollout_len, device)
        stats.update(ppo_update(model, optimizer, rollout, cfg))
        stats["env_steps"] = update * cfg.num_envs * cfg.rollout_len
        if update % cfg.val_every == 0 or update == num_updates:
            stats.update(validate(SLMAgentPolicy(model, device=device, greedy=True), tasks, cfg.val_episodes))
            model.train()
            save_model(model, out_dir / "last.pt", stage=stage.number, update=update, **stats)
            if stats["val_success"] > best:
                best = stats["val_success"]
                save_model(model, out_dir / "best.pt", stage=stage.number, update=update, **stats)
        logger.log(stage=stage.number, update=update, **stats)
    env.close()
    return model
