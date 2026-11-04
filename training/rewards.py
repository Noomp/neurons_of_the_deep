"""Reward maths, independent of the environment code.

* Immediate reward: environment reward + potential-based shaping
  ``lambda * (gamma * Phi(s') - Phi(s))`` (Ng et al., 1999), which does not
  change the optimal policy. ``Phi`` comes from ``training.vec_env``.
* Delayed reward: the discounted return, estimated with GAE against the value head.
"""

from __future__ import annotations

import torch


def shaped_reward(
    env_reward: float, potential: float, next_potential: float, terminal: bool, gamma: float, weight: float
) -> float:
    """Environment reward plus potential-based shaping.

    The potential of a terminal state is taken as 0, which keeps the shaping
    term from changing which policy is optimal in episodic tasks.
    """
    next_potential = 0.0 if terminal else next_potential
    return env_reward + weight * (gamma * next_potential - potential)


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor, gamma: float, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over a rollout.

    Args:
        rewards: (N, T) reward received after the action at each step.
        values: (N, T) value estimates of the states where the actions were taken.
        dones: (N, T) True if the episode ended after that step.
        last_value: (N,) value of the state after the last step.

    Returns:
        advantages, returns: both (N, T); ``returns = advantages + values``.
    """
    n, t_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(n, device=rewards.device)
    next_value = last_value
    for t in reversed(range(t_len)):
        not_done = 1.0 - dones[:, t].float()
        delta = rewards[:, t] + gamma * next_value * not_done - values[:, t]
        gae = delta + gamma * lam * not_done * gae
        advantages[:, t] = gae
        next_value = values[:, t]
    return advantages, advantages + values
