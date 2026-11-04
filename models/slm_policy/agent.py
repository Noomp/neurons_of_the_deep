"""``SLMAgentPolicy``: an ``SLMPolicy`` as a babytext policy (``obs -> action``)."""

from __future__ import annotations

import torch

from .inputs import observations_from_env
from .policy import PolicyOutput, SLMPolicy


class SLMAgentPolicy:
    """Wraps a trained ``SLMPolicy`` for ``babytext.run_episode`` / ``evaluate``.

    Keeps the memory and step counter of the current episode; ``reset`` (which
    ``run_episode`` calls after every ``env.reset``) clears them.

    Args:
        model: the policy network.
        device: where to run it.
        greedy: take the most likely action (evaluation) instead of sampling.
        keep_attention: store each step's space-time weights in ``self.trace``
            (for figures and failure analysis).
    """

    def __init__(self, model: SLMPolicy, device: str | torch.device = "cpu", greedy: bool = True, keep_attention: bool = False):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.greedy = greedy
        self.keep_attention = keep_attention
        self.trace: list[PolicyOutput] = []
        self.reset()

    def reset(self, env=None) -> None:
        self.memory = self.model.initial_memory(1, self.device)
        self.t = 0
        self.trace = []

    @torch.no_grad()
    def __call__(self, obs: dict) -> int:
        block = observations_from_env([obs], [self.t], [self.t == 0]).to(self.device)
        out = self.model(block, self.memory, return_attention=self.keep_attention)
        self.memory = out.memory
        self.t += 1
        if self.keep_attention:
            self.trace.append(out)
        logits = out.logits[0, 0]
        if self.greedy:
            return int(logits.argmax())
        return int(torch.distributions.Categorical(logits=logits).sample())
