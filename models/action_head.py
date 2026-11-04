"""P: a lightweight action head on the hidden states of a frozen LLM (the proposed method).

The frozen LLM reads the GLAM prompt (``obs["prompt"]``, the same text B1 and
B2 read). The head takes the hidden states of one layer, pools the ~300
token vectors into one, and maps it with a small MLP to the 6 action logits,
in a single forward pass of the (truncated) LLM per decision:

    obs["prompt"] ─► frozen LLM, layers 1 .. l ─► token vectors (L, D)
                  ─► pooling: last token | mean | learned attention ─► (D,)
                  ─► LayerNorm ─► MLP (1-2 hidden layers) ─► 6 logits

Because the LLM is frozen and the prompts are fixed text, the hidden states
of the demonstrations are computed once and cached
(``training.tools.cache_hidden``); training a head then needs no LLM pass.
Only acting in the environment runs the LLM, truncated to layer ``l``.

Settings: ``config.HEAD``. Checkpoints (``save_model`` / ``load_model``) store
the head and the LLM it reads, not the LLM's weights.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from babytext.constants import NUM_ACTIONS
from config import HEAD, LLM, HeadConfig, LLMConfig

from .frozen_lm import FrozenLM, PromptStates

__all__ = ["POOLINGS", "ActionHead", "HeadPolicy", "load_checkpoint_info", "load_model", "load_policy", "pool", "save_model"]

POOLINGS = ("last", "mean", "attention")


class ActionHead(nn.Module):
    """Pooling + MLP from one layer's hidden states to 6 action logits.

    Args:
        cfg: pooling, MLP sizes and the layer it reads (``config.HEAD``).
        input_dim: the LLM's hidden size D.
    """

    def __init__(self, cfg: HeadConfig = HEAD, input_dim: int = 1024):
        super().__init__()
        if cfg.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {cfg.pooling!r}")
        self.cfg, self.input_dim = cfg, input_dim
        self.norm = nn.LayerNorm(input_dim)
        if cfg.pooling == "attention":
            self.key = nn.Linear(input_dim, cfg.attention_dim)
            # A zero query gives uniform weights: the head starts as mean pooling.
            self.query = nn.Parameter(torch.zeros(cfg.attention_dim))
        layers, width = [], input_dim
        for _ in range(cfg.hidden_layers):
            layers += [nn.Linear(width, cfg.hidden_dim), nn.ReLU()]
            width = cfg.hidden_dim
        self.mlp = nn.Sequential(*layers, nn.Linear(width, NUM_ACTIONS))

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Action logits (B, 6).

        Args:
            features: (B, D) pooled vectors for "last" / "mean" pooling, or
                (B, L, D) token vectors for "attention" pooling.
            mask: (B, L) bool, False on padding; attention pooling only.
        """
        x = self.norm(features.float())
        if self.cfg.pooling == "attention":
            scores = self.key(x) @ self.query / math.sqrt(self.cfg.attention_dim)  # (B, L)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            x = (torch.softmax(scores, dim=-1)[..., None] * x).sum(1)
        return self.mlp(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pool(states: PromptStates, layer: int, pooling: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The head's ``(features, mask)`` inputs from live hidden states, exactly as they are cached."""
    if pooling == "last":
        return states.last(layer), None
    if pooling == "mean":
        return states.mean(layer), None
    return states.hidden[layer], states.mask


class HeadPolicy:
    """A trained head as a babytext policy (``obs -> action``), greedy.

    The GLAM prompt in ``obs["prompt"]`` already carries the last observations
    and actions, so the policy keeps no state of its own.

    Args:
        head: the trained head.
        lm: the frozen LLM, with at least ``head.cfg.layer`` layers (load it
            truncated to that layer to skip the layers above).
        device: where the head runs (the LM keeps its own device).
    """

    def __init__(self, head: ActionHead, lm: FrozenLM, device: str | torch.device = "cpu"):
        self.head = head.to(device).eval()
        self.lm = lm
        self.device = torch.device(device)

    def reset(self, env=None) -> None:
        pass

    @torch.no_grad()
    def __call__(self, obs: dict) -> int:
        layer = self.head.cfg.layer
        features, mask = pool(self.lm.run([obs["prompt"]], [layer]), layer, self.head.cfg.pooling)
        logits = self.head(features.to(self.device), None if mask is None else mask.to(self.device))
        return int(logits[0].argmax())


# --- checkpoints -------------------------------------------------------------------


def save_model(head: ActionHead, path: str | Path, llm: LLMConfig = LLM, **extra) -> None:
    """Save the head's config, input size and weights, the LLM config it reads, and ``extra``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(head.cfg),
            "input_dim": head.input_dim,
            "llm": asdict(llm),
            "state_dict": head.state_dict(),
            "extra": extra,
        },
        path,
    )


def load_checkpoint_info(path: str | Path) -> dict:
    """The ``extra`` dict saved with a checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=False)["extra"]


def load_model(path: str | Path) -> tuple[ActionHead, LLMConfig]:
    """The head (on the CPU, eval mode) and the config of the LLM it was trained on."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in ckpt["config"].items()}
    head = ActionHead(HeadConfig(**cfg), ckpt["input_dim"])
    head.load_state_dict(ckpt["state_dict"])
    return head.eval(), LLMConfig(**ckpt["llm"])


def load_policy(path: str | Path, device: str | torch.device = "cpu") -> HeadPolicy:
    """A ready ``HeadPolicy``: the head, plus the LLM it was trained on truncated to its layer."""
    head, llm = load_model(path)
    return HeadPolicy(head, FrozenLM.load(llm, num_layers=head.cfg.layer, device=device), device)
