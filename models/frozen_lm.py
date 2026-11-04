"""The frozen LLM shared by B1, B2 and the action head: loading, and hidden states of prompts.

Layer numbering: layer ``l`` (1 .. N) is the output of decoder layer ``l``.
The last layer, N, includes the model's final norm (it is the hidden state
the LM head reads); the layers below do not. ``FrozenLM.load(cfg, num_layers=l)``
keeps only the first ``l`` layers (and drops the final norm unless ``l = N``),
so its last layer is the same vector and the layers above never run. This is
what makes a head trained on cached hidden states of the full model act
identically on the truncated one.

Prompts in a batch are left-padded with explicit position ids, so each
prompt's hidden states do not depend on what it is batched with.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from config import LLM, LLMConfig

__all__ = ["FrozenLM", "PromptStates", "load_causal_lm"]


@dataclass
class PromptStates:
    """Hidden states of B prompts at some layers.

    Attributes:
        hidden: layer -> (B, L, D) token vectors, left-padded, in the model's dtype.
        mask: (B, L) bool, False on padding.
    """

    hidden: dict[int, torch.Tensor]
    mask: torch.Tensor

    def last(self, layer: int) -> torch.Tensor:
        """(B, D): the last token's vector (left padding puts it in the last column)."""
        return self.hidden[layer][:, -1]

    def mean(self, layer: int) -> torch.Tensor:
        """(B, D): the mean over each prompt's tokens (padding excluded), computed in float32."""
        m = self.mask[..., None].float()
        return ((self.hidden[layer].float() * m).sum(1) / m.sum(1)).to(self.hidden[layer].dtype)

    def sequences(self, layer: int) -> list[torch.Tensor]:
        """B tensors of shape (L_i, D): each prompt's token vectors without padding."""
        return [h[m] for h, m in zip(self.hidden[layer], self.mask)]


class FrozenLM:
    """A frozen Hugging Face base model (no LM head) and its tokenizer.

    Args:
        model: e.g. ``Qwen3Model``, with a ``layers`` ModuleList; frozen and put in eval mode here.
        tokenizer: the matching tokenizer.
        full_layers: the checkpoint's number of layers before any truncation.
        device: where the model runs.
    """

    def __init__(self, model: nn.Module, tokenizer, full_layers: int, device: str | torch.device = "cpu"):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.full_layers = full_layers
        self.device = torch.device(device)

    @classmethod
    def load(cls, cfg: LLMConfig = LLM, num_layers: int | None = None, device: str | torch.device = "cpu") -> FrozenLM:
        """Load ``cfg.name`` as a base model, keeping only its first ``num_layers`` layers (None = all)."""
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(cfg.name, dtype=getattr(torch, cfg.dtype))
        full = model.config.num_hidden_layers
        if num_layers is not None and not 1 <= num_layers <= full:
            raise ValueError(f"num_layers must be in 1..{full}, got {num_layers}")
        if num_layers is not None and num_layers < full:
            model.layers = model.layers[:num_layers]
            model.config.num_hidden_layers = num_layers
            if hasattr(model.config, "layer_types"):
                model.config.layer_types = model.config.layer_types[:num_layers]
            model.norm = nn.Identity()  # the final norm belongs to the last layer
        return cls(model, AutoTokenizer.from_pretrained(cfg.name), full, device)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def tokenize(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-padded ``(ids, mask, positions)``, each (B, L), on the model's device."""
        rows = [self.tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
        length = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), length, dtype=torch.long)
        mask = torch.zeros(len(rows), length, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, length - len(r):] = torch.tensor(r)
            mask[i, length - len(r):] = 1
        positions = (mask.cumsum(1) - 1).clamp_min(0)
        return ids.to(self.device), mask.to(self.device), positions.to(self.device)

    @torch.no_grad()
    def run(self, prompts: list[str], layers: list[int]) -> PromptStates:
        """Hidden states of ``prompts`` at ``layers`` (each in 1 .. ``num_layers``)."""
        bad = [l for l in layers if not 1 <= l <= self.num_layers]
        if bad:
            raise ValueError(f"layers {bad} outside 1..{self.num_layers}")
        captured: dict[int, torch.Tensor] = {}
        hooks = [
            self.model.layers[l - 1].register_forward_hook(
                lambda _m, _i, out, l=l: captured.__setitem__(l, out[0] if isinstance(out, tuple) else out)
            )
            for l in layers
            if l < self.num_layers
        ]
        try:
            ids, mask, positions = self.tokenize(prompts)
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        captured = {l: captured[l] if l < self.num_layers else out.last_hidden_state for l in layers}
        return PromptStates(hidden=captured, mask=mask.bool())


def load_causal_lm(cfg: LLMConfig = LLM, device: str | torch.device = "cpu"):
    """``(model, tokenizer)`` for generation and scoring (B1, B2): the full LM with its LM head, frozen."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(cfg.name, dtype=getattr(torch, cfg.dtype)).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, AutoTokenizer.from_pretrained(cfg.name)
