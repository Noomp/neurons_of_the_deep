"""Backbones: turn the H read-outs (and the mission) into one decision feature.

Two interchangeable implementations of ``Backbone``:

* ``ReadoutBackbone``: a small MLP. Training stage 1 (model M1), and the
  "no SLM" ablation.
* ``SLMBackbone``: a pretrained causal LM. The read-outs become H soft tokens
  placed after the mission text; the last token's hidden state is the
  feature. LoRA adapters sit on the low-level layers from the start (they are
  zero-initialised, so they change nothing until a stage trains them).

The rest of the policy only sees ``Backbone.forward`` and ``feature_dim``,
so swapping the SLM, the prompt or the read-out layer touches only this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import SLMPolicyConfig


class Backbone(nn.Module):
    """Interface: (N, H, d) read-outs + missions -> (N, feature_dim) features."""

    feature_dim: int

    def forward(
        self, readouts: torch.Tensor, mission: torch.Tensor, mission_index: torch.Tensor, mission_texts: list[str]
    ) -> torch.Tensor:
        """Compute decision features for N independent steps.

        Args:
            readouts: (N, H, d) attention read-outs.
            mission: (N, mission_dim) mission vectors from the mission GRU.
            mission_index: (N,) index of each step's mission in ``mission_texts``.
            mission_texts: U distinct mission strings.
        """
        raise NotImplementedError

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named trainable parameter groups (see ``models.slm_policy.parameter_groups``)."""
        raise NotImplementedError

    def pretrained_parameter_names(self) -> set[str]:
        """Names (relative to this module) of frozen pretrained weights, which
        checkpoints do not store because they are reloaded from the hub."""
        return set()


class ReadoutBackbone(Backbone):
    """MLP over the concatenated read-outs and the mission vector."""

    def __init__(self, dim: int, heads: int, mission_dim: int, hidden: int):
        super().__init__()
        self.feature_dim = hidden
        self.mlp = nn.Sequential(nn.Linear(heads * dim + mission_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, readouts, mission, mission_index, mission_texts):
        return self.mlp(torch.cat([readouts.flatten(1), mission], dim=-1))

    def parameter_groups(self):
        return {"readout": list(self.mlp.parameters())}


class SLMBackbone(Backbone):
    """A frozen causal LM reading ``<mission text> <H soft tokens> <"Action:">``.

    Args:
        lm: a Hugging Face *base* model (``AutoModel``, no LM head) with a
            ``layers`` ModuleList, e.g. ``Qwen3Model``. Its weights are frozen here.
        tokenizer: the matching tokenizer.
        dim: read-out width d.
        cfg: ``low_level_layers``, ``lora_rank``, ``lora_targets``,
            ``readout_layer`` and ``gradient_checkpointing`` are used.

    Use ``SLMBackbone.from_pretrained(cfg, dim)`` to load ``cfg.slm_name``.
    """

    PREFIX = "Goal: {mission}\nWhat you see and remember:"
    SUFFIX = "\nNext action:"

    def __init__(self, lm: nn.Module, tokenizer, dim: int, cfg: SLMPolicyConfig):
        super().__init__()
        self.tokenizer = tokenizer
        self.heads = cfg.read_heads
        self._truncate(lm, cfg.readout_layer)
        for p in lm.parameters():
            p.requires_grad_(False)
        if cfg.lora_rank > 0 and cfg.low_level_layers > 0:
            self._add_lora(lm, cfg)
        self._pretrained = {f"lm.{n}" for n, _ in lm.named_parameters() if "lora_" not in n}
        if cfg.gradient_checkpointing:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.lm = lm
        hidden = lm.config.hidden_size
        self.feature_dim = hidden
        self.adapter = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden))
        # Soft tokens start at the scale of real token embeddings.
        with torch.no_grad():
            self.adapter[1].weight.fill_(float(lm.get_input_embeddings().weight.float().std()))
        self._prompt_cache: dict[str, tuple[list[int], list[int]]] = {}

    @classmethod
    def from_pretrained(cls, cfg: SLMPolicyConfig, dim: int) -> SLMBackbone:
        from transformers import AutoModel, AutoTokenizer

        lm = AutoModel.from_pretrained(cfg.slm_name, dtype=getattr(torch, cfg.slm_dtype))
        return cls(lm, AutoTokenizer.from_pretrained(cfg.slm_name), dim, cfg)

    @staticmethod
    def _truncate(lm: nn.Module, readout_layer: int | None) -> None:
        """Drop the layers above ``readout_layer`` (and the final norm, which belongs to the last layer)."""
        if readout_layer is None or readout_layer >= lm.config.num_hidden_layers:
            return
        lm.layers = lm.layers[:readout_layer]
        lm.config.num_hidden_layers = readout_layer
        if hasattr(lm.config, "layer_types"):
            lm.config.layer_types = lm.config.layer_types[:readout_layer]
        lm.norm = nn.Identity()

    @staticmethod
    def _add_lora(lm: nn.Module, cfg: SLMPolicyConfig) -> None:
        from peft import LoraConfig, inject_adapter_in_model

        lora = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=2 * cfg.lora_rank,
            target_modules=list(cfg.lora_targets),
            layers_to_transform=list(range(min(cfg.low_level_layers, len(lm.layers)))),
            layers_pattern="layers",
        )
        inject_adapter_in_model(lora, lm)

    def _prompt_ids(self, mission: str) -> tuple[list[int], list[int]]:
        if mission not in self._prompt_cache:
            encode = lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"]  # noqa: E731
            self._prompt_cache[mission] = (encode(self.PREFIX.format(mission=mission)), encode(self.SUFFIX))
        return self._prompt_cache[mission]

    def _prompt_table(self, texts: list[str], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-padded token ids (U, L), attention mask (U, L) and soft-token mask (U, L)."""
        rows = [self._prompt_ids(text) for text in texts]
        length = max(len(pre) + self.heads + len(suf) for pre, suf in rows)
        ids = torch.zeros(len(rows), length, dtype=torch.long)
        attn = torch.zeros(len(rows), length, dtype=torch.long)
        soft = torch.zeros(len(rows), length, dtype=torch.bool)
        for i, (pre, suf) in enumerate(rows):
            n = len(pre) + self.heads + len(suf)
            start = length - n
            ids[i, start:] = torch.tensor(pre + [0] * self.heads + suf)
            attn[i, start:] = 1
            soft[i, start + len(pre) : start + len(pre) + self.heads] = True
        return ids.to(device), attn.to(device), soft.to(device)

    def forward(self, readouts, mission, mission_index, mission_texts):
        ids, attn, soft = self._prompt_table(mission_texts, readouts.device)
        ids, attn, soft = ids[mission_index], attn[mission_index], soft[mission_index]
        embeds = self.lm.get_input_embeddings()(ids)
        soft_tokens = self.adapter(readouts).to(embeds.dtype)  # (N, H, hidden)
        embeds = embeds.masked_scatter(soft[..., None], soft_tokens)
        positions = (attn.cumsum(dim=1) - 1).clamp_min(0)
        out = self.lm(inputs_embeds=embeds, attention_mask=attn, position_ids=positions, use_cache=False)
        return out.last_hidden_state[:, -1].float()

    def parameter_groups(self):
        lora = [p for n, p in self.lm.named_parameters() if "lora_" in n]
        return {"adapter": list(self.adapter.parameters()), "lora": lora}

    def pretrained_parameter_names(self):
        return set(self._pretrained)


def build_backbone(cfg: SLMPolicyConfig, dim: int, mission_dim: int) -> Backbone:
    """The backbone named by ``cfg.backbone`` ("readout" or "slm")."""
    if cfg.backbone == "readout":
        return ReadoutBackbone(dim, cfg.read_heads, mission_dim, cfg.readout_hidden)
    if cfg.backbone == "slm":
        return SLMBackbone.from_pretrained(cfg, dim)
    raise ValueError(f"unknown backbone {cfg.backbone!r}")
