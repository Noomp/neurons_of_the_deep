"""The SLM policy: encoder -> memory -> space-time attention -> backbone -> heads.

One forward pass handles a (B, T) block. The only recurrence is the memory
write, which is cheap; the SLM never feeds back into the memory, so it runs
once on all B * T steps after the recurrent loop. Acting in the environment
is the same call with T = 1, so rollout and training compute identical
log-probabilities (which PPO relies on).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn

from babytext.constants import NUM_ACTIONS
from config import ENV, SLM, SLMPolicyConfig

from .attention import SpaceTimeAttention
from .backbones import Backbone, build_backbone
from .encoders import MissionEncoder, TimeEncoding, ViewEncoder
from .inputs import MissionTable, Observations
from .memory import MemoryState, SlotMemory


@dataclass
class PolicyOutput:
    """Result of ``SLMPolicy.forward`` on a (B, T) block.

    Attributes:
        logits: (B, T, 6) action logits.
        value: (B, T) state-value estimates (meaningful once the value head is trained).
        memory: memory after the last step; pass it to the next call.
        write_gate: (B, T) the memory write gate g_t of every step.
        attention: (B, T, H, K + 1, P) space-time weights, only if requested.
    """

    logits: torch.Tensor
    value: torch.Tensor
    memory: MemoryState
    write_gate: torch.Tensor
    attention: torch.Tensor | None = None

    def distribution(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits)


class SLMPolicy(nn.Module):
    """The main model (see models/README.md and the report).

    Args:
        cfg: sizes and backbone choice, default ``config.SLM``.
        backbone: a ready-made backbone (e.g. a tiny LM in tests); built from
            ``cfg`` when omitted.

    Typical use::

        model = SLMPolicy()
        memory = model.initial_memory(batch_size=B)
        out = model(obs_block, memory)          # obs_block: Observations (B, T)
        loss = F.cross_entropy(out.logits.flatten(0, 1), actions.flatten())
    """

    def __init__(self, cfg: SLMPolicyConfig = SLM, backbone: Backbone | None = None):
        super().__init__()
        self.cfg = cfg
        d, v = cfg.feature_dim, ENV.agent_view_size
        self.num_cells = v * v
        self.time = TimeEncoding(d)
        self.encoder = ViewEncoder(d, cfg.num_conv_layers, v, self.time, cfg.use_direction)
        self.mission = MissionEncoder(cfg.word_dim, cfg.mission_dim)
        self.memory = SlotMemory(d, cfg.mission_dim, cfg.memory_slots, cfg.write_gate_bias, cfg.write_mode)
        self.attention = SpaceTimeAttention(d, cfg.mission_dim, cfg.read_heads, v, self.time)
        self.backbone = backbone if backbone is not None else build_backbone(cfg, d, cfg.mission_dim)
        f = self.backbone.feature_dim
        self.head_norm = nn.LayerNorm(f)
        self.action_head = nn.Linear(f, NUM_ACTIONS)
        self.value_head = nn.Linear(f, 1)

    def initial_memory(self, batch_size: int, device: str | torch.device | None = None) -> MemoryState:
        """An empty memory for ``batch_size`` sequences."""
        device = device if device is not None else self.action_head.weight.device
        return MemoryState.empty(batch_size, self.cfg.memory_slots, self.num_cells, self.cfg.feature_dim, device)

    def forward(self, obs: Observations, memory: MemoryState, return_attention: bool = False) -> PolicyOutput:
        """Run the policy over a (B, T) block of observations.

        At each step t the memory is first cleared where ``obs.reset`` is set,
        then read (together with the current view) to choose the action, then
        the current view is offered to the memory.

        Args:
            obs: inputs on the model's device.
            memory: memory before the first step of the block.
            return_attention: also return the space-time weights (for figures).
        """
        b, t_len = obs.shape
        table = MissionTable.from_missions(obs.missions, device=obs.image.device)
        mission = self.mission(table.texts)[table.index]  # (B, T, mission_dim)

        views = self.encoder(obs.image.flatten(0, 1), obs.direction.flatten(), obs.t.flatten())
        views = views.reshape(b, t_len, self.num_cells, -1)
        t_enc = self.time(obs.t)  # (B, T, d)

        readouts, gates, weights = [], [], []
        for step in range(t_len):
            memory = memory.reset(obs.reset[:, step])
            read, attn = self.attention(views[:, step], memory, mission[:, step])
            memory, write = self.memory.write(memory, views[:, step], mission[:, step], t_enc[:, step])
            readouts.append(read)
            gates.append(write.gate)
            if return_attention:
                weights.append(attn)

        readouts = torch.stack(readouts, dim=1)  # (B, T, H, d)
        features = self.backbone(
            readouts.flatten(0, 1), mission.flatten(0, 1), table.index.flatten(), table.texts
        )
        features = self.head_norm(features)
        return PolicyOutput(
            logits=self.action_head(features).reshape(b, t_len, -1),
            value=self.value_head(features).reshape(b, t_len),
            memory=memory,
            write_gate=torch.stack(gates, dim=1),
            attention=torch.stack(weights, dim=1) if return_attention else None,
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


# --- parameter groups -------------------------------------------------------------

GROUPS = ("encoder", "mission", "memory", "attention", "readout", "adapter", "lora", "action_head", "value_head")


def parameter_groups(model: SLMPolicy) -> dict[str, list[nn.Parameter]]:
    """The model's parameters by role, for per-stage freezing and learning rates.

    Keys are a subset of ``GROUPS``: "readout" exists only with the readout
    backbone, "adapter" and "lora" only with the SLM backbone. Frozen
    pretrained SLM weights belong to no group.
    """
    groups = {
        "encoder": [*model.encoder.parameters()],  # includes the shared time encoding
        "mission": [*model.mission.parameters()],
        "memory": [*model.memory.parameters()],
        "attention": [p for n, p in model.attention.named_parameters() if not n.startswith("time.")],
        **model.backbone.parameter_groups(),
        "action_head": [*model.head_norm.parameters(), *model.action_head.parameters()],
        "value_head": [*model.value_head.parameters()],
    }
    return {k: v for k, v in groups.items() if v}


# --- checkpoints ------------------------------------------------------------------


def save_model(model: SLMPolicy, path: str | Path, **extra) -> None:
    """Save the config and every weight except the frozen pretrained SLM.

    A random SLM (``slm_pretrained = False``) is not stored either: its
    fingerprint is, so ``load_model`` can check the regenerated weights.
    ``extra`` (e.g. stage, step, validation score) is stored alongside and
    returned by ``load_checkpoint_info``.
    """
    skip = {f"backbone.{n}" for n in model.backbone.pretrained_parameter_names()}
    state = {k: v for k, v in model.state_dict().items() if k not in skip}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.cfg),
            "state_dict": state,
            "extra": extra,
            "lm_fingerprint": model.backbone.regenerated_fingerprint(),
        },
        path,
    )


def load_checkpoint_info(path: str | Path) -> dict:
    """The ``extra`` dict saved with a checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=False)["extra"]


def load_model(path: str | Path, backbone_module: Backbone | None = None, **overrides) -> SLMPolicy:
    """Build a model from a checkpoint, optionally changing its config.

    Pass config overrides to move to the next training stage, e.g.
    ``load_model("m1.pt", backbone="slm")``. When the backbone type changes,
    the backbone and the heads start fresh and everything else (encoder,
    mission, memory, attention) is loaded. The model's config is the saved
    one plus ``overrides``; ``config.SLM`` is not read.

    Raises:
        RuntimeError: weights are missing or unexpected, or a random SLM
            regenerated with different weights than it was trained with
            (e.g. after a torch / transformers upgrade).

    Args:
        path: file written by ``save_model``.
        backbone_module: optional ready-made backbone (tests); otherwise built from the config.
        overrides: ``SLMPolicyConfig`` fields to change.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    names = {f.name for f in fields(SLMPolicyConfig)}
    merged = {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k in names}
    merged.setdefault("use_direction", True)  # checkpoints from before the field always used the heading
    unknown = set(overrides) - names
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    merged.update(overrides)
    cfg = SLMPolicyConfig(**merged)
    model = SLMPolicy(cfg, backbone=backbone_module)

    state = ckpt["state_dict"]
    if cfg.backbone != saved["backbone"]:
        fresh = ("backbone.", "head_norm.", "action_head.", "value_head.")
        state = {k: v for k, v in state.items() if not k.startswith(fresh)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has weights the model does not: {unexpected}")
    allowed = {f"backbone.{n}" for n in model.backbone.pretrained_parameter_names()}
    if cfg.backbone == saved["backbone"]:
        bad = [k for k in missing if k not in allowed]
        if bad:
            raise RuntimeError(f"checkpoint is missing weights: {bad}")
        expected = ckpt.get("lm_fingerprint")
        if expected is not None and model.backbone.regenerated_fingerprint() != expected:
            raise RuntimeError(
                "the random SLM did not regenerate to the weights this checkpoint was trained with "
                "(different slm_init_seed or slm_name, or a different torch / transformers version)"
            )
    return model
