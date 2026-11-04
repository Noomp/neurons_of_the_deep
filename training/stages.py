"""The four training stages as data: which backbone, which algorithm, and the
learning-rate scale of every parameter group (0 or missing = frozen).

To change what a stage trains, edit the table below; ``make_optimizer`` does
the freezing. Group names are ``models.slm_policy.GROUPS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from models.slm_policy import SLMPolicy, parameter_groups

_CORE = ("encoder", "mission", "memory", "attention")


@dataclass(frozen=True)
class Stage:
    """One training stage.

    Attributes:
        number: 1-4.
        description: one line for logs.
        backbone: "readout" (no SLM) or "slm".
        algorithm: "bc" (behaviour cloning on demos) or "ppo" (reinforcement learning).
        lr_scale: learning-rate multiplier per parameter group; groups not
            listed (or with 0) are frozen.
    """

    number: int
    description: str
    backbone: str
    algorithm: str
    lr_scale: dict[str, float] = field(default_factory=dict)


STAGES: dict[int, Stage] = {
    1: Stage(
        1,
        "warm start without the SLM",
        backbone="readout",
        algorithm="bc",
        lr_scale={**{g: 1.0 for g in _CORE}, "readout": 1.0, "action_head": 1.0},
    ),
    2: Stage(
        2,
        "connect the frozen SLM (adapter + new action head)",
        backbone="slm",
        algorithm="bc",
        lr_scale={**{g: 0.1 for g in _CORE}, "adapter": 1.0, "action_head": 1.0},
    ),
    3: Stage(
        3,
        "LoRA on the SLM's low-level layers",
        backbone="slm",
        algorithm="bc",
        lr_scale={**{g: 0.1 for g in _CORE}, "adapter": 1.0, "action_head": 1.0, "lora": 1.0},
    ),
    4: Stage(
        4,
        "recurrent PPO with immediate + delayed rewards",
        backbone="slm",
        algorithm="ppo",
        lr_scale={**{g: 0.1 for g in _CORE}, "adapter": 1.0, "action_head": 1.0, "lora": 1.0, "value_head": 1.0},
    ),
}


def make_optimizer(model: SLMPolicy, stage: Stage, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Freeze everything the stage does not train and build an AdamW optimiser.

    Every parameter outside the stage's groups (including the pretrained SLM)
    gets ``requires_grad = False``; each trained group gets ``lr * scale``.
    Raises if the stage names a group the model does not have (e.g. "lora" on
    the readout backbone), which usually means the wrong checkpoint was loaded.
    """
    groups = parameter_groups(model)
    missing = [g for g, s in stage.lr_scale.items() if s > 0 and g not in groups]
    if missing:
        raise ValueError(f"stage {stage.number} trains {missing}, which this model does not have")

    for p in model.parameters():
        p.requires_grad_(False)
    param_groups = []
    for name, params in groups.items():
        scale = stage.lr_scale.get(name, 0.0)
        if scale <= 0:
            continue
        for p in params:
            p.requires_grad_(True)
        param_groups.append({"params": params, "lr": lr * scale, "name": name})
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
