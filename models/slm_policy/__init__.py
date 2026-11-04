"""The SLM policy: a small language model with a learned spatio-temporal memory.

Per environment step::

    obs["image"], direction, t ─► ViewEncoder ─► F_t (49 cells x d)
    obs["mission"] ─► MissionEncoder (word GRU) ─► q
    [F_t ; memory] ─► SpaceTimeAttention(q) ─► H read-outs ─► Backbone ─► action / value heads
    F_t ─► SlotMemory.write (gate g_t: save or not; address w_t: which slot) ─► next memory

Public API (also listed in models/README.md):

``SLMPolicy``
    The network. ``forward(obs: Observations, memory) -> PolicyOutput`` on a
    (B, T) block; ``initial_memory(batch_size)``.
``SLMAgentPolicy``
    A babytext policy (``obs -> action``) around a model, for ``babytext.evaluate``.
``Observations``, ``observations_from_env``
    The model's input block, and how to build one from env observation dicts.
``PolicyOutput``, ``MemoryState``
    What ``forward`` returns, and the memory it carries between calls.
``parameter_groups``, ``GROUPS``
    Parameters by role, which training stages freeze or train.
``save_model``, ``load_model``, ``load_checkpoint_info``
    Checkpoints (the frozen pretrained SLM weights are not stored).
``Backbone``, ``ReadoutBackbone``, ``SLMBackbone``
    The swappable part between the read-outs and the heads.

Everything else (``encoders``, ``attention``, ``memory.SlotMemory``) is
internal but documented, and can be edited without touching training code.
"""

from .agent import SLMAgentPolicy
from .backbones import Backbone, ReadoutBackbone, SLMBackbone
from .inputs import Observations, observations_from_env
from .memory import MemoryState
from .policy import GROUPS, PolicyOutput, SLMPolicy, load_checkpoint_info, load_model, parameter_groups, save_model

__all__ = [
    "GROUPS",
    "Backbone",
    "MemoryState",
    "Observations",
    "PolicyOutput",
    "ReadoutBackbone",
    "SLMAgentPolicy",
    "SLMBackbone",
    "SLMPolicy",
    "load_checkpoint_info",
    "load_model",
    "observations_from_env",
    "parameter_groups",
    "save_model",
]
