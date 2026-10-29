"""Models: the policies trained in this project.

Modules are imported directly (``from models.symbolic_agent import SymbolicAgent``)
so that ``import models`` does not pull in PyTorch. Each module lists its
public names in ``__all__``; everything else is internal. Requires the
``models`` extra (``pip install -e ".[models]"``, or the conda env).

``models.symbolic_agent``  (done)
    B3, the non-LLM baseline: frame-stacked CNN + FiLM on the symbolic grid.
    ``SymbolicAgent``, ``SymbolicPolicy`` (a ``babytext`` policy),
    ``SymbolicHistory``, ``stack_demo_steps``, ``MissionTokenizer``, ``NO_ACTION``.

``models.slm_policy``  (planned, see models/README.md)
    The main model: spatial encoder, time encoding, gated memory, space-time
    attention, adapter into a small language model (SLM), action and value
    heads. Planned public names: ``SLMPolicyConfig``, ``SLMPolicy``
    (``forward(step_inputs, memory) -> (action_logits, value, new_memory)``),
    ``initial_memory``, ``parameter_groups`` (named groups that training
    freezes / unfreezes per stage), ``SLMAgentPolicy`` (a ``babytext`` policy
    whose ``reset`` clears the memory), ``save_model`` / ``load_model``.

``models.prompted``  (planned)
    Zero-shot baselines on the same SLM reading ``obs["prompt"]``:
    B1 generate-and-map and B2 log-probability scoring.

Contract with the environment
    A model is evaluated by wrapping it as a policy: a callable
    ``policy(obs) -> int`` (index into ``babytext.ACTION_NAMES``), optionally
    with ``reset(env)`` to clear per-episode state, passed to
    ``babytext.run_episode`` / ``babytext.evaluate``. Use only names exported
    by ``babytext`` (plus ``babytext.constants``), and settings from ``config``.

Must not import from ``training``.
"""
