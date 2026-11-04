"""Models: the policies compared in this project.

Modules are imported directly (``from models.action_head import ActionHead``)
so that ``import models`` does not pull in PyTorch. Each module lists its
public names in ``__all__``; everything else is internal. Requires the
``models`` extra (``pip install -e ".[models]"``, or the conda env).

``models.action_head``  (done)
    P, the proposed method: a small head on one layer's hidden states of the
    frozen LLM reading the GLAM prompt. ``ActionHead``, ``HeadPolicy`` (a
    ``babytext`` policy), ``pool``, ``save_model`` / ``load_model`` /
    ``load_policy`` / ``load_checkpoint_info``. Settings: ``config.HEAD``.

``models.prompted``  (done)
    B1 and B2, the language-grounded zero-shot baselines on the same LLM and
    prompt: ``GenerationPolicy`` (generate and map), ``ScoringPolicy``
    (log-probability scoring), ``SentenceEmbedder``. Settings: ``config.PROMPTED``.

``models.symbolic_agent``  (done)
    B3, the non-language baseline: frame-stacked CNN + FiLM on the symbolic grid.
    ``SymbolicAgent``, ``SymbolicPolicy`` (a ``babytext`` policy),
    ``SymbolicHistory``, ``stack_demo_steps``, ``MissionTokenizer``, ``NO_ACTION``,
    ``save_model`` / ``load_model`` / ``load_checkpoint_info``.

``models.frozen_lm``  (done)
    The frozen LLM shared by P, B1 and B2: ``FrozenLM`` (base model, layer
    truncation, hidden states of prompts), ``PromptStates``, ``load_causal_lm``.
    Settings: ``config.LLM``.

Contract with the environment
    A model is evaluated by wrapping it as a policy: a callable
    ``policy(obs) -> int`` (index into ``babytext.ACTION_NAMES``), optionally
    with ``reset(env)`` to clear per-episode state, passed to
    ``babytext.run_episode`` / ``babytext.evaluate``. Use only names exported
    by ``babytext`` (plus ``babytext.constants``), and settings from ``config``.

Must not import from ``training``.
"""
