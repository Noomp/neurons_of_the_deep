"""Training: building datasets and (planned) training the models in ``models``.

Depends on ``babytext`` (to roll out the expert and the policy), ``models``
(the networks being trained) and ``config``. Nothing in ``babytext`` or
``models`` may import from here, so a trained model can be loaded and
evaluated without the training code.

Public API
----------
Command-line tools (``training.tools``)
    ``collect_demos``   expert demonstrations -> ``data/demos/<task>_<split>.{jsonl,npz}``
    ``dataset_stats``   charts of the demonstrations and eval sets

Demonstration files are written and read with ``babytext.data``
(``save_demos``, ``load_episodes``, ``load_steps``, ``load_symbolic``),
which belongs to the environment because it defines the episode format.

Planned, not yet implemented (see training/README.md)
    immediate (shaped) and delayed rewards; stage 1-3 behaviour cloning on
    whole demonstration episodes; stage 4 recurrent PPO; optional DAgger
    relabelling with ``babytext.ExpertPolicy(obs, action_taken=...)``;
    demonstration-count learning curves (``config.TRAIN.demo_counts``).
"""

__all__: list[str] = []
