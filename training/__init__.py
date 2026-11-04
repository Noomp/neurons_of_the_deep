"""Training: building datasets and training the models in ``models`` by behaviour cloning.

Depends on ``babytext`` (to roll out the expert and the policy), ``models``
(the networks being trained) and ``config``. Nothing in ``babytext`` or
``models`` may import from here, so a trained model can be loaded and
evaluated without the training code.

Public API
----------
Command-line tools (``training.tools``)
    ``collect_demos``   expert demonstrations -> ``data/demos/<task>_<split>.{jsonl,npz}``
    ``dataset_stats``   charts of the demonstrations and eval sets
    ``cache_hidden``    frozen-LLM hidden states of every demo step -> ``data/hidden/``

Demonstration files are written and read with ``babytext.data``
(``save_demos``, ``load_episodes``, ``load_steps``, ``load_symbolic``),
which belongs to the environment because it defines the episode format.

Training (``python -m training.train [--model head|b3]``; every run of the
comparison: ``python -m training.run_plan``)
    ``training.bc``            ``train_bc``, ``bc_loss``, ``evaluate_bc``: the one loop for P and B3
    ``training.bc_head``       ``HeadDemos``, ``head_demos``, ``train_head_bc`` (P, from the cache)
    ``training.bc_symbolic``   ``SymbolicDemos``, ``symbolic_demos``, ``train_symbolic_bc`` (B3)
    ``training.hidden_cache``  ``cache_dir``, ``check_cache``, ``load_pooled``, ``open_sequences``
    ``training.demos``         ``load_demo_episodes``, ``batch_indices``, ``check_demo_config``
    ``training.overrides``     ``parse_overrides``, ``effective_configs`` (``--set``)
    ``training.common``        ``set_seed``, ``RunLogger``, ``validate``, ``git_state``

Config objects are passed explicitly to every model and trainer; the
``config`` defaults bound as default arguments are only a convenience.

Not implemented: DAgger relabelling (``config.TRAIN.dagger_rounds`` is unused).
"""

__all__: list[str] = []
