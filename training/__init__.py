"""Training: building datasets and training the models in ``models``.

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

Training the SLM policy (``python -m training.train --stage N``)
    ``training.stages``   ``STAGES`` table and ``make_optimizer`` (freezing per stage)
    ``training.bc``       ``train_bc``, ``bc_loss`` (stages 1-3)
    ``training.ppo``      ``train_ppo``, ``collect_rollout``, ``ppo_update`` (stage 4)
    ``training.rewards``  ``shaped_reward``, ``compute_gae``
    ``training.vec_env``  ``VecTextEnv``, ``goal_potential``: all environment plumbing for RL
    ``training.demos``    ``load_demo_episodes``, ``collate``, ``iterate_batches``

Not implemented yet: DAgger relabelling, a KL penalty towards the BC policy.
"""

__all__: list[str] = []
