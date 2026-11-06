"""Final evaluation of every condition on the test sets (proposal 6.5).

``evaluation.evaluate`` runs the conditions and writes one result cell per
(model, tier, run, task, eval set); ``evaluation.summarise`` turns the cells
into the report's tables; ``evaluation.record`` holds the recorded rollouts
both use. Depends on ``babytext``, ``models`` and ``config`` (and on
``training.common`` for the git state); nothing imports it.
"""
