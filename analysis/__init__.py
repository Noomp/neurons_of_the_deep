"""Analysis of the final evaluation for the report.

``analysis.failures`` replays and categorises every failed test episode and
builds the action confusion matrices; ``analysis.figures`` draws the report's
result figures. Both read the cells written by ``evaluation`` (and
``analysis.figures`` the training runs' ``run.json``); they run no model.
Depends on ``babytext``, ``evaluation`` and ``config``; nothing imports it.
"""
