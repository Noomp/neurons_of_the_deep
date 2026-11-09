# analysis

The failure analysis (proposal §6.6) and the report's result figures, both from the evaluation's recorded cells: no
model is run, so this section needs neither a GPU nor the models' dependencies. Depends on `babytext`, `evaluation`
and `config`; nothing imports it.

| module | what |
|---|---|
| [`failures.py`](failures.py) | replays every failed test episode and gives it a failure category; action confusion matrices and agreement with the expert by step |
| [`figures.py`](figures.py) | the report's result figures, written to `report/figures/` |

## Run

After `python -m evaluation.evaluate` (and `python -m babytext.tools.reference_baselines` for the random val floor in
the layer figure):

```bash
python -m analysis.failures    # outputs/eval -> outputs/results/failures.{md,csv,json}   (about 4 min, 8 processes)
python -m analysis.figures     # -> report/figures/results_*.png and .svg                 (about 1 min)
```

`failures` replays the failed episodes of `config.ANALYSIS.models` (B2, B3, P-L12-attn-h1, P-L28-attn-h1); B1 is left
out because it acts like B2 (evaluation README, design choices), random because it is the floor. The confusion
matrices and the agreement by step cover every evaluated condition.

## Outputs

- `outputs/results/failures.md`: failure categories per tier and per task, what the wrongly picked-up objects were
  (with chance), non-exclusive flags of failed episodes, and the confusion matrices on `test`.
- `outputs/results/failures.csv`: one row per failed episode: identity (`model`, `tier`, `run`, `task`, `eval_set`,
  `seed`), `category`, `stuck_kind`, the flags, `wrong_pickup_kind` and `wrong_pickup_t`, the chance shares of its
  room, and `first_disagreement` (the first step off the expert's choice).
- `outputs/results/failures.json`: per `model|tier|eval_set`, the confusion counts (6x6: row = the expert's action
  in the policy's state, column = the policy's action) and agreement by step for successful and failed episodes.
- `report/figures/results_*.png` (300 dpi) and `.svg` (live text): tracked in git, because the report uses them.

| figure | shows |
|---|---|
| `results_success` | success on the four test sets per condition, both tiers, 95% intervals |
| `results_gaps` | generalisation gap (shifted set minus `test`) of B3, P (L12) and P (L28) |
| `results_demo_curve` | success on `test` against demonstrations per task, B3 vs P, 95% bands |
| `results_cost` | success on `test` against median latency per decision (log scale) |
| `results_layers` | val success of P by layer and pooling, easy tier (selection data, not test) |
| `results_failures` | share of episodes failing in each category, per condition and test set |
| `results_confusion` | action confusion against the expert on `test`, B3 and P, both tiers |
| `results_drift` | agreement with the expert by step, failed vs successful episodes |
| `results_examples` | four failed episodes, six frames each |

## Public API

| module | public names |
|---|---|
| `analysis.failures` | `main`, `episode_features`, `classify`, `failure_rows`, `category_shares`, `confusion`, `agreement_by_step`, `CATEGORIES`, `STUCK_KINDS` |
| `analysis.figures` | `main`, `COLOURS`, `LABELS` |

## Failure categories

Each failed episode gets the first category that applies:

| category | the agent... |
|---|---|
| `wrong_object` | picked up an object the instruction does not ask for (for putnext and pickup_then_goto, anything but the first object) |
| `first_step_only` | (hard tier) picked up the right object, then failed the second step |
| `reached_no_pickup` | (tasks needing `pick up`) faced the object to pick up, never picked it up |
| `stuck` | spent its last 16 steps in at most 2 cells: `spinning` (turns only), `blocked` (forward moves into an object or wall) or `other` |
| `never_saw_goal` | never had the object it needed first in its 7x7 view |
| `wandered` | none of the above |

## Design choices

- **Categories from the replayed state, not from the text.** The replay reads the environment (positions, carried
  object, what is ahead and in view), so a category means the same for B3 and P, and under `test_nonce` it refers to
  the real objects. Replaying is exact: every replay is checked against the recorded outcome.
- **Ordered, exclusive categories plus flags.** One category per failure (so the shares add up to the failure rate),
  ordered from the earliest decisive error (a wrong pickup cannot be undone in time) to the least specific; the flags
  (`saw_goal`, `faced_goal`, `faced_lookalike`, empty pickups, blocked forwards) are kept for every failure, so a
  different ordering can be read off the CSV.
- **Chance for the wrong object.** A wrong pickup is a look-alike (shares the colour or type of an instruction
  object), the other instruction object, or unrelated. How often each would occur if the wrong object were picked at
  random is computed from each failed episode's own room, so "picks look-alikes" is measured against chance.
- **No "drift" category.** Drifting after an early mistake is shown by agreement with the expert at each step for
  failed vs successful episodes (`results_drift`), not as a category: almost every failure leaves the expert's choice
  within its first 4 steps (73% of P's, 83% of B3's; `first_disagreement` in the CSV), so "drifted" would not
  separate failures from one another.
- **One colour per condition in every figure** (`COLOURS`), in an order where neighbouring bars are neighbouring
  slots of the project's validated categorical palette; failure categories use the three slots validated for every
  pair for the three large categories.

## Limitations

- "Stuck" is a heuristic (at most 2 cells in the last 16 steps); an agent oscillating over 3 cells counts as
  `wandered`. The thresholds are `config.ANALYSIS.stuck_window` and `stuck_cells`.
- `never_saw_goal` is used once in 38,299 failures. The 2,259 failures that never saw the goal (almost all B2) are
  `stuck` first; the room (6x6 inside) is small enough for any agent that moves around to see the goal.
- Agreement with the expert is not success (several shortest paths), and agreement by step for successful episodes is
  a selection effect at later steps (only harder successful episodes last that long).
