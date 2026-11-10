# evaluation

The final evaluation (proposal §6.5): every condition on the test sets, once, with everything kept per decision so
the analysis never needs a re-run. Depends on `babytext`, `models` and `config` (and `training.common` for the git
state); only `analysis` imports it (the failure analysis and the report's figures read its cells).

| module | what |
|---|---|
| [`evaluate.py`](evaluate.py) | the command: plans and checks every run, then evaluates each cell; resumable |
| [`record.py`](record.py) | recorded rollouts: one cell (a policy on one task and eval set) with every decision kept |
| [`summarise.py`](summarise.py) | the report's tables: success with 95% intervals, generalisation gaps, cost, B1's invalid outputs, expert agreement, demo-count curves |
| [`failures.py`](failures.py) | failure analysis (§6.6): each failed episode's category, action confusion matrices, example trajectories |

## Run

```bash
python -m evaluation.evaluate --models random B3 P-L12-attn-h1 B2 B1 --tiers easy --dry-run   # check the plan
python -m evaluation.evaluate --models random B3 P-L12-attn-h1 B2 B1 --tiers easy             # then run it
python -m evaluation.evaluate --models random B3 P-L12-attn-h1 P-L28-attn-h1 B2 --tiers hard   # L28: H2 control
python -m evaluation.evaluate --models B1 --tiers hard --sets test                             # B1: test only (design choices)
python -m evaluation.evaluate --models B3-n100 B3-n250 B3-n500 B3-n1000 P-L12-attn-h1-n100 P-L12-attn-h1-n250 \
    P-L12-attn-h1-n500 P-L12-attn-h1-n1000 --tiers easy --sets test                             # demo-count curves
python -m evaluation.evaluate --models P-L12-last-h1 P-L12-mean-h1 --tiers easy --sets test test_dist16   # H2: pooling under clutter
python -m evaluation.summarise                                                                 # -> outputs/results/evaluation.md
python -m evaluation.failures --models B3 P-L12-attn-h1 B2 B1                                   # -> outputs/results/failures/
```

`--models` takes `random`, `B1`, `B2` (no training; one run each) and training run IDs (`B3`, `P-L12-attn-h1`,
`B3-n100`, ...), evaluated for every seed in `config.TRAIN.seeds` from `outputs/runs/<ID>/<tier>/seed<k>/best.pt`.
Defaults: both tiers, the four test sets (`config.EVAL.test_sets`), `config.EVAL.n_episodes` = 500 episodes per task
and eval set, the same seeds for every condition.

**Resuming.** Each cell is written when it finishes, so an interrupted evaluation resumes with the same command and
loses at most the cell in progress. A run whose checkpoint does not exist yet (e.g. P on the hard tier before it is
trained) is listed and skipped; run the command again once it exists.

**Run nothing else on the GPU during an evaluation.** Latency is measured during the rollouts, so a training run in
parallel would inflate it.

### Time (RTX 5080, measured on a 3-episode pilot)

| condition | per decision | easy tier, 4 test sets | hard tier, 4 test sets |
|---|---|---|---|
| random | - | seconds | seconds |
| B3 (3 seeds) | 1.2 ms | minutes | minutes |
| P-L12-attn-h1 (3 seeds) | 10-12 ms | about 1 h | about 2 h |
| P-L28-attn-h1 (3 seeds, hard only) | 21 ms | - | about 4-5 h |
| B2 | 44 ms | about 3 h | about 6 h |
| B1 | 170 ms | about 11 h | about 26 h; `test` only, as run: about 5.5 h |

Failing episodes run to the step limit (64 easy, 128 hard), so the baselines that fail most cost the most: B1 on all
four hard-tier sets alone would take more than a day, which is why it runs on `test` only there.

## What is checked before the first episode

- a trained run must be finished (`run.json` says so and `best.pt` exists);
- it must have been trained under the current `config.ENV` and `config.EVAL_SETS` (from its `run.json`);
- every head must read the same LLM as B1 and B2 (`config.LLM`), so all LLM conditions share the backbone;
- every existing cell must be the same measurement: its stored **identity** (model, tier, run, task, eval set,
  number and first/last seed of the episodes, environment and eval-set config, checkpoint SHA-256 for trained runs,
  `LLM` and `PROMPTED` for B1/B2) must equal the planned one. A different identity (e.g. a retrained checkpoint, a
  different `--n-episodes`) stops the evaluation; `--overwrite` replaces those cells.

## Outputs

`outputs/eval/<ID>/<tier>/<seed<k> | zero-shot>/`, two files per cell `<task>.<eval set>`:

- **`.json`**: the summary, written last (a cell is done when it exists): `identity`; `success_rate`, `mean_reward`,
  `mean_length`, `truncated_rate`; `decisions` and `latency_ms` (median, p90, mean); `expert_agreement` over
  `expert_labelled_steps`, and `label_errors`; `invalid_rate` (B1); for trained runs the checkpoint, model type,
  trainable parameters, training wall time and val results; git commit, GPU, library versions, wall time.
- **`.episodes.jsonl.gz`**: one JSON object per episode: `seed`, `mission`, `goal`, `actions`, `expert` (the expert's
  action in each state, `null` after a labelling failure), `ms` (time of each decision), `success`, `reward`,
  `length`, `truncated`, `label_error`, and per decision `logits` (P, B3), `scores` (B2) or `text` and `valid` (B1).
  Observations are not stored: `(task, eval set, seed)` and `actions` replay an episode exactly.

`python -m evaluation.summarise` writes `outputs/results/evaluation.md` (the tables, per tier) and
`evaluation_cells.csv` (one row per cell).

## Public API

| module | public names |
|---|---|
| `evaluation.evaluate` | `main`, `plan_runs`, `PlannedRun`, `check_run`, `cell_identity`, `ZERO_SHOT` |
| `evaluation.record` | `evaluate_cell`, `run_recorded_episode`, `summarise_episodes`, `read_episodes`, `decision_details` |
| `evaluation.summarise` | `main`, `tables`, `load_cells`, `group_runs`, `success_interval`, `gap_interval`, `mean_interval`, `wilson`, `Interval` |

## Design choices

- **Expert labels on the policy's own states.** Before each decision, `babytext.ExpertPolicy` is told the previous
  executed action and asked what it would do (its relabelling protocol); it only reads the environment, so the episode
  is identical with or without labels (tested on all four tasks). The labels give the action confusion matrices of
  §6.6. When the bot cannot continue (a policy opened a box with `toggle`, which makes it vanish), labelling stops for
  that episode and `label_error` says why. Agreement with the expert is not success: there are often several equally
  short paths.
- **Latency is the policy call alone**, at batch size 1, timed around `policy(obs)` (which ends with `int()`, so the
  GPU has finished). It includes tokenisation and the policy's own Python, not the environment step. A few untimed
  decisions first (`config.EVAL.warmup_decisions`) keep GPU start-up out of it.
- **Zero-shot conditions are deterministic** (greedy), so they run once; `random` is re-seeded for every cell, so each
  cell is reproducible on its own.
- **B1 on the hard tier: `test` only.** B1 succeeded in none of the 1000 hard-tier episodes it ran first
  (`putnext` on `test` and `test_dist16`), and both hard tasks need `pick up`, which B1 chose in none of its 128,000
  hard-tier decisions (10 of 241,396 on the easy tier, all in one episode). It also behaves like B2: same outcome in
  all 5000 episodes compared, same action in 99.5% (easy) and 99.9% (hard) of identical states, and B2 scores 0 on
  every hard-tier cell. Its remaining shifted-set cells (about 16 h) could not show a generalisation gap from a
  success of 0, so B1 runs on the hard tier on `test` only (both tasks). The `putnext`/`test_dist16` cell finished
  before the change and is kept; `summarise` shows it per task, and "-" for B1's hard-tier shifted sets and gaps.
- **Intervals.** Trained conditions: a t-interval over the training seeds, each seed's success being the mean over the
  tier's tasks. Zero-shot conditions: a Wilson interval over episodes; their generalisation gaps use a normal interval
  for a difference of proportions. Both are 95%.

## Limitations

- B1 has no generalisation gap on the hard tier (it runs on `test` only there; see design choices).
- B1 and B2 are untrained, so their intervals cover episode noise only, while the trained conditions' intervals
  cover seed-to-seed variation; the two kinds are not interchangeable.
- The zero-shot gap intervals treat `test` and the shifted set as independent samples, although they share seeds
  (for `test_nonce`, the same rooms): conservative.
- GPU results are not bit-for-bit reproducible (non-deterministic kernels); success can differ in rare tie-breaking
  decisions between runs of the same checkpoint.
