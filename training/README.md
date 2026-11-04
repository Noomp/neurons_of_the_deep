# training

Builds the training data and trains the models in [`models/`](../models/README.md) by behaviour cloning on expert
demonstrations: the action head on the frozen LLM (P) and the symbolic agent (B3), under one shared protocol. B1 and
B2 are zero-shot and need no training. Depends on `babytext`, `models` and `config`; nothing imports from it.

Status: **dataset tools, hidden-state cache, the shared behaviour-cloning trainer, the command line and the run plan
done** (untrained so far). The method is in the [root README](../README.md#method) and
[project_proposal.md](../project_proposal.md) §6.4.

## Install

Same shared environment as the rest of the repository ([root README](../README.md#setup)). The dataset tools need
only the base install; caching and training need the model dependencies (`pip install -e ".[models]"`, or the conda
env).

## Run

From the repo root, inside the activated env:

```bash
python -m training.tools.collect_demos                       # expert demos -> data/demos/ (counts from config.DATA, ~30 s)
python -m training.tools.dataset_stats                       # charts of demos and eval sets -> outputs/figures/
python -m training.tools.cache_hidden                        # LLM hidden states of every demo step -> data/hidden/
python -m training.train --set head.layer=16 --set head.pooling=mean --out outputs/runs/p   # one head
python -m training.train --model b3 --out outputs/runs/b3                                     # B3
python -m training.run_plan --dry-run                        # every run of the comparison
```

`data/` and `outputs/` are git-ignored and can always be regenerated. `data/demos/summary.json` records the full
`config` each dataset was generated with; after changing `config.ENV` or `config.EVAL_SETS`, re-collect, then re-cache.

### Hidden-state cache

`python -m training.tools.cache_hidden [--tasks ...] [--layers ...] [--sequence-layers ...]` runs the frozen LLM once
over every demo step's prompt and writes, per demo dataset, into `data/hidden/<model>-<dtype>/`:

| file | contents |
|---|---|
| `<task>_<split>.last.npy`, `.mean.npy` | (steps, layers, 1024) last-token and mean vectors at `--layers` (default `config.HEAD.sweep_layers`) |
| `<task>_<split>.seq<l>.npy`, `.offsets.npy` | every token's vector at layer *l*, for attention pooling (only `--sequence-layers`), and each step's token range |
| `<task>_<split>.meta.json` | LLM config, layers, step count, SHA-256 of the demo file |

Vectors are the raw bits of bfloat16 (the LLM's own precision). Measured on an RTX 5080: about 96 prompts/s, so about
14 min for all 78k demo steps, or 82 s for `goto` train + val including one sequence layer. Sizes: about 2 KB per step
per layer and pooling (about 2.2 GB for all tasks at the 7 sweep layers), and about 0.6 MB per step per sequence layer
(about 14 GB for the easy tier, 28 GB for the hard tier, per layer). Training refuses a cache whose demo-file hash,
LLM or layers do not match, and says what to re-run.

## Training a model: `python -m training.train`

`--model head` (the default) trains P from the cache; `--model b3` trains B3. Other options: `--tasks` (default
`goto pickup`), `--n-demos N` (first N training demos per task), `--seed k` (default: the first of
`config.TRAIN.seeds`), `--id` (model ID recorded in `run.json`), `--device`, `--demos`, `--hidden`, and `--set` (below).

### The shared protocol (`bc.py`)

Both models go through the same `train_bc` loop, so the protocol is identical by construction:

- **Data:** the first `--n-demos` training episodes per task, and all val demo episodes.
- **Batches:** every step of `config.BC.batch_episodes` whole episodes, in the order `demos.batch_indices` draws from
  `--seed` alone, so P and B3 see the same batches for the same seed.
- **Optimiser:** AdamW with `config.BC`'s learning rate, weight decay and gradient clipping; cross-entropy on the
  expert's actions.
- **Learning-rate decay and early stopping:** after each epoch, val cross-entropy and accuracy on the val demos
  (eval mode). With `lr_decay="plateau"` (default) the learning rate halves (`lr_factor`) after `lr_patience` = 2
  epochs without a lower val cross-entropy (down to `min_lr`); `"cosine"` anneals it over `epochs`, `"none"` keeps it.
  Training stops after `patience` = 6 epochs without improvement, so every model trains until it stops improving;
  `epochs` = 200 is only an upper bound. Each epoch's `lr` is logged.
- **Selection:** the epoch with the lowest val cross-entropy is kept as `best.pt` (`last.pt` is the latest), and
  greedy success on `config.BC.val_episodes` val episodes per task is measured **once**, on that epoch. Configurations
  are compared by that success (`run_plan --summary`). Measuring success after every epoch would run the LLM at every
  decision of every val episode of every epoch: days of GPU time over the sweep (proposal §14).

**Why these defaults.** With the earlier fixed schedule (lr 3e-4, 20 epochs) both models under-trained: the head's val
cross-entropy flattened early and its val success stayed at the random floor. Measured on `goto` (all 2000 demos, val
cross-entropy of the selected epoch, plateau decay and early stopping unless noted):

| model | 3e-4, fixed, 20 epochs | 3e-4 | 1e-3 | 3e-3 | **1e-2** | 3e-2 |
|---|---|---|---|---|---|---|
| P, layer 16, last token | 0.722 | 0.713 | 0.679 | **0.638** | 0.639 | – |
| P, layer 16, mean | 0.678 | 0.616 | 0.586 | 0.510 | **0.505** | – |
| P, layer 28, last token | 0.746 | 0.728 | 0.723 | **0.626** | 0.638 | – |
| B3 | 0.568 | 0.569 | 0.518 | 0.490 | **0.413** | 0.536 |

An initial learning rate of 1e-2 is the best or within 0.012 of the best for every model, so one shared value keeps
the protocol identical for P and B3; runs then stop after 16-35 epochs. This used val demos only (no test data).

For P, training reads only the cache; the LLM runs once at the end, truncated to the head's layer, for the val
episodes (about 65 s for 100 `goto` episodes on an RTX 5080). For B3, each step is an independent frame-stacked sample
(`stack_demo_steps`).

### Outputs

Each run writes to `--out`:

- `metrics.jsonl`: one line per epoch (`ce`, `accuracy`, `val_ce`, `val_accuracy`), then one line with the selection:
  `selected_epoch`, its metrics, `val_success` and `val_<task>`.
- `best.pt` (the selected epoch, with its val success) and `last.pt`. They load with
  `models.action_head.load_policy` / `load_model` or `models.symbolic_agent.load_model`.
- `run.json`, which records what was actually used:

| key | contents |
|---|---|
| `model` | `type` (`head` / `b3`), the effective model `cfg`, for P the `llm` config and the `cache` meta data, trainable parameters |
| `bc`, `train` | the effective configs, after `--set` |
| `overrides` | the parsed `--set` values |
| `command`, `args`, `model_id`, `device` | the full command line and its parsed arguments |
| `git` | commit hash and whether the working tree was dirty |
| `demos` | `datasets` and `config` from `data/demos/summary.json` |
| `file_config` | `config.as_dict()`: config.py as it was, *before* overrides |
| `selected` | the selected epoch, its metrics and its val success |
| `warnings`, `finished`, `wall_time_s` | printed warnings; `finished` turns true when training ends |

### Changing a setting for one run: `--set`

`--set section.field=value` (repeatable) changes one field of `config.HEAD` (`head`), `config.LLM` (`llm`),
`config.BC` (`bc`), `config.SYMBOLIC` (`symbolic`) or `config.TRAIN` (`train`) for this run only. The value is parsed
with the field's type: `int`, `float`, `bool` (`true` / `false`), `none` for optional fields, and tuples as
comma-separated lists.

```bash
python -m training.train --set head.layer=12 --set head.pooling=attention --set bc.lr=1e-3 --out outputs/runs/try
```

- Unknown sections or fields, wrong types and a field set twice are errors. `env`, `eval_sets` and `data` cannot be
  set: they change which episodes the demos contain, so edit `config.py` and re-collect. `head.sweep_layers` cannot be
  set per run either: it only decides what `cache_hidden` caches. Setting a section the model never reads (e.g. `head`
  with `--model b3`) prints a warning.
- A changed `llm` needs its own cache (`cache_hidden --set llm.<field>=<value>`); training checks.
- The config instances are frozen and imported when a module loads (often as default arguments, e.g.
  `train_bc(..., cfg=BC)`), so reassigning `config.BC` from code does **not** reach the trainers. `train.py` builds the
  effective configs with `dataclasses.replace` and passes them explicitly to `ActionHead`, `SymbolicAgent`,
  `train_head_bc` and `train_symbolic_bc`. Do the same when calling these from code.

### Demos must match the config

Before training, `train.py` compares the `env` and `eval_sets` sections of `data/demos/summary.json` with config.py
and refuses to run if any value differs, if the summary is missing, or if it does not list a requested task (e.g. after
`collect_demos --tasks goto`). Re-collect with `python -m training.tools.collect_demos`. `--n-demos` larger than the
collected count is refused too.

## Run plan: `python -m training.run_plan`

```bash
python -m training.run_plan --dry-run                                  # every command
python -m training.run_plan --ids B3 "P-L*" --tiers easy               # step 1 on the easy tier, in order
python -m training.run_plan --summary --tiers easy                     # val success per ID, best first
python -m training.run_plan --attention-layers 12 16 --ids "P-*-attn-*" --tiers easy   # step 2
python -m training.run_plan --curve B3 P-L16-mean-h1 --ids "*-n*"      # demo-count curves
```

One command per model ID x seed (`config.TRAIN.seeds`) x tier (easy = `goto pickup`, hard =
`putnext pickup_then_goto`), into `outputs/runs/<id>/<tier>/seed<k>/`, after the `cache_hidden` commands the heads
need:

| ID | what | step |
|---|---|---|
| `B3` | the symbolic agent | 1 |
| `P-L<l>-<last\|mean>-h<n>` | head at layer *l* of `config.HEAD.sweep_layers` (4, 8, ..., 28), last-token or mean pooling, *n* = 1 or 2 MLP hidden layers: 28 IDs | 1 |
| `P-L<l>-attn-h<n>` | learned attention pooling at the layers passed with `--attention-layers` (the 1-2 best of step 1 by val success) | 2 |
| `<ID>-n<k>` | `--curve ID`: the ID trained on the first *k* demos per task, for every *k* in `config.TRAIN.demo_counts` below the full count (the full count is the ID's own run) | 3 |

`--ids` takes names or glob patterns. Without `--dry-run` the commands run in order, missing caches are built first,
and runs whose `run.json` says `"finished": true` are skipped, so an interrupted plan can be restarted. `--set` and
`--n-demos` are passed to every run (a per-ID `--set` of the same field wins).

Step 1 is 29 IDs x 3 seeds x 2 tiers = 174 runs. A head trains in a few seconds from the cache (one `goto` run until
early stopping: 4-7 s); the val success measurement (live LLM) is about a minute per run on the easy tier, longer on
the hard tier and for failing policies.

## Reproducibility

`--seed` seeds Python, NumPy and PyTorch (CPU and all GPUs) before the model is built, and the order of the training
episodes depends only on `--seed` and the data. The cache is deterministic for a given LLM, dtype and demo file.
**GPU runs are not bit-for-bit reproducible**: some CUDA kernels are non-deterministic, so two GPU runs with the same
seed can differ slightly. CPU runs repeat exactly.

## Public API

| what | where |
|---|---|
| dataset tools | `training.tools.collect_demos`, `training.tools.dataset_stats`, `training.tools.cache_hidden` (command line) |
| training | `python -m training.train` (one run), `python -m training.run_plan` (all IDs, summary) |
| the shared trainer | `training.bc`: `train_bc`, `bc_loss`, `evaluate_bc`, the `BCData` interface |
| per-model data and trainers | `training.bc_head`: `HeadDemos`, `head_demos`, `train_head_bc`; `training.bc_symbolic`: `SymbolicDemos`, `symbolic_demos`, `train_symbolic_bc` |
| the cache | `training.hidden_cache`: `cache_dir`, `check_cache`, `load_pooled`, `open_sequences`, `CacheError` |
| demos | `training.demos`: `load_demo_episodes`, `batch_indices`, `check_demo_config` |
| per-run settings | `training.overrides`: `parse_overrides`, `effective_configs` |
| helpers | `training.common`: `set_seed`, `RunLogger`, `validate` (any babytext policy), `git_state` |
| reading and writing demonstration files | `babytext.data`: `save_demos`, `load_episodes`, `load_steps`, `load_symbolic` |

The demo file format belongs to `babytext` because it stores `babytext.Episode`s and `babytext.tools.visualise_episode`
reads it too; this keeps the environment free of imports from `training`. The file layout is described in the
[babytext README](../babytext/README.md#demonstration-files).

## Flow

```
collect_demos ─► data/demos/<task>_<split>.{jsonl,npz} + summary.json
                     │                                   │
                     │ prompts                           │ views, actions, missions
                     ▼                                   ▼
cache_hidden: frozen LLM, once ─► data/hidden/...   stack_demo_steps (B3's frame-stacked inputs)
                     │                                   │
                     ▼                                   ▼
               HeadDemos                           SymbolicDemos
                     └───────────► bc.train_bc ◄─────────┘   same batches, optimiser, selection
                                        │
                                        ▼
        best.pt / last.pt / metrics.jsonl / run.json ─► HeadPolicy / SymbolicPolicy ─► babytext.evaluate
```

## Design choices

- **Behaviour cloning only.** Expert demonstrations give a working policy cheaply; reinforcement learning is out of
  scope (proposal §4, §14).
- **One loop for every trained condition.** P and B3 differ in their model and data class only, so a difference in
  results cannot come from a difference in training.
- **Cache once, sweep cheaply.** The LLM is frozen and prompts are fixed text, so the head's inputs never change; the
  layer x pooling x depth sweep needs no LLM pass until validation.
- **Prefix datasets.** Using the first N seeds of a split makes every smaller dataset a prefix of a larger one, so the
  demonstration-count curves (`config.TRAIN.demo_counts`) need no extra collection or caching.
- **Model selection on `val` only**; `test*` is evaluated once at the end (proposal §6.5).

## Limitations

- **Demonstrations are short and optimal.** The expert needs about 6 steps (easy tier) or 12 (hard tier), and the
  policy never sees how to recover from a mistake. DAgger (`babytext.ExpertPolicy` with `action_taken`) would address
  it; it is not implemented (`config.TRAIN.dagger_rounds` is unused).
- **Validation is sequential.** `training.common.validate` runs one environment at a time; batched evaluation would
  make the head's val episodes several times faster.
- **Attention pooling needs large caches** (about 14-28 GB per layer per tier), which is why it runs at two layers only.
