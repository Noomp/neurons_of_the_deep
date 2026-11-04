# training

Builds the training data and trains the models in [`models/`](../models/README.md): behaviour cloning on
expert demonstrations, then reinforcement learning with immediate and delayed rewards. Depends on `babytext`, `models`
and `config`; nothing imports from it.

Status: **dataset tools, rewards, all four training stages and the B3 trainer done**; every model ID of the comparison
can be trained with one command (untrained so far). The stages are specified in the
[Training section of the root README](../README.md#training); the feasibility assessment and compute estimates are in
[report/architecture/architecture_feasibility.pdf](../report/architecture/architecture_feasibility.pdf).

## Install

Same shared environment as the rest of the repository ([root README](../README.md#setup)). The dataset tools need
only the base install; training needs the model dependencies (`pip install -e ".[models]"`, or the conda env).

## Run

From the repo root, inside the activated env:

```bash
python -m training.tools.collect_demos                       # expert demos -> data/demos/ (counts from config.DATA, ~30 s)
python -m training.tools.collect_demos --n-train 500 --tasks goto
python -m training.tools.dataset_stats                       # charts of demos and eval sets -> outputs/figures/
```

`data/` and `outputs/` are git-ignored and can always be regenerated. `data/demos/summary.json` records the full
`config` each dataset was generated with; after changing `config.ENV` or `config.EVAL_SETS`, re-collect.

## Training a model

```bash
python -m training.train --stage 1 --tasks goto pickup --out outputs/runs/m1          # CPU is enough
python -m training.train --stage 2 --tasks goto pickup --init outputs/runs/m1/best.pt --out outputs/runs/m2
python -m training.train --stage 3 --tasks goto pickup --init outputs/runs/m2/best.pt --out outputs/runs/m3
python -m training.train --stage 4 --tasks goto pickup --init outputs/runs/m3/best.pt --out outputs/runs/m4
python -m training.train --model b3 --tasks goto pickup --out outputs/runs/b3          # B3, CPU is enough
```

`--model slm` (the SLM policy, needs `--stage`) is the default. Other options: `--n-demos N` (first N training demos
per task), `--seed k` (default: the first of `config.TRAIN.seeds`), `--id` (model ID recorded in `run.json`),
`--device`, `--demos`, and `--set` (below).

Each run writes to `--out`:

- `metrics.jsonl`: one line per epoch (BC) or update (PPO). Both BC trainers log `ce`, `accuracy`, `val_ce`,
  `val_accuracy`, `val_success` and `val_<task>`; the SLM policy also logs `write_rate`.
- `last.pt` (every epoch) and `best.pt` (best val success, or lowest val CE when `bc.val_episodes=0`). They load with
  `models.slm_policy.load_model` or `models.symbolic_agent.load_model`.
- `run.json`, which records what was actually used:

| key | contents |
|---|---|
| `model` | `type` (`slm` / `b3`), stage, `--init`, the **effective model config** `cfg`, trainable parameters |
| `bc`, `ppo`, `symbolic`, `train` | the effective configs, after `--set` |
| `overrides` | the parsed `--set` values |
| `command`, `args`, `model_id`, `device` | the full command line and its parsed arguments |
| `git` | commit hash and whether the working tree was dirty |
| `demos` | `datasets` and `config` from `data/demos/summary.json` |
| `file_config` | `config.as_dict()`: config.py as it was, *before* overrides and checkpoint configs |
| `warnings`, `finished`, `wall_time_s` | printed warnings; `finished` turns true when training ends |

### Changing a setting for one run: `--set`

`--set section.field=value` (repeatable) changes one field of `config.SLM` (`slm`), `config.BC` (`bc`), `config.PPO`
(`ppo`), `config.SYMBOLIC` (`symbolic`) or `config.TRAIN` (`train`) for this run only. The value is parsed with the
field's type: `int`, `float`, `bool` (`true` / `false`), `none` for optional fields, and tuples as comma-separated lists.

```bash
python -m training.train --stage 1 --set slm.memory_slots=0 --set bc.lr=1e-3 --out outputs/runs/try
python -m training.train --stage 2 --init outputs/runs/m1/best.pt --set slm.lora_targets=q_proj,k_proj,v_proj --out ...
```

- Unknown sections or fields, wrong types and a field set twice are errors. `env`, `eval_sets` and `data` cannot be
  set: they change which episodes the demos contain, so edit `config.py` and re-collect. `slm.backbone` follows
  `--stage`. Setting a section the run never reads (e.g. `ppo` in stage 1) prints a warning.
- The config instances are frozen and imported when a module loads (often as default arguments, e.g.
  `train_bc(..., cfg=BC)`), so reassigning `config.BC` from code does **not** reach the trainers. `train.py` builds the
  effective configs with `dataclasses.replace` and passes them explicitly to `SLMPolicy`, `load_model`, `SymbolicAgent`,
  `train_bc`, `train_symbolic_bc` and `train_ppo`. Do the same when calling these from code.
- **With `--init`, `config.SLM` is not used.** The model config is the checkpoint's saved config, then the stage's
  backbone, then the `slm` overrides; so stage 2+ keeps the architecture its checkpoint was trained with. If
  `config.SLM` differs from that config on a field you did not `--set`, a warning names the fields (e.g. after editing
  `readout_layer` in config.py, which has no effect on a stage-2 run; use `--set slm.readout_layer=10`).

### Demos must match the config

Before training, `train.py` compares the `env` and `eval_sets` sections of `data/demos/summary.json` with config.py
and refuses to run if any value differs, if the summary is missing, or if it does not list a requested task (e.g. after
`collect_demos --tasks goto`). Re-collect with `python -m training.tools.collect_demos`. `--n-demos` larger than the
collected count is refused too.

### Model IDs

Every trained model of the comparison (root README, [Evaluation](../README.md#evaluation)) is one command:

| ID | command | differs from |
|---|---|---|
| B3 | `--model b3` | - |
| M1 / M2 / M3 | `--stage 1..3`, `--init` the previous stage's `best.pt` | - |
| M1-nomem | `--stage 1 --set slm.memory_slots=0` | M1 |
| M1-long | `--stage 1 --set bc.epochs=40` (stage-1 + stage-2 epochs, `2 x config.BC.epochs`) | M1 |
| M2-randlm | `--stage 2 --init <M1> --set slm.slm_pretrained=false` | M2 |
| M2-L{10,18} | `--stage 2 --init <M1> --set slm.readout_layer=N` (layer 28 is all of Qwen3-0.6B, i.e. M2) | M2 |

All of them share the protocol: same demos (first `--n-demos` per task), same `config.BC` optimiser settings, same
seeds, same selection rule (val success after every epoch). Controls are recorded in the checkpoint's saved config,
so an evaluation needs no special cases. The core results are behaviour cloning only, so M4 (`--stage 4 --init <M3>`,
PPO) is a stretch goal that still works by hand but is not in the run plan.

### Run plan

```bash
python -m training.run_plan --dry-run                                # every command
python -m training.run_plan --dry-run --ids M1 M2 --tiers easy --seeds 0
python -m training.run_plan --ids B3 M1 M2 --tiers easy              # run them one after another
```

One command per model ID x seed (`config.TRAIN.seeds`) x tier (easy = `goto pickup`, hard =
`putnext pickup_then_goto`), into `outputs/runs/<id>/<tier>/seed<k>/`. A later stage's `--init` is the `best.pt` of the
same tier and seed one stage earlier, and models are listed after the runs they start from. Without `--dry-run` it
runs the commands in order and skips runs whose `run.json` says `"finished": true`. `--set` and `--n-demos` are passed
to every run (a per-ID `--set` of the same field wins; M1-long doubles the effective `bc.epochs`).

### Reproducibility

`--seed` seeds Python, NumPy and PyTorch (CPU and all GPUs) before the model is built, and the order of the training
episodes depends only on `--seed` and the data: both BC trainers draw the same batches from `demos.batch_indices` for
the same seed. The random LM of M2-randlm is drawn from its own seed (`slm.slm_init_seed`), independent of `--seed`.
**GPU runs are not bit-for-bit reproducible**: some CUDA kernels (e.g. atomic adds in backward passes) are
non-deterministic, so two GPU runs with the same seed can differ slightly. CPU runs repeat exactly.

### Modules

| module | role |
|---|---|
| `stages.py` | `STAGES` (backbone, algorithm, learning-rate scale per parameter group) and `make_optimizer` |
| `demos.py` | demonstration files -> padded (B, T) episode batches; `batch_indices` (shared batch order); `check_demo_config` |
| `bc.py` | behaviour cloning of the SLM policy (stages 1-3) |
| `bc_symbolic.py` | behaviour cloning of B3: `train_symbolic_bc`, `symbolic_demos`, `symbolic_bc_loss` |
| `vec_env.py` | the environment side of RL: N envs, auto-reset on fresh train seeds, step counters, shaping potential |
| `rewards.py` | potential-based shaping and GAE, no environment code |
| `ppo.py` | recurrent PPO (stage 4): one sequence per env per rollout, memory stored at its start |
| `overrides.py` | `--set` parsing: `parse_overrides`, `effective_configs` |
| `common.py` | `set_seed`, `RunLogger`, `validate` (any babytext policy), `git_state` |
| `train.py` | command line for one run |
| `run_plan.py` | command lines for every model ID x seed x tier |

### B3 trainer

`train_symbolic_bc` follows `train_bc` wherever B3 allows: the same `DemoEpisode`s, the same batches of whole episodes
for the same seed, AdamW with `config.BC`'s epochs, lr, weight decay and gradient clipping, and the same
selection and metric names. B3 has no memory, so there is no write penalty, and one batch is **all steps of
`batch_episodes` episodes** (each step a separate frame-stacked sample from `stack_demo_steps`), so every update sees
the same data as the SLM trainer's. BatchNorm is in eval mode for the val loss and the val episodes.

## Public API

| what | where |
|---|---|
| dataset tools | `training.tools.collect_demos`, `training.tools.dataset_stats` (command line) |
| training | `python -m training.train` (one run), `python -m training.run_plan` (all model IDs) |
| trainers from code | `training.bc.train_bc`, `training.bc_symbolic.train_symbolic_bc`, `training.ppo.train_ppo`; pass every config explicitly |
| per-run settings | `training.overrides.parse_overrides`, `effective_configs` |
| reading and writing demonstration files | `babytext.data`: `save_demos`, `load_episodes`, `load_steps`, `load_symbolic` |
| training protocol settings | `config.TRAIN` (seeds, demonstration counts, DAgger rounds) |

The file format belongs to `babytext` because it stores `babytext.Episode`s and `babytext.tools.visualise_episode`
reads it too; this keeps the environment free of imports from `training`. The file layout is described in the
[babytext README](../babytext/README.md#demonstration-files).

## Flow

```
training.tools.collect_demos ─► babytext.data.save_demos ─► data/demos/<task>_<split>.{jsonl,npz}              done
                                                                  │
          babytext.data.load_symbolic + load_episodes ◄───────────┘
                     │ cut into episodes by `episode`, ordered by `t`
                     ▼
 stage 1  behaviour cloning, no SLM (encoder, memory, attention, small action head)                   done   
 stage 2  behaviour cloning with the SLM frozen (adds adapter + action head on the SLM)               done   
 stage 3  behaviour cloning + LoRA on the SLM's low-level layers                                      done   
 stage 4  recurrent PPO in the environment, immediate + delayed rewards                               done   
 B3       behaviour cloning of the symbolic agent on the same episodes and batches (bc_symbolic.py)   done
                     │
                     ▼
 SLMAgentPolicy(model) / SymbolicPolicy(model) ─► babytext.evaluate(policy, task, "val", n) for selection; test* once at the end
```

What each stage trains and freezes is in the root README's Training section. Stage 4 (M4) is a stretch goal in the
proposal (§6.3). B3 (`models.symbolic_agent`) is trained by `bc_symbolic.py` with the same demonstrations, seeds,
batches and `config.BC` settings as stages 1-3 (`python -m training.train --model b3`).

## Design choices

- **Expert first, RL second.** Rewards in this environment are sparse (one reward at the end of a successful
  episode), so learning from reward alone with an SLM in the loop would be slow. Behaviour cloning gives a working
  policy cheaply; RL then improves on it and corrects compounding errors.
- **Whole episodes, not shuffled steps.** The SLM policy carries memory across time, so behaviour cloning replays each
  demonstration in time order (truncated back-propagation through time), using the `episode` and `t` arrays.
- **Prefix datasets.** Using the first N seeds of a split makes every smaller dataset a prefix of a larger one, so the
  demonstration-count curves (`config.TRAIN.demo_counts`) need no extra collection.
- **Model selection on `val` only**; `test*` is evaluated once at the end (proposal §6.5).

## Limitations

- **Demonstrations carry no per-step reward.** Only the final `reward` and `success` per episode are stored. The shaped
  immediate reward must be recomputed by replaying `(task, eval_set, seed, actions)`, and needs the agent's position,
  which `babytext` does not put in `info` yet.
- **Demonstrations are short and optimal.** The expert needs ~6 steps (easy tier) or ~12 (hard tier), so (a) memory rarely
  has to span long in stage 1-3 data, and (b) the policy never sees how to recover from a mistake. Stage 4 and optional
  DAgger (`babytext.ExpertPolicy` with `action_taken`) address (b).
- **No thinking traces.** The expert gives actions, not reasoning, so behaviour cloning cannot teach the SLM's thinking
  mode, and it is too slow inside PPO; it is not used (proposal §14).
- **Cost of training through the SLM.** The SLM's inputs depend on the learned memory and its low-level layers are tuned,
  so hidden states cannot be cached once (as the submitted proposal planned). Every update runs the SLM forward and backward;
  stage 4 also runs it inside the rollout loop.
- **The M2-randlm LM is regenerated, not stored.** Checkpoints skip the frozen LM, so a random one is rebuilt from
  `slm_init_seed` on load. Its fingerprint is saved and checked: loading with a different torch / transformers
  version that draws other weights raises an error instead of silently using a different LM.
- **Not implemented yet:** DAgger relabelling (`config.TRAIN.dagger_rounds` is unused), a KL penalty towards the BC
  policy during PPO, and batched validation (`training.common.validate` runs one environment at a time).
