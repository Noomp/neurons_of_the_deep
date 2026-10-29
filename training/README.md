# training

Builds the training data and (planned) trains the models in [`models/`](../models/README.md): behaviour cloning on
expert demonstrations, then reinforcement learning with immediate and delayed rewards. Depends on `babytext`, `models`
and `config`; nothing imports from it.

Status: **dataset tools done**; rewards and the training stages are **not implemented yet**. The stages are specified
in the [Training section of the root README](../README.md#training).

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

## Public API

| what | where |
|---|---|
| dataset tools | `training.tools.collect_demos`, `training.tools.dataset_stats` (command line) |
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
 stage 1  behaviour cloning, no SLM (encoder, memory, attention, small action head)                   planned
 stage 2  behaviour cloning with the SLM frozen (adds adapter + action head on the SLM)               planned
 stage 3  behaviour cloning + LoRA on the SLM's low-level layers                                      planned
 stage 4  recurrent PPO in the environment, immediate + delayed rewards                               planned
                     │
                     ▼
 SLMAgentPolicy(model) ─► babytext.evaluate(policy, task, "val", n) for selection; test* once at the end
```

What each stage trains and freezes is in the root README's Training section. B3 (`models.symbolic_agent`) is trained
with the same demonstrations, seeds and DAgger settings as stage 1-3 (`config.TRAIN`).

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
  mode; it can only be shaped by reward in stage 4, or used zero-shot.
- **Cost of training through the SLM.** The SLM's inputs depend on the learned memory and its low-level layers are tuned,
  so hidden states cannot be cached once (as the proposal planned). Every update runs the SLM forward and backward;
  stage 4 also runs it inside the rollout loop.
- **`config.TRAIN` still describes the proposal's frozen-LLM sweep** (`layer_stride`, `poolings`, Qwen2.5 backbone). Its
  seeds, demonstration counts and DAgger settings still apply; the rest should be replaced when the training code lands.
