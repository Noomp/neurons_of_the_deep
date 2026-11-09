# very deep neurotica

SC4001 group project: **can a lightweight action head trained on the hidden states of a frozen LLM serve as an
effective and efficient policy for a text-based gridworld (BabyAI-Text), and what does it trade away compared with
language-grounded action selection?** A frozen Qwen3-0.6B reads the GLAM text prompt; a small MLP reads one layer's
hidden state and picks one of six actions in a single forward pass. It is compared with the same LLM choosing actions
through language (generation, log-probability scoring) and with a small non-language agent trained on the same
demonstrations ([project_proposal.md](project_proposal.md), §3).

> **Changed from the submitted proposal:** see [project_proposal.md §14](project_proposal.md#14-changes-since-the-submitted-proposal-23-oct-2026).
> A different design (an SLM reading a learned spatio-temporal memory, M1-M4) was pursued between 23 Oct and 4 Nov and is
> kept on git branch `archive/slm-memory-policy`.

## Layout

The code is split into five sections. Each has its own README (install, run, public API, flow, design choices,
limitations).

| section | status | contents | README |
|---|---|---|---|
| `babytext/` | done | the environment: BabyAI-Text-style gridworld, eval sets and seeds, expert / random policies, rollout and evaluation, demonstration file format, visualisation, environment tools and tests | [babytext](babytext/README.md) |
| `models/` | done | the policies: the action head on the frozen LLM (P), the zero-shot baselines B1 / B2, the symbolic agent B3, and the shared frozen-LLM wrapper | [models](models/README.md) |
| `training/` | done | demonstration collection and statistics, hidden-state caching, one behaviour-cloning trainer for P and B3, `--set` overrides and the run plan | [training](training/README.md) |
| `evaluation/` | done (run) | the final evaluation: every condition on the test sets with every decision recorded (resumable), and the report's tables | [evaluation](evaluation/README.md) |
| `analysis/` | done | failure analysis of the recorded test episodes (categories, confusion matrices, drift) and the report's result figures | [analysis](analysis/README.md) |

```
grp_proj/
├── project_proposal.md
├── config.py            all settings: environment, eval sets, data, evaluation, training, model sizes
├── uv.lock              locked dependency versions (source of requirements.txt)
├── requirements.txt     pinned dependencies for a venv, exported from uv.lock
├── environment.yml      conda env ("neurons") for the whole project, including GPU PyTorch
├── pyproject.toml       installs babytext, models, training, evaluation and config (pip install -e .)
├── babytext/            the environment: package, tools/, tests/, README.md
├── models/              the policies: action_head.py (P), prompted.py (B1, B2), symbolic_agent.py (B3), frozen_lm.py
├── training/            datasets, hidden-state cache and training: tools/, README.md
├── evaluation/          final evaluation on the test sets and its tables: README.md
├── analysis/            failure analysis and the report's result figures: README.md
├── tests/               model, training and evaluation tests
├── report/              val_results.md and test_results.md (all results), figures/ (architecture diagrams, result figures)
└── data/, outputs/      generated demos, hidden-state caches, figures and results (gitignored)
```

Rules for using another section's code:

- Use only a section's public names: the `__all__` of its `__init__.py` or module, as listed in its README. Names
  starting with `_` are private.
- Dependencies point one way: `babytext` <- `models` <- `training`, and all three may read `config`. The environment
  imports nothing from the other two, and `models` never imports `training`, so a trained model can be evaluated
  without the training code. `evaluation` uses `babytext`, `models` and `training.common`; `analysis` uses `babytext`
  and `evaluation` (no model). Only `analysis` imports `evaluation`, and nothing imports `analysis`.
- Models reach the environment as policies: a callable `policy(obs) -> action index`, optionally with `reset(env)` to
  clear per-episode state, passed to `babytext.run_episode` / `babytext.evaluate`.

## Setup

Python 3.10 or newer (3.11 pinned). Pick one of the three; all install the same pinned versions and run from the repo
root, because the data and output paths in `config.py` are relative to it.

**venv + pip**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # pinned environment + test dependencies
pip install -e .                   # babytext, models, training and config (editable); the environment works from here
pip install -e ".[models]"         # PyTorch + Transformers for the models (install the right CUDA build of torch first)
```

**uv** (uses `uv.lock` directly, and Python 3.11 from `.python-version`)

```bash
uv sync --extra dev                   # creates .venv/ with the locked versions and installs the project (editable)
uv sync --extra dev --extra models    # also PyTorch + Transformers
source .venv/bin/activate             # Windows: .venv\Scripts\activate
```

**conda**

```bash
conda env create -f environment.yml   # creates the "neurons" env, including GPU PyTorch
conda activate neurons
pip install -e .                      # add ".[models]" for Transformers
```

Always work inside the activated conda env (or `conda run -n neurons ...`): the conda PyTorch build loads its DLLs from
the env's `Library/bin`, which is only on the search path when the env is active; calling the env's `python.exe`
directly makes `import torch` fail with "DLL load failed".

**Tests:** `python -m pytest` (~80 s; the model and training tests are skipped when PyTorch or Transformers is
not installed, and use a tiny random Qwen3, so nothing is downloaded).

**Changing dependencies:** edit `pyproject.toml`, then run `uv lock` and
`uv export --no-hashes --no-emit-project --extra dev --format requirements-txt -o requirements.txt` (`requirements.txt`
is exported from `uv.lock`), and update the pins in `environment.yml` to match. Commit all three.

## Configuration

All settings live in [config.py](config.py) (importable as `config`); edit the defaults there to change them project-wide.

| section | contents |
|---|---|
| `ENV` | room size, distractors, view size, time limit, held-out goals, invented words, prompt window |
| `EVAL_SETS` | the six eval sets: seed split and how each one's levels differ from `ENV` |
| `DATA` | demo counts per task, data, hidden-state cache and output folders |
| `EVAL` | episodes per evaluation, the test sets of the final evaluation, warm-up decisions before timing |
| `ANALYSIS` | conditions in the failure analysis, the "stuck" thresholds, the report's figure folder |
| `TRAIN` | training protocol: seeds, demo-count curve, DAgger |
| `LLM` | the frozen LLM shared by B1, B2 and P (Qwen3-0.6B, bfloat16) |
| `HEAD` | the action head P: layer read, pooling, MLP size, and the layers swept |
| `PROMPTED` | B1's generation length and mapping model, B2's length normalisation |
| `SYMBOLIC` | sizes of the B3 symbolic agent |
| `BC` | behaviour cloning (P and B3): epochs, batch, learning rate, weight decay, gradient clip, val episodes |

Changing `ENV` or `EVAL_SETS` changes the generated episodes, so re-collect demos afterwards; changing the demos,
`LLM` or `HEAD.sweep_layers` needs a new hidden-state cache. `data/demos/summary.json` records the full config each
dataset was generated with, each cache records its LLM and the hash of its demo file, and training refuses to start
when either does not match.

**Per-run changes without editing `config.py`:** `python -m training.train ... --set section.field=value`
(repeatable) changes `HEAD`, `LLM`, `BC`, `SYMBOLIC` or `TRAIN` for one run, e.g. `--set head.layer=16` or
`--set bc.epochs=40`; this is how the layer and pooling sweep is run. `ENV`, `EVAL_SETS` and `DATA` cannot be set this
way, because they change the demos. The instances in `config.py` are frozen and imported at load time, so reassigning
`config.BC` from code reaches nothing: build a new one with `dataclasses.replace` and pass it explicitly. Each run's
`run.json` records the effective configs, the command line, the git commit and `config.py` as it was. Details:
[training README](training/README.md#changing-a-setting-for-one-run---set).

## Quick start

```python
from babytext import evaluate, get_seeds, make_env

env = make_env("goto", "train")                     # task in babytext.TASKS
obs, info = env.reset(seed=get_seeds("goto", "train", 1)[0])
obs["prompt"]                                       # GLAM prompt: input to the LLM (B1, B2, P)
obs["image"]                                        # 7x7 symbolic view: input to B3
obs["mission"]                                      # goal text
obs, reward, terminated, truncated, info = env.step(2)   # "go forward"

result = evaluate(my_policy, "goto", "val", n_episodes=500)
result["success_rate"], result["mean_reward"], result["episodes"]
```

```bash
python -m training.tools.collect_demos              # expert demonstrations -> data/demos/
python -m training.tools.cache_hidden               # frozen-LLM hidden states of every demo step -> data/hidden/
python -m training.run_plan --dry-run               # every training command
python -m evaluation.evaluate --models random B3 P-L12-attn-h1 B2 B1 --dry-run   # final evaluation plan
python -m analysis.failures && python -m analysis.figures   # failure analysis, then the report's figures
python -m babytext.tools.play --task goto --eval-set test_nonce   # play by hand
```

## Method

**P, the action head** (proposed; [models README](models/README.md#p-the-action-head-action_headpy)):

```
obs["prompt"] (GLAM prompt, ~270-300 tokens) ─► frozen Qwen3-0.6B, layers 1..l ─► token vectors at layer l
     ─► pooling: last token | mean | learned attention ─► LayerNorm ─► MLP (1-2 hidden layers) ─► 6 action logits
```

Only the head is trained (about 0.26-0.4M parameters). The LLM never changes, so the hidden states of every
demonstration step are computed once and cached (about 14 min on an RTX 5080 for all 78k steps); a head then trains in
seconds to minutes. Acting runs the LLM once per decision, truncated at layer *l* (21 ms for all 28 layers at batch
size 1).

**Language-grounded baselines** on the same LLM and prompt, zero-shot: **B1** generates a short continuation and maps
it to the nearest action name by sentence-embedding similarity; **B2** scores the six action names by their
length-normalised log-probability after the prompt. Both choose through the *meaning* of the action names; the head's
six outputs mean nothing until demonstrations teach them. What the head gives up by that is RQ3.

**B3, the non-language agent**: a frame-stacked CNN with FiLM on the last three 7x7 symbolic views and the two actions
between them (the same window as the prompt), plus the goal read word by word. It needs no LLM, so P vs B3 asks
whether running the LLM pays for its cost.

**Training** ([training README](training/README.md)): behaviour cloning on 2000 expert demonstrations per task, with
one shared loop for P and B3 (same demonstrations, seeds, batches, optimiser settings and selection rule). The
learning rate (initially 3e-3) halves when val cross-entropy stalls for 3 epochs, and training stops once it has not
improved for 10 epochs. Each run keeps the epoch with the lowest val cross-entropy and measures val greedy success once on it;
layers, poolings and other configurations are compared by that success. The layer sweep has two steps: last-token and mean pooling at
layers 4, 8, ..., 28 with 1 or 2 MLP hidden layers, then learned attention pooling at the best layers (12, 16 and 20 on the easy tier, which tied) and at the final layer 28 as a control.
No reinforcement learning, no LLM fine-tuning, and no thinking mode (proposal §4, §14).

## Evaluation

Four single-room tasks in two tiers: `goto` and `pickup` (easy, 64-step limit) and `putnext` and `pickup_then_goto`
(hard, two objects, 128-step limit); six actions; reward `1 - 0.9 * steps / max_steps` on success, else 0. Every
episode is identified by `(task, eval_set, seed)`, so every condition sees the same rooms. Task details and the
reference results are in the [babytext README](babytext/README.md#tasks).

| eval set | seeds | distractors | goal objects | names | tests |
|---|---|---|---|---|---|
| `train` | train | 8 | all except held-out | real | demonstrations |
| `val` | val | 8 | all except held-out | real | all model selection |
| `test` | test | 8 | all except held-out | real | in-distribution result |
| `test_dist16` | test | **16** | all except held-out | real | clutter: many more descriptions in the prompt |
| `test_nonce` | test | 8 | all except held-out | **invented** (`red`→`dueno`, `ball`→`tism`, ...) | language grounding: names the LLM has never seen (B3's `image` codes are unchanged) |
| `test_heldout` | test | 8 | **one named object held out** (yellow box, red key, green ball) | real | generalisation to unseen goals |

Use `val` for all model selection and evaluate `test*` once at the end (proposal §6.5), with
`python -m evaluation.evaluate` ([evaluation README](evaluation/README.md)). Report every result next to
the random floor (high on `goto`: ~0.3), as mean and 95% CI over the training seeds in `config.TRAIN.seeds`.

Comparisons, all on the same seeds (proposal §6.3; research questions in §3):

| id | condition | trained by | question it answers |
|---|---|---|---|
| Random | uniform random actions | - | chance floor (report next to every result) |
| Expert | MiniGrid's BabyAI bot, reads the full state | - | ceiling; teacher for the demonstrations |
| B1 | Qwen3 reads the GLAM prompt and generates text, mapped to the nearest valid action | - | RQ1, RQ3: can the untrained LLM act through language? |
| B2 | Qwen3 log-probability scoring of the six action strings | - | RQ1, RQ3: same, without invalid outputs |
| B3 | frame-stacked CNN + FiLM on the last 3 symbolic views and 2 actions, goal read word by word, no LLM | BC | RQ1: does the LLM pay for its cost? |
| P-L*l*-*pool*-h*n* | action head on Qwen3 layer *l* (4, 8, ..., 28), pooling last / mean (all layers) or attention (best layers 12, 16, 20, and 28 as a control), *n* = 1-2 MLP hidden layers | BC | RQ1 (best variant), RQ2 (layer and pooling), RQ3 |

## Acknowledgements

Text description format and prompt template are ported from GLAM (Carta et al., 2023,
[flowersteam/Grounding_LLMs_with_online_RL](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)).
Environment dynamics, BabyAI levels and the expert bot are from Farama
[MiniGrid](https://github.com/Farama-Foundation/Minigrid) (Chevalier-Boisvert et al., 2019, 2023).
