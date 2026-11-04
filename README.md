# very deep neurotica

SC4001 group project: a **small language model (SLM) with a learned spatio-temporal memory** as the policy for a
text-based gridworld (BabyAI-Text). A spatial encoder reads the agent's view, a GRU-style gate decides which views to
keep in a fixed-size memory, a 2-D space-time attention matrix reads from that memory, and the SLM receives the result
next to the mission text and chooses one of six actions. Training starts from expert demonstrations and continues
with reinforcement learning (see [Training](#training)).

> **Changed from the submitted proposal:** see [project_proposal.md §14](project_proposal.md#14-changes-since-the-submitted-proposal-23-oct-2026).

## Layout

The code is split into three sections. Each has its own README (install, run, public API, flow, design choices,
limitations).

| section | status | contents | README |
|---|---|---|---|
| `babytext/` | done | the environment: BabyAI-Text-style gridworld, eval sets and seeds, expert / random policies, rollout and evaluation, demonstration file format, visualisation, environment tools and tests | [babytext](babytext/README.md) |
| `models/` | B3 and SLM policy done (untrained) | the policies: B3 symbolic agent, the SLM policy (`slm_policy/`); prompted-SLM baselines to come | [models](models/README.md) |
| `training/` | done (untrained) | demonstration collection and statistics; rewards, behaviour cloning (stages 1-3, B3), recurrent PPO (stage 4); `--set` overrides and the run plan | [training](training/README.md) |

```
grp_proj/
├── project_proposal.md
├── config.py            all settings: environment, eval sets, data, evaluation, training, model sizes
├── uv.lock              locked dependency versions (source of requirements.txt)
├── requirements.txt     pinned dependencies for a venv, exported from uv.lock
├── environment.yml      conda env ("neurons") for the whole project, including GPU PyTorch
├── pyproject.toml       installs babytext, models, training and config (pip install -e .)
├── babytext/            the environment: package, tools/, tests/, README.md
├── models/              the policies: symbolic_agent.py (B3), README.md
├── training/            datasets and training: tools/, README.md
├── tests/               model and training tests
├── report/              figures; architecture/ holds the design + feasibility note (LaTeX + PDF)
└── data/, outputs/      generated demos, figures and results (gitignored)
```

Rules for using another section's code:

- Use only a section's public names: the `__all__` of its `__init__.py` or module, as listed in its README. Names
  starting with `_` are private.
- Dependencies point one way: `babytext` <- `models` <- `training`, and all three may read `config`. The environment
  imports nothing from the other two, and `models` never imports `training`, so a trained model can be evaluated
  without the training code.
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

**Tests:** `python -m pytest` (~2 min; the B3 tests are skipped when PyTorch is not installed).

**Changing dependencies:** edit `pyproject.toml`, then run `uv lock` and
`uv export --no-hashes --no-emit-project --extra dev --format requirements-txt -o requirements.txt` (`requirements.txt`
is exported from `uv.lock`), and update the pins in `environment.yml` to match. Commit all three.

## Configuration

All settings live in [config.py](config.py) (importable as `config`); edit the defaults there to change them project-wide.

| section | contents |
|---|---|
| `ENV` | room size, distractors, view size, time limit, held-out goals, invented words, prompt window |
| `EVAL_SETS` | the six eval sets: seed split and how each one's levels differ from `ENV` |
| `DATA` | demo counts per task, data and output folders |
| `EVAL` | episodes per evaluation |
| `TRAIN` | training protocol: seeds, demo-count curve, DAgger |
| `SYMBOLIC` | sizes of the B3 symbolic agent |
| `SLM` | sizes of the SLM policy: encoder, memory, attention, backbone, LoRA |
| `BC`, `PPO` | optimiser and algorithm settings of training stages 1-3 and 4 |

Changing `ENV` or `EVAL_SETS` changes the generated episodes, so re-collect demos afterwards.
`data/demos/summary.json` records the full config each dataset was generated with, and training refuses to start
when its `env` or `eval_sets` differ from `config.py`.

**Per-run changes without editing `config.py`:** `python -m training.train ... --set section.field=value`
(repeatable) changes `SLM`, `BC`, `PPO`, `SYMBOLIC` or `TRAIN` for one run, e.g. `--set slm.memory_slots=0` or
`--set bc.epochs=40`; this is how the controls are trained. `ENV`, `EVAL_SETS` and `DATA` cannot be set this way,
because they change the demos. The instances in `config.py` are frozen and imported at load time, so reassigning
`config.BC` from code reaches nothing: build a new one with `dataclasses.replace` and pass it explicitly. A run
started from a checkpoint (`--init`) uses the checkpoint's saved model config plus its `--set` values, not
`config.SLM`. Each run's `run.json` records the effective configs, the command line, the git commit and `config.py`
as it was. Details: [training README](training/README.md#changing-a-setting-for-one-run---set).

## Quick start

```python
from babytext import evaluate, get_seeds, make_env

env = make_env("goto", "train")                     # task in babytext.TASKS
obs, info = env.reset(seed=get_seeds("goto", "train", 1)[0])
obs["image"]                                        # 7x7 symbolic view: input to the spatial encoder (and B3)
obs["mission"]                                      # goal text: input to the SLM
obs["prompt"]                                       # GLAM prompt: input to the prompted-SLM baselines
obs, reward, terminated, truncated, info = env.step(2)   # "go forward"

result = evaluate(my_policy, "goto", "val", n_episodes=500)
result["success_rate"], result["mean_reward"], result["episodes"]
```

```bash
python -m training.tools.collect_demos              # expert demonstrations -> data/demos/
python -m babytext.tools.play --task goto --eval-set test_nonce   # play by hand
```

## Training

### Architecture

Per environment step *t* the model runs once (full diagram and shapes in the [models README](models/README.md#the-slm-policy-slm_policy); design and feasibility in [report/architecture/architecture_feasibility.pdf](report/architecture/architecture_feasibility.pdf)):

1. **Spatial encoder.** MiniGrid's 7x7 symbolic view (`obs["image"]`) is embedded per cell and passed through
   3x3 convolutions, giving one feature vector per cell. A **time encoding** of the step index *t* is added. The
   compass heading (`obs["direction"]`) is not used, because B3 and the GLAM prompt do not get it either.
2. **Gated memory.** A fixed-size memory of K slots, each holding one past view's features. After every step a
   GRU-style update gate decides *whether* to save the current view and a softmax address decides *which* slot it replaces.
3. **Space-time attention.** The current view and the K memory slots form a (K+1) x 49 grid of (time, cell)
   positions. A small convolutional scorer, conditioned on the mission, gives a weight to every position: this 2-D
   matrix is the space-time weight matrix. H read heads turn it into H weighted sums of features.
4. **SLM.** An adapter projects the H read-outs into H soft tokens. The SLM (Qwen3-0.6B, 28 layers)
   reads the mission as text followed by the soft tokens. Its layers are split into **low-level** (the first third,
   layers 0-9, which first process the new soft tokens) and **high-level** (layers 10-27, which carry the
   pretrained language and reasoning ability).
5. **Heads.** An action head (6 logits) and a value head (for PPO) read the SLM's last-token hidden state.

### Rewards

- **Immediate reward** for each action: the environment's reward (non-zero only on the final, successful step) plus
  potential-based shaping `λ · (γ·Φ(s_{t+1}) - Φ(s_t))` with `Φ(s) = -distance from the agent to the nearest goal object`.
  Potential-based shaping does not change which policy is optimal (Ng et al., 1999). It needs the agent's position,
  which the environment must first add to `info`.
- **Delayed reward over space-time**: the discounted return `G_t = Σ_k γ^k r_{t+k}`, estimated with GAE against the
  value head. Gradients from it flow back through the memory across the whole rollout, and this signal teaches
  the memory gate and the space-time weights which past (time, cell) positions mattered for success.

### Training steps

Every stage starts from the previous stage's weights and is selected on `val` only.

| stage | data and loss | trains | frozen |
|---|---|---|---|
| 0. demonstrations | `python -m training.tools.collect_demos`: 2000 train + 200 val expert episodes per task (`config.DATA`) | nothing | — |
| 1. warm start without SLM | behaviour cloning (cross-entropy on expert actions) on whole episodes in time order, small action head directly on the read-outs | spatial encoder, time encoding, memory gate, space-time scorer, small action head | — (no SLM yet) |
| 2. connect the SLM | behaviour cloning, same data | adapter, action head; encoder, gate and scorer continue at a lower learning rate | **all SLM parameters (low- and high-level)** |
| 3. adapt the SLM's input side | behaviour cloning, same data | **LoRA on the SLM's low-level layers** (attention q/v projections of layers 0-9), plus everything from stage 2 | **the SLM's high-level layers** and the LM head |
| 4. reinforcement learning | recurrent PPO in the environment on `train` seeds, immediate + delayed reward | everything trained in stage 3, plus the value head | **the SLM's high-level layers** and the LM head |

The model after stage 1 (M1) is the SLM policy without its SLM, and stages 2, 3 and 4 give M2, M3 and M4. Stage 4
(M4) is a stretch goal in the proposal (§6.3); the core results come from stages 1-3. B3 (`models/symbolic_agent.py`)
is trained like stage 1 on the same demonstrations and batches, with its own fixed-window architecture
(`python -m training.train --model b3`). Every model ID and control is one command; `python -m training.run_plan
--dry-run` prints them all ([training README](training/README.md#model-ids)). An optional DAgger round between stages 3 and 4 relabels states the model visits with the
expert's action. The SLM's thinking mode is not used: demonstrations contain no reasoning, and it is too slow inside
PPO (proposal §14).

### Tunable parameters

**Learned by gradient descent:**

| group | parameters | trained in stages |
|---|---|---|
| spatial encoder | cell embeddings and 3x3 conv kernels (the spatial part of the space-time weights) | 1-4 |
| time encoding | learned embedding of the step index (0 to the 64 / 128 step limit) | 1-4 |
| memory gate | GRU-style update gate `W_g, b_g` and the slot-address weights: what to keep and what to throw away | 1-4 |
| space-time attention | conv scorer that produces the (K+1) x 49 weight matrix, per read head | 1-4 |
| adapter | linear map from read-outs to the SLM's embedding width | 2-4 |
| SLM low-level | LoRA matrices on layers 0-9 | 3-4 |
| SLM high-level | layers 10-27, LM head | never |
| heads | action head (1-4), value head (4) | see left |

**Chosen on `val`** (not learned): memory slots K (8), feature width d (64), read heads H (4), number of low-level
SLM layers (10), LoRA rank (8), learning rate per group, shaping weight λ, discount γ, GAE λ, PPO clip and entropy
coefficient, number of demonstrations. Values in brackets are the starting points; they live in `config.SLM`,
`config.BC` and `config.PPO`. The thinking mode is not implemented (see the feasibility note).

Assumptions made where the original brief was ambiguous (what "learn the weights" of the reward means, and whether
memory should get pickier over the episode or over training) are listed in the
[models README](models/README.md#open-questions).

<details>
<summary>Original design brief</summary>

The model is an adapted SLM model to encode for temporatal information as well as spacial information. The input parameters will be encoded with space-time matrix with the corresponding weights corresponding to the attention to that point in spacetime. The model will be provided with the following rewards - immediate reward as a direct result from its action, and reward over space time in which the model will learn the weights for. The spacetime weight matrix will be a 2 dimensional matrix. At time $t$, the "memory" which is decided by the model whether to "save" the memory, similar to a GRU. The memory will be a fix length vector to fit the fixed-size space-time matrix with the corresponding weights.

The model will contain a SLM with the capability to think, as well as an encoder that encodes the observation input to encode for space time, as well as a GRU component that learns which feature observation is worth storing in its fixed length memory array, the input observation will be time encoded, to hopefully encourage more memorization at the start and being more picky of its memory as the training progresses. The tunable parameters will include: the GRU on which decides which memory to keep and which to throw, as well as the weights of spacetime matrix (similar to a coonvolutional neural network), as well as the SLM to take in these new inputs.

</details>

## Evaluation

Four single-room tasks in two tiers: `goto` and `pickup` (easy, 64-step limit) and `putnext` and `pickup_then_goto`
(hard, two objects, 128-step limit); six actions; reward `1 - 0.9 * steps / max_steps` on success, else 0. Every
episode is identified by `(task, eval_set, seed)`, so every condition sees the same rooms. Task details and the
reference results are in the [babytext README](babytext/README.md#tasks).

| eval set | seeds | distractors | goal objects | names | tests |
|---|---|---|---|---|---|
| `train` | train | 8 | all except held-out | real | demonstrations and RL rollouts |
| `val` | val | 8 | all except held-out | real | all model selection |
| `test` | test | 8 | all except held-out | real | in-distribution result |
| `test_dist16` | test | **16** | all except held-out | real | clutter: attention and memory under more distractors |
| `test_nonce` | test | 8 | all except held-out | **invented** (`red`→`dueno`, `ball`→`tism`, ...) | language grounding of the mission (the `image` codes are unchanged) |
| `test_heldout` | test | 8 | **one named object held out** (yellow box, red key, green ball) | real | generalisation to unseen goals |

Use `val` for all model selection and evaluate `test*` once at the end (proposal §6.5). Report every result next to
the random floor (high on `goto`: ~0.3), as mean and 95% CI over the training seeds in `config.TRAIN.seeds`.

Comparisons, all on the same seeds (proposal §6.3; research questions in §3):

| id | condition | trained by | question it answers |
|---|---|---|---|
| Random | uniform random actions | - | chance floor (report next to every result) |
| Expert | MiniGrid's BabyAI bot, reads the full state | - | ceiling; teacher for the demonstrations |
| B1 | Qwen3 reads the GLAM text prompt and generates text, mapped to the nearest valid action | - | can an untrained LM act from text? |
| B2 | Qwen3 log-probability scoring of the six action strings | - | same, without invalid outputs |
| B3 | frame-stacked CNN + FiLM on the last 3 symbolic views, mission read word by word, no LM | BC | standard BabyAI-style reference |
| M1 | SLM policy with an MLP in place of the SLM (encoder, memory, attention) | BC (stage 1) | base of the M chain |
| M2 | M1 + frozen Qwen3 reading the mission and 4 soft tokens | BC (stage 2) | does a pretrained LM help? |
| M3 | M2 + LoRA on Qwen3 layers 0-9 | BC (stage 3) | does tuning the LM's input layers help? |
| M4 *(stretch)* | M3 + recurrent PPO | PPO (stage 4) | does RL fix behaviour-cloning errors? |

Controls, each changing one thing:

| id | change | question it answers | status |
|---|---|---|---|
| M1-nomem | memory off (K = 0, `--set slm.memory_slots=0`) | does the memory help? | planned; trainable |
| M1-long | M1 trained for as many epochs as stages 1 and 2 combined (`--set bc.epochs=40`) | makes M1 vs M2 fair (M2 gets extra training) | planned; trainable |
| M2-L{10,18} | read Qwen3 at layer 10 or 18 (`--set slm.readout_layer=N`); layer 28 is M2 itself | where the decision-relevant information sits | planned (easy tier); trainable |
| M2-randlm | Qwen3 architecture with random weights (`--set slm.slm_pretrained=false`) | the LM's knowledge, or just its size? | stretch; trainable |
| M1-view{3,5} | agent view reduced to 3x3 or 5x5, with and without memory | makes memory necessary, so the K = 0 control can show a difference | stretch |

## Acknowledgements

Text description format and prompt template are ported from GLAM (Carta et al., 2023,
[flowersteam/Grounding_LLMs_with_online_RL](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)).
Environment dynamics, BabyAI levels and the expert bot are from Farama
[MiniGrid](https://github.com/Farama-Foundation/Minigrid) (Chevalier-Boisvert et al., 2019, 2023).
