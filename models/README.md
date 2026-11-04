# models

The policies trained in this project. Depends on `babytext` and `config`; must never import `training`, so a trained
model can be loaded and evaluated without the training code.

| module | status | what |
|---|---|---|
| [`symbolic_agent.py`](symbolic_agent.py) | done (untrained) | **B3**, the non-LLM baseline: frame-stacked CNN + FiLM on the symbolic grid |
| [`slm_policy/`](slm_policy/__init__.py) | done (untrained) | **the main model**: SLM with spatial encoder, write-sparse slot memory and space-time attention |
| `prompted.py` | not implemented | **B1 / B2**: the same SLM prompted with the GLAM prompt (generate-and-map, log-probability scoring) |

Where the original design brief (kept in the root README's [Training](../README.md#training) section) was ambiguous,
the choice made here is marked *(interpretation)* and listed under [Open questions](#open-questions).

## Install

Same shared environment as the rest of the repository ([root README](../README.md#setup)), plus the model dependencies:

```bash
pip install -e ".[models]"     # venv: torch + transformers;  conda: torch already comes from environment.yml
python -m pytest tests/test_symbolic_agent.py   # skipped automatically when torch is not installed
```

LoRA adaptation (training stages 3-4) needs `peft`, now in the `models` extra (conda: `pip install peft`).
`python -m pytest tests/test_slm_policy.py` tests the SLM policy with a tiny random Qwen3, so nothing is downloaded.

## Public API

Import from the module (`from models.symbolic_agent import SymbolicAgent`), so `import models` does not load PyTorch.
Each module lists its public names in `__all__`.

| module | public names |
|---|---|
| `models.symbolic_agent` | `SymbolicAgent`, `SymbolicPolicy`, `SymbolicHistory`, `stack_demo_steps`, `MissionTokenizer`, `NO_ACTION`, `save_model`, `load_model`, `load_checkpoint_info` |
| `models.slm_policy` | `SLMPolicy` (`forward(obs, memory) -> PolicyOutput`, `initial_memory(batch_size)`), `SLMAgentPolicy`, `Observations`, `observations_from_env`, `PolicyOutput`, `MemoryState`, `parameter_groups`, `GROUPS`, `save_model`, `load_model`, `load_checkpoint_info`, `Backbone`, `ReadoutBackbone`, `SLMBackbone` |

Every model reaches the environment the same way: a policy object, `policy(obs) -> action index`, with `reset(env)`
clearing any per-episode state, handed to `babytext.run_episode` / `babytext.evaluate`:

```python
from babytext import evaluate
from models.symbolic_agent import SymbolicAgent, SymbolicPolicy

policy = SymbolicPolicy(trained_agent)          # trained SymbolicAgent
result = evaluate(policy, "goto", "val", n_episodes=500)
```

## B3: symbolic agent (`symbolic_agent.py`)

Sees the same information as the GLAM prompt, without language pretraining: the last N = `config.ENV.prompt_num_obs`
egocentric views stacked along the channel axis, the N - 1 actions between them, and the mission read word by word
with a fixed vocabulary (so invented names in `test_nonce` become unknown tokens). It deliberately does not get
`obs["direction"]`, because the prompt never gives a compass direction.

Per-cell embeddings → 3x3 convolutions, each modulated by the mission through FiLM (as in the BabyAI baselines) →
MLP with the mission and previous-action embeddings → 6 action logits. Sizes are in `config.SYMBOLIC`.
`stack_demo_steps` builds training inputs from a demo file and `SymbolicHistory` builds the same inputs during a
rollout. Architecture figure: [report/figures/symbolic_agent_architecture.svg](../report/figures/symbolic_agent_architecture.svg).
Trained by `python -m training.train --model b3` ([training README](../training/README.md#b3-trainer)).

**Checkpoints.** `save_model(model, path, **extra)` stores the config, the input window (`num_frames`, `view_size`),
the full `state_dict` (including BatchNorm running statistics) and `extra` (epoch, val scores).
`load_model(path)` rebuilds the model from the saved config and window, not from `config.SYMBOLIC` / `config.ENV`,
and returns it on the CPU in eval mode; `load_checkpoint_info(path)` returns `extra`.

```python
from models.symbolic_agent import SymbolicPolicy, load_model

policy = SymbolicPolicy(load_model("outputs/runs/B3/easy/seed0/best.pt"), device="cuda")
```

## The SLM policy (`slm_policy/`)

Design, feasibility assessment and the comparison with a plain GRU: [report/architecture/architecture_feasibility.pdf](../report/architecture/architecture_feasibility.pdf).
Implemented as described below, with two changes. (1) The memory is a *write-sparse* slot memory: the gate starts
biased towards "do not write", a write penalty is added to the loss, `write_mode="hard"` gives 0/1 writes, and each slot
records its age and fill level. (2) The mission is encoded by a word GRU for the memory and the attention, so the SLM is
never inside the recurrence and runs once on all steps of a training sequence. The thinking mode is not implemented.

| file | contents |
|---|---|
| `inputs.py` | `Observations`, the (B, T) input block; `observations_from_env` |
| `encoders.py` | `TimeEncoding`, `ViewEncoder` (spatial encoder), `MissionEncoder` |
| `memory.py` | `MemoryState`, `SlotMemory` (write gate + slot address) |
| `attention.py` | `SpaceTimeAttention`: (K+1) x 49 weights per head, H read-outs |
| `backbones.py` | `Backbone` interface, `ReadoutBackbone` (stage 1), `SLMBackbone` (soft tokens, LoRA, layer truncation) |
| `policy.py` | `SLMPolicy`, `parameter_groups`, `save_model` / `load_model` |
| `agent.py` | `SLMAgentPolicy` for `babytext.evaluate` |

One forward pass per environment step *t*:

```
obs["image"] (7x7x3) ─► cell embedding ─► spatial encoder (3x3 conv layers) ─► F_t  (49 cells x d)
step t ─► time encoding e(t) ──────────────────────────────────────────────────────┘ added to F_t

memory M_{t-1}  (K slots x 49 cells x d) ─┐
                                          ├─► space-time tensor S_t = [F_t ; M_{t-1}]  ((K+1) x 49 x d)
F_t ──────────────────────────────────────┘
mission embedding q ─► space-time attention: A_t = softmax(score(S_t, q))  ((K+1) x 49, per read head)
                      ─► H read-outs r_h = Σ_{τ,p} A_t^h[τ,p] · S_t[τ,p]     (H vectors of size d)
                      ─► adapter (linear d -> SLM width) ─► H soft tokens

SLM input:  <instruction + mission text> <soft token 1> ... <soft token H> "Action:"
SLM low-level layers (LoRA from stage 3) ─► SLM high-level layers (frozen) ─► last-token hidden state
            ─► action head (6 logits) and value head (1 scalar, for PPO)

memory gate (after acting):
   g_t = σ(W_g [pool(F_t), pool(M_{t-1}), e(t)] + b_g)          "save this observation?"   (GRU update gate)
   w_t = softmax over K slots                                     which slot to overwrite
   M_t[k] = (1 - g_t·w_t[k]) · M_{t-1}[k] + g_t·w_t[k] · F_t
```

| part | role | from the design brief |
|---|---|---|
| cell embedding + spatial encoder | turns MiniGrid's (object, colour, state) codes into features per cell; 3x3 convolutions share weights over space | "encoder that encodes the observation input to encode for space" |
| time encoding `e(t)` | learned embedding of the step index (0 to the time limit), added to the features and fed to the gate | "the input observation will be time encoded" |
| gated memory | fixed number K of slots; the gate decides whether the current view is saved and which slot it replaces | "GRU component that learns which feature observation is worth storing in its fixed length memory array" |
| space-time attention `A_t` | a 2-D matrix of weights over (time slot, grid cell); the scorer is a small conv net shared across cells and slots, conditioned on the mission | "space-time matrix with weights corresponding to the attention to that point in spacetime", "similar to a convolutional neural network" *(interpretation: rows = memory slots + current view, columns = the 49 cells)* |
| adapter + soft tokens | projects the read-outs into the SLM's embedding space | "the SLM to take in these new inputs" |
| SLM | an instruction-tuned model with a thinking mode, e.g. Qwen3-0.6B (28 layers); reads the mission as text and the scene as soft tokens | "a SLM with the capability to think" |
| action / value heads | linear layers on the SLM's last-token hidden state | one of six actions in a single pass; value estimate for PPO |

### Hyperparameters

In `config.SLM` (`SLMPolicyConfig`). The main ones:

| name | meaning | starting value |
|---|---|---|
| `memory_slots` (K) | number of past views kept | 8 |
| `feature_dim` (d) | features per cell | 64 |
| `num_conv_layers`, kernel | depth and kernel of the spatial encoder | 3, 3x3 |
| `use_direction` | add an embedding of `obs["direction"]` to every cell (off: B3 gets no heading, so no compared model does; with `false` there is no `direction_emb` at all) | False |
| `read_heads` (H) | number of soft tokens given to the SLM | 4 |
| `write_gate_bias`, `write_mode` | initial gate bias ("rarely write") and `soft` / `hard` (straight-through) writes | -2.0, `soft` |
| `backbone` | `readout` (MLP, stage 1) or `slm` (stages 2-4) | `readout` |
| `slm_name`, `slm_dtype` | Hugging Face model id (or local folder) and precision | `Qwen/Qwen3-0.6B`, `bfloat16` |
| `slm_pretrained`, `slm_init_seed` | `false` is the **M2-randlm** control: `slm_name`'s architecture and tokenizer with random weights drawn with `slm_init_seed` | True, 0 |
| `low_level_layers` | how many of the first SLM layers count as low-level (tuned from stage 3) | 10 of 28 |
| `lora_rank`, `lora_targets` | LoRA on the low-level layers | 8, attention q/v projections |
| `readout_layer` | read the action from this layer's output; higher layers are dropped (None = all) | None |
| `gradient_checkpointing` | recompute SLM activations in the backward pass (needed to fit an 8 GB GPU) | True |

**Checkpoints** (`save_model` / `load_model`) store the config and every weight except the frozen LM, which is reloaded
from the hub. A random LM (`slm_pretrained=False`) is not stored either: `load_model` regenerates it from
`slm_init_seed` (drawn on the CPU from a forked RNG, so the global seed neither affects it nor is changed by it) and
compares it with a SHA-256 fingerprint saved in the checkpoint, raising `RuntimeError` if the weights came out
differently (e.g. other torch / transformers versions). `load_model(path, **overrides)` uses the saved config plus
`overrides` and never reads `config.SLM`.

### Design choices

- **Scene through `image`, goal through text.** The spatial encoder reads the symbolic 7x7 grid because it already is a
  spatial grid that a convolution can exploit directly; the text descriptions contain the same information but would
  have to be parsed back into positions. The mission stays text, so the SLM's language knowledge maps "the red ball"
  to what the encoder sees.
- **Soft tokens instead of the GLAM prompt.** The GLAM prompt (and B3) keep a fixed window of 3 observations and 2
  actions. Here history lives in the learned memory, so the SLM sees a short, fixed-size input (mission + H tokens) at
  every step, however long the episode is.
- **Fixed-size memory with a write gate.** A GRU-style update gate decides how much of the current view to store and a
  softmax address picks the slot, so memory and compute stay constant over the episode.
- **Low-level SLM layers tuned, high-level frozen.** The soft tokens are a new input "modality", so the layers that first
  process them are adapted with LoRA. The high-level layers stay frozen to keep the SLM's language ability, which the
  nonce / held-out tests rely on.
- **No heading input.** B3 and the GLAM prompt never see `obs["direction"]`, so the SLM policy does not either
  (`use_direction=False`): every compared model gets the same observation, and differences between them come from
  the models alone. Its memory stores egocentric views from different headings anyway (see Limitations).

### Limitations

- **Egocentric memory.** Each slot stores a view in the agent's own frame. After the agent turns or moves, cell *p* in an
  old slot is no longer the same place as cell *p* now. Without a heading input, the model must infer how views relate
  from the views themselves and the time encodings.
  An allocentric (room-aligned) memory would avoid it but needs the agent's position, which `babytext` does not expose yet.
- **No thinking mode.** Thinking tokens would multiply the cost of each of up to 64 / 128 steps, and no demonstration
  contains reasoning to imitate, so the thinking mode is not used (proposal §14).
- **`test_nonce` mostly tests the mission.** Renaming does not change `image`, so on `test_nonce` only the mission text
  differs. Mapping invented words to grid codes is not learnable from the training data, so a large drop is expected.
- **No caching.** Hidden states depend on the learned memory and on tuned SLM layers, so they cannot be pre-computed
  as the submitted proposal planned (proposal §14).

### Open questions

These come from the original design brief and are decided here provisionally:

1. *"Reward over space time in which the model will learn the weights for."* Taken to mean that the delayed (episode)
   reward is propagated back through time and is what trains the space-time weights and the memory gate. If the intent
   is for the model to learn the **reward weights themselves**, that needs an outer objective (e.g. validation
   success): a policy that sets its own reward weights by gradient can simply make its reward large.
2. *"Encourage more memorisation at the start and being more picky … as the training progresses."* If this means *as the
   episode progresses*, the time encoding fed to the gate is enough, optionally with the gate bias initialised to
   decrease with *t*. If it means *over training epochs*, it needs a schedule, e.g. a write-rate penalty that grows
   during training; the per-step time encoding alone cannot do it.
3. *"The spacetime weight matrix will be a 2-dimensional matrix."* Taken as (time slot × grid cell), computed per step by a
   shared conv scorer, rather than one static learned matrix, which could not follow where the goal currently is.
