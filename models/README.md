# models

The policies trained in this project. Depends on `babytext` and `config`; must never import `training`, so a trained
model can be loaded and evaluated without the training code.

| module | status | what |
|---|---|---|
| [`symbolic_agent.py`](symbolic_agent.py) | done (untrained) | **B3**, the non-LLM baseline: frame-stacked CNN + FiLM on the symbolic grid |
| `slm_policy.py` | specified here, not implemented | **the main model**: SLM with spatial encoder, gated memory and space-time attention |
| `prompted.py` | not implemented | **B1 / B2**: the same SLM prompted with the GLAM prompt (generate-and-map, log-probability scoring) |

Where the original design brief (kept in the root README's [Training](../README.md#training) section) was ambiguous,
the choice made here is marked *(interpretation)* and listed under [Open questions](#open-questions).

## Install

Same shared environment as the rest of the repository ([root README](../README.md#setup)), plus the model dependencies:

```bash
pip install -e ".[models]"     # venv: torch + transformers;  conda: torch already comes from environment.yml
python -m pytest tests/test_symbolic_agent.py   # skipped automatically when torch is not installed
```

LoRA adaptation (training stage 3) will also need `peft`; add it to the `models` extra in `pyproject.toml` (and to
`environment.yml`) when that code lands.

## Public API

Import from the module (`from models.symbolic_agent import SymbolicAgent`), so `import models` does not load PyTorch.
Each module lists its public names in `__all__`.

| module | public names |
|---|---|
| `models.symbolic_agent` | `SymbolicAgent`, `SymbolicPolicy`, `SymbolicHistory`, `stack_demo_steps`, `MissionTokenizer`, `NO_ACTION` |
| `models.slm_policy` (planned) | `SLMPolicy` (`forward(step_inputs, memory) -> (action_logits, value, new_memory)`), `initial_memory(batch_size)`, `parameter_groups(model)`, `SLMAgentPolicy`, `save_model`, `load_model` |

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

## The SLM policy (`slm_policy.py`, planned)

One forward pass per environment step *t*:

```
obs["image"] (7x7x3) ─► cell embedding ─► spatial encoder (3x3 conv layers) ─► F_t  (49 cells x d)
obs["direction"], step t ─► direction embedding + time encoding e(t) ─────────────┘ added to F_t

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

To be added to `config.py` as an `SLM` section next to `SYMBOLIC`, with these starting values:

| name | meaning | starting value |
|---|---|---|
| `memory_slots` (K) | number of past views kept | 8 |
| `feature_dim` (d) | features per cell | 64 |
| `num_conv_layers`, kernel | depth and kernel of the spatial encoder | 3, 3x3 |
| `read_heads` (H) | number of soft tokens given to the SLM | 4 |
| `time_encoding` | `learned` or `sinusoidal` | `learned` |
| `backbone` | Hugging Face model id (replaces `config.TRAIN.backbone`, which still names the proposal's Qwen2.5) | `Qwen/Qwen3-0.6B` |
| `low_level_layers` | how many of the first SLM layers count as low-level (tuned from stage 3) | 10 of 28 |
| `lora_rank`, `lora_targets` | LoRA on the low-level layers | 8, attention q/v projections |
| `action_readout` | `head` (linear head, one pass) or `score` (log-probability of the six action strings) | `head` |
| `thinking_budget` | max thinking tokens generated before the action; 0 disables thinking | 0 during training |

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
  process them are adapted with LoRA. The high-level layers stay frozen to keep the SLM's language and reasoning
  ability, which the thinking mode and the nonce / held-out tests rely on.
- **Direction as an input.** Unlike B3, the SLM policy gets `obs["direction"]`, because its memory stores egocentric views
  from different headings (see Limitations). Comparing it with B3 therefore also changes this input; an ablation
  without direction makes the comparison exact.

### Limitations

- **Egocentric memory.** Each slot stores a view in the agent's own frame. After the agent turns or moves, cell *p* in an
  old slot is no longer the same place as cell *p* now; the model must learn this from direction and time encodings.
  An allocentric (room-aligned) memory would avoid it but needs the agent's position, which `babytext` does not expose yet.
- **Thinking is expensive and untrained.** Thinking tokens multiply the cost of each of up to 64 / 128 steps, and no
  demonstration contains reasoning, so thinking can only be learned from reward (stage 4) or used zero-shot.
- **`test_nonce` mostly tests the mission.** Renaming does not change `image`, so on `test_nonce` only the mission text
  differs. Mapping invented words to grid codes is not learnable from the training data, so a large drop is expected.
- **No caching.** Hidden states depend on the learned memory and on tuned SLM layers, so they cannot be pre-computed
  as the proposal planned.

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
