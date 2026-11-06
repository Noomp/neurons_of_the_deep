# models

The policies compared in this project. Depends on `babytext` and `config`; must never import `training`, so a trained
model can be loaded and evaluated without the training code.

| module | status | what |
|---|---|---|
| [`action_head.py`](action_head.py) | done | **P**, the proposed method: a small head on one layer's hidden states of the frozen LLM |
| [`prompted.py`](prompted.py) | done | **B1 / B2**, the language-grounded baselines: the same frozen LLM generating (B1) or scoring (B2) the action names, zero-shot |
| [`symbolic_agent.py`](symbolic_agent.py) | done | **B3**, the non-language baseline: frame-stacked CNN + FiLM on the symbolic grid |
| [`frozen_lm.py`](frozen_lm.py) | done | the frozen LLM shared by P, B1 and B2: loading, truncation, hidden states of prompts |

The design that was here before 4 Nov (an SLM reading a learned spatio-temporal memory, `slm_policy/`) is kept on git
branch `archive/slm-memory-policy`.

## Install

Same shared environment as the rest of the repository ([root README](../README.md#setup)), plus the model dependencies:

```bash
pip install -e ".[models]"     # venv: torch + transformers;  conda: torch already comes from environment.yml
python -m pytest tests          # uses a tiny random Qwen3, so nothing is downloaded
```

The first use of the real LLM downloads `Qwen/Qwen3-0.6B` (about 1.2 GB) and, for B1, `sentence-transformers/all-MiniLM-L6-v2`
(about 90 MB) into the Hugging Face cache.

## Public API

Import from the module (`from models.action_head import ActionHead`), so `import models` does not load PyTorch.
Each module lists its public names in `__all__`.

| module | public names |
|---|---|
| `models.action_head` | `ActionHead`, `HeadPolicy`, `pool`, `POOLINGS`, `save_model`, `load_model`, `load_policy`, `load_checkpoint_info` |
| `models.prompted` | `GenerationPolicy` (B1), `ScoringPolicy` (B2), `SentenceEmbedder` |
| `models.symbolic_agent` | `SymbolicAgent`, `SymbolicPolicy`, `SymbolicHistory`, `stack_demo_steps`, `MissionTokenizer`, `NO_ACTION`, `save_model`, `load_model`, `load_checkpoint_info` |
| `models.frozen_lm` | `FrozenLM` (`load`, `run`, `tokenize`), `PromptStates` (`last`, `mean`, `sequences`), `load_causal_lm` |

Every model reaches the environment the same way: a policy object, `policy(obs) -> action index`, with `reset(env)`
clearing any per-episode state, handed to `babytext.run_episode` / `babytext.evaluate`:

```python
from babytext import evaluate
from models.action_head import load_policy
from models.prompted import ScoringPolicy
from models.symbolic_agent import SymbolicPolicy, load_model

evaluate(load_policy("outputs/runs/P-L16-mean-h1/easy/seed0/best.pt", device="cuda"), "goto", "val", 500)
evaluate(ScoringPolicy.load(device="cuda"), "goto", "val", 500)                          # B2
evaluate(SymbolicPolicy(load_model("outputs/runs/B3/easy/seed0/best.pt")), "goto", "val", 500)
```

## The frozen LLM (`frozen_lm.py`)

Qwen3-0.6B (`config.LLM`: 28 layers, width 1024, bfloat16), never trained. `FrozenLM` is the base model without its LM
head (P); `load_causal_lm` loads the full model with its LM head (B1, B2).

- **Layer numbering.** Layer *l* (1-28) is the output of decoder layer *l*. Layer 28 includes the final norm (it is what
  the LM head reads); layers below do not.
- **Truncation.** `FrozenLM.load(cfg, num_layers=l)` keeps only layers 1-*l* (and drops the final norm unless *l* = 28),
  so its last layer is the same vector as layer *l* of the full model and the layers above never run. A head trained on
  cached states of the full model therefore acts identically on the truncated one (tested), and reading a lower layer
  is cheaper.
- **Batching.** Prompts are left-padded with explicit position ids, so a prompt's hidden states do not depend on what it
  is batched with (tested).

## P: the action head (`action_head.py`)

```
obs["prompt"] ─► frozen LLM, layers 1..l ─► token vectors (L x 1024) at layer l
             ─► pooling ─► LayerNorm ─► MLP (1-2 hidden layers of 256, ReLU) ─► 6 action logits
```

| pooling | the one vector the MLP reads | cost |
|---|---|---|
| `last` | the last token's vector. The LLM reads left to right, so only the last token has seen the whole prompt; the prompt ends in `Action N: `, where the LLM would predict the action and where B2 reads its scores | cached |
| `mean` | the mean of all token vectors (padding excluded) | cached |
| `attention` | a learned weighted sum of the token vectors: a key projection (1024 -> 64) scored against one learned query, softmax over tokens. The query starts at zero, so the head starts as mean pooling over LayerNorm-ed tokens | needs every token's vector: cached only for chosen layers |

Settings are `config.HEAD` (`layer`, `pooling`, `hidden_layers`, `hidden_dim`, `attention_dim`; `sweep_layers` are the
layers the cache and the run plan cover). With 1024-wide hidden states, a head has about 0.26M (1 hidden layer) to
0.4M (2 hidden layers, attention) trainable parameters.
Architecture figure: [report/figures/action_head_architecture.svg](../report/figures/action_head_architecture.svg)
(drawn by `python report/figures/action_head_architecture.py`, which reads the parameter counts from the code).

`HeadPolicy(head, lm)` is the babytext policy: one truncated LLM pass on `obs["prompt"]` per decision, greedy. It keeps
no state, because the GLAM prompt already carries the last three observations and two actions. `last_logits` holds the
latest decision's six logits (recorded by the evaluation).

**Checkpoints.** `save_model(head, path, llm, **extra)` stores the head's config, input size and weights, the LLM config
it reads (not the LLM's weights) and `extra` (epoch, val scores). `load_model(path)` returns `(head, llm_config)`;
`load_policy(path, device)` returns a ready `HeadPolicy` with the LLM truncated to the head's layer.

## B1 / B2: language-grounded action selection (`prompted.py`)

Both read the same GLAM prompt as P and choose through the meaning of the six action names, with no training.

- **B2, `ScoringPolicy`** (GLAM; TWOSOME): the log-probability of each action name as the continuation of the prompt,
  divided by its number of tokens (`config.PROMPTED.length_normalise`). The prompt runs once; its key/value cache is
  repeated for the six names, so a decision costs one prefill plus a few tokens (tested equal to scoring six full
  sequences). `last_scores` holds the latest six scores.
- **B1, `GenerationPolicy`** (Huang et al., 2022): greedy generation of `config.PROMPTED.max_new_tokens` tokens. If the
  first line is exactly an action name, that action; otherwise the action whose name is nearest by cosine similarity of
  `sentence-transformers/all-MiniLM-L6-v2` embeddings (`SentenceEmbedder`). `stats` counts decisions and invalid
  outputs (not exactly an action name) for the invalid-output rate; `last_text` and `last_valid` describe the latest
  decision.

The prompt ends in `"Action N: "` (with a space); action names are tokenized on their own and appended to the prompt's
tokens, identically for every action. The thinking mode is not used: the prompt is raw text, not a chat template.

## B3: symbolic agent (`symbolic_agent.py`)

Sees the same information as the GLAM prompt, without language pretraining: the last N = `config.ENV.prompt_num_obs`
egocentric views stacked along the channel axis, the N - 1 actions between them, and the mission read word by word
with a fixed vocabulary (so invented names in `test_nonce` become unknown tokens). It deliberately does not get
`obs["direction"]`, because the prompt never gives a compass direction.

Per-cell embeddings → 3x3 convolutions, each modulated by the mission through FiLM (as in the BabyAI baselines) →
MLP with the mission and previous-action embeddings → 6 action logits. Sizes are in `config.SYMBOLIC`.
`stack_demo_steps` builds training inputs from a demo file and `SymbolicHistory` builds the same inputs during a
rollout; `SymbolicPolicy.last_logits` holds the latest decision's six logits. Architecture figure: [report/figures/symbolic_agent_architecture.svg](../report/figures/symbolic_agent_architecture.svg).
Trained by `python -m training.train --model b3` ([training README](../training/README.md)).

**Checkpoints.** `save_model(model, path, **extra)` stores the config, the input window (`num_frames`, `view_size`),
the full `state_dict` (including BatchNorm running statistics) and `extra` (epoch, val scores).
`load_model(path)` rebuilds the model from the saved config and window, not from `config.SYMBOLIC` / `config.ENV`,
and returns it on the CPU in eval mode; `load_checkpoint_info(path)` returns `extra`.

## Design choices

- **The same prompt for every LLM condition.** P, B1 and B2 read exactly `obs["prompt"]`, so they differ only in how the
  action is read out of the same frozen LLM. B3 reads the same window (3 views, 2 actions, goal) symbolically.
- **Frozen LLM, cached features.** Nothing in the LLM is trained, so its hidden states on the demonstrations are fixed
  and computed once; the layer sweep costs seconds per head. LayerNorm on the head's input absorbs the very different
  scales of different layers.
- **Hidden states in bfloat16.** The cache stores the LLM's own precision exactly; float16 could overflow on the largest
  activations.

## Limitations

- **Zero-shot baselines vs a trained head.** B1 and B2 are never trained on the task while P is, so P vs B2 compares a
  trained head with untrained language scoring, not two equally trained methods. GLAM trained scoring with PPO, which
  is out of scope.
- **`test_nonce`.** Invented words are unknown to the LLM as well, so neither P nor B1/B2 can know what they refer to;
  the question (H3) is how much each degrades.
- **No thinking mode, no fine-tuning, no RL** (proposal §4).
