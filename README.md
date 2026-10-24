# very deep neurotica

SC4001 group project: *Reading Actions from Frozen Language Models* (see [project_proposal.md](project_proposal.md)).

This repository currently contains the **environment side of O1**: a BabyAI-Text-style text gridworld,
expert demonstrations, fixed train/val/test splits, the three shifted test sets, and visualisation tools.
LLM feature extraction and all models are not included yet.

## Why not the `babyai-text` package?

GLAM's BabyAI-Text ships as a bundle of forked `babyai` + `gym-minigrid` repos on the old `gym` API,
which is awkward to install on current Python/Windows. (Lamorel, GLAM's LLM serving library, is only needed for
their PPO fine-tuning, not for the environment.) Following the fallback in proposal §6.1 we instead
wrap Farama's maintained [`minigrid`](https://minigrid.farama.org/), which already contains the BabyAI levels and
the BabyAI expert bot, with a small text layer that reproduces BabyAI-Text's output **exactly**. The tests check our
descriptions and prompts against code transcribed from GLAM's repository on every visited state.

## Setup

```bash
conda env create -f environment.yml   # creates the "neurons" env
conda activate neurons
pip install -e .                      # installs the babytext package (editable)
python -m pytest                      # 29 tests, ~40 s
```

## Quick start

```python
from babytext import make_env, get_seeds, ExpertPolicy, run_episode, evaluate

env = make_env("goto", "train")                     # task in {goto, pickup}
obs, info = env.reset(seed=get_seeds("goto", "train", 1)[0])
print(obs["prompt"])                                # GLAM prompt for the LLM conditions
obs["image"], obs["direction"]                      # symbolic view for the B3 baseline
obs, reward, terminated, truncated, info = env.step(2)   # "go forward"

# Any callable obs -> action index is a policy:
result = evaluate(my_policy, "goto", "test_dist16", n_episodes=500)
result["success_rate"], result["mean_reward"], result["episodes"]
```

## The environment

| | |
|---|---|
| Tasks | `goto` ("go to the red ball") and `pickup` ("pick up the red ball") |
| Layout | one 8x8 room (6x6 interior), 1 goal object + 8 distractors (keys, balls, boxes in 6 colours); every object is reachable |
| Actions | `turn left, turn right, go forward, pick up, drop, toggle` (indices 0-5, `babytext.ACTION_NAMES`) |
| Observation | 7x7 egocentric partial view (walls block sight) |
| Reward | `1 - 0.9 * steps / max_steps` on success, else 0; `max_steps = 64` |
| Expert | MiniGrid's `BabyAIBot`, 100% success on every eval set |

Each observation dict carries:

- `image` (7, 7, 3) uint8 and `direction`: MiniGrid's symbolic view;
- `mission`: the goal string;
- `descriptions`: BabyAI-Text descriptions, e.g. `You see a red ball 1 step left and 2 steps forward`, `You see a wall 3 steps right`, `You carry a grey key`;
- `prompt`: the GLAM prompt (goal, last 3 observations, last 2 actions), identical for every LLM-based condition.

The text is a deterministic function of `image`, so the text and symbolic views contain the same information.
`info` additionally gives `goal` (type, colour) and `goal_positions`.

Example prompt (`goto`, train seed 0, after `turn left`, `go forward`; Observation 0 shortened):

```
Possible action of the agent: turn left, turn right, go forward, pick up, drop, toggle
 Goal of the agent: go to a grey key
 Observation 0: You see a wall 2 steps left, You see a grey key 1 step left and 5 steps forward, ..., You see a grey key 2 steps right and 3 steps forward,
 Action 0: turn left
 Observation 1: You see a wall 2 steps forward, You see a wall 1 step left, You see a red ball 2 steps right, You see a red box 3 steps right and 1 step forward, You see a purple key 3 steps right,
 Action 1: go forward
 Observation 2: You see a wall 1 step forward, You see a wall 1 step left, You see a red box 3 steps right,
 Action 2:
```

The missions use BabyAI's articles: "a grey key" when several objects match (any of them counts), "the" when exactly one does.

### Eval sets and seeds

| eval set | seeds | distractors | goal objects | names |
|---|---|---|---|---|
| `train` | train | 8 | all except held-out | real |
| `val` | val | 8 | all except held-out | real |
| `test` | test | 8 | all except held-out | real |
| `test_dist16` | test | **16** | all except held-out | real |
| `test_nonce` | test | 8 | all except held-out | **invented** (`red`→`dax`, `ball`→`tiv`, ...) |
| `test_heldout` | test | 8 | **only held-out** | real |

- Held-out goals follow GLAM: **yellow box, red key, green ball**. They still appear as distractors during training; only the goal is held out.
- `get_seeds(task, eval_set, n)` returns the first `n` seeds of a fixed, disjoint range per (split, task).
  `test_nonce` uses the same seeds as `test` and gives the **identical layout and expert trajectory**, with only the words changed.
  `test_heldout` with the same seed has the same object positions with a different goal identity.
- Use `val` for all model selection and touch `test*` once at the end (proposal §6.5).

Reference results (`scripts/reference_baselines.py`, 500 episodes per cell):

| task | eval set | expert success | random success |
|---|---|---|---|
| goto | train / val / test | 1.000 | 0.32 / 0.31 / 0.32 |
| goto | dist16 / nonce / heldout | 1.000 | 0.35 / 0.32 / 0.31 |
| pickup | train / val / test | 1.000 | 0.16 / 0.15 / 0.15 |
| pickup | dist16 / nonce / heldout | 1.000 | 0.14 / 0.15 / 0.13 |

Note the high random floor on `goto` (walking around a cluttered room often ends up facing the goal); report it next to every result.

## Scripts

```bash
python scripts/collect_demos.py                  # expert demos -> data/demos/ (2000 train + 200 val per task, ~15 s)
python scripts/dataset_stats.py                  # charts of demos and eval sets -> outputs/figures/
python scripts/reference_baselines.py            # expert / random success table -> outputs/results/
python scripts/visualise_eval_sets.py            # one figure comparing all eval sets -> outputs/figures/
python scripts/visualise_episode.py --task pickup --eval-set test_dist16 --seed-index 3
python scripts/visualise_episode.py --demos data/demos/goto_train --episode 10 --show-prompt
python scripts/play.py --task goto --eval-set test_nonce   # play by hand; text printed in the terminal
```

`visualise_episode.py` writes a GIF (room with field of view and goal highlighted | agent's view | text) and a PNG
filmstrip. For failure analysis, render any evaluated episode with
`babytext.viz.replay(task, eval_set, episode.seed, episode.actions)` followed by `save_gif` / `save_strip`.

### Demonstration files

`data/demos/<task>_<split>.jsonl` holds one episode per line (seed, mission, goal, outcome, and per-step
`descriptions`, `prompt`, `action`). `data/demos/<task>_<split>.npz` holds flat per-step arrays
(`image`, `direction`, `action`, `episode`, `t`) for the symbolic baseline. Row *i* of the npz is step *i* of
`babytext.data.load_steps(prefix)`, so cached hidden states can be aligned to both. Episodes use the first N seeds of
the split, so the first *k* episodes are a valid smaller training set for demonstration-count curves.

## Layout

```
babytext/
  constants.py   actions, colours, held-out goals, invented words
  levels.py      LocalLevel: single-room goto / pickup on MiniGrid's BabyAI RoomGridLevel
  text.py        describe(): BabyAI-Text descriptions; Renamer for invented words
  prompt.py      GLAM prompt template and history
  wrapper.py     TextEnv gymnasium wrapper (text + symbolic observations, 6 actions)
  splits.py      eval sets, seed ranges, make_env()
  policies.py    ExpertPolicy (BabyAIBot), RandomPolicy
  rollout.py     run_episode(), evaluate()
  data.py        save / load demonstrations
  viz.py         frame rendering, GIFs, filmstrips
scripts/         collection, statistics, visualisation, manual play
tests/           unit tests; glam_reference.py holds code transcribed from GLAM for equivalence checks
```

## Differences from GLAM's BabyAI-Text

- Only the single-room `goto` and `pickup` tasks (GLAM's MixedTrainLocal also mixes in open, put-next and sequences).
- The goal (type, colour) is sampled first from the allowed pool and then the distractors, instead of picking a random
  placed object and rejecting forbidden missions. The goal distribution is the same (uniform over allowed pairs).
- Objects are placed one at a time and an object is re-placed if it would make anything unreachable. BabyAI instead rejects
  whole layouts, which needs ~1300 attempts per layout with 16 distractors (~0.5 s per reset vs ~12 ms here). With 8
  distractors the two are practically identical.
- Doors (unused here) would be described as "an open ... door" rather than GLAM's "a open ... door".

## Acknowledgements

Text description format and prompt template are ported from GLAM (Carta et al., 2023,
[flowersteam/Grounding_LLMs_with_online_RL](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)).
Environment dynamics, BabyAI levels and the expert bot are from Farama
[MiniGrid](https://github.com/Farama-Foundation/Minigrid) (Chevalier-Boisvert et al., 2019, 2023).
