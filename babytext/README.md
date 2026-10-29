# babytext

The environment section of the project: a BabyAI-Text-style text gridworld on Farama MiniGrid, with fixed
train/val/test splits, three shifted test sets, the expert and random policies, rollout / evaluation, the
demonstration file format and visualisation tools. It imports nothing from `models` or `training`.

Status: **done**. Collecting demonstrations is a training step and lives in [`training/`](../training/README.md).

## Install

The virtual environment (or conda env) is shared by the whole repository; see the [root README](../README.md#setup).
The environment needs only the base dependencies, so it can be installed without PyTorch:

```bash
python -m venv .venv && source .venv/bin/activate    # or: conda env create -f environment.yml && conda activate neurons
pip install -r requirements.txt                       # venv only: pinned versions
pip install -e .                                      # installs babytext (with config, models, training; editable)
python -m pytest babytext/tests                       # environment tests
```

## Public API

Import from `babytext` (the names in `__all__` of [`__init__.py`](__init__.py)), plus three modules other code may
import directly: `babytext.data` (demonstration files), `babytext.viz` (rendering; it switches matplotlib to `Agg`)
and `babytext.constants` (fixed vocabulary). `levels`, `wrapper`, `text` and `splits` internals, and names starting with
`_`, are implementation details.

| purpose | names |
|---|---|
| environments and seeds | `make_env(task, eval_set, **overrides)`, `get_seeds(task, eval_set, n)`, `TASKS`, `EVAL_SETS` |
| actions | `ACTION_NAMES`, `ACTION_TO_IDX`, `NUM_ACTIONS` |
| policies and evaluation | `ExpertPolicy`, `RandomPolicy`, `DisappearedBoxError`, `run_episode`, `evaluate`, `Episode`, `Step` |
| text helpers | `describe`, `glam_prompt`, `PromptHistory`, `Renamer` |
| demonstration files (`babytext.data`) | `save_demos`, `load_episodes`, `load_steps`, `load_symbolic` |
| rendering (`babytext.viz`) | `replay`, `capture`, `draw_frame`, `save_gif`, `save_strip`, `outline_goals`, `Frame` |

A policy is any callable `policy(obs) -> action index`. It may define `reset(env)`, which `run_episode` calls after
every `env.reset`; policies with memory use it to clear their state and must not read anything else from `env` there.

## Quick start

```python
from babytext import make_env, get_seeds, ExpertPolicy, run_episode, evaluate

env = make_env("goto", "train")                     # task in babytext.TASKS
obs, info = env.reset(seed=get_seeds("goto", "train", 1)[0])
obs["image"], obs["direction"]                      # symbolic view: input to the SLM policy's encoder and to B3
obs["mission"]                                      # goal text: read by the SLM
print(obs["prompt"])                                # GLAM prompt: input to the prompted-SLM baselines B1 / B2
obs, reward, terminated, truncated, info = env.step(2)   # "go forward"

# Any callable obs -> action index is a policy:
result = evaluate(my_policy, "goto", "test_dist16", n_episodes=500)
result["success_rate"], result["mean_reward"], result["episodes"]
```

## The environment

| | |
|---|---|
| Tasks | four tasks in two tiers, see below |
| Layout | one 8x8 room (6x6 interior), the 1 or 2 objects named in the instruction + 8 distractors (keys, balls, boxes in 6 colours); every object is reachable |
| Actions | `turn left, turn right, go forward, pick up, drop, toggle` (indices 0-5, `babytext.ACTION_NAMES`) |
| Observation | 7x7 egocentric partial view (walls block sight) |
| Reward | `1 - 0.9 * steps / max_steps` on success, else 0; `max_steps` = 64 (easy tier) or 128 (hard tier) |
| Expert | MiniGrid's `BabyAIBot`, 100% success on every eval set |

### Tasks

| tier | task | example mission | expert steps | random success |
|---|---|---|---|---|
| easy | `goto` | go to the red ball | 5.5 | 0.32 |
| easy | `pickup` | pick up the red ball | 6.4 | 0.16 |
| hard | `putnext` | put the red ball next to the blue key | 11.9 | 0.02 |
| hard | `pickup_then_goto` | pick up the red ball, then go to the blue key | 11.3 | 0.08 |

(Expert steps and random success on `train`, 500 episodes.) The easy tier matches GLAM's single-object tasks. In the
hard tier the agent has to handle two objects, carry one while navigating, and (for `putnext`) use `drop`.
Episodes are about twice as long. Both hard tasks come from GLAM's training mix (`Level_MixedTrainLocal`). The two
objects in an instruction always differ in type or colour. `babytext.constants.TASK_TIERS` lists the tiers.

The task stays observable from the prompt: once the first object is picked up, every observation starts with
`You carry a ...`. For this reason the expert in `pickup_then_goto` keeps holding the object (see "Differences from GLAM").

Each observation dict carries:

- `image` (7, 7, 3) uint8: per cell (object index, colour index, state), MiniGrid's encoding; the agent sits at cell
  (3, 6), facing up the grid. Input to the spatial encoder of the SLM policy and to B3;
- `direction`: absolute heading 0-3;
- `mission`: the goal string, the text the SLM policy reads;
- `descriptions`: BabyAI-Text descriptions, e.g. `You see a red ball 1 step left and 2 steps forward`, `You see a wall 3 steps right`, `You carry a grey key`;
- `prompt`: the GLAM prompt (goal, last 3 observations, last 2 actions), used by the prompted-SLM baselines.

The text is a deterministic function of `image`, so the text and symbolic views contain the same information.
`info` additionally gives `goal`, the (type, colour) of each object named in the instruction in mission order, and
`goal_positions`, the current grid positions of the matching objects (one list per named object).

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
| `test_nonce` | test | 8 | all except held-out | **invented** (`red`→`dueno`, `ball`→`tism`, ...) |
| `test_heldout` | test | 8 | **exactly one named object is held out** | real |

- Held-out goals follow GLAM: **yellow box, red key, green ball**. They never appear in training instructions, but
  still appear as distractors.
- `get_seeds(task, eval_set, n)` returns the first `n` seeds of a fixed, disjoint range per (split, task).
  `test_nonce` uses the same seeds as `test` and gives the **identical layout and expert trajectory**, with only the words changed.
  For the single-object tasks, `test_heldout` with the same seed has the same object positions and a different goal identity.
- Seed ranges are fixed per (split, task) and do not move when tasks are added (`train` starts at
  `task_index × 10⁶`, `val` at 10⁷ + that, `test` at 2·10⁷ + that).
- Use `val` for all model selection and touch `test*` once at the end (proposal §6.5).

Reference results (`python -m babytext.tools.reference_baselines`, 500 episodes per cell):

| task | eval set | expert success | random success |
|---|---|---|---|
| goto | train / val / test | 1.000 | 0.32 / 0.31 / 0.29 |
| goto | dist16 / nonce / heldout | 1.000 | 0.34 / 0.29 / 0.30 |
| pickup | train / val / test | 1.000 | 0.16 / 0.13 / 0.12 |
| pickup | dist16 / nonce / heldout | 1.000 | 0.12 / 0.12 / 0.14 |
| putnext | train / val / test | 1.000 | 0.02 / 0.02 / 0.02 |
| putnext | dist16 / nonce / heldout | 1.000 | 0.01 / 0.02 / 0.01 |
| pickup_then_goto | train / val / test | 1.000 | 0.08 / 0.07 / 0.07 |
| pickup_then_goto | dist16 / nonce / heldout | 1.000 | 0.05 / 0.07 / 0.07 |

Note the high random floor on `goto` (walking around a cluttered room often ends up facing the goal); report it next to every result.

## Tools

Run from the repo root inside the activated env (outputs go to `data/` and `outputs/` there):

```bash
python -m babytext.tools.reference_baselines   # expert / random success table -> outputs/results/
python -m babytext.tools.visualise_eval_sets   # one figure comparing all eval sets -> outputs/figures/
python -m babytext.tools.visualise_episode --task pickup --eval-set test_dist16 --seed-index 3
python -m babytext.tools.visualise_episode --demos data/demos/goto_train --episode 10 --show-prompt
python -m babytext.tools.play --task goto --eval-set test_nonce   # play by hand; text printed in the terminal
```

Demonstrations are collected with `python -m training.tools.collect_demos` (see [training](../training/README.md)).
`play` needs a display (it opens a pygame window); every other tool renders off-screen. `visualise_episode` writes a GIF (room with field of view highlighted and the named objects outlined, solid for the
first and dashed for the second | agent's view | text) and a PNG
filmstrip. For failure analysis, render any evaluated episode with
`babytext.viz.replay(task, eval_set, episode.seed, episode.actions)` followed by `save_gif` / `save_strip`.

### Demonstration files

`data/demos/<task>_<split>.jsonl` holds one episode per line (seed, mission, goal, outcome, and per-step
`descriptions`, `prompt`, `action`). `data/demos/<task>_<split>.npz` holds flat per-step arrays
(`image`, `direction`, `action`, `episode`, `t`) for the symbolic baseline. Row *i* of the npz is step *i* of
`babytext.data.load_steps(prefix)`, so any per-step array computed later (shaped rewards, features) can be
aligned to both; `episode` and `t` cut the rows back into episodes for models with memory. Episodes use the first N seeds of
the split, so the first *k* episodes are a valid smaller training set for demonstration-count curves.

### Expert relabelling (DAgger)

To label states visited by a trained policy, pass the action actually executed on the previous step to the expert:

```python
expert = ExpertPolicy()
obs, info = env.reset(seed=seed)
expert.reset(env)
prev, done = None, False
while not done:
    label = expert(obs, action_taken=prev)   # training target for this state
    prev = head(obs)                         # the policy decides what is executed
    obs, reward, terminated, truncated, info = env.step(prev)
    done = terminated or truncated
```

Without `action_taken` the bot assumes its own suggestion was followed. After an off-plan pick up or drop it then
crashes or wrongly reports the mission done (25-60% of episodes on the tasks with pick up, with 30% random actions).
If the policy toggles a box, the box opens and vanishes from the room (MiniGrid rule) and the expert raises
`babytext.DisappearedBoxError`. Catch it and stop relabelling that episode.

## Flow

```
make_env(task, eval_set, **overrides)                splits.py   seed range from config.EVAL_SETS[eval_set]
 ├─ LocalLevel(task, goal_pool, num_dists, ...)       levels.py   MiniGrid RoomGridLevel: layout, verifier, reward, time limit
 └─ TextEnv(level, renamer)                           wrapper.py
     reset(seed) / step(action)
      ├─ MiniGrid observation (image, direction, mission)
      ├─ describe(image) -> descriptions              text.py     BabyAI-Text descriptions
      ├─ Renamer (test_nonce only)                    text.py     applied to mission and descriptions, not to image
      └─ PromptHistory -> GLAM prompt                 prompt.py

run_episode(env, policy, seed) -> Episode            rollout.py  policy.reset(env) if defined, then policy(obs) each step
evaluate(policy, task, eval_set, n_episodes)         rollout.py  run_episode over get_seeds(...) -> dict with success_rate, ...
save_demos / load_*                                  data.py     Episode list <-> .jsonl + .npz
replay(task, eval_set, seed, actions)                viz.py      re-runs the actions to produce frames for GIFs / strips
```

## Layout

```
babytext/
  constants.py   fixed definitions: actions, tasks, object vocabulary
  levels.py      LocalLevel: single-room goto / pickup / putnext / pickup_then_goto on MiniGrid's BabyAI RoomGridLevel
  text.py        describe(): BabyAI-Text descriptions; Renamer for invented words
  prompt.py      GLAM prompt template and history
  wrapper.py     TextEnv gymnasium wrapper (text + symbolic observations, 6 actions)
  splits.py      seed ranges, make_env()
  policies.py    ExpertPolicy (BabyAIBot, keeps holding in pickup_then_goto), RandomPolicy
  rollout.py     run_episode(), evaluate()
  data.py        demonstration file format: save / load
  viz.py         frame rendering, GIFs, filmstrips
  tools/         command-line tools: reference baselines, visualisation, manual play
  tests/         environment tests; glam_reference.py holds code transcribed from GLAM for equivalence checks
```

Environment settings (room size, distractors, held-out goals, invented words, prompt window, eval sets) live in the
project-wide [config.py](../config.py) under `ENV` and `EVAL_SETS`. Changing them changes the generated episodes, so
re-collect demos afterwards. For a one-off change, pass overrides to `make_env`, e.g. `make_env("goto", "train", num_dists=4)`.

## Design choices

- **Re-implementation instead of the `babyai-text` package.** GLAM's BabyAI-Text ships as a bundle of forked `babyai` +
  `gym-minigrid` repos on the old `gym` API, which is awkward to install on current Python/Windows. (Lamorel, GLAM's
  LLM serving library, is only needed for their PPO fine-tuning, not for the environment.) Following the fallback in
  proposal §6.1 we wrap Farama's maintained [`minigrid`](https://minigrid.farama.org/), which already contains the
  BabyAI levels and the BabyAI expert bot, with a small text layer that reproduces BabyAI-Text's output **exactly**.
  The tests check our descriptions and prompts against code transcribed from GLAM's repository on every visited state.
- **Seeds identify episodes.** Every episode is `(task, eval_set, seed)`, so any result can be replayed and rendered
  after the fact, and every condition sees exactly the same rooms.
- **Shifted test sets reuse the test seeds**, so a drop on a shifted set comes from the shift, not from different layouts.
- **Both views in one observation.** Text and symbolic observations describe the same step, so text-based and
  grid-based models are compared on identical states.
- **Settings in one place.** Everything that is a choice lives in [config.py](../config.py); `constants.py` holds only
  what is fixed. Each demonstration dataset records the config that produced it.

### Differences from GLAM's BabyAI-Text

- Four separate single-room tasks rather than GLAM's mixture (`MixedTrainLocal` also has open-door, which needs two
  rooms, and "go to Y after you pick up X", which we leave out).
- Instruction objects are sampled first from the allowed pool, then the distractors, instead of picking random placed
  objects and rejecting forbidden missions. Each named object is still uniform over the allowed pairs.
- Two-object instructions always name two different (type, colour) pairs. GLAM allowed "pick up a X, then go to a X"
  when two X exist, but MiniGrid's go-to verifier keeps the positions recorded at reset, so picking up X already
  counts as facing X and the go-to half disappears.
- In `pickup_then_goto` the stock BabyAI bot picks X up and immediately drops it back before going to Y. Two steps later,
  nothing in the 3-observation prompt shows that X was ever picked up, so the expert's labels would depend on
  information the model never sees. Our expert removes that drop and carries X to Y; the bot still puts X down
  occasionally to clear an object blocking its path (about 1% of steps, 6% with 16 distractors).
- Objects are placed one at a time and an object is re-placed if it would make anything unreachable. BabyAI instead rejects
  whole layouts, which needs ~1300 attempts per layout with 16 distractors (~0.5 s per reset vs ~12 ms here). With 8
  distractors the two are practically identical.
- Doors (unused here) would be described as "an open ... door" rather than GLAM's "a open ... door".

## Limitations

- **Sparse reward only.** The reward is non-zero only on the final step of a successful episode. The SLM policy's
  *immediate* reward has to be shaped in `training/`, and the agent's position (needed for distance-based shaping) is
  not in `info`; it is only reachable through `env.unwrapped.agent_pos`. Adding it to `info` is the planned fix.
- **`test_nonce` changes text only.** The renamer rewrites the mission and descriptions; `image` keeps MiniGrid's object
  and colour codes. A model that reads the scene from `image` sees an unchanged room and is only tested on whether it
  can still interpret the renamed mission.
- **Short expert episodes.** The expert needs ~6 steps on the easy tier and ~12 on the hard tier, far below the 64 / 128
  step limits, so demonstrations rarely exercise long-term memory. `toggle` never appears in demonstrations.
- **Random floor on `goto` is high** (~0.3); report it next to every result.
- **Expert relabelling has edge cases**: pass `action_taken`, and catch `DisappearedBoxError` (see above).
- `make_env(..., **overrides)` changes only that one environment; anything that should hold for the whole project
  belongs in `config.py`, followed by re-collecting demonstrations.

## Acknowledgements

Text description format and prompt template are ported from GLAM (Carta et al., 2023,
[flowersteam/Grounding_LLMs_with_online_RL](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)).
Environment dynamics, BabyAI levels and the expert bot are from Farama
[MiniGrid](https://github.com/Farama-Foundation/Minigrid) (Chevalier-Boisvert et al., 2019, 2023).
