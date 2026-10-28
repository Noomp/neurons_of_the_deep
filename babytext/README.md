# babytext

The environment for the project: a BabyAI-Text-style text gridworld on Farama MiniGrid, with expert demonstrations,
fixed train/val/test splits, three shifted test sets and visualisation tools. This is objective O1 of the
[proposal](../project_proposal.md), minus the LLM side.

## Why not the `babyai-text` package?

GLAM's BabyAI-Text ships as a bundle of forked `babyai` + `gym-minigrid` repos on the old `gym` API,
which is awkward to install on current Python/Windows. (Lamorel, GLAM's LLM serving library, is only needed for
their PPO fine-tuning, not for the environment.) Following the fallback in proposal §6.1 we instead
wrap Farama's maintained [`minigrid`](https://minigrid.farama.org/), which already contains the BabyAI levels and
the BabyAI expert bot, with a small text layer that reproduces BabyAI-Text's output **exactly**. The tests check our
descriptions and prompts against code transcribed from GLAM's repository on every visited state.

## Quick start

```python
from babytext import make_env, get_seeds, ExpertPolicy, run_episode, evaluate

env = make_env("goto", "train")                     # task in babytext.TASKS
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

- `image` (7, 7, 3) uint8 and `direction`: MiniGrid's symbolic view;
- `mission`: the goal string;
- `descriptions`: BabyAI-Text descriptions, e.g. `You see a red ball 1 step left and 2 steps forward`, `You see a wall 3 steps right`, `You carry a grey key`;
- `prompt`: the GLAM prompt (goal, last 3 observations, last 2 actions), identical for every LLM-based condition.

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
python -m babytext.tools.collect_demos         # expert demos -> data/demos/ (2000 train + 200 val per task, ~30 s)
python -m babytext.tools.dataset_stats         # charts of demos and eval sets -> outputs/figures/
python -m babytext.tools.reference_baselines   # expert / random success table -> outputs/results/
python -m babytext.tools.visualise_eval_sets   # one figure comparing all eval sets -> outputs/figures/
python -m babytext.tools.visualise_episode --task pickup --eval-set test_dist16 --seed-index 3
python -m babytext.tools.visualise_episode --demos data/demos/goto_train --episode 10 --show-prompt
python -m babytext.tools.play --task goto --eval-set test_nonce   # play by hand; text printed in the terminal
```

`visualise_episode` writes a GIF (room with field of view highlighted and the named objects outlined, solid for the
first and dashed for the second | agent's view | text) and a PNG
filmstrip. For failure analysis, render any evaluated episode with
`babytext.viz.replay(task, eval_set, episode.seed, episode.actions)` followed by `save_gif` / `save_strip`.

### Demonstration files

`data/demos/<task>_<split>.jsonl` holds one episode per line (seed, mission, goal, outcome, and per-step
`descriptions`, `prompt`, `action`). `data/demos/<task>_<split>.npz` holds flat per-step arrays
(`image`, `direction`, `action`, `episode`, `t`) for the symbolic baseline. Row *i* of the npz is step *i* of
`babytext.data.load_steps(prefix)`, so cached hidden states can be aligned to both. Episodes use the first N seeds of
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
  data.py        save / load demonstrations
  viz.py         frame rendering, GIFs, filmstrips
  tools/         command-line tools: collection, statistics, visualisation, manual play
  tests/         environment tests; glam_reference.py holds code transcribed from GLAM for equivalence checks
```

Environment settings (room size, distractors, held-out goals, invented words, prompt window, eval sets) live in the
project-wide [config.py](../config.py) under `ENV` and `EVAL_SETS`. Changing them changes the generated episodes, so
re-collect demos afterwards. For a one-off change, pass overrides to `make_env`, e.g. `make_env("goto", "train", num_dists=4)`.

## Differences from GLAM's BabyAI-Text

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

## Acknowledgements

Text description format and prompt template are ported from GLAM (Carta et al., 2023,
[flowersteam/Grounding_LLMs_with_online_RL](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)).
Environment dynamics, BabyAI levels and the expert bot are from Farama
[MiniGrid](https://github.com/Farama-Foundation/Minigrid) (Chevalier-Boisvert et al., 2019, 2023).
