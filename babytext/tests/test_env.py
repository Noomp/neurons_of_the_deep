"""Tests for the environment, text interface, splits and data pipeline.

Run with ``python -m pytest`` from the repository root.
"""

import re
from collections import deque

import numpy as np
import pytest
from glam_reference import glam_gen_graph, glam_generate_prompt
from minigrid.core.world_object import Ball

from babytext import (
    ACTION_NAMES,
    ACTION_TO_IDX,
    EVAL_SETS,
    DisappearedBoxError,
    TASKS,
    ExpertPolicy,
    RandomPolicy,
    Renamer,
    describe,
    evaluate,
    get_seeds,
    make_env,
    run_episode,
)
from config import ENV
from babytext.constants import TASK_NUM_OBJECTS

HELD_OUT_GOALS = ENV.held_out_goals
from babytext.data import load_episodes, load_steps, load_symbolic, save_demos


# --- text descriptions ---------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("eval_set", ["train", "test_dist16"])
def test_descriptions_match_glam(task, eval_set):
    """Our descriptions equal GLAM's gen_graph on every visited state, including
    states where the agent carries an object (random policy, 30 episodes)."""
    env = make_env(task, eval_set)
    policy = RandomPolicy(seed=0)
    n_carry = 0
    for seed in get_seeds(task, eval_set, 30):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            assert obs["descriptions"] == glam_gen_graph(env)["descriptions"]
            n_carry += env.unwrapped.carrying is not None
            obs, _, term, trunc, _ = env.step(policy(obs))
            done = term or trunc
    if task != "pickup":  # pickup ends as soon as the goal is picked up
        assert n_carry > 0, "no carrying states were exercised"


def test_description_hand_built_scene():
    """A known layout gives the expected strings."""
    env = make_env("goto", "train", num_dists=0)
    env.reset(seed=0)
    level = env.unwrapped
    for x in range(1, 7):  # clear the room interior
        for y in range(1, 7):
            level.grid.set(x, y, None)
    level.agent_pos, level.agent_dir = (3, 6), 3  # bottom row, facing up (north)
    level.grid.set(2, 4, Ball("red"))  # 1 left, 2 forward
    level.grid.set(3, 2, Ball("blue"))  # 4 forward
    level.grid.set(5, 6, Ball("green"))  # 2 right, same row (hides the right wall)
    level.carrying = Ball("grey")
    obs = level.gen_obs()
    assert describe(obs["image"]) == [
        "You carry a grey ball",
        "You see a wall 3 steps left",  # agent x=3, wall x=0
        "You see a red ball 1 step left and 2 steps forward",
        "You see a blue ball 4 steps forward",
        "You see a green ball 2 steps right",
    ]


# --- prompt --------------------------------------------------------------------


def test_prompt_matches_glam():
    env = make_env("pickup", "train")
    obs, _ = env.reset(seed=get_seeds("pickup", "train", 1)[0])
    obs_q, act_q = deque([obs["descriptions"]], maxlen=3), deque(maxlen=2)
    for action in [0, 2, 1, 2, 0]:
        assert obs["prompt"] == glam_generate_prompt(obs["mission"], ACTION_NAMES, obs_q, act_q)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
        act_q.append(ACTION_NAMES[action])
        obs_q.append(obs["descriptions"])
    assert obs["prompt"].endswith("\n Action 2: ")
    assert obs["prompt"].count("Observation") == 3


def test_prompt_history_resets_between_episodes():
    env = make_env("goto", "train")
    env.reset(seed=1)
    env.step(0)
    env.step(0)
    obs, _ = env.reset(seed=2)
    assert obs["prompt"].count("Observation") == 1


# --- levels and eval sets --------------------------------------------------------


def test_seed_splits_disjoint():
    seen = set()
    for task in TASKS:
        for split in ("train", "val", "test"):
            seeds = set(get_seeds(task, split, 1000))
            assert not seeds & seen
            seen |= seeds


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("eval_set", list(EVAL_SETS))
def test_expert_solves_every_eval_set(task, eval_set):
    result = evaluate(ExpertPolicy(), task, eval_set, 100)
    assert result["success_rate"] == 1.0
    assert all(set(e.actions) <= set(range(len(ACTION_NAMES))) for e in result["episodes"])


def _instr_objects(task, eval_set, n):
    env = make_env(task, eval_set)
    return [env.reset(seed=s)[1]["goal"] for s in get_seeds(task, eval_set, n)]


@pytest.mark.parametrize("task", TASKS)
def test_goal_pools(task):
    """In-distribution instructions never name a held-out object; in test_heldout
    exactly one instruction object is held out."""
    for eval_set in ("train", "val", "test", "test_dist16", "test_nonce"):
        goals = _instr_objects(task, eval_set, 300)
        assert all(len(g) == TASK_NUM_OBJECTS[task] for g in goals)
        named = {obj for g in goals for obj in g}
        assert not named & HELD_OUT_GOALS, eval_set
        assert len(named) == 18 - len(HELD_OUT_GOALS)
    goals = _instr_objects(task, "test_heldout", 100)
    assert all(sum(obj in HELD_OUT_GOALS for obj in g) == 1 for g in goals)
    assert {obj for g in goals for obj in g if obj in HELD_OUT_GOALS} == set(HELD_OUT_GOALS)


@pytest.mark.parametrize("task", ["putnext", "pickup_then_goto"])
def test_two_object_tasks_name_different_objects(task):
    assert all(a != b for a, b in _instr_objects(task, "train", 200))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("eval_set", ["train", "test_dist16"])
def test_object_count(task, eval_set):
    expected = EVAL_SETS[eval_set].level_kwargs()["num_dists"] + TASK_NUM_OBJECTS[task]
    env = make_env(task, eval_set)
    for seed in get_seeds(task, eval_set, 20):
        _, info = env.reset(seed=seed)
        objs = [c for c in env.unwrapped.grid.grid if c is not None and c.type != "wall"]
        assert len(objs) == expected
        assert all(len(p) > 0 for p in info["goal_positions"])  # every named object is on the floor


@pytest.mark.parametrize(
    "task,pattern",
    [
        ("goto", r"^go to (the|a) \w+ \w+$"),
        ("pickup", r"^pick up (the|a) \w+ \w+$"),
        ("putnext", r"^put (the|a) \w+ \w+ next to (the|a) \w+ \w+$"),
        ("pickup_then_goto", r"^pick up (the|a) \w+ \w+, then go to (the|a) \w+ \w+$"),
    ],
)
def test_mission_format(task, pattern):
    env = make_env(task, "train")
    for seed in get_seeds(task, "train", 20):
        obs, _ = env.reset(seed=seed)
        assert re.match(pattern, obs["mission"]), obs["mission"]


def test_seed_layout_is_stable():
    """Seed ranges must not move when tasks are added (data would silently change)."""
    assert get_seeds("goto", "train", 1) == [0]
    assert get_seeds("pickup", "train", 1) == [1_000_000]
    assert get_seeds("goto", "val", 1) == [10_000_000]
    assert get_seeds("pickup_then_goto", "test_nonce", 1) == [23_000_000]


def test_expert_keeps_carrying_in_pickup_then_goto():
    """The expert must not drop the picked-up object (see ExpertPolicy), so the
    prompt always shows whether the first half of the task is done."""
    result = evaluate(ExpertPolicy(), "pickup_then_goto", "train", 100)
    after_pickup = carrying = 0
    for ep in result["episodes"]:
        first = ep.actions.index(ACTION_TO_IDX["pick up"])
        assert ep.actions[first + 1] != ACTION_TO_IDX["drop"]
        for step in ep.steps[first + 1 :]:
            after_pickup += 1
            carrying += step.descriptions[0].startswith("You carry")
    assert carrying / after_pickup > 0.95


def test_determinism():
    for eval_set in EVAL_SETS:
        seed = get_seeds("pickup", eval_set, 1)[0]
        a = run_episode(make_env("pickup", eval_set), ExpertPolicy(), seed)
        b = run_episode(make_env("pickup", eval_set), ExpertPolicy(), seed)
        assert a.mission == b.mission and a.actions == b.actions
        assert [s.prompt for s in a.steps] == [s.prompt for s in b.steps]


@pytest.mark.parametrize("task", TASKS)
def test_nonce_renaming(task):
    """test_nonce has the same layout as test, and no real colour/object word survives."""
    real_words = set(ENV.nonce_words)
    for seed in get_seeds(task, "test", 20):
        a = run_episode(make_env(task, "test"), ExpertPolicy(), seed)
        b = run_episode(make_env(task, "test_nonce"), ExpertPolicy(), seed)
        assert a.actions == b.actions
        assert b.mission == Renamer()(a.mission)
        for step in b.steps:
            text = step.prompt.replace("Possible action of the agent", "")
            assert not real_words & set(text.replace(",", " ").split())


def test_text_is_function_of_symbolic_obs():
    env = make_env("goto", "train")
    obs, _ = env.reset(seed=3)
    assert obs["descriptions"] == describe(obs["image"])
    assert obs["image"].shape == (7, 7, 3) and obs["image"].dtype == np.uint8


def test_invalid_action_rejected():
    env = make_env("goto", "train")
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(6)  # MiniGrid's "done" is not part of the action space


# --- data ----------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    env = make_env("goto", "train")
    episodes = [run_episode(env, ExpertPolicy(), s, "goto", "train") for s in get_seeds("goto", "train", 5)]
    prefix = tmp_path / "goto_train"
    save_demos(episodes, prefix)

    records = load_episodes(prefix)
    steps = load_steps(prefix)
    arrays = load_symbolic(prefix)
    n = sum(e.length for e in episodes)
    assert len(records) == 5 and len(steps) == n and arrays["action"].shape == (n,)
    assert [r["action"] for r in steps] == arrays["action"].tolist()
    flat = [s for e in episodes for s in e.steps]
    assert all(np.array_equal(s.image, img) for s, img in zip(flat, arrays["image"]))
    assert [r["prompt"] for r in steps] == [s.prompt for s in flat]


# --- expert relabelling (DAgger) -------------------------------------------------


def _mixed_rollout(task, seed, eps, allow_toggle, rng):
    """Expert labels every step; with probability eps a random action is executed
    instead and reported back via action_taken. Returns True on success."""
    env = make_env(task, "train")
    obs, _ = env.reset(seed=seed)
    expert = ExpertPolicy()
    expert.reset(env)
    prev = None
    while True:
        action = expert(obs, action_taken=prev)
        if rng.random() < eps:
            action = int(rng.integers(len(ACTION_NAMES) if allow_toggle else ACTION_TO_IDX["toggle"]))
        obs, reward, term, trunc, _ = env.step(action)
        prev = action
        if term or trunc:
            return term and reward > 0


@pytest.mark.parametrize("task", TASKS)
def test_expert_relabels_off_policy_states(task):
    """Told what was actually executed, the expert recovers from every deviation."""
    rng = np.random.default_rng(0)
    assert all(_mixed_rollout(task, s, 0.3, False, rng) for s in get_seeds(task, "train", 50))


@pytest.mark.parametrize("task", TASKS)
def test_expert_only_gives_up_when_a_box_is_opened(task):
    rng = np.random.default_rng(1)
    for seed in get_seeds(task, "train", 50):
        try:
            assert _mixed_rollout(task, seed, 0.3, True, rng)
        except DisappearedBoxError:
            pass


@pytest.mark.parametrize("task", TASKS)
def test_action_taken_matches_demos_when_expert_is_followed(task):
    """Reporting the expert's own actions back changes nothing."""
    env = make_env(task, "train")
    for seed in get_seeds(task, "train", 20):
        demo = run_episode(env, ExpertPolicy(), seed).actions
        obs, _ = env.reset(seed=seed)
        expert = ExpertPolicy()
        expert.reset(env)
        actions, prev, done = [], None, False
        while not done:
            prev = expert(obs, action_taken=prev)
            actions.append(prev)
            obs, _, term, trunc, _ = env.step(prev)
            done = term or trunc
        assert actions == demo
