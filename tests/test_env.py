"""Tests for the environment, text interface, splits and data pipeline.

Run with ``python -m pytest`` from the repository root.
"""

from collections import deque

import numpy as np
import pytest
from glam_reference import glam_gen_graph, glam_generate_prompt
from minigrid.core.world_object import Ball

from babytext import (
    ACTION_NAMES,
    EVAL_SETS,
    HELD_OUT_GOALS,
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
from babytext.constants import NONCE_WORDS
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
    if task == "goto":  # pickup ends as soon as the goal is picked up
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


@pytest.mark.parametrize("task", TASKS)
def test_goal_pools(task):
    for eval_set in ("train", "val", "test", "test_dist16", "test_nonce"):
        goals = {e.goal for e in evaluate(RandomPolicy(0), task, eval_set, 300)["episodes"]}
        assert not goals & HELD_OUT_GOALS, eval_set
        assert len(goals) == 18 - len(HELD_OUT_GOALS)
    held = {e.goal for e in evaluate(RandomPolicy(0), task, "test_heldout", 100)["episodes"]}
    assert held == set(HELD_OUT_GOALS)


@pytest.mark.parametrize("eval_set,expected", [("train", 9), ("test_dist16", 17)])
def test_object_count(eval_set, expected):
    env = make_env("goto", eval_set)
    for seed in get_seeds("goto", eval_set, 20):
        env.reset(seed=seed)
        objs = [c for c in env.unwrapped.grid.grid if c is not None and c.type != "wall"]
        assert len(objs) == expected


def test_determinism():
    for eval_set in EVAL_SETS:
        seed = get_seeds("pickup", eval_set, 1)[0]
        a = run_episode(make_env("pickup", eval_set), ExpertPolicy(), seed)
        b = run_episode(make_env("pickup", eval_set), ExpertPolicy(), seed)
        assert a.mission == b.mission and a.actions == b.actions
        assert [s.prompt for s in a.steps] == [s.prompt for s in b.steps]


def test_nonce_renaming():
    """test_nonce has the same layout as test, and no real colour/object word survives."""
    real_words = set(NONCE_WORDS)
    for seed in get_seeds("goto", "test", 20):
        a = run_episode(make_env("goto", "test"), ExpertPolicy(), seed)
        b = run_episode(make_env("goto", "test_nonce"), ExpertPolicy(), seed)
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
