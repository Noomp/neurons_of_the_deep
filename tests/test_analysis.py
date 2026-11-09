"""Tests for the failure analysis and the report figures (no model is run: random and expert policies only)."""

import csv
import json

import numpy as np
import pytest

from analysis import failures, figures
from babytext import ACTION_TO_IDX, ExpertPolicy, RandomPolicy, get_seeds, make_env, run_episode
from evaluation.record import evaluate_cell
from evaluation.summarise import load_cells

LEFT, PICKUP = ACTION_TO_IDX["turn left"], ACTION_TO_IDX["pick up"]


@pytest.mark.parametrize("task", ["goto", "pickup", "putnext", "pickup_then_goto"])
def test_replay_reproduces_expert_success(task):
    """The replay is the recorded episode: an expert episode succeeds, picks nothing wrong and sees its goal."""
    env = make_env(task, "test")
    seed = get_seeds(task, "test", 1)[0]
    episode = run_episode(env, ExpertPolicy(), seed)
    f = failures.episode_features(task, "test", seed, episode.actions)
    assert f["success"] and f["length"] == len(episode.actions)
    assert not f["wrong_pickup"] and f["saw_goal"]
    assert f["carried_goal"] == (task != "goto")
    assert abs(sum(f[f"chance_{k}"] for k in ("other_instruction_object", "lookalike", "unrelated")) - 1) < 1e-9


def test_turning_on_the_spot_is_stuck_spinning():
    seed = get_seeds("goto", "test", 1)[0]
    f = failures.episode_features("goto", "test", seed, [LEFT] * 64, stuck_window=16, stuck_cells=2)
    assert not f["success"] and f["stuck"] and f["stuck_kind"] == "spinning"
    assert failures.classify("goto", f) == "stuck"


def test_wrong_pickup_is_found_and_classified():
    """A random policy on pickup eventually picks up a non-goal object: recorded with its kind and step."""
    for seed in get_seeds("pickup", "test", 200):
        episode = run_episode(make_env("pickup", "test"), RandomPolicy(seed=seed), seed)
        f = failures.episode_features("pickup", "test", seed, episode.actions)
        if f["wrong_pickup"]:
            assert f["wrong_pickup_kind"] in ("lookalike", "unrelated")  # one instruction object: never "other"
            assert episode.actions[f["wrong_pickup_t"]] == PICKUP
            assert failures.classify("pickup", f) == "wrong_object"
            return
    pytest.fail("no wrong pickup in 200 random episodes")


def test_category_order():
    base = dict(wrong_pickup=False, carried_goal=False, faced_goal=False, stuck=False, saw_goal=True)
    assert failures.classify("putnext", {**base, "wrong_pickup": True, "carried_goal": True}) == "wrong_object"
    assert failures.classify("putnext", {**base, "carried_goal": True, "stuck": True}) == "first_step_only"
    assert failures.classify("pickup", {**base, "faced_goal": True, "stuck": True}) == "reached_no_pickup"
    assert failures.classify("goto", {**base, "faced_goal": True, "stuck": True}) == "stuck"  # goto needs no pickup
    assert failures.classify("goto", {**base, "saw_goal": False}) == "never_saw_goal"
    assert failures.classify("goto", base) == "wandered"


def test_confusion_and_agreement_by_step():
    episodes = [{"actions": [0, 2, 3], "expert": [0, 1, None], "success": True},
                {"actions": [1, 1], "expert": [1, 2], "success": False}]
    m = failures.confusion(episodes)
    assert m.sum() == 4 and m[0, 0] == 1 and m[1, 2] == 1 and m[1, 1] == 1 and m[2, 1] == 1
    a = failures.agreement_by_step(episodes, 3)
    assert a["success"] == [1.0, 0.0, None] and a["n_success"] == [1, 1, 0]
    assert a["failure"] == [1.0, 0.0, None]


@pytest.fixture(scope="module")
def eval_dir(tmp_path_factory):
    """A small evaluation output: a random 'trained' condition (one seed) and the random baseline, easy tier."""
    root = tmp_path_factory.mktemp("eval")
    for model, run in (("P-L12-attn-h1", "seed0"), ("random", "zero-shot")):
        for task in ("goto", "pickup"):
            for eval_set in ("test", "test_nonce"):
                identity = {"model": model, "tier": "easy", "run": run, "task": task, "eval_set": eval_set}
                evaluate_cell(RandomPolicy(seed=1), task, eval_set, 6, root / model / "easy" / run / f"{task}.{eval_set}",
                              {"identity": identity})
    return root


def test_failure_rows_cover_every_failed_episode(eval_dir):
    rows = failures.failure_rows(eval_dir, models=("P-L12-attn-h1",), workers=1)
    cells = [c for c in load_cells(eval_dir) if c["identity"]["model"] == "P-L12-attn-h1"]
    assert len(rows) == sum(round((1 - c["success_rate"]) * c["n_episodes"]) for c in cells)
    assert all(r["category"] in failures.CATEGORIES for r in rows)
    shares = failures.category_shares(rows, cells)
    for key, d in shares.items():
        assert abs(sum(d["shares"].values()) - d["failed"] / d["episodes"]) < 1e-9


def test_main_writes_tables_and_figures(eval_dir, tmp_path):
    results, figs = tmp_path / "results", tmp_path / "figures"
    failures.main(["--eval", str(eval_dir), "--out", str(results), "--models", "P-L12-attn-h1", "--workers", "1"])
    text = (results / "failures.md").read_text(encoding="utf-8")
    assert "Failure categories, per tier" in text and "What the wrong object was" in text
    with open(results / "failures.csv", encoding="utf-8") as f:
        assert {r["model"] for r in csv.DictReader(f)} <= {"P-L12-attn-h1"}
    tables = json.loads((results / "failures.json").read_text(encoding="utf-8"))
    assert np.array(tables["random|easy|test"]["confusion"]).shape == (6, 6)

    figures.main(["--eval", str(eval_dir), "--runs", str(tmp_path / "no_runs"), "--results", str(results),
                  "--out", str(figs)])
    for name in ("results_success", "results_gaps", "results_cost", "results_failures", "results_confusion",
                 "results_drift"):
        assert (figs / f"{name}.png").stat().st_size > 0 and (figs / f"{name}.svg").exists()
