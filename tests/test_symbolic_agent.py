"""Tests for the B3 symbolic agent's inputs and forward pass (no training)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from babytext import TASKS, ExpertPolicy, evaluate, get_seeds, make_env, run_episode  # noqa: E402
from babytext.data import load_steps, load_symbolic, save_demos  # noqa: E402
from config import ENV  # noqa: E402
from models.symbolic_agent import (  # noqa: E402
    NO_ACTION,
    MissionTokenizer,
    SymbolicAgent,
    SymbolicHistory,
    SymbolicPolicy,
    stack_demo_steps,
)


def _batch(model, stacked, missions):
    tokens, lengths = model.tokenizer.encode(missions)
    return {
        **{k: torch.from_numpy(v) for k, v in stacked.items()},
        "mission_tokens": tokens,
        "mission_lengths": lengths,
    }


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """A small saved demo file for the two-object pick-up task."""
    env = make_env("pickup_then_goto", "train")
    episodes = [run_episode(env, ExpertPolicy(), s, "pickup_then_goto", "train") for s in get_seeds("pickup_then_goto", "train", 10)]
    prefix = tmp_path_factory.mktemp("demos") / "pickup_then_goto_train"
    save_demos(episodes, prefix)
    return episodes, load_symbolic(prefix), load_steps(prefix)


def test_parameter_budget():
    """Proposal 6.3: B3 has fewer than 1M trainable parameters."""
    assert SymbolicAgent().num_parameters() < 1_000_000


def test_forward_shape(demo):
    _, arrays, steps = demo
    model = SymbolicAgent().eval()
    stacked = stack_demo_steps(arrays["image"], arrays["action"], arrays["t"])
    logits = model(**_batch(model, stacked, [s["mission"] for s in steps]))
    assert logits.shape == (len(steps), 6) and torch.isfinite(logits).all()


def test_demo_stacking_matches_rollout_history(demo):
    """Inputs built from saved demos equal those built live during a rollout."""
    episodes, arrays, _ = demo
    stacked = stack_demo_steps(arrays["image"], arrays["action"], arrays["t"])
    row = 0
    for ep in episodes:
        history = SymbolicHistory()
        for step in ep.steps:
            history.add_observation(step.image)
            live = history.inputs()
            for key in stacked:
                np.testing.assert_array_equal(live[key][0], stacked[key][row], err_msg=f"{key} row {row}")
            history.add_action(step.action)
            row += 1


def test_episode_start_is_padded(demo):
    _, arrays, _ = demo
    stacked = stack_demo_steps(arrays["image"], arrays["action"], arrays["t"])
    first = np.flatnonzero(arrays["t"] == 0)
    n = ENV.prompt_num_obs
    assert (stacked["frame_mask"][first] == [False] * (n - 1) + [True]).all()
    assert (stacked["prev_actions"][first] == NO_ACTION).all()
    assert (stacked["frames"][first][:, : n - 1] == 0).all()


def test_missing_frames_are_ignored(demo):
    """Whatever sits in a masked-out frame slot cannot change the output."""
    _, arrays, steps = demo
    model = SymbolicAgent().eval()
    stacked = stack_demo_steps(arrays["image"], arrays["action"], arrays["t"])
    first = np.flatnonzero(arrays["t"] == 0)
    sub = {k: v[first] for k, v in stacked.items()}
    missions = [steps[i]["mission"] for i in first]
    noisy = dict(sub, frames=sub["frames"].copy())
    rng = np.random.default_rng(0)
    for channel, high in enumerate((11, 6, 3)):  # object, colour, state index ranges
        noisy["frames"][:, 0, ..., channel] = rng.integers(0, high, size=noisy["frames"][:, 0, ..., channel].shape)
    with torch.no_grad():
        torch.testing.assert_close(model(**_batch(model, sub, missions)), model(**_batch(model, noisy, missions)))


def test_tokenizer_covers_missions_and_flags_invented_words():
    tok = MissionTokenizer()
    for task in TASKS:
        env = make_env(task, "train")
        for seed in get_seeds(task, "train", 30):
            tokens, _ = tok.encode([env.reset(seed=seed)[0]["mission"]])
            assert MissionTokenizer.UNK not in tokens.tolist()[0]
    nonce = make_env("putnext", "test_nonce").reset(seed=get_seeds("putnext", "test_nonce", 1)[0])[0]["mission"]
    assert tok.encode([nonce])[0].tolist()[0].count(MissionTokenizer.UNK) == 4  # 2 colours + 2 objects


@pytest.mark.parametrize("task", TASKS)
def test_policy_runs_in_evaluate(task):
    result = evaluate(SymbolicPolicy(SymbolicAgent()), task, "val", 2)
    assert result["n_episodes"] == 2
