"""Tests for the final evaluation: recorded rollouts, the resumable evaluation command and the summary tables."""

import json
from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import config  # noqa: E402
from babytext import ExpertPolicy, RandomPolicy, get_seeds, make_env, run_episode  # noqa: E402
from config import HEAD  # noqa: E402
from evaluation import evaluate, summarise  # noqa: E402
from evaluation.record import evaluate_cell, read_episodes, run_recorded_episode  # noqa: E402
from models.action_head import ActionHead, save_model  # noqa: E402
from models.frozen_lm import load_causal_lm  # noqa: E402
from models.prompted import GenerationPolicy  # noqa: E402
from tests.test_prompted import char_embed  # noqa: E402


@pytest.mark.parametrize("task", ["goto", "pickup", "putnext", "pickup_then_goto"])
def test_expert_labels_do_not_change_the_episode(task):
    """Labelling queries the bot on the policy's states; the rollout must be the one babytext produces without it."""
    env = make_env(task, "test")
    for seed in get_seeds(task, "test", 3):
        plain = run_episode(env, RandomPolicy(seed=1), seed)
        recorded = run_recorded_episode(env, RandomPolicy(seed=1), seed)
        assert recorded["actions"] == plain.actions and recorded["success"] == plain.success
        assert len(recorded["expert"]) == len(recorded["ms"]) == recorded["length"]
    expert = run_recorded_episode(env, ExpertPolicy(), get_seeds(task, "test", 1)[0])
    assert expert["success"] and expert["expert"] == expert["actions"]  # the expert agrees with itself


def make_run(runs, tiny_llm, seed=0, std=1.0):
    """A finished training run of a small attention head on the tiny LLM, as ``training.train`` leaves it."""
    out = runs / "P-test" / "easy" / f"seed{seed}"
    head = ActionHead(replace(HEAD, layer=2, pooling="attention"), input_dim=32)
    torch.manual_seed(seed)
    torch.nn.init.normal_(head.mlp[-1].weight, std=std)
    save_model(head, out / "best.pt", tiny_llm, epoch=0)
    info = {"finished": True, "file_config": config.as_dict(), "wall_time_s": 1.5, "selected": {"val_success": 0.5},
            "model": {"type": "head", "llm": asdict(tiny_llm), "trainable_parameters": head.num_parameters()}}
    (out / "run.json").write_text(json.dumps(info))
    return out


def test_evaluate_resumes_and_refuses_other_measurements(tiny_llm, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "LLM", tiny_llm)  # B2 and the head share the backbone, as in the real setup
    runs, out = tmp_path / "runs", tmp_path / "eval"
    run_dir = make_run(runs, tiny_llm)

    def argv(n_episodes=2, *extra):
        return ["--models", "random", "B2", "P-test", "--tiers", "easy", "--sets", "test", "test_nonce", "--seeds", "0",
                "1", "--n-episodes", str(n_episodes), "--runs", str(runs), "--out", str(out), "--device", "cpu",
                "--no-progress", *extra]

    evaluate.main(argv())
    printed = capsys.readouterr().out
    assert "seed1     skipped: no finished training run" in printed  # seed 1 was never trained

    cells = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.json"))
    assert len(cells) == 3 * 2 * 2  # 3 runs x 2 tasks x 2 eval sets
    head_cell = json.loads((out / "P-test/easy/seed0/goto.test.json").read_text())
    assert head_cell["n_episodes"] == 2 and head_cell["identity"]["checkpoint_sha256"]
    assert head_cell["trainable_parameters"] > 0 and head_cell["val"] == {"val_success": 0.5}
    episode = read_episodes(out / "P-test/easy/seed0/goto.test")[0]
    assert len(episode["logits"]) == episode["length"] and len(episode["logits"][0]) == 6
    assert len(episode["expert"]) == len(episode["ms"]) == episode["length"]
    assert len(read_episodes(out / "B2/easy/zero-shot/pickup.test_nonce")[0]["scores"][0]) == 6

    mtimes = {p: p.stat().st_mtime_ns for p in out.rglob("*.json")}
    evaluate.main(argv())  # everything done: nothing is re-evaluated
    assert "0 cells to evaluate" in capsys.readouterr().out
    assert mtimes == {p: p.stat().st_mtime_ns for p in out.rglob("*.json")}

    with pytest.raises(SystemExit):  # a different episode count is a different measurement
        evaluate.main(argv(3))
    make_run(runs, tiny_llm, std=2.0)  # retrained: different weights, same path
    with pytest.raises(SystemExit):
        evaluate.main(argv())
    evaluate.main(argv(2, "--overwrite"))
    assert "4 cells to evaluate" in capsys.readouterr().out  # only the head's cells are redone

    info = json.loads((run_dir / "run.json").read_text())
    info["file_config"]["env"]["num_dists"] += 1
    (run_dir / "run.json").write_text(json.dumps(info))
    with pytest.raises(SystemExit):  # trained in another environment: refused before any episode
        evaluate.main(argv(2, "--dry-run"))

    summarise.main(["--eval", str(out), "--out", str(tmp_path / "results")])
    report = (tmp_path / "results" / "evaluation.md").read_text()
    for expected in ("| random |", "| B2 |", "| P-test |", "Generalisation gap", "test_nonce"):
        assert expected in report


def test_b1_cell_records_text_and_invalid_rate(tiny_llm, tmp_path):
    lm, tok = load_causal_lm(tiny_llm)
    summary = evaluate_cell(GenerationPolicy(lm, tok, char_embed, max_new_tokens=3), "goto", "test", 1,
                            tmp_path / "goto.test", {"identity": {}})
    episode = read_episodes(tmp_path / "goto.test")[0]
    assert len(episode["text"]) == len(episode["valid"]) == episode["length"]
    assert summary["invalid_rate"] == pytest.approx(1 - sum(episode["valid"]) / episode["length"])


def test_intervals():
    i = summarise.mean_interval([0.8, 0.9, 1.0])
    assert i.value == pytest.approx(0.9) and i.high - i.value == pytest.approx(4.303 * 0.1 / 3**0.5)
    w = summarise.wilson(50, 100)
    assert (round(w.low, 3), round(w.high, 3)) == (0.404, 0.596)
    assert summarise.mean_interval([0.7]).low is None  # one seed: no interval
