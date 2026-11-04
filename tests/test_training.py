"""Tests for the training command line: --set overrides, the demo and cache checks,
the shared behaviour-cloning protocol for the action head and B3, and the run plan."""

import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import config  # noqa: E402
from babytext import get_seeds, make_env, run_episode  # noqa: E402
from config import SYMBOLIC  # noqa: E402
from models import action_head, symbolic_agent  # noqa: E402
from models.symbolic_agent import SymbolicAgent  # noqa: E402
from training import run_plan  # noqa: E402
from training.bc_symbolic import symbolic_demos  # noqa: E402
from training.demos import batch_indices, load_demo_episodes  # noqa: E402
from training.overrides import OverrideError, effective_configs, parse_overrides  # noqa: E402
from training.train import FIXED, main  # noqa: E402

QUICK_BC = ["bc.epochs=1", "bc.batch_episodes=4", "bc.val_episodes=1"]


def sets(*items):
    return [a for item in items for a in ("--set", item)]


def run_args(demos, out, *extra):
    return ["--tasks", "goto", "--demos", str(demos), "--out", str(out), "--device", "cpu", *extra]


def head_args(demos, hidden, out, llm_sets, *extra):
    return run_args(demos, out, "--hidden", str(hidden), *llm_sets, *extra)


def read_run(out):
    return json.loads((out / "run.json").read_text())


def metrics(out):
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]


# --- --set parsing ----------------------------------------------------------------


def test_parse_overrides_types():
    parsed = parse_overrides(["head.layer=16", "head.pooling=mean", "bc.lr=1e-3", "train.seeds=3,4",
                              "llm.dtype=float32", "symbolic.hidden_dim=64"])
    assert parsed == {"head": {"layer": 16, "pooling": "mean"}, "bc": {"lr": 1e-3}, "train": {"seeds": (3, 4)},
                      "llm": {"dtype": "float32"}, "symbolic": {"hidden_dim": 64}}
    cfgs = effective_configs(parsed)
    assert cfgs["head"].layer == 16 and cfgs["head"].hidden_dim == config.HEAD.hidden_dim
    assert cfgs["bc"].lr == 1e-3 and cfgs["symbolic"] == replace(SYMBOLIC, hidden_dim=64)


@pytest.mark.parametrize("item, message", [
    ("bc.epochs=2.5", "not a valid int"),
    ("head.layer=ten", "not a valid int"),
    ("train.seeds=1,x", "not a valid int"),
    ("bc.epoch=3", "has no field 'epoch'"),
    ("slm.memory_slots=0", "unknown section"),
    ("env.num_dists=4", "re-collect"),
    ("eval_sets.val=x", "re-collect"),
    ("data.n_train_demos=5", "re-collect"),
    ("bc.epochs", "expected section.field=value"),
    ("head.sweep_layers=4,8", "only decides which layers"),
])
def test_parse_overrides_errors(item, message):
    with pytest.raises(OverrideError, match=message):
        parse_overrides([item], FIXED)


def test_override_set_twice_and_cli_error(demos, tmp_path, capsys):
    with pytest.raises(OverrideError, match="set twice"):
        parse_overrides(["bc.epochs=1", "bc.epochs=2"])
    with pytest.raises(SystemExit):
        main(run_args(demos, tmp_path, "--model", "b3", *sets("bc.epochs=x")))
    assert "not a valid int" in capsys.readouterr().err


# --- the action head --------------------------------------------------------------------


@pytest.mark.parametrize("layer, pooling", [(2, "last"), (4, "mean"), (2, "attention")])
def test_head_trains_from_cache(demos, hidden, llm_sets, tmp_path, layer, pooling):
    out = tmp_path / "p"
    main(head_args(demos, hidden, out, llm_sets, "--id", "P",
                   *sets(f"head.layer={layer}", f"head.pooling={pooling}", "head.hidden_layers=2",
                         "bc.epochs=4", "bc.batch_episodes=2", "bc.lr=3e-3", "bc.val_episodes=1")))
    run = read_run(out)
    assert run["finished"] and run["model_id"] == "P" and run["model"]["cfg"]["layer"] == layer
    assert run["model"]["cfg"]["pooling"] == pooling and run["model"]["llm"]["dtype"] == "float32"
    assert {"selected_epoch", "val_ce", "val_success", "val_goto"} <= set(run["selected"])
    lines = metrics(out)
    assert len(lines) == 4 + 1  # one line per epoch, then the selection
    assert lines[-1]["selected_epoch"] == min(range(4), key=lambda e: lines[e]["val_ce"])
    assert lines[3]["ce"] < lines[0]["ce"]
    assert all((out / f).exists() for f in ("best.pt", "last.pt"))
    info = action_head.load_checkpoint_info(out / "best.pt")
    assert info["epoch"] == run["selected"]["selected_epoch"] and info["val_success"] == run["selected"]["val_success"]
    policy = action_head.load_policy(out / "best.pt")
    assert policy.head.cfg.layer == layer and policy.lm.num_layers == layer
    assert run_episode(make_env("goto", "val"), policy, get_seeds("goto", "val", 1)[0]).length >= 1


def test_head_refuses_missing_or_stale_cache(demos, hidden, llm_sets, tmp_path):
    with pytest.raises(SystemExit, match="layer 3 has no cached pooled vectors"):
        main(head_args(demos, hidden, tmp_path / "a", llm_sets, *sets("head.layer=3", *QUICK_BC)))
    with pytest.raises(SystemExit, match="layer 4 has no cached token sequences"):
        main(head_args(demos, hidden, tmp_path / "b", llm_sets, *sets("head.layer=4", "head.pooling=attention", *QUICK_BC)))
    other_dtype = [a.replace("llm.dtype=float32", "llm.dtype=bfloat16") for a in llm_sets]
    with pytest.raises(SystemExit, match="no hidden-state cache"):
        main(head_args(demos, hidden, tmp_path / "c", other_dtype, *sets("head.layer=2", *QUICK_BC)))
    changed = tmp_path / "demos"
    shutil.copytree(demos, changed)
    lines = (changed / "goto_train.jsonl").read_text().splitlines()
    (changed / "goto_train.jsonl").write_text("\n".join(json.dumps(json.loads(x), indent=None, separators=(",", ":"))
                                                        for x in lines) + "\n")
    with pytest.raises(SystemExit, match="other demos"):
        main(head_args(changed, hidden, tmp_path / "d", llm_sets, *sets("head.layer=2", *QUICK_BC)))


def test_demo_config_mismatch_refuses(demos, tmp_path):
    stale = tmp_path / "stale"
    shutil.copytree(demos, stale)
    summary = json.loads((stale / "summary.json").read_text())
    summary["config"]["env"]["num_dists"] = 4
    (stale / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit, match=r"env\.num_dists: demos 4, config\.py 8"):
        main(run_args(stale, tmp_path / "run"))
    (stale / "summary.json").unlink()
    with pytest.raises(SystemExit, match="re-collect"):
        main(run_args(stale, tmp_path / "run", "--model", "b3"))
    assert not (tmp_path / "run").exists()


# --- B3 under the same protocol -----------------------------------------------------------


def test_b3_trains_with_the_same_protocol(demos, tmp_path):
    out = tmp_path / "b3"
    main(run_args(demos, out, "--model", "b3", "--n-demos", "10", "--id", "B3",
                  *sets("bc.epochs=2", "bc.batch_episodes=2", "bc.lr=1e-3", "bc.val_episodes=1", "head.layer=2")))
    run = read_run(out)
    assert run["finished"] and run["model"]["type"] == "b3" and "val_success" in run["selected"]
    assert any("--set head.* has no effect" in w for w in run["warnings"])
    lines = metrics(out)
    assert len(lines) == 3 and {"ce", "accuracy", "val_ce", "val_accuracy"} <= set(lines[0])
    assert all((out / f).exists() for f in ("best.pt", "last.pt"))


def test_b3_save_load_same_logits(demos, tmp_path):
    model = SymbolicAgent(replace(SYMBOLIC, conv_channels=8, hidden_dim=16))
    data = symbolic_demos(load_demo_episodes(demos / "goto_train", 4), model)
    inputs, _ = data.batch(np.arange(len(data)), "cpu")
    model.train()(**inputs)  # move the BatchNorm running statistics away from their initial values
    symbolic_agent.save_model(model, tmp_path / "b3.pt", epoch=3)
    loaded = symbolic_agent.load_model(tmp_path / "b3.pt")
    assert symbolic_agent.load_checkpoint_info(tmp_path / "b3.pt") == {"epoch": 3}
    assert loaded.cfg.conv_channels == 8 and not loaded.training
    with torch.no_grad():
        torch.testing.assert_close(model.eval()(**inputs), loaded(**inputs))


def test_both_trainers_see_the_same_batches(demos):
    lengths = [len(e) for e in load_demo_episodes(demos / "goto_train")]
    first = list(batch_indices(lengths, 3, np.random.default_rng(5)))
    again = list(batch_indices(lengths, 3, np.random.default_rng(5)))
    assert [list(b) for b in first] == [list(b) for b in again]
    assert sorted(i for b in first for i in b) == list(range(len(lengths)))


# --- early stopping and learning-rate decay -----------------------------------------------


class FixedData:
    """Random features with fixed labels; one episode per 4 steps."""

    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def lengths(self):
        return [4] * (len(self.labels) // 4)

    def batch(self, episodes, device):
        rows = torch.tensor([4 * e + i for e in episodes for i in range(4)])
        return {"x": self.features[rows]}, self.labels[rows]


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 6)

    def forward(self, x):
        return self.fc(x)


def fit(tmp_path, val_labels_shift, **cfg):
    from training.bc import train_bc

    torch.manual_seed(0)
    x, y = torch.randn(64, 8), torch.randint(0, 6, (64,))
    model = Linear()
    _, summary = train_bc(model, FixedData(x, y), FixedData(x, (y + val_labels_shift) % 6), ["goto"], tmp_path,
                          make_policy=None, save=lambda *a, **k: None,
                          cfg=replace(config.BC, batch_episodes=4, val_episodes=0, **cfg))
    return summary, metrics(tmp_path)


def test_early_stopping_ends_when_val_stops_improving(tmp_path):
    # Val labels differ from train labels, so fitting the train set makes val cross-entropy worse.
    summary, lines = fit(tmp_path, 1, epochs=50, lr=0.1, patience=2, lr_decay="none")
    assert summary["stopped_early"] and summary["epochs_run"] == summary["selected_epoch"] + 3
    assert len(lines) == summary["epochs_run"] + 1


def test_plateau_decay_lowers_lr_when_val_stalls(tmp_path):
    summary, lines = fit(tmp_path, 1, epochs=4, lr=0.1, patience=0, lr_decay="plateau", lr_patience=0, lr_factor=0.5)
    lrs = [line["lr"] for line in lines[:-1]]
    assert lrs[0] == 0.1 and lrs[-1] < 0.1 and not summary["stopped_early"]


def test_cosine_decay_and_unknown_schedule(tmp_path):
    _, lines = fit(tmp_path / "c", 0, epochs=4, lr=0.1, patience=0, lr_decay="cosine", min_lr=0.0)
    lrs = [line["lr"] for line in lines[:-1]]
    assert lrs == sorted(lrs, reverse=True) and lrs[0] == 0.1
    with pytest.raises(ValueError, match="lr_decay"):
        fit(tmp_path / "x", 0, epochs=1, lr_decay="step")


# --- run plan ---------------------------------------------------------------------------


def plan_lines(capsys, *argv):
    run_plan.main(["--dry-run", *argv])
    return [line for line in capsys.readouterr().out.splitlines() if line.startswith("python")]


def test_run_plan(capsys):
    lines = plan_lines(capsys)
    training = [line for line in lines if "training.train" in line]
    n_ids = 1 + len(config.HEAD.sweep_layers) * 2 * 2
    assert len(training) == n_ids * len(run_plan.TIERS) * len(config.TRAIN.seeds)
    assert sum("cache_hidden" in line for line in lines) == len(run_plan.TIERS)
    p = next(line for line in training if "--id P-L16-mean-h2 " in line and "hard/seed1" in line)
    assert "--set head.layer=16 --set head.pooling=mean --set head.hidden_layers=2" in p
    assert "--tasks putnext pickup_then_goto" in p

    attn = plan_lines(capsys, "--attention-layers", "12", "--ids", "P-*-attn-*", "--tiers", "easy", "--seeds", "0")
    assert any("--sequence-layers 12" in line for line in attn)
    assert sum("training.train" in line for line in attn) == 2

    curve = plan_lines(capsys, "--curve", "B3", "--ids", "B3-n*", "--tiers", "easy", "--seeds", "0")
    counts = [k for k in config.TRAIN.demo_counts if k < config.DATA.n_train_demos]
    assert [line.split("--n-demos ")[1].split()[0] for line in curve] == [str(k) for k in counts]
