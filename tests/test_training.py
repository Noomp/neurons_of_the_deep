"""Tests for the training command line: --set overrides, --init configs, the
demo/config check, the B3 trainer and checkpoints, and the new SLM controls.

The SLM is the tiny random Qwen3 of tests/test_slm_policy.py, saved to a local
folder with a byte-level tokenizer so ``slm_name`` can point at it: the real
``from_pretrained`` code path runs and nothing is downloaded.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")

import config  # noqa: E402
from babytext import ExpertPolicy, get_seeds, make_env, run_episode  # noqa: E402
from babytext.data import save_demos  # noqa: E402
from config import SLM, SYMBOLIC  # noqa: E402
from models import symbolic_agent  # noqa: E402
from models.slm_policy import SLMAgentPolicy, SLMPolicy, load_model, observations_from_env, save_model  # noqa: E402
from models.symbolic_agent import SymbolicAgent  # noqa: E402
from training import run_plan  # noqa: E402
from training.bc_symbolic import symbolic_bc_loss, symbolic_demos  # noqa: E402
from training.common import set_seed  # noqa: E402
from training.demos import load_demo_episodes  # noqa: E402
from training.overrides import OverrideError, effective_configs, parse_overrides  # noqa: E402
from training.train import FIXED, main  # noqa: E402

LM_CONFIG = dict(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                 num_attention_heads=4, num_key_value_heads=2, head_dim=8)


@pytest.fixture(scope="module")
def tiny_hub(tmp_path_factory):
    """A folder that ``AutoModel`` / ``AutoConfig`` / ``AutoTokenizer.from_pretrained`` accept."""
    path = tmp_path_factory.mktemp("tiny_qwen3")
    torch.manual_seed(0)
    transformers.Qwen3Model(transformers.Qwen3Config(**LM_CONFIG)).save_pretrained(path)
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tok = tokenizers.Tokenizer(tokenizers.models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    transformers.PreTrainedTokenizerFast(tokenizer_object=tok).save_pretrained(path)
    return path.as_posix()


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    """10 goto train demos and 4 val demos, with a summary.json matching config.py."""
    out = tmp_path_factory.mktemp("demos")
    datasets = {}
    for split, n in (("train", 10), ("val", 4)):
        env = make_env("goto", split)
        eps = [run_episode(env, ExpertPolicy(), s, "goto", split) for s in get_seeds("goto", split, n)]
        save_demos(eps, out / f"goto_{split}")
        datasets[f"goto_{split}"] = {"episodes": n}
    (out / "summary.json").write_text(json.dumps({"datasets": datasets, "config": config.as_dict()}))
    return out


def tiny_slm_sets(hub):
    sizes = dict(feature_dim=16, mission_dim=16, word_dim=8, num_conv_layers=1, memory_slots=3, read_heads=2,
                 readout_hidden=32, low_level_layers=2, lora_rank=2, gradient_checkpointing="false",
                 slm_dtype="float32", slm_name=hub)
    return [f"slm.{k}={v}" for k, v in sizes.items()]


QUICK_BC = ["bc.epochs=1", "bc.batch_episodes=4", "bc.val_episodes=1"]


def sets(*items):
    return [a for item in items for a in ("--set", item)]


def run_args(demos, out, *extra):
    return ["--tasks", "goto", "--demos", str(demos), "--out", str(out), "--device", "cpu", *extra]


def read_run(out):
    return json.loads((out / "run.json").read_text())


def metrics(out):
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]


# --- --set parsing ----------------------------------------------------------------


def test_parse_overrides_types():
    parsed = parse_overrides([
        "slm.memory_slots=0", "bc.lr=1e-3", "slm.use_direction=false", "slm.readout_layer=none",
        "slm.lora_targets=q_proj, k_proj", "train.seeds=3,4", "slm.write_mode=hard",
    ])
    assert parsed == {
        "slm": {"memory_slots": 0, "use_direction": False, "readout_layer": None,
                "lora_targets": ("q_proj", "k_proj"), "write_mode": "hard"},
        "bc": {"lr": 1e-3},
        "train": {"seeds": (3, 4)},
    }
    assert parse_overrides(["slm.readout_layer=10"])["slm"]["readout_layer"] == 10
    cfgs = effective_configs(parsed)
    assert cfgs["slm"].memory_slots == 0 and cfgs["slm"].feature_dim == SLM.feature_dim
    assert cfgs["bc"].lr == 1e-3 and cfgs["ppo"] == config.PPO and cfgs["symbolic"] == SYMBOLIC


@pytest.mark.parametrize("item, message", [
    ("bc.epochs=2.5", "not a valid int"),
    ("slm.use_direction=maybe", "not a bool"),
    ("slm.readout_layer=ten", "not a valid int"),
    ("train.seeds=1,x", "not a valid int"),
    ("bc.epoch=3", "has no field 'epoch'"),
    ("model.size=3", "unknown section"),
    ("env.num_dists=4", "re-collect"),
    ("eval_sets.val=x", "re-collect"),
    ("data.n_train_demos=5", "re-collect"),
    ("bc.epochs", "expected section.field=value"),
    ("slm.backbone=slm", "follows --stage"),
])
def test_parse_overrides_errors(item, message):
    with pytest.raises(OverrideError, match=message):
        parse_overrides([item], FIXED)


def test_override_set_twice_and_cli_error(demos, tmp_path, capsys):
    with pytest.raises(OverrideError, match="set twice"):
        parse_overrides(["bc.epochs=1", "bc.epochs=2"])
    with pytest.raises(SystemExit):
        main(run_args(demos, tmp_path, "--stage", "1", *sets("bc.epochs=x")))
    assert "not a valid int" in capsys.readouterr().err


# --- overrides reach the model and the trainers -----------------------------------------


def test_overrides_reach_model_and_bc(demos, tiny_hub, tmp_path):
    out = tmp_path / "m1"
    main(run_args(demos, out, "--stage", "1", "--id", "M1", *sets(*tiny_slm_sets(tiny_hub), *QUICK_BC[1:], "bc.epochs=2")))
    run = read_run(out)
    assert run["finished"] and run["model_id"] == "M1" and run["bc"]["epochs"] == 2
    assert run["model"]["cfg"]["feature_dim"] == 16 and run["file_config"]["slm"]["feature_dim"] == SLM.feature_dim
    assert run["demos"]["config"]["env"] == run["file_config"]["env"]
    assert set(run["git"]) == {"commit", "dirty"} and run["command"].startswith("python -m training.train --tasks goto")
    assert len(metrics(out)) == 2  # bc.epochs=2 reached train_bc
    assert {"ce", "accuracy", "val_ce", "val_success", "val_goto"} <= set(metrics(out)[0])
    model = load_model(out / "best.pt")
    assert model.cfg.feature_dim == 16 and model.cfg.memory_slots == 3


def test_overrides_reach_ppo(demos, tiny_hub, tmp_path):
    m1 = tmp_path / "m1"
    main(run_args(demos, m1, "--stage", "1", *sets(*tiny_slm_sets(tiny_hub), *QUICK_BC)))
    out = tmp_path / "m4"
    ppo = ["ppo.total_steps=8", "ppo.num_envs=2", "ppo.rollout_len=4", "ppo.minibatch_envs=1", "ppo.epochs=1",
           "ppo.val_every=1", "ppo.val_episodes=1"]
    main(run_args(demos, out, "--stage", "4", "--init", str(m1 / "best.pt"), *sets(*ppo, "bc.epochs=7")))
    (log,) = metrics(out)  # 8 steps // (2 envs * 4 steps) = 1 update
    assert log["env_steps"] == 8 and read_run(out)["ppo"]["num_envs"] == 2
    assert any("--set bc.* has no effect" in w for w in read_run(out)["warnings"])


def test_init_applies_override_and_warns(demos, tiny_hub, tmp_path, capsys):
    """The pitfall: with --init, config.SLM must not silently apply, and --set must."""
    m1 = tmp_path / "m1"
    main(run_args(demos, m1, "--stage", "1", *sets(*tiny_slm_sets(tiny_hub), *QUICK_BC)))
    capsys.readouterr()
    out = tmp_path / "m2"
    main(run_args(demos, out, "--stage", "2", "--init", str(m1 / "best.pt"), *sets("slm.readout_layer=1", *QUICK_BC)))
    printed = capsys.readouterr().out
    assert "warning: config.SLM differs" in printed and "feature_dim" in printed and "readout_layer" not in printed
    run = read_run(out)
    assert run["model"]["cfg"]["readout_layer"] == 1 and run["model"]["cfg"]["feature_dim"] == 16
    assert run["model"]["cfg"]["backbone"] == "slm" and run["warnings"]
    model = load_model(out / "best.pt")
    assert model.cfg.readout_layer == 1 and len(model.backbone.lm.layers) == 1


# --- demos must match the config --------------------------------------------------------


def test_demo_config_mismatch_refuses(demos, tmp_path):
    stale = tmp_path / "stale"
    stale.mkdir()
    for f in demos.iterdir():
        (stale / f.name).write_bytes(f.read_bytes())
    summary = json.loads((stale / "summary.json").read_text())
    summary["config"]["env"]["num_dists"] = 4
    (stale / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit, match=r"env\.num_dists: demos 4, config\.py 8"):
        main(run_args(stale, tmp_path / "run", "--stage", "1"))
    (stale / "summary.json").unlink()
    with pytest.raises(SystemExit, match="re-collect"):
        main(run_args(stale, tmp_path / "run", "--model", "b3"))
    assert not (tmp_path / "run").exists()


# --- B3 ---------------------------------------------------------------------------------


def test_b3_epoch_lowers_loss_and_writes_files(demos, tmp_path):
    out = tmp_path / "b3"
    main(run_args(demos, out, "--model", "b3", "--n-demos", "10", "--id", "B3",
                  *sets("bc.epochs=1", "bc.batch_episodes=2", "bc.lr=1e-3", "bc.val_episodes=1")))
    assert all((out / f).exists() for f in ("best.pt", "last.pt", "run.json", "metrics.jsonl"))
    (log,) = metrics(out)
    assert {"ce", "accuracy", "val_ce", "val_accuracy", "val_success", "val_goto"} <= set(log)
    assert read_run(out)["model"]["type"] == "b3" and read_run(out)["finished"]

    set_seed(0)  # main builds the model right after set_seed(seed), seed 0 by default
    before = SymbolicAgent(SYMBOLIC)
    after = symbolic_agent.load_model(out / "last.pt")
    data = symbolic_demos(load_demo_episodes(demos / "goto_train", 10), before)
    batch = data.batch(np.arange(len(data)), "cpu")
    with torch.no_grad():  # train-mode BatchNorm for both, so only the weights differ
        ce = [symbolic_bc_loss(m.train(), *batch)[1]["ce"] for m in (before, after)]
    assert ce[1] < ce[0]


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


# --- new SLM controls -------------------------------------------------------------------


def env_block(n=2):
    env = make_env("goto", "train")
    obs = [env.reset(seed=s)[0] for s in get_seeds("goto", "train", n)]
    return observations_from_env(obs, [0] * n, [True] * n)


def test_random_lm_regenerates_identically(tiny_hub, tmp_path):
    cfg = replace(SLM, backbone="slm", slm_name=tiny_hub, slm_pretrained=False, slm_init_seed=3, slm_dtype="float32",
                  feature_dim=16, mission_dim=16, word_dim=8, num_conv_layers=1, memory_slots=3, read_heads=2,
                  low_level_layers=2, lora_rank=2, gradient_checkpointing=False)
    model = SLMPolicy(cfg).eval()
    pretrained = SLMPolicy(replace(cfg, slm_pretrained=True)).backbone.lm
    assert not torch.equal(model.backbone.lm.layers[0].mlp.up_proj.weight, pretrained.layers[0].mlp.up_proj.weight)
    save_model(model, tmp_path / "m.pt")
    state = torch.load(tmp_path / "m.pt", weights_only=False)["state_dict"]
    assert not any(k.startswith("backbone.lm.") and "lora_" not in k for k in state), "random LM weights saved"
    loaded = load_model(tmp_path / "m.pt").eval()
    obs = env_block()
    torch.testing.assert_close(model(obs, model.initial_memory(2)).logits, loaded(obs, loaded.initial_memory(2)).logits,
                               rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="did not regenerate"):
        load_model(tmp_path / "m.pt", slm_init_seed=4)


def test_no_direction_by_default_end_to_end(demos, tiny_hub, tmp_path):
    """B3 gets no heading, so by default no SLM policy does either."""
    out = tmp_path / "default"
    main(run_args(demos, out, "--stage", "1", *sets(*tiny_slm_sets(tiny_hub), *QUICK_BC)))
    model = load_model(out / "best.pt").eval()
    assert model.encoder.direction_emb is None and read_run(out)["model"]["cfg"]["use_direction"] is False
    obs = env_block()
    turned = replace(obs, direction=(obs.direction + 1) % 4)
    torch.testing.assert_close(model(obs, model.initial_memory(2)).logits, model(turned, model.initial_memory(2)).logits)
    ep = run_episode(make_env("goto", "val"), SLMAgentPolicy(model), get_seeds("goto", "val", 1)[0])
    assert ep.length >= 1


def test_checkpoint_from_before_use_direction_keeps_heading(tmp_path):
    model = SLMPolicy(replace(SLM, feature_dim=16, mission_dim=16, word_dim=8, memory_slots=2, read_heads=2,
                              readout_hidden=32, use_direction=True))
    save_model(model, tmp_path / "old.pt")
    ckpt = torch.load(tmp_path / "old.pt", weights_only=False)
    del ckpt["config"]["use_direction"]
    torch.save(ckpt, tmp_path / "old.pt")
    assert load_model(tmp_path / "old.pt").encoder.direction_emb is not None


# --- run plan ---------------------------------------------------------------------------


def test_run_plan_dry_run(capsys):
    run_plan.main(["--dry-run", "--set", "bc.epochs=5"])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("python")]
    ids = [m.id for m in run_plan.models(5)]
    assert len(lines) == len(ids) * len(run_plan.TIERS) * len(config.TRAIN.seeds)
    assert "M4" not in ids and "M2-L28" not in ids  # BC only; layer 28 is M2
    m3 = next(line for line in lines if "--id M3 " in line and "seed1" in line and "hard" in line)
    assert "--init outputs/runs/M2/hard/seed1/best.pt" in m3 and "--tasks putnext pickup_then_goto" in m3
    long = next(line for line in lines if "--id M1-long " in line)
    assert "--set bc.epochs=10" in long and "bc.epochs=5" not in long
