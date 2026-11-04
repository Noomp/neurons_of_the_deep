"""Tests for the SLM policy and its training code.

A tiny randomly initialised Qwen3 stands in for the real SLM, so nothing is
downloaded and the tests run on CPU in seconds.
"""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from babytext import ExpertPolicy, get_seeds, make_env, run_episode  # noqa: E402
from babytext.data import save_demos  # noqa: E402
from config import BC, PPO, SLM  # noqa: E402
from models.slm_policy import (  # noqa: E402
    SLMAgentPolicy,
    SLMBackbone,
    SLMPolicy,
    load_model,
    observations_from_env,
    parameter_groups,
    save_model,
)
from training.bc import bc_loss, train_bc  # noqa: E402
from training.demos import collate, load_demo_episodes  # noqa: E402
from training.ppo import collect_rollout, ppo_update  # noqa: E402
from training.rewards import compute_gae, shaped_reward  # noqa: E402
from training.stages import STAGES, make_optimizer  # noqa: E402
from training.vec_env import VecTextEnv  # noqa: E402

SMALL = replace(SLM, feature_dim=16, mission_dim=16, word_dim=8, num_conv_layers=1, memory_slots=3, read_heads=2,
                readout_hidden=32, low_level_layers=2, lora_rank=2, gradient_checkpointing=False)


class CharTokenizer:
    """Stand-in for a Hugging Face tokenizer: one id per character."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + ord(c) % 60 for c in text]}


def tiny_slm_backbone(cfg=SMALL):
    lm_cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    torch.manual_seed(0)
    return SLMBackbone(transformers.Qwen3Model(lm_cfg), CharTokenizer(), cfg.feature_dim, cfg)


def make_model(backbone="readout", cfg=SMALL):
    torch.manual_seed(0)
    cfg = replace(cfg, backbone=backbone)
    return SLMPolicy(cfg, backbone=tiny_slm_backbone(cfg) if backbone == "slm" else None)


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    out = tmp_path_factory.mktemp("demos")
    for task, split in (("goto", "train"), ("pickup", "train"), ("goto", "val")):
        env = make_env(task, split)
        eps = [run_episode(env, ExpertPolicy(), s, task, split) for s in get_seeds(task, split, 8)]
        save_demos(eps, out / f"{task}_{split}")
    return out


def env_block(n=2, task="goto"):
    env = make_env(task, "train")
    obs = [env.reset(seed=s)[0] for s in get_seeds(task, "train", n)]
    return observations_from_env(obs, [0] * n, [True] * n)


# --- model ------------------------------------------------------------------------


@pytest.mark.parametrize("backbone", ["readout", "slm"])
def test_forward_shapes(backbone):
    model = make_model(backbone)
    out = model(env_block(), model.initial_memory(2), return_attention=True)
    k1, p = SMALL.memory_slots + 1, 49
    assert out.logits.shape == (2, 1, 6) and out.value.shape == (2, 1)
    assert out.attention.shape == (2, 1, SMALL.read_heads, k1, p)
    torch.testing.assert_close(out.attention.sum(dim=(-1, -2)), torch.ones(2, 1, SMALL.read_heads))


@pytest.mark.parametrize("backbone", ["readout", "slm"])
def test_sequence_equals_stepwise(backbone, demos):
    """One (B, T) call gives the same outputs as T calls with T = 1, including a
    reset in the middle: PPO relies on this to recompute rollout log-probs."""
    model = make_model(backbone).eval()
    batch = collate(load_demo_episodes(demos / "goto_train", 2))
    obs = batch.obs
    obs.reset[1, 2] = True  # an episode boundary inside a sequence
    full = model(obs, model.initial_memory(2))
    memory, logits = model.initial_memory(2), []
    for t in range(obs.shape[1]):
        step = obs.select([0, 1])
        step.image, step.direction, step.t, step.reset = (x[:, t : t + 1] for x in (obs.image, obs.direction, obs.t, obs.reset))
        step.missions = [[m[t]] for m in obs.missions]
        out = model(step, memory)
        memory = out.memory
        logits.append(out.logits)
    torch.testing.assert_close(full.logits, torch.cat(logits, dim=1), atol=1e-4, rtol=1e-4)


def test_memory_untouched_when_gate_closed():
    model = make_model(cfg=replace(SMALL, write_mode="hard", write_gate_bias=-50.0))
    memory = model.initial_memory(2)
    memory.slots.normal_()
    memory.fill.fill_(1.0)
    obs = env_block()
    obs.reset[:] = False
    out = model(obs, memory)
    torch.testing.assert_close(out.memory.slots, memory.slots)
    assert out.write_gate.sum() == 0


def test_reset_clears_memory():
    model = make_model()
    memory = model.initial_memory(2)
    memory.slots.normal_()
    memory.fill.fill_(1.0)
    cleared = memory.reset(torch.tensor([True, False]))
    assert cleared.slots[0].abs().sum() == 0 and cleared.fill[0].sum() == 0
    torch.testing.assert_close(cleared.slots[1], memory.slots[1])


def test_no_memory_ablation():
    model = make_model(cfg=replace(SMALL, memory_slots=0))
    out = model(env_block(), model.initial_memory(2), return_attention=True)
    assert out.attention.shape[-2] == 1 and torch.isfinite(out.logits).all()


def test_lora_starts_as_identity_and_padding_is_harmless():
    """Zero-initialised LoRA changes nothing, and a step gives the same output
    alone as when batched next to a longer (left-padded) mission."""
    with_lora = make_model("slm").eval()
    without = SLMPolicy(replace(SMALL, backbone="slm"), backbone=tiny_slm_backbone(replace(SMALL, lora_rank=0)))
    without.load_state_dict(with_lora.state_dict(), strict=False)
    without.eval()
    obs = env_block(3)
    obs.missions[1] = ["go to the red ball and also somewhere very far away"]
    a = with_lora(obs, with_lora.initial_memory(3)).logits
    b = without(obs, without.initial_memory(3)).logits
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    alone = with_lora(obs.select([0]), with_lora.initial_memory(1)).logits
    torch.testing.assert_close(alone, a[:1], atol=1e-4, rtol=1e-4)


def test_agent_policy_runs_episode():
    env = make_env("goto", "val")
    ep = run_episode(env, SLMAgentPolicy(make_model()), get_seeds("goto", "val", 1)[0])
    assert 1 <= ep.length <= 64


# --- checkpoints and stages ---------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    model = make_model("slm").eval()
    save_model(model, tmp_path / "m.pt", stage=3)
    state = torch.load(tmp_path / "m.pt", weights_only=False)["state_dict"]
    assert not any(k.startswith("backbone.lm.") and "lora_" not in k for k in state), "pretrained SLM weights saved"
    loaded = load_model(tmp_path / "m.pt", backbone_module=tiny_slm_backbone()).eval()
    obs = env_block()
    torch.testing.assert_close(model(obs, model.initial_memory(2)).logits, loaded(obs, loaded.initial_memory(2)).logits)


def test_stage_switch_keeps_core(tmp_path):
    m1 = make_model("readout")
    save_model(m1, tmp_path / "m1.pt")
    m2 = load_model(tmp_path / "m1.pt", backbone_module=tiny_slm_backbone(), backbone="slm")
    assert m2.cfg.backbone == "slm"
    torch.testing.assert_close(m1.encoder.cell_pos, m2.encoder.cell_pos)
    torch.testing.assert_close(m1.memory.gate.weight, m2.memory.gate.weight)


def test_stage_freezing():
    model = make_model("slm")
    make_optimizer(model, STAGES[2], lr=1e-3)
    groups = parameter_groups(model)
    assert all(not p.requires_grad for p in groups["lora"]) and all(p.requires_grad for p in groups["adapter"])
    assert not any(p.requires_grad for n, p in model.backbone.lm.named_parameters() if "lora_" not in n)
    make_optimizer(model, STAGES[3], lr=1e-3)
    assert all(p.requires_grad for p in groups["lora"]) and not any(p.requires_grad for p in groups["value_head"])
    with pytest.raises(ValueError):
        make_optimizer(make_model("readout"), STAGES[3], lr=1e-3)


# --- training -----------------------------------------------------------------------


def test_gae_and_shaping():
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.5, 0.5]])
    adv, ret = compute_gae(rewards, values, torch.tensor([[False, True]]), torch.tensor([9.0]), gamma=0.9, lam=1.0)
    torch.testing.assert_close(ret, torch.tensor([[0.9, 1.0]]))  # the bootstrap value 9 is cut by done
    assert shaped_reward(0.0, -0.5, -0.25, False, 1.0, 1.0) == pytest.approx(0.25)
    assert shaped_reward(1.0, -0.1, -5.0, True, 1.0, 1.0) == pytest.approx(1.1)  # terminal potential is 0


def test_bc_learns(demos, tmp_path):
    model = make_model()
    episodes = load_demo_episodes(demos / "goto_train") + load_demo_episodes(demos / "pickup_train")
    batch = collate(episodes)
    before = bc_loss(model, batch, 0.0)[1]["ce"]
    cfg = replace(BC, epochs=15, batch_episodes=8, lr=3e-3, val_episodes=0)
    train_bc(model, STAGES[1], episodes, load_demo_episodes(demos / "goto_val"), ["goto"], tmp_path, cfg)
    assert bc_loss(model, batch, 0.0)[1]["ce"] < 0.7 * before
    assert (tmp_path / "best.pt").exists() and (tmp_path / "metrics.jsonl").exists()


def test_vec_env_autoreset():
    env = VecTextEnv(["goto", "pickup"], num_envs=2, shaping=0.1)
    current = env.reset()
    assert current.reset.all() and (current.t == 0).all()
    for _ in range(70):  # longer than the 64-step limit: every env must restart at least once
        current = env.step(np.zeros(2, dtype=int))  # always turn left
        assert ((current.t == 0) == current.reset).all()
    assert env._next_seed["goto"] > env._next_seed["pickup"] - 10  # both tasks drew new seeds


def test_ppo_update_runs():
    model = make_model()
    env = VecTextEnv(["goto"], num_envs=2, shaping=0.1)
    cfg = replace(PPO, rollout_len=8, epochs=1, minibatch_envs=1)
    optimizer = make_optimizer(model, replace(STAGES[4], backbone="readout",
                                              lr_scale={"readout": 1.0, "action_head": 1.0, "value_head": 1.0}), cfg.lr)
    rollout, _, _, stats = collect_rollout(model, env, env.reset(), model.initial_memory(2), cfg.rollout_len, "cpu")
    assert rollout.actions.shape == (2, 8)
    metrics = ppo_update(model, optimizer, rollout, cfg)
    assert np.isfinite(list(metrics.values())).all()
    # First epoch, first minibatch: the policy has not moved yet, so the ratio is 1.
    assert abs(metrics["approx_kl"]) < 0.05
