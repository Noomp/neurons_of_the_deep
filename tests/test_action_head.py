"""Tests for the frozen LLM wrapper, the hidden-state cache and the action head (P)."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from babytext import get_seeds, make_env, run_episode  # noqa: E402
from babytext.data import load_steps  # noqa: E402
from config import HEAD  # noqa: E402
from models.action_head import ActionHead, HeadPolicy, load_model, load_policy, pool, save_model  # noqa: E402
from models.frozen_lm import FrozenLM  # noqa: E402
from training.hidden_cache import cache_dir, from_bits, load_pooled, open_sequences, read_meta  # noqa: E402

BF16 = dict(atol=1e-4, rtol=8e-3)  # the cache stores bfloat16; the tiny test model computes in float32


def bf16(x):
    return x.to(torch.bfloat16).float()


PROMPTS = ["Goal of the agent: go to the red ball \n Action 0: ", "short", "Observation 0: You see a wall 2 steps left, "]


def test_truncated_lm_matches_full_model_batched(tiny_llm):
    """Layer l of the full model, batched with left padding, equals the last layer of
    the model truncated to l, run alone: the head trains on one and acts on the other."""
    full = FrozenLM.load(tiny_llm)
    batched = full.run(PROMPTS, [2, 4])
    for layer in (2, 4):
        truncated = FrozenLM.load(tiny_llm, num_layers=layer)
        assert truncated.num_layers == layer
        for i, prompt in enumerate(PROMPTS):
            alone = truncated.run([prompt], [layer])
            torch.testing.assert_close(alone.last(layer)[0], batched.last(layer)[i], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(alone.mean(layer)[0], batched.mean(layer)[i], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(alone.sequences(layer)[0], batched.sequences(layer)[i], atol=1e-5, rtol=1e-4)
    # The final layer includes the final norm; a truncated model drops it.
    assert not isinstance(full.model.norm, torch.nn.Identity)
    assert isinstance(FrozenLM.load(tiny_llm, num_layers=2).model.norm, torch.nn.Identity)


def test_cache_rows_match_live_states(tiny_llm, demos, hidden):
    directory = cache_dir(hidden, tiny_llm)
    meta = read_meta(directory, "goto_train")
    prompts = [s["prompt"] for s in load_steps(demos / "goto_train")]
    assert meta["steps"] == len(prompts) and meta["layers"] == [2, 4] and meta["sequence_layers"] == [2]
    live = FrozenLM.load(tiny_llm).run(prompts[:5], [2, 4])
    for layer in (2, 4):
        for pooling in ("last", "mean"):
            cached = load_pooled(directory, "goto_train", layer, pooling, meta, 5).float()
            torch.testing.assert_close(cached, bf16(getattr(live, pooling)(layer)), **BF16)
    tokens, offsets = open_sequences(directory, "goto_train", 2)
    assert len(offsets) == len(prompts) + 1 and offsets[-1] == len(tokens)
    torch.testing.assert_close(from_bits(tokens[offsets[1]:offsets[2]]).float(), bf16(live.sequences(2)[1]), **BF16)


@pytest.mark.parametrize("pooling", ["last", "mean", "attention"])
def test_head_shapes_and_policy(tiny_llm, pooling):
    lm = FrozenLM.load(tiny_llm, num_layers=2)
    head = ActionHead(replace(HEAD, layer=2, pooling=pooling, hidden_layers=2), input_dim=lm.hidden_size)
    features, mask = pool(lm.run(PROMPTS, [2]), 2, pooling)
    assert head(features, mask).shape == (3, 6)
    ep = run_episode(make_env("goto", "val"), HeadPolicy(head, lm), get_seeds("goto", "val", 1)[0])
    assert 1 <= ep.length <= 64


def test_attention_pooling_ignores_padding(tiny_llm):
    head = ActionHead(replace(HEAD, layer=2, pooling="attention"), input_dim=32)
    torch.nn.init.normal_(head.query)
    x, mask = torch.randn(2, 5, 32), torch.tensor([[True] * 3 + [False] * 2] * 2)
    noisy = x.clone()
    noisy[:, 3:] = 100.0
    torch.testing.assert_close(head(x, mask), head(noisy, mask))


def test_head_save_load(tiny_llm, tmp_path):
    head = ActionHead(replace(HEAD, layer=2, pooling="mean"), input_dim=32)
    torch.nn.init.normal_(head.mlp[-1].weight)
    save_model(head, tmp_path / "p.pt", tiny_llm, epoch=1)
    loaded, llm = load_model(tmp_path / "p.pt")
    assert llm == tiny_llm and loaded.cfg == head.cfg
    x = torch.randn(4, 32)
    torch.testing.assert_close(head.eval()(x), loaded(x))
    policy = load_policy(tmp_path / "p.pt")
    assert policy.lm.num_layers == 2  # the layers above the head's are never loaded
    assert isinstance(policy(make_env("goto", "val").reset(seed=get_seeds("goto", "val", 1)[0])[0]), int)


def test_cache_stores_bfloat16_bits_exactly():
    from training.hidden_cache import to_bits

    x = torch.tensor([1e5, -3.25, 0.0, 7e-3], dtype=torch.bfloat16)
    assert torch.equal(from_bits(to_bits(x)), x) and np.isfinite(from_bits(to_bits(x)).float().numpy()).all()
