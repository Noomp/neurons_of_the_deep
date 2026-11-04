"""Tests for the zero-shot baselines B1 (generate and map) and B2 (log-probability scoring)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from babytext import evaluate, get_seeds, make_env  # noqa: E402
from babytext.constants import ACTION_NAMES  # noqa: E402
from models.frozen_lm import load_causal_lm  # noqa: E402
from models.prompted import GenerationPolicy, ScoringPolicy  # noqa: E402


def first_prompt():
    return make_env("goto", "val").reset(seed=get_seeds("goto", "val", 1)[0])[0]["prompt"]


@pytest.mark.parametrize("normalise", [True, False])
def test_scoring_with_shared_cache_equals_full_sequences(tiny_llm, normalise):
    """B2 runs the prompt once and reuses its cache; the scores must equal scoring
    each prompt + action name as a separate full sequence."""
    lm, tok = load_causal_lm(tiny_llm)
    policy = ScoringPolicy(lm, tok, length_normalise=normalise)
    prompt = first_prompt()
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    expected = []
    for name in ACTION_NAMES:
        action_ids = tok(name, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            logp = torch.log_softmax(lm(input_ids=torch.tensor([prompt_ids + action_ids])).logits[0].float(), -1)
        total = sum(logp[len(prompt_ids) - 1 + j, t] for j, t in enumerate(action_ids))
        expected.append(total / len(action_ids) if normalise else total)
    torch.testing.assert_close(policy.scores(prompt), torch.stack(expected), atol=1e-4, rtol=1e-4)
    assert policy({"prompt": prompt}) == int(torch.stack(expected).argmax())


def test_scoring_policy_runs_in_evaluate(tiny_llm):
    lm, tok = load_causal_lm(tiny_llm)
    assert evaluate(ScoringPolicy(lm, tok), "goto", "val", 1)["n_episodes"] == 1


def char_embed(texts):
    """Stand-in sentence embedder: normalised letter counts."""
    vectors = torch.zeros(len(texts), 26)
    for i, text in enumerate(texts):
        for c in text.lower():
            if "a" <= c <= "z":
                vectors[i, ord(c) - 97] += 1
    return torch.nn.functional.normalize(vectors + 1e-6, dim=-1)


def test_generation_maps_text_to_actions(tiny_llm):
    lm, tok = load_causal_lm(tiny_llm)
    policy = GenerationPolicy(lm, tok, char_embed, max_new_tokens=3)
    assert policy.map_text(" go forward\nAction 3: turn left") == (2, True)
    assert policy.map_text("Pick up.") == (3, True)
    action, valid = policy.map_text("toggles the door")
    assert not valid and action == ACTION_NAMES.index("toggle")
    policy({"prompt": first_prompt()})
    assert policy.stats["decisions"] == 1 and isinstance(policy.last_text, str)
