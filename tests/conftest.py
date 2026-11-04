"""Shared test fixtures: a tiny random Qwen3 saved as a local "hub" folder, small demos, and their cache.

The tiny model (4 layers, width 32, byte-level tokenizer) goes through the real
``from_pretrained`` code paths, so nothing is downloaded and the tests run on
CPU in seconds.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

import config  # noqa: E402
from babytext import ExpertPolicy, get_seeds, make_env, run_episode  # noqa: E402
from babytext.data import save_demos  # noqa: E402
from config import LLMConfig  # noqa: E402

LM_CONFIG = dict(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                 num_attention_heads=4, num_key_value_heads=2, head_dim=8)
CACHED_LAYERS, SEQUENCE_LAYERS = [2, 4], [2]


@pytest.fixture(scope="session")
def tiny_hub(tmp_path_factory) -> str:
    """A folder that ``AutoModel``, ``AutoModelForCausalLM`` and ``AutoTokenizer`` load."""
    path = tmp_path_factory.mktemp("tiny_qwen3")
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**LM_CONFIG)).save_pretrained(path)
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tok = tokenizers.Tokenizer(tokenizers.models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    transformers.PreTrainedTokenizerFast(tokenizer_object=tok).save_pretrained(path)
    return path.as_posix()


@pytest.fixture(scope="session")
def tiny_llm(tiny_hub) -> LLMConfig:
    return LLMConfig(name=tiny_hub, dtype="float32", cache_batch=4)


@pytest.fixture(scope="session")
def llm_sets(tiny_llm) -> list[str]:
    """``--set`` arguments that point a command at the tiny LLM."""
    return ["--set", f"llm.name={tiny_llm.name}", "--set", "llm.dtype=float32", "--set", "llm.cache_batch=4"]


@pytest.fixture(scope="session")
def demos(tmp_path_factory) -> Path:
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


@pytest.fixture(scope="session")
def hidden(demos, llm_sets, tmp_path_factory) -> Path:
    """The tiny LLM's cache of ``demos``: pooled vectors at layers 2 and 4, token sequences at layer 2."""
    from training.tools import cache_hidden

    out = tmp_path_factory.mktemp("hidden")
    cache_hidden.main(["--tasks", "goto", "--demos", str(demos), "--out", str(out), "--device", "cpu",
                       "--layers", *map(str, CACHED_LAYERS), "--sequence-layers", *map(str, SEQUENCE_LAYERS), *llm_sets])
    return out
