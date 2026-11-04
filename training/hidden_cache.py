"""Cached hidden states of the frozen LLM on every demonstration step.

The LLM is frozen and each step's prompt is fixed text, so the head's inputs
are computed once (``python -m training.tools.cache_hidden``) and training
needs no LLM pass. Files, per demo dataset ``<task>_<split>``, in
``<DATA.hidden_dir>/<model>-<dtype>/``:

* ``<task>_<split>.last.npy``, ``.mean.npy``: (S, n_layers, D) last-token and
  mean-pooled vectors at every cached layer, one row per demo step (the row
  order of ``babytext.data.load_steps``);
* ``<task>_<split>.seq<l>.npy``: (total tokens, D) every token vector at layer
  ``l``, for attention pooling (only for ``--sequence-layers``), with
  ``.offsets.npy`` (S + 1,) giving each step's token range;
* ``<task>_<split>.meta.json``: the LLM config, layers, step count and the
  SHA-256 of the demo file, so a cache never silently mixes with other demos
  or another LLM.

Vectors are stored as the raw bits of bfloat16 (the LLM's own precision, in
``uint16``): exact, half the size of float32, and free of the overflow
float16 would risk on the LLM's largest activations.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from config import LLMConfig

FILES = ("last", "mean")


class CacheError(ValueError):
    """The cache is missing, or does not match the demos, the LLM or the requested layer."""


def cache_dir(root: str | Path, llm: LLMConfig) -> Path:
    """``<root>/<model>-<dtype>``, e.g. ``data/hidden/Qwen--Qwen3-0.6B-bfloat16``."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", llm.name).strip("-")[-80:]
    return Path(root) / f"{slug}-{llm.dtype}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_bits(x: torch.Tensor) -> np.ndarray:
    """A tensor as the uint16 bits of its bfloat16 value."""
    return x.detach().to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)


def from_bits(bits: np.ndarray) -> torch.Tensor:
    """Inverse of ``to_bits``: a bfloat16 tensor (copied into memory)."""
    return torch.from_numpy(np.array(bits, dtype=np.uint16).view(np.int16)).view(torch.bfloat16)


def read_meta(directory: Path, prefix: str) -> dict:
    path = directory / f"{prefix}.meta.json"
    if not path.exists():
        raise CacheError(f"no hidden-state cache {path}; run `python -m training.tools.cache_hidden`")
    return json.loads(path.read_text(encoding="utf-8"))


def check_cache(directory: Path, prefix: str, demos_dir: str | Path, llm: LLMConfig, layer: int,
                sequences: bool) -> dict:
    """The cache's meta data, after checking it matches the demo file, the LLM and the layer.

    Raises:
        CacheError: with what to run to fix it.
    """
    meta = read_meta(directory, prefix)
    fix = "re-run `python -m training.tools.cache_hidden`"
    demo_file = Path(demos_dir) / f"{prefix}.jsonl"
    if meta["demos_sha256"] != file_sha256(demo_file):
        raise CacheError(f"the cache of {prefix} was built from other demos than {demo_file}; {fix}")
    built, wanted = (meta["llm"]["name"], meta["llm"]["dtype"]), (llm.name, llm.dtype)
    if built != wanted:
        raise CacheError(f"the cache of {prefix} was built with LLM {built}, not {wanted}; {fix}")
    available = meta["sequence_layers"] if sequences else meta["layers"]
    if layer not in available:
        what = "token sequences (for attention pooling)" if sequences else "pooled vectors"
        option = "--sequence-layers" if sequences else "--layers"
        raise CacheError(f"layer {layer} has no cached {what} for {prefix} (cached: {available}); "
                         f"{fix} with {option} including {layer}")
    return meta


def load_pooled(directory: Path, prefix: str, layer: int, pooling: str, meta: dict, rows: int) -> torch.Tensor:
    """(rows, D) bfloat16 vectors of the first ``rows`` steps at ``layer`` ("last" or "mean")."""
    data = np.load(directory / f"{prefix}.{pooling}.npy", mmap_mode="r")
    return from_bits(data[:rows, meta["layers"].index(layer)])


def open_sequences(directory: Path, prefix: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """``(tokens, offsets)``: the memory-mapped (total tokens, D) bits and (S + 1,) step offsets."""
    return (np.load(directory / f"{prefix}.seq{layer}.npy", mmap_mode="r"),
            np.load(directory / f"{prefix}.offsets.npy"))
