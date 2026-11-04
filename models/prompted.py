"""B1 and B2: the frozen LLM choosing actions through language, zero-shot.

Both read the GLAM prompt (``obs["prompt"]``), the same text the action head
reads, and pick an action through the *meaning* of the action names, with no
training:

* **B1, generate and map** (``GenerationPolicy``; Huang et al., 2022): the LLM
  writes a short continuation greedily; if it is exactly an action name that
  action is taken, otherwise the action whose name is closest in
  sentence-embedding space. Outputs that are not exactly an action name are
  counted as invalid (``stats``).
* **B2, log-probability scoring** (``ScoringPolicy``; GLAM, TWOSOME): the
  log-probability of each of the six action names as the continuation of the
  prompt, divided by its number of tokens (length normalisation); the best
  scoring action is taken. The prompt runs once and its key/value cache is
  shared by the six continuations.

The prompt ends in ``"Action N: "``; action names are tokenized on their own
and appended to the prompt's tokens, identically for every action.
Settings: ``config.LLM`` (the model) and ``config.PROMPTED``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from babytext.constants import ACTION_NAMES
from config import LLM, PROMPTED, LLMConfig, PromptedConfig

from .frozen_lm import load_causal_lm

__all__ = ["GenerationPolicy", "ScoringPolicy", "SentenceEmbedder"]


class ScoringPolicy:
    """B2: choose the action name the LLM finds most likely after the prompt.

    Args:
        lm: a causal LM (with LM head), e.g. from ``models.frozen_lm.load_causal_lm``.
        tokenizer: its tokenizer.
        length_normalise: divide each action's log-probability by its token count.
        device: where ``lm`` is.

    After each call, ``last_scores`` holds the (6,) scores of that decision.
    """

    def __init__(self, lm, tokenizer, length_normalise: bool = True, device: str | torch.device = "cpu"):
        self.lm, self.tokenizer = lm, tokenizer
        self.length_normalise = length_normalise
        self.device = torch.device(device)
        rows = [tokenizer(name, add_special_tokens=False)["input_ids"] for name in ACTION_NAMES]
        self.action_lengths = torch.tensor([len(r) for r in rows], device=self.device)
        self.action_ids = torch.zeros(len(rows), max(len(r) for r in rows), dtype=torch.long)
        for i, r in enumerate(rows):
            self.action_ids[i, : len(r)] = torch.tensor(r)
        self.action_ids = self.action_ids.to(self.device)
        self.last_scores: torch.Tensor | None = None

    @classmethod
    def load(cls, llm: LLMConfig = LLM, cfg: PromptedConfig = PROMPTED, device: str | torch.device = "cpu") -> ScoringPolicy:
        lm, tokenizer = load_causal_lm(llm, device)
        return cls(lm, tokenizer, cfg.length_normalise, device)

    def reset(self, env=None) -> None:
        pass

    @torch.no_grad()
    def scores(self, prompt: str) -> torch.Tensor:
        """(6,) score of each action name after ``prompt``: summed (or, normalised, mean) token log-probability."""
        ids = torch.tensor([self.tokenizer(prompt, add_special_tokens=False)["input_ids"]], device=self.device)
        out = self.lm(input_ids=ids, use_cache=True)
        n_actions, width = self.action_ids.shape
        logp = F.log_softmax(out.logits[0, -1].float(), dim=-1)[self.action_ids[:, 0]]  # first token of each name
        if width > 1:
            cache = out.past_key_values
            cache.batch_repeat_interleave(n_actions)
            valid = torch.arange(width, device=self.device)[None] < self.action_lengths[:, None]  # (A, W)
            attention = torch.cat([torch.ones(n_actions, ids.shape[1], device=self.device, dtype=torch.long),
                                   valid[:, :-1].long()], dim=1)
            rest = self.lm(input_ids=self.action_ids[:, :-1], past_key_values=cache, attention_mask=attention,
                           use_cache=False)
            token_logp = F.log_softmax(rest.logits.float(), dim=-1).gather(-1, self.action_ids[:, 1:, None])[..., 0]
            logp = logp + (token_logp * valid[:, 1:]).sum(1)
        return logp / self.action_lengths if self.length_normalise else logp

    def __call__(self, obs: dict) -> int:
        self.last_scores = self.scores(obs["prompt"])
        return int(self.last_scores.argmax())


class SentenceEmbedder:
    """Mean-pooled, L2-normalised sentence embeddings from a Hugging Face encoder
    (the standard recipe for ``sentence-transformers/all-MiniLM-L6-v2``)."""

    def __init__(self, name: str, device: str | torch.device = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        """(N, E) unit-length embeddings."""
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        hidden = self.model(**enc).last_hidden_state
        m = enc["attention_mask"][..., None].float()
        return F.normalize((hidden * m).sum(1) / m.sum(1), dim=-1)


class GenerationPolicy:
    """B1: let the LLM write the action, then map the text to the nearest valid action.

    Args:
        lm, tokenizer: a causal LM and its tokenizer.
        embed: texts -> (N, E) unit-length embeddings (a ``SentenceEmbedder``).
        max_new_tokens: length of the greedy continuation.
        device: where ``lm`` is.

    ``stats`` counts ``decisions`` and ``invalid`` (outputs that are not
    exactly an action name); ``last_text`` is the latest generated text.
    """

    def __init__(self, lm, tokenizer, embed: Callable[[list[str]], torch.Tensor], max_new_tokens: int = 8,
                 device: str | torch.device = "cpu"):
        self.lm, self.tokenizer, self.embed = lm, tokenizer, embed
        self.max_new_tokens = max_new_tokens
        self.device = torch.device(device)
        self.action_embeddings = embed(list(ACTION_NAMES))
        self.stats = {"decisions": 0, "invalid": 0}
        self.last_text = ""

    @classmethod
    def load(cls, llm: LLMConfig = LLM, cfg: PromptedConfig = PROMPTED, device: str | torch.device = "cpu") -> GenerationPolicy:
        lm, tokenizer = load_causal_lm(llm, device)
        return cls(lm, tokenizer, SentenceEmbedder(cfg.mapping_model, device), cfg.max_new_tokens, device)

    def reset(self, env=None) -> None:
        pass

    def map_text(self, text: str) -> tuple[int, bool]:
        """``(action, valid)``: the action named by the first line of ``text``, or the nearest one by embedding."""
        cleaned = text.strip().split("\n")[0].strip().strip(".,;:").lower()
        if cleaned in ACTION_NAMES:
            return ACTION_NAMES.index(cleaned), True
        similarity = self.embed([cleaned]) @ self.action_embeddings.T  # (1, 6)
        return int(similarity.argmax()), False

    @torch.no_grad()
    def __call__(self, obs: dict) -> int:
        ids = torch.tensor([self.tokenizer(obs["prompt"], add_special_tokens=False)["input_ids"]], device=self.device)
        pad = next((t for t in (self.tokenizer.pad_token_id, self.tokenizer.eos_token_id) if t is not None), 0)
        out = self.lm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_new_tokens,
                               do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=pad)
        self.last_text = self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        action, valid = self.map_text(self.last_text)
        self.stats["decisions"] += 1
        self.stats["invalid"] += int(not valid)
        return action
