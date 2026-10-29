"""B3: the non-LLM symbolic agent (proposal section 6.3).

A small frame-stacked CNN trained on MiniGrid's symbolic grid observation with
the same demonstrations, and the same behaviour cloning / DAgger procedure, as
the frozen-LLM action heads. Its inputs match what the LLM prompt contains:

* the last N egocentric 7x7 views, oldest first (N = config.ENV.prompt_num_obs,
  the prompt's observation window), stacked along the channel axis;
* the N - 1 actions taken between those views;
* the mission, read word by word with a fixed vocabulary, so invented names in
  test_nonce become unknown tokens (the grid itself is never renamed).

It deliberately does *not* get ``obs["direction"]``: the prompt only gives
egocentric positions ("2 steps left"), never the compass direction.

Architecture: per-cell embeddings -> 3x3 convolutions, each modulated by the
mission through FiLM (as in the BabyAI baselines) -> MLP together with the
mission and previous-action embeddings -> 6 action logits.

Inputs are built by ``stack_demo_steps`` (from saved demonstrations) or
``SymbolicHistory`` (during rollouts); both produce identical tensors for the
same state. ``SymbolicPolicy`` wraps a trained model for ``babytext.evaluate``.
"""

from __future__ import annotations

import re
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from torch.nn.utils.rnn import pack_padded_sequence

from babytext.constants import COLORS, NUM_ACTIONS, OBJ_TYPES
from config import ENV, SYMBOLIC, SymbolicAgentConfig

__all__ = ["NO_ACTION", "MissionTokenizer", "SymbolicAgent", "SymbolicHistory", "SymbolicPolicy", "stack_demo_steps"]

NO_ACTION = NUM_ACTIONS # represents `no-action`, used as `previous-action` at the start of episodes


# --- mission tokenizer ------------------------------------------------------------


class MissionTokenizer:
    """Word-level tokenizer with a fixed vocabulary covering every mission the
    environment generates. Unknown words (e.g. invented names) map to <unk>."""

    PAD, UNK = 0, 1
    WORDS = ("go", "to", "the", "a", "pick", "up", "put", "next", "then", *COLORS, *OBJ_TYPES)

    def __init__(self):
        self.vocab = {"<pad>": self.PAD, "<unk>": self.UNK}
        for word in self.WORDS:
            self.vocab.setdefault(word, len(self.vocab))

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, missions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn a batch of mission strings into word-ID tensors for the model.

        Each mission is lowercased and split into words (punctuation is
        dropped), and each word is replaced by its vocabulary ID (<unk> = 1 if
        unknown). Shorter missions are filled up with <pad> = 0 at the end so
        all rows have the same length.

        Returns:
            tokens: (batch, longest mission) word IDs.
            lengths: (batch,) number of real words per mission, so the model
                can skip the padding.
        """
        ids = [[self.vocab.get(w, self.UNK) for w in re.findall(r"[a-z]+", m.lower())] for m in missions]
        lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
        tokens = torch.full((len(ids), int(lengths.max())), self.PAD, dtype=torch.long)
        for i, x in enumerate(ids):
            tokens[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        return tokens, lengths


# --- model -------------------------------------------------------------------------


class FiLM(nn.Module):
    """Feature-wise linear modulation: scale and shift each channel from the mission."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)  # start as the identity transform
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + gamma) + beta


class SymbolicAgent(nn.Module):
    """Frame-stacked CNN policy over symbolic observations (B3)."""

    def __init__(self, cfg: SymbolicAgentConfig = SYMBOLIC, num_frames: int | None = None, view_size: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.num_frames = ENV.prompt_num_obs if num_frames is None else num_frames
        self.view_size = ENV.agent_view_size if view_size is None else view_size
        self.tokenizer = MissionTokenizer()

        # Each cell is (object, colour, state); the three embeddings are summed.
        self.obj_emb = nn.Embedding(num_embeddings = len(OBJECT_TO_IDX), embedding_dim = cfg.cell_dim)
        self.color_emb = nn.Embedding(num_embeddings = len(COLOR_TO_IDX), embedding_dim = cfg.cell_dim)
        self.state_emb = nn.Embedding(num_embeddings = 3, embedding_dim = cfg.cell_dim)

        self.word_emb = nn.Embedding(num_embeddings = len(self.tokenizer), 
                                     embedding_dim = cfg.word_dim, padding_idx=MissionTokenizer.PAD)
        self.mission_rnn = nn.GRU(cfg.word_dim, cfg.mission_dim, batch_first=True)

        self.action_emb = nn.Embedding(NUM_ACTIONS + 1, cfg.action_dim)  # + NO_ACTION

        # Input channels: cell embeddings of every frame + one "frame present" channel per frame.
        in_ch = self.num_frames * (cfg.cell_dim + 1)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.films = nn.ModuleList()
        for i in range(cfg.num_conv_layers):
            self.convs.append(nn.Conv2d(in_ch if i == 0 else cfg.conv_channels, cfg.conv_channels, 3, padding=1))
            self.norms.append(nn.BatchNorm2d(cfg.conv_channels))
            self.films.append(FiLM(cfg.mission_dim, cfg.conv_channels))

        # Flatten (not pool): where an object sits relative to the agent matters.
        flat = cfg.conv_channels * self.view_size**2 + cfg.mission_dim + (self.num_frames - 1) * cfg.action_dim
        self.head = nn.Sequential(nn.Linear(flat, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, NUM_ACTIONS))

    def encode_mission(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.word_emb(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.mission_rnn(packed)
        return h[-1]  # (B, mission_dim)

    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        prev_actions: torch.Tensor,
        mission_tokens: torch.Tensor,
        mission_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Action logits (B, 6).

        Args:
            frames: (B, N, V, V, 3) integer grids, oldest first; missing frames are zeros.
            frame_mask: (B, N) bool, False for frames before the episode start.
            prev_actions: (B, N - 1) long, NO_ACTION where the episode was too short.
            mission_tokens, mission_lengths: from ``self.tokenizer.encode``.
        """
        frames = frames.long()
        cells = self.obj_emb(frames[..., 0]) + self.color_emb(frames[..., 1]) + self.state_emb(frames[..., 2])
        present = frame_mask.to(cells.dtype)[:, :, None, None, None]
        cells = cells * present  # (B, N, V, V, D), zero for missing frames
        b, n, v, _, d = cells.shape
        x = cells.permute(0, 1, 4, 2, 3).reshape(b, n * d, v, v)
        x = torch.cat([x, present[..., 0].expand(b, n, v, v)], dim=1)

        mission = self.encode_mission(mission_tokens, mission_lengths)
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            x = torch.relu(film(norm(conv(x)), mission))

        actions = self.action_emb(prev_actions).flatten(1)
        return self.head(torch.cat([x.flatten(1), mission, actions], dim=1))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- inputs from saved demonstrations ----------------------------------------------


def stack_demo_steps(
    image: np.ndarray, action: np.ndarray, t: np.ndarray, num_frames: int | None = None
) -> dict[str, np.ndarray]:
    """Build the model's frame-stacked inputs for every step of a demo file.

    For each step, collects the last N views and the N - 1 actions between
    them, exactly as ``SymbolicHistory`` does during a rollout. Steps near the
    start of an episode have fewer than N views: the missing slots are zeros,
    marked False in ``frame_mask``, and their actions are ``NO_ACTION``.

    Args:
        image, action, t: the arrays of the same names returned by
            ``babytext.data.load_symbolic(prefix)``.
        num_frames: N; defaults to ``ENV.prompt_num_obs`` (the prompt window).

    Returns:
        A dict with one row per step (S steps in total), oldest view first:
            frames: (S, N, 7, 7, 3) uint8 views.
            frame_mask: (S, N) bool, False where no view exists yet.
            prev_actions: (S, N - 1) int64 actions between the views.

    Row i matches row i of ``load_symbolic`` and step i of ``load_steps``,
    which is where the missions come from::

        arrays = load_symbolic("data/demos/goto_train")
        inputs = stack_demo_steps(arrays["image"], arrays["action"], arrays["t"])
        missions = [s["mission"] for s in load_steps("data/demos/goto_train")]
    """
    n = ENV.prompt_num_obs if num_frames is None else num_frames
    rows = np.arange(len(t))
    frames = np.zeros((len(t), n, *image.shape[1:]), dtype=np.uint8)
    frame_mask = np.zeros((len(t), n), dtype=bool)
    prev_actions = np.full((len(t), n - 1), NO_ACTION, dtype=np.int64)
    for k in range(n):  # slot k holds the view from (n - 1 - k) steps ago
        back = n - 1 - k
        valid = t >= back  # rows are consecutive within an episode
        frames[valid, k] = image[rows[valid] - back]
        frame_mask[:, k] = valid
    for j in range(n - 1):  # slot j holds the action taken (n - 1 - j) steps ago
        back = n - 1 - j
        valid = t >= back
        prev_actions[valid, j] = action[rows[valid] - back]
    return {"frames": frames, "frame_mask": frame_mask, "prev_actions": prev_actions}


# --- inputs during rollouts --------------------------------------------------------


class SymbolicHistory:
    """Rolling window of views and actions; the symbolic twin of PromptHistory."""

    def __init__(self, num_frames: int | None = None):
        self.num_frames = ENV.prompt_num_obs if num_frames is None else num_frames
        self.reset()

    def reset(self):
        self.frames: deque[np.ndarray] = deque(maxlen=self.num_frames)
        self.actions: deque[int] = deque(maxlen=self.num_frames - 1)

    def add_observation(self, image: np.ndarray):
        self.frames.append(np.asarray(image, dtype=np.uint8))

    def add_action(self, action: int):
        self.actions.append(int(action))

    def inputs(self) -> dict[str, np.ndarray]:
        """The same arrays as one row of ``stack_demo_steps``, with a batch dim of 1."""
        n = self.num_frames
        frames = np.zeros((1, n, *self.frames[-1].shape), dtype=np.uint8)
        frame_mask = np.zeros((1, n), dtype=bool)
        for k, img in enumerate(self.frames, start=n - len(self.frames)):  # right-align, oldest first
            frames[0, k] = img
            frame_mask[0, k] = True
        prev_actions = np.full((1, n - 1), NO_ACTION, dtype=np.int64)
        for j, a in enumerate(self.actions, start=n - 1 - len(self.actions)):
            prev_actions[0, j] = a
        return {"frames": frames, "frame_mask": frame_mask, "prev_actions": prev_actions}


class SymbolicPolicy:
    """Greedy policy around a trained ``SymbolicAgent``, usable with
    ``babytext.evaluate`` / ``run_episode`` (callable ``obs -> action``)."""

    def __init__(self, model: SymbolicAgent, device: str | torch.device = "cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.history = SymbolicHistory(model.num_frames)
        self._mission = None

    def reset(self, env=None):
        self.history.reset()
        self._mission = None

    @torch.no_grad()
    def __call__(self, obs: dict) -> int:
        if self._mission is None:
            tokens, lengths = self.model.tokenizer.encode([obs["mission"]])
            self._mission = (tokens.to(self.device), lengths)
        self.history.add_observation(obs["image"])
        inputs = {k: torch.from_numpy(v).to(self.device) for k, v in self.history.inputs().items()}
        logits = self.model(**inputs, mission_tokens=self._mission[0], mission_lengths=self._mission[1])
        action = int(logits.argmax(dim=-1))
        self.history.add_action(action)  # the policy's own action is the one executed
        return action
