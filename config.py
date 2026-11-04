"""Project configuration: every experimental setting in one place.

Edit the defaults below to change a setting project-wide. ``constants.py``
holds what is *not* a choice (MiniGrid's action indices, the task list, the
object vocabulary); everything here is.

Settings are grouped into frozen dataclasses, each with one module-level
instance that the code reads:

* ``ENV``: level generation and the text interface;
* ``EVAL_SETS``: the train / val / test sets and the three shifted test sets;
* ``DATA``: demonstration counts and file locations;
* ``EVAL``: evaluation sizes;
* ``TRAIN``: the training protocol from the proposal;
* ``SYMBOLIC``: sizes of the B3 symbolic agent;
* ``LLM``: the frozen LLM shared by B1, B2 and the action head;
* ``HEAD``: the action head (P): read-out layer, pooling, MLP size;
* ``PROMPTED``: the zero-shot baselines B1 and B2;
* ``BC``: behaviour cloning settings, shared by B3 and the action head.

Changing ``ENV`` or ``EVAL_SETS`` changes the generated episodes: re-run
``python -m training.tools.collect_demos`` afterwards; changing ``LLM`` (or
``HEAD.sweep_layers``) changes the hidden-state cache: re-run
``python -m training.tools.cache_hidden``. ``training.tools.collect_demos`` stores ``as_dict()`` next to the demos so every dataset
records the settings that produced it.

One-off overrides do not need edits here, e.g.
``make_env("goto", "train", num_dists=4)``. Training runs change ``HEAD``,
``BC``, ``SYMBOLIC`` and ``TRAIN`` per run with
``python -m training.train --set section.field=value`` (see training/README.md):
the instances below are frozen and imported at load time, so reassigning
them at run time does not reach the code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EnvConfig:
    """Level generation and text interface (defaults follow GLAM / BabyAI-Text)."""

    # One room_size x room_size room including walls (8 -> 6x6 interior).
    room_size: int = 8
    # Random distractors added besides the 1-2 objects named in the instruction.
    num_dists: int = 8
    # Side of the agent's square egocentric view (odd, >= 3).
    agent_view_size: int = 7
    # Fixed episode time limit; None uses BabyAI's room_size**2 per navigation
    # the instruction needs (64 for goto/pickup, 128 for the two-object tasks).
    max_steps: int | None = None

    # Objects never named in training instructions (GLAM's choice; they may
    # still appear as distractors). In test_heldout exactly one named object
    # of each instruction is held out.
    held_out_goals: frozenset[tuple[str, str]] = frozenset({("box", "yellow"), ("key", "red"), ("ball", "green")})

    # Word substitution for test_nonce. Words must be unique and none may be
    # a real colour or object word.
    nonce_words: dict[str, str] = field(
        default_factory=lambda: {
            # colours
            "red": "dueno",
            "green": "hfao",
            "blue": "noomp",
            "purple": "clet",
            "yellow": "toma",
            "grey": "zup",
            # object types
            "key": "glorp",
            "ball": "tism",
            "box": "marl",
        }
    )

    # GLAM prompt history: the last N observations and the N-1 actions between them.
    prompt_num_obs: int = 3

    def level_kwargs(self) -> dict:
        """Settings passed to the level generator (``levels.LocalLevel``)."""
        kwargs = {"room_size": self.room_size, "num_dists": self.num_dists, "agent_view_size": self.agent_view_size}
        if self.max_steps is not None:
            kwargs["max_steps"] = self.max_steps
        return kwargs


ENV = EnvConfig()


@dataclass(frozen=True)
class EvalSet:
    """One evaluation set: which seed range it draws from and how its levels
    differ from ``ENV``.

    ``level`` overrides any ``EnvConfig.level_kwargs()`` entry, plus
    ``goal_pool`` ("train", "heldout" or "any"; default "train").
    """

    seed_split: str  # "train", "val" or "test"
    level: dict = field(default_factory=dict)
    nonce: bool = False  # rename colours and objects with ENV.nonce_words

    def level_kwargs(self) -> dict:
        return {**ENV.level_kwargs(), "goal_pool": "train", **self.level}


EVAL_SETS: dict[str, EvalSet] = {
    "train": EvalSet("train"),
    "val": EvalSet("val"),
    "test": EvalSet("test"),
    "test_dist16": EvalSet("test", level={"num_dists": 16}),
    "test_nonce": EvalSet("test", nonce=True),
    "test_heldout": EvalSet("test", level={"goal_pool": "heldout"}),
}


@dataclass(frozen=True)
class DataConfig:
    # Expert demonstrations per task (the first N seeds of each split).
    n_train_demos: int = 2000
    n_val_demos: int = 200
    demos_dir: Path = Path("data/demos")
    hidden_dir: Path = Path("data/hidden")  # cached LLM hidden states of the demos
    outputs_dir: Path = Path("outputs")


DATA = DataConfig()


@dataclass(frozen=True)
class EvalConfig:
    # Episodes per (task, eval set) for final evaluation and reference baselines.
    n_episodes: int = 500
    # Expert episodes per (task, eval set) for the dataset statistics charts.
    n_stats_episodes: int = 200


EVAL = EvalConfig()


@dataclass(frozen=True)
class TrainConfig:
    """Training protocol shared by every trained condition (proposal 6.4-6.5).
    Optimiser settings live in ``BC``; model sizes in ``HEAD`` and ``SYMBOLIC``."""

    # 6.5: at least 3 training seeds per configuration, reported as mean with 95% CI.
    seeds: tuple[int, ...] = (0, 1, 2)
    # 6.4: sample-efficiency curves; each count uses the first k demo episodes.
    demo_counts: tuple[int, ...] = (100, 250, 500, 1000, 2000)
    # 6.4: DAgger only if behaviour cloning drifts (0 = off). Every trained
    # condition, including the symbolic baseline, must use the same values.
    dagger_rounds: int = 0
    dagger_episodes_per_round: int = 500


TRAIN = TrainConfig()


@dataclass(frozen=True)
class SymbolicAgentConfig:
    """Sizes of the B3 symbolic agent (models/symbolic_agent.py).

    Its history length is not set here: it is ENV.prompt_num_obs, so the agent
    always sees the same window as the LLM prompt.
    """

    cell_dim: int = 16  # embedding size of each grid cell (object + colour + state)
    conv_channels: int = 64
    num_conv_layers: int = 2  # 3x3 convolutions, each conditioned on the mission (FiLM)
    word_dim: int = 32  # mission word embedding size
    mission_dim: int = 128  # mission GRU size
    action_dim: int = 8  # embedding size of each previous action
    hidden_dim: int = 128  # MLP before the 6 action logits


SYMBOLIC = SymbolicAgentConfig()


@dataclass(frozen=True)
class LLMConfig:
    """The frozen LLM shared by every LLM-based condition (B1, B2 and the action head P).

    Changing it invalidates the hidden-state cache: re-run
    ``python -m training.tools.cache_hidden`` afterwards.
    """

    name: str = "Qwen/Qwen3-0.6B"  # Hugging Face id or local folder; 28 layers, width 1024
    dtype: str = "bfloat16"
    cache_batch: int = 32  # prompts per forward pass when caching hidden states


LLM = LLMConfig()


@dataclass(frozen=True)
class HeadConfig:
    """The action head (P): one pooled hidden state of the frozen LLM -> MLP -> 6 action logits."""

    # Read the output of this LLM layer (1 .. 28; 28 is the final layer, after the final norm).
    layer: int = 28
    # How the ~300 token vectors of that layer become one: "last" token, "mean"
    # over tokens, or learned "attention" (needs cached token sequences).
    pooling: str = "last"
    hidden_layers: int = 1  # MLP hidden layers (1 or 2)
    hidden_dim: int = 256
    attention_dim: int = 64  # key size of the learned attention pooling
    # Layers whose last-token and mean vectors are cached and swept (RQ2):
    # every 4th layer and the final one.
    sweep_layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28)


HEAD = HeadConfig()


@dataclass(frozen=True)
class PromptedConfig:
    """The zero-shot baselines B1 (generate and map) and B2 (log-probability scoring)."""

    max_new_tokens: int = 8  # B1: greedy generation length
    # B1: sentence-embedding model that maps generated text to the nearest action (Huang et al., 2022).
    mapping_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # B2: score each action by its mean token log-probability (length normalisation, as in TWOSOME).
    length_normalise: bool = True


PROMPTED = PromptedConfig()


@dataclass(frozen=True)
class BCConfig:
    """Behaviour cloning, shared by every trained condition (B3 and the action head).

    One batch is every step of ``batch_episodes`` demo episodes. Each run keeps
    the epoch with the lowest val cross-entropy (on the val demos) and measures
    greedy val success once on it; configurations are compared by that success.
    Training stops early once val cross-entropy has stopped improving, and the
    learning rate decays when it stalls, so ``epochs`` is only an upper bound.
    """

    epochs: int = 200  # upper bound; early stopping ends runs after ~15-40 epochs
    batch_episodes: int = 32  # episodes per batch; the batch holds all their steps
    # Initial learning rate. 1e-2 gave the lowest (or tied lowest) val cross-entropy for
    # both B3 and the head (goto, 4 Nov; 3e-4 under-trained both); see training/README.md.
    lr: float = 1e-2
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # Learning-rate decay: "plateau" multiplies the lr by lr_factor after lr_patience
    # epochs without a lower val cross-entropy; "cosine" anneals it to min_lr over
    # ``epochs``; "none" keeps it constant.
    lr_decay: str = "plateau"
    lr_factor: float = 0.5
    lr_patience: int = 2
    min_lr: float = 1e-6
    # Early stopping: stop after this many epochs without a lower val cross-entropy (0 = never).
    patience: int = 6
    # Greedy success on this many val episodes per task, measured once on the selected epoch (0 = skip).
    val_episodes: int = 100


BC = BCConfig()


def as_dict() -> dict:
    """All settings as plain JSON-serialisable data (for logging next to results)."""

    def clean(value):
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, (frozenset, set)):
            return sorted(clean(v) for v in value)
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value

    return clean(
        {
            "env": asdict(ENV),
            "eval_sets": {name: asdict(s) for name, s in EVAL_SETS.items()},
            "data": asdict(DATA),
            "eval": asdict(EVAL),
            "train": asdict(TRAIN),
            "symbolic": asdict(SYMBOLIC),
            "llm": asdict(LLM),
            "head": asdict(HEAD),
            "prompted": asdict(PROMPTED),
            "bc": asdict(BC),
        }
    )
