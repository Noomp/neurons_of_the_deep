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
* ``SLM``: sizes of the SLM policy (encoder, memory, attention, backbone);
* ``BC``: behaviour cloning settings (training stages 1-3);
* ``PPO``: reinforcement learning settings (training stage 4).

Changing ``ENV`` or ``EVAL_SETS`` changes the generated episodes: re-run
``python -m training.tools.collect_demos`` (and any hidden-state caching) afterwards.
``training.tools.collect_demos`` stores ``as_dict()`` next to the demos so every dataset
records the settings that produced it.

One-off overrides do not need edits here, e.g.
``make_env("goto", "train", num_dists=4)``.
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
    Optimiser settings live in ``BC`` and ``PPO``; model sizes in ``SLM``."""

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
class SLMPolicyConfig:
    """Sizes of the SLM policy (models/slm_policy). See models/README.md.

    The policy is: spatial encoder -> write-sparse slot memory -> space-time
    attention -> backbone (a small MLP in stage 1, the SLM from stage 2) ->
    action and value heads.
    """

    # Spatial encoder: per-cell embedding, then 3x3 convolutions.
    feature_dim: int = 64  # d, features per grid cell
    num_conv_layers: int = 3

    # Mission encoder (word GRU, independent of the SLM so the memory path never runs it).
    word_dim: int = 32
    mission_dim: int = 128

    # Write-sparse slot memory (the "modified GRU"; report section 4).
    memory_slots: int = 8  # K; 0 disables the memory (ablation)
    # Initial bias of the write gate: sigmoid(-2) ~ 0.12, i.e. "do not save" by default.
    write_gate_bias: float = -2.0
    # "soft": g in (0, 1), fully differentiable. "hard": 0/1 forward with a
    # straight-through gradient (Skip-RNN style), so memory really stays untouched.
    write_mode: str = "soft"

    # Space-time attention: H read heads over (K + 1) x 49 (time, cell) positions.
    read_heads: int = 4

    # Backbone. "readout": small MLP on the read-outs (stage 1, model M1).
    # "slm": Hugging Face causal LM reading the mission text + H soft tokens.
    backbone: str = "readout"
    readout_hidden: int = 256
    slm_name: str = "Qwen/Qwen3-0.6B"
    slm_dtype: str = "bfloat16"
    # Layers 0 .. low_level_layers-1 get LoRA from stage 3; the rest stay frozen.
    low_level_layers: int = 10
    lora_rank: int = 8
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    # Read the action from the output of this many layers (None = all). Layers
    # above it are dropped, so a smaller value is also cheaper.
    readout_layer: int | None = None
    gradient_checkpointing: bool = True


SLM = SLMPolicyConfig()


@dataclass(frozen=True)
class BCConfig:
    """Behaviour cloning on whole demonstration episodes (training stages 1-3)."""

    epochs: int = 20
    batch_episodes: int = 32  # episodes per batch (full back-propagation through time)
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # Penalty on the mean write gate: makes memory writes rare.
    write_penalty: float = 1e-2
    # Validation: success rate on this many val episodes after each epoch (0 = loss only).
    val_episodes: int = 100


BC = BCConfig()


@dataclass(frozen=True)
class PPOConfig:
    """Recurrent PPO in the environment (training stage 4)."""

    total_steps: int = 200_000  # environment steps over all parallel envs
    num_envs: int = 16
    rollout_len: int = 128  # steps per env per update; one recurrent sequence per env
    epochs: int = 4
    minibatch_envs: int = 4  # sequences per minibatch
    lr: float = 1e-4
    grad_clip: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    write_penalty: float = 1e-2
    # Potential-based shaping weight (0 = environment reward only).
    shaping: float = 0.1
    val_every: int = 10  # updates between validation runs
    val_episodes: int = 100


PPO = PPOConfig()


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
            "slm": asdict(SLM),
            "bc": asdict(BC),
            "ppo": asdict(PPO),
        }
    )
