"""Rendering episodes for inspection, failure analysis and the report.

Every frame shows three views of the same state side by side:

1. the full room, with the agent's 7x7 field of view highlighted and the
   instruction objects outlined in white (first object solid, second dashed,
   e.g. "put the <solid> next to the <dashed>");
2. the agent's egocentric view (what ``obs["image"]`` encodes);
3. the text the language model receives: mission, descriptions and the
   action chosen at this step (optionally the full GLAM prompt).

Episodes are reproduced by replaying ``(task, eval_set, seed, actions)``, so
any saved demonstration or evaluation episode can be rendered after the fact.

Figures are drawn off-screen on standalone matplotlib ``Figure`` objects (not
pyplot), so importing this module never changes your matplotlib backend. To
look at a frame interactively: ``plt.imshow(fig_to_array(draw_frame(frame)))``.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from .constants import ACTION_NAMES
from .splits import make_env

TILE = 32  # pixels per grid cell

# Page tokens (light theme of the project's chart palette).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
SUCCESS = "#008300"
FAILURE = "#e34948"


@dataclass
class Frame:
    t: int
    full: np.ndarray  # full-room render, highlighted field of view
    pov: np.ndarray  # egocentric render
    mission: str
    descriptions: list[str]
    prompt: str
    goal_positions: list[list[tuple[int, int]]]  # per instruction object
    action: int | None  # action taken from this state (None at the final state)
    reward: float  # reward received on entering this state
    done: bool
    success: bool


def outline_goals(ax, goal_positions, linewidth: float = 2.5):
    """Outline instruction objects: first object solid, second dashed."""
    for k, positions in enumerate(goal_positions):
        for x, y in positions:
            ax.add_patch(
                Rectangle(
                    (x * TILE + 1.5, y * TILE + 1.5), TILE - 3, TILE - 3, fill=False,
                    edgecolor="white", linewidth=linewidth, linestyle="-" if k == 0 else (0, (2, 1.5)),
                )
            )


def capture(env, obs, info, action=None, reward=0.0, done=False) -> Frame:
    """Snapshot the current state of a ``TextEnv``."""
    level = env.unwrapped
    return Frame(
        t=level.step_count,
        full=level.get_full_render(highlight=True, tile_size=TILE),
        pov=level.get_pov_render(tile_size=TILE),
        mission=obs["mission"],
        descriptions=list(obs["descriptions"]),
        prompt=obs["prompt"],
        goal_positions=[list(p) for p in info["goal_positions"]],
        action=action,
        reward=reward,
        done=done,
        success=done and reward > 0,
    )


def replay(task: str, eval_set: str, seed: int, actions: list[int]) -> list[Frame]:
    """Re-run an episode and return one frame per state (len(actions) + 1)."""
    env = make_env(task, eval_set)
    obs, info = env.reset(seed=seed)
    frames = []
    reward, done = 0.0, False
    for a in actions:
        frames.append(capture(env, obs, info, action=a, reward=reward))
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        if done:
            break
    frames.append(capture(env, obs, info, action=None, reward=reward, done=done))
    env.close()
    return frames


def _wrap(lines: list[str], width: int) -> str:
    return "\n".join(textwrap.fill(line, width, subsequent_indent="   ") for line in lines)


def draw_frame(frame: Frame, title: str = "", show_prompt: bool = False) -> Figure:
    """Draw one state as a figure (room | agent view | text)."""
    fig = Figure(figsize=(14, 5.6), dpi=100, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 0.9, 1.6], wspace=0.06, left=0.01, right=0.99, top=0.88, bottom=0.02)

    ax_full = fig.add_subplot(grid[0])
    ax_full.imshow(frame.full)
    outline_goals(ax_full, frame.goal_positions)
    ax_full.set_title("Room (field of view highlighted, goal objects outlined)", fontsize=9, color=INK_SECONDARY)

    ax_pov = fig.add_subplot(grid[1])
    ax_pov.imshow(frame.pov)
    ax_pov.set_title("Agent's egocentric view", fontsize=9, color=INK_SECONDARY)

    for ax in (ax_full, ax_pov):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_txt = fig.add_subplot(grid[2])
    ax_txt.axis("off")
    if frame.action is not None:
        status, status_color = f"action -> {ACTION_NAMES[frame.action]}", INK
    elif frame.success:
        status, status_color = f"success (reward {frame.reward:.3f})", SUCCESS
    elif frame.done:
        status, status_color = "failed (time limit)", FAILURE
    else:
        status, status_color = "", INK
    ax_txt.text(0, 1.0, f"Goal: {frame.mission}", fontsize=11, fontweight="bold", color=INK, va="top", transform=ax_txt.transAxes)
    ax_txt.text(0, 0.93, f"Step {frame.t}   {status}", fontsize=10, color=status_color, va="top", transform=ax_txt.transAxes)
    body = frame.prompt if show_prompt else _wrap(frame.descriptions or ["(nothing visible)"], 70)
    if show_prompt:
        body = "\n".join(textwrap.fill(line, 90) for line in body.split("\n"))
    ax_txt.text(
        0, 0.85, body, fontsize=7.5 if show_prompt else 9, color=INK_SECONDARY, va="top", family="monospace", transform=ax_txt.transAxes
    )

    if title:
        fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    return fig


def fig_to_array(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return img


def save_gif(frames: list[Frame], path: str | Path, title: str = "", fps: float = 2.0, show_prompt: bool = False) -> Path:
    """Save an animated GIF of an episode (last frame held for 2 s)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(fig_to_array(draw_frame(f, title, show_prompt))) for f in frames]
    durations = [int(1000 / fps)] * (len(images) - 1) + [2000]
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)
    return path


def save_strip(frames: list[Frame], path: str | Path, title: str = "", max_frames: int = 12, ncols: int = 6) -> Path:
    """Save a compact PNG filmstrip of the room view, one panel per step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) > max_frames:  # keep first/last, subsample the middle
        idx = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames = [frames[i] for i in idx]
    ncols = min(ncols, len(frames))
    nrows = int(np.ceil(len(frames) / ncols))
    fig = Figure(figsize=(2.3 * ncols, 2.6 * nrows + 0.5), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, f in zip(axes.flat, frames):
        ax.imshow(f.full)
        outline_goals(ax, f.goal_positions, linewidth=1.5)
        if f.action is not None:
            label, color = f"t={f.t}: {ACTION_NAMES[f.action]}", INK
        else:
            label, color = (f"t={f.t}: success", SUCCESS) if f.success else (f"t={f.t}: end", FAILURE)
        ax.set_title(label, fontsize=8.5, color=color)
    if title:
        fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    return path
