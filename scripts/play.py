"""Play the environment by hand: the room is shown in a window and the text
observation (or full GLAM prompt) is printed to the terminal after each move.

Keys: left/right = turn, up = forward, tab or pageup = pick up,
      left shift or pagedown = drop, space = toggle,
      backspace = restart episode, n = next seed, escape = quit.

Usage:
    python scripts/play.py --task pickup --eval-set test_nonce --prompt
"""

from __future__ import annotations

import argparse

import pygame
from minigrid.manual_control import ManualControl

from babytext import ACTION_TO_IDX, EVAL_SETS, TASKS, get_seeds, make_env

KEY_TO_ACTION = {
    "left": "turn left",
    "right": "turn right",
    "up": "go forward",
    "tab": "pick up",
    "page up": "pick up",
    "left shift": "drop",
    "page down": "drop",
    "space": "toggle",
}


class TextManualControl(ManualControl):
    def __init__(self, env, seeds, show_prompt=False):
        super().__init__(env, seed=seeds[0])
        self.seeds = seeds
        self.seed_idx = 0
        self.show_prompt = show_prompt

    def show(self, obs):
        print("\n" + "=" * 80)
        if self.show_prompt:
            print(obs["prompt"])
        else:
            print(f"Goal: {obs['mission']}  (step {self.env.unwrapped.step_count})")
            for d in obs["descriptions"]:
                print(f"  {d}")

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        print(f"\n### seed {seed}")
        self.show(obs)
        self.env.render()

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            print(f"\n*** {'success' if reward > 0 else 'failure'} (reward {reward:.3f}) ***")
            self.next_seed()
        else:
            self.show(obs)
            self.env.render()

    def next_seed(self):
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        self.seed = self.seeds[self.seed_idx]
        self.reset(self.seed)

    def key_handler(self, event):
        key = event.key
        if key == "escape":
            self.env.close()
            self.closed = True
        elif key == "backspace":
            self.reset(self.seed)
        elif key == "n":
            self.next_seed()
        elif key in KEY_TO_ACTION:
            self.step(ACTION_TO_IDX[KEY_TO_ACTION[key]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="goto", choices=TASKS)
    parser.add_argument("--eval-set", default="train", choices=list(EVAL_SETS))
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--prompt", action="store_true", help="print the full GLAM prompt")
    args = parser.parse_args()

    env = make_env(args.task, args.eval_set, render_mode="human")
    seeds = get_seeds(args.task, args.eval_set, 1000)[args.seed_index :]
    pygame.init()
    TextManualControl(env, seeds, show_prompt=args.prompt).start()


if __name__ == "__main__":
    main()
