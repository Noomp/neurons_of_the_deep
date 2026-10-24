"""Render an episode as an animated GIF and a PNG filmstrip.

The episode is either rolled out fresh with a reference policy, or replayed
from a saved demonstration file.

Usage:
    # fresh rollout with the expert on the 4th test seed of "pick up"
    python scripts/visualise_episode.py --task pickup --eval-set test --seed-index 3

    # random agent on the 16-distractor shift, with the full GLAM prompt shown
    python scripts/visualise_episode.py --task goto --eval-set test_dist16 --policy random --show-prompt

    # replay episode 10 of a saved demonstration file
    python scripts/visualise_episode.py --demos data/demos/goto_train --episode 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from babytext import TASKS, EVAL_SETS, ExpertPolicy, RandomPolicy, get_seeds, make_env, run_episode
from babytext.data import load_episodes
from babytext.viz import replay, save_gif, save_strip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="goto", choices=TASKS)
    parser.add_argument("--eval-set", default="train", choices=list(EVAL_SETS))
    parser.add_argument("--seed-index", type=int, default=0, help="index into the eval set's seed list")
    parser.add_argument("--policy", default="expert", choices=["expert", "random"])
    parser.add_argument("--demos", type=Path, help="dataset prefix to replay from, e.g. data/demos/goto_train")
    parser.add_argument("--episode", type=int, default=0, help="episode index in --demos")
    parser.add_argument("--show-prompt", action="store_true", help="show the full GLAM prompt instead of the descriptions")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/episodes"))
    args = parser.parse_args()

    if args.demos:
        ep = load_episodes(args.demos)[args.episode]
        task, eval_set, seed = ep["task"], ep["eval_set"], ep["seed"]
        actions = [s["action"] for s in ep["steps"]]
        source = f"demo {args.demos.name}#{args.episode}"
    else:
        task, eval_set = args.task, args.eval_set
        seed = get_seeds(task, eval_set, args.seed_index + 1)[-1]
        policy = ExpertPolicy() if args.policy == "expert" else RandomPolicy(seed)
        actions = run_episode(make_env(task, eval_set), policy, seed).actions
        source = args.policy

    frames = replay(task, eval_set, seed, actions)
    outcome = "success" if frames[-1].success else "failure"
    title = f"{task} / {eval_set} / seed {seed} / {source} / {outcome} in {len(frames) - 1} steps"
    stem = f"{task}_{eval_set}_seed{seed}_{source.split()[0]}"
    gif = save_gif(frames, args.out / f"{stem}.gif", title, fps=args.fps, show_prompt=args.show_prompt)
    strip = save_strip(frames, args.out / f"{stem}_strip.png", title)
    print(f"wrote {gif}\nwrote {strip}")


if __name__ == "__main__":
    main()
