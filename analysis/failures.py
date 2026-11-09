"""Failure analysis of the final evaluation (proposal 6.6): why test episodes fail.

    python -m analysis.failures      # outputs/eval -> outputs/results/failures.csv, failures.md, failures.json

Every failed episode of the conditions in ``config.ANALYSIS.models`` is
replayed from its recorded ``(task, eval set, seed, actions)`` (no model is
run) and given one category, the first that applies in this order:

1. ``wrong_object``: picked up an object the instruction does not ask for
   (for putnext and pickup_then_goto, anything but the first object).
2. ``first_step_only`` (hard tier): picked up the right object, then failed the
   second step (placing it, or going to the second object).
3. ``reached_no_pickup`` (tasks that need ``pick up``): stood facing the object
   to pick up at least once, but never picked it up.
4. ``stuck``: spent its last ``stuck_window`` steps in at most ``stuck_cells``
   cells; ``stuck_kind`` says how: ``spinning`` (turns only), ``blocked``
   (most forward moves hit an object or a wall) or ``other``.
5. ``never_saw_goal``: the object it needed first never entered its 7x7 view.
6. ``wandered``: none of the above (saw the object, kept moving, ran out of time).

Every failure in this evaluation is a time-out (no episode ends early without
success), so the categories describe how the step limit was reached. Next to
the category, each failed episode keeps non-exclusive flags (saw the goal,
faced it, faced a look-alike sharing its colour or type, pickups that did
nothing, toggles) and the first step at which it left the expert's choice.

Two analyses read the recorded decisions only, for every condition: the action
confusion matrix (expert's action in the policy's state vs the policy's
action) and agreement with the expert by step, separately for episodes that
succeeded and failed (whether failures drift away from the expert over time).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import config
from babytext import ACTION_TO_IDX, NUM_ACTIONS, make_env
from babytext.constants import OBJ_TYPES, TASK_TIERS
from evaluation.record import read_episodes
from evaluation.summarise import load_cells

__all__ = [
    "CATEGORIES",
    "STUCK_KINDS",
    "agreement_by_step",
    "category_shares",
    "classify",
    "confusion",
    "episode_features",
    "failure_rows",
    "main",
]

CATEGORIES = ("wrong_object", "first_step_only", "reached_no_pickup", "stuck", "never_saw_goal", "wandered")
STUCK_KINDS = ("spinning", "blocked", "other")
PICKUP, DROP, TOGGLE, FORWARD = (ACTION_TO_IDX[a] for a in ("pick up", "drop", "toggle", "go forward"))
TURNS = {ACTION_TO_IDX["turn left"], ACTION_TO_IDX["turn right"]}
NEEDS_PICKUP = {"pickup", "putnext", "pickup_then_goto"}
CSV_FIELDS = ["model", "tier", "run", "task", "eval_set", "seed", "length", "category", "stuck_kind", "stuck",
              "saw_goal", "faced_goal", "faced_lookalike", "wrong_pickup", "wrong_pickup_kind", "wrong_pickup_t", "chance_lookalike",
              "chance_other_instruction_object", "chance_unrelated", "carried_goal", "empty_pickups", "toggles",
              "blocked_forward", "first_disagreement"]


def episode_features(task: str, eval_set: str, seed: int, actions: list[int], stuck_window: int | None = None,
                     stuck_cells: int | None = None) -> dict:
    """Replay one episode and measure what the agent did.

    Args:
        task, eval_set, seed: identify the room (as recorded by ``evaluation``).
        actions: the recorded actions, length T.
        stuck_window, stuck_cells: the "stuck" test (default ``config.ANALYSIS``).

    Returns:
        ``success`` (as replayed), ``length`` T, ``saw_goal`` (the first instruction object was in view at some
        state), ``faced_goal`` (it was in the cell ahead), ``faced_lookalike`` (a non-instruction object sharing an
        instruction object's colour or type was ahead), ``wrong_pickup`` and ``wrong_pickup_kind`` (what the first
        wrong pickup was: ``other_instruction_object``, a ``lookalike`` or ``unrelated``; ``wrong_pickup_t``: its
        step, -1 if none), ``carried_goal`` (picked
        up the first instruction object), ``chance_<kind>`` (the share of the room's objects, other than the first
        instruction object, of each wrong-pickup kind: what a uniformly random wrong pickup would be), ``empty_pickups`` (``pick up`` or ``drop`` that changed nothing), ``toggles``,
        ``blocked_forward`` (``go forward`` that did not move the agent), ``stuck`` and ``stuck_kind``.
    """
    stuck_window = config.ANALYSIS.stuck_window if stuck_window is None else stuck_window
    stuck_cells = config.ANALYSIS.stuck_cells if stuck_cells is None else stuck_cells
    env = make_env(task, eval_set)
    obs, info = env.reset(seed=seed)
    level = env.unwrapped
    goal = [tuple(g) for g in info["goal"]]  # (type, colour) per instruction object, mission order
    kinds, colours = {k for k, _ in goal}, {c for _, c in goal}

    def relation(key):
        return ("other_instruction_object" if key in goal else
                "lookalike" if key[0] in kinds or key[1] in colours else "unrelated")

    # Chance: what a pickup of a uniformly random object other than the first instruction object would be.
    others = [relation((o.type, o.color)) for o in level.grid.grid
              if o is not None and o.type in OBJ_TYPES and (o.type, o.color) != goal[0]]
    chance = {k: others.count(k) / len(others) for k in ("other_instruction_object", "lookalike", "unrelated")}

    f = dict(saw_goal=False, faced_goal=False, faced_lookalike=False, wrong_pickup=False, wrong_pickup_kind="", wrong_pickup_t=-1,
             carried_goal=False, empty_pickups=0, toggles=0, blocked_forward=0)
    cells, blocked = [], []  # per step: agent cell before the action; did a forward move fail
    reward, terminated = 0.0, False
    for t, a in enumerate(actions):
        pos = tuple(int(v) for v in level.agent_pos)
        cells.append(pos)
        # info["goal_positions"][0]: where the first instruction object lies now (empty while carried).
        f["saw_goal"] |= any(level.in_view(x, y) for x, y in info["goal_positions"][0])
        ahead = level.grid.get(*level.front_pos)
        if ahead is not None and ahead.type in OBJ_TYPES:
            key = (ahead.type, ahead.color)
            f["faced_goal"] |= key == goal[0]
            f["faced_lookalike"] |= key not in goal and (ahead.type in kinds or ahead.color in colours)
        carrying_before = level.carrying
        obs, reward, terminated, truncated, info = env.step(a)
        carrying = level.carrying
        if a == PICKUP:
            if carrying_before is None and carrying is not None:
                key = (carrying.type, carrying.color)
                if key == goal[0]:
                    f["carried_goal"] = True
                elif not f["wrong_pickup"]:
                    f["wrong_pickup"] = True
                    f["wrong_pickup_kind"] = relation(key)
                    f["wrong_pickup_t"] = t
            else:
                f["empty_pickups"] += 1
        elif a == DROP and (carrying_before is None or carrying is not None):
            f["empty_pickups"] += 1
        elif a == TOGGLE:
            f["toggles"] += 1
        failed_move = a == FORWARD and tuple(int(v) for v in level.agent_pos) == pos
        f["blocked_forward"] += failed_move
        blocked.append(failed_move)
        if terminated or truncated:
            break
    env.close()

    tail = slice(-stuck_window, None)
    stuck = len(cells) >= stuck_window and len(set(cells[tail])) <= stuck_cells
    kind = ""
    if stuck:
        last = actions[: len(cells)][tail]
        forwards = sum(a == FORWARD for a in last)
        if all(a in TURNS for a in last):
            kind = "spinning"
        elif forwards and sum(blocked[tail]) >= forwards / 2:
            kind = "blocked"
        else:
            kind = "other"
    return {**f, "success": bool(terminated and reward > 0), "length": len(cells), "stuck": stuck, "stuck_kind": kind,
            **{f"chance_{k}": v for k, v in chance.items()}}


def classify(task: str, features: dict) -> str:
    """The failure category of a failed episode's ``episode_features`` (order in the module docstring)."""
    if features["wrong_pickup"]:
        return "wrong_object"
    if features["carried_goal"] and task in TASK_TIERS["hard"]:
        return "first_step_only"
    if features["faced_goal"] and task in NEEDS_PICKUP:
        return "reached_no_pickup"
    if features["stuck"]:
        return "stuck"
    if not features["saw_goal"]:
        return "never_saw_goal"
    return "wandered"


def first_disagreement(episode: dict) -> int | None:
    """Index of the first labelled decision where the policy did not take the expert's action (None: never)."""
    return next((t for t, (a, e) in enumerate(zip(episode["actions"], episode["expert"])) if e is not None and a != e),
                None)


def _cell_failures(args: tuple) -> list[dict]:
    """Rows for the failed episodes of one cell (a worker process; module level so it can be pickled)."""
    prefix, identity = args
    rows = []
    for e in read_episodes(prefix):
        if e["success"]:
            continue
        f = episode_features(identity["task"], identity["eval_set"], e["seed"], e["actions"])
        if f["success"]:
            raise RuntimeError(f"{prefix}: seed {e['seed']} succeeds on replay but was recorded as a failure")
        rows.append({**{k: identity[k] for k in ("model", "tier", "run", "task", "eval_set")}, "seed": e["seed"],
                     **f, "category": classify(identity["task"], f), "first_disagreement": first_disagreement(e)})
    return rows


def cell_prefix(root: Path, identity: dict) -> Path:
    return root / identity["model"] / identity["tier"] / identity["run"] / f"{identity['task']}.{identity['eval_set']}"


def failure_rows(root: Path, models: tuple[str, ...] | None = None, workers: int | None = None) -> list[dict]:
    """Replay and categorise every failed episode of ``models`` under ``root`` (one row per episode)."""
    models = config.ANALYSIS.models if models is None else models
    workers = workers if workers is not None else config.ANALYSIS.workers
    workers = workers or min(8, os.cpu_count() or 1)
    jobs = [(cell_prefix(root, c["identity"]), c["identity"]) for c in load_cells(root)
            if c["identity"]["model"] in models and c["success_rate"] < 1]
    if workers == 1:
        return [r for job in jobs for r in _cell_failures(job)]
    with ProcessPoolExecutor(workers) as pool:
        return [r for rows in pool.map(_cell_failures, jobs) for r in rows]


def confusion(episodes: list[dict]) -> np.ndarray:
    """Counts (NUM_ACTIONS, NUM_ACTIONS): row = the expert's action in the policy's state, column = the policy's."""
    m = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=np.int64)
    for e in episodes:
        for a, x in zip(e["actions"], e["expert"]):
            if x is not None:
                m[x, a] += 1
    return m


def agreement_by_step(episodes: list[dict], max_steps: int) -> dict[str, list]:
    """Agreement with the expert at each step t < ``max_steps``, for successful and failed episodes.

    Returns ``{"success": [...], "failure": [...], "n_success": [...], "n_failure": [...]}``, each of length
    ``max_steps``: the share of labelled decisions at step t that match the expert (None when no episode lasted that
    long) and how many decisions it rests on.
    """
    hits = {k: np.zeros(max_steps) for k in ("success", "failure")}
    counts = {k: np.zeros(max_steps, dtype=np.int64) for k in ("success", "failure")}
    for e in episodes:
        k = "success" if e["success"] else "failure"
        for t, (a, x) in enumerate(zip(e["actions"][:max_steps], e["expert"][:max_steps])):
            if x is not None:
                hits[k][t] += a == x
                counts[k][t] += 1
    out = {}
    for k in hits:
        out[k] = [float(h / n) if n else None for h, n in zip(hits[k], counts[k])]
        out[f"n_{k}"] = counts[k].tolist()
    return out


def _decision_tables(root: Path) -> dict:
    """Confusion matrices and agreement by step per (model, tier, eval set), from every recorded cell."""
    by = defaultdict(list)
    for c in load_cells(root):
        i = c["identity"]
        by[(i["model"], i["tier"], i["eval_set"])].append(cell_prefix(root, i))
    out = {}
    for (model, tier, eval_set), prefixes in sorted(by.items()):
        episodes = [e for p in prefixes for e in read_episodes(p)]
        max_steps = max(e["length"] for e in episodes)
        out[f"{model}|{tier}|{eval_set}"] = {"confusion": confusion(episodes).tolist(),
                                             "agreement_by_step": agreement_by_step(episodes, max_steps)}
    return out


def category_shares(rows: list[dict], cells: list[dict], by_task: bool = False) -> dict[tuple, dict]:
    """Per (model, tier, eval set[, task]): episodes, failures, and each category's share of all episodes.

    Shares are of *all* episodes, so they sum to the failure rate. ``stuck_kinds`` counts the stuck episodes by kind.
    """
    def key_of(d):
        return (d["model"], d["tier"], d["eval_set"]) + ((d["task"],) if by_task else ())

    out = {}
    for c in cells:
        k = key_of(c["identity"])
        out.setdefault(k, {"episodes": 0, "failed": 0, "counts": Counter(), "stuck_kinds": Counter()})
        out[k]["episodes"] += c["n_episodes"]
    for r in rows:
        d = out[key_of(r)]
        d["failed"] += 1
        d["counts"][r["category"]] += 1
        if r["category"] == "stuck":
            d["stuck_kinds"][r["stuck_kind"]] += 1
    for d in out.values():
        d["shares"] = {c: d["counts"][c] / d["episodes"] for c in CATEGORIES}
    return out


def _order(key: tuple) -> tuple:
    """Sort by tier, eval set (``config.EVAL.test_sets`` order), then the rest of the key."""
    model, tier, eval_set, *rest = key
    return (tier, config.EVAL.test_sets.index(eval_set), *rest, model)


def category_table(rows: list[dict], cells: list[dict], by_task: bool = False, eval_sets: tuple[str, ...] | None = None
                   ) -> str:
    """Markdown: for each (tier, eval set[, task], model), the share of all episodes failing in each category."""
    shares = category_shares(rows, cells, by_task)
    task_col = " task |" if by_task else ""
    title = "per task, `test`" if by_task else "per tier"
    out = [f"## Failure categories, {title} (share of all episodes; a row sums to the failure rate)", "",
           f"| tier | eval set |{task_col} condition | failed | " + " | ".join(CATEGORIES)
           + " | stuck: spinning / blocked / other |",
           "|---" * (5 + by_task + len(CATEGORIES)) + "|"]
    for key in sorted(shares, key=_order):
        if eval_sets and key[2] not in eval_sets:
            continue
        d = shares[key]
        cols = " | ".join(f"{d['shares'][c]:.3f}" for c in CATEGORIES)
        stuck = " / ".join(str(d["stuck_kinds"][k]) for k in STUCK_KINDS)
        task = f" {key[3]} |" if by_task else ""
        out.append(f"| {key[1]} | {key[2]} |{task} {key[0]} | {d['failed'] / d['episodes']:.3f} | {cols} | {stuck} |")
    return "\n".join(out)


def wrong_pickup_table(rows: list[dict]) -> str:
    """Markdown: what the first wrong pickup was, as shares of each condition's wrong-object failures."""
    kinds = ("other_instruction_object", "lookalike", "unrelated")
    by = defaultdict(list)
    for r in rows:
        if r["category"] == "wrong_object":
            by[(r["model"], r["tier"], r["eval_set"])].append(r)
    out = ["## What the wrong object was (share of `wrong_object` failures; chance in brackets)", "",
           "`other_instruction_object`: the other object of a two-object instruction (e.g. the target of putnext); "
           "`lookalike`: shares a colour or type with an instruction object; `unrelated`: shares neither. Chance: the "
           "share a uniformly random wrong pickup would have, from the same failed episodes' rooms.", "",
           "| tier | eval set | condition | wrong-object failures | " + " | ".join(kinds) + " |",
           "|---" * (4 + len(kinds)) + "|"]
    for key in sorted(by, key=_order):
        rs = by[key]
        cols = " | ".join(f"{np.mean([r['wrong_pickup_kind'] == k for r in rs]):.2f} "
                          f"({np.mean([r[f'chance_{k}'] for r in rs]):.2f})" for k in kinds)
        out.append(f"| {key[1]} | {key[2]} | {key[0]} | {len(rs)} | {cols} |")
    return "\n".join(out)


def flag_table(rows: list[dict]) -> str:
    """Markdown: the non-exclusive flags, as shares of each condition's failed episodes."""
    flags = ["saw_goal", "faced_goal", "faced_lookalike", "stuck"]
    out = ["## Flags of failed episodes (share of failed episodes; not exclusive)", "",
           "| tier | eval set | condition | failed episodes | " + " | ".join(flags)
           + " | empty pickups / drops per episode | toggles per episode | blocked forwards per episode "
             "| first step off the expert (median) |",
           "|---" * (8 + len(flags)) + "|"]
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["tier"], r["eval_set"])].append(r)
    for key in sorted(by, key=_order):
        rs = by[key]
        firsts = [r["first_disagreement"] for r in rs if r["first_disagreement"] is not None]
        shares = " | ".join(f"{np.mean([r[f] for r in rs]):.2f}" for f in flags)
        out.append(f"| {key[1]} | {key[2]} | {key[0]} | {len(rs)} | {shares} | {np.mean([r['empty_pickups'] for r in rs]):.1f} "
                   f"| {np.mean([r['toggles'] for r in rs]):.1f} | {np.mean([r['blocked_forward'] for r in rs]):.1f} "
                   f"| {np.median(firsts) if firsts else '-'} |")
    return "\n".join(out)


def confusion_table(tables: dict, models: tuple[str, ...]) -> str:
    """Markdown: row-normalised confusion matrices on ``test`` for ``models``."""
    names = ["left", "right", "forward", "pick up", "drop", "toggle"]
    out = ["## Action confusion on `test` (row: the expert's action in the policy's state; share of that row)", ""]
    for key, t in tables.items():
        model, tier, eval_set = key.split("|")
        if eval_set != "test" or model not in models:
            continue
        m = np.array(t["confusion"], dtype=float)
        out += [f"### {model}, {tier} tier", "", "| expert \\ policy | " + " | ".join(names) + " | decisions |",
                "|---" * (len(names) + 2) + "|"]
        for i, name in enumerate(names):
            n = m[i].sum()
            if n:
                out.append(f"| {name} | " + " | ".join(f"{v / n:.2f}" for v in m[i]) + f" | {int(n):,} |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, default=config.DATA.outputs_dir / "eval", help="evaluation cells")
    parser.add_argument("--out", type=Path, default=config.DATA.outputs_dir / "results")
    parser.add_argument("--models", nargs="+", default=list(config.ANALYSIS.models))
    parser.add_argument("--workers", type=int, default=None, help="replay processes (default config.ANALYSIS)")
    args = parser.parse_args(argv)
    cells = load_cells(args.eval)
    if not cells:
        parser.error(f"no evaluation cells under {args.eval}")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = failure_rows(args.eval, tuple(args.models), args.workers)
    with open(args.out / "failures.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tables = _decision_tables(args.eval)
    (args.out / "failures.json").write_text(json.dumps(tables), encoding="utf-8")
    analysed = [c for c in cells if c["identity"]["model"] in args.models]
    report = "\n\n".join(["# Failure analysis", __doc__.split("\n\n", 2)[2].strip(), category_table(rows, analysed),
                          category_table(rows, analysed, by_task=True, eval_sets=("test",)), wrong_pickup_table(rows),
                          flag_table(rows), confusion_table(tables, tuple(args.models))])
    (args.out / "failures.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"wrote {args.out / 'failures.csv'}, failures.md and failures.json ({len(rows):,} failed episodes)")


if __name__ == "__main__":
    main()
