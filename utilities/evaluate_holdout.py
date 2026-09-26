"""Evaluate a checkpoint on the fifteen held-out Mesh Quest levels.

Three protocols, because progress is visible in the weaker ones long before
greedy play solves anything:

  greedy     one deterministic episode
  best-of-N  N sampled episodes, solved if any reaches par
  beam       policy-guided beam search over the env's own action vocabulary

The win test is the game's: vertex score at par, every face a quad, and min
scaled Jacobian at or above `quality_threshold`.

    venv/bin/python utilities/evaluate_holdout.py -input <run-dir> -n_samples 16 -beam 32
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from src.holdout import ALL_LEVEL_NAMES, LEVEL_NAMES, make_level_env  # noqa: E402
from src.search import beam_search, state_from_env  # noqa: E402
from src.utils import load_yaml_config, load_model_from_checkpoint  # noqa: E402

EXPERT_TRAJECTORIES = os.path.join("experiments", "expert_bc", "expert_trajectories.json")


def _expert_lengths():
    try:
        with open(EXPERT_TRAJECTORIES) as handle:
            data = json.load(handle)
    except OSError:
        return {}
    return {name: len(rec["actions"]) for name, rec in data.items() if rec.get("actions")}


def _obs_batch(obs):
    return {key: np.expand_dims(value, 0) for key, value in obs.items()}


def run_episode(model, env, deterministic):
    """One episode. Returns (solved, moves, face_score, vertex_excess, quality)."""
    obs, _ = env.reset()
    solved = env.is_at_par()
    steps = 0
    while not solved and steps < env.max_steps:
        action, _ = model.predict(_obs_batch(obs), deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(int(np.asarray(action).ravel()[0]))
        steps += 1
        solved = bool(info["at_par"])
        if terminated or truncated:
            break
    return {
        "solved": solved,
        "moves": steps,
        "face_score": env.global_face_score,
        "vertex_excess": env.global_vertex_score - env.par,
        "quality": float(env.min_element_quality()),
    }


def evaluate_level(model, env_config, level, n_samples=16, beam=0, beam_budget_s=60.0,
                   beam_depth=24, beam_top_k=6):
    env = make_level_env(env_config, level)
    par = env.par

    greedy = run_episode(model, env, deterministic=True)

    def rank(result):
        return (not result["solved"], result["face_score"], abs(result["vertex_excess"]),
                -result["quality"])

    # the greedy rollout is one of the N: "best of 5" means five episodes, and
    # `evaluate_geo2d.evaluate_geometry` counts them the same way, so the two
    # datasets report the same quantity
    best = (rank(greedy), greedy)
    sampled_solved = 0
    for _ in range(n_samples):
        result = run_episode(model, env, deterministic=False)
        sampled_solved += int(result["solved"])
        if rank(result) < best[0]:
            best = (rank(result), result)
    best_result = best[1]

    row = {
        "level": level,
        "par": par,
        "greedy_solved": int(greedy["solved"]),
        "greedy_moves": greedy["moves"],
        "greedy_face": greedy["face_score"],
        "greedy_vertex_excess": greedy["vertex_excess"],
        "greedy_quality": round(greedy["quality"], 3),
        "bestN_solved": int(best_result["solved"]),
        "bestN_hits": sampled_solved,
        "bestN_moves": best_result["moves"] if best_result["solved"] else -1,
        "beam_solved": -1,
        "beam_moves": -1,
        "beam_expanded": -1,
    }

    if beam > 0:
        env.reset()
        actions, stats = beam_search(
            model.policy, env, state=state_from_env(env), par=par,
            beam=beam, top_k=beam_top_k, max_depth=beam_depth, budget_s=beam_budget_s,
        )
        row["beam_solved"] = int(actions is not None)
        row["beam_moves"] = len(actions) if actions is not None else -1
        row["beam_expanded"] = stats["expanded"]

    return row


def evaluate_holdout(model, env_config, levels=None, n_samples=16, beam=0,
                     beam_budget_s=60.0, verbose=True):
    levels = levels or LEVEL_NAMES
    expert = _expert_lengths()
    rows = []
    if verbose:
        header = (f"{'level':14s} {'par':>3} {'greedy':>7} {'bestN':>7} {'hits':>5} "
                  f"{'beam':>6} {'moves':>6} {'expert':>6}")
        print(header)
        print("-" * len(header))
    for level in levels:
        row = evaluate_level(model, env_config, level, n_samples=n_samples,
                             beam=beam, beam_budget_s=beam_budget_s)
        row["expert_moves"] = expert.get(level, -1)
        rows.append(row)
        if verbose:
            beam_txt = "-" if row["beam_solved"] < 0 else ("WIN" if row["beam_solved"] else "fail")
            moves = row["beam_moves"] if row["beam_solved"] == 1 else (
                row["bestN_moves"] if row["bestN_solved"] else row["greedy_moves"])
            print(f"{row['level']:14s} {row['par']:3d} "
                  f"{'WIN' if row['greedy_solved'] else 'fail':>7} "
                  f"{'WIN' if row['bestN_solved'] else 'fail':>7} "
                  f"{row['bestN_hits']:5d} {beam_txt:>6} {moves:6d} "
                  f"{row['expert_moves']:6d}")

    game = [r for r in rows if r["level"] in LEVEL_NAMES]
    summary = {
        "greedy": sum(r["greedy_solved"] for r in rows),
        "bestN": sum(r["bestN_solved"] for r in rows),
        "beam": sum(max(r["beam_solved"], 0) for r in rows),
        "total": len(rows),
        # the game's fifteen on their own, so the headline stays comparable
        "game_greedy": sum(r["greedy_solved"] for r in game),
        "game_bestN": sum(r["bestN_solved"] for r in game),
        "game_total": len(game),
    }
    if verbose:
        print("-" * 60)
        print(f"greedy {summary['greedy']}/{summary['total']}   "
              f"best-of-{n_samples} {summary['bestN']}/{summary['total']}   "
              f"beam {summary['beam']}/{summary['total']}")
        if summary["game_total"] != summary["total"]:
            print(f"  of which the game's fifteen: greedy {summary['game_greedy']}"
                  f"/{summary['game_total']}, best-of-{n_samples} "
                  f"{summary['game_bestN']}/{summary['game_total']}")
    return rows, summary


def _resolve_paths(args):
    if args.input:
        config_fn = os.path.join(args.input, "config.yml")
        checkpoint = args.checkpoint
        if checkpoint is None:
            for name in ("best_holdout_model.zip", "expert_iteration_model.zip",
                         "bc_model.zip", "best_model.zip", "final_model.zip"):
                candidate = os.path.join(args.input, name)
                if os.path.isfile(candidate):
                    checkpoint = candidate
                    break
        return config_fn, checkpoint
    return args.config, args.checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate on the held-out Mesh Quest levels")
    parser.add_argument("-input", default=None, help="run directory holding config.yml and a checkpoint")
    parser.add_argument("-config", default=None)
    parser.add_argument("-checkpoint", default=None)
    parser.add_argument("-n_samples", default=16, type=int)
    parser.add_argument("-beam", default=0, type=int, help="beam width; 0 disables the beam protocol")
    parser.add_argument("-beam_budget", default=60.0, type=float)
    parser.add_argument("-levels", default=None, help="comma-separated subset of level names")
    parser.add_argument("-extra", action="store_true",
                        help="include the extra held-out domains, not just the game's fifteen")
    parser.add_argument("-out", default=None, help="CSV output path")
    args = parser.parse_args()

    config_fn, checkpoint = _resolve_paths(args)
    if config_fn is None or checkpoint is None:
        parser.error("need -input <run-dir>, or both -config and -checkpoint")

    config = load_yaml_config(config_fn)
    model = load_model_from_checkpoint(checkpoint, config_fn)
    levels = args.levels.split(",") if args.levels else (
        ALL_LEVEL_NAMES if args.extra else None)

    rows, summary = evaluate_holdout(
        model, config["environment"], levels=levels, n_samples=args.n_samples,
        beam=args.beam, beam_budget_s=args.beam_budget,
    )

    out = args.out or (os.path.join(args.input, "holdout.csv") if args.input else None)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("wrote", out)


if __name__ == "__main__":
    main()
