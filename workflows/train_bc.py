"""Behaviour-clone the certified solutions from `envs.solved_instances`.

Random play reaches par on about 2% of 5-8-gons and never above eight sides,
so PPO from scratch has almost nothing to bootstrap on. The generator sidesteps
that: it runs the problem backwards from meshes that are at par by
construction, which yields as many (state, optimal move) pairs as we care to
make, on the training distribution, without any search.

The result is a checkpoint in PPO's own format, so `train_curriculum.py
-checkpoint ...` picks it up and continues with reinforcement learning.

    venv/bin/python workflows/train_bc.py -config <run-dir>/config.yml \\
        -num_instances 4000 -epochs 8
"""

import argparse
import datetime
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.append(os.getcwd())
from envs.environment_maker import initialize_environment  # noqa: E402
from envs.solved_instances import generate_instances  # noqa: E402
from src.behaviour_cloning import (  # noqa: E402
    Dataset, balance_replays, build_model, collect, drop_holdout_domains, train_epoch)
from src.utils import load_yaml_config  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description="Behaviour-clone certified solutions")
    parser.add_argument("-config", required=True)
    parser.add_argument("-instances", default=None,
                        help="comma-separated pickles of solved instances")
    parser.add_argument("-checkpoint", default=None,
                        help="warm-start from this checkpoint instead of a fresh net")
    parser.add_argument("-num_instances", default=4000, type=int)
    parser.add_argument("-min_cells", default=2, type=int)
    parser.add_argument("-max_cells", default=14, type=int)
    parser.add_argument("-replays", default=2, type=int)
    parser.add_argument("-hole_probability", default=0.30, type=float,
                        help="share of CANDIDATES that try for a hole; the achieved "
                             "fraction is lower because a hole also needs eight "
                             "cells, so 0.30 over cells 2-14 lands near 20 percent")
    parser.add_argument("-pinwheel_probability", default=0.40, type=float,
                        help="share of hole instances whose hole is turned against "
                             "its rim, forcing a circulating mesh. Needs "
                             "-gate untangle: the tightest member sits at 0.196 "
                             "under a laplacian and the default gate refuses it")
    parser.add_argument("-gate", default="untangle",
                        choices=("laplacian", "untangle", "topology"),
                        help="acceptance test for a candidate solution. `untangle` "
                             "matches the environment's evaluator: topology at par, "
                             "then geo2d's optimiser, then a low degeneracy bar. "
                             "`laplacian` is the old test and refuses a topology "
                             "the Laplacian merely cannot draw")
    parser.add_argument("-balance", action="store_true",
                        help="replay rare buckets more often to flatten the mixture")
    parser.add_argument("-exclude_holdout", action="store_true",
                        help="drop instances posing the same problem as a held-out level")
    parser.add_argument("-epochs", default=8, type=int)
    parser.add_argument("-batch_size", default=256, type=int)
    parser.add_argument("-lr", default=3.0e-4, type=float)
    parser.add_argument("-vf_coef", default=0.25, type=float)
    parser.add_argument("-holdout_every", default=2, type=int)
    parser.add_argument("-extra_levels", action="store_true",
                        help="score the extra held-out domains too, not just the game's fifteen")
    parser.add_argument("-seed", default=0, type=int)
    parser.add_argument("-out", default=None)
    parser.add_argument("-save_instances", default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = config.get("output_dir", os.path.dirname(args.config))
    os.makedirs(output_dir, exist_ok=True)
    env_config = dict(config["environment"])
    env_config.setdefault("logdir", output_dir)
    rng = np.random.default_rng(args.seed)

    print("BC START :", datetime.datetime.now())
    if args.instances:
        instances = []
        for path in args.instances.split(","):
            with open(path.strip(), "rb") as handle:
                loaded = pickle.load(handle)
            print(f"loaded {len(loaded)} solved instances from {path.strip()}")
            instances += loaded
    else:
        t0 = time.time()
        instances, stats = generate_instances(
            args.num_instances, cell_range=(args.min_cells, args.max_cells),
            rng=rng, hole_probability=args.hole_probability,
            pinwheel_probability=args.pinwheel_probability, gate=args.gate,
            verbose=True)
        print(f"generated {len(instances)} solved instances in {time.time() - t0:.0f}s "
              f"({stats})")

    if args.exclude_holdout:
        instances = drop_holdout_domains(instances)

    if args.save_instances:
        os.makedirs(os.path.dirname(args.save_instances) or ".", exist_ok=True)
        with open(args.save_instances, "wb") as handle:
            pickle.dump(instances, handle)
        print("saved instances ->", args.save_instances)

    degrees = np.array([i.polygon_degree for i in instances])
    pars = np.array([i.par for i in instances])
    print(f"  degree {degrees.min()}-{degrees.max()}, "
          f"par {dict(zip(*np.unique(pars, return_counts=True)))}")

    targets = sorted({instance.target for instance in instances})
    envs = {target: initialize_environment(dict(env_config, face_desired_degree=target))
            for target in targets}
    print(f"  targets present: {targets}  "
          + "  ".join(f"{t}: {sum(i.target == t for i in instances)}" for t in targets))
    env = envs if len(envs) > 1 else envs[targets[0]]
    replays = (balance_replays(instances, args.replays) if args.balance
               else args.replays)
    if args.balance:
        print(f"  balanced replays: min {min(replays)} median "
              f"{int(np.median(replays))} max {max(replays)}")
    t0 = time.time()
    dataset = collect(env, instances, replays, rng).finalize()
    print(f"collected {len(dataset)} state-action pairs in {time.time() - t0:.0f}s")

    model = build_model(config, env_config, output_dir)
    if args.checkpoint:
        print("warm starting from", args.checkpoint)
        loaded = PPO.load(args.checkpoint, custom_objects={"policy_kwargs": None})
        model.policy.load_state_dict(loaded.policy.state_dict())
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    from src.holdout import ALL_LEVEL_NAMES, LEVEL_NAMES  # noqa: E402
    from utilities.evaluate_holdout import evaluate_holdout  # noqa: E402
    holdout_levels = ALL_LEVEL_NAMES if args.extra_levels else LEVEL_NAMES

    best = (-1, -1)
    for epoch in range(args.epochs):
        cross_entropy, accuracy = train_epoch(
            policy, optimizer, dataset, rng, batch_size=args.batch_size,
            vf_coef=args.vf_coef)
        print(f"epoch {epoch + 1}/{args.epochs}  ce {cross_entropy:.4f}  "
              f"top-1 {accuracy:.1%}", flush=True)

        if args.holdout_every and (epoch + 1) % args.holdout_every == 0:
            policy.eval()
            greedy = best_of_n = 0
            for target in targets:
                print(f"\n  held-out, face target {target}:")
                _, summary = evaluate_holdout(
                    model, dict(env_config, face_desired_degree=target),
                    levels=holdout_levels, n_samples=8, verbose=True)
                greedy += summary["greedy"]
                best_of_n += summary["bestN"]
            score = (greedy, best_of_n)
            print(f"  combined across targets: greedy {greedy}, best-of-8 {best_of_n}")
            if score > best:
                best = score
                model.save(os.path.join(output_dir, "best_holdout_model"))
                print("  new best held-out score", score)

    out = args.out or os.path.join(output_dir, "bc_model")
    model.save(out)
    print("saved", out + ".zip")
    print("BC STOP :", datetime.datetime.now())


if __name__ == "__main__":
    main()
