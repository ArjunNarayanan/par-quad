"""Score a checkpoint on the held-out par > 0 domains.

These are domains where a perfectly regular mesh is IMPOSSIBLE -- the
Gauss-Bonnet bound is nonzero -- so reaching par means finding the LEAST
irregular mesh rather than a flawless one. No geo2d domain has this property, so
until now the benchmark never tested it, while about 35% of training instances
had it.

Only the domains are used; the generator's own solutions are discarded and the
agent has to find its own.

    venv/bin/python utilities/score_par_set.py \\
        -config experiments/self-play/quad/pinwheel-v1/config.yml \\
        -checkpoint experiments/self-play/quad/pinwheel-v1/warm_ppo_model.zip \\
        -set data/par_test_set.pkl -n_samples 5 -repeats 4
"""

import argparse
import collections
import json
import os
import pickle
import sys
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())

from envs.environment_maker import initialize_environment  # noqa: E402
from src.geo2d_bridge import load_model  # noqa: E402
from src.utils import load_yaml_config  # noqa: E402
from utilities.evaluate_geo2d import rollout  # noqa: E402


class FixedDomain:
    """Hand the env one domain, built fresh each reset."""

    def __init__(self, instance):
        self.instance = instance
        self.n = len(set(instance.loop))

    def __call__(self):
        graph, desired = self.instance.build()
        return deepcopy(graph), dict(desired)


def make_env(env_config, instance):
    config = deepcopy(env_config)
    config.pop("initializer", None)
    config["graph_initializer"] = FixedDomain(instance)
    # a held-out domain is evaluated exactly as given
    config["resample_if_at_par"] = False
    return initialize_environment(config)


def score(model, env_config, instances, n_samples):
    rows = []
    for index, instance in enumerate(instances):
        env = make_env(env_config, instance)
        greedy, _ = rollout(model, env, deterministic=True)
        best, hits = greedy, 0
        for _ in range(n_samples):
            sampled, _ = rollout(model, env, deterministic=False)
            hits += int(sampled["solved"])
            if (sampled["face_score"], abs(sampled["vertex_excess"])) < \
               (best["face_score"], abs(best["vertex_excess"])):
                best = sampled
        rows.append({"index": index, "par": int(instance.par),
                     "corners": len(set(instance.loop)),
                     "hole": bool(instance.has_hole),
                     "greedy_solved": int(greedy["solved"]),
                     "bestN_solved": int(best["solved"]),
                     "bestN_hits": hits,
                     "bestN_moves": best["moves"],
                     "bestN_excess": int(abs(best["vertex_excess"])),
                     "bestN_face": int(best["face_score"]),
                     "bestN_quality": round(best["min_quality"], 3)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-config", required=True)
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-set", default="data/par_test_set.pkl")
    parser.add_argument("-n_samples", default=5, type=int)
    parser.add_argument("-repeats", default=1, type=int)
    parser.add_argument("-out", default=None)
    args = parser.parse_args()

    with open(args.set, "rb") as handle:
        instances = pickle.load(handle)
    config = load_yaml_config(args.config)
    env_config = dict(config["environment"])
    model = load_model(args.checkpoint, args.config,
                       template_size=env_config.get("template_size", 128))

    passes = []
    for _ in range(max(args.repeats, 1)):
        rows = score(model, env_config, instances, args.n_samples)
        passes.append(sum(r["bestN_solved"] for r in rows))
    rows_last = rows

    total = len(instances)
    by_par = collections.defaultdict(lambda: [0, 0])
    for r in rows_last:
        by_par[r["par"]][0] += r["bestN_solved"]
        by_par[r["par"]][1] += 1
    greedy = sum(r["greedy_solved"] for r in rows_last)

    print(f"\npar > 0 held-out set: {total} domains")
    print(f"  greedy      {greedy}/{total}")
    print(f"  best-of-{args.n_samples + 1}   {np.mean(passes):.2f}/{total}"
          f"   passes {passes}   spread {max(passes) - min(passes)}")
    print(f"  {'par':>4}{'at par':>10}")
    for par in sorted(by_par):
        solved, count = by_par[par]
        print(f"  {par:>4}{solved:>7}/{count:<3}")
    excess = [r["bestN_excess"] for r in rows_last if not r["bestN_solved"]]
    if excess:
        print(f"  unsolved: {len(excess)}, mean excess over par {np.mean(excess):.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"checkpoint": args.checkpoint, "set": args.set,
                       "total": total, "greedy": greedy,
                       "best_of_n": float(np.mean(passes)), "passes": passes,
                       "rows": rows_last}, handle, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
