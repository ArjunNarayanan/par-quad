"""Gmsh's quad meshers on the held-out par > 0 domains.

The interesting half of the benchmark: these are domains where a perfectly
regular mesh is IMPOSSIBLE, so reaching the bound means finding the LEAST
irregular mesh rather than a flawless one. Every geo2d domain is par 0, so this
is the first time any mesher has been scored on the case the Gauss-Bonnet bound
is actually about.

Each mesher is given the element count the agent used, so nobody wins by
refining, and is scored on the agent's own metric: all-quad, and vertex
irregularity equal to par.

    venv/bin/python utilities/gmsh_par_set.py \\
        -config experiments/self-play/quad/pinwheel-v1/config.yml \\
        -checkpoint experiments/self-play/quad/pinwheel-v1/warm_ppo_model.zip
"""

import argparse
import collections
import csv
import os
import pickle
import sys

import numpy as np

sys.path.append(os.getcwd())

from src.geo2d_bridge import load_model  # noqa: E402
from src.utils import load_yaml_config  # noqa: E402
from utilities.compare_gmsh import (VARIANTS, domain_corner_wants,  # noqa: E402
                                    mesh_at_element_count, score_mesh)
from utilities.level_geometry import level_geometry  # noqa: E402
from utilities.score_par_set import make_env  # noqa: E402
from utilities.evaluate_geo2d import rollout  # noqa: E402


def agent_best(model, env_config, instance, n_samples):
    env = make_env(env_config, instance)
    greedy, _ = rollout(model, env, deterministic=True)
    best = greedy
    for _ in range(n_samples):
        sampled, _ = rollout(model, env, deterministic=False)
        if (sampled["face_score"], abs(sampled["vertex_excess"])) < \
           (best["face_score"], abs(best["vertex_excess"])):
            best = sampled
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-config", required=True)
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-set", default="data/par_test_set.pkl")
    parser.add_argument("-n_samples", default=5, type=int)
    parser.add_argument("-variants", default=",".join(VARIANTS))
    parser.add_argument("-out", default="out/gmsh-par-set")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.set, "rb") as handle:
        instances = pickle.load(handle)
    config = load_yaml_config(args.config)
    env_config = dict(config["environment"])
    model = load_model(args.checkpoint, args.config,
                       template_size=env_config.get("template_size", 128))
    variants = [v for v in args.variants.split(",") if v in VARIANTS]

    rows = []
    for index, instance in enumerate(instances):
        par = int(instance.par)
        best = agent_best(model, env_config, instance, args.n_samples)
        elements = best["faces"]
        rows.append({"index": index, "par": par, "method": "agent",
                     "elements": elements,
                     "all_quad": int(best["face_score"] == 0),
                     "vertex_score": int(best["vertex_excess"]) + par,
                     "excess_over_par": int(abs(best["vertex_excess"])),
                     "at_par": int(best["solved"]),
                     "min_quality": round(best["min_quality"], 4)})
        graph, _ = instance.build()
        geometry = level_geometry(graph)
        wants, tol = domain_corner_wants(geometry)
        for variant in variants:
            try:
                found = mesh_at_element_count(geometry, elements, VARIANTS[variant])
                if found is None:
                    continue
                _, _, nodes, elems = found
                scored = score_mesh(nodes, elems, wants, tol, par)
            except Exception:
                continue
            rows.append({"index": index, "par": par, "method": variant, **{
                k: scored[k] for k in ("elements", "all_quad", "vertex_score",
                                       "excess_over_par", "at_par", "min_quality")}})
        print(f"#{index} par={par} agent={elements}q "
              f"{'AT PAR' if best['solved'] else 'MISS'}", flush=True)

    path = os.path.join(args.out, "par_set.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["method"]]["at_par"].append(r["at_par"])
        by[r["method"]]["all_quad"].append(r["all_quad"])
        by[r["method"]]["excess"].append(r["excess_over_par"])
        by[r["method"]]["q"].append(float(r["min_quality"]))
    total = len(instances)
    print(f"\n{'method':<20}{'at par':>10}{'all quad':>11}{'excess':>9}{'min q':>8}")
    print("-" * 58)
    for method, d in sorted(by.items(), key=lambda kv: -sum(kv[1]['at_par'])):
        print(f"{method:<20}{sum(d['at_par']):>6}/{total:<3}{sum(d['all_quad']):>7}/{total:<3}"
              f"{np.mean(d['excess']):>9.1f}{np.mean(d['q']):>8.3f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
