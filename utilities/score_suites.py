"""Score one checkpoint on every evaluation suite and write a JSON record.

The headline metric of the warm-vs-cold study is **best-of-5 against par**: five
rollouts per domain (the greedy one plus four sampled), the best of them scored
against the domain's own Gauss-Bonnet lower bound, so a suite score of 24/24
means every domain was solved to a provable topological optimum.

    venv/bin/python utilities/score_suites.py -config <config.yml> \
        -checkpoint <agent.zip> -out out/2x2/warm-sticky/scores/0600000.json

Suites are declared in `SUITES`: five geo2d families (three hole-free, two with
holes) at a held-out seed, plus the fifteen Mesh Quest levels. `-seed` picks the
geo2d seed, so the same script produces the dev (seed 0) and held-out (seed 7)
tables.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.append(os.getcwd())

from src.geo2d_bridge import import_geo2d, load_model  # noqa: E402
from src.utils import load_yaml_config  # noqa: E402
from utilities.evaluate_geo2d import evaluate_geometry  # noqa: E402

# name -> geo2d.generate_many keyword arguments
SUITES = {
    "straight":        dict(preset="straight", n_holes=0),
    "polycube":        dict(preset="polycube", n_holes=0),
    "polycube-holes":  dict(preset="polycube", n_holes=(1, 2), ratio=8),
    "straight-holes":  dict(preset="straight", n_holes=(1, 2), ratio=8, n_ops=(1, 2), n_mods=(0, 2)),
}

# Retired from the default set. Measured on the seed-7 draw: nineteen of its 24
# domains solve with fewer than five elements, eleven with two or fewer, and SIX
# are bare quadrilaterals -- four corners, one element, zero moves -- so they
# arrive already at par and are free to every method, Gmsh included. A 24/24
# there carried almost no information. Still reachable by name for anyone who
# wants it.
RETIRED_SUITES = {
    "straight-small":  dict(preset="straight", n_holes=0, ratio=5, n_ops=(1, 2), n_mods=(0, 1)),
}
ALL_SUITES = {**SUITES, **RETIRED_SUITES}

# A domain with fewer sides than this is not a meshing problem. Seven of the
# original 120 were quadrilaterals; the floor keeps the benchmark about the work
# rather than about how many freebies a draw happened to contain.
MIN_CORNERS = 8


def geometries(name, seed, n, max_corners=24, min_corners=MIN_CORNERS):
    """`n` geometries of a suite, within the corner-count window."""
    geo2d = import_geo2d()
    from src.geo2d_bridge import geometry_corner_count
    out = []
    draw = seed
    while len(out) < n and draw < seed + 200 * n:
        try:
            geometry = geo2d.generate(draw, **ALL_SUITES[name])
            corners = geometry_corner_count(geometry)
            if min_corners <= corners <= max_corners:
                out.append(geometry)
        except Exception:
            pass
        draw += 1
    return out


def score_suite(model, env_config, name, seed, n, n_samples, template_size,
                max_corners, min_corners=MIN_CORNERS):
    rows = []
    for index, geometry in enumerate(geometries(name, seed, n, max_corners, min_corners)):
        row, *_ = evaluate_geometry(model, env_config, geometry, index, n_samples,
                                    template_size, drop_collinear=True, keep_frames=False,
                                    resmooth="optimize", resmooth_iters=10)
        rows.append(row)
    total = len(rows)
    return {
        "suite": name, "seed": seed, "total": total,
        "greedy": sum(r["greedy_solved"] for r in rows),
        "best_of_n": sum(r["bestN_solved"] for r in rows),
        "best_of_n_resmoothed": sum(r["bestN_solved_resmoothed"] for r in rows),
        "mean_quality": round(float(np.mean([r["bestN_quality_resmoothed"] for r in rows])), 3)
                        if total else 0.0,
        "mean_moves": round(float(np.mean([r["bestN_moves"] for r in rows if r["bestN_moves"] > 0])), 1)
                      if any(r["bestN_moves"] > 0 for r in rows) else -1.0,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Score a checkpoint on every suite")
    parser.add_argument("-config", required=True)
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-seed", default=7, type=int, help="geo2d seed of the suites")
    parser.add_argument("-n", default=24, type=int)
    parser.add_argument("-repeats", default=1, type=int,
                        help="evaluate the whole suite this many times and report the mean; "
                             "one pass carries about +/-3 domains of sampling noise on the "
                             "135-domain total, so a difference below three is not "
                             "measurable from a single pass (utilities/measure_metric_noise.py)")
    parser.add_argument("-n_samples", default=4, type=int,
                        help="sampled rollouts on top of the greedy one; 4 gives best-of-5")
    parser.add_argument("-template_size", default=None, type=int)
    parser.add_argument("-max_corners", default=24, type=int)
    parser.add_argument("-min_corners", default=MIN_CORNERS, type=int,
                        help="skip domains with fewer sides; 8 keeps the benchmark "
                             "about the work rather than about freebies")
    parser.add_argument("-suites", default=",".join(SUITES))
    parser.add_argument("-holdout", action="store_true", help="also score the Mesh Quest levels")
    parser.add_argument("-steps", default=-1, type=int, help="training steps, recorded in the JSON")
    parser.add_argument("-out", required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    env_config = dict(config["environment"])
    template_size = args.template_size or env_config.get("template_size", 128)
    model = load_model(args.checkpoint, args.config, template_size=template_size)

    record = {"checkpoint": args.checkpoint, "config": args.config, "steps": args.steps,
              "seed": args.seed, "n_samples": args.n_samples, "repeats": args.repeats,
              "suites": {}}
    for name in args.suites.split(","):
        t0 = time.time()
        passes = [score_suite(model, env_config, name, args.seed, args.n,
                              args.n_samples, template_size, args.max_corners,
                              args.min_corners)
                  for _ in range(max(args.repeats, 1))]
        summary = passes[0]
        if len(passes) > 1:
            # keep the first pass's rows, but report the mean over passes: one pass
            # is +/-3 on the total, which is wider than most differences worth reading.
            # Read every pass's value out BEFORE writing any mean back -- `summary`
            # aliases passes[0], so mutating it first overwrites that pass's own number.
            per_pass = {key: [p[key] for p in passes]
                        for key in ("greedy", "best_of_n", "best_of_n_resmoothed")}
            summary["passes"] = list(per_pass["best_of_n"])
            summary["best_of_n_spread"] = int(max(per_pass["best_of_n"])
                                              - min(per_pass["best_of_n"]))
            for key, values in per_pass.items():
                summary[key] = float(np.mean(values))
        record["suites"][name] = summary
        spread = f"  spread {summary['best_of_n_spread']}" if args.repeats > 1 else ""
        print(f"{name:16s} greedy {summary['greedy']:5.1f}/{summary['total']:2d}  "
              f"best-of-{args.n_samples + 1} {summary['best_of_n']:5.1f}/{summary['total']:2d}"
              f"{spread}  ({time.time() - t0:.0f}s)", flush=True)

    if args.holdout:
        from src.holdout import LEVEL_NAMES  # noqa: E402
        from utilities.evaluate_holdout import evaluate_holdout  # noqa: E402
        rows, summary = evaluate_holdout(model, env_config, levels=LEVEL_NAMES,
                                         n_samples=args.n_samples, verbose=False)
        record["holdout"] = {"greedy": summary["greedy"], "best_of_n": summary["bestN"],
                             "total": summary["total"], "rows": rows}
        print(f"{'mesh-quest':16s} greedy {summary['greedy']:2d}/{summary['total']:2d}  "
              f"best-of-{args.n_samples + 1} {summary['bestN']:2d}/{summary['total']:2d}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(record, handle, indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
