"""Gmsh on the scorer's own curved suites, matched domain by domain to the agent.

    venv/bin/python utilities/gmsh_curved_suite.py -agent out/x/curved.json \
        -suite curved -out out/gmsh-curved-suite/curved.json

Draws EXACTLY the domains `score_quality_objective.py` draws (same seed, same
filter, same order), so the two can be paired. For each domain Gmsh (blossom
recombination, the strongest variant on curved domains) is searched to the
agent's element count and scored on the shape metric, raw and after the same
interior untangling the agent's mesh gets with the boundary pinned -- the
conservative treatment from `gmsh_untangled.py`. Reports the per-mesh minimum
median, p10 and usable count at the bar, the same three numbers as the agent.

Without `-agent`, `-target` sets one element count for every domain.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from utilities.score_quality_objective import SUITES, _median, _percentile  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-suite", required=True)
    parser.add_argument("-agent", default=None, help="scorer JSON with per-domain records")
    parser.add_argument("-target", default=0, type=int)
    parser.add_argument("-scale", default=1.0, type=float,
                        help="multiply the agent's element count by this before matching")
    parser.add_argument("-n", default=48, type=int)
    parser.add_argument("-seed", default=7, type=int)
    parser.add_argument("-usable_at", default=0.2, type=float)
    parser.add_argument("-out", default=None)
    args = parser.parse_args()

    from src.geo2d_bridge import import_geo2d, make_initializer, resmooth_mesh
    from utilities.compare_gmsh import VARIANTS, mesh_at_element_count
    geo2d = import_geo2d()

    per_domain = {}
    if args.agent:
        report = json.load(open(args.agent))["suites"][args.suite]
        per_domain = {d["draw"]: d for d in report["domains"]}

    options = dict(SUITES[args.suite]); options.setdefault("max_corners", 24)
    source = make_initializer(seed=args.seed, **options)
    rows = []
    last = max(per_domain) if per_domain else args.n - 1   # pair by draw index
    for draw in range(last + 1):
        try:
            source()
        except Exception:
            continue
        geometry = source.last_geometry
        record = per_domain.get(draw)
        if args.agent and record is None:
            continue
        target = int(round((record["elements"] if record else args.target) * args.scale))
        found = mesh_at_element_count(geometry, max(target, 1), VARIANTS["blossom"])
        if found is None:
            print(f"  draw {draw}: gmsh failed", flush=True)
            continue
        count, _, nodes, elements = found
        all_quad = all(len(e) == 4 for e in elements)
        mesh = geo2d.Mesh(nodes, elements)
        raw = float(np.min(np.asarray(mesh.quality(), float)))
        try:
            out = resmooth_mesh(geo2d.Mesh(nodes, elements), np.ones(len(nodes), bool),
                                iters=4, method="optimize", slide=False)
            fixed = float(np.min(np.asarray(out.quality(), float)))
        except Exception:
            fixed = raw
        rows.append(dict(draw=draw, target=target, elements=int(count), all_quad=all_quad,
                         raw=raw, fixed=fixed,
                         agent=(record["quality"] if record else None),
                         agent_elements=(record["elements"] if record else None)))
        print(f"  draw {draw:2d} target {target:3d} gmsh {count:3d}q "
              f"raw {raw:+.3f} untangled {fixed:+.3f}"
              + (f"  agent {record['quality']:+.3f}" if record else ""), flush=True)

    quads = [r for r in rows if r["all_quad"]]
    summary = {}
    for key in ("raw", "fixed"):
        values = [r[key] for r in quads]
        summary[key] = dict(median=_median(values), p10=_percentile(values, 10),
                            usable=sum(v >= args.usable_at for v in values),
                            all_quad=len(quads), total=len(rows))
    if args.agent:
        a = [r["agent"] for r in rows]
        summary["agent"] = dict(median=_median(a), p10=_percentile(a, 10),
                                usable=sum(v >= args.usable_at for v in a),
                                wins_vs_fixed=sum(r["agent"] > r["fixed"] for r in rows))
    print(f"=== gmsh blossom, {args.suite}, n={len(rows)} ===")
    for key, r in summary.items():
        print(f"   {key:6} q(med) {r['median']:+.3f} q(p10) {r['p10']:+.3f} usable {r['usable']}"
              + (f"  agent wins {r['wins_vs_fixed']}/{len(rows)}" if key == "agent" else ""))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump({"suite": args.suite, "summary": summary, "rows": rows},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
