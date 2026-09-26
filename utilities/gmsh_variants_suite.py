"""All seven Gmsh variants on a scorer suite, matched domain by domain to the agent's
element count, scored on the agent's scoreboard: all-quad, usable, quality, at par,
excess over par, irregular vertices per vertex.

    venv/bin/python utilities/gmsh_variants_suite.py -suite straight \
        -agent out/paper/pinwheel/straight.0.json -out out/paper/gmsh/straight.json

Draws the domains exactly as `score_quality_objective.py` (same seed, same filter,
same order; transfer suites pair by draw index), searches each variant to the
agent's element count, untangles with the boundary fixed as `gmsh_curved_suite.py`
does, and reads the vertex score with the wants from the tangents.
"""
import argparse
import json
import os
import statistics
import sys

import numpy as np

sys.path.append(os.getcwd())
from utilities.score_quality_objective import SUITES, _percentile  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-suite", required=True)
    parser.add_argument("-agent", required=True)
    parser.add_argument("-seed", default=7, type=int)
    parser.add_argument("-variants", default=None)
    parser.add_argument("-usable_at", default=0.3, type=float)
    parser.add_argument("-out", default=None)
    args = parser.parse_args()
    from src.geo2d_bridge import import_geo2d, make_initializer, resmooth_mesh
    from utilities.compare_gmsh import VARIANTS, mesh_at_element_count, score_mesh
    from utilities.compare_gmsh_curved import curved_corner_wants
    geo2d = import_geo2d()
    variants = [v for v in (args.variants.split(",") if args.variants else VARIANTS) if v in VARIANTS]
    records = {d["draw"]: d for d in json.load(open(args.agent))["suites"][args.suite]["domains"]}
    options = dict(SUITES[args.suite]); options.setdefault("max_corners", 24)
    source = make_initializer(seed=args.seed, **options)
    rows = {v: [] for v in variants}
    for draw in range(max(records) + 1):
        try:
            source()
        except Exception:
            continue
        rec = records.get(draw)
        if rec is None:
            continue
        geometry = source.last_geometry
        try:
            wants, tol = curved_corner_wants(geometry)
        except Exception:
            wants, tol = None, None
        for v in variants:
            found = mesh_at_element_count(geometry, max(int(rec["elements"]), 1), VARIANTS[v])
            if found is None:
                rows[v].append(dict(draw=draw, failed=True)); continue
            count, _, nodes, elements = found
            all_quad = all(len(e) == 4 for e in elements)
            quality = float(np.min(np.asarray(geo2d.Mesh(nodes, elements).quality(), float)))
            try:
                out = resmooth_mesh(geo2d.Mesh(nodes, elements), np.ones(len(nodes), bool), iters=4, method="optimize", slide=False)
                quality = float(np.min(np.asarray(out.quality(), float)))
            except Exception:
                pass
            vscore = None
            if wants is not None:
                try:
                    vscore = int(score_mesh(nodes, elements, wants, tol, rec["par"])["vertex_score"])
                except Exception:
                    vscore = None
            rows[v].append(dict(draw=draw, failed=False, all_quad=all_quad, quality=quality, elements=int(count),
                                vertices=len(nodes), par=rec["par"], vertex_score=vscore,
                                agent_elements=rec["elements"], agent_quality=rec["quality"]))
        print(f"  draw {draw:2d} done", flush=True)
    summary = {}
    for v in variants:
        r = rows[v]; ok = [x for x in r if not x["failed"]]; quad = [x for x in ok if x["all_quad"]]
        q = [x["quality"] for x in quad]
        exc = [x["vertex_score"] - x["par"] for x in quad if x["vertex_score"] is not None]
        frac = [x["vertex_score"] / max(x["vertices"], 1) for x in quad if x["vertex_score"] is not None]
        summary[v] = dict(n=len(r), all_quad=len(quad), usable=sum(x >= args.usable_at for x in q),
                          usable_02=sum(x >= 0.2 for x in q),
                          quality_median=statistics.median(q) if q else None, quality_p10=_percentile(q, 10) if q else None,
                          at_par=sum(e == 0 for e in exc), excess_median=statistics.median(exc) if exc else None,
                          excess_mean=statistics.mean(exc) if exc else None,
                          irregular_per_vertex_median=statistics.median(frac) if frac else None,
                          elements_mean=statistics.mean(x["elements"] for x in ok) if ok else None)
        s = summary[v]
        print(f"=== {args.suite} {v:18s} n {s['n']:2d} all-quad {s['all_quad']:2d} usable {s['usable']:2d} q med {s['quality_median'] if s['quality_median'] is None else round(s['quality_median'],3)} "
              f"at par {s['at_par']} exc med {s['excess_median']} irr/vtx {s['irregular_per_vertex_median'] if s['irregular_per_vertex_median'] is None else round(s['irregular_per_vertex_median'],3)}")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump({"suite": args.suite, "summary": summary, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
