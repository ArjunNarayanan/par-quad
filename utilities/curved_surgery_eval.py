"""Surgery-and-continue on the curved suites, scored REGULARITY FIRST.

    venv/bin/python utilities/curved_surgery_eval.py \
        -config models/paper-2026-09-24/pinwheel-mitq-e2e-v1.config.yml \
        -checkpoint models/paper-2026-09-24/pinwheel-mitq-e2e-v1-4M.zip \
        -suite curved -n 24 -workers 4 -out out/surgery/released.curved.json

Per domain, from ONE set of agent rollouts (greedy + sampled, arcs densified to
45 degrees as in the paper's curved suites), three selections:

  quality-first   the paper scorer's rule: best min shape quality, then excess
  regular-first   fewest irregular vertices among meshes over the bar
  surgery         `src.surgery.surgery_search` from those same candidates

Reported per selection: all-quad, usable at 0.3 and 0.2, irregular per vertex
(median and mean over all-quad domains), at par, elements, quality median/p10.
"""

import argparse
import json
import os
import statistics
import sys
import time
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())


def suite_domains(suite, n, seed=7, densify=45.0, with_geometry=False):
    """The scorer's domain draws for `suite`, densified; [(draw, graph, desired)]
    (plus the geo2d geometry as a fourth entry with `with_geometry`)."""
    if suite == "frozen":
        # the ten real geo2d `rounded` draws frozen in src/curved_domains.json: the only
        # REAL geo2d geometry available without geogen (chosen as hard cases)
        from src.curved_levels import from_spec, load_domains
        from src.geo2d_bridge import densify_arcs
        out = []
        for draw, (name, spec) in enumerate(sorted(load_domains().items())):
            graph, desired = from_spec(spec)
            if densify:
                densify_arcs(graph, desired, densify)
            out.append((draw, graph, desired, None) if with_geometry else (draw, graph, desired))
        return out[:n]
    from src.geo2d_bridge import make_initializer
    from utilities.score_quality_objective import (MAX_EDGE_RATIO, MIN_CORNERS, SUITES,
                                                   boundary_edge_ratio)
    options = dict(SUITES[suite])
    capped = "min_corners" in options
    options.setdefault("max_corners", 24)
    options.setdefault("min_corners", MIN_CORNERS)
    source = make_initializer(seed=seed, densify_sweep=densify or None, **options)
    out, draw = [], -1
    while len(out) < n and draw < 40 * n:
        draw += 1
        try:
            graph, desired = source()
        except Exception:
            continue
        if capped and boundary_edge_ratio(graph) > MAX_EDGE_RATIO:
            continue
        out.append((draw, graph, desired, source.last_geometry) if with_geometry
                   else (draw, graph, desired))
    return out


_MODEL = {}


def _worker_init(config_path, checkpoint, template):
    import torch
    torch.set_num_threads(1)
    from src.geo2d_bridge import load_model
    _MODEL["model"] = load_model(checkpoint, config_path, template_size=template)


def run_domain(job):
    (draw, graph, desired, env_config, args) = job
    import random

    import torch
    from src import surgery as S
    random.seed(args["rollout_seed"] + draw)
    np.random.seed(args["rollout_seed"] + draw)
    torch.manual_seed(args["rollout_seed"] + draw)
    model = _MODEL["model"]
    env = S.domain_env(env_config, graph, desired, max_steps_factor=args["budget"])
    bar = args["bar"]
    t0 = time.time()
    found = S.rollouts(env, model, desired, None, n=args["n_initial"])
    par = int(env.par)
    rescued = False
    if not found and args.get("rescue", 0):
        # no all-quad state at all: more samples at a larger budget
        env = S.domain_env(env_config, graph, desired, max_steps_factor=args["budget"] * 1.5)
        found = S.rollouts(env, model, desired, None, n=args["rescue"], deterministic_first=False)
        rescued = bool(found)
    pool = {}
    for fp, (irr, g) in sorted(found.items(), key=lambda kv: kv[1][0])[:args["max_measure"]]:
        pool[fp] = S.measure(env, g, desired, "agent")
    t_agent = time.time() - t0

    def pack(c):
        if c is None:
            return None
        return dict(irr=c.irr, q=c.q, vertices=c.vertices, faces=c.faces, origin=c.origin,
                    irr_per_vertex=c.irr_per_vertex)

    rec = dict(draw=draw, par=par, corners=len(desired), candidates=len(pool), rescued=rescued)
    keep = {}
    if pool:
        cands = list(pool.values())
        qf = min(cands, key=lambda c: (-c.q, c.irr))
        rf = min(cands, key=lambda c: S.key_regular_first(c, bar))
        rec["quality_first"], rec["regular_first"] = pack(qf), pack(rf)
        keep.update(quality_first=qf.graph, regular_first=rf.graph)
    else:
        rec["quality_first"] = rec["regular_first"] = None
    t1 = time.time()
    if pool and args["rounds"] > 0:
        best, allc, stats = S.surgery_search(
            env, model, desired, bar=bar, n_continue=args["n_continue"], rounds=args["rounds"],
            beam=args["beam"], faces_per_seed=args["faces"], continue_budget=args["continue_budget"],
            moves=tuple(args["moves"].split(",")), pool=pool,
            polish=args["polish"], time_limit=args["time_limit"] or None)
        rec["surgery"] = pack(best)
        keep["surgery"] = best.graph
        # the chain that produced it, root first: (origin, seed graph, cut graph, result graph)
        chain, c = [], best
        while c is not None and c.parent is not None:
            chain.append(dict(origin=c.origin, seed=c.parent.graph, cut=c.cut, result=c.graph,
                              seed_q=c.parent.q, seed_irr=c.parent.irr, q=c.q, irr=c.irr))
            c = c.parent
        keep["chain"] = chain[::-1]
        rec["surgery_stats"] = stats
        # the full trade-off curve the search found: best quality at each irregularity
        front = {}
        for c in allc:
            if c.irr not in front or c.q > front[c.irr]["q"]:
                front[c.irr] = pack(c)
        rec["front"] = [front[k] for k in sorted(front)]
    else:
        rec["surgery"] = rec["regular_first"]
    rec["seconds"] = dict(agent=t_agent, surgery=time.time() - t1)
    if args.get("save_meshes"):
        import pickle
        os.makedirs(args["save_meshes"], exist_ok=True)
        with open(os.path.join(args["save_meshes"], f"draw{draw:03d}.pkl"), "wb") as handle:
            pickle.dump(dict(start=graph, desired=desired, par=par,
                             wants={k: S.wants_for(env, g, desired) for k, g in keep.items()
                                    if k != "chain"},
                             **keep), handle)
    return rec


def summarize(records, key, bar=0.3):
    rows = [r[key] for r in records]
    quad = [x for x in rows if x is not None]
    q = [x["q"] for x in quad]
    ipv = [x["irr_per_vertex"] for x in quad]
    usable = [x for x in quad if x["q"] >= bar]
    return dict(
        n=len(rows), all_quad=len(quad), usable=len(usable),
        usable_02=sum(x["q"] >= 0.2 for x in quad),
        irr_pv_median=statistics.median(ipv) if ipv else float("nan"),
        irr_pv_mean=statistics.mean(ipv) if ipv else float("nan"),
        irr_pv_usable_mean=(statistics.mean(x["irr_per_vertex"] for x in usable)
                            if usable else float("nan")),
        at_par=sum(x["irr"] == r["par"] for r, x in zip(records, rows) if x is not None),
        at_par_usable=sum(x["irr"] == r["par"] and x["q"] >= bar
                          for r, x in zip(records, rows) if x is not None),
        elements_median=statistics.median(x["faces"] for x in quad) if quad else float("nan"),
        q_median=statistics.median(q) if q else float("nan"),
        q_p10=float(np.percentile(q, 10)) if q else float("nan"),
        q_min=min(q) if q else float("nan"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-config", required=True)
    p.add_argument("-checkpoint", required=True)
    p.add_argument("-suite", default="curved")
    p.add_argument("-n", default=24, type=int)
    p.add_argument("-seed", default=7, type=int)
    p.add_argument("-rollout_seed", default=0, type=int)
    p.add_argument("-bar", default=0.3, type=float)
    p.add_argument("-budget", default=6.0, type=float, help="max_steps_factor for the initial rollouts")
    p.add_argument("-n_initial", default=5, type=int, help="greedy + (n-1) sampled rollouts")
    p.add_argument("-max_measure", default=60, type=int)
    p.add_argument("-rounds", default=6, type=int)
    p.add_argument("-beam", default=3, type=int)
    p.add_argument("-faces", default=2, type=int)
    p.add_argument("-n_continue", default=3, type=int)
    p.add_argument("-continue_budget", default=40, type=int)
    p.add_argument("-moves", default="sheet,insert,chord")
    p.add_argument("-densify", default=45.0, type=float)
    p.add_argument("-polish", default=0, type=int,
                   help="rounds of defect-driven surgery on usable meshes still above par")
    p.add_argument("-time_limit", default=0, type=float, help="seconds of search per domain (0: none)")
    p.add_argument("-rescue", default=0, type=int,
                   help="sampled rollouts at 1.5x budget when the first pass finds no all-quad mesh")
    p.add_argument("-workers", default=4, type=int)
    p.add_argument("-draws", default="", help="comma-separated draw indices to keep")
    p.add_argument("-save_meshes", default="", help="directory for per-domain pickles")
    p.add_argument("-out", default=None)
    args = p.parse_args()

    from src.utils import load_yaml_config
    config = load_yaml_config(args.config)
    env_config = dict(config["environment"])
    template = env_config.get("template_size", 200)
    domains = suite_domains(args.suite, args.n, args.seed, args.densify)
    if args.draws:
        keep = {int(x) for x in args.draws.split(",")}
        domains = [d for d in domains if d[0] in keep]
    jobs = [(d, g, w, env_config, vars(args)) for d, g, w in domains]

    records = []
    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers, initializer=_worker_init,
                      initargs=(args.config, args.checkpoint, template)) as pool:
            for rec in pool.imap_unordered(run_domain, jobs):
                records.append(rec)
                _print(rec)
    else:
        _worker_init(args.config, args.checkpoint, template)
        for job in jobs:
            rec = run_domain(job)
            records.append(rec)
            _print(rec)
    records.sort(key=lambda r: r["draw"])
    summary = {k: summarize(records, k, args.bar) for k in ("quality_first", "regular_first", "surgery")}
    print(f"\n=== {args.suite}  n={len(records)}  bar {args.bar} ===")
    for k, s in summary.items():
        print(f"  {k:14} quad {s['all_quad']:2d}/{s['n']}  usable {s['usable']:2d} (0.2: {s['usable_02']:2d})"
              f"  irr/v med {s['irr_pv_median']:.3f} mean {s['irr_pv_mean']:.3f}"
              f"  par {s['at_par']:2d} (usable {s['at_par_usable']:2d})  elems {s['elements_median']:.0f}"
              f"  q med {s['q_median']:+.2f} p10 {s['q_p10']:+.2f} min {s['q_min']:+.2f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(dict(args=vars(args), summary=summary, records=records), open(args.out, "w"), indent=1)


def _print(rec):
    def f(x):
        return "   --   " if x is None else f"{x['irr']:3d}/{x['vertices']:<3d} {x['q']:+.2f}"
    print(f"draw {rec['draw']:3d} par {rec['par']:2d}  quality-first {f(rec['quality_first'])}"
          f"  regular-first {f(rec['regular_first'])}  surgery {f(rec['surgery'])}"
          f"  ({rec['seconds']['agent']:.0f}s + {rec['seconds']['surgery']:.0f}s)"
          + (f"  [{rec['surgery']['origin']}]" if rec.get("surgery") else ""), flush=True)


if __name__ == "__main__":
    main()
