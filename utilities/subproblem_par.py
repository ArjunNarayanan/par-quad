"""The bound of every sub-problem the agent meets while meshing a par-0 domain.

    venv/bin/python utilities/subproblem_par.py -config <cfg> -checkpoint <zip> \
        -suites straight,polycube,straight-holes,polycube-holes -n 24 -out <json>

Every geo2d domain has par 0, but the agent never meshes the domain in one piece:
each chord splits a face into two, and every unfinished face is a domain of its
own with its own Gauss-Bonnet bound. For a face F, a vertex on its loop already
has deg(v) edges, two of them F's own sides, so as a corner of F it wants
d(v) - deg(v) + 2 edges (at least two); F is a disc (a slit face is one too), so

    par(F) = | sum over F's loop of (want_F(v) - 3) + 4 |,

scanned over tie corners as in `compute_par`. Applied to the start state, which
is the whole domain as a single face, this must return the domain's own par; the
script counts any mismatch. A face whose loop visits a vertex twice -- the slit
that joins a hole to the outer boundary, until a chord separates its two sides --
has corners the formula cannot tell apart, so such faces are skipped and counted. A vertex shared by two unfinished faces is charged its
full remaining degree in each, which is the only assignment available before the
agent has decided where its edges go.

Each distinct unfinished face (by its vertex loop) is counted once per episode,
from the greedy attempt of the paper's procedure (budget doubled); the start face
is excluded, so what is counted are the sub-problems the agent created.
"""
import argparse
import collections
import json
import os
import sys
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())
from utilities.score_quality_objective import (MAX_EDGE_RATIO, MIN_CORNERS, SUITES,  # noqa: E402
                                               boundary_edge_ratio)


def face_par(env, face):
    g = env.graph
    loop = g.generate_half_edge_face_loop(g.first_face_halfedge(face))
    base, ties = 4, 0
    for h in loop:
        v = g.source_vertex(h, tag=False)
        options = env.desired_degrees(v)
        remaining = g.vertex_degree(v) - 2
        want = max(min(options) - remaining, 2)
        base += want - 3
        ties += len(options) - 1
    return min(abs(base + t) for t in range(ties + 1)), len(loop)


def unfinished(env):
    g = env.graph
    out = []
    for face in g.face_list():
        loop = g.generate_half_edge_face_loop(g.first_face_halfedge(face))
        if len(loop) == 4:
            continue
        verts = [g.source_vertex(h, tag=False) for h in loop]
        k = verts.index(min(verts))
        out.append((tuple(verts[k:] + verts[:k]), face, len(set(verts)) != len(verts)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-config", required=True)
    p.add_argument("-checkpoint", required=True)
    p.add_argument("-suites", required=True)
    p.add_argument("-n", default=24, type=int)
    p.add_argument("-seed", default=7, type=int)
    p.add_argument("-max_steps_factor", default=6.0, type=float)
    p.add_argument("-out", required=True)
    args = p.parse_args()

    from envs.environment_maker import initialize_environment
    from src.geo2d_bridge import load_model, make_initializer
    from src.utils import load_yaml_config
    config = load_yaml_config(args.config)
    env_config = dict(config["environment"])
    env_config["max_steps_factor"] = args.max_steps_factor
    model = load_model(args.checkpoint, args.config, template_size=env_config.get("template_size", 200))

    counts = collections.Counter()
    by_suite, episodes, mismatches = {}, [], 0
    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        options = dict(SUITES[suite])
        capped = "min_corners" in options
        options.setdefault("max_corners", 24)
        options.setdefault("min_corners", MIN_CORNERS)
        source = make_initializer(seed=args.seed, **options)
        built, draw = 0, -1
        suite_counts = collections.Counter()
        while built < args.n and draw < 40 * args.n:
            draw += 1
            try:
                graph, desired = source()
            except Exception:
                continue
            if capped and boundary_edge_ratio(graph) > MAX_EDGE_RATIO:
                continue
            built += 1

            class Fixed:
                def __call__(self):
                    return deepcopy(graph), dict(desired)
            cfg = dict(env_config)
            cfg.pop("initializer", None)
            cfg["graph_initializer"] = Fixed()
            cfg["resample_if_at_par"] = False
            env = initialize_environment(cfg)
            observation, _ = env.reset()
            start = unfinished(env)
            start_mismatch = None
            if len(start) == 1 and not start[0][2]:
                start_mismatch = face_par(env, start[0][1])[0] != env.par
                mismatches += int(start_mismatch)
            seen = {key for key, _, _ in start}
            skipped = 0
            local = collections.Counter()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, done, truncated, _ = env.step(action)
                for key, face, slit in unfinished(env):
                    if key in seen:
                        continue
                    seen.add(key)
                    if slit:
                        skipped += 1
                        continue
                    local[face_par(env, face)[0]] += 1
            counts.update(local); suite_counts.update(local)
            episodes.append(dict(suite=suite, draw=draw, par=int(env.par), slit_skipped=skipped,
                                 start_checked=start_mismatch is not None, start_mismatch=bool(start_mismatch),
                                 subproblems=sum(local.values()),
                                 nonzero=sum(c for k, c in local.items() if k > 0),
                                 hist={int(k): int(c) for k, c in local.items()}))
        by_suite[suite] = {int(k): int(c) for k, c in sorted(suite_counts.items())}
        print(f"{suite}: {built} episodes, sub-problem par histogram {dict(sorted(suite_counts.items()))}", flush=True)

    total = sum(counts.values())
    nonzero = sum(c for k, c in counts.items() if k > 0)
    with_nonzero = sum(e["nonzero"] > 0 for e in episodes)
    checked = sum(e["start_checked"] for e in episodes)
    skipped = sum(e["slit_skipped"] for e in episodes)
    print(f"=== {len(episodes)} episodes, {total} sub-problems, {nonzero} ({100 * nonzero / max(total, 1):.1f}%) "
          f"with par > 0; {with_nonzero} episodes contain one; start-face check: {mismatches} mismatches "
          f"of {checked} checked; {skipped} slit faces skipped")
    for k in sorted(counts):
        print(f"   par {k}: {counts[k]} ({100 * counts[k] / total:.1f}%)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(histogram={int(k): int(c) for k, c in sorted(counts.items())}, by_suite=by_suite,
                   episodes=episodes, start_face_mismatches=mismatches), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
