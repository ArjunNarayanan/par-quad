"""Why does a transfer domain fail? Two failure modes, diagnosed per domain.

    venv/bin/python utilities/transfer_failures.py -config <cfg> -checkpoint <zip> \
        -suite straight-transfer [-n 16] [-out <json>]

Reproduces the paper protocol (seed 7 domains, best of 4 sampled + 1 greedy, every all-quad
state untangled, quality-first selection) and then, for each domain:

  INCOMPLETE (no rollout ever reached all-quad): per rollout, the steps used against the
  budget, the number of non-quad faces left and their degrees, and how many odd faces the
  mesh ends with. Then a probe: the greedy rollout again with DOUBLE the step budget, to
  separate "ran out of moves" from "cannot close it".

  BELOW THE BAR (all-quad, min shape quality < 0.3 after the untangler): the worst corner's
  class (convex corner wanting 2, concave corner, inserted boundary vertex wanting 3,
  interior vertex), its aspect ratio, whether its face touches an irregular vertex, and
  the count of faces below the bar -- via `utilities/worst_element.classify_corner`.
"""
import argparse
import json
import os
import sys
from collections import Counter
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())
from utilities.score_quality_objective import SUITES, MAX_EDGE_RATIO, boundary_edge_ratio  # noqa: E402
from utilities.worst_element import classify_corner  # noqa: E402


def leftover(env):
    g = env.graph
    degs = sorted(int(g.face_degree(f)) for f in g.face_list() if g.face_degree(f) != 4)
    return dict(non_quad_faces=len(degs), degrees=degs[-6:], odd_faces=int(env.number_of_odd_faces()),
                faces=len(g.face_list()), steps=int(env.num_steps), max_steps=int(env.max_steps))


def rollout(env, model, deterministic, collect):
    obs, _ = env.reset()
    done = False
    while True:
        if collect is not None and int(env.global_face_score) == 0:
            collect.append((abs(int(env.global_vertex_score) - env.par), deepcopy(env.graph)))
        if done:
            break
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, trunc, _ = env.step(action)
        done = done or trunc
    return leftover(env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-config", required=True); p.add_argument("-checkpoint", required=True)
    p.add_argument("-suite", default="straight-transfer"); p.add_argument("-n", default=16, type=int)
    p.add_argument("-n_samples", default=4, type=int); p.add_argument("-seed", default=7, type=int)
    p.add_argument("-rollout_seed", default=0, type=int); p.add_argument("-bar", default=0.3, type=float)
    p.add_argument("-out", default=None)
    args = p.parse_args()

    from envs.environment_maker import initialize_environment
    from src.geo2d_bridge import make_initializer, resmooth_env, load_model
    from src.utils import load_yaml_config
    import random, torch
    config = load_yaml_config(args.config); env_config = dict(config["environment"]); env_config["quality_metric"] = "shape"
    model = load_model(args.checkpoint, args.config, template_size=env_config.get("template_size", 200))
    random.seed(args.rollout_seed); np.random.seed(args.rollout_seed); torch.manual_seed(args.rollout_seed)
    options = dict(SUITES[args.suite]); options.setdefault("max_corners", 24); options.setdefault("min_corners", 8)
    source = make_initializer(seed=args.seed, **options)

    records = []; built = 0; draw = -1
    while built < args.n and draw < 40 * args.n:
        draw += 1
        try:
            graph, desired = source()
        except Exception:
            continue
        if "min_corners" in SUITES[args.suite] and boundary_edge_ratio(graph) > MAX_EDGE_RATIO:
            continue
        built += 1

        class Fixed:
            def __init__(self): self.n = len(desired)
            def __call__(self): return deepcopy(graph), dict(desired)

        def make_env(factor=None):
            cfg = dict(env_config); cfg.pop("initializer", None); cfg["graph_initializer"] = Fixed(); cfg["resample_if_at_par"] = False
            if factor is not None:
                cfg["max_steps_factor"] = factor
            return initialize_environment(cfg)

        env = make_env()
        candidates, ends = [], []
        for trial in range(args.n_samples + 1):
            ends.append(rollout(env, model, trial == 0, candidates))
        rec = dict(draw=draw, corners=len(desired), par=int(env.par), rollouts=ends)
        if not candidates:
            rec["mode"] = "incomplete"
            # budget probe: greedy again with twice the budget
            env2 = make_env(factor=2.0 * float(env_config.get("max_steps_factor", 1.5)))
            probe = []
            end2 = rollout(env2, model, True, probe)
            rec["double_budget"] = dict(completed=bool(probe), **end2)
            worst = min(ends, key=lambda e: e["non_quad_faces"])
            print(f"draw {draw:3d} corners {len(desired):2d} INCOMPLETE: best rollout leaves {worst['non_quad_faces']} non-quad faces "
                  f"(degrees {worst['degrees']}), odd {worst['odd_faces']}, steps {worst['steps']}/{worst['max_steps']}; "
                  f"double budget -> {'COMPLETES' if probe else 'still ' + str(end2['non_quad_faces']) + ' open'} at {end2['steps']}/{end2['max_steps']}", flush=True)
        else:
            best = None
            for excess, cand in candidates:
                env.graph = cand; env._update_half_edge_angles()
                try:
                    resmooth_env(env)
                except Exception:
                    pass
                key = (-float(env.min_element_quality()), excess)
                if best is None or key < best[0]:
                    best = (key, deepcopy(env.graph))
            env.graph = best[1]; env._update_half_edge_angles()
            q = float(env.min_element_quality()); rec["quality"] = q; rec["excess"] = best[0][1]
            if q < args.bar:
                rec["mode"] = "below_bar"
                corners = env.graph.corner_shape_qualities(); worst = min(corners, key=corners.get)
                rec["worst"] = classify_corner(env, worst, corners)
                faces = {}
                for h in corners:
                    if corners[h] < args.bar:
                        f = env.graph.face(h)
                        if f not in faces or corners[h] < corners[faces[f]]:
                            faces[f] = h
                rec["faces_below_bar"] = [classify_corner(env, h, corners) for h in faces.values()]
                w = rec["worst"]
                print(f"draw {draw:3d} corners {len(desired):2d} BELOW BAR q {q:+.3f} exc {rec['excess']}: worst at {w['where']}, "
                      f"deg {w['degree']} wants {w['wants']}, aspect {w['aspect']:.1f}, face irregular {w['face_irregular_vertices']}, "
                      f"faces<bar {len(faces)}", flush=True)
            else:
                rec["mode"] = "usable"
                print(f"draw {draw:3d} corners {len(desired):2d} usable q {q:+.3f} exc {rec['excess']}", flush=True)
        records.append(rec)

    modes = Counter(r["mode"] for r in records)
    print(f"\n{args.suite}: {dict(modes)}")
    inc = [r for r in records if r["mode"] == "incomplete"]
    if inc:
        print("INCOMPLETE:", len(inc), "domains;", "double budget completes", sum(r["double_budget"]["completed"] for r in inc))
        print("  best-rollout non-quad faces left:", sorted(min(e["non_quad_faces"] for e in r["rollouts"]) for r in inc))
        print("  rollouts ending at the step cap:", sum(e["steps"] >= e["max_steps"] for r in inc for e in r["rollouts"]), "of", sum(len(r["rollouts"]) for r in inc))
        print("  largest leftover face degree (best rollout):", sorted(max(min(r["rollouts"], key=lambda e: e["non_quad_faces"])["degrees"] or [0]) for r in inc))
    bb = [r for r in records if r["mode"] == "below_bar"]
    if bb:
        print("BELOW BAR:", len(bb), "domains; worst corner by class:", Counter(r["worst"]["where"] for r in bb))
        print("  worst corner at an irregular vertex:", sum(r["worst"]["vertex_irregular"] for r in bb), "; face touches irregular:", sum(r["worst"]["face_irregular_vertices"] > 0 for r in bb),
              "; aspect >= 3:", sum(r["worst"]["aspect"] >= 3 for r in bb), "; aspect median %.1f" % np.median([r["worst"]["aspect"] for r in bb]))
        allf = [f for r in bb for f in r["faces_below_bar"]]
        print("  all faces below bar:", len(allf), Counter(f["where"] for f in allf), "; touching irregular", sum(f["face_irregular_vertices"] > 0 for f in allf), "; one boundary side", sum(f["face_boundary_sides"] == 1 for f in allf))
    if args.out:
        json.dump(dict(args=vars(args), records=records), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
