"""Where is the worst element of a transfer mesh, and what is it touching?

    venv/bin/python utilities/worst_element.py -config <cfg> -checkpoint <zip> \
        -suite curved-transfer -scores <scorer json> -below 0.3 [-limit 12] -out <json>

Reproduces the scorer's protocol (same domain seed, same draw sequence, best of
`n_samples` + greedy, every all-quad state untangled and ranked quality-first) on
the domains whose scored quality is below `-below`, then classifies the WORST
CORNER of the selected mesh: is its vertex on the boundary, on an arc joint, at a
convex or concave corner; are its two sides boundary edges; is the vertex or the
face irregular; the tangent angle and the aspect ratio of the two sides. Also
counts every corner below the bar by the same classes, so the tail is described,
not just its minimum. The scorer records no meshes, which is why this reruns.
"""
import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())
from utilities.score_quality_objective import SUITES, MAX_EDGE_RATIO, boundary_edge_ratio  # noqa: E402


def classify_corner(env, h, corners):
    g = env.graph
    v = g.source_vertex(h, tag=False)
    prev = g.previous_half_edge(h)
    ahead = g.target_vertex(h, tag=False)
    behind = g.source_vertex(prev, tag=False)
    arcs = getattr(g, "boundary_arcs", None)
    out_boundary = g.is_boundary_half_edge(g.twin_half_edge(h))
    in_boundary = g.is_boundary_half_edge(g.twin_half_edge(prev))
    boundary_vertex = g.is_boundary_vertex(v)
    wants = sorted(env.desired_degrees(v)) if boundary_vertex else [4]
    degree = g.vertex_degree(v)
    c = np.asarray(g.vertex_coordinate(v), float)
    a = np.asarray(g.vertex_coordinate(ahead), float) - c
    b = np.asarray(g.vertex_coordinate(behind), float) - c
    aspect = float(max(np.linalg.norm(a), np.linalg.norm(b)) / max(min(np.linalg.norm(a), np.linalg.norm(b)), 1e-12))
    on_arc = bool(arcs and (arcs.get(v, ahead) is not None or arcs.get(v, behind) is not None))
    face = g.face(h)
    loop = g.face_half_edges(face)
    face_vertices = [g.source_vertex(x, tag=False) for x in loop]
    irregular_in_face = 0
    for fv in face_vertices:
        w = env.desired_degrees(fv) if g.is_boundary_vertex(fv) else {4}
        if g.vertex_degree(fv) not in w:
            irregular_in_face += 1
    boundary_sides = sum(g.is_boundary_half_edge(g.twin_half_edge(x)) for x in loop)
    if not boundary_vertex:
        where = "interior vertex"
    elif on_arc and min(wants) >= 3 and max(wants) <= 3:
        where = "arc joint (wants 3)"
    elif min(wants) <= 2:
        where = "convex corner (wants 2)"
    elif min(wants) >= 4:
        where = "concave corner (wants >= 4)"
    else:
        where = "flat straight vertex (wants 3)"
    return dict(quality=float(corners[h]), where=where, on_arc=on_arc, wants=wants, degree=degree,
                vertex_irregular=degree not in set(wants), sides_on_boundary=int(in_boundary) + int(out_boundary),
                aspect=aspect, face_boundary_sides=int(boundary_sides), face_irregular_vertices=irregular_in_face,
                face_degree=len(loop))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", required=True)
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-suite", default="curved-transfer")
    parser.add_argument("-scores", required=True, help="scorer json; picks the draws to rerun")
    parser.add_argument("-below", default=0.3, type=float)
    parser.add_argument("-limit", default=12, type=int)
    parser.add_argument("-n", default=48, type=int)
    parser.add_argument("-n_samples", default=4, type=int)
    parser.add_argument("-seed", default=7, type=int)
    parser.add_argument("-densify", default=45.0, type=float)
    parser.add_argument("-bar", default=0.3, type=float)
    parser.add_argument("-out", required=True)
    args = parser.parse_args()

    from envs.environment_maker import initialize_environment
    from src.geo2d_bridge import make_initializer, resmooth_env, load_model
    from src.utils import load_yaml_config

    scored = json.load(open(args.scores))["suites"][args.suite]["domains"]
    picked = sorted([d for d in scored if d["face"] == 0 and d["quality"] < args.below], key=lambda d: d["quality"])[:args.limit]
    wanted = {d["draw"]: d for d in picked}
    print(f"{len(wanted)} domains below {args.below}: draws {sorted(wanted)}", flush=True)

    config = load_yaml_config(args.config)
    env_config = dict(config["environment"])
    env_config["quality_metric"] = "shape"
    model = load_model(args.checkpoint, args.config, template_size=env_config.get("template_size", 200))
    import random, torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    options = dict(SUITES[args.suite]); options.setdefault("max_corners", 24)
    source = make_initializer(seed=args.seed, densify_sweep=args.densify or None, **options)

    results = []
    built, draw = 0, -1
    while built < args.n and draw < 40 * args.n and len(results) < len(wanted):
        draw += 1
        try:
            graph, desired = source()
        except Exception:
            continue
        if boundary_edge_ratio(graph) > MAX_EDGE_RATIO:
            continue
        built += 1
        if draw not in wanted:
            continue

        class Fixed:
            def __init__(self):
                self.n = len(desired)

            def __call__(self):
                return deepcopy(graph), dict(desired)

        cfg = dict(env_config); cfg.pop("initializer", None)
        cfg["graph_initializer"] = Fixed(); cfg["resample_if_at_par"] = False
        env = initialize_environment(cfg)
        candidates = []
        for trial in range(args.n_samples + 1):
            observation, _ = env.reset(); done = False
            while True:
                if int(env.global_face_score) == 0:
                    candidates.append((abs(int(env.global_vertex_score) - env.par), deepcopy(env.graph)))
                if done:
                    break
                action, _ = model.predict(observation, deterministic=(trial == 0))
                observation, _, done, truncated, _ = env.step(action); done = done or truncated
        best = None
        for excess, candidate in candidates:
            env.graph = candidate; env._update_half_edge_angles()
            try:
                resmooth_env(env)
            except Exception:
                pass
            key = (-float(env.min_element_quality()), excess)
            if best is None or key < best[0]:
                best = (key, deepcopy(env.graph))
        if best is None:
            results.append(dict(draw=draw, scored=wanted[draw]["quality"], all_quad=False)); continue
        env.graph = best[1]; env._update_half_edge_angles()
        corners = env.graph.corner_shape_qualities()
        worst = min(corners, key=corners.get)
        record = dict(draw=draw, scored=wanted[draw]["quality"], rerun=float(env.min_element_quality()),
                      excess=best[0][1], corners=wanted[draw]["corners"], elements=len(env.graph.face_list()),
                      worst=classify_corner(env, worst, corners))
        below = [classify_corner(env, h, corners) for h in corners if corners[h] < args.bar]
        # one entry per FACE below the bar, at its worst corner
        faces = {}
        for h in corners:
            if corners[h] < args.bar:
                f = env.graph.face(h)
                if f not in faces or corners[h] < corners[faces[f]]:
                    faces[f] = h
        record["faces_below_bar"] = [classify_corner(env, h, corners) for h in faces.values()]
        record["corners_below_bar"] = len(below)
        results.append(record)
        w = record["worst"]
        print(f"draw {draw:3d} scored {record['scored']:+.3f} rerun {record['rerun']:+.3f} exc {record['excess']:2d} "
              f"faces<{args.bar}: {len(faces):2d}  worst: {w['where']}, sides on boundary {w['sides_on_boundary']}, "
              f"deg {w['degree']} wants {w['wants']}, aspect {w['aspect']:.1f}, face irregular {w['face_irregular_vertices']}", flush=True)

    json.dump(dict(args=vars(args), results=results), open(args.out, "w"), indent=1)
    # summary over the worst corners and over every face below the bar
    from collections import Counter
    ok = [r for r in results if "worst" in r]
    print("\nWORST corner by class:", Counter(r["worst"]["where"] for r in ok))
    print("worst corner at an irregular vertex:", sum(r["worst"]["vertex_irregular"] for r in ok), "of", len(ok))
    print("worst face touching an irregular vertex:", sum(r["worst"]["face_irregular_vertices"] > 0 for r in ok), "of", len(ok))
    print("worst corner with both sides on the boundary:", sum(r["worst"]["sides_on_boundary"] == 2 for r in ok), "of", len(ok))
    allf = [f for r in ok for f in r["faces_below_bar"]]
    print(f"\nALL faces below {args.bar} ({len(allf)}):", Counter(f["where"] for f in allf))
    print("  touching an irregular vertex:", sum(f["face_irregular_vertices"] > 0 for f in allf), "; at an irregular vertex:", sum(f["vertex_irregular"] for f in allf))
    print("  face boundary sides:", Counter(f["face_boundary_sides"] for f in allf))
    print("  aspect median:", float(np.median([f["aspect"] for f in allf])) if allf else None)


if __name__ == "__main__":
    main()
