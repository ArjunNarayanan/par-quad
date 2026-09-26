"""Split-and-continue repair at evaluation: force one move at the worst corner, let the agent finish.

    venv/bin/python utilities/repair_eval.py -config <cfg> -checkpoint <zip> -suite straight-transfer \
        [-rounds 3] [-bar 0.3] [-n 16] [-n_samples 4] -out <json>

Arjun's idea (2026-09-23): the below-bar meshes are slivers and bad angles at boundary corners.
After the paper protocol picks its best mesh, if its minimum shape quality is under the bar,
take the worst corner and apply the env's own move that breaks it -- a VERTEX INSERT on the
longer of its two sides when the corner is thin (aspect >= `-aspect`), a CHORD from the corner
to the opposite vertex when it is a bad angle -- then re-enter the episode from that mesh
(`AngleEnv._reset_to_state`) and let the policy run to termination, greedy plus `-n_samples`
sampled rollouts, every all-quad state untangled and the best kept. Repeat up to `-rounds`
times from the best mesh so far. Reports, per domain, quality before and after, rounds used,
and elements added; and the suite totals before and after, so the trick is judged on the
same protocol as everything else.
"""
import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())
from utilities.score_quality_objective import SUITES, MAX_EDGE_RATIO, boundary_edge_ratio  # noqa: E402


def wants_for(env, graph, base):
    """The want table for `graph`: the domain's own for its original vertices, and the
    env's rule for every vertex a move added since -- degree three on the boundary, four
    inside (what `_step_insert_vertex` registers)."""
    table = dict(base)
    for v in graph.vertex_list(tag=False):
        if v not in table:
            table[v] = env.boundary_vertex_desired_degree if graph.is_boundary_vertex(v) else env.interior_vertex_desired_degree
    return table


def best_all_quad(env, model, n_samples, from_graph=None, desired=None):
    """Run greedy + n_samples rollouts (from the env's start, or from `from_graph`), untangle
    every all-quad state, return (quality, excess, graph) of the best, or None."""
    from src.geo2d_bridge import resmooth_env
    candidates = []
    for trial in range(n_samples + 1):
        if from_graph is None:
            obs, _ = env.reset()
        else:
            start = deepcopy(from_graph)
            obs, _ = env._reset_to_state(start, wants_for(env, start, desired))
        done = False
        while True:
            if int(env.global_face_score) == 0:
                candidates.append((abs(int(env.global_vertex_score) - env.par), deepcopy(env.graph)))
            if done:
                break
            action, _ = model.predict(obs, deterministic=(trial == 0))
            obs, _, done, trunc, _ = env.step(action); done = done or trunc
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
    if best is None:
        return None
    return -best[0][0], best[0][1], best[1]


def forced_move(env, graph, aspect_bar, mode="split"):
    """Apply one repair move to `graph` at its worst corner. Returns a description.

    mode "split": insert on the long side of a thin corner, chord across a bad angle (the
    original). mode "delete": remove the INTERIOR side of the worst corner, merging the sliver
    with its neighbour into a hexagon and lowering the corner vertex's degree by one -- the
    move an over-connected boundary vertex (five chords fanned into a notch tip wanting three)
    needs and the monotone vocabulary lacks; falls back to "split" when no interior side can be
    deleted. mode "auto": "delete" when the corner's vertex is above its want, else "split"."""
    env.graph = graph; env._update_half_edge_angles()
    env.vertex_desired_degree = wants_for(env, graph, env.vertex_desired_degree)
    env.vertex_desired_options = env._compute_desired_options()
    corners = graph.corner_shape_qualities()
    h = min(corners, key=corners.get)
    prev = graph.previous_half_edge(h)
    v = graph.source_vertex(h, tag=False)
    c = np.asarray(graph.vertex_coordinate(v), float)
    a = np.asarray(graph.vertex_coordinate(graph.target_vertex(h, tag=False)), float) - c
    b = np.asarray(graph.vertex_coordinate(graph.source_vertex(prev, tag=False)), float) - c
    la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    aspect = max(la, lb) / max(min(la, lb), 1e-12)
    over = graph.vertex_degree(v) > max(env.desired_degrees(v)) if graph.is_boundary_vertex(v) else graph.vertex_degree(v) > 4
    if mode == "delete" or (mode == "auto" and over):
        # prefer the interior side that is a chord (twin exists); if both are interior, the shorter
        # one, since the long one is the sliver's spine and deleting the short one merges across it
        sides = [(la, h), (lb, prev)]
        sides = [(L, e) for L, e in sides if not graph.is_boundary_half_edge(graph.twin_half_edge(e))]
        sides.sort()
        for L, e in sides:
            if graph.is_valid_delete_half_edge(e):
                graph.delete_half_edge(e)
                return f"delete interior side at the corner (deg {graph.vertex_degree(v) + 1}->{graph.vertex_degree(v)}, aspect {aspect:.1f}, q {corners[h]:+.2f})"
        # no deletable side: fall through to the split move
    if aspect >= aspect_bar:
        long_edge = h if la >= lb else prev
        graph.insert_vertex(long_edge)
        return f"insert on long side (aspect {aspect:.1f}, q {corners[h]:+.2f})"
    face = graph.face(h)
    if graph.face_degree(face) >= 4 and graph.is_valid_chord_insert(h, 1, 3):
        graph.insert_half_edge(h, 1)
        return f"chord across the corner (aspect {aspect:.1f}, q {corners[h]:+.2f})"
    graph.insert_vertex(h if la >= lb else prev)
    return f"insert (fallback; aspect {aspect:.1f}, q {corners[h]:+.2f})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-config", required=True); p.add_argument("-checkpoint", required=True)
    p.add_argument("-suite", required=True); p.add_argument("-n", default=16, type=int)
    p.add_argument("-n_samples", default=4, type=int); p.add_argument("-seed", default=7, type=int)
    p.add_argument("-rollout_seed", default=0, type=int); p.add_argument("-bar", default=0.3, type=float)
    p.add_argument("-rounds", default=3, type=int); p.add_argument("-aspect", default=2.0, type=float)
    p.add_argument("-mode", default="split", choices=["split", "delete", "auto"])
    p.add_argument("-out", default=None)
    args = p.parse_args()

    from envs.environment_maker import initialize_environment
    from src.geo2d_bridge import make_initializer, load_model
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

        cfg = dict(env_config); cfg.pop("initializer", None); cfg["graph_initializer"] = Fixed(); cfg["resample_if_at_par"] = False
        env = initialize_environment(cfg)
        first = best_all_quad(env, model, args.n_samples)
        rec = dict(draw=draw, corners=len(desired))
        if first is None:
            rec.update(before=None, after=None, rounds=0, note="incomplete")
            print(f"draw {draw:3d}: incomplete", flush=True); records.append(rec); continue
        q0, e0, g0 = first
        rec.update(before=q0, before_excess=e0, before_elements=len(g0.face_list()))
        q, e, g = q0, e0, g0; moves = []
        for r in range(args.rounds):
            if q >= args.bar:
                break
            trial = deepcopy(g)
            moves.append(forced_move(env, trial, args.aspect, args.mode))
            cont = best_all_quad(env, model, args.n_samples, from_graph=trial, desired=desired)
            if cont is None:
                moves[-1] += " -> agent could not close"; break
            q1, e1, g1 = cont
            if q1 > q:
                q, e, g = q1, e1, g1
            else:
                moves[-1] += f" -> no gain ({q1:+.2f})"
        rec.update(after=q, after_excess=e, after_elements=len(g.face_list()), rounds=len(moves), moves=moves)
        print(f"draw {draw:3d}: {q0:+.3f} -> {q:+.3f}  exc {e0}->{e}  elements {rec['before_elements']}->{rec['after_elements']}  " + " | ".join(moves), flush=True)
        records.append(rec)

    done = [r for r in records if r["before"] is not None]
    print(f"\n{args.suite}: all-quad {len(done)}/{len(records)}; usable before {sum(r['before'] >= args.bar for r in done)} -> after {sum(r['after'] >= args.bar for r in done)}; "
          f"median before {np.median([r['before'] for r in done]):+.3f} -> after {np.median([r['after'] for r in done]):+.3f}; "
          f"at par before {sum(r['before_excess'] == 0 for r in done)} -> after {sum(r['after_excess'] == 0 for r in done)}; "
          f"repaired domains {sum(r['rounds'] > 0 for r in done)}, improved {sum(r['after'] > r['before'] for r in done)}")
    if args.out:
        json.dump(dict(args=vars(args), records=records), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
