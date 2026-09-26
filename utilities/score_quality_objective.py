"""Score the objective "lowest irregularity at acceptable quality".

    venv/bin/python utilities/score_quality_objective.py -checkpoint <zip> -steps N

Solve rate is the wrong instrument for this. par is not always reachable -- the
par-0 mesh of some curved domains inverts -- so a scoreboard that only counts
par hides both the good outcome (a clean mesh two off par) and the bad one (a
par mesh that is degenerate). Two numbers instead:

Reported per suite, as components rather than one verdict, because a binary
hides the movement that matters:

  QUAD     how many came out all-quad at all
  Q(MED)   median per-mesh MINIMUM shape quality, over the all-quad ones
  Q(P10)   the tenth percentile of the same -- the tail is what fails downstream
  EXC(MED) median irregularity above par
  PAR      how many reached par exactly
  USABLE   all-quad AND quality >= the bar; the old single number, kept

The quality is the SHAPE metric, `2J / (|a|^2 + |b|^2)`, which geo2d uses and
which sees aspect ratio. The angle metric this used to report cannot: it scores
a comb of paper-thin quads perfectly, and on one checkpoint's own meshes it read
0.707 where geo2d read 0.056. For an engineering mesh that difference is the
whole question, so par is demoted to a tiebreak and quality decides.

Reported per suite, straight and curved, hole-free and holed, so a gain on one
cannot hide a loss on another. A run that trades a point of excess for a
collapsed mesh is going backwards and this says so.
"""

import argparse
import numpy as np
import json
import os
import sys
from copy import deepcopy

sys.path.append(os.getcwd())

SUITES = {
    "straight":       dict(preset="straight", n_holes=0, allow_curves=False),
    "straight-holes": dict(preset="straight", n_holes=[1, 2], ratio=8, allow_curves=False),
    "polycube":       dict(preset="polycube", n_holes=0, allow_curves=False),
    "curved":         dict(preset="rounded", n_holes=0, allow_curves=True),
    "curved-holes":   dict(preset="rounded", n_holes=[1, 2], allow_curves=True),
    # TRANSFER suites: 25-50 corners, out of the 8-24 training range. The size
    # comes from the generator's own ratio / n_ops / n_mods (max_corners is only
    # a rejection filter, see utilities/transfer_set.py), the corner window from
    # min/max_corners, and MAX_EDGE_RATIO caps the boundary edge-length ratio so a
    # drop here is SIZE and not geometric extremity (same cap as transfer_set.py).
    "straight-transfer":     dict(preset="straight", n_holes=0, ratio=24, n_ops=(6, 12), n_mods=(4, 8),
                                  allow_curves=False, min_corners=25, max_corners=50),
    "curved-transfer":       dict(preset="rounded", n_holes=0, ratio=24, n_ops=(6, 12), n_mods=(4, 8),
                                  allow_curves=True, min_corners=25, max_corners=50),
    "curved-holes-transfer": dict(preset="rounded", n_holes=[1, 2], ratio=24, n_ops=(6, 12), n_mods=(4, 8),
                                  allow_curves=True, min_corners=25, max_corners=50),
    # the paper's in-distribution fourth family, and the 25-50 corner versions of the
    # two families above that were missing: the "larger boundaries" set is these four
    # straight-sided transfer suites at 16 domains each
    "polycube-holes":          dict(preset="polycube", n_holes=[1, 2], allow_curves=False),
    "straight-holes-transfer": dict(preset="straight", n_holes=[1, 2], ratio=24, n_ops=(6, 12), n_mods=(4, 8),
                                    allow_curves=False, min_corners=25, max_corners=50),
    "polycube-transfer":       dict(preset="polycube", n_holes=0, ratio=24, n_ops=(6, 12),
                                    allow_curves=False, min_corners=25, max_corners=50),
    "polycube-holes-transfer": dict(preset="polycube", n_holes=[1, 2], ratio=24, n_ops=(6, 12),
                                    allow_curves=False, min_corners=25, max_corners=50),
}
MAX_EDGE_RATIO = 10.0   # applied to the transfer suites only
# The paper's floor: `73b47a4` retired every domain below 8 corners from the
# benchmark, and `score_suites.py` applies it, but this scorer only set a floor on
# the transfer suites -- so `straight-holes` carried a bare triangle (draw 8) and a
# 6-gon (draw 22) into the paper's Table 1 set. The other three in-distribution
# suites never drew below 8, so applying the floor here changes straight-holes only.
MIN_CORNERS = 8


def boundary_edge_ratio(graph):
    """Longest over shortest boundary edge of the start polygon, on the Tiler the
    agent is handed (so arcs, densified, count by their chords). The same cap as
    `transfer_set.edge_length_ratio`, which only reads straight geometries."""
    import numpy as np
    lengths = []
    for h in graph.half_edge_list():
        if graph.is_boundary_half_edge(graph.twin_half_edge(h)):
            a = np.asarray(graph.vertex_coordinate(graph.source_vertex(h, tag=False)), float)
            b = np.asarray(graph.vertex_coordinate(graph.target_vertex(h, tag=False)), float)
            lengths.append(float(np.hypot(*(b - a))))
    return max(lengths) / max(min(lengths), 1e-12) if lengths else float("inf")


def _median(values):
    if not values:
        return float("nan")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float(0.5 * (ordered[middle - 1] + ordered[middle]))


def _percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def _save_selected_mesh(directory, suite, env, record, base_wants):
    """Pickle the mesh the protocol selected, with what a figure needs to draw it
    exactly as it was scored: node coordinates, face loops, each vertex's degree
    and the degrees it wants (ties as a set), each face's shape quality, and the
    per-domain record. Irregular vertices are recomputed from these and checked
    against `record["irregular"]` by the figure script."""
    import pickle
    from src.geo2d_bridge import tiler_faces
    from utilities.repair_eval import wants_for
    graph = env.graph
    # the env's want table belongs to the LAST trial's episode, not necessarily the
    # selected mesh's; rebuild it for this graph as the repair path does
    env.vertex_desired_degree = wants_for(env, graph, base_wants)
    env.vertex_desired_options = env._compute_desired_options()
    nodes, elements, index = tiler_faces(graph)
    corner_quality = graph.corner_shape_qualities()
    face_quality = []
    for face in graph.face_list():
        loop = graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))
        values = [corner_quality[h] for h in loop if h in corner_quality]
        face_quality.append(float(min(values)) if values else float("nan"))
    vertices = []
    for vertex, i in index.items():
        vertices.append(dict(node=i, degree=int(graph.vertex_degree(vertex)),
                             wants=tuple(int(w) for w in env.desired_degrees(vertex))))
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{suite}.draw{record['draw']:03d}.pkl")
    with open(path, "wb") as handle:
        pickle.dump(dict(suite=suite, record=dict(record), nodes=np.asarray(nodes, float),
                         elements=[list(e) for e in elements], face_quality=face_quality,
                         vertices=vertices), handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config",
                        default="experiments/self-play/quad/all-domains-v1/config.yml")
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-n", default=16, type=int)
    parser.add_argument("-n_samples", default=4, type=int)
    parser.add_argument("-seed", default=7, type=int)
    parser.add_argument("-rollout_seed", default=0, type=int,
                        help="seeds the SAMPLED rollouts only; the domains stay fixed by -seed, so "
                             "repeated passes at different rollout seeds measure the protocol's own noise")
    parser.add_argument("-suites", default="", help="comma-separated subset")
    parser.add_argument("-usable_at", default=0.3, type=float,
                        help="shape quality a mesh must reach to count as usable; "
                             "0.2 is Gmsh's own coarse p10-median band, so the "
                             "claim is 'no worse than a production mesher at the "
                             "same element count'")
    parser.add_argument("-steps", default=0, type=int)
    parser.add_argument("-out", default=None)
    parser.add_argument("-max_steps_factor", default=None, type=float,
                        help="override the config's step budget at evaluation (moves per boundary "
                             "half-edge); the paper default is 3.0, and 6.0 completes the larger domains")
    parser.add_argument("-repair_bar", default=0.3, type=float,
                        help="the quality below which -repair_rounds fires (independent of -usable_at, "
                             "which the paper queue sets to 0.2 for its logs)")
    parser.add_argument("-repair_budget", default=None, type=int,
                        help="cap on the step budget of each repair continuation (default: the env's own "
                             "budget for the repaired mesh, which on large curved meshes at 6x runs to thousands "
                             "of moves per continuation; one forced move needs only a few closing moves)")
    parser.add_argument("-repair_rounds", default=0, type=int,
                        help="split-and-continue: when the selected mesh is below -usable_at, force one "
                             "move at its worst corner (insert on the long side of a thin corner, chord "
                             "across a bad angle) and let the policy finish, up to this many rounds; "
                             "see utilities/repair_eval.py")
    parser.add_argument("-densify", default=0.0, type=float,
                        help="degrees of sweep per boundary-arc segment; 0 leaves "
                             "arcs as drawn. Applied after the draw, so the "
                             "domains are the same ones as at 0")
    parser.add_argument("-draws", default="",
                        help="comma-separated draw indices to run; every other draw is still drawn (so the "
                             "domains and their indices are unchanged) but not rolled out. For figures.")
    parser.add_argument("-save_meshes", default=None,
                        help="directory: pickle each selected mesh, with the per-vertex degree and wants and "
                             "the record the tables use, as <suite>.draw<NNN>.pkl (read by paper_figure.py)")
    args = parser.parse_args()
    only_draws = {int(x) for x in args.draws.split(",") if x.strip()}

    from envs.environment_maker import initialize_environment
    from src.geo2d_bridge import make_initializer, resmooth_env
    from src.utils import load_yaml_config

    config = load_yaml_config(args.config)
    env_config = dict(config["environment"])
    template = env_config.get("template_size", 200)
    # The REWARD target and the USABILITY bar are different numbers and were
    # conflated. Gmsh at ~10 elements has a per-mesh minimum whose median is
    # 0.21-0.39 by suite, so 0.4 demanded better-than-Gmsh at matched coarseness;
    # it only reaches 0.4-0.65 when allowed 3-10x more elements. 0.4 stays as the
    # reward's target, where an aspirational number gives gradient. Usability is
    # judged here, against what a production mesher actually delivers.
    threshold = args.usable_at
    env_config["quality_metric"] = "shape"     # score what matters downstream
    if args.max_steps_factor is not None:
        env_config["max_steps_factor"] = args.max_steps_factor
    from src.geo2d_bridge import load_model
    model = load_model(args.checkpoint, args.config, template_size=template)
    import random, torch
    random.seed(args.rollout_seed); np.random.seed(args.rollout_seed); torch.manual_seed(args.rollout_seed)

    chosen = [x.strip() for x in args.suites.split(",") if x.strip()] or list(SUITES)
    report, line = {}, []
    for suite in chosen:
        options = dict(SUITES[suite])
        capped = "min_corners" in options          # a transfer suite
        options.setdefault("max_corners", 24)
        options.setdefault("min_corners", MIN_CORNERS)
        source = make_initializer(seed=args.seed, densify_sweep=args.densify or None, **options)
        usable, excesses, built = 0, [], 0
        all_quad, at_par, qualities, quad_qualities = 0, 0, [], []
        domains = []
        # `draw` indexes the generator's draw sequence, so a Gmsh line can pair
        # by it; a transfer suite rejects most draws on size and extremity, so
        # it draws until `n` are KEPT (capped at 40 n tries)
        draw = -1
        while built < args.n and draw < 40 * args.n:
            draw += 1
            try:
                graph, desired = source()
            except Exception:
                continue
            if capped and boundary_edge_ratio(graph) > MAX_EDGE_RATIO:
                continue
            built += 1
            if only_draws and draw not in only_draws:
                continue

            class Fixed:
                def __init__(self):
                    self.n = len(desired)

                def __call__(self):
                    return deepcopy(graph), dict(desired)

            cfg = dict(env_config)
            cfg.pop("initializer", None)
            cfg["graph_initializer"] = Fixed()
            cfg["resample_if_at_par"] = False
            env = initialize_environment(cfg)
            # Pick the mesh an engineer would keep: all-quad first, then the
            # best-shaped, and only then closeness to par -- searched over EVERY
            # state the rollout passes through, not just the one it ends on.
            #
            # Reading the final state only is right when the episode stops at
            # its own goal, and wrong the moment it does not. With
            # `terminate_when_usable` and `terminate_at_par` off, the episode
            # runs to max_steps and the final mesh is whatever the monotone
            # vocabulary did after the good one: measured on the warm start,
            # scoring the last state gives 31 of 48 all-quad where scoring the
            # best gives 47. The reward already maximises over the trajectory
            # (`AngleEnv.candidate_score`); this is the evaluator agreeing with
            # it about which mesh is the answer.
            # Collect every ALL-QUAD state the rollouts pass through, then rank
            # them AFTER untangling. Ranking before it meant choosing on one
            # number and reporting another: measured on the trained policy, the
            # untangler changes which candidate wins on 4 of 10 domains, since
            # it became tangent-aware and can now repair curved elements it
            # previously left alone. Untangling is too slow for the inner loop,
            # so it runs once per candidate at the end rather than per step.
            candidates, fallback = [], None
            for trial in range(args.n_samples + 1):
                observation, _ = env.reset()
                done = False
                while True:
                    face = int(env.global_face_score)
                    excess = abs(int(env.global_vertex_score) - env.par)
                    if face == 0:
                        candidates.append((excess, deepcopy(env.graph)))
                    key = (face, -float(env.min_element_quality()), excess)
                    if fallback is None or key < fallback[0]:
                        fallback = (key, deepcopy(env.graph))
                    if done:
                        break
                    action, _ = model.predict(observation, deterministic=(trial == 0))
                    observation, _, done, truncated, _ = env.step(action)
                    done = done or truncated

            best = None
            # "certified optimum reached": did ANY all-quad candidate across the
            # attempts sit exactly at par and clear the degeneracy bar after the
            # untangler? Recorded beside the quality-first selection so the tables
            # can report at-par-on-any-attempt (the par-first protocol) and usable
            # (the quality-first one) from the same run.
            par_reached, par_reached_topological = False, False
            for excess, candidate in candidates:
                env.graph = candidate
                env._update_half_edge_angles()
                try:
                    resmooth_env(env)
                except Exception:
                    pass
                key = (0, -float(env.min_element_quality()), excess)
                if excess == 0:
                    par_reached_topological = True
                    if float(env.min_element_quality()) >= 0.1:
                        par_reached = True
                if best is None or key < best[0]:
                    best = (key, deepcopy(env.graph))
            if best is None:
                best = fallback
            env.graph = best[1]
            env._update_half_edge_angles()
            if not candidates:
                # the fallback was never an all-quad candidate, so it has not
                # been untangled yet; every real candidate already has been
                try:
                    resmooth_env(env)
                except Exception:
                    pass
            repaired = 0
            if args.repair_rounds and candidates and -best[0][1] < args.repair_bar:
                # Split-and-continue (Arjun, 2026-09-23): the below-bar meshes are slivers at
                # boundary corners; one forced move there plus the policy's own closing moves
                # lifts most of them over the bar (utilities/repair_eval.py: 42 -> 53 of 60).
                from utilities.repair_eval import forced_move, wants_for
                base_wants = dict(desired)
                cur_q, cur_e, cur_g = -best[0][1], best[0][2], best[1]
                for _round in range(args.repair_rounds):
                    if cur_q >= args.repair_bar:
                        break
                    trial_graph = deepcopy(cur_g)
                    try:
                        forced_move(env, trial_graph, 2.0)
                    except Exception:
                        break
                    repaired += 1
                    found = []
                    for trial in range(args.n_samples + 1):
                        start = deepcopy(trial_graph)
                        observation, _ = env._reset_to_state(start, wants_for(env, start, base_wants))
                        if args.repair_budget:
                            env.max_steps = min(env.max_steps, args.repair_budget)
                        done = False
                        while True:
                            if int(env.global_face_score) == 0:
                                found.append((abs(int(env.global_vertex_score) - env.par), deepcopy(env.graph)))
                            if done:
                                break
                            action, _ = model.predict(observation, deterministic=(trial == 0))
                            observation, _, done, truncated, _ = env.step(action)
                            done = done or truncated
                    for e1, g1 in found:
                        env.graph = g1
                        env._update_half_edge_angles()
                        env.vertex_desired_degree = wants_for(env, g1, base_wants)
                        env.vertex_desired_options = env._compute_desired_options()
                        try:
                            resmooth_env(env)
                        except Exception:
                            pass
                        q1 = float(env.min_element_quality())
                        if (q1, -e1) > (cur_q, -cur_e):
                            cur_q, cur_e, cur_g = q1, e1, deepcopy(env.graph)
                best = ((0, -cur_q, cur_e), cur_g)
                env.graph = cur_g
                env._update_half_edge_angles()
                env.vertex_desired_degree = wants_for(env, cur_g, base_wants)
                env.vertex_desired_options = env._compute_desired_options()
            face, _, excess = best[0]
            quality = float(env.min_element_quality())
            qualities.append(quality)
            # per-domain record, so a Gmsh baseline can be matched to THIS
            # mesh's element count and the comparison paired domain by domain
            domains.append({
                "draw": draw, "face": int(face), "excess": int(excess),
                "par": int(env.par), "quality": quality,
                "elements": int(env.graph.number_of_faces()),
                "vertices": int(env.graph.number_of_vertices()),
                # NOT `env.global_vertex_score`: that is cached by `step` and STALE
                # after `env.graph` is assigned (the previous trial's number, wrong
                # on 46 of 48 records of one run). Recomputing on the candidate
                # raises KeyError for vertices another trial inserted, so use the
                # identity score = par + excess, with excess from the candidate key.
                "irregular": int(env.par + excess),
                "corners": int(getattr(source, "last_geometry", None).outer.n)
                if getattr(source, "last_geometry", None) is not None else -1,
                "par_reached": bool(par_reached),
                "par_reached_topological": bool(par_reached_topological),
                "repair_rounds": int(repaired),
            })
            if args.save_meshes:
                _save_selected_mesh(args.save_meshes, suite, env, domains[-1], dict(desired))
            if face == 0:
                all_quad += 1
                quad_qualities.append(quality)
                excesses.append(excess)
                at_par += int(excess == 0)
                if quality >= threshold:
                    usable += 1
        report[suite] = {
            "total": built, "all_quad": all_quad, "usable": usable, "at_par": at_par,
            "par_reached": sum(d["par_reached"] for d in domains),
            "par_reached_topological": sum(d["par_reached_topological"] for d in domains),
            "quality_median": _median(quad_qualities),
            "quality_p10": _percentile(quad_qualities, 10),
            "excess_median": _median(excesses),
            "excess_mean": (sum(excesses) / len(excesses)) if excesses else float("nan"),
            "domains": domains,
        }
        r = report[suite]
        line.append(f"{suite:14} quad {all_quad:2d}/{built:<2d} "
                    f"q(med) {r['quality_median']:+.2f} q(p10) {r['quality_p10']:+.2f} "
                    f"exc(med) {r['excess_median']:.1f} par {at_par:2d} "
                    f"usable {usable:2d} par-any {r['par_reached']:2d}")
    print(f"=== {args.steps/1e6:6.2f}M ===", flush=True)
    for entry in line:
        print("   " + entry, flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"steps": args.steps, "suites": report}, handle, indent=2)


if __name__ == "__main__":
    main()
