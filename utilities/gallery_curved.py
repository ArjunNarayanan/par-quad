"""What curved domains look like, and how the agent does on them.

Two sources, which are NOT the same thing and are drawn side by side for that
reason:

  bent      a certified instance whose straight edges were bent into arcs by
            `src/bend_outline.py`. Every corner want is preserved by
            construction, so the recorded solution is still exactly optimal --
            this is TRAINING data, and it comes with an answer.
  geo2d     geo2d's own `rounded`/`default`/`complex` presets, which build
            fillets natively. These are TEST domains and come with no answer.

Meshes are untangled before drawing, and any face the agent never split is
HATCHED rather than filled. Without that an unfinished domain reads as a
rendering bug: a big leftover polygon overlaps its neighbours and looks like an
inverted element, when the truth is simply that the mesh is not finished. The
untangler cannot help there either -- `resmooth_env` reverts on a mesh holding
odd faces, because `optimize` is not monotone when the topology is unfinished.

Boundaries are drawn by sampling the arcs, which is a rendering choice only --
nothing in the mesh or the scoring ever discretises them.

    venv/bin/python utilities/gallery_curved.py -out out/curved
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.append(os.getcwd())

QUALITY_CMAP = plt.cm.YlGnBu
SAMPLES = 24          # points per arc, for DRAWING only


def boundary_path(graph):
    """The outline as a dense polyline, following arcs where they exist."""
    pieces = []
    for half_edge in graph.half_edge_list():
        if not graph.half_edge_on_boundary(half_edge):
            continue
        a = graph.source_vertex(half_edge, tag=False)
        b = graph.target_vertex(half_edge, tag=False)
        arc = graph.boundary_arcs.get(a, b) if graph.boundary_arcs else None
        if arc is None:
            pieces.append(np.array([graph.vertex_coordinate(a),
                                    graph.vertex_coordinate(b)], dtype=float))
        else:
            start = np.asarray(graph.vertex_coordinate(a), dtype=float)
            forward = np.linalg.norm(arc.point(0.0) - start) < \
                np.linalg.norm(arc.point(1.0) - start)
            ts = np.linspace(0.0, 1.0, SAMPLES) if forward else np.linspace(1.0, 0.0, SAMPLES)
            pieces.append(np.array([arc.point(t) for t in ts]))
    return pieces


def draw(axis, graph, title, colour_by_quality=True):
    axis.clear()
    axis.axis("off")
    angles = graph.half_edge_angles()
    for face in graph.face_list():
        loop = graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))
        # follow the arc on any side of this face that is a curved boundary, or
        # the fill stops short of the outline and every panel looks broken
        pieces = []
        for half_edge in loop:
            a = graph.source_vertex(half_edge, tag=False)
            b = graph.target_vertex(half_edge, tag=False)
            arc = graph.boundary_arcs.get(a, b) if graph.boundary_arcs else None
            if arc is None or not graph.half_edge_on_boundary(half_edge):
                pieces.append(np.asarray(graph.vertex_coordinate(a), dtype=float)[None, :])
                continue
            start = np.asarray(graph.vertex_coordinate(a), dtype=float)
            forward = np.linalg.norm(arc.point(0.0) - start) < \
                np.linalg.norm(arc.point(1.0) - start)
            ts = np.linspace(0.0, 1.0, SAMPLES) if forward else np.linspace(1.0, 0.0, SAMPLES)
            pieces.append(np.array([arc.point(t) for t in ts[:-1]]))
        points = np.vstack(pieces)
        degree = graph.face_degree(face)
        if degree != 4:
            # An UNFINISHED face -- the agent never split it. Drawing it like a
            # quad is what made these panels look like a rendering bug: a big
            # leftover polygon overlaps its neighbours and reads as an
            # inversion, when really the mesh simply is not done. Hatch it.
            axis.fill(points[:, 0], points[:, 1], facecolor="#f2d0cb",
                      edgecolor="#c1440e", linewidth=0.8, hatch="//////",
                      zorder=1, alpha=0.9)
            continue
        if colour_by_quality and angles:
            worst = min(np.sin(np.radians(angles[h])) for h in loop if h in angles)
            shade = float(np.clip((worst - 0.2) / 0.8, 0.0, 1.0))
            colour = QUALITY_CMAP(0.15 + 0.7 * shade)
        else:
            colour = "#e9eef4"
        axis.fill(points[:, 0], points[:, 1], facecolor=colour,
                  edgecolor="#8a97a5", linewidth=0.5, zorder=1)
    for piece in boundary_path(graph):
        axis.plot(piece[:, 0], piece[:, 1], color="#17242e", linewidth=1.6, zorder=3)
    for vertex in graph.vertex_list(tag=False):
        if graph.is_boundary_vertex(vertex):
            point = graph.vertex_coordinate(vertex)
            axis.plot(point[0], point[1], "o", markersize=2.6, color="#c1440e", zorder=4)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=7, pad=3)


def sheet(panels, path, title, per_row=6):
    rows = int(np.ceil(len(panels) / per_row))
    figure, axes = plt.subplots(rows, per_row, figsize=(2.2 * per_row, 2.5 * rows),
                                squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for index, (graph, caption) in enumerate(panels):
        draw(axes[index // per_row][index % per_row], graph, caption)
    figure.suptitle(title, fontsize=11, y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(path, dpi=125)
    plt.close(figure)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-config", default="experiments/self-play/quad/pinwheel-v1/config.yml")
    parser.add_argument("-checkpoint",
                        default="experiments/self-play/quad/pinwheel-v1/warm_ppo_model.zip")
    parser.add_argument("-n", default=12, type=int)
    parser.add_argument("-n_samples", default=5, type=int)
    parser.add_argument("-seed", default=17, type=int)
    parser.add_argument("-out", default="out/curved")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import geo2d
    from copy import deepcopy
    from envs.solved_instances import (default_scratch_env, generate_instances,
                                       replay)
    from src.geo2d_bridge import load_model, make_curved_env, resmooth_env
    from src.utils import load_yaml_config
    from utilities.evaluate_geo2d import rollout

    config = load_yaml_config(args.config)
    env_config = dict(config["environment"])
    model = load_model(args.checkpoint, args.config,
                       template_size=env_config.get("template_size", 128))

    # --- 1. bent training domains, raw, and with the CERTIFIED solution ------
    instances, _ = generate_instances(
        160, cell_range=(2, 12), hole_probability=0.0, curve_probability=1.0,
        curve_edge_probability=0.7, rng=np.random.default_rng(args.seed), verbose=False)
    curved = [i for i in instances if i.curved][:args.n]
    scratch = default_scratch_env(3, face_desired_degree=4)

    raw, solved = [], []
    for index, instance in enumerate(curved):
        graph, desired = instance.build()
        raw.append((deepcopy(graph), f"#{index}  {len(instance.arcs)} arcs  "
                                     f"par {instance.par}"))
        *_, played, ok = replay(scratch, instance)
        if ok:
            solved.append((deepcopy(scratch.graph),
                           f"#{index}  certified: {len(scratch.graph.face_list())}q  "
                           f"par {scratch.par}"))
    sheet(raw, os.path.join(args.out, "bent_domains.png"),
          "bent training domains (arcs preserve every corner want)")
    sheet(solved, os.path.join(args.out, "bent_certified.png"),
          "the generator's own certified solution, on the bent domain")

    # --- 2. geo2d's native curved domains, and the AGENT on them -------------
    panels, at_par = [], 0
    drawn = 0
    seed = 0
    while drawn < args.n and seed < 3000:
        try:
            geometry = geo2d.generate(seed, preset="rounded")
        except Exception:
            seed += 1
            continue
        seed += 1
        if len(geometry.loops) != 1 or not any(geometry.outer.is_arc()):
            continue
        if not (8 <= geometry.outer.n <= 20):
            continue
        env = make_curved_env(env_config, geometry, template_size=160)
        best, _ = rollout(model, env, deterministic=True)
        for _ in range(args.n_samples):
            sampled, _ = rollout(model, env, deterministic=False)
            if (sampled["face_score"], abs(sampled["vertex_excess"])) < \
               (best["face_score"], abs(best["vertex_excess"])):
                best = sampled
        at_par += int(best["solved"])
        # Untangle before drawing, so a mesh is judged on the geometry a
        # downstream tool would get. It is a no-op on a mesh that still holds
        # odd faces -- resmooth_env reverts there, because `optimize` is not
        # monotone when the topology is unfinished.
        try:
            resmooth_env(env, iters=15, method="optimize", slide="optimize")
        except Exception:
            pass
        mark = "AT PAR" if best["solved"] else \
            (f"all quads, {abs(best['vertex_excess'])} off par"
             if best["face_score"] == 0
             else f"UNFINISHED: {best['face_score']} face score")
        panels.append((deepcopy(env.graph), f"#{drawn}  par {env.par}  {mark}"))
        drawn += 1
    sheet(panels, os.path.join(args.out, "geo2d_agent.png"),
          f"geo2d `rounded` domains -- the agent reaches par on {at_par}/{len(panels)}")
    print(f"\nagent at par on {at_par}/{len(panels)} geo2d curved domains")

    # --- 3. the two sources side by side ------------------------------------
    # They are different animals and the paper should not blur them: one comes
    # with a certified answer and is TRAINING data, the other does not and is a
    # TEST set. Drawn as raw domains, no meshes, so the shapes are the subject.
    side = []
    for index, instance in enumerate(curved[:6]):
        graph, _ = instance.build()
        side.append((graph, f"BENT #{index}  {len(instance.arcs)} arcs\n"
                            f"certified solution: YES"))
    drawn, seed = 0, 0
    while drawn < 6 and seed < 3000:
        try:
            geometry = geo2d.generate(seed, preset="rounded")
        except Exception:
            seed += 1
            continue
        seed += 1
        if len(geometry.loops) != 1 or not any(geometry.outer.is_arc()):
            continue
        if not (8 <= geometry.outer.n <= 20):
            continue
        env = make_curved_env(env_config, geometry, template_size=160)
        env.reset()
        side.append((env.graph, f"GEO2D #{drawn}  "
                                f"{int(sum(geometry.outer.is_arc()))} arcs\n"
                                f"certified solution: no"))
        drawn += 1
    sheet(side, os.path.join(args.out, "sources_side_by_side.png"),
          "top row: bent certified instances (training)   "
          "bottom row: geo2d's own curved generator (test)")


if __name__ == "__main__":
    main()
