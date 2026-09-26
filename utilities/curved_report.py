"""Agent against Gmsh on curved domains: galleries and quality tables.

    venv/bin/python utilities/curved_report.py -checkpoint <zip> -config <yml> \
        -n 8 -out out/report

Produces, for the same curved domains:

  * a side-by-side gallery, agent and Gmsh, each element shaded by its own SHAPE
    quality on one shared scale, so a bad element is visible rather than inferred
  * a table of per-mesh minimum and median quality, element count and all-quad
    rate, for the agent and for Gmsh at BOTH coarsenesses -- matched element
    count, which is the like-for-like comparison, and Gmsh's natural size, which
    is what a solver would actually be handed

Quality is geo2d's shape metric throughout, for both meshers. The angle metric
this repository used until today cannot see aspect ratio and reads 0.707 on a
mesh geo2d scores 0.056, so it flatters exactly the failure that matters.
"""

import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.colors import Normalize                 # noqa: E402

QUALITY_CMAP = plt.get_cmap("RdYlGn")
SHADE = Normalize(vmin=-0.2, vmax=0.8)


def shade(value):
    return QUALITY_CMAP(SHADE(value))


def draw_tiler(axis, graph, title):
    """Our own mesh, each quad shaded by its worst corner."""
    from utilities.gallery_curved import boundary_path
    corner = graph.corner_shape_qualities()
    for face in graph.face_list():
        loop = graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))
        points = np.array([graph.vertex_coordinate(
            graph.source_vertex(h, tag=False)) for h in loop], dtype=float)
        worst = min(corner[h] for h in loop if h in corner)
        axis.fill(points[:, 0], points[:, 1], facecolor=shade(worst),
                  edgecolor="#37474f", linewidth=0.6, zorder=1)
    for piece in boundary_path(graph):
        axis.plot(piece[:, 0], piece[:, 1], color="#11181c", linewidth=1.7, zorder=3)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=8)


def normalizer(geometry, drop_collinear=True):
    """The centre and scale `curved_geometry_to_tiler` applies, so everything can
    be compared in ONE frame. The agent's mesh is normalized to about [-1, 1];
    Gmsh and the raw outline are not. Shape quality is a ratio and does not care,
    but any DISTANCE does, and comparing them unnormalized silently measured the
    gap between two coordinate systems."""
    from src.geo2d_bridge import _loop_arrays
    points, _, _ = _loop_arrays(geometry.outer, ccw=True, drop_collinear=drop_collinear)
    centre = points.mean(axis=0)
    scale = max(float(np.abs(points - centre).max()), 1e-9)
    return centre, scale


def outline_of(geometry, per_edge=60):
    """The true domain boundary, densely sampled along EVERY edge.

    Straight edges have to be sampled too. `_sampled_loop` emits only the start
    point of a straight edge, which is right for drawing a polyline and wrong
    for a nearest-point distance: a mesh vertex halfway along a long straight
    edge is far from both of its endpoints, and the error comes out as the
    length of the edge rather than zero.
    """
    from src.geo2d_bridge import _loop_arrays
    loops = []
    for index, loop in enumerate(geometry.loops):
        points, _, arcs = _loop_arrays(loop, ccw=(index == 0), drop_collinear=False)
        count = len(points)
        pieces = []
        for i in range(count):
            j = (i + 1) % count
            entry = arcs.get((i, j)) or arcs.get((j, i))
            ts = np.linspace(0.0, 1.0, per_edge, endpoint=False)
            if entry is None:
                a, b = points[i], points[j]
                pieces.append(a + np.outer(ts, b - a))
            else:
                arc, start = entry
                sampled = np.array([arc.point(t) for t in np.linspace(0.0, 1.0, per_edge)])
                if start != i:
                    sampled = sampled[::-1]
                pieces.append(sampled[:-1])
        loops.append(np.vstack(pieces))
    return loops


def chordal_error(boundary_points, outline, scale=1.0):
    """Worst distance from a mesh boundary VERTEX to the true outline.

    Kept for reference, but it is the weaker of the two: a vertex placed exactly
    on the curve scores zero however badly the EDGE between two such vertices
    cuts across. See `chordal_edge_error`.
    """
    if len(boundary_points) == 0:
        return float("nan")
    every = np.vstack(outline)
    worst = 0.0
    for point in boundary_points:
        worst = max(worst, float(np.min(np.linalg.norm(every - point, axis=1))))
    return worst / scale


def chordal_edge_error(edges, outline, samples=9):
    """Worst distance from a boundary EDGE to the true outline.

    This is the honest measure of how well a mesh approximates a curved
    boundary, and the vertex version misses it entirely. The agent keeps the
    minimum four control points on a circular hole, so the hole is a 4-gon
    inscribed in the circle: every vertex is exactly on the curve and scores
    zero, while the edges cut across by r(1 - cos 45) or about 0.29r. Gmsh puts
    more nodes on the same hole and approximates it far better. The galleries
    show this plainly -- the agent's holes render as diamonds and Gmsh's as
    circles -- and the vertex metric called the two equal.
    """
    if not len(edges):
        return float("nan")
    every = np.vstack(outline)
    worst = 0.0
    for start, end in edges:
        for t in np.linspace(0.0, 1.0, samples)[1:-1]:
            point = start + t * (end - start)
            worst = max(worst, float(np.min(np.linalg.norm(every - point, axis=1))))
    return worst


def boundary_edges_of_tiler(graph):
    """Boundary edges of a Tiler, as coordinate pairs."""
    out = []
    for h in graph.half_edge_list():
        if not graph.half_edge_on_boundary(h):
            continue
        a = np.asarray(graph.vertex_coordinate(graph.source_vertex(h, tag=False)), float)
        b = np.asarray(graph.vertex_coordinate(graph.target_vertex(h, tag=False)), float)
        out.append((a, b))
    return out


def strict_element_qualities(nodes, elements):
    """Per-element shape quality that SEES a flipped element.

    geo2d's `Mesh.quality()` is orientation-blind -- a perfect unit square
    traversed clockwise scores +1.0, exactly like the counter-clockwise one --
    so an element that has turned inside out relative to its neighbours reads as
    excellent. The agent's own meshes are measured corner-wise through the DCEL,
    which is orientation-AWARE, so scoring Gmsh with geo2d's metric and
    ourselves with the strict one would hold the two to different standards.

    The mesh's own convention supplies the sign: whichever orientation the bulk
    of the elements have is taken as correct, and an element opposing it is
    reported negative.
    """
    nodes = np.asarray(nodes, dtype=float)
    signed = []
    for quad in elements:
        loop = nodes[list(quad)]
        area = 0.0
        for k in range(len(loop)):
            a, b = loop[k], loop[(k + 1) % len(loop)]
            area += a[0] * b[1] - b[0] * a[1]
        signed.append(0.5 * area)
    sign = 1.0 if np.median(signed) >= 0 else -1.0

    out = []
    for quad in elements:
        loop = nodes[list(quad)]
        worst = None
        for k in range(len(loop)):
            point, ahead, behind = loop[k], loop[(k + 1) % len(loop)], loop[k - 1]
            first, second = ahead - point, behind - point
            jacobian = first[0] * second[1] - first[1] * second[0]
            value = sign * 2.0 * jacobian / max(first @ first + second @ second, 1e-30)
            worst = value if worst is None else min(worst, value)
        out.append(worst)
    return np.asarray(out, dtype=float)


def boundary_edges_of_quads(nodes, elements):
    """Boundary edges of a quad mesh: the sides belonging to exactly one element."""
    seen = {}
    for quad in elements:
        n = len(quad)
        for k in range(n):
            a, b = int(quad[k]), int(quad[(k + 1) % n])
            seen[(min(a, b), max(a, b))] = seen.get((min(a, b), max(a, b)), 0) + 1
    nodes = np.asarray(nodes, float)
    return [(nodes[a], nodes[b]) for (a, b), count in seen.items() if count == 1]


def draw_gmsh_mesh(axis, nodes, elements, title, outline=None):
    """Gmsh's mesh with every edge drawn and no quality shading.

    The companion to `gallery_curved.draw(..., colour_by_quality=False)`: the
    shaded galleries answer "how good is each element", and these answer "what
    does the mesh look like", which a colour ramp actively gets in the way of.
    Same neutral fill and edge colour on both sides so the two are comparable.
    """
    nodes = np.asarray(nodes, dtype=float)
    for element in elements:
        points = nodes[list(element)]
        axis.fill(points[:, 0], points[:, 1], facecolor="#e9eef4",
                  edgecolor="#8a97a5", linewidth=0.5, zorder=1)
    if outline is not None:
        for piece in outline:
            closed = np.vstack([piece, piece[:1]])
            axis.plot(closed[:, 0], closed[:, 1], color="#17242e",
                      linewidth=1.6, zorder=3)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=8)


def draw_gmsh(axis, nodes, elements, title, outline=None):
    """Gmsh's mesh, on the same colour scale."""
    from src.geo2d_bridge import import_geo2d
    geo2d = import_geo2d()
    quality = np.asarray(geo2d.Mesh(nodes, elements).quality(), dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    for element, value in zip(elements, quality):
        points = nodes[list(element)]
        axis.fill(points[:, 0], points[:, 1], facecolor=shade(value),
                  edgecolor="#37474f", linewidth=0.5, zorder=1)
    if outline is not None:
        for piece in outline:
            closed = np.vstack([piece, piece[:1]])
            axis.plot(closed[:, 0], closed[:, 1], color="#11181c", linewidth=1.7, zorder=3)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=8)


def agent_mesh(env_config, geometry, model, samples):
    """Best-of-N by the criterion an engineer would use: all-quad, then shape."""
    from envs.environment_maker import initialize_environment
    from src.geo2d_bridge import curved_geometry_to_tiler, resmooth_env

    graph, desired = curved_geometry_to_tiler(geometry)

    class Fixed:
        def __init__(self):
            self.n = len(desired)

        def __call__(self):
            return deepcopy(graph), dict(desired)

    config = dict(env_config)
    config.pop("initializer", None)
    config["graph_initializer"] = Fixed()
    config["resample_if_at_par"] = False
    config["quality_metric"] = "shape"
    env = initialize_environment(config)

    # Every state the rollout passes through is a candidate, not just the one
    # it ends on. With `terminate_when_usable` and `terminate_at_par` off the
    # episode runs to max_steps, so the final mesh is whatever the monotone
    # vocabulary did AFTER the good one -- measured on the warm start, keeping
    # the last state scores 31 of 48 all-quad where keeping the best scores 47.
    # The reward maximises over the trajectory too (`candidate_score`).
    # All-quad states are collected and ranked AFTER untangling, because the
    # untangler became tangent-aware and now changes which candidate wins on
    # 4 of 10 domains. Ranking before it chose on one number and reported
    # another.
    candidates, fallback = [], None
    for trial in range(samples + 1):
        observation, _ = env.reset()
        done = False
        while True:
            face = int(env.global_face_score)
            if face == 0:
                candidates.append(deepcopy(env.graph))
            key = (face, -float(env.min_element_quality()))
            if fallback is None or key < fallback[0]:
                fallback = (key, deepcopy(env.graph))
            if done:
                break
            action, _ = model.predict(observation, deterministic=(trial == 0))
            observation, _, done, truncated, _ = env.step(action)
            done = done or truncated

    best, keep = None, None
    for candidate in candidates:
        env.graph = candidate
        env._update_half_edge_angles()
        try:
            resmooth_env(env)
        except Exception as error:
            print(f"    resmooth skipped: {error}", flush=True)
        key = (0, -float(env.min_element_quality()))
        if best is None or key < best:
            best, keep = key, deepcopy(env.graph)
    if best is None:
        best, keep = fallback[0], fallback[1]
    env.graph = keep
    env._update_half_edge_angles()
    if not candidates:
        # only the fallback needs it; every candidate is already untangled
        try:
            resmooth_env(env)
        except Exception as error:
            print(f"    resmooth skipped: {error}", flush=True)
    return env, best[0]


def stats(values):
    if len(values) == 0:
        return float("nan"), float("nan")
    ordered = np.sort(np.asarray(values, dtype=float))
    return float(ordered.min()), float(np.median(ordered))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-config", required=True)
    parser.add_argument("-n", default=8, type=int)
    parser.add_argument("-n_samples", default=6, type=int)
    parser.add_argument("-holes", action="store_true")
    parser.add_argument("-out", default="out/curved_report")
    parser.add_argument("-label", default="agent")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from src.geo2d_bridge import (import_geo2d, load_model, tiler_faces)
    from src.utils import load_yaml_config
    from utilities.compare_gmsh import VARIANTS, gmsh_quad_mesh, mesh_at_element_count
    from utilities.compare_gmsh_curved import curved_corner_wants
    from utilities.compare_gmsh import score_mesh
    geo2d = import_geo2d()

    env_config = dict(load_yaml_config(args.config)["environment"])
    model = load_model(args.checkpoint, args.config,
                       template_size=env_config.get("template_size", 200))

    panels, rows, seed, drawn = [], [], 0, 0
    while drawn < args.n and seed < 4000:
        this, seed = seed, seed + 1
        try:
            geometry = geo2d.generate(this, preset="rounded")
        except Exception:
            continue
        holed = len(geometry.loops) > 1
        if holed != bool(args.holes):
            continue
        if not any(any(loop.is_arc()) for loop in geometry.loops):
            continue
        if not (8 <= geometry.outer.n <= 14):
            continue

        env, face = agent_mesh(env_config, geometry, model, args.n_samples)
        ours = np.asarray(list(env.graph.corner_shape_qualities().values()), float)
        # The SAME mesh read as a linear Q4 mesh, for a like-for-like line
        # against Gmsh. Gmsh emits straight-sided quads with no arcs attached,
        # so its number is a chord reading whichever metric we name; the agent's
        # is a tangent reading. Which is right depends on what the mesh IS -- a
        # curved block decomposition or a linear mesh -- and the two readings
        # bracket the answer rather than one of them being the answer.
        stripped = deepcopy(env.graph)
        if stripped.boundary_arcs:
            stripped.boundary_arcs._arcs.clear()
        chord_reading = min(stripped.corner_shape_qualities().values())
        # per-ELEMENT minimum, to compare like with like against geo2d
        per_face = []
        for f in env.graph.face_list():
            loop = env.graph.generate_half_edge_face_loop(
                env.graph.first_face_halfedge(f))
            corner = env.graph.corner_shape_qualities()
            per_face.append(min(corner[h] for h in loop if h in corner))
        a_min, a_med = stats(per_face)
        a_count = env.graph.number_of_faces()

        # IRREGULARITY per vertex: what a block decomposition is actually for.
        # A structured mesh puts its vertices on the degree the domain wants;
        # an unstructured one scatters irregular vertices through the interior.
        # No shape metric sees this, and the galleries show it plainly.
        a_vertices = env.graph.number_of_vertices()
        a_irregular = float(int(env.global_vertex_score)) / max(a_vertices, 1)

        found = mesh_at_element_count(geometry, a_count, VARIANTS["blossom"])
        if found is None:
            continue
        _, _, g_nodes, g_elements = found
        g_q = np.asarray(geo2d.Mesh(g_nodes, g_elements).quality(), float)
        g_min, g_med = stats(g_q)
        try:
            wants, tol = curved_corner_wants(geometry)
            scored = score_mesh(g_nodes, g_elements, wants, tol, env.par)
            g_irregular = float(scored["vertex_score"]) / max(len(g_nodes), 1)
        except Exception as error:
            print(f"    gmsh irregularity skipped: {error}", flush=True)
            g_irregular = float("nan")

        # one frame for everything: normalize the outline and Gmsh the way the
        # bridge normalizes the agent's mesh
        centre, scale = normalizer(geometry)
        outline_raw = outline_of(geometry)
        outline = [(piece - centre) / scale for piece in outline_raw]
        ours_boundary = np.array([env.graph.vertex_coordinate(v)
                                  for v in env.graph.vertex_list(tag=False)
                                  if env.graph.is_boundary_vertex(v)], dtype=float)
        a_chord = chordal_error(ours_boundary, outline)
        a_edge_chord = chordal_edge_error(boundary_edges_of_tiler(env.graph), outline)
        g_nodes_array = (np.asarray(g_nodes, dtype=float) - centre) / scale
        # BOUNDARY nodes only. Every interior node is legitimately far from the
        # outline, and including them measured mesh depth rather than fidelity --
        # which looked fine on the agent's coarse meshes, with almost no interior
        # nodes, and wrongly indicted Gmsh's finer ones.
        boundary_index = np.asarray(
            geo2d.Mesh(g_nodes, g_elements).boundary_nodes(), dtype=int)
        g_chord = chordal_error(g_nodes_array[boundary_index], outline)
        g_edge_chord = chordal_edge_error(
            boundary_edges_of_quads(g_nodes_array, g_elements), outline)

        try:
            n_nodes, n_elements = gmsh_quad_mesh(geometry, 0.5, VARIANTS["blossom"])
            n_q = np.asarray(geo2d.Mesh(n_nodes, n_elements).quality(), float)
            n_min, n_med = stats(n_q)
            n_count = len(n_elements)
        except Exception:
            n_min = n_med = float("nan"); n_count = 0

        rows.append(dict(seed=this, all_quad=(face == 0), agent_elements=a_count,
                         agent_min=a_min, agent_median=a_med,
                         agent_min_as_linear=float(chord_reading),
                         gmsh_matched_elements=len(g_elements),
                         gmsh_matched_min=g_min, gmsh_matched_median=g_med,
                         gmsh_natural_elements=n_count,
                         gmsh_natural_min=n_min, gmsh_natural_median=n_med,
                         agent_chordal=a_chord, gmsh_matched_chordal=g_chord,
                         agent_edge_chordal=a_edge_chord,
                         gmsh_matched_edge_chordal=g_edge_chord,
                         agent_irregular=a_irregular,
                         gmsh_matched_irregular=g_irregular))
        panels.append((env.graph, g_nodes, g_elements, this, a_min, g_min, a_count,
                       len(g_elements), outline_raw, a_chord, g_chord))
        drawn += 1
        print(f"  seed {this:4d}  agent {a_count:3d}q min {a_min:+.3f}"
              f" chord v{a_chord:.3f} e{a_edge_chord:.3f}"
              f"  |  gmsh {len(g_elements):3d}q min {g_min:+.3f}"
              f" chord v{g_chord:.3f} e{g_edge_chord:.3f}",
              flush=True)

    if not rows:
        print("no domains scored"); return

    columns = 4
    figure, axes = plt.subplots(2 * ((len(panels) + columns - 1) // columns), columns,
                                figsize=(3.1 * columns,
                                         3.3 * 2 * ((len(panels) + columns - 1) // columns)),
                                squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for index, panel in enumerate(panels):
        graph, nodes, elements, this, a_min, g_min, aq, gq, outline, a_ch, g_ch = panel
        block, column = divmod(index, columns)
        draw_tiler(axes[2 * block][column], graph,
                   f"agent  seed {this}\n{aq} quads  min {a_min:+.2f}  chord {a_ch:.3f}")
        draw_gmsh(axes[2 * block + 1][column], nodes, elements,
                  f"gmsh  seed {this}\n{gq} quads  min {g_min:+.2f}  chord {g_ch:.3f}",
                  outline=outline)
    kind = "with holes" if args.holes else "hole-free"
    figure.suptitle(f"{args.label} against Gmsh at matched element count, curved {kind}"
                    f"\nshaded by element SHAPE quality: red degenerate, green regular",
                    fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    name = f"gallery_{'holes' if args.holes else 'plain'}.png"
    figure.savefig(os.path.join(args.out, name), dpi=115)
    plt.close(figure)
    print("wrote", os.path.join(args.out, name))

    # The same panels again, drawn as MESHES rather than as heat maps. A quality
    # ramp is the right thing for "is any element bad" and the wrong thing for
    # "what does this mesh look like" -- the structure the agent is actually
    # producing, block by block, is much easier to read without it.
    from utilities.gallery_curved import draw as draw_plain
    figure, axes = plt.subplots(2 * ((len(panels) + columns - 1) // columns), columns,
                                figsize=(3.1 * columns,
                                         3.3 * 2 * ((len(panels) + columns - 1) // columns)),
                                squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for index, panel in enumerate(panels):
        graph, nodes, elements, this, a_min, g_min, aq, gq, outline, a_ch, g_ch = panel
        block, column = divmod(index, columns)
        draw_plain(axes[2 * block][column], graph,
                   f"agent  seed {this}   {aq} quads", colour_by_quality=False)
        draw_gmsh_mesh(axes[2 * block + 1][column], nodes, elements,
                       f"gmsh  seed {this}   {gq} quads", outline=outline)
    figure.suptitle(f"{args.label} against Gmsh at matched element count, curved {kind}"
                    f"\nevery edge drawn, no quality shading", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    mesh_name = f"mesh_{'holes' if args.holes else 'plain'}.png"
    figure.savefig(os.path.join(args.out, mesh_name), dpi=115)
    plt.close(figure)
    print("wrote", os.path.join(args.out, mesh_name))

    # Keep the meshes. Re-rendering these panels any other way has meant
    # re-running the agent over every domain, twenty minutes for a picture.
    import pickle
    keep = [dict(seed=this, graph=graph, gmsh_nodes=np.asarray(nodes, dtype=float),
                 gmsh_elements=[list(e) for e in elements], outline=outline,
                 agent_min=a_min, gmsh_min=g_min, agent_quads=aq, gmsh_quads=gq)
            for graph, nodes, elements, this, a_min, g_min, aq, gq, outline, _, _ in panels]
    meshes = os.path.join(args.out, f"meshes_{'holes' if args.holes else 'plain'}.pkl")
    with open(meshes, "wb") as handle:
        pickle.dump(keep, handle)
    print("wrote", meshes)

    with open(os.path.join(args.out,
                           f"table_{'holes' if args.holes else 'plain'}.json"), "w") as handle:
        json.dump(rows, handle, indent=2)
    kind = "holes" if args.holes else "plain"
    quad_rows = [r for r in rows if r["all_quad"]]
    a = np.array([[r["agent_min"], r["gmsh_matched_min"], r["gmsh_natural_min"],
                   r["agent_chordal"], r["gmsh_matched_chordal"]]
                  for r in rows], dtype=float)
    # Quality among ALL-QUAD meshes only. A mesh with a leftover n-gon is not a
    # quad mesh at all, and letting its corners into the median compares the
    # agent's unfinished work against Gmsh's finished work.
    q = np.array([[r["agent_min"], r["gmsh_matched_min"], r["gmsh_natural_min"]]
                  for r in quad_rows], dtype=float) if quad_rows else np.zeros((0, 3))
    print(f"\nRESULT {args.label} {kind}: all-quad "
          f"{len(quad_rows)}/{len(rows)}", flush=True)
    if len(q):
        print(f"RESULT {args.label} {kind}: per-mesh MIN shape quality among ALL-QUAD "
              f"(median, n={len(q)}) -- agent {np.nanmedian(q[:,0]):+.3f}  "
              f"gmsh@matched {np.nanmedian(q[:,1]):+.3f}  "
              f"gmsh@natural {np.nanmedian(q[:,2]):+.3f}", flush=True)
    print(f"RESULT {args.label} {kind}: same over ALL meshes (median) -- "
          f"agent {np.nanmedian(a[:,0]):+.3f}  gmsh@matched {np.nanmedian(a[:,1]):+.3f}  "
          f"gmsh@natural {np.nanmedian(a[:,2]):+.3f}", flush=True)
    linear = np.array([r["agent_min_as_linear"] for r in rows], dtype=float)
    print(f"RESULT {args.label} {kind}: agent min quality, SAME meshes, two readings -- "
          f"as curved blocks {np.nanmedian(a[:,0]):+.3f}  "
          f"as a linear mesh {np.nanmedian(linear):+.3f}  "
          f"(gmsh, linear either way, {np.nanmedian(a[:,1]):+.3f})", flush=True)
    print(f"RESULT {args.label} {kind}: chordal error at VERTICES (median) -- "
          f"agent {np.nanmedian(a[:,3]):.4f}  gmsh@matched {np.nanmedian(a[:,4]):.4f}",
          flush=True)
    # The honest one. A vertex sits on the curve by construction; the EDGE
    # between two of them is what actually approximates the boundary, and a
    # circular hole carried on four control points is a diamond.
    edge = np.array([[r["agent_edge_chordal"], r["gmsh_matched_edge_chordal"]]
                     for r in rows], dtype=float)
    print(f"RESULT {args.label} {kind}: chordal error at EDGES (median) -- "
          f"agent {np.nanmedian(edge[:,0]):.4f}  "
          f"gmsh@matched {np.nanmedian(edge[:,1]):.4f}", flush=True)
    print(f"RESULT {args.label} {kind}: chordal error at EDGES (worst domain) -- "
          f"agent {np.nanmax(edge[:,0]):.4f}  "
          f"gmsh@matched {np.nanmax(edge[:,1]):.4f}", flush=True)
    irr = np.array([[r["agent_irregular"], r["gmsh_matched_irregular"]]
                    for r in rows], dtype=float)
    print(f"RESULT {args.label} {kind}: irregular degree per vertex (median) -- "
          f"agent {np.nanmedian(irr[:,0]):.3f}  gmsh@matched {np.nanmedian(irr[:,1]):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
