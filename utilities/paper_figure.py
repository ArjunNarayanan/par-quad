"""The paper's mesh figures: the agent against Gmsh on the same domains.

Three steps, each re-runnable on its own so the drawing never reruns the agent:

    # 1. pick the domains: per suite, the one closest to that suite's medians
    venv/bin/python utilities/paper_figure.py select -scores out/paper/mit-q-2M-b6-repair3 \\
        -suites straight-holes,straight-transfer

    # 2. compute: the agent's mesh under the paper's procedure (the scorer itself,
    #    via -draws / -save_meshes), then Gmsh blossom at the agent's element count
    #    and Gmsh quasi-structured as close to it as it can get, both untangled as
    #    in the tables
    venv/bin/python utilities/paper_figure.py compute -suite straight-holes -draw 11 \\
        -config models/paper-2026-09-24/pinwheel-mit-q-v1.config.yml \\
        -checkpoint models/paper-2026-09-24/pinwheel-mit-q-v1-2M.zip -out out/figure

    # 3. render: vector PDF at the paper's text width
    venv/bin/python utilities/paper_figure.py render -rows straight-holes:11,straight-transfer:3 \\
        -meshes out/figure -out figures/fig-gallery.pdf

What is drawn: every element in one neutral fill; elements below the usability
bar (shape quality < 0.3, the tables' bar) tinted and hatched; non-quadrilateral
elements tinted; every irregular vertex marked, over-connected and
under-connected with different markers. No colour ramp.
"""
import argparse
import glob
import json
import os
import pickle
import subprocess
import sys

import numpy as np

sys.path.append(os.getcwd())

PROCEDURE = ["-n_samples", "4", "-max_steps_factor", "6.0", "-repair_rounds", "3", "-repair_bar", "0.3"]


# --------------------------------------------------------------------------- select

def _median(values):
    return float(np.median(values)) if values else float("nan")


def select(args):
    """Per suite: among domains whose outcome is the suite's typical one (all-quad,
    usable, excess over par equal to the median), the one closest to the medians of
    corner count, element count and quality, each scaled by its spread."""
    for suite in [s for s in args.suites.split(",") if s]:
        path = os.path.join(args.scores, f"{suite}.{args.rollout_seed}.json")
        domains = json.load(open(path))["suites"][suite]["domains"]
        quads = [d for d in domains if d["face"] == 0]
        med = {k: _median([d[k] for d in quads]) for k in ("quality", "elements", "corners", "excess")}
        spread = {k: (np.percentile([d[k] for d in quads], 75) - np.percentile([d[k] for d in quads], 25)) or 1.0
                  for k in ("quality", "elements", "corners", "excess")}
        typical = [d for d in quads if d["quality"] >= args.bar and d["excess"] == round(med["excess"])]
        keys = ("quality", "elements", "corners")
        if not typical:
            # no domain has exactly the typical outcome (a spread-out suite): rank every
            # all-quad domain, with excess over par as a fourth distance term
            print(f"{suite}: no domain with the typical outcome; ranking all all-quad domains")
            typical, keys = quads, ("quality", "elements", "corners", "excess")

        def distance(d):
            return sum(abs(d[k] - med[k]) / spread[k] for k in keys)
        ranked = sorted(typical, key=distance)
        if args.require_arcs:
            # a curved preset can draw a straight-sided part; a curved gallery row must be curved
            kept = []
            for d in ranked:
                geometry, _ = _geometry(suite, d["draw"], args.seed)
                if any(any(loop.is_arc()) for loop in geometry.loops):
                    kept.append(d)
                if len(kept) >= args.top:
                    break
            ranked = kept
        print(f"{suite}: median corners {med['corners']:.0f}, elements {med['elements']:.0f}, "
              f"quality {med['quality']:+.2f}, excess {med['excess']:.0f}")
        for d in ranked[:args.top]:
            print(f"   draw {d['draw']:3d}  corners {d['corners']:2d}  elements {d['elements']:3d}  "
                  f"q {d['quality']:+.3f}  excess {d['excess']}  distance {distance(d):.2f}")


# --------------------------------------------------------------------------- compute

def _geometry(suite, draw, seed):
    """The domain the scorer drew at index `draw` (same options, same filters), and its
    corner count as the suite filter counts it (boundary vertices before densification,
    holes included -- what the paper's 8-24 / 25-50 ranges mean)."""
    from src.geo2d_bridge import make_initializer
    from utilities.score_quality_objective import SUITES, MIN_CORNERS, MAX_EDGE_RATIO, boundary_edge_ratio
    options = dict(SUITES[suite])
    capped = "min_corners" in options
    options.setdefault("max_corners", 24)
    options.setdefault("min_corners", MIN_CORNERS)
    # no densification here: it is applied after the draw (the domains are the same with or
    # without it) and it adds entries to the corner table, which is counted below
    source = make_initializer(seed=seed, **options)
    index = -1
    while index < draw:
        index += 1
        try:
            graph, _desired = source()
            _corner_count = len(_desired)
        except Exception:
            if index == draw:
                raise RuntimeError(f"draw {draw} of {suite} failed to build")
            continue
        if index == draw:
            if capped and boundary_edge_ratio(graph) > MAX_EDGE_RATIO:
                raise RuntimeError(f"draw {draw} of {suite} is filtered out by the edge-ratio cap")
            return source.last_geometry, _corner_count
    raise RuntimeError("unreachable")


def _node_degrees(elements):
    from collections import defaultdict
    corners, uses = defaultdict(int), defaultdict(int)
    for element in elements:
        for i in range(len(element)):
            a, b = int(element[i]), int(element[(i + 1) % len(element)])
            corners[a] += 1
            uses[(min(a, b), max(a, b))] += 1
    boundary = {n for (a, b), u in uses.items() if u == 1 for n in (a, b)}
    return {v: k + (1 if v in boundary else 0) for v, k in corners.items()}, boundary


def _boundary_nodes(elements):
    from collections import Counter
    uses = Counter()
    for element in elements:
        for i in range(len(element)):
            a, b = int(element[i]), int(element[(i + 1) % len(element)])
            uses[(min(a, b), max(a, b))] += 1
    return {n for (a, b), u in uses.items() if u == 1 for n in (a, b)}


def _gmsh(geometry, target, variant, par, agent=None):
    """Gmsh's mesh as the tables score it: searched to `target` elements, interior
    untangled with the same smoother, degrees read against the domain's wants."""
    from src.geo2d_bridge import import_geo2d, resmooth_mesh
    from utilities.compare_gmsh import VARIANTS, corner_wants, domain_corner_wants, mesh_at_element_count
    geo2d = import_geo2d()
    found = mesh_at_element_count(geometry, max(int(target), 1), VARIANTS[variant])
    if found is None:
        return None
    _, _, nodes, elements = found
    elements = [list(map(int, e)) for e in elements]
    mesh = geo2d.Mesh(nodes, elements)
    try:
        mesh = resmooth_mesh(mesh, mesh.is_boundary(), iters=4, method="optimize", slide=False)
    except Exception:
        pass
    nodes = np.asarray(mesh.nodes, float)
    face_quality = [float(v) for v in np.asarray(mesh.quality(), float)]
    # the agent's Tiler is centred and scaled by the bridge; put Gmsh in the same
    # frame so a row's panels share one scale (quality is scale-free, drawing is not)
    from src.geo2d_bridge import geometry_loops
    curved = any(any(loop.is_arc()) for loop in geometry.loops)
    if curved:
        # the frame `curved_geometry_to_tiler` applies (curved_report.normalizer)
        from utilities.curved_report import normalizer
        centre, scale = normalizer(geometry)
    else:
        outer, _ = geometry_loops(geometry, drop_collinear=True)
        centre = outer.mean(axis=0)
        scale = max(float(np.abs(outer - centre).max()), 1e-9)
    # straight domains: corner wants from the raw-coordinate table; curved ones start
    # from the generic wants and take their corners from the agent's table below
    wants, tol = ({}, 1e-6) if curved else domain_corner_wants(geometry, normalize=False)
    degree, boundary = _node_degrees(elements)
    vertices = []
    for node, deg in degree.items():
        if node in boundary:
            key = (round(float(nodes[node][0]) / tol), round(float(nodes[node][1]) / tol))
            options = wants.get(key) or corner_wants(180.0)
        else:
            options = (4,)
        vertices.append(dict(node=node, degree=int(deg), wants=tuple(int(w) for w in options)))
    if curved and agent is not None:
        # corner wants on a curved domain are TANGENT angles, which the raw-coordinate
        # table above (chord angles) gets wrong at arc joints; read them off the agent's
        # own mesh instead: its boundary vertices that want something other than 3 are
        # the domain's corners, and they never move, so Gmsh's corner nodes coincide
        # with them in the common frame
        framed = (nodes - centre) / scale
        matched = 0
        a_nodes = np.asarray(agent["nodes"], float)
        a_boundary = _boundary_nodes(agent["elements"])
        corners = [(a_nodes[v["node"]], v["wants"]) for v in agent["vertices"]
                   if v["node"] in a_boundary and tuple(v["wants"]) != (3,)]
        for vertex in vertices:
            if vertex["node"] not in boundary:
                continue
            here = framed[vertex["node"]]
            match = [w for p, w in corners if float(np.hypot(*(p - here))) < 1e-4]
            vertex["wants"] = tuple(match[0]) if match else (3,)
            matched += bool(match)
        # every corner of the domain is a node of Gmsh's mesh too; if they do not all
        # match, the two frames disagree and Gmsh's irregularity would be misread
        print(f"  {variant}: matched {matched} of {len(corners)} agent corners", flush=True)
        if matched < len(corners):
            raise RuntimeError(f"{variant}: only {matched} of {len(corners)} corners matched; frames disagree")
    return dict(nodes=(nodes - centre) / scale, elements=elements, face_quality=face_quality,
                vertices=vertices, variant=variant, par=par,
                corners=None)   # set by compute(): the suite's own corner count


def compute(args):
    os.makedirs(args.out, exist_ok=True)
    agent_path = os.path.join(args.out, f"{args.suite}.draw{args.draw:03d}.pkl")
    if not os.path.exists(agent_path) or args.force:
        command = [sys.executable, "utilities/score_quality_objective.py", "-config", args.config,
                   "-checkpoint", args.checkpoint, "-suites", args.suite, "-n", str(args.n),
                   "-seed", str(args.seed), "-rollout_seed", str(args.rollout_seed),
                   "-draws", str(args.draw), "-save_meshes", args.out] + PROCEDURE
        if args.densify:
            command += ["-densify", str(args.densify)]
        if args.repair_budget:
            command += ["-repair_budget", str(args.repair_budget)]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
    agent = pickle.load(open(agent_path, "rb"))
    record = agent["record"]
    print(f"agent: {record}", flush=True)
    geometry, corner_count = _geometry(args.suite, args.draw, args.seed)
    for variant in ("blossom", "quasi-structured"):
        mesh = _gmsh(geometry, record["elements"], variant, record["par"], agent=agent)
        if mesh is not None:
            mesh["corners"] = int(corner_count)
        path = os.path.join(args.out, f"{args.suite}.draw{args.draw:03d}.{variant}.pkl")
        pickle.dump(mesh, open(path, "wb"))
        if mesh is not None:
            print(f"{variant}: {len(mesh['elements'])} elements, min q {min(mesh['face_quality']):+.3f}, "
                  f"irregular {sum(_deviation(v) != 0 for v in mesh['vertices'])}", flush=True)


# --------------------------------------------------------------------------- arcs

def _domain_arcs(meshes_dir, suite, draw, seed=7):
    """The domain's arcs, densely sampled, in the frame the meshes are drawn in
    (cached next to the meshes). Empty for a straight-sided domain."""
    path = os.path.join(meshes_dir, f"{suite}.draw{draw:03d}.arcs.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    arcs = []
    if suite.startswith("curved"):
        from src.geo2d_bridge import _loop_arrays
        from utilities.curved_report import normalizer
        geometry, _ = _geometry(suite, draw, seed)
        centre, scale = normalizer(geometry)
        for index, loop in enumerate(geometry.loops):
            _, _, entries = _loop_arrays(loop, ccw=(index == 0), drop_collinear=False)
            for arc, _start in entries.values():
                points = np.array([arc.point(t) for t in np.linspace(0.0, 1.0, 400)], dtype=float)
                arcs.append((points - centre) / scale)
    pickle.dump(arcs, open(path, "wb"))
    return arcs


def _on_polyline(point, polyline):
    """(distance from `point` to the polyline, the fractional index of its foot)."""
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30), 0, 1)
    foot = a + t[:, None] * ab
    d = np.hypot(*(foot - point).T)
    k = int(np.argmin(d))
    return float(d[k]), k + float(t[k])


def _arc_path(p, q, arcs, tol):
    """Points from `p` to `q` along the domain arc both lie on, or None if they do not
    share one. Endpoints are exactly `p` and `q`."""
    for polyline in arcs:
        dp, sp = _on_polyline(p, polyline)
        if dp > tol:
            continue
        dq, sq = _on_polyline(q, polyline)
        if dq > tol:
            continue
        lo, hi = sorted((sp, sq))
        inner = polyline[int(np.floor(lo)) + 1:int(np.ceil(hi))]
        if sp > sq:
            inner = inner[::-1]
        return np.vstack([p, inner, q]) if len(inner) else np.vstack([p, q])
    return None


def _element_outline(element, nodes, boundary_edges, arcs, tol):
    """The element's outline, with every boundary edge that lies on a domain arc drawn
    along the arc: the side the paper judges an element on."""
    points = []
    for i in range(len(element)):
        a, b = element[i], element[(i + 1) % len(element)]
        segment = None
        if arcs and (min(a, b), max(a, b)) in boundary_edges:
            segment = _arc_path(nodes[a], nodes[b], arcs, tol)
        points.append(segment[:-1] if segment is not None else nodes[[a]])
    return np.vstack(points)


# --------------------------------------------------------------------------- render

def _deviation(vertex):
    """Signed distance of a vertex's degree from the nearest degree it wants."""
    return min((vertex["degree"] - w for w in vertex["wants"]), key=abs)


# Okabe-Ito, so the two marker colours and the tint survive colour-blindness;
# every category also differs in shape or hatching, so the figure survives greyscale
FILL = "#eef1f4"
EDGE = "#7b8794"
BOUNDARY = "#1f2933"
LOW_FILL = "#f4b183"      # below the usability bar
LOW_EDGE = "#b5541c"
NONQUAD_FILL = "#c9c3e6"  # not a quadrilateral
OVER = "#0072B2"          # vertex with more edges than it wants
UNDER = "#D55E00"         # vertex with fewer edges than it wants
INK = "#1f2933"
MUTED = "#52606d"

COLUMNS = [("agent", "Ours"), ("blossom", "Gmsh blossom, closest count"),
           ("quasi-structured", "Gmsh quasi-structured")]
SUITE_LABEL = {
    "straight": "chamfered", "polycube": "rectilinear",
    "straight-holes": "chamfered, holes", "polycube-holes": "rectilinear, holes",
    "straight-transfer": "chamfered", "polycube-transfer": "rectilinear",
    "straight-holes-transfer": "chamfered, holes", "polycube-holes-transfer": "rectilinear, holes",
    "curved": "curved", "curved-holes": "curved, holes",
    "curved-transfer": "curved", "curved-holes-transfer": "curved, holes",
}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "pdf.fonttype": 42,      # TrueType, not Type 3: what conference PDF checks want
        "ps.fonttype": 42,
        "hatch.linewidth": 0.45,
        "axes.linewidth": 0.0,
    })


def _load(meshes, suite, draw):
    base = os.path.join(meshes, f"{suite}.draw{draw:03d}")
    agent = pickle.load(open(base + ".pkl", "rb"))
    out = {"agent": agent}
    for variant in ("blossom", "quasi-structured"):
        path = f"{base}.{variant}.pkl"
        if os.path.exists(path):          # a gallery may carry only some of the Gmsh columns
            out[variant] = pickle.load(open(path, "rb"))
    return out


def _stats(mesh, bar):
    quads = all(len(e) == 4 for e in mesh["elements"])
    deviations = [_deviation(v) for v in mesh["vertices"]]
    irregular = sum(d != 0 for d in deviations)
    low = sum(q < bar for q, e in zip(mesh["face_quality"], mesh["elements"]) if len(e) == 4)
    # a curved gallery carries chord quality beside the tangent one (hatching stays on the tangent)
    chord = min(mesh["face_quality_chord"]) if mesh.get("face_quality_chord") else None
    return dict(n=len(mesh["elements"]), quads=quads, irregular=irregular, low=low,
                q=min(mesh["face_quality"]), q_chord=chord,
                nonquad=sum(len(e) != 4 for e in mesh["elements"]))


def _draw(axis, mesh, bar, scale, arcs=()):
    from collections import Counter
    from matplotlib.patches import Polygon
    nodes = np.asarray(mesh["nodes"], float)
    edge_width = 0.45 * scale
    uses = Counter()
    for element in mesh["elements"]:
        for i in range(len(element)):
            a, b = element[i], element[(i + 1) % len(element)]
            uses[(min(a, b), max(a, b))] += 1
    boundary_edges = {edge for edge, count in uses.items() if count == 1}
    extent = float(np.ptp(nodes, axis=0).max()) if len(nodes) else 1.0
    tol = 2e-3 * extent
    for element, quality in zip(mesh["elements"], mesh["face_quality"]):
        points = _element_outline(element, nodes, boundary_edges, arcs, tol)
        if len(element) != 4:
            patch = Polygon(points, closed=True, facecolor=NONQUAD_FILL, edgecolor=EDGE,
                            linewidth=edge_width, zorder=2)
        elif quality < bar:
            patch = Polygon(points, closed=True, facecolor=LOW_FILL, edgecolor=LOW_EDGE,
                            linewidth=edge_width * 1.3, hatch="////", zorder=3)
        else:
            patch = Polygon(points, closed=True, facecolor=FILL, edgecolor=EDGE,
                            linewidth=edge_width, zorder=1)
        axis.add_patch(patch)
    # the domain boundary: every edge used by exactly one face, drawn over the mesh,
    # along the domain's arc where it lies on one
    for (a, b) in boundary_edges:
        segment = _arc_path(nodes[a], nodes[b], arcs, tol) if arcs else None
        if segment is None:
            segment = nodes[[a, b]]
        axis.plot(segment[:, 0], segment[:, 1], color=BOUNDARY, linewidth=1.0 * scale,
                  solid_capstyle="round", solid_joinstyle="round", zorder=4)
    over = np.array([nodes[v["node"]] for v in mesh["vertices"] if _deviation(v) > 0]).reshape(-1, 2)
    under = np.array([nodes[v["node"]] for v in mesh["vertices"] if _deviation(v) < 0]).reshape(-1, 2)
    size = 11 * scale
    axis.scatter(over[:, 0], over[:, 1], s=size, marker="o", facecolor=OVER, edgecolor="white",
                 linewidth=0.35, zorder=6)
    axis.scatter(under[:, 0], under[:, 1], s=size * 1.25, marker="^", facecolor=UNDER, edgecolor="white",
                 linewidth=0.35, zorder=6)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def _caption(stats, bar):
    first = f"{stats['n']} elements, {stats['irregular']} irregular"
    fmt = lambda q: f"{q + 0.0:.2f}".replace("-0.00", "0.00")
    if stats.get("q_chord") is not None:
        second = f"min $q$ = {fmt(stats['q'])} tangent, {fmt(stats['q_chord'])} chord"
        third = (f"{stats['nonquad']} not quad" if not stats["quads"] else
                 f"{stats['low']} below {bar:g} on the tangent" if stats["low"] else "")
        return first + "\n" + second + ("\n" + third if third else "")
    second = f"min $q$ = {fmt(stats['q'])}"
    if not stats["quads"]:
        second += f", {stats['nonquad']} not quad"
    elif stats["low"]:
        second += f", {stats['low']} below {bar:g}"
    return first + "\n" + second


def render(args):
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rows = []
    for token in args.rows.split(","):
        suite, draw = token.split(":")
        rows.append((suite, int(draw)))
    panel_height = args.row_height
    columns = [c for c in COLUMNS if c[0] in args.columns.split(",")]
    figure, axes = plt.subplots(len(rows), len(columns), figsize=(args.width, panel_height * len(rows) + 0.2),
                                squeeze=False, gridspec_kw=dict(wspace=0.04, hspace=0.36))
    for r, (suite, draw) in enumerate(rows):
        meshes = _load(args.meshes, suite, draw)
        agent = meshes["agent"]
        # one scale per row: the union of the three panels' extents
        points = np.vstack([np.asarray(m["nodes"], float) for m in meshes.values() if m is not None])
        low, high = points.min(axis=0), points.max(axis=0)
        pad = 0.04 * float((high - low).max())
        # thinner lines and smaller markers on a finer mesh
        count = max(len(agent["elements"]), 1)
        scale = float(np.clip((20.0 / count) ** 0.25, 0.55, 1.0))
        for c, (key, title) in enumerate(columns):
            axis = axes[r][c]
            mesh = meshes[key]
            if mesh is None:
                axis.text(0.5, 0.5, "Gmsh failed", ha="center", va="center", transform=axis.transAxes)
                axis.axis("off")
                continue
            _draw(axis, mesh, args.bar, scale, arcs=_domain_arcs(args.meshes, suite, draw))
            axis.set_xlim(low[0] - pad, high[0] + pad)
            axis.set_ylim(low[1] - pad, high[1] + pad)
            stats = _stats(mesh, args.bar)
            axis.set_xlabel(_caption(stats, args.bar), fontsize=7, color=MUTED, labelpad=2.5,
                            linespacing=1.15)
            if r == 0:
                axis.set_title(title, fontsize=8.5, color=INK, pad=4,
                               fontweight="bold" if key == "agent" else "normal")
        corners = next((m["corners"] for m in meshes.values() if m is not None and "corners" in m),
                       agent["record"].get("corners", -1))
        mixed = any(not s.startswith("curved") for s, _ in rows)
        if suite.startswith("curved") and mixed:
            # beside straight-sided rows (the teaser): the curved row is its own setting
            rest = SUITE_LABEL.get(suite, suite).replace("curved", "").strip(", ")
            label = f"curved boundary\n{rest + ', ' if rest else ''}{corners} corners"
        else:
            if suite.startswith("curved"):
                # label a curved gallery by the suite's corner range
                setting = "25–50-corner suite" if "transfer" in suite else "8–24-corner suite"
            else:
                setting = "larger boundary" if "transfer" in suite else "in-distribution"
            label = f"{setting}\n{SUITE_LABEL.get(suite, suite)}, {corners} corners"
        axes[r][0].set_ylabel(label, fontsize=7.5, color=INK, labelpad=4)
    handles = [
        Patch(facecolor=FILL, edgecolor=EDGE, linewidth=0.5, label="quadrilateral"),
        Patch(facecolor=LOW_FILL, edgecolor=LOW_EDGE, hatch="////", linewidth=0.5,
              label=f"below the usability bar ($q<{args.bar:g}$)"),
        Patch(facecolor=NONQUAD_FILL, edgecolor=EDGE, linewidth=0.5, label="not a quadrilateral"),
        Line2D([], [], marker="o", linestyle="none", markersize=4, markerfacecolor=OVER,
               markeredgecolor="white", markeredgewidth=0.35, label="too many edges"),
        Line2D([], [], marker="^", linestyle="none", markersize=4.4, markerfacecolor=UNDER,
               markeredgecolor="white", markeredgewidth=0.35, label="too few edges"),
    ]
    figure.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.075)
    # the legend hangs below the last row's labels; bbox_inches="tight" keeps it
    figure.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7,
                  handlelength=1.3, handletextpad=0.5, columnspacing=1.1,
                  bbox_to_anchor=(0.5, 0.0), bbox_transform=figure.transFigure)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    figure.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    if args.png:
        figure.savefig(args.png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print("wrote", args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select")
    p.add_argument("-scores", required=True)
    p.add_argument("-suites", required=True)
    p.add_argument("-rollout_seed", default=0, type=int)
    p.add_argument("-bar", default=0.3, type=float)
    p.add_argument("-top", default=5, type=int)
    p.add_argument("-seed", default=7, type=int)
    p.add_argument("-require_arcs", action="store_true", help="only domains with at least one arc")
    p = sub.add_parser("compute")
    p.add_argument("-suite", required=True)
    p.add_argument("-draw", required=True, type=int)
    p.add_argument("-config", required=True)
    p.add_argument("-checkpoint", required=True)
    p.add_argument("-n", default=24, type=int, help="the suite size the draw index came from")
    p.add_argument("-seed", default=7, type=int)
    p.add_argument("-rollout_seed", default=0, type=int)
    p.add_argument("-out", default="out/figure")
    p.add_argument("-force", action="store_true")
    p.add_argument("-densify", default=0.0, type=float, help="arc densification (degrees), as scored")
    p.add_argument("-repair_budget", default=None, type=int, help="cap per repair continuation")
    p = sub.add_parser("render")
    p.add_argument("-rows", required=True, help="suite:draw,suite:draw,...")
    p.add_argument("-meshes", default="out/figure")
    p.add_argument("-out", required=True)
    p.add_argument("-bar", default=0.3, type=float)
    p.add_argument("-width", default=5.5, type=float, help="inches; the ICLR text width")
    p.add_argument("-row_height", default=1.45, type=float, help="inches per domain row")
    p.add_argument("-png", default=None, help="also write a 300-dpi PNG (for review)")
    p.add_argument("-columns", default="agent,blossom,quasi-structured", help="which panels, in COLUMNS order")
    args = parser.parse_args()
    {"select": select, "compute": compute, "render": render}[args.command](args)


if __name__ == "__main__":
    main()
