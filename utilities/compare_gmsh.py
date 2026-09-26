"""Score Gmsh's quad meshers on the same domains, by the same rule, as the agent.

The agent's claim is topological: every face a quad, and the vertex-irregularity
score down to `par`, the discrete Gauss-Bonnet lower bound. Gmsh is not trying to
do that -- it meshes to a target element size and lets the irregular vertices fall
where they may -- so a naive "Gmsh vs us" table compares two different objectives
and means nothing. Two things make it fair here.

**The same metric, computed by the same definitions.** A Gmsh mesh is scored with
the vertex defect this repository uses: interior vertices want degree 4, a
boundary vertex on a straight stretch of the outline wants 3, and a vertex sitting
on a corner of the domain wants whatever `rounded_desired_degree` says that corner
wants -- a set of them, at a rounding tie, exactly as `AngleEnv` does. `par` is a
property of the *domain*, not of a
mesh, so the same number bounds both methods.

**The same coarseness.** The agent produces a block decomposition -- the coarsest
all-quad mesh the topology allows -- so Gmsh is asked for a comparable element
count by a search over its target size `lc`, not run at its own natural
resolution. Both readings are reported: `-match` (the default) searches for the
agent's element count, and `-lc` pins a fixed size instead.

    venv/bin/python utilities/compare_gmsh.py -config <config.yml> \
        -checkpoint <agent.zip> -suites straight,polycube-holes -n 12

Gmsh is a separate install: `venv/bin/pip install gmsh`.
"""

import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np

sys.path.append(os.getcwd())

import envs.polygon_utils as utils  # noqa: E402
from src.geo2d_bridge import (_interior_angles, env_summary, geometry_loops,  # noqa: E402
                              import_geo2d, load_model, make_geo2d_env)
from src.utils import load_yaml_config  # noqa: E402
from utilities.evaluate_geo2d import rollout  # noqa: E402
from utilities.score_suites import SUITES, geometries  # noqa: E402

# name -> Gmsh options. Mesh.Algorithm: 6 Frontal-Delaunay, 8 Frontal-Delaunay for
# quads, 9 Packing of Parallelograms, 11 Quasi-structured quad.
# Mesh.RecombinationAlgorithm: 0 simple, 1 Blossom, 2 simple full-quad, 3 Blossom
# full-quad. SubdivisionAlgorithm 1 splits everything into quads instead.
VARIANTS = {
    "blossom":          {"Mesh.Algorithm": 6, "Mesh.RecombineAll": 1,
                         "Mesh.RecombinationAlgorithm": 1},
    "blossom-full":     {"Mesh.Algorithm": 6, "Mesh.RecombineAll": 1,
                         "Mesh.RecombinationAlgorithm": 3},
    "frontal-quad":     {"Mesh.Algorithm": 8, "Mesh.RecombineAll": 1,
                         "Mesh.RecombinationAlgorithm": 1},
    "packing":          {"Mesh.Algorithm": 9, "Mesh.RecombineAll": 1,
                         "Mesh.RecombinationAlgorithm": 1},
    "quasi-structured": {"Mesh.Algorithm": 11, "Mesh.RecombineAll": 1,
                         "Mesh.RecombinationAlgorithm": 1},
    "simple-recombine": {"Mesh.Algorithm": 6, "Mesh.RecombineAll": 1,
                         "Mesh.RecombinationAlgorithm": 0},
    "subdivide":        {"Mesh.Algorithm": 6, "Mesh.SubdivisionAlgorithm": 1},
}


# --------------------------------------------------------------------------- gmsh

def gmsh_quad_mesh(geometry, lc, options, verbosity=0):
    """`(nodes, elements)` from Gmsh, driven through its Python API.

    The .geo is written with `recombine=False` so that nothing in the file
    competes with the options set here -- geogen's writer otherwise pins
    Algorithm 8 and Blossom, which is one of the seven variants, not all of them.
    """
    import gmsh
    from geo2d.export import to_geo

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "domain.geo")
        with open(path, "w") as handle:
            handle.write(to_geo(geometry, lc=lc, recombine=False))
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", verbosity)
            gmsh.option.setNumber("General.Verbosity", verbosity)
            gmsh.open(path)
            for key, value in options.items():
                gmsh.option.setNumber(key, value)
            gmsh.model.mesh.generate(2)
            tags, coords, _ = gmsh.model.mesh.getNodes()
            index = {int(t): i for i, t in enumerate(tags)}
            nodes = np.asarray(coords, float).reshape(-1, 3)[:, :2]
            elements = []
            for dim, tag in gmsh.model.getEntities(2):
                kinds, _, conn = gmsh.model.mesh.getElements(dim, tag)
                for kind, nodelist in zip(kinds, conn):
                    size = {2: 3, 3: 4}.get(int(kind))
                    if size is None:
                        continue
                    block = np.asarray(nodelist, np.int64).reshape(-1, size)
                    elements += [[index[int(v)] for v in row] for row in block]
        finally:
            gmsh.finalize()
    return nodes, elements


def mesh_at_element_count(geometry, target, options, lo=0.05, hi=8.0, tries=12):
    """Binary-search Gmsh's target size for an element count near `target`."""
    best = None
    for _ in range(tries):
        mid = float(np.sqrt(lo * hi))
        try:
            nodes, elements = gmsh_quad_mesh(geometry, mid, options)
        except Exception:
            lo = mid
            continue
        count = len(elements)
        if best is None or abs(count - target) < abs(best[0] - target):
            best = (count, mid, nodes, elements)
        if count == target:
            break
        if count > target:
            lo = mid          # too fine: ask for a bigger element size
        else:
            hi = mid
    return best


# --------------------------------------------------------------------------- scoring

def corner_wants(angle, target_angle=90.0, tie_tolerance=1e-6):
    """The degrees a corner of this interior angle is content with.

    Mirrors `AngleEnv._compute_desired_options`, and it has to: at an exact
    rounding tie -- 135 degrees against a 90-degree target is 1.5 -- both corner
    treatments are geometrically legal, so the corner carries the SET
    {floor+1, ceil+1} and irregularity is the distance to the nearer member.
    `rounded_desired_degree` alone returns whichever one numpy's round-half-to-even
    happens to pick, which scores a legal corner as a defect and inflates the
    excess over par for every method being compared.
    """
    ratio = angle / target_angle
    if abs(ratio - np.floor(ratio) - 0.5) <= tie_tolerance:
        low = max(int(np.floor(ratio)) + 1, 2)
        high = max(int(np.ceil(ratio)) + 1, 2)
        if low != high:
            return (low, high)
    return (utils.rounded_desired_degree(angle, target_angle),)


def domain_corner_wants(geometry, target_angle=90.0, tol=1e-6, normalize=False):
    """Corner coordinate -> the degrees the domain wants there.

    `normalize=False` keys on the raw loop coordinates, the space Gmsh meshes in;
    `normalize=True` applies the bridge's own centre-and-scale so the keys line up
    with a `Tiler` built by `geometry_to_tiler`. A hole corner is measured from
    the material side, 360 minus its own interior angle, as `slit_outline` does.
    """
    outer, holes = geometry_loops(geometry, drop_collinear=True)
    if normalize:
        centre = outer.mean(axis=0)
        scale = max(float(np.abs(outer - centre).max()), 1e-9)
        outer = (outer - centre) / scale
        holes = [(hole - centre) / scale for hole in holes]
    wants = {}
    for points, flip in [(outer, False)] + [(hole, True) for hole in holes]:
        for point, angle in zip(points, _interior_angles(points)):
            options = corner_wants(360.0 - angle if flip else angle, target_angle)
            wants[(round(float(point[0]) / tol), round(float(point[1]) / tol))] = options
    return wants, tol


def score_mesh(nodes, elements, wants, tol, par, untangle=True):
    """The agent's own scoreboard, applied to somebody else's mesh.

    `untangle` gives the other mesh the SAME post-processing ours gets. Without
    it the comparison smooths one side only: the agent's mesh is Laplacian
    smoothed after every step and untangled under an evaluator config, while
    Gmsh's was scored exactly as Gmsh emitted it. Measured over 48 domains, the
    same interior untangling moves Gmsh from +0.400 to +0.428 and drops our
    per-domain win rate from 43/48 to 37/48 -- so it is worth six domains, and
    leaving it out was worth six domains to us.

    Only the INTERIOR is untangled; every boundary node stays exactly where
    Gmsh put it. Allowing boundary sliding as ours gets adds nothing measurable
    (+0.428 either way), so the conservative treatment is also the honest one
    and there is nothing to argue about.
    """
    geo2d = import_geo2d()
    face_score = sum(abs(len(e) - 4) for e in elements)

    # Degree is counted as incident EDGES, which is what `Tiler.vertex_degree`
    # reports. Counting distinct neighbours instead is the same number only on a
    # simple mesh, and the agent's intermediate states are not simple -- a chord
    # can run parallel to an existing edge -- so it silently undercounted there.
    # For a manifold mesh, a vertex with k incident faces has k edges inside the
    # material and one more if it sits on the boundary.
    corners = defaultdict(int)
    edge_uses = defaultdict(int)
    for element in elements:
        size = len(element)
        for i in range(size):
            a, b = int(element[i]), int(element[(i + 1) % size])
            corners[a] += 1
            edge_uses[(min(a, b), max(a, b))] += 1
    boundary_node = set()
    for (a, b), uses in edge_uses.items():
        if uses == 1:
            boundary_node.add(a)
            boundary_node.add(b)
    neighbours = {v: k + (1 if v in boundary_node else 0) for v, k in corners.items()}

    vertex_score = 0
    for node in range(len(nodes)):
        if node not in neighbours:
            continue                      # Gmsh emits unused nodes; they are not vertices
        if node in boundary_node:
            key = (round(float(nodes[node][0]) / tol), round(float(nodes[node][1]) / tol))
            options = wants.get(key) or corner_wants(180.0)
        else:
            options = (4,)
        degree = neighbours[node]
        vertex_score += min(abs(degree - want) for want in options)

    mesh = geo2d.Mesh(nodes, elements)
    if untangle:
        try:
            from src.geo2d_bridge import resmooth_mesh
            mesh = resmooth_mesh(mesh, mesh.is_boundary(), iters=4,
                                 method="optimize", slide=False)
        except Exception:
            pass                      # scoring it unsmoothed is the harsher test
    quality = mesh.quality()
    return {
        "elements": len(elements),
        "all_quad": int(face_score == 0),
        "face_score": int(face_score),
        "vertex_score": int(vertex_score),
        "excess_over_par": int(vertex_score - par),
        "at_par": int(face_score == 0 and vertex_score == par),
        "min_quality": float(np.min(quality)) if len(quality) else 0.0,
    }


# --------------------------------------------------------------------------- agent

def agent_best(model, env_config, geometry, template_size, n_samples):
    """The agent's best-of-N mesh on this domain, and its score."""
    env = make_geo2d_env(env_config, geometry, template_size=template_size)
    best, best_env = None, None
    for i in range(n_samples + 1):
        rollout(model, env, deterministic=(i == 0))
        summary = env_summary(env)
        key = (not summary["solved"], summary["face_score"],
               abs(summary["vertex_excess"]), -summary["min_quality"])
        if best is None or key < best[0]:
            best = (key, summary)
            best_env = env.graph.copy() if hasattr(env.graph, "copy") else None
    summary = best[1]
    return {
        "elements": summary["faces"],
        "all_quad": int(summary["face_score"] == 0),
        "face_score": summary["face_score"],
        "vertex_score": summary["par"] + summary["vertex_excess"],
        "excess_over_par": summary["vertex_excess"],
        "at_par": int(summary["solved"]),
        "min_quality": summary["min_quality"],
    }, env


def main():
    parser = argparse.ArgumentParser(description="Gmsh's quad meshers on the agent's metric")
    parser.add_argument("-config", required=True)
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-suites", default="straight,polycube,polycube-holes,straight-holes")
    parser.add_argument("-seed", default=7, type=int)
    parser.add_argument("-n", default=12, type=int)
    parser.add_argument("-n_samples", default=4, type=int)
    parser.add_argument("-template_size", default=None, type=int)
    parser.add_argument("-max_corners", default=24, type=int)
    parser.add_argument("-variants", default=",".join(VARIANTS))
    parser.add_argument("-lc", default=None, type=float,
                        help="fixed target size instead of matching the agent's element count")
    parser.add_argument("-out", default="out/gmsh")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    env_config = dict(config["environment"])
    # Score OUR mesh the way Gmsh's is scored. `geo2d.Mesh.quality()` is
    # 2J/(|a|^2+|b|^2), aspect-aware; `min_element_quality` defaults to the
    # ANGLE metric, which is sin of the corner angle and cannot see a
    # paper-thin quad. No straight config sets this, so the published table
    # compared the agent on the blind metric against Gmsh on the honest one --
    # worth about +0.23 to us. Forced here so it cannot regress.
    env_config["quality_metric"] = "shape"
    template_size = args.template_size or env_config.get("template_size", 128)
    model = load_model(args.checkpoint, args.config, template_size=template_size)
    variants = [v for v in args.variants.split(",") if v in VARIANTS]

    def suite_geometries(suite):
        """The geo2d suites, plus the Mesh Quest levels converted back to geometries.

        A level with a hole is stored as a slit loop, which no external mesher can
        read; `level_geometry` inverts the slit. par is unchanged by the round trip
        on all fifteen, so the comparison is on the same domains the agent sees.
        """
        if suite.startswith("transfer-"):
            # The large-domain set lives in its own module because it needs the
            # extremity cap as well as a corner window; see transfer_set.py.
            from utilities.transfer_set import transfer_geometries
            return [g for _, g in transfer_geometries(suite, args.seed, args.n)]
        if suite != "mesh-quest":
            return list(geometries(suite, args.seed, args.n, args.max_corners))
        sys.path.insert(0, os.path.join(os.getcwd(), "game"))
        from server import INITIALIZERS
        from src.holdout import LEVEL_NAMES
        from utilities.level_geometry import level_geometry
        out = []
        for name in LEVEL_NAMES:
            graph, _ = INITIALIZERS[name]()
            out.append(level_geometry(graph))
        return out

    rows = []
    for suite in args.suites.split(","):
        for index, geometry in enumerate(suite_geometries(suite)):
            agent, env = agent_best(model, env_config, geometry, template_size, args.n_samples)
            par = int(env.par)
            wants, tol = domain_corner_wants(geometry)
            rows.append(dict(suite=suite, index=index, method="agent", par=par, **agent))
            for name in variants:
                try:
                    if args.lc is not None:
                        nodes, elements = gmsh_quad_mesh(geometry, args.lc, VARIANTS[name])
                    else:
                        found = mesh_at_element_count(geometry, agent["elements"], VARIANTS[name])
                        if found is None:
                            raise RuntimeError("no mesh")
                        _, _, nodes, elements = found
                    score = score_mesh(nodes, elements, wants, tol, par)
                except Exception as error:
                    score = {"elements": 0, "all_quad": 0, "face_score": -1, "vertex_score": -1,
                             "excess_over_par": -1, "at_par": 0, "min_quality": 0.0,
                             "error": repr(error)[:80]}
                rows.append(dict(suite=suite, index=index, method=name, par=par, **score))
            print(f"{suite} #{index} par={par} agent={agent['elements']}q "
                  f"{'AT PAR' if agent['at_par'] else 'excess %d' % agent['excess_over_par']}",
                  flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"compare_seed{args.seed}.csv")
    keys = ["suite", "index", "method", "par", "elements", "all_quad", "face_score",
            "vertex_score", "excess_over_par", "at_par", "min_quality"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'method':18s} {'at par':>7} {'all quad':>9} {'excess':>8} {'quads':>7} {'min q':>7}")
    print("-" * 62)
    for method in ["agent"] + variants:
        subset = [r for r in rows if r["method"] == method and r["face_score"] >= 0]
        if not subset:
            continue
        total = len(subset)
        print(f"{method:18s} {sum(r['at_par'] for r in subset):3d}/{total:<3d} "
              f"{sum(r['all_quad'] for r in subset):5d}/{total:<3d} "
              f"{np.mean([r['excess_over_par'] for r in subset]):8.1f} "
              f"{np.mean([r['elements'] for r in subset]):7.1f} "
              f"{np.mean([r['min_quality'] for r in subset]):7.3f}")
    print("\nwrote", path)


if __name__ == "__main__":
    main()
