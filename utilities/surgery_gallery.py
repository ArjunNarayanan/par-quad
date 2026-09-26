"""The curved gallery from the surgery run: the released straight agent plus surgery, and Gmsh.

    venv/bin/python utilities/surgery_gallery.py -results <dir of released.<suite>.json and meshes/>
        -paper_records <dir of the paper's curved scorer JSONs> -out out/figure-surgery

For each curved suite, picks one domain by the gallery rule of `paper_figure.select` (closest to
the suite's medians of corner count, element count and minimum quality, among domains that have
at least one arc and whose mesh is usable), converts the surgery mesh the run pickled into
`paper_figure`'s format, and meshes the same domain with Gmsh blossom and quasi-structured at the same target count,
untangled and scored on the arc TANGENTS as the agent is (`gmsh_curved_regularity.tangent_quality`,
reproduced per element here). Then `paper_figure.py render -meshes <out>` draws the rows; the
three columns are the other galleries' own.
"""
import argparse
import json
import os
import pickle
import statistics
import sys

import numpy as np

sys.path.append(os.getcwd())

PAPER = {"curved": "mixmask.b6r.curved", "curved-holes": "mixmask.b6r.curved-holes",
         "curved-transfer": "mixmask.b6r.curved-transfer",
         "curved-holes-transfer": "mixmask.b6r.n16.curved-holes-transfer"}


def _on_arc(point, arc, tol):
    c = np.asarray(arc.center, float); r = float(arc.radius)
    v = np.asarray(point, float) - c
    if abs(np.linalg.norm(v) - r) > tol * max(r, 1.0):
        return False
    u0, u1, um = (np.asarray(x, float) - c for x in (arc.p0, arc.p1, arc.point(0.5)))
    ang = lambda x, y: np.arccos(np.clip(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1))
    total = ang(u0, um) + ang(um, u1)
    return abs(ang(u0, v) + ang(v, u1) - total) < 1e-6 or \
        (ang(u0, v) <= ang(u0, um) + 1e-9 and ang(v, um) <= ang(u0, um) + 1e-9) or \
        (ang(u1, v) <= ang(u1, um) + 1e-9 and ang(v, um) <= ang(u1, um) + 1e-9)


def arc_tangents(nodes, mesh, geometry, tol=1e-6):
    arcs = [loop.edge(i) for loop in geometry.loops for i, a in enumerate(loop.is_arc()) if a]
    table = {}
    for a, b in mesh.boundary_edges():
        a, b = int(a), int(b)
        for arc in arcs:
            if not (_on_arc(nodes[a], arc, tol * 1e3) and _on_arc(nodes[b], arc, tol * 1e3)):
                continue
            c = np.asarray(arc.center, float)
            for here, there in ((a, b), (b, a)):
                radial = nodes[here] - c
                t = np.array([-radial[1], radial[0]]) / np.linalg.norm(radial)
                if t @ (nodes[there] - nodes[here]) < 0:
                    t = -t
                table[(here, there)] = t
            break
    return table


def gmsh_tangent(geometry, target, agent, variant="blossom"):
    """Gmsh blossom at `target` elements, interior untangled on the arc tangents, per-element
    tangent-aware quality, in the agent's frame, with wants read off the agent's corners."""
    from geo2d.smooth import _corner_directions, _corner_terms
    from src.geo2d_bridge import import_geo2d, resmooth_mesh
    from utilities.compare_gmsh import VARIANTS, mesh_at_element_count
    from utilities.curved_report import normalizer
    from utilities.paper_figure import _boundary_nodes, _node_degrees
    geo2d = import_geo2d()
    found = mesh_at_element_count(geometry, max(int(target), 1), VARIANTS[variant])
    if found is None:
        return None
    _, _, nodes, elements = found
    elements = [list(map(int, e)) for e in elements]
    mesh = geo2d.Mesh(np.asarray(nodes, float), elements)
    table = arc_tangents(mesh.nodes, mesh, geometry)
    try:
        mesh = resmooth_mesh(mesh, mesh.is_boundary(), iters=4, method="optimize", slide=False,
                             tangents=table or None)
    except Exception:
        pass
    c = mesh.corners()
    tri = np.column_stack([c["prev"], c["node"], c["next"]])
    sn = np.sin(np.pi * (c["n"] - 2) / c["n"])
    J, L2 = _corner_terms(mesh.nodes[tri], _corner_directions(tri, table))
    q = 2 * J / L2 / sn
    face_quality = [float(np.min(q[c["elem"] == e])) for e in range(len(elements))]
    face_quality_chord = [float(v) for v in np.asarray(mesh.quality(), float)]
    centre, scale = normalizer(geometry)
    framed = (np.asarray(mesh.nodes, float) - centre) / scale
    degree, boundary = _node_degrees(elements)
    a_nodes = np.asarray(agent["nodes"], float)
    a_boundary = _boundary_nodes(agent["elements"])
    corners = [(a_nodes[v["node"]], v["wants"]) for v in agent["vertices"]
               if v["node"] in a_boundary and tuple(v["wants"]) != (3,)]
    vertices, matched = [], 0
    for node, deg in degree.items():
        if node in boundary:
            here = framed[node]
            match = [w for p, w in corners if float(np.hypot(*(p - here))) < 1e-4]
            wants = tuple(match[0]) if match else (3,)
            matched += bool(match)
        else:
            wants = (4,)
        vertices.append(dict(node=node, degree=int(deg), wants=tuple(int(w) for w in wants)))
    if matched < len(corners):
        raise RuntimeError(f"only {matched} of {len(corners)} corners matched; frames disagree")
    return dict(nodes=framed, elements=elements, face_quality=face_quality,
                face_quality_chord=face_quality_chord, vertices=vertices, variant=variant)


def convert(result_pkl, record, corners):
    """The surgery mesh as `paper_figure` stores an agent mesh; checked against the record."""
    from src.geo2d_bridge import tiler_faces
    d = pickle.load(open(result_pkl, "rb"))
    graph, wants = d["surgery"], d["wants"]["surgery"]
    nodes, elements, index = tiler_faces(graph)
    cq = graph.corner_shape_qualities()
    face_quality = []
    for face in graph.face_list():
        loop = graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))
        values = [cq[h] for h in loop if h in cq]
        face_quality.append(float(min(values)) if values else float("nan"))
    from src.geo2d_bridge import import_geo2d
    chord = [float(v) for v in np.asarray(import_geo2d().Mesh(np.asarray(nodes, float),
                                                             [list(e) for e in elements]).quality(), float)]
    vertices = [dict(node=i, degree=int(graph.vertex_degree(v)), wants=(int(wants[v]),))
                for v, i in index.items()]
    irr = sum(abs(v["degree"] - v["wants"][0]) for v in vertices)
    assert irr == record["surgery"]["irr"], (result_pkl, irr, record["surgery"]["irr"])
    assert abs(min(face_quality) - record["surgery"]["q"]) < 1e-6, (min(face_quality), record["surgery"]["q"])
    rec = dict(draw=record["draw"], par=record["par"], corners=corners, elements=len(elements),
               irregular=irr, quality=float(min(face_quality)), face=0)
    return dict(suite=None, record=rec, nodes=np.asarray(nodes, float), elements=[list(e) for e in elements],
                face_quality=face_quality, face_quality_chord=chord, vertices=vertices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-results", required=True)
    p.add_argument("-paper_records", required=True)
    p.add_argument("-out", required=True)
    p.add_argument("-bar", default=0.3, type=float)
    args = p.parse_args()
    from utilities.paper_figure import _geometry
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for suite, paper in PAPER.items():
        recs = json.load(open(os.path.join(args.results, f"released.{suite}.json")))["records"]
        true_corners = {d["draw"]: d["corners"] for d in
                        json.load(open(os.path.join(args.paper_records, f"{paper}.json")))["suites"][suite]["domains"]}
        med = [statistics.median(true_corners[r["draw"]] for r in recs),
               statistics.median(r["surgery"]["faces"] for r in recs),
               statistics.median(r["surgery"]["q"] for r in recs)]
        ranked = sorted(recs, key=lambda r: sum(abs(x - m) / max(abs(m), 1e-9) for x, m in
                                                zip((true_corners[r["draw"]], r["surgery"]["faces"], r["surgery"]["q"]), med)))
        for r in ranked:
            if r["surgery"]["q"] < args.bar:
                continue
            geometry, _ = _geometry(suite, r["draw"], 7)
            if not any(any(loop.is_arc()) for loop in geometry.loops):
                continue
            base = os.path.join(args.out, f"{suite}.draw{r['draw']:03d}")
            agent = convert(os.path.join(args.results, "meshes", suite, f"draw{r['draw']:03d}.pkl"),
                            r, true_corners[r["draw"]])
            agent["suite"] = suite
            pickle.dump(agent, open(base + ".pkl", "wb"))
            for variant in ("blossom", "quasi-structured"):
                gm = gmsh_tangent(geometry, r["surgery"]["faces"], agent, variant)
                if gm is not None:
                    gm["par"], gm["corners"] = r["par"], true_corners[r["draw"]]
                pickle.dump(gm, open(f"{base}.{variant}.pkl", "wb"))
            print(f"{suite}: draw {r['draw']} corners {true_corners[r['draw']]} agent {agent['record']['elements']}e "
                  f"irr {agent['record']['irregular']} (par {r['par']}) q {agent['record']['quality']:+.2f} | "
                  + ("gmsh failed" if gm is None else
                     f"gmsh {len(gm['elements'])}e q {min(gm['face_quality']):+.2f}"), flush=True)
            rows.append(f"{suite}:{r['draw']}")
            break
    print("ROWS", ",".join(rows))


if __name__ == "__main__":
    main()
