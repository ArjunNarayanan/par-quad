"""Gmsh on the curved suites, scored on the same regularity and quality as the agent.

    venv/bin/python utilities/gmsh_curved_regularity.py -suite curved -agent out/surgery/released.curved.s6.json \
        -key surgery -out out/surgery/gmsh.curved.json

For each domain the agent's record names, Gmsh is searched to the agent's element
count (`mesh_at_element_count`) under each variant in `-variants`, and the mesh
is scored with `compare_gmsh.score_mesh`: irregularity is sum |deg - want| with
corner wants from the TANGENT angles (a point on an arc wants 3, like any flat
boundary point), interior want 4; quality is geo2d's shape metric after interior
untangling, recorded both against the arc tangents (`min_quality`) and on chords
(`chord_quality`). They differ where one element owns both boundary edges at a node on a
convex arc: flat against the tangent, 180 degrees less the arc's sweep on the chords. Gmsh
spaces its arc nodes coarsely, so its chord reading is often far better than its tangent one.
"""

import argparse
import json
import os
import statistics
import sys

import numpy as np

sys.path.append(os.getcwd())


def curved_wants(geometry, tol=1e-6):
    import envs.polygon_utils as utils  # noqa: F401
    from utilities.compare_gmsh import corner_wants
    wants = {}
    for loop in geometry.loops:
        for point, angle in zip(np.asarray(loop.points, float), loop.interior_angles()):
            wants[(round(float(point[0]) / tol), round(float(point[1]) / tol))] = corner_wants(float(angle))
    return wants, tol


def _on_arc(point, arc, tol):
    c = np.asarray(arc.center, float)
    r = float(arc.radius)
    v = np.asarray(point, float) - c
    if abs(np.linalg.norm(v) - r) > tol * max(r, 1.0):
        return False
    u0 = np.asarray(arc.p0, float) - c
    u1 = np.asarray(arc.p1, float) - c
    um = np.asarray(arc.point(0.5), float) - c

    def ang(x, y):
        return np.arccos(np.clip(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1))
    # inside the span iff the two part-angles through the midpoint's side add up
    total = ang(u0, um) + ang(um, u1)
    return abs(ang(u0, v) + ang(v, u1) - total) < 1e-6 or \
        (ang(u0, v) <= ang(u0, um) + 1e-9 and ang(v, um) <= ang(u0, um) + 1e-9) or \
        (ang(u1, v) <= ang(u1, um) + 1e-9 and ang(v, um) <= ang(u1, um) + 1e-9)


def arc_tangents(nodes, mesh, geometry, tol=1e-6):
    """{(node, neighbour): unit tangent} for every mesh boundary edge lying on an arc."""
    arcs = [loop.edge(i) for loop in geometry.loops for i, a in enumerate(loop.is_arc()) if a]
    table = {}
    for a, b in mesh.boundary_edges():
        a, b = int(a), int(b)
        pa, pb = nodes[a], nodes[b]
        for arc in arcs:
            if not (_on_arc(pa, arc, tol * 1e3) and _on_arc(pb, arc, tol * 1e3)):
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


def tangent_quality(nodes, elements, geometry, untangle=True):
    """(min tangent-aware shape quality, min chord quality) after interior untangling."""
    from geo2d.smooth import _corner_directions, _corner_terms

    from src.geo2d_bridge import import_geo2d, resmooth_mesh
    geo2d = import_geo2d()
    mesh = geo2d.Mesh(np.asarray(nodes, float), elements)
    table = arc_tangents(mesh.nodes, mesh, geometry)
    if untangle:
        try:
            mesh = resmooth_mesh(mesh, mesh.is_boundary(), iters=4, method="optimize",
                                 slide=False, tangents=table or None)
        except Exception:
            pass
    c = mesh.corners()
    tri = np.column_stack([c["prev"], c["node"], c["next"]])
    sn = np.sin(np.pi * (c["n"] - 2) / c["n"])
    J, L2 = _corner_terms(mesh.nodes[tri], _corner_directions(tri, table))
    return float(np.min(2 * J / L2 / sn)), float(np.min(mesh.quality())), len(table)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-suite", required=True)
    p.add_argument("-agent", required=True)
    p.add_argument("-key", default="surgery")
    p.add_argument("-variants", default="blossom,blossom-full,frontal-quad,packing,quasi-structured")
    p.add_argument("-scales", default="1,1.5,2,3,4,6", help="multiples of the agent's element count tried in order")
    p.add_argument("-tries", default=8, type=int, help="binary-search steps per element-count target")
    p.add_argument("-seed", default=7, type=int)
    p.add_argument("-bar", default=0.3, type=float)
    p.add_argument("-out", default=None)
    args = p.parse_args()

    from utilities.compare_gmsh import VARIANTS, mesh_at_element_count, score_mesh
    from utilities.curved_surgery_eval import suite_domains

    scales = [float(x) for x in args.scales.split(",")]
    agent = json.load(open(args.agent))
    recs = {r["draw"]: r for r in agent["records"]}
    n = agent["args"]["n"]
    # densify as the agent was scored: the transfer suites filter on the densified edge ratio
    domains = suite_domains(args.suite, n, args.seed, densify=agent["args"].get("densify", 45.0),
                            with_geometry=True)
    rows = []
    for draw, graph, desired, geometry in domains:
        rec = recs.get(draw)
        if rec is None or rec.get(args.key) is None:
            continue
        target = rec[args.key]["faces"]
        wants, tol = curved_wants(geometry)
        row = dict(draw=draw, par=rec["par"], target=target,
                   agent=dict(irr=rec[args.key]["irr"], vertices=rec[args.key]["vertices"],
                              q=rec[args.key]["q"]))
        for name in args.variants.split(","):
            # each variant "as is": the COARSEST mesh it can make that is all-quad
            # and clears the bar, searching up from the agent's element count
            chosen = None
            for scale in scales:
                found = mesh_at_element_count(geometry, max(int(round(target * scale)), 1),
                                              VARIANTS[name], tries=args.tries)
                if found is None:
                    continue
                count, _, nodes, elements = found
                sc = score_mesh(nodes, elements, wants, tol, rec["par"], untangle=False)
                sc["min_quality"], sc["chord_quality"], sc["arc_edges"] = tangent_quality(nodes, elements, geometry)
                sc["vertices"] = len({int(v) for e in elements for v in e})
                sc["scale"] = scale
                ok = bool(sc["all_quad"]) and sc["min_quality"] >= args.bar
                key = (not sc["all_quad"], -sc["min_quality"])
                if chosen is None or ok or key < chosen[0]:
                    chosen = (key, sc)
                if ok:
                    break
            row[name] = None if chosen is None else chosen[1]
        rows.append(row)
        print(f"draw {draw:3d} agent {target:3d}e {row['agent']['irr']:3d}/{row['agent']['vertices']:<3d} "
              f"q {row['agent']['q']:+.2f}  " + "  ".join(
                  f"{k} " + ("--" if row[k] is None else
                             f"{row[k]['elements']}e quad {row[k]['all_quad']} "
                             f"{row[k]['vertex_score']}/{row[k]['vertices']} q {row[k]['min_quality']:+.2f}")
                  for k in args.variants.split(",")), flush=True)
    summary = {}
    for name in args.variants.split(","):
        sel = [(r, r[name]) for r in rows if r[name] is not None]
        usable = [(r, x) for r, x in sel if x["all_quad"] and x["min_quality"] >= args.bar]
        ipv = [x["vertex_score"] / x["vertices"] for _, x in usable]
        ratio = [x["elements"] / r["target"] for r, x in usable]
        summary[name] = dict(n=len(rows), usable=len(usable),
                             all_quad=sum(bool(x["all_quad"]) for _, x in sel),
                             irr_pv_median=statistics.median(ipv) if ipv else float("nan"),
                             irr_pv_mean=statistics.mean(ipv) if ipv else float("nan"),
                             elements_median=statistics.median(x["elements"] for _, x in usable) if usable else float("nan"),
                             element_ratio_median=statistics.median(ratio) if ratio else float("nan"),
                             q_median=statistics.median(x["min_quality"] for _, x in usable) if usable else float("nan"))
    ag = [r["agent"] for r in rows]
    summary["agent"] = dict(n=len(rows), usable=sum(x["q"] >= args.bar for x in ag),
                            irr_pv_median=statistics.median(x["irr"] / x["vertices"] for x in ag),
                            elements_median=statistics.median(r["target"] for r in rows))
    print(f"=== gmsh on {args.suite}: coarsest usable mesh per variant (tangent quality >= {args.bar}), "
          f"searched from 1x to {scales[-1]}x the agent's count ===")
    for k, s in summary.items():
        print(f"  {k:17} usable {s['usable']}/{s['n']}  irr/v med {s['irr_pv_median']:.3f}"
              f"  elems med {s['elements_median']}"
              + (f" ({s['element_ratio_median']:.1f}x agent)  q med {s['q_median']:+.2f}" if k != "agent" else ""))
    if args.out:
        json.dump(dict(args=vars(args), summary=summary, rows=rows), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
