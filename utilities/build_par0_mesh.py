"""Build a domain's par-0 mesh directly, and report whether it is realizable.

    venv/bin/python utilities/build_par0_mesh.py -level Hook

`par0_structure.py` says what a zero-score mesh must be: a rectilinear polygon
whose side lengths are fixed by the wants. That fixes the whole mesh up to where
the vertices sit, so it can be built outright rather than searched for --
boundary vertices spaced along the real outline, interior vertices smoothed into
place, and the quality measured.

This exists because guessing was wrong. A large forced edge-length ratio looked
like it implied a degenerate mesh, and it does for Hook (-0.71) and does NOT for
Bridge (+0.14) which has a worse ratio. The only way to know is to build it.

Reports the quality the EVALUATOR would see, so "gate accepts" here means the
agent solving this domain would have its answer accepted.
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from utilities.par0_structure import DIRECTIONS, sides_of, solve  # noqa: E402


def boundary_positions(graph, frm, to, count, n):
    """`count + 1` points spaced by arc length along the outline from `frm` to `to`."""
    pieces = []
    i = frm
    while i != to:
        j = (i + 1) % n
        arc = graph.boundary_arcs.get(i, j)
        if arc is None:
            a = np.asarray(graph.vertex_coordinates[i], float)
            b = np.asarray(graph.vertex_coordinates[j], float)
            pieces.append((float(np.linalg.norm(b - a)),
                           lambda t, a=a, b=b: a + t * (b - a)))
        else:
            start = arc if np.allclose(arc.point(0.0), graph.vertex_coordinates[i],
                                       atol=1e-9) else None
            length = abs(arc.radius * arc._sweep())
            forward = start is not None
            pieces.append((length,
                           lambda t, arc=arc, f=forward: arc.point(t if f else 1.0 - t)))
        i = j
    total = sum(p[0] for p in pieces)
    out = []
    for step in range(count + 1):
        target = total * step / count
        walked = 0.0
        for length, at in pieces:
            if walked + length >= target - 1e-12:
                out.append(np.asarray(at(min(1.0, max(0.0, (target - walked) / length))), float))
                break
            walked += length
        else:
            out.append(np.asarray(pieces[-1][1](1.0), float))
    return out


def inside(polygon, point):
    x, y = point
    hit = False
    for k in range(len(polygon)):
        x1, y1 = polygon[k]
        x2, y2 = polygon[(k + 1) % len(polygon)]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) / (y2 - y1) * (x2 - x1):
                hit = not hit
    return hit


def build(level, seed=None, preset="rounded"):
    from src.tiler import Tiler

    if seed is None:
        from src.curved_levels import from_spec, load_domains
        graph, wants = from_spec(load_domains()[level])
    else:
        from src.geo2d_bridge import curved_geometry_to_tiler, import_geo2d
        geo2d = import_geo2d()
        graph, wants = curved_geometry_to_tiler(geo2d.generate(seed, preset=preset))
    n = len(wants)
    sides = sides_of([wants[v] for v in sorted(wants)])
    if sides is None:
        return None, "not a par-0 domain"
    found = solve(sides, 8)
    if found is None:
        return None, "no simple closure"
    _, lengths, area = found

    # lattice coordinate of every boundary mesh vertex, and its domain position
    lattice, position, corner = [], [], (0, 0)
    for side, length in zip(sides, lengths):
        dx, dy = DIRECTIONS[side["dir"]]
        pts = boundary_positions(graph, side["from"], side["to"], length, n)
        for step in range(length):
            lattice.append(corner)
            position.append(pts[step])
            corner = (corner[0] + dx, corner[1] + dy)
    index = {c: i for i, c in enumerate(lattice)}
    coordinates = {i: p for i, p in enumerate(position)}
    outline = [tuple(map(float, c)) for c in lattice]

    # every unit cell inside the rectilinear polygon becomes a quad
    xs = [c[0] for c in lattice]; ys = [c[1] for c in lattice]
    cells = [(i, j) for i in range(min(xs), max(xs)) for j in range(min(ys), max(ys))
             if inside(outline, (i + 0.5, j + 0.5))]
    if len(cells) != area:
        return None, f"cell count {len(cells)} != area {area}"

    # interior lattice points get a vertex too, placed by averaging later
    for cell in cells:
        for c in ((cell[0], cell[1]), (cell[0] + 1, cell[1]),
                  (cell[0] + 1, cell[1] + 1), (cell[0], cell[1] + 1)):
            if c not in index:
                index[c] = len(index)
                coordinates[index[c]] = None
    interior = [c for c, i in index.items() if coordinates[i] is None]
    for c in interior:                      # start them on the centroid
        coordinates[index[c]] = np.mean([p for p in position], axis=0)

    loops = [[index[(i, j)], index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]]
             for i, j in cells]
    mesh = Tiler.from_face_loops(loops, {k: np.asarray(v, float)
                                         for k, v in coordinates.items()},
                                 user_vertices=set(range(len(position))))
    return (mesh, len(position), len(interior), area), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-level", default=None)
    parser.add_argument("-seed", default=None, type=int)
    parser.add_argument("-preset", default="rounded")
    args = parser.parse_args()
    built, why = build(args.level, args.seed, args.preset)
    if built is None:
        print(f"{args.level or args.seed}: {why}")
        return
    mesh, boundary, interior, area = built
    import math
    mesh.smooth_vertices(num_iter=400)
    angles = mesh.half_edge_angles()
    quality = min(math.sin(math.radians(a)) for a in angles.values())
    print(f"{args.level or args.seed}: {area} quads, {boundary} boundary vertices, {interior} interior")
    print(f"  min quality after smoothing   {quality:+.4f}")
    try:
        from src.geo2d_bridge import import_geo2d, resmooth_mesh, tiler_faces
        geo2d = import_geo2d()
        nodes, elements, idx = tiler_faces(mesh)
        corners = np.zeros(len(nodes), dtype=bool)
        for vertex, i in idx.items():
            corners[i] = mesh.is_user_defined_vertex(vertex)
        out = resmooth_mesh(geo2d.Mesh(nodes, elements), corners, iters=60,
                            method="optimize", slide="optimize")
        for vertex, i in idx.items():
            mesh.set_vertex_coordinate(vertex, out.nodes[i])
        angles = mesh.half_edge_angles()
        quality = min(math.sin(math.radians(a)) for a in angles.values())
        print(f"  min quality after untangler   {quality:+.4f}")
    except Exception as error:
        print(f"  untangle unavailable: {error}")
    print(f"  gate accepts at 0.1  ->  {'YES' if quality >= 0.1 else 'NO, degenerate'}")


if __name__ == "__main__":
    main()
