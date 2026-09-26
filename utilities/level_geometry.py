"""Mesh Quest levels as geo2d geometries, so external meshers can be run on them.

The levels are stored as half-edge face loops, and a level with a hole is stored
in SLIT form -- one loop that walks out to the hole along a zero-width cut and
back. That is the representation the agent needs and the one an external mesher
cannot use, so `unslit` inverts it: the hole-side anchor's two visits bracket the
hole walk, and the boundary-side anchor is left doubled once the walk is removed.

The geometry that comes back is the same domain, not an approximation of it, so
par is unchanged -- `test_level_geometry.py` holds it to the environment's own
number for every level.
"""

import collections

import numpy as np


def unslit(loop):
    """`(outer, holes)` from a face loop that may visit slit endpoints twice."""
    outer, holes = list(loop), []
    while True:
        seen = collections.defaultdict(list)
        for index, vertex in enumerate(outer):
            seen[vertex].append(index)
        repeated = [(v, p) for v, p in seen.items() if len(p) > 1]
        if not repeated:
            return outer, holes
        # the innermost repeat is the hole-side anchor; its visits bracket the walk
        _, positions = min(repeated, key=lambda item: item[1][1] - item[1][0])
        first, second = positions[0], positions[1]
        holes.append(outer[first:second])
        outer = outer[:first] + outer[second + 1:]
        outer = [v for i, v in enumerate(outer) if v != outer[i - 1]]


def level_geometry(graph):
    """A `geo2d.Geometry` for a level's `Tiler`, holes separated back out."""
    from src.geo2d_bridge import import_geo2d
    geo2d = import_geo2d()

    face = graph.face_list()[0]
    loop = [graph.source_vertex(h, tag=False)
            for h in graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))]
    coordinates = {v: np.asarray(graph.vertex_coordinate(v), dtype=float)
                   for v in graph.vertex_list(tag=False)}
    outer, holes = unslit(loop)

    def as_loop(vertices, want_ccw):
        points = np.array([coordinates[v] for v in vertices], dtype=float)
        candidate = geo2d.Loop(points)
        if candidate.is_ccw() != want_ccw:
            candidate = geo2d.Loop(points[::-1])
        return candidate

    # geo2d wants the outer loop counter-clockwise and each hole clockwise
    return geo2d.Geometry([as_loop(outer, True)] + [as_loop(h, False) for h in holes])
