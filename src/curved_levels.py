"""Named curved domains, small enough to reason about by hand.

These exist to answer one question directly: can the agent find a correct
decomposition when the boundary is genuinely curved? A semicircle is the
smallest domain where that question is not trivial -- three vertices, one
straight edge, two arcs, and par 2, so a perfectly regular mesh is IMPOSSIBLE
and the agent has to place two units of irregularity in the right place.

The tangent angles are what make it work and are worth spelling out. At either
end of the diameter the circle's tangent is perpendicular to it, so the material
turns through 90 degrees and that corner wants two. At the apex the two arcs
join smoothly, so the angle is 180 and it wants three, exactly like a flat point
on a straight edge. Nothing here is discretised.
"""

import json
import os

import numpy as np

import envs.polygon_utils as utils
from src.boundary_arcs import Arc, corner_angles
from src.tiler import Tiler


def _finish(loop, coordinates, arcs, target_angle):
    graph = Tiler.from_face_loops([loop], {k: v.copy() for k, v in coordinates.items()})
    for (a, b), (arc, start) in arcs.items():
        graph.boundary_arcs.add(a, b, arc, start_vertex=start)
    angles = corner_angles(graph)
    desired = {v: utils.rounded_desired_degree(angles[v], target_angle) for v in loop}
    return graph, desired


def semicircle(target_angle=90.0):
    """Half a disc: one straight diameter, two quarter arcs. par 2."""
    coordinates = {0: np.array([-1.0, 0.0]), 1: np.array([1.0, 0.0]),
                   2: np.array([0.0, 1.0])}
    loop = [0, 1, 2]
    arcs = {(1, 2): (Arc((0, 0), 1.0, 0.0, np.pi / 2, ccw=True), 1),
            (2, 0): (Arc((0, 0), 1.0, np.pi / 2, np.pi, ccw=True), 2)}
    return _finish(loop, coordinates, arcs, target_angle)


def quarter_disc(target_angle=90.0):
    """A quarter of a disc: two straight radii meeting at 90, one arc."""
    coordinates = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0]),
                   2: np.array([0.0, 1.0])}
    loop = [0, 1, 2]
    arcs = {(1, 2): (Arc((0, 0), 1.0, 0.0, np.pi / 2, ccw=True), 1)}
    return _finish(loop, coordinates, arcs, target_angle)


def stadium(target_angle=90.0):
    """A rectangle capped with semicircles -- the classic slot."""
    coordinates = {0: np.array([-1.0, -1.0]), 1: np.array([1.0, -1.0]),
                   2: np.array([2.0, 0.0]), 3: np.array([1.0, 1.0]),
                   4: np.array([-1.0, 1.0]), 5: np.array([-2.0, 0.0])}
    loop = [0, 1, 2, 3, 4, 5]
    arcs = {
        (1, 2): (Arc((1, 0), 1.0, -np.pi / 2, 0.0, ccw=True), 1),
        (2, 3): (Arc((1, 0), 1.0, 0.0, np.pi / 2, ccw=True), 2),
        (4, 5): (Arc((-1, 0), 1.0, np.pi / 2, np.pi, ccw=True), 4),
        (5, 0): (Arc((-1, 0), 1.0, np.pi, 3 * np.pi / 2, ccw=True), 5),
    }
    return _finish(loop, coordinates, arcs, target_angle)


def rounded_square(target_angle=90.0, radius=0.4):
    """A square with all four corners filleted -- the shape bending cannot make."""
    half = 1.0
    inner = half - radius
    coordinates, loop, arcs = {}, [], {}
    corners = [(inner, -half, inner, -inner, -np.pi / 2, 0.0),
               (half, inner, inner, inner, 0.0, np.pi / 2),
               (-inner, half, -inner, inner, np.pi / 2, np.pi),
               (-half, -inner, -inner, -inner, np.pi, 3 * np.pi / 2)]
    index = 0
    for x0, y0, cx, cy, start, end in corners:
        a = index
        coordinates[a] = np.array([x0, y0])
        b = index + 1
        coordinates[b] = np.array([cx + radius * np.cos(end),
                                   cy + radius * np.sin(end)])
        loop += [a, b]
        arcs[(a, b)] = (Arc((cx, cy), radius, start, end, ccw=True), a)
        index += 2
    return _finish(loop, coordinates, arcs, target_angle)


# --------------------------------------------------------------- frozen domains

_DOMAINS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "src", "curved_domains.json")


def load_domains():
    """The curved domains frozen out of geo2d, as plain numbers.

    `game/js/curved_domains.js` is generated from this same file; keeping one
    source is the only thing that stops a level meaning two different shapes.
    """
    if not os.path.exists(_DOMAINS_PATH):
        return {}
    with open(_DOMAINS_PATH) as handle:
        return json.load(handle)


def circle_arcs(centre, radius, count=4, phase=0.0):
    """`count` control points on a circle and the arcs between them, CCW.

    A circle needs at least two control points to be a loop at all, and four is
    what geo2d uses: the quarter arcs stay well-conditioned and every point sits
    on an axis. Each point is a smooth joint, so each wants degree three.
    """
    angles = [phase + 2 * np.pi * k / count for k in range(count)]
    points = [np.array([centre[0] + radius * np.cos(a),
                        centre[1] + radius * np.sin(a)]) for a in angles]
    arcs = {}
    for k in range(count):
        arcs[(k, (k + 1) % count)] = (
            Arc(centre, radius, angles[k], angles[(k + 1) % count], ccw=True), k)
    return np.array(points), arcs


def plate_with_hole(width=2.4, height=1.4, radius=0.55, target_angle=90.0):
    """The classic: a rectangular plate with one circular hole.

    par is 4 and the reason is worth seeing on the board. The plate's own four
    corners want two and supply the whole Gauss-Bonnet budget on their own; the
    hole is smooth everywhere, so its four control points want three and supply
    nothing. Four defects therefore have to sit somewhere, and the O-grid puts
    one at each control point -- which is exactly the mesh a human draws for
    this part without being told to.
    """
    from src.geo2d_bridge import slit_outline

    outer = np.array([[-width, -height], [width, -height],
                      [width, height], [-width, height]], dtype=float)
    hole, hole_arcs = circle_arcs((0.0, 0.0), radius, count=4)
    coordinates, loop, angles, arcs = slit_outline(
        outer, [hole], target_angle,
        outer_angles=[90.0] * 4, hole_angles=[[180.0] * 4],
        outer_arcs={}, hole_arcs=[hole_arcs])
    graph = Tiler.from_face_loops([loop], {k: np.asarray(v, dtype=float)
                                           for k, v in coordinates.items()})
    for (first, second), (arc, start) in arcs.items():
        graph.boundary_arcs.add(first, second, arc, start_vertex=start)
    desired = {v: utils.rounded_desired_degree(a, target_angle)
               for v, a in angles.items()}
    return graph, desired


def from_spec(spec, target_angle=90.0):
    """Build a `(Tiler, desired)` from one frozen domain.

    The wants are recomputed from the tangents rather than read from the file:
    a level records a SHAPE, and what its corners want is a property of the
    shape plus the meshing target, not a number to be stored and go stale.
    """
    coordinates = {index: np.asarray(point, dtype=float)
                   for index, point in enumerate(spec["points"])}
    # a domain with a hole is cut open by a slit, so its face loop revisits the
    # two slit endpoints and is not just range(n)
    loop = list(spec.get("loop") or range(len(coordinates)))
    arcs = {}
    for entry in spec["arcs"]:
        first, second = entry["edge"]
        arcs[(first, second)] = (Arc(entry["centre"], entry["radius"],
                                     entry["from"], entry["to"], ccw=entry["ccw"]),
                                 entry["start"])
    return _finish(loop, coordinates, arcs, target_angle)


def make_domain(name):
    """A LEVELS-style builder for one frozen domain."""
    def build(target_angle=90.0):
        return from_spec(load_domains()[name], target_angle)
    return build


LEVELS = {
    "plate-with-hole": plate_with_hole,
    "semicircle": semicircle,
    "quarter-disc": quarter_disc,
    "stadium": stadium,
    "rounded-square": rounded_square,
}


def make_initializer(name_of_level="semicircle", target_angle=90.0, **_):
    """Factory for the `Custom` initializer hook: one fixed curved domain."""
    from copy import deepcopy

    graph, desired = LEVELS[name_of_level](target_angle)

    class _Fixed:
        def __init__(self):
            self.n = len(desired)

        def __call__(self):
            return deepcopy(graph), dict(desired)

    return _Fixed()
