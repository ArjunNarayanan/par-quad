"""Corner quality takes its ANGLE from the tangent and its LENGTHS from the chord.

On a curved edge the element's side follows the arc, so the direction it leaves
a corner is the arc's tangent. Building the Jacobian from the chord measures an
element nobody has. Aspect ratio stays on the straight-sided patch.

The same rule already governs what a corner WANTS, so the two now agree: a
smooth arc joint reads 180 degrees, wants degree three, and scores as a flat
corner if one face owns both of its sides.
"""

import numpy as np
import pytest

from src.boundary_arcs import Arc, edge_direction
from src.tiler import Tiler


def _unit_square():
    coordinates = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0]),
                   2: np.array([1.0, 1.0]), 3: np.array([0.0, 1.0])}
    return Tiler.from_face_loops([[0, 1, 2, 3]], coordinates)


def test_a_straight_domain_is_bit_for_bit_unchanged():
    """No arcs means the tangent IS the chord, so nothing may move."""
    graph = _unit_square()
    assert graph.boundary_arcs is None or not len(list(graph.boundary_arcs.items()))
    values = graph.corner_shape_qualities()
    assert all(abs(v - 1.0) < 1e-12 for v in values.values())


def test_the_tangent_changes_the_angle_on_an_arc():
    """A square whose bottom is a bulging arc: the chord reading says perfect."""
    graph = _unit_square()
    centre = np.array([0.5, -1.0])
    radius = float(np.linalg.norm(np.array([0.0, 0.0]) - centre))
    start = np.arctan2(0 - centre[1], 0 - centre[0])
    end = np.arctan2(0 - centre[1], 1 - centre[0])
    graph.boundary_arcs.add(0, 1, Arc(centre, radius, start, end, ccw=False),
                            start_vertex=0)
    values = graph.corner_shape_qualities()
    worst = min(values.values())
    # the two corners on the arc are no longer square: the tangent leaves at an
    # angle to the vertical side, so the corner is worse than 1
    assert worst < 1.0 - 1e-6
    # ... and the two corners OFF the arc are untouched
    assert max(values.values()) == pytest.approx(1.0, abs=1e-12)


def test_a_face_owning_both_sides_of_a_smooth_joint_has_a_flat_corner():
    """The case the chord reading called fine.

    Two quarter arcs of one circle meeting at the top of a square. In tangent
    terms that joint is 180 degrees -- it is a point ON a smooth curve, which is
    why it wants degree three -- so a face holding both sides has a flat corner
    and must score about zero, where the chord through it subtends 90 degrees
    and scores well.
    """
    coordinates = {0: np.array([0.0, -1.0]), 1: np.array([1.0, 0.0]),
                   2: np.array([0.0, 1.0]), 3: np.array([-1.0, 0.0])}
    graph = Tiler.from_face_loops([[0, 1, 2, 3]], coordinates)
    chord_only = min(graph.corner_shape_qualities().values())
    graph.boundary_arcs.add(1, 2, Arc((0, 0), 1.0, 0.0, np.pi / 2, ccw=True),
                            start_vertex=1)
    graph.boundary_arcs.add(2, 3, Arc((0, 0), 1.0, np.pi / 2, np.pi, ccw=True),
                            start_vertex=2)
    with_arcs = graph.corner_shape_qualities()
    # vertex 2 is the smooth joint; its corner is the one that must collapse
    joint = [h for h in graph.half_edge_list()
             if graph.source_vertex(h, tag=False) == 2]
    assert chord_only > 0.5
    for half_edge in joint:
        assert abs(with_arcs[half_edge]) < 1e-9


def test_quality_and_wants_now_read_the_same_tangent():
    """One rule, `edge_direction`, behind both -- that is the point of the fix."""
    graph = _unit_square()
    centre = np.array([0.5, -1.0])
    radius = float(np.linalg.norm(np.array([0.0, 0.0]) - centre))
    start = np.arctan2(0 - centre[1], 0 - centre[0])
    end = np.arctan2(0 - centre[1], 1 - centre[0])
    graph.boundary_arcs.add(0, 1, Arc(centre, radius, start, end, ccw=False),
                            start_vertex=0)
    # the direction quality now uses at vertex 0 along the arc
    tangent = np.asarray(edge_direction(graph, 0, 1), dtype=float)
    tangent /= np.linalg.norm(tangent)
    chord = np.array([1.0, 0.0])
    assert not np.allclose(tangent, chord, atol=1e-3)
    # and the corner built from it is the one the metric reports
    corner = [h for h in graph.half_edge_list()
              if graph.source_vertex(h, tag=False) == 0
              and graph.target_vertex(h, tag=False) == 1][0]
    behind = graph.source_vertex(graph.previous_half_edge(corner), tag=False)
    b = np.asarray(graph.vertex_coordinate(behind), float) - np.array([0.0, 0.0])
    a = tangent * 1.0                       # chord length along the arc edge is 1
    expected = 2.0 * (a[0] * b[1] - a[1] * b[0]) / (a @ a + b @ b)
    assert graph.corner_shape_qualities()[corner] == pytest.approx(expected, abs=1e-12)
