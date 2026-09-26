"""Snapping to a curved boundary must not undo a legitimate slide.

A vertex where two arcs meet -- a joint, or the midpoint a refinement split --
belongs to both. Projecting onto each in turn and keeping the last clamps a
vertex that has slid along one arc back to the shared endpoint, because it is
outside the other arc's span. The nearest projection is the right one.
"""

import numpy as np

from src.boundary_arcs import Arc
from src.tiler import Tiler


def _quad_on_a_quarter_circle():
    """One quad whose outer side is two quarter-arcs of the unit circle."""
    coordinates = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0]),
                   2: np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)]),
                   3: np.array([0.0, 1.0])}
    graph = Tiler.from_face_loops([[0, 1, 2, 3]], coordinates)
    graph.boundary_arcs.add(1, 2, Arc((0, 0), 1.0, 0.0, np.pi / 4, ccw=True),
                            start_vertex=1)
    graph.boundary_arcs.add(2, 3, Arc((0, 0), 1.0, np.pi / 4, np.pi / 2, ccw=True),
                            start_vertex=2)
    return graph


def test_snap_leaves_a_vertex_that_is_already_on_one_of_its_arcs():
    graph = _quad_on_a_quarter_circle()
    # slide vertex 2 along its first arc, to 30 degrees: still exactly on the
    # boundary, but now outside the span of its SECOND arc (45 to 90)
    angle = np.pi / 6
    slid = np.array([np.cos(angle), np.sin(angle)])
    graph.set_vertex_coordinate(2, slid)
    graph.snap_to_boundary_arcs()
    assert np.allclose(graph.vertex_coordinate(2), slid, atol=1e-12)


def test_snap_still_pulls_a_vertex_back_onto_the_curve():
    graph = _quad_on_a_quarter_circle()
    angle = np.pi / 6
    off = 0.8 * np.array([np.cos(angle), np.sin(angle)])   # inside the circle
    graph.set_vertex_coordinate(2, off)
    graph.snap_to_boundary_arcs()
    point = np.asarray(graph.vertex_coordinate(2), dtype=float)
    assert abs(float(np.linalg.norm(point)) - 1.0) < 1e-12
    assert np.allclose(point, [np.cos(angle), np.sin(angle)], atol=1e-12)
