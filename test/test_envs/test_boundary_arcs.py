"""Curved boundaries carried natively, not discretised.

A curve is not a different problem: topology never asks what shape an edge is,
and the corner want at a vertex comes from the TANGENT angle there, which geo2d
already reports. Discretising would invent corners that are an artefact of the
sampling and make par depend on how finely the curve was sampled -- the opposite
of a bound. So only two operations need to know about arcs: inserting a vertex
on one, and smoothing a vertex that sits on one.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.getcwd())

from src.boundary_arcs import Arc, BoundaryArcs  # noqa: E402
from src.tiler import Tiler  # noqa: E402


def _quarter_circle():
    """A quarter arc of the unit circle, centre at the origin, from +x to +y."""
    return Arc(centre=(0.0, 0.0), radius=1.0, start=0.0, end=np.pi / 2, ccw=True)


def test_the_arc_evaluates_where_geo2d_does():
    geo2d = pytest.importorskip("geo2d")
    loop = geo2d.generate(3, preset="rounded").outer
    index = int(np.flatnonzero(loop.is_arc())[0])
    edge = loop.edge(index)
    arc = Arc.from_geo2d(edge)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        np.testing.assert_allclose(arc.point(t), np.asarray(edge.point(t)), atol=1e-9)


def test_a_split_reproduces_the_original_arc():
    arc = _quarter_circle()
    first, second = arc.split(0.5)
    np.testing.assert_allclose(first.point(0.0), arc.point(0.0), atol=1e-12)
    np.testing.assert_allclose(second.point(1.0), arc.point(1.0), atol=1e-12)
    np.testing.assert_allclose(first.point(1.0), second.point(0.0), atol=1e-12)
    for t in (0.2, 0.6, 0.9):
        assert abs(np.linalg.norm(first.point(t) - arc.centre) - arc.radius) < 1e-12


def test_projection_lands_on_the_circle_and_within_the_span():
    arc = _quarter_circle()
    far = arc.point(0.5) + np.array([0.4, 0.4])
    projected = arc.project(far)
    assert abs(np.linalg.norm(projected - arc.centre) - arc.radius) < 1e-12
    # a point past the end clamps to the end rather than wrapping round
    beyond = np.array([-1.0, -1.0])
    clamped = arc.project(beyond)
    assert abs(np.linalg.norm(clamped - arc.centre) - arc.radius) < 1e-12
    fractions = [np.linalg.norm(clamped - arc.point(0.0)),
                 np.linalg.norm(clamped - arc.point(1.0))]
    assert min(fractions) < 1e-9, "a point outside the span should clamp to an end"


def test_the_split_hands_each_half_to_the_right_pair():
    arc = _quarter_circle()
    arcs = BoundaryArcs()
    arcs.add(0, 1, arc, start_vertex=0)
    point = arcs.split(0, 1, middle=7)
    np.testing.assert_allclose(point, arc.point(0.5), atol=1e-12)
    assert arcs.get(0, 1) is None, "the original edge is gone"
    np.testing.assert_allclose(arcs.get(0, 7).point(0.0), arc.point(0.0), atol=1e-12)
    np.testing.assert_allclose(arcs.get(7, 1).point(1.0), arc.point(1.0), atol=1e-12)


def test_the_split_respects_which_end_the_arc_runs_from():
    """Registered against the pair in the other order, the halves must still match."""
    arc = _quarter_circle()
    arcs = BoundaryArcs()
    arcs.add(0, 1, arc, start_vertex=1)          # the arc runs 1 -> 0
    arcs.split(0, 1, middle=7)
    np.testing.assert_allclose(arcs.get(1, 7).point(0.0), arc.point(0.0), atol=1e-12)
    np.testing.assert_allclose(arcs.get(7, 0).point(1.0), arc.point(1.0), atol=1e-12)


def _square_with_one_arc():
    coordinates = {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [1.0, 1.0], 3: [0.0, 1.0]}
    graph = Tiler.from_face_loops([[0, 1, 2, 3]], coordinates)
    # bulge the edge 1->2 out into a quarter arc centred at (0, 1)... any arc
    # through both endpoints will do; centre at (0,0) with radius sqrt(2) does
    arc = Arc(centre=(0.0, 0.0), radius=float(np.hypot(1.0, 1.0)),
              start=float(np.arctan2(0.0, 1.0)), end=float(np.arctan2(1.0, 1.0)),
              ccw=True)
    return graph, arc


def test_a_straight_domain_is_untouched():
    """boundary_arcs is empty by default and every path is a no-op."""
    graph, _ = _square_with_one_arc()
    assert not graph.boundary_arcs
    half_edge = graph.half_edge_list()[0]
    source = graph.source_vertex(half_edge, tag=False)
    target = graph.target_vertex(half_edge, tag=False)
    expected = 0.5 * (graph.vertex_coordinate(source) + graph.vertex_coordinate(target))
    np.testing.assert_allclose(
        graph.get_new_vertex_coordinate(target, source), expected, atol=1e-12)


def test_inserting_on_an_arc_lands_on_the_arc():
    graph, arc = _square_with_one_arc()
    # find the boundary half-edge joining 1 and 2 and register the arc on it
    graph.boundary_arcs.add(1, 2, arc, start_vertex=1)
    target = None
    for half_edge in graph.half_edge_list():
        pair = {graph.source_vertex(half_edge, tag=False),
                graph.target_vertex(half_edge, tag=False)}
        if pair == {1, 2} and graph.half_edge_on_boundary(half_edge):
            target = half_edge
            break
    assert target is not None
    graph.insert_vertex(target)

    new = [v for v in graph.vertex_list(tag=False) if v not in (0, 1, 2, 3)]
    assert len(new) == 1
    point = np.asarray(graph.vertex_coordinate(new[0]), dtype=float)
    assert abs(np.linalg.norm(point - arc.centre) - arc.radius) < 1e-9, \
        "the new vertex sits on the chord, not the arc"
    assert graph.boundary_arcs.get(1, 2) is None, "the split arc was not replaced"
    assert len(graph.boundary_arcs) == 2


def test_smoothing_puts_boundary_vertices_back_on_their_arc():
    graph, arc = _square_with_one_arc()
    graph.boundary_arcs.add(1, 2, arc, start_vertex=1)
    graph.smooth_vertices(num_iter=5)
    for vertex in (1, 2):
        point = np.asarray(graph.vertex_coordinate(vertex), dtype=float)
        assert abs(np.linalg.norm(point - arc.centre) - arc.radius) < 1e-9


def test_a_chord_between_two_arc_ENDPOINTS_is_not_treated_as_the_arc():
    """The bug this nearly shipped with.

    `boundary_arcs` is keyed by vertex PAIR, and a chord can join the same two
    vertices through the interior. Looking the arc up from the pair alone put an
    INTERIOR vertex out on the boundary curve -- which showed up only after a
    rollout, as vertices drifting 6e-2 off arcs that were exact at construction.
    """
    graph, arc = _square_with_one_arc()
    graph.boundary_arcs.add(1, 3, arc, start_vertex=1)   # pretend 1-3 is an arc
    # 1 and 3 are opposite corners of the square, so the chord between them runs
    # through the interior
    interior = None
    for half_edge in graph.half_edge_list():
        if graph.half_edge_on_boundary(half_edge):
            continue
        pair = {graph.source_vertex(half_edge, tag=False),
                graph.target_vertex(half_edge, tag=False)}
        if pair == {1, 3}:
            interior = half_edge
            break
    if interior is None:                       # no such chord yet: make one
        for half_edge in graph.half_edge_list():
            if graph.source_vertex(half_edge, tag=False) == 1 and \
               graph.is_valid_edge_insert(half_edge, 1):
                graph.insert_half_edge(half_edge, 1)
                break
        for half_edge in graph.half_edge_list():
            if graph.half_edge_on_boundary(half_edge):
                continue
            pair = {graph.source_vertex(half_edge, tag=False),
                    graph.target_vertex(half_edge, tag=False)}
            if pair == {1, 3}:
                interior = half_edge
                break
    assert interior is not None, "fixture failed to produce an interior chord"

    before = set(graph.vertex_list(tag=False))
    graph.insert_vertex(interior)
    new = (set(graph.vertex_list(tag=False)) - before).pop()
    point = np.asarray(graph.vertex_coordinate(new), dtype=float)
    expected = 0.5 * (np.asarray(graph.vertex_coordinate(1), dtype=float)
                      + np.asarray(graph.vertex_coordinate(3), dtype=float))
    np.testing.assert_allclose(point, expected, atol=1e-12,
                               err_msg="an interior vertex was placed on the boundary arc")
