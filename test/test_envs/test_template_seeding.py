"""The template window should be centred on a FACE, not on one of its half-edges.

Breadth-first from a single half-edge reaches `twin(h)` -- a half-edge of the
neighbouring face -- at depth one, while the far side of h's own face waits
until depth two. So the window leaks across h before closing the face it is
supposed to be centred on, and which way it leaks depends on which half-edge of
that face happened to be picked as the centre. Seeding the walk with the whole
face loop removes the choice.

The effect is a small-k one: by the time the window holds a few rings the
frontier has evened out by itself, which is why this was invisible at the
template sizes the experiments use. It is OFF by default for that reason -- it
changes which half-edges a truncated window holds, so it is a change to the
observation and belongs in an ablation, not beside other changes.
"""

import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from src.tiler import Tiler  # noqa: E402


def _grid(n):
    """An n x n patch of unit quads."""
    coordinates, loops = {}, []
    for i in range(n + 1):
        for j in range(n + 1):
            coordinates[i * (n + 1) + j] = [float(j), float(i)]
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            loops.append([a, a + 1, a + n + 2, a + n + 1])
    return Tiler.from_face_loops(loops, coordinates)


def _centre_half_edges(graph, n):
    face = graph.face_list()[(n // 2) * n + n // 2]
    return [h for h in graph.half_edge_list() if graph.face(h) == face], face


def test_the_centre_face_is_in_the_window_before_anything_else():
    graph = _grid(5)
    half_edges, _ = _centre_half_edges(graph, 5)
    face = set(half_edges)
    window = graph.knn_half_edges(half_edges[0], 4, seed_face=True)
    assert face <= set(window), "the four half-edges of the centre face come first"


def test_seeding_from_one_half_edge_does_not():
    """The behaviour this replaces; if this ever passes, the fix is redundant."""
    graph = _grid(5)
    half_edges, _ = _centre_half_edges(graph, 5)
    window = graph.knn_half_edges(half_edges[0], 4, seed_face=False)
    assert not set(half_edges) <= set(window)


def test_the_window_is_centred_on_the_face():
    graph = _grid(9)
    half_edges, _ = _centre_half_edges(graph, 9)
    centre = np.mean([graph.vertex_coordinate(graph.source_vertex(h, tag=False))
                      for h in half_edges], axis=0)

    def offset(seed_face):
        window = graph.knn_half_edges(half_edges[0], 16, seed_face=seed_face)
        points = np.array([graph.vertex_coordinate(graph.source_vertex(h, tag=False))
                           for h in window])
        return float(np.linalg.norm(points.mean(axis=0) - centre))

    assert offset(True) < offset(False)
    assert offset(True) < 1e-9, "seeded from the face, the window is symmetric"


def test_a_completed_frontier_gives_the_same_window_from_any_half_edge():
    """Only at a ring boundary, and that is the honest claim.

    Seeding does not make the window independent of which half-edge named the
    face: the seeds are the loop rotated to start there, so a k falling
    mid-frontier still cuts a start-dependent set. What it does fix is the
    boundary case -- and every k, for the centre face and the centroid.
    """
    graph = _grid(7)
    half_edges, _ = _centre_half_edges(graph, 7)
    for k in (4, 8, 16):                      # frontiers that close exactly
        windows = {frozenset(graph.knn_half_edges(h, k, seed_face=True)) for h in half_edges}
        assert len(windows) == 1, f"k={k} should not depend on the half-edge"
        biased = {frozenset(graph.knn_half_edges(h, k, seed_face=False))
                  for h in half_edges}
        assert len(biased) > 1, f"k={k}: the old behaviour was supposed to differ"

    # and mid-frontier it still varies, which this pins so nobody assumes otherwise
    mid = {frozenset(graph.knn_half_edges(h, 20, seed_face=True)) for h in half_edges}
    assert len(mid) > 1


def test_the_centre_is_still_index_zero():
    """Callers index the template; the centre must stay where they expect it."""
    graph = _grid(5)
    half_edges, _ = _centre_half_edges(graph, 5)
    for half_edge in half_edges:
        assert graph.knn_half_edges(half_edge, 12, seed_face=True)[0] == half_edge
        assert graph.knn_half_edges_with_boundary(half_edge, 12, seed_face=True)[0] == half_edge


def test_it_returns_the_same_number_of_half_edges():
    graph = _grid(6)
    half_edges, _ = _centre_half_edges(graph, 6)
    for k in (1, 3, 8, 30, 400):
        window = graph.knn_half_edges(half_edges[0], k, seed_face=True)
        assert len(window) == min(k, len(graph.half_edge_list()))
        assert len(set(window)) == len(window), "no half-edge twice"
