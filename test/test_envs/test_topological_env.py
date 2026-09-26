"""The topological env: counts for features, an evaluator for the quality gate.

The point of the split is that a topology which cannot be drawn well is a
different failure from a mesh that merely needs smoothing. A vertex inserted on
the edge of a triangle gives a single quad with a 180-degree corner: topologically
at par, geometrically impossible, and no smoother repairs it because the flat
corner is forced. A mesh at quality 0.25 is fine and is smoothing's problem.
"""

import os
import sys
from copy import deepcopy

import numpy as np
import pytest

sys.path.append(os.getcwd())

import envs.polygon_utils as utils  # noqa: E402
from envs.environment_initializers import RandomPolygon  # noqa: E402
from envs.topological_env import TopologicalEnv  # noqa: E402
from src.tiler import Tiler  # noqa: E402


class _Fixed:
    """An initializer that hands back the same domain every time."""

    def __init__(self, points):
        self.points = points

    def __call__(self):
        coordinates = {index: list(point) for index, point in enumerate(self.points)}
        loop = list(range(len(self.points)))
        graph = Tiler.from_face_loops([loop], coordinates)
        angles = utils.get_polygon_interior_angles(loop, graph.vertex_coordinates)
        desired = {v: utils.rounded_desired_degree(a, 90) for v, a in angles.items()}
        return deepcopy(graph), dict(desired)


def _triangle_env(**kwargs):
    return TopologicalEnv(
        face_desired_degree=4,
        graph_initializer=_Fixed([(0.0, 0.0), (1.0, 0.0), (0.5, 0.8660254)]),
        template_size=32, max_steps_factor=8.0, resample_if_at_par=False, **kwargs)


def test_features_carry_no_geometry():
    """Seven features, all counts, and none of them moves when the mesh does."""
    env = TopologicalEnv(
        face_desired_degree=4,
        graph_initializer=RandomPolygon(range(8, 12), utils.average_face_angle(4)),
        template_size=32, max_steps_factor=4.0)
    assert TopologicalEnv.get_feature_size() == 7
    observation, _ = env.reset()
    assert observation["features"].shape == (32, 7)

    before = observation["features"].copy()
    # move every vertex: a purely topological observation cannot notice
    for vertex in env.graph.vertex_list(tag=False):
        x, y = env.graph.vertex_coordinate(vertex)
        env.graph.set_vertex_coordinate(vertex, np.array([x * 1.7 + 0.3, y * 0.4 - 0.2]))
    env._update_half_edge_angles()
    after = env._get_feature_matrix()
    np.testing.assert_allclose(before, after, atol=0, rtol=0)


def test_smoothing_is_off_by_default():
    env = _triangle_env()
    assert env.smooth_iterations == 0


def test_a_forced_flat_corner_is_rejected():
    """A vertex on a triangle's edge: at par, and no embedding saves it."""
    env = _triangle_env(degeneracy_threshold=0.2)
    env.reset()
    half_edge = env.graph.half_edge_list()[0]
    env.graph.insert_vertex(half_edge)
    new_vertex = env.graph.target_vertex(half_edge, tag=False)
    env.vertex_desired_degree[new_vertex] = env.boundary_vertex_desired_degree
    env._update_half_edge_angles()
    env._update_scores_on_reset()

    assert env.is_topologically_at_par(), "the single quad should be at par"
    accepted, face = env.evaluate_candidate()
    assert not accepted, "a 180-degree corner must not pass the degeneracy gate"
    assert face is not None, "the evaluator has to say which element it objected to"
    assert not env.is_at_par()
    assert env._rejected is True


def test_rejection_moves_the_template_onto_the_offending_element():
    env = _triangle_env(degeneracy_threshold=0.2)
    env.reset()
    half_edge = env.graph.half_edge_list()[0]
    env.graph.insert_vertex(half_edge)
    env.vertex_desired_degree[env.graph.target_vertex(half_edge, tag=False)] = \
        env.boundary_vertex_desired_degree
    env._update_half_edge_angles()
    env._update_scores_on_reset()
    env.is_at_par()                      # sets the rejection and its face

    centre = env._sticky_center()
    assert centre is not None
    assert env.graph.face(centre) == env._rejected_face


def test_the_rejection_bit_reaches_the_observation():
    """`at par and accepted` and `at par and rejected` must not look identical."""
    env = _triangle_env(degeneracy_threshold=0.2)
    env.reset()
    clean = env._get_global_features().copy()
    env._rejected = True
    marked = env._get_global_features()
    assert not np.allclose(clean, marked), "nothing in the observation records rejection"


def test_a_generous_threshold_still_rejects_a_flat_corner():
    """The gate is about degeneracy, not quality: 0.05 must still say no."""
    env = _triangle_env(degeneracy_threshold=0.05)
    env.reset()
    half_edge = env.graph.half_edge_list()[0]
    env.graph.insert_vertex(half_edge)
    env.vertex_desired_degree[env.graph.target_vertex(half_edge, tag=False)] = \
        env.boundary_vertex_desired_degree
    env._update_half_edge_angles()
    env._update_scores_on_reset()
    accepted, _ = env.evaluate_candidate()
    assert not accepted
