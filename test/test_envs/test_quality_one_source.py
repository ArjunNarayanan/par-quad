"""The gate and the reward must read the SAME quality, from one source.

They diverged once already in this repo's history -- the shortfall sat behind an
`elif` so a solved episode never paid it -- and a metric that judges the agent
differently from the metric that trains it is the same class of bug. These pin
the wiring: both paths reach `Tiler.corner_shape_qualities`, which is where the
tangent correction lives, so a curved domain cannot be trained on one reading
and scored on another.
"""

import numpy as np
import pytest

import envs.polygon_utils as utils
from envs.global_angle_env import AngleEnv
from src.boundary_arcs import Arc, corner_angles
from src.tiler import Tiler


def _curved_env():
    """A square with one arc side, as an env with the shape metric."""
    coordinates = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0]),
                   2: np.array([1.0, 1.0]), 3: np.array([0.0, 1.0])}
    graph = Tiler.from_face_loops([[0, 1, 2, 3]], coordinates)
    centre = np.array([0.5, -1.0])
    radius = float(np.linalg.norm(np.array([0.0, 0.0]) - centre))
    graph.boundary_arcs.add(
        0, 1, Arc(centre, radius,
                  np.arctan2(0 - centre[1], 0 - centre[0]),
                  np.arctan2(0 - centre[1], 1 - centre[0]), ccw=False),
        start_vertex=0)
    angles = corner_angles(graph)
    desired = {v: utils.rounded_desired_degree(angles[v], 90.0) for v in angles}

    class _Fixed:
        n = len(desired)

        def __call__(self):
            from copy import deepcopy
            return deepcopy(graph), dict(desired)

    env = AngleEnv(face_desired_degree=4, graph_initializer=_Fixed(),
                   template_size=32, resample_if_at_par=False,
                   quality_metric="shape", quality_potential_weight=1.0,
                   quality_threshold=0.4)
    env.reset()
    for vertex in graph.vertex_list(tag=False):
        env.graph.set_vertex_coordinate(vertex, graph.vertex_coordinate(vertex))
    env._update_half_edge_angles()
    return env


def test_the_gate_reads_corner_shape_qualities():
    env = _curved_env()
    ideal = float(np.sin(np.radians(env.desired_angle)))
    expected = min(env.graph.corner_shape_qualities().values()) / ideal
    assert env.min_element_quality() == pytest.approx(expected, abs=1e-12)


def test_the_reward_reads_the_same_values():
    env = _curved_env()
    corners = env.graph.corner_shape_qualities()
    for value in env._target_face_qualities():
        assert any(abs(value - v) < 1e-12 for v in corners.values())


def test_gate_and_reward_agree_on_an_all_quad_mesh():
    """Every corner belongs to a quad here, so the two sets coincide."""
    env = _curved_env()
    ideal = float(np.sin(np.radians(env.desired_angle)))
    assert min(env._target_face_qualities()) / ideal == \
        pytest.approx(env.min_element_quality(), abs=1e-12)


def test_the_arc_reaches_both_the_gate_and_the_reward():
    """Strip the arc and both readings must move -- otherwise they are blind.

    Measured on the quantity itself rather than on the shortfall: a shallow arc
    can leave every corner above `quality_threshold`, so the shortfall is zero
    with the arc and zero without it while the underlying quality has moved.
    """
    env = _curved_env()
    gate_with = env.min_element_quality()
    reward_with = min(env._target_face_qualities())
    env.graph.boundary_arcs._arcs.clear()
    env._update_half_edge_angles()
    assert abs(gate_with - env.min_element_quality()) > 1e-9
    assert abs(reward_with - min(env._target_face_qualities())) > 1e-9
    # and the chord reading is the OPTIMISTIC one, which is the whole point
    assert env.min_element_quality() > gate_with
