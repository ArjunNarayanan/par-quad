"""The clearance term sees a thin element against a curved boundary.

On the chord a thin block scores perfectly well; after refinement onto the
curve it is the element that fails. `clearance_shortfall` is the cheap proxy
for that, and these pin its two ends: zero where there is room, positive where
there is not, and identically zero on a straight domain.
"""

import numpy as np
import pytest

import envs.polygon_utils as utils
from envs.global_angle_env import AngleEnv
from src.boundary_arcs import Arc, corner_angles
from src.tiler import Tiler


def _env_from(graph, **overrides):
    angles = corner_angles(graph)
    desired = {v: utils.rounded_desired_degree(angles[v], 90.0)
               for v in graph.vertex_list(tag=False)}

    class _Fixed:
        n = len(desired)

        def __call__(self):
            from copy import deepcopy
            return deepcopy(graph), dict(desired)

    settings = dict(face_desired_degree=4, graph_initializer=_Fixed(),
                    template_size=64, resample_if_at_par=False,
                    clearance_potential_weight=1.0, clearance_target=4.0)
    settings.update(overrides)
    env = AngleEnv(**settings)
    env.reset()
    # `reset` smooths, and with no vertex pinned a Laplacian pulls this one quad
    # in on itself -- so the depth under test would be the smoother's, not the
    # one the test set up. Put the prototype's coordinates back.
    for vertex in graph.vertex_list(tag=False):
        env.graph.set_vertex_coordinate(vertex, graph.vertex_coordinate(vertex))
    env._update_half_edge_angles()
    return env


def _slab_on_an_arc(depth, radius=1.0, half=0.5):
    """One quad whose bottom side bulges downward on an arc of `radius`."""
    centre = np.array([0.0, np.sqrt(max(radius ** 2 - half ** 2, 1e-9))])
    start = np.arctan2(-centre[1], -half)
    end = np.arctan2(-centre[1], half)
    coordinates = {0: np.array([-half, 0.0]), 1: np.array([half, 0.0]),
                   2: np.array([half, depth]), 3: np.array([-half, depth])}
    graph = Tiler.from_face_loops([[0, 1, 2, 3]], coordinates)
    # counter-clockwise from vertex 0 at -120 degrees to vertex 1 at -60 passes
    # through the BOTTOM of the circle; clockwise takes the long way round and
    # gives a sagitta bigger than the circle's own radius
    graph.boundary_arcs.add(0, 1, Arc(centre, radius, start, end, ccw=True),
                            start_vertex=0)
    return graph


def test_a_deep_element_on_a_gentle_arc_is_charged_nothing():
    env = _env_from(_slab_on_an_arc(depth=3.0))
    assert env.clearance_shortfall() == pytest.approx(0.0)


def test_a_shallow_element_on_the_same_arc_is_charged():
    shallow = _env_from(_slab_on_an_arc(depth=0.05))
    deep = _env_from(_slab_on_an_arc(depth=3.0))
    assert shallow.clearance_shortfall() > 0.0
    assert shallow.clearance_shortfall() > deep.clearance_shortfall()


def test_the_charge_falls_as_the_element_deepens():
    values = [_env_from(_slab_on_an_arc(depth=d)).clearance_shortfall()
              for d in (0.05, 0.1, 0.2, 0.4)]
    assert values == sorted(values, reverse=True)


def test_a_straight_domain_is_untouched():
    coordinates = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0]),
                   2: np.array([1.0, 1.0]), 3: np.array([0.0, 1.0])}
    graph = Tiler.from_face_loops([[0, 1, 2, 3]], coordinates)
    assert _env_from(graph).clearance_shortfall() == pytest.approx(0.0)


def test_the_term_is_off_by_default():
    graph = _slab_on_an_arc(depth=0.05)
    env = _env_from(graph, clearance_potential_weight=0.0)
    assert env.clearance_potential_weight == 0.0
    # the potential must not move when the weight is zero, however thin the mesh
    deep = _env_from(_slab_on_an_arc(depth=3.0), clearance_potential_weight=0.0)
    assert env._compute_potential() == pytest.approx(deep._compute_potential())
