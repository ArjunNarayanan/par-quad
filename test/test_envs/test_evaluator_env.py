"""Geometry in the loop, the win test at the untangler.

The point of the split is that a topology which cannot be drawn well is a
different failure from a mesh that merely needs smoothing. A vertex inserted on
the edge of a triangle gives a single quad with a 180-degree corner:
topologically at par, geometrically impossible, and no smoother repairs it
because the flat corner is forced. A mesh the Laplacian leaves at 0.2 is fine
and is smoothing's problem -- and the k=4 pinwheel is exactly that case.
"""

import os
import sys
from copy import deepcopy

import numpy as np
import pytest

sys.path.append(os.getcwd())

import envs.polygon_utils as utils  # noqa: E402
from envs.angle_env_with_length import AngleEnvWithLength  # noqa: E402
from envs.environment_initializers import RandomPolygon  # noqa: E402
from envs.evaluator_env import EvaluatorAngleEnv  # noqa: E402
from envs.solved_instances import pinwheel_annulus_mesh  # noqa: E402
from src.tiler import Tiler  # noqa: E402


class _Fixed:
    def __init__(self, points):
        self.points = points

    def __call__(self):
        coordinates = {i: list(p) for i, p in enumerate(self.points)}
        loop = list(range(len(self.points)))
        graph = Tiler.from_face_loops([loop], coordinates)
        angles = utils.get_polygon_interior_angles(loop, graph.vertex_coordinates)
        desired = {v: utils.rounded_desired_degree(a, 90) for v, a in angles.items()}
        return deepcopy(graph), dict(desired)


def _triangle_env(**kwargs):
    return EvaluatorAngleEnv(
        face_desired_degree=4,
        graph_initializer=_Fixed([(0.0, 0.0), (1.0, 0.0), (0.5, 0.8660254)]),
        template_size=32, max_steps_factor=8.0, resample_if_at_par=False, **kwargs)


def _flat_corner(env):
    """Insert a vertex on a triangle edge: at par, and no embedding saves it."""
    env.reset()
    half_edge = env.graph.half_edge_list()[0]
    env.graph.insert_vertex(half_edge)
    env.vertex_desired_degree[env.graph.target_vertex(half_edge, tag=False)] = \
        env.boundary_vertex_desired_degree
    env._update_half_edge_angles()
    env._update_scores_on_reset()
    env._verdict_step = None


def test_geometry_is_still_in_the_loop():
    """The whole point of this env over TopologicalEnv: ten features, smoothing on."""
    assert EvaluatorAngleEnv.get_feature_size() == AngleEnvWithLength.get_feature_size()
    env = EvaluatorAngleEnv(
        face_desired_degree=4,
        graph_initializer=RandomPolygon(range(8, 12), utils.average_face_angle(4)),
        template_size=32, max_steps_factor=4.0)
    assert env.smooth_iterations > 0, "the Laplacian must still run between moves"
    observation, _ = env.reset()
    assert observation["features"].shape[1] == 10


def test_the_observation_is_unchanged_from_the_parent():
    """No extra entry: a state at par with poor quality already says so through
    the quality scalar the global vector has always carried."""
    env = EvaluatorAngleEnv(
        face_desired_degree=4,
        graph_initializer=RandomPolygon(range(8, 12), utils.average_face_angle(4)),
        template_size=32, max_steps_factor=4.0)
    plain = AngleEnvWithLength(
        face_desired_degree=4,
        graph_initializer=RandomPolygon(range(8, 12), utils.average_face_angle(4)),
        template_size=32, max_steps_factor=4.0)
    observation, _ = env.reset()
    assert env.global_feature_size == plain.global_feature_size
    assert observation["global"].shape == (plain.global_feature_size,)


def test_a_forced_flat_corner_is_still_rejected():
    env = _triangle_env(degeneracy_threshold=0.1)
    _flat_corner(env)
    assert env.is_topologically_at_par(), "the single quad should be at par"
    assert not env.is_at_par(), "a 180-degree corner must not pass"


def test_a_rejected_candidate_does_not_end_the_episode():
    env = _triangle_env(degeneracy_threshold=0.1)
    _flat_corner(env)
    assert not env.is_terminated(), "a rejection hands the board back, it does not end it"


def test_the_verdict_is_cached_within_a_step():
    """`is_terminated` and the reward path both ask; the untangler is not cheap."""
    env = _triangle_env(degeneracy_threshold=0.1)
    _flat_corner(env)
    calls = {"n": 0}
    real = env.evaluate_candidate

    def counted():
        calls["n"] += 1
        return real()

    env.evaluate_candidate = counted
    env.is_at_par(); env.is_at_par(); env.is_at_par()
    assert calls["n"] == 1


@pytest.mark.parametrize("k", [4, 5, 6])
def test_every_pinwheel_is_accepted(k):
    """k=4 sits at 0.203 under a Laplacian; the old gate refused exactly it."""
    pytest.importorskip("geo2d")
    graph, desired, _ = pinwheel_annulus_mesh(k=k, rng=np.random.default_rng(0))
    env = EvaluatorAngleEnv(
        face_desired_degree=4,
        graph_initializer=RandomPolygon(range(8, 12), utils.average_face_angle(4)),
        template_size=160, max_steps_factor=4.0, degeneracy_threshold=0.1)
    env._reset_to_state(deepcopy(graph), dict(desired))
    env.graph.smooth_vertices(num_iter=env.smooth_iterations)
    env._update_half_edge_angles()
    env._update_scores_on_reset()
    env._verdict_step = None
    assert env.is_topologically_at_par()
    assert env.is_at_par(), f"k={k} rejected at laplacian quality {env.min_element_quality()}"


def test_the_old_env_refuses_the_tightest_pinwheel():
    """The regression this env exists to fix; if this ever passes, drop the env."""
    pytest.importorskip("geo2d")
    graph, desired, _ = pinwheel_annulus_mesh(k=4, rng=np.random.default_rng(0))
    env = AngleEnvWithLength(
        face_desired_degree=4,
        graph_initializer=RandomPolygon(range(8, 12), utils.average_face_angle(4)),
        template_size=160, max_steps_factor=4.0)
    env._reset_to_state(deepcopy(graph), dict(desired))
    env.graph.smooth_vertices(num_iter=env.smooth_iterations)
    env._update_half_edge_angles()
    env._update_scores_on_reset()
    assert env.is_topologically_at_par()
    assert not env.is_at_par()


def test_the_untangler_actually_runs():
    """The bug this nearly shipped with.

    The evaluator used to deepcopy the ENV, which raises because
    `graph_initializer` can hold a geo2d module reference. The except swallowed
    it and the gate degraded to raw angles without saying so, which happens to
    accept the pinwheels for the wrong reason. So assert the untangler MOVES the
    number, not just that the verdict is right.
    """
    pytest.importorskip("geo2d")
    graph, desired, _ = pinwheel_annulus_mesh(k=4, rng=np.random.default_rng(0))
    env = EvaluatorAngleEnv(
        face_desired_degree=4,
        graph_initializer=RandomPolygon(range(8, 12), utils.average_face_angle(4)),
        template_size=160, max_steps_factor=4.0, degeneracy_threshold=0.1)
    env._reset_to_state(deepcopy(graph), dict(desired))
    env.graph.smooth_vertices(num_iter=env.smooth_iterations)
    env._update_half_edge_angles()
    env._update_scores_on_reset()

    laplacian = env.min_element_quality()
    accepted = env.evaluate_candidate()
    kept = env.min_element_quality()
    assert kept > laplacian + 0.1, (
        f"the untangler did not run: {laplacian:.3f} -> {kept:.3f}")
    assert accepted
    # and the improvement is KEPT: the agent works from the better drawing
    assert env.min_element_quality() == pytest.approx(kept, abs=1e-12)


def test_an_uncopyable_initializer_does_not_silently_degrade_the_gate():
    """A geo2d-backed Mixture cannot be deepcopied; the gate must not care."""
    pytest.importorskip("geo2d")
    import sys as _sys

    class _Uncopyable(RandomPolygon):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.module = _sys                    # what a geo2d initializer holds

    graph, desired, _ = pinwheel_annulus_mesh(k=4, rng=np.random.default_rng(0))
    env = EvaluatorAngleEnv(
        face_desired_degree=4,
        graph_initializer=_Uncopyable(range(8, 12), utils.average_face_angle(4)),
        template_size=160, max_steps_factor=4.0, degeneracy_threshold=0.1)
    with pytest.raises(TypeError):
        deepcopy(env)                              # the env itself still cannot copy
    env._reset_to_state(deepcopy(graph), dict(desired))
    env.graph.smooth_vertices(num_iter=env.smooth_iterations)
    env._update_half_edge_angles()
    env._update_scores_on_reset()
    laplacian = env.min_element_quality()
    env.evaluate_candidate()
    assert env.min_element_quality() > laplacian + 0.1


