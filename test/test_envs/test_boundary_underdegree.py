"""`boundary_underdegree_weight` charges boundary vertices sitting below their want."""

import numpy as np

from envs.environment_maker import initialize_environment


def _env(**overrides):
    config = dict(name="GlobalAngleEnv", initializer=dict(name="LEnv"), face_desired_degree=4,
                  template_size=64, max_edge_addition_steps=3, reward_mode="normalized",
                  quality_metric="shape", quality_aggregate="sum", quality_potential_weight=2.5)
    config.update(overrides)
    env = initialize_environment(config)
    env.reset(seed=0)
    return env


def test_under_degree_counts_boundary_gaps_only():
    env = _env(boundary_underdegree_weight=1.0)
    g = env.graph
    expected = 0
    for v in g.vertex_list(tag=False):
        if g.is_boundary_vertex(v):
            expected += max(0, env.effective_desired_degree(v) - g.vertex_degree(v))
    assert env.boundary_underdegree() == expected
    assert expected > 0  # a raw polygon has every flat point below its want


def test_weight_enters_the_potential_and_d0():
    base, charged = _env(), _env(boundary_underdegree_weight=1.0)
    gap = base._compute_potential() - charged._compute_potential()
    assert np.isclose(gap, charged.boundary_underdegree())
    assert np.isclose(charged.initial_defect - base.initial_defect, charged.boundary_underdegree())


def test_default_is_off():
    env = _env()
    assert env.boundary_underdegree_weight == 0.0
    assert np.isclose(env._compute_potential(), -(env.global_face_score + env.vertex_potential_weight * abs(env.global_vertex_score - env.par) + 2.5 * env.quality_shortfall()))
