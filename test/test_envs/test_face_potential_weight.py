"""`face_potential_weight` scales the face term of the potential and nothing else.

At its default of 1 the potential is what it always was; at w it charges w per
non-quad face, so that closing a face can be made worth more than any element
quality it could cost (all-quad first).
"""

import numpy as np

from envs.environment_maker import initialize_environment


def _env(**overrides):
    config = dict(name="GlobalAngleEnv", initializer=dict(name="LEnv"), face_desired_degree=4,
                  template_size=64, max_edge_addition_steps=3, reward_mode="normalized",
                  quality_metric="shape", quality_potential_weight=15.0, vertex_potential_weight=0.5)
    config.update(overrides)
    env = initialize_environment(config)
    env.reset(seed=0)
    return env


def test_default_weight_is_the_old_potential():
    env = _env()
    assert env.face_potential_weight == 1.0
    face, vertex = env.global_face_score, abs(env.global_vertex_score - env.par)
    expected = -(face + 0.5 * vertex + 15.0 * env.quality_shortfall())
    assert np.isclose(env._compute_potential(), expected)


def test_face_weight_scales_only_the_face_term():
    base, heavy = _env(), _env(face_potential_weight=5.0)
    assert heavy.global_face_score == base.global_face_score > 0
    gap = base._compute_potential() - heavy._compute_potential()
    assert np.isclose(gap, 4.0 * base.global_face_score)


def test_face_weight_reaches_the_initial_defect():
    base, heavy = _env(), _env(face_potential_weight=5.0)
    assert np.isclose(heavy.initial_defect - base.initial_defect, 4.0 * base.global_face_score)
