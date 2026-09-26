"""`global_features: intensive` keeps every global entry inside a fixed range on any
domain size; the default "raw" is unchanged."""

import numpy as np

from envs.environment_maker import initialize_environment


def _env(**overrides):
    config = dict(name="GlobalAngleEnv", initializer=dict(name="LEnv"), face_desired_degree=4,
                  template_size=64, max_edge_addition_steps=3, reward_mode="normalized", best_so_far_reward=True)
    config.update(overrides)
    env = initialize_environment(config)
    env.reset(seed=0)
    return env


def test_raw_is_the_default_and_unchanged():
    env = _env()
    assert env.global_features == "raw"
    g = env._get_global_features()
    assert g.shape == (env.global_feature_size,)
    assert np.isclose(g[6], min(len(env.graph.face_list()), 40) / 10.0)


def test_intensive_entries_are_bounded_and_per_face():
    env = _env(global_features="intensive")
    g = env._get_global_features()
    assert g.shape == (env.global_feature_size,)
    faces = max(len(env.graph.face_list()), 1)
    assert np.isclose(g[0], min(abs(env.global_vertex_score - env.par) / faces, 2.0))
    assert np.isclose(g[6], np.tanh(faces / 40.0))
    assert np.all(np.abs(g) <= 2.0)
