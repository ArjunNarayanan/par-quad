"""`quality_aggregate: sum` charges one shortfall per finished quad, its worst corner.

Extensive, unlike min / mean / min+mean, which are one bounded number per mesh.
"""

import numpy as np

from envs.environment_maker import initialize_environment


def _env(aggregate):
    config = dict(name="GlobalAngleEnv", initializer=dict(name="LEnv"), face_desired_degree=4,
                  template_size=64, max_edge_addition_steps=3, reward_mode="normalized",
                  quality_metric="shape", quality_potential_weight=2.5, quality_aggregate=aggregate)
    env = initialize_environment(config)
    env.reset(seed=0)
    return env


def _play_until_some_quads(env, n=3):
    rng = np.random.default_rng(0)
    for _ in range(200):
        if len(env._target_face_worst_corners()) >= n:
            return True
        mask = env._get_obs()["mask"]
        valid = np.flatnonzero(np.isfinite(mask))
        env.step(int(rng.choice(valid)))
    return False


def test_sum_is_the_sum_of_per_face_worst_corner_shortfalls():
    env = _env("sum")
    assert _play_until_some_quads(env)
    worst = env._target_face_worst_corners()
    expected = sum(max(0.0, 0.4 - q) for q in worst)
    assert np.isclose(env.quality_shortfall(), expected)
    assert len(worst) == sum(1 for f in env.graph.face_list() if env.graph.face_degree(f) == 4)


def test_sum_charges_each_bad_face_in_full():
    env = _env("sum")
    assert _play_until_some_quads(env)
    worst = env._target_face_worst_corners()
    bad = [q for q in worst if q < 0.4]
    if bad:
        assert env.quality_shortfall() >= max(0.4 - q for q in bad)


def test_other_aggregates_are_unchanged():
    env = _env("min+mean")
    assert _play_until_some_quads(env)
    corners = env._target_face_qualities()
    shortfalls = [max(0.0, 0.4 - q) for q in corners]
    assert np.isclose(env.quality_shortfall(), 0.7 * max(shortfalls) + 0.3 * sum(shortfalls) / len(shortfalls))
