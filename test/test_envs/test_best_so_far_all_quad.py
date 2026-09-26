"""`best_so_far_all_quad`: only an all-quad state may set the running best.

An open-face state earns nothing and does not move the best, however high its
potential; the first all-quad state is paid against the START state's score.
"""

import numpy as np

from envs.environment_maker import initialize_environment


def _env(**overrides):
    config = dict(name="GlobalAngleEnv", initializer=dict(name="LEnv"), face_desired_degree=4,
                  template_size=64, max_edge_addition_steps=3, reward_mode="normalized",
                  quality_metric="shape", quality_aggregate="sum", quality_potential_weight=2.5,
                  best_so_far_reward=True, terminate_at_par=False)
    config.update(overrides)
    env = initialize_environment(config)
    env.reset(seed=0)
    return env


def _random_walk(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    rewards, faces, bests = [], [], []
    for _ in range(steps):
        mask = env._get_obs()["mask"]
        valid = np.flatnonzero(np.isfinite(mask))
        if len(valid) == 0:
            break
        _, r, term, trunc, _ = env.step(int(rng.choice(valid)))
        rewards.append(r); faces.append(int(env.global_face_score)); bests.append(env.best_candidate)
        if term or trunc:
            break
    return rewards, faces, bests


def test_open_face_states_earn_nothing_and_do_not_move_the_best():
    env = _env(best_so_far_all_quad=True)
    start = env.best_candidate
    rewards, faces, bests = _random_walk(env, 60)
    for r, f, b in zip(rewards, faces, bests):
        if f > 0:
            assert r == 0.0
    # the best only ever changes on an all-quad step
    prev = start
    for f, b in zip(faces, bests):
        if b != prev:
            assert f == 0
        prev = b


def test_default_is_the_old_rule():
    a, b = _env(), _env(best_so_far_all_quad=False)
    ra, _, _ = _random_walk(a, 40)
    rb, _, _ = _random_walk(b, 40)
    assert ra == rb
    assert a.best_so_far_all_quad is False
