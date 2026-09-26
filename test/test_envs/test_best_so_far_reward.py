"""The episode runs to max_steps and the BEST state along it is what counts.

Equivalent to cutting the trajectory at its best candidate and learning from
that prefix, without touching SB3's rollout collection -- which this repo has
deliberately avoided reimplementing before, so that a library upgrade cannot
silently change the objective.
"""

import numpy as np
import pytest

import envs.polygon_utils as utils
from envs.global_angle_env import AngleEnv
from src.tiler import Tiler


def _env(**overrides):
    coordinates = {0: np.array([0.0, 0.0]), 1: np.array([2.0, 0.0]),
                   2: np.array([2.0, 1.0]), 3: np.array([1.0, 1.0]),
                   4: np.array([1.0, 2.0]), 5: np.array([0.0, 2.0])}
    loop = [0, 1, 2, 3, 4, 5]
    graph = Tiler.from_face_loops([loop], coordinates)
    angles = utils.get_polygon_interior_angles(loop, graph.vertex_coordinates)
    desired = {v: utils.rounded_desired_degree(a, 90) for v, a in angles.items()}

    class _Fixed:
        n = len(desired)

        def __call__(self):
            from copy import deepcopy
            return deepcopy(graph), dict(desired)

    settings = dict(face_desired_degree=4, graph_initializer=_Fixed(),
                    template_size=64, resample_if_at_par=False,
                    best_so_far_reward=True, terminate_at_par=False,
                    terminate_when_usable=False)
    settings.update(overrides)
    env = AngleEnv(**settings)
    env.reset()
    return env


def _rollout(env, seed=0, limit=200):
    rng = np.random.default_rng(seed)
    observation = env._get_obs()
    start = env.candidate_score()
    scores, total, done, steps = [start], 0.0, False, 0
    while not done and steps < limit:
        mask = observation["mask"].reshape(-1)
        legal = np.nonzero(np.isfinite(mask))[0]
        if not len(legal):
            break
        observation, reward, done, truncated, _ = env.step(int(rng.choice(legal)))
        done = done or truncated
        total += reward
        scores.append(env.candidate_score())
        steps += 1
    return start, scores, total


def test_the_return_is_exactly_best_minus_start():
    """The defining identity: this IS the post-hoc truncation objective."""
    for seed in range(4):
        env = _env()
        start, scores, total = _rollout(env, seed=seed)
        assert total == pytest.approx(max(scores) - start, abs=1e-9)


def test_no_step_is_ever_punished():
    """Wrecking the mesh after a good candidate must cost nothing.

    The vocabulary is monotone and there is no stop action, so the agent cannot
    decline to keep moving; charging it for moves it has to make is what the
    old rule did.
    """
    env = _env()
    rng = np.random.default_rng(7)
    observation = env._get_obs()
    done = False
    while not done:
        mask = observation["mask"].reshape(-1)
        legal = np.nonzero(np.isfinite(mask))[0]
        if not len(legal):
            break
        observation, reward, done, truncated, _ = env.step(int(rng.choice(legal)))
        done = done or truncated
        assert reward >= -1e-12


def test_the_step_charge_is_inside_the_max():
    """A candidate reached sooner must score higher than the same one later.

    Charged at the end instead, the total is `step_cost * max_steps`, fixed at
    reset, a constant offset that cancels out of the policy gradient -- and the
    agent has no reason to be brief.
    """
    env = _env()
    env.num_steps = 5
    early = env.candidate_score()
    env.num_steps = 40
    late = env.candidate_score()
    assert early > late


def test_the_par_bonus_is_not_paid_twice():
    """It lives in `candidate_score`, so the terminal path must not add it."""
    env = _env()
    assert env.par_bonus > 0
    source = __import__("inspect").getsource(type(env).step)
    assert "at_par and not self.best_so_far_reward" in source


def test_the_observation_carries_the_gap_to_the_best():
    """Otherwise the reward depends on history the agent cannot see."""
    with_flag = _env()
    without = _env(best_so_far_reward=False)
    assert with_flag.global_feature_size == without.global_feature_size + 1
    assert with_flag._get_obs()["global"].shape == (with_flag.global_feature_size,)


def test_the_default_path_is_untouched():
    """Everything off by default; the old reward must still be the old reward."""
    env = _env(best_so_far_reward=False)
    assert env.best_candidate is None
    start, scores, total = _rollout(env, seed=3)
    # the old rule can and does go negative, which is the difference
    assert total != pytest.approx(max(scores) - start, abs=1e-9)
