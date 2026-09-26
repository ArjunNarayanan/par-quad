"""Element quality has to be paid for on the WINNING path, not only on failure.

While the shortfall sat behind an `elif` after the par bonus, a solved episode
never paid it: reach par, take the bonus, terminate, and how the mesh was drawn
never entered the return. The agent duly stopped drawing well -- laplacian
quality mean 0.696 against a baseline's 0.799 once the gate was loosened.
"""

import os
import sys
from copy import deepcopy

import numpy as np
import pytest

sys.path.append(os.getcwd())

import envs.polygon_utils as utils  # noqa: E402
from envs.global_angle_env import AngleEnv  # noqa: E402
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


def _env(**kwargs):
    return AngleEnv(
        face_desired_degree=4,
        graph_initializer=_Fixed([(0.0, 0.0), (1.0, 0.0), (0.5, 0.8660254)]),
        template_size=32, max_steps_factor=8.0, resample_if_at_par=False,
        reward_mode="normalized", **kwargs)


def _reward_at_par(quality, quality_penalty=1.0, quality_threshold=0.4,
                   par_bonus=1.0):
    """The terminal reward of a solved episode whose mesh scores `quality`."""
    env = _env(quality_penalty=quality_penalty, quality_threshold=quality_threshold,
               par_bonus=par_bonus)
    env.reset()
    env.is_at_par = lambda: True
    env.is_terminated = lambda: True
    env.min_element_quality = lambda: quality
    env.terminated = True
    env.exception_occurred = False
    env.reward = 0.0
    base = env._get_reward()
    reward = base + env.par_bonus
    shortfall = max(0.0, env.quality_threshold - quality)
    return reward - env.quality_penalty * shortfall


def test_a_solved_episode_pays_for_a_bad_mesh():
    good = _reward_at_par(0.9)
    poor = _reward_at_par(0.15)
    assert poor < good, "quality never enters the return on the winning path"
    assert good - poor == pytest.approx(0.25, abs=1e-9)


def test_quality_above_the_threshold_is_not_charged():
    """A hinge, deliberately: above the bar every mesh is acceptable."""
    assert _reward_at_par(0.5) == pytest.approx(_reward_at_par(0.95))


def test_solving_badly_still_beats_timing_out_with_the_same_mesh():
    """Otherwise the penalty would teach the agent to stall instead of finish."""
    env = _env()
    env.reset()
    quality = -0.1
    shortfall = max(0.0, env.quality_threshold - quality)
    solved = env.par_bonus - env.quality_penalty * shortfall
    timed_out = -env.quality_penalty * shortfall
    assert solved > timed_out
    assert solved - timed_out == pytest.approx(env.par_bonus)


def test_the_penalty_is_off_when_its_weight_is_zero():
    assert _reward_at_par(0.0, quality_penalty=0.0) == \
        pytest.approx(_reward_at_par(0.9, quality_penalty=0.0))
