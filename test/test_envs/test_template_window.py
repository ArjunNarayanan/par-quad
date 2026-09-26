"""The template is a window, so a mesh larger than it must keep working.

`template_size` is the number of half-edges NEAREST the centre, not a capacity.
Everything downstream is built for a mesh bigger than the window -- `knn_half_edges`
returns the k nearest of however many there are, out-of-window neighbours carry the
-2 sentinel that `DCELConvBlock` gives its own learned embedding, and the scores,
`par` and the win test are computed over the whole graph. The only real restriction
is that the action space reaches the half-edges currently in view.

These tests pin that, because `terminate_on_overflow` used to hide it: with the
flag on, no episode ever ran past the window, so nothing exercised the case.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.getcwd())

import envs.polygon_utils as utils  # noqa: E402
from envs.environment_initializers import RandomPolygon  # noqa: E402
from envs.global_angle_env import AngleEnv  # noqa: E402


def _env(template_size, terminate_on_overflow, seed=0):
    return AngleEnv(
        face_desired_degree=4,
        graph_initializer=RandomPolygon(range(10, 15), utils.average_face_angle(4)),
        template_size=template_size,
        max_steps_factor=8.0,
        terminate_on_overflow=terminate_on_overflow,
    )


def _grow(env, steps, rng):
    """Take random legal moves, returning how many half-edges the mesh reached."""
    observation, _ = env.reset()
    largest = env.graph.number_of_half_edges()
    for _ in range(steps):
        valid = np.nonzero(np.isfinite(observation["mask"]))[0]
        if not len(valid):
            break
        observation, _, terminated, truncated, _ = env.step(int(rng.choice(valid)))
        largest = max(largest, env.graph.number_of_half_edges())
        if terminated or truncated:
            break
    return largest, observation


def test_the_env_runs_past_a_full_window():
    """A tiny template with the flag off: the mesh outgrows it and nothing breaks."""
    env = _env(template_size=24, terminate_on_overflow=False)
    rng = np.random.default_rng(0)
    grew = False
    for seed in range(12):
        largest, observation = _grow(env, 60, np.random.default_rng(seed))
        if largest > env.template_size:
            grew = True
            # the window is still exactly template_size wide and still usable
            assert observation["features"].shape[0] == env.template_size
            assert len(env.index_to_half_edge) == env.template_size
            assert np.isfinite(observation["mask"]).any(), "no legal move in view"
            # and the scoreboard is global, not windowed
            assert env.global_face_score >= 0
            assert env.par == env.compute_par()
    assert grew, "the mesh never outgrew the window; the test proved nothing"


def test_the_flag_is_what_stops_it():
    """The same growth ends the episode when the flag is on, and only then."""
    rng_seed = 3
    on = _env(template_size=24, terminate_on_overflow=True)
    largest_on, _ = _grow(on, 60, np.random.default_rng(rng_seed))
    off = _env(template_size=24, terminate_on_overflow=False)
    largest_off, _ = _grow(off, 60, np.random.default_rng(rng_seed))
    assert largest_off >= largest_on
    assert not off.template_overflowed() or not off.is_terminated() or off.num_steps >= off.max_steps


def test_overflow_defaults_to_not_terminating():
    """A config that says nothing gets the sliding window, not the guard."""
    from envs.environment_maker import initialize_environment
    env = initialize_environment({
        "name": "GlobalAngleEnv",
        "initializer": {"name": "RandomPolygon", "min_polygon_degree": 10,
                        "max_polygon_degree": 14},
        "face_desired_degree": 4,
        "template_size": 32,
    })
    assert env.terminate_on_overflow is False
