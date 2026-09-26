"""The certified anchor must be able to come from the generator, not a pickle.

The geo2d half of the training mixture draws a new domain every episode. A
fixed pickle makes the certified half incomparable to it -- and since geo2d
produces only par 0 domains, that finite list is where every par > 0 instance
the agent ever sees comes from.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.getcwd())

from envs.environment_maker import get_env_initializer  # noqa: E402
from envs.solved_instances import GeneratedInstanceInitializer  # noqa: E402


def _config(**overrides):
    base = {"name": "GeneratedInstances", "batch_size": 12, "refresh_after": 12,
            "seed": 3, "cell_range": [2, 8], "hole_probability": 0.0}
    base.update(overrides)
    return base


def test_the_maker_dispatches_to_it():
    initializer = get_env_initializer(_config())
    assert isinstance(initializer, GeneratedInstanceInitializer)


def test_nothing_is_generated_until_it_is_called():
    """The batch must be built per worker, not forked into every worker."""
    initializer = get_env_initializer(_config())
    assert initializer._inner is None
    initializer()
    assert initializer._inner is not None


def test_it_refreshes_instead_of_recycling():
    initializer = get_env_initializer(_config(batch_size=8, refresh_after=8))
    for _ in range(20):
        graph, desired = initializer()
        assert len(graph.face_list()) >= 1
        assert desired
    assert initializer._batches >= 2, "the pool never refreshed"


def test_two_workers_do_not_draw_the_same_pool():
    """Each worker is its own process; identical streams would defeat the point."""
    first = GeneratedInstanceInitializer(batch_size=8, refresh_after=8, seed=1,
                                         cell_range=(2, 8), hole_probability=0.0)
    second = GeneratedInstanceInitializer(batch_size=8, refresh_after=8, seed=2,
                                          cell_range=(2, 8), hole_probability=0.0)
    first(), second()
    left = sorted(tuple(i.loop) for i in first._inner.instances)
    right = sorted(tuple(i.loop) for i in second._inner.instances)
    assert left != right


def test_the_generator_settings_reach_the_generator():
    initializer = get_env_initializer(
        _config(batch_size=6, refresh_after=6, hole_probability=1.0,
                cell_range=[8, 12]))
    initializer()
    # hole_probability 1.0 over cells that admit a hole: some must have one
    assert any(i.has_hole for i in initializer._inner.instances)
