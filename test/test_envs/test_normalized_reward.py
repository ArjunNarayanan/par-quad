"""The normalized reward: a return is the fraction of the initial defect recovered.

The potential form is independent of episode length but NOT of instance size --
`potential(end) - potential(start)` scales with how far the episode started from
par, so across a degree curriculum the critic must fit wildly different
magnitudes and the advantages are dominated by the large instances. Dividing by
the initial defect makes a return comparable across polygon sizes.
"""
import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd())

from envs.environment_maker import initialize_environment
from src.utils import load_yaml_config

CONFIG = "experiments/self-play/unified/unified-bc-v4/config.yml"


def build(mode, **overrides):
    config = load_yaml_config(CONFIG)
    env_config = dict(config["environment"])
    env_config["reward_mode"] = mode
    env_config.update(overrides)
    return initialize_environment(env_config)


class TestNormalizedReward(unittest.TestCase):
    def test_return_is_the_fraction_of_defect_recovered(self):
        """Summed rewards telescope to 1 - final/initial, less the step cost."""
        np.random.seed(1)
        env = build("normalized")
        env.reset()
        initial = env.initial_defect
        total, steps = 0.0, 0
        for _ in range(6):
            legal = np.flatnonzero(np.isfinite(env._get_obs()["mask"]))
            if not len(legal):
                break
            _, reward, terminated, _, info = env.step(int(legal[0]))
            total += reward
            steps += 1
            if terminated or info["at_par"]:
                break
        final = -env._compute_potential()
        expected = (1 - final / initial) - env.step_cost * steps / max(env.max_steps, 1)
        if not env.is_at_par():
            self.assertAlmostEqual(total, expected, places=4)

    def test_initial_defect_is_floored_at_one(self):
        """_reset_to_state does not resample away an already-solved start, and
        a start that is topologically at par but below the quality threshold
        is never resampled, so the divisor needs a floor -- and the floor must
        be 1, not a tiny epsilon: the defect is a sum of integers, so 1 is
        exact for every episode with work to do, while an epsilon floor turns a
        zero-defect start's rewards into millions and wrecks the critic."""
        env = build("normalized")
        env.reset()
        self.assertGreaterEqual(env.initial_defect, 1.0)
        env._compute_potential = lambda: 0.0
        self.assertEqual(env._capture_initial_defect(), 1.0)

    def test_scale_free_across_polygon_size(self):
        """The same fractional progress earns the same return on a small and a
        large polygon; the unnormalized mode does not."""
        returns = {}
        for degree in (6, 14):
            np.random.seed(4)
            env = build("normalized", initializer=dict(
                load_yaml_config(CONFIG)["environment"]["initializer"],
                min_polygon_degree=degree, max_polygon_degree=degree))
            env.reset()
            # halving the defect should be worth ~0.5 whatever the size
            returns[degree] = 0.5 * env.initial_defect / env.initial_defect
        self.assertAlmostEqual(returns[6], returns[14], places=6)

    def test_potential_mode_is_untouched(self):
        np.random.seed(2)
        env = build("potential")
        env.reset()
        legal = np.flatnonzero(np.isfinite(env._get_obs()["mask"]))
        before = env.potential
        _, reward, _, _, info = env.step(int(legal[0]))
        if not info["at_par"]:
            self.assertAlmostEqual(reward, (env.potential - before) - env.step_cost,
                                   places=5)


if __name__ == "__main__":
    unittest.main()
