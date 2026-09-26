"""Weighted cloning: a near-miss teaches completion without teaching mediocrity."""
import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd())

from src.behaviour_cloning import Dataset, add_trajectory, excess_weight


def fake_obs(template=4, features=3):
    return {
        "features": np.zeros((template, features), dtype=np.float32),
        "next": np.zeros(template, dtype=np.int64),
        "previous": np.zeros(template, dtype=np.int64),
        "twin": np.zeros(template, dtype=np.int64),
        "mask": np.zeros(template * 2, dtype=np.float32),
        "progress": np.zeros(1, dtype=np.float32),
    }


class TestExcessWeight(unittest.TestCase):
    def test_par_is_full_weight(self):
        self.assertAlmostEqual(excess_weight(0), 1.0)

    def test_weight_decays_with_excess(self):
        weights = [excess_weight(e) for e in range(5)]
        self.assertTrue(all(a > b for a, b in zip(weights, weights[1:])))
        self.assertTrue(all(w > 0 for w in weights))

    def test_negative_excess_is_clamped(self):
        """Excess is never below zero: par is the bound."""
        self.assertAlmostEqual(excess_weight(-3), 1.0)


class TestValueTargets(unittest.TestCase):
    def test_par_bonus_is_withheld_from_a_trajectory_that_missed_par(self):
        """Recording the bonus on a miss teaches the critic to expect a payment
        the episode never made."""
        observations = [fake_obs() for _ in range(3)]
        actions = [0, 1, 2]
        potentials = [-3.0, -2.0, -1.0]

        won = add_trajectory(Dataset(), observations, actions, potentials,
                             step_cost=0.05, par_bonus=1.0, at_par=True).finalize()
        missed = add_trajectory(Dataset(), observations, actions, potentials,
                                step_cost=0.05, par_bonus=1.0,
                                weight=0.25, at_par=False).finalize()
        for a, b in zip(won.values, missed.values):
            self.assertAlmostEqual(float(a - b), 1.0, places=5)

    def test_weight_is_carried_through_to_the_batch(self):
        observations = [fake_obs() for _ in range(2)]
        data = add_trajectory(Dataset(), observations, [0, 1], [-2.0, -1.0],
                              step_cost=0.05, par_bonus=1.0,
                              weight=0.25, at_par=False).finalize()
        self.assertTrue(np.allclose(data.weights, 0.25))
        *_, weights = data.batch(np.arange(len(data)))
        self.assertTrue(np.allclose(weights.numpy(), 0.25))

    def test_default_trajectory_is_unweighted(self):
        """The existing cloning path must be unchanged by the new argument."""
        observations = [fake_obs() for _ in range(2)]
        data = add_trajectory(Dataset(), observations, [0, 1], [-2.0, -1.0],
                              step_cost=0.05, par_bonus=1.0).finalize()
        self.assertTrue(np.allclose(data.weights, 1.0))


if __name__ == "__main__":
    unittest.main()
