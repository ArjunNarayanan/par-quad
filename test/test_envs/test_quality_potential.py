"""Element quality as part of the objective, not just a parting charge.

The objective this supports is "get as close to par as you can while keeping
the mesh usable", which is not the same as "reach par". It has to be different
because on some domains par is not reachable at all -- the par-0 mesh of several
curved outlines inverts, so chasing par there chases a mesh no smoother can
draw (see `utilities/build_par0_mesh.py`).
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _env(weight, **extra):
    from copy import deepcopy

    from envs.environment_initializers import SquareHole
    from envs.global_angle_env import AngleEnv

    graph, desired = SquareHole(90)()
    holder = {"g": deepcopy(graph), "d": dict(desired)}

    class Fixed:
        n = len(desired)

        def __call__(self):
            return deepcopy(holder["g"]), dict(holder["d"])

    return AngleEnv(4, Fixed(), template_size=96, resample_if_at_par=False,
                    quality_potential_weight=weight, **extra)


class TestQualityPotential(unittest.TestCase):
    def test_off_by_default(self):
        self.assertEqual(_env(0.0).quality_potential_weight, 0.0)

    def test_an_unfinished_face_is_never_judged(self):
        """A raw polygon scores -1 for a reflex corner; that is not a mesh."""
        weighted = _env(4.0)
        weighted.reset()
        self.assertGreater(weighted.global_face_score, 0)
        # no face is a quad yet, so there is nothing to charge for
        self.assertEqual(weighted.quality_shortfall(), 0.0)

    def test_it_charges_from_the_first_finished_face(self):
        """Not gated on the WHOLE mesh being done.

        Gating on `face_score == 0` made the term exactly zero until the mesh
        was already complete, so the agent got no guidance on how to finish --
        only that finishing paid. Measuring the faces that ARE quads gives the
        signal from the first one onwards and still never judges an unfinished
        face.
        """
        env = _env(3.0)
        env.reset()
        env.quality_threshold = 0.4
        # one perfect quad and one poor one: the poor corner must be felt
        env._target_face_qualities = lambda: [1.0, 1.0, 1.0, 0.1]
        env.quality_aggregate = "min"
        self.assertAlmostEqual(env.quality_shortfall(), 0.4 - 0.1, places=12)

    def test_the_aggregates_differ_as_documented(self):
        env = _env(3.0)
        env.reset()
        env.quality_threshold = 0.4
        # three fine corners and one bad one
        env._target_face_qualities = lambda: [1.0, 1.0, 1.0, 0.0]
        worst, average = 0.4, 0.4 / 4
        env.quality_aggregate = "min"
        self.assertAlmostEqual(env.quality_shortfall(), worst, places=12)
        env.quality_aggregate = "mean"
        self.assertAlmostEqual(env.quality_shortfall(), average, places=12)
        env.quality_aggregate = "min+mean"
        self.assertAlmostEqual(env.quality_shortfall(),
                               0.7 * worst + 0.3 * average, places=12)
        # and "mean" is the one that notices a SECOND poor element
        env.quality_aggregate = "mean"
        one_bad = env.quality_shortfall()
        env._target_face_qualities = lambda: [1.0, 1.0, 0.0, 0.0]
        self.assertGreater(env.quality_shortfall(), one_bad)
        env.quality_aggregate = "min"
        env._target_face_qualities = lambda: [1.0, 1.0, 1.0, 0.0]
        only_worst = env.quality_shortfall()
        env._target_face_qualities = lambda: [1.0, 1.0, 0.0, 0.0]
        self.assertAlmostEqual(env.quality_shortfall(), only_worst, places=12,
                               msg="min cannot see a second poor element, which "
                                   "is exactly why it is a weak reward")

    def test_the_terminal_charge_is_not_applied_twice(self):
        """With the shortfall in the potential, the end-of-episode charge goes."""
        import inspect

        from envs.global_angle_env import AngleEnv
        source = inspect.getsource(AngleEnv.step)
        self.assertIn("not self.quality_potential_weight", source,
                      "the terminal quality charge must be skipped when the "
                      "shortfall is already telescoping through the potential")


if __name__ == "__main__":
    unittest.main()
