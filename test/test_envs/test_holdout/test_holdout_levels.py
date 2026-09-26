"""The fifteen Mesh Quest levels are the held-out test set.

These tests guard the interface to them: every level must build through the
training environment, and every known expert solution must still be
expressible in the environment's action vocabulary and reach par when
replayed. That second check is what would catch an action-space change that
silently puts a known solution out of reach.
"""

import json
import os
import unittest

from src.holdout import LEVELS, LEVEL_NAMES, make_level_env

ENV_CONFIG = {
    "name": "AngleEnvWithLength",
    "face_desired_degree": 4,
    "template_size": 64,
    "max_edge_addition_steps": 3,
    "max_steps_factor": 4,
    "smooth_iterations": 5,
}

TRAJECTORIES = os.path.join("experiments", "expert_bc", "expert_trajectories.json")


def load_expert_trajectories():
    with open(TRAJECTORIES) as handle:
        return json.load(handle)


class TestLevelsBuild(unittest.TestCase):
    def test_fifteen_levels(self):
        # the FIFTEEN are the frozen held-out set; the game may carry more
        from src.holdout import LEVEL_NAMES
        self.assertEqual(len(LEVEL_NAMES), 15)
        self.assertTrue(set(LEVEL_NAMES) <= set(LEVELS))

    def test_every_level_builds(self):
        for level in LEVEL_NAMES:
            env = make_level_env(ENV_CONFIG, level)
            obs, _ = env.reset()
            self.assertEqual(obs["features"].shape,
                             (env.template_size, env.num_features))
            self.assertGreaterEqual(env.par, 0)

    def test_no_level_starts_solved(self):
        for level in LEVEL_NAMES:
            env = make_level_env(ENV_CONFIG, level)
            env.reset()
            self.assertFalse(env.is_at_par(), f"{level} starts at par")


class TestExpertReplay(unittest.TestCase):
    """Every certified solution must survive the action-space definition."""

    def setUp(self):
        self.trajectories = load_expert_trajectories()

    def _replay(self, level, actions):
        env = make_level_env(ENV_CONFIG, level)
        env.reset()
        for tagged, kind in actions:
            half_edge = (tagged[0], tagged[1])
            if kind == "V":
                local = env._insert_vertex_action
            else:
                local = kind - env._chord_offset
                self.assertTrue(0 <= local < env.max_edge_addition_steps,
                                f"{level}: chord k={kind} is outside the vocabulary")
            slot = env.half_edge_to_index.get(half_edge)
            self.assertIsNotNone(slot, f"{level}: half-edge {half_edge} outside the template")
            linear = slot * env.num_actions_per_half_edge + local
            self.assertTrue(env.is_valid_action(linear),
                            f"{level}: expert move masked as illegal")
            env.step(linear)
        return env

    def test_expert_solutions_reach_par(self):
        for level, record in self.trajectories.items():
            if not record.get("actions"):
                continue
            with self.subTest(level=level):
                env = self._replay(level, record["actions"])
                self.assertTrue(env.is_at_par(), f"{level}: replay did not reach par")

    def test_expert_solutions_use_no_deletes(self):
        """The monotone vocabulary is only safe while this holds."""
        for level, record in self.trajectories.items():
            if not record.get("actions"):
                continue
            kinds = {action[1] for action in record["actions"]}
            self.assertFalse(kinds & {"D", "X"},
                             f"{level} needs a delete; the monotone action space would lose it")


if __name__ == "__main__":
    unittest.main()


class TestNoTrainTestLeakage(unittest.TestCase):
    """A generated domain must not pose the same problem as a held-out level.

    The generator never sees the levels, but it does build some of them:
    L-shape, T-bracket, Z-shape and Plus are small polyominoes, and the polar
    family at three quads is a triangle with par 1, which is the Triangle
    level. `drop_holdout_domains` is what makes a held-out score honest, and
    this pins that it actually removes them.
    """

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from envs.solved_instances import generate_instances
        rng = np.random.default_rng(11)
        cls.instances, _ = generate_instances(400, cell_range=(2, 12), rng=rng)

    def test_signature_is_invariant_to_rotation_and_reflection(self):
        from src.holdout import canonical_signature
        degrees = [2, 2, 4, 3, 2]
        rotated = degrees[2:] + degrees[:2]
        self.assertEqual(canonical_signature(degrees), canonical_signature(rotated))
        self.assertEqual(canonical_signature(degrees),
                         canonical_signature(list(reversed(degrees))))

    def test_the_generator_really_does_build_held_out_domains(self):
        """If this ever stops being true the exclusion has silently stopped mattering."""
        from src.holdout import canonical_signature, holdout_signatures
        banned = set(holdout_signatures().values())
        hits = sum(canonical_signature([i.desired[v] for v in i.loop]) in banned
                   for i in self.instances)
        self.assertGreater(hits, 0)

    def test_exclusion_removes_every_held_out_signature(self):
        from src.behaviour_cloning import drop_holdout_domains
        from src.holdout import canonical_signature, holdout_signatures
        banned = set(holdout_signatures().values())
        kept = drop_holdout_domains(self.instances, verbose=False)
        for instance in kept:
            signature = canonical_signature([instance.desired[v] for v in instance.loop])
            self.assertNotIn(signature, banned)

    def test_exclusion_removes_every_triangle(self):
        """Triangle's signature is the only 3-gon one, so none may survive."""
        from src.behaviour_cloning import drop_holdout_domains
        kept = drop_holdout_domains(self.instances, verbose=False)
        self.assertEqual([i for i in kept if i.polygon_degree == 3], [])
