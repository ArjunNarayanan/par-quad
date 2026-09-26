"""The WP1 properties of the environment, each pinned by the defect it fixes."""

import unittest

import numpy as np

from envs.angle_env_with_length import AngleEnvWithLength
from envs.environment_initializers import LEnv, RandomPolygon
from envs.global_angle_env import AngleEnv, GLOBAL_FEATURE_SIZE


def make_env(**kwargs):
    settings = dict(template_size=48, max_edge_addition_steps=3, smooth_iterations=5)
    settings.update(kwargs)
    return AngleEnv(4, LEnv(90), **settings)


class TestActionVocabulary(unittest.TestCase):
    def test_monotone_vocabulary_has_no_deletes(self):
        env = make_env()
        self.assertEqual(env.num_actions_per_half_edge, 4)
        self.assertIsNone(env._delete_edge_action)
        self.assertIsNone(env._delete_vertex_action)
        self.assertEqual(env._insert_vertex_action, 3)

    def test_full_vocabulary_keeps_the_legacy_layout(self):
        env = make_env(allow_delete=True, allow_digon=True)
        self.assertEqual(env.num_actions_per_half_edge, 6)
        self.assertEqual(env._delete_edge_action, 3)
        self.assertEqual(env._insert_vertex_action, 4)
        self.assertEqual(env._delete_vertex_action, 5)

    def test_no_action_can_make_a_two_gon(self):
        """Local chord k maps to insert_half_edge(h, k + 1): the smallest chord
        makes a triangle, never a digon."""
        env = make_env()
        env.reset()
        for _ in range(6):
            legal = np.nonzero(env._get_action_mask() == 0)[0]
            self.assertGreater(len(legal), 0)
            env.step(int(np.random.choice(legal)))
            degrees = [env.graph.face_degree(f) for f in env.graph.face_list()]
            self.assertGreaterEqual(min(degrees), 3)

    def test_digon_flag_restores_the_old_mapping(self):
        env = make_env(allow_digon=True)
        self.assertEqual(env.chord_steps(0), 0)
        env = make_env(allow_digon=False)
        self.assertEqual(env.chord_steps(0), 1)


class TestResetNeverSolved(unittest.TestCase):
    def test_reset_resamples_past_solved_states(self):
        """A 4-gon is at par on arrival; SB3 steps it anyway and the only moves
        available make it worse, so it must never be handed out."""
        initializer = RandomPolygon([4, 5, 6], 90, scale=0.6)
        env = AngleEnvWithLength(4, initializer, template_size=48,
                                 max_edge_addition_steps=3)
        for _ in range(40):
            env.reset()
            self.assertFalse(env.is_at_par())

    def test_flag_off_allows_solved_starts(self):
        env = make_env(resample_if_at_par=False)
        env.reset()  # L-shape is not at par; the point is that it does not raise


class TestPotentialReward(unittest.TestCase):
    def test_return_is_potential_difference_minus_step_cost(self):
        env = make_env(step_cost=0.05, par_bonus=1.0)
        env.reset()
        start_potential = env.potential
        total, steps = 0.0, 0
        solved = False
        for _ in range(env.max_steps):
            legal = np.nonzero(env._get_action_mask() == 0)[0]
            _, reward, terminated, _, info = env.step(int(np.random.choice(legal)))
            total += reward
            steps += 1
            solved = info["at_par"]
            if terminated:
                break
        expected = env.potential - start_potential - 0.05 * steps
        if solved:
            expected += 1.0
        else:
            shortfall = max(0.0, env.quality_threshold - env.min_element_quality())
            expected -= env.quality_penalty * shortfall
        self.assertAlmostEqual(total, expected, places=5)

    def test_legacy_reward_still_reachable(self):
        env = make_env(reward_mode="legacy", allow_delete=True, allow_digon=True)
        env.reset()
        env.step(0)
        self.assertIsInstance(env.reward, float)


class TestObservation(unittest.TestCase):
    def test_global_vector_matches_env_scalars(self):
        env = make_env()
        obs, _ = env.reset()
        self.assertIn("global", obs)
        self.assertEqual(obs["global"].shape, (GLOBAL_FEATURE_SIZE,))
        expected_defect = abs(env.global_vertex_score - env.par)
        self.assertAlmostEqual(obs["global"][0], min(expected_defect, 20) / 5.0, places=5)
        self.assertAlmostEqual(obs["global"][1], min(env.global_face_score, 40) / 5.0, places=5)
        self.assertAlmostEqual(obs["global"][2], min(env.number_of_odd_faces(), 20) / 5.0,
                               places=5)

    def test_feature_sizes(self):
        self.assertEqual(AngleEnv.get_feature_size(), 11)
        self.assertEqual(AngleEnvWithLength.get_feature_size(), 10)
        env = make_env()
        obs, _ = env.reset()
        self.assertEqual(obs["features"].shape[1], AngleEnv.get_feature_size())

    def test_padded_slots_are_masked(self):
        env = make_env(template_size=48)
        obs, _ = env.reset()
        occupied = len(env.index_to_half_edge)
        padded = obs["mask"][occupied * env.num_actions_per_half_edge:]
        self.assertTrue(np.all(np.isneginf(padded)))


class TestDeterministicCentre(unittest.TestCase):
    def test_tie_break_is_stable(self):
        env = make_env(deterministic_center=True)
        env.reset()
        centres = set()
        for _ in range(10):
            env._set_half_edge_template_center(env.graph.half_edge_list())
            centres.add(env.template_center)
        self.assertEqual(len(centres), 1)


class TestInverseMasking(unittest.TestCase):
    def test_immediate_undo_is_masked(self):
        env = make_env(allow_delete=True, allow_digon=False, mask_inverse=True)
        env.reset()
        legal = np.nonzero(env._get_action_mask() == 0)[0]
        chord = next(a for a in legal
                     if a % env.num_actions_per_half_edge < env.max_edge_addition_steps)
        env.step(int(chord))
        kind, payload = env._inverse_block
        self.assertEqual(kind, "delete_edge")
        for half_edge in payload:
            slot = env.half_edge_to_index.get(half_edge)
            if slot is None:
                continue
            linear = slot * env.num_actions_per_half_edge + env._delete_edge_action
            self.assertFalse(env.is_valid_action(linear),
                             "the chord that was just inserted can be deleted for free")


class TestUnwinnableInstancesRejected(unittest.TestCase):
    def test_corner_angle_filter(self):
        initializer = RandomPolygon(list(range(12, 21)), 90, scale=0.8, min_quality=0.4)
        for _ in range(200):
            graph, desired = initializer()
            loop = [graph.source_vertex(h, tag=False)
                    for h in graph.generate_half_edge_face_loop(
                        graph.first_face_halfedge(graph.face_list()[0]))]
            import envs.polygon_utils as utils
            angles = utils.get_polygon_interior_angles(loop, graph.vertex_coordinates)
            self.assertTrue(initializer.is_winnable(angles, desired))


if __name__ == "__main__":
    unittest.main()
