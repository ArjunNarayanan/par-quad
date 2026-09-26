from envs.global_angle_env import AngleEnv
import unittest
from envs.environment_initializers import CenterCrack

# The legacy MDP: digon chords allowed, deletes in the vocabulary, and the
# score-difference reward. These tests pin the scoring pipeline, which the
# WP1 switches leave untouched, so they keep asserting against it.
LEGACY = dict(allow_digon=True, allow_delete=True, reward_mode="legacy",
              max_steps_factor=2, min_max_steps=0)



class TestCenterCrack(unittest.TestCase):
    def setUp(self) -> None:
        initializer = CenterCrack(90)
        env = AngleEnv(4, initializer, template_size=30, smooth_iterations=40, **LEGACY)

        env.template_center = (0, env.graph.half_edge_tag)
        env._build_template()

        self.env = env

    def test_scores(self):
        self.assertEqual(self.env.global_face_score, 7)
        self.assertEqual(self.env.global_vertex_score, 7)
        self.assertAlmostEqual(self.env.global_angle_score, 7)

    def test_step(self):
        self.env._step_half_edge_action(0, 4)
        self.env._step_half_edge_action(0, 4)
        self.env._step_half_edge_action(0, 4)

        self.env._step_half_edge_action(2, 4)
        self.env._step_half_edge_action(2, 4)
        self.env._step_half_edge_action(2, 4)

        self.env._step_half_edge_action(1, 4)

        self.env._step_half_edge_action(9, 2)
        self.env._step_half_edge_action(8, 2)
        self.env._step_half_edge_action(7, 2)
        self.env._step_half_edge_action(23, 2)
        self.env._step_half_edge_action(25, 2)
        self.env._step_half_edge_action(6, 2)
        self.env._step_half_edge_action(5, 2)

        self.env.graph.smooth_vertices(num_iter=50)
        self.env._update_half_edge_angles()
        self.env._update_scores_on_step()

        self.assertEqual(self.env.global_face_score, 0)
        self.assertEqual(self.env.global_vertex_score, 0)
        self.assertAlmostEqual(self.env.global_angle_score, 0, places=4)
        self.assertAlmostEqual(self.env.reward, 1, places=4)


if __name__ == "__main__":
    unittest.main()
