from envs.angle_env_with_length import AngleEnvWithLength as AngleEnv
import unittest
from envs.environment_initializers import LEnv

# The legacy MDP: digon chords allowed, deletes in the vocabulary, and the
# score-difference reward. These tests pin the scoring pipeline, which the
# WP1 switches leave untouched, so they keep asserting against it.
LEGACY = dict(allow_digon=True, allow_delete=True, reward_mode="legacy",
              max_steps_factor=2, min_max_steps=0)



class TestLEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.initializer = LEnv(90)
        env = AngleEnv(4, self.initializer, **LEGACY)
        env.template_center = (0, env.graph.half_edge_tag)
        env._build_template()

        self.env = env

    def test_length_feature(self):
        features = self.env._get_length_features()
        test_features = [2, 1, 2, 1, 1, 1]
        self.assertTrue((features == test_features).all())

    def test_step1(self):
        self.env.step(4)
        self.env.template_center = (0, self.env.graph.half_edge_tag)
        self.env._build_template()

        features = self.env._get_length_features()
        test_features = [1, 1, 2, 1, 1, 1, 1]
        self.assertTrue((features == test_features).all())
        self.assertEqual(self.env.global_face_score, 3)
        self.assertEqual(self.env.global_vertex_score, 3)
        self.assertEqual(self.env.global_angle_score, 3)

    def test_step2(self):
        self.env.step(4)
        self.env.template_center = (0, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env.step(8)
        self.env.template_center = (0, self.env.graph.half_edge_tag)
        self.env._build_template()

        features = self.env._get_length_features()
        test_features = [1, 1, 2, 1, 1, 1, 1, 1, 1]
        self.assertTrue((features == test_features).all())

        self.assertEqual(self.env.global_face_score, 1)
        self.assertEqual(self.env.global_vertex_score, 1)
        self.assertEqual(self.env.global_angle_score, 1)
        self.assertEqual(self.env.reward, 1)

    def test_step3(self):
        self.env.step(4)
        self.env.template_center = (0, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env.step(8)
        self.env.template_center = (0, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env.step(16)
        self.env.template_center = (0, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env.step(14)

        features = self.env._get_length_features()
        self.assertTrue((features == 1).all())

        self.assertEqual(self.env.global_face_score, 0)
        self.assertEqual(self.env.global_vertex_score, 0)
        self.assertEqual(self.env.global_angle_score, 0)
        self.assertEqual(self.env.reward, 1)


if __name__ == "__main__":
    unittest.main()
