from envs.substep_angle_env import AngleEnv
import unittest
from envs.environment_initializers import LEnv


class TestLEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.initializer = LEnv(90)
        env = AngleEnv(4, self.initializer)
        env.template_center = (0, env.graph.half_edge_tag)
        env._build_template()

        self.env = env

    def test_scores(self):
        self.assertEqual(self.env.template_face_score, 2)
        self.assertEqual(self.env.template_vertex_score, 2)
        self.assertEqual(self.env.template_angle_score, 2)

    def test_step1(self):
        self.env.step(4)
        self.assertEqual(self.env.template_face_score, 3)
        self.assertEqual(self.env.template_vertex_score, 3)
        self.assertEqual(self.env.template_angle_score, 3)
        self.assertEqual(self.env.reward, -1)
        self.assertTrue(self.env.template_center == (0, self.env.graph.half_edge_tag))

    def test_step2(self):
        self.env.step(4)
        self.env.step(8)
        self.assertEqual(self.env.template_face_score, 1)
        self.assertEqual(self.env.template_vertex_score, 1)
        self.assertEqual(self.env.template_angle_score, 1)
        self.assertEqual(self.env.reward, 2)
        self.assertTrue(self.env.template_center == (0, self.env.graph.half_edge_tag))

    def test_step3(self):
        self.env.step(4)
        self.env.step(8)
        self.env.step(16)
        self.env.step(14)
        self.assertEqual(self.env.template_face_score, 0)
        self.assertEqual(self.env.template_vertex_score, 0)
        self.assertEqual(self.env.template_angle_score, 0)
        self.assertEqual(self.env.reward, 2)
        self.assertTrue(self.env.template_center == (0, self.env.graph.half_edge_tag))


if __name__ == "__main__":
    unittest.main()
