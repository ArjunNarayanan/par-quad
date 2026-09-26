from envs.substep_angle_env import AngleEnv
import unittest
from envs.environment_initializers import SquareHole


class TestSquareHoleEnv(unittest.TestCase):
    def setUp(self) -> None:
        initializer = SquareHole(90)
        env = AngleEnv(4, initializer)
        env.template_center = (0, env.graph.half_edge_tag)
        env._build_template()

        self.env = env

    def test_scores(self):
        self.assertEqual(self.env.template_face_score, 6)
        self.assertEqual(self.env.template_vertex_score, 8)
        self.assertAlmostEqual(self.env.template_angle_score, 8)

    def test_step1(self):
        self.env.step(26)
        self.assertEqual(self.env.template_face_score, 4)
        self.assertEqual(self.env.template_vertex_score, 8)
        self.assertEqual(self.env.template_angle_score, 8)
        self.assertAlmostEqual(self.env.reward, 2 / 3)
        self.assertTrue(self.env.template_center == (0, self.env.graph.half_edge_tag))

    def test_step1(self):
        self.env.step(26)
        self.env.step(44)
        self.env.step(56)

        self.assertEqual(self.env.template_face_score, 0)
        self.assertEqual(self.env.template_vertex_score, 8)
        self.assertEqual(self.env.template_angle_score, 8)
        self.assertAlmostEqual(self.env.reward, 2 / 3)
        self.assertTrue(self.env.template_center == (0, self.env.graph.half_edge_tag))


if __name__ == "__main__":
    unittest.main()
