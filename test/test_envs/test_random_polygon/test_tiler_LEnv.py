from envs.random_polygon_tiler_env import RandomPolygonEnv
import unittest
from envs.environment_initializers import LEnv


class TestLEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.initializer = LEnv(90)
        env = RandomPolygonEnv(4, self.initializer)
        env.template_center = (0, env.graph.half_edge_tag)
        env._build_template()

        self.env = env

    def test_scores(self):
        self.assertEqual(self.env.face_score, 2)
        self.assertEqual(self.env.vertex_score, 2)

    def test_step1(self):
        self.env.step(4)
        self.assertEqual(self.env.face_score, 3)
        self.assertEqual(self.env.vertex_score, 3)
        self.assertEqual(self.env.reward, -0.5)

    def test_step2(self):
        self.env.step(4)
        self.env.template_center = (0, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env.step(8)

        self.assertEqual(self.env.face_score, 1)
        self.assertEqual(self.env.vertex_score, 1)
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

        self.assertEqual(self.env.face_score, 0)
        self.assertEqual(self.env.vertex_score, 0)
        self.assertEqual(self.env.reward, 1)

    def test_reset(self):
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

        self.assertEqual(self.env.face_score, 0)
        self.assertEqual(self.env.vertex_score, 0)
        self.assertEqual(self.env.reward, 1)

        self.env.reset()
        self.assertEqual(self.env.face_score, 2)
        self.assertEqual(self.env.vertex_score, 2)


if __name__ == "__main__":
    unittest.main()
