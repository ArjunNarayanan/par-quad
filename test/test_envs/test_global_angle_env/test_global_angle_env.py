from envs.global_angle_env import AngleEnv
import unittest
import numpy as np
from src.tiler import Tiler
from envs.environment_initializers import Hexagon


class TestHexEnvInitial(unittest.TestCase):
    def setUp(self) -> None:
        initializer = Hexagon(60)
        self.env = AngleEnv(3, initializer, template_size=6)

        self.env.template_center = (3, self.env.graph.half_edge_tag)
        self.env._build_template()
        self.env._update_scores_on_reset()

    def test_scores(self):
        self.assertAlmostEqual(self.env.global_face_score, 3)
        self.assertAlmostEqual(self.env.global_angle_score, 6)

    def test_feature_matrix(self):
        obs = self.env._get_obs()
        coords = obs["features"][:, [0, 1]]
        test_coords = np.array(
            [
                [0, 0],
                [1, 0],
                [-0.5, np.sqrt(3) / 2],
                [1.5, np.sqrt(3) / 2],
                [0, np.sqrt(3)],
                [1, np.sqrt(3)]
            ]
        )
        # self.assertTrue(np.isclose(coords, test_coords, atol=1e-5).all())


class TestHexEnvSteps(unittest.TestCase):
    def setUp(self) -> None:
        initializer = Hexagon(60)
        self.env = AngleEnv(3, initializer, template_size=8)

        self.env._step_insert_edge(5, 2)
        self.env.template_center = (3, self.env.graph.half_edge_tag)
        self.env.half_edge_angles = self.env.graph.half_edge_angles()
        self.env._build_template()
        self.env._update_scores_on_reset()

    def test_scores(self):
        self.assertAlmostEqual(self.env.global_face_score, 2)
        self.assertAlmostEqual(self.env.global_angle_score, 4)


class TestCompositeActions(unittest.TestCase):
    def setUp(self) -> None:
        initializer = Hexagon(60)
        self.env = AngleEnv(
            3,
            initializer,
            face_reward_weight=0.5,
            angle_reward_weight=0.5,
            vertex_reward_weight=0,
            template_size=14,
            smooth_iterations=0
        )

        self.env._step_insert_edge(0, 2)
        self.env._step_insert_vertex(7)
        self.env._step_insert_edge(8, 1)
        self.env._step_insert_edge(5, 1)
        self.env._step_insert_edge(6, 1)
        self.env._step_insert_edge(2, 1)

        self.env.graph.smooth_vertices(num_iter=50)
        self.env._update_half_edge_angles()
        self.env._update_scores_on_step()

    def test_scores(self):
        self.assertEqual(self.env.global_face_score, 0)
        self.assertAlmostEqual(self.env.global_angle_score, 0, places=5)


if __name__ == "__main__":
    unittest.main()
