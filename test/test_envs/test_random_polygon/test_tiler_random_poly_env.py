from envs.random_polygon_tiler_env import RandomPolygonEnv
import unittest
import numpy as np
from envs.environment_initializers import Hexagon


class TestInsertHalfedge(unittest.TestCase):
    def setUp(self) -> None:
        initializer = Hexagon(60)
        self.env = RandomPolygonEnv(
            3,
            initializer,
            template_size=6,
            face_reward_weight=1,
            vertex_reward_weight=1
        )
        for vidx, _ in self.env.vertex_desired_degree.items():
            self.env.vertex_desired_degree[vidx] = 3
        self.env._update_scores_on_reset()

    def test_insert_edge_reward(self):
        self.env._step_insert_edge(0, 2)

        self.assertEqual(self.env.initial_face_score, 3)
        self.assertEqual(self.env.initial_vertex_score, 6)
        self.assertEqual(self.env.initial_score, 9)

        self.assertEqual(self.env.face_score, 2)
        self.assertEqual(self.env.vertex_score, 4)
        self.assertEqual(self.env.score, 6)

    def test_vertex_reward(self):
        self.env._step_insert_vertex(4)

        self.assertEqual(self.env.initial_face_score, 3)
        self.assertEqual(self.env.initial_vertex_score, 6)
        self.assertEqual(self.env.initial_score, 9)

        self.assertEqual(self.env.face_score, 4)
        self.assertEqual(self.env.vertex_score, 8)
        self.assertEqual(self.env.score, 12)

    def test_delete_edge_score(self):
        self.assertEqual(self.env.score, 9)

        self.env._step_insert_edge(2, 2)
        self.assertEqual(self.env.face_score, 2)
        self.assertEqual(self.env.vertex_score, 4)
        self.assertEqual(self.env.score, 6)

        self.env._step_insert_vertex(6)
        self.assertEqual(self.env.face_score, 4)
        self.assertEqual(self.env.vertex_score, 8)
        self.assertEqual(self.env.score, 12)

        self.env._step_insert_edge(8, 2)
        self.assertEqual(self.env.face_score, 3)
        self.assertEqual(self.env.vertex_score, 6)
        self.assertEqual(self.env.score, 9)

        self.env._step_delete_edge(8)
        self.assertEqual(self.env.face_score, 4)
        self.assertEqual(self.env.vertex_score, 8)
        self.assertEqual(self.env.score, 12)

        self.env._step_delete_edge(10)
        self.assertEqual(self.env.face_score, 4)
        self.assertEqual(self.env.vertex_score, 8)
        self.assertEqual(self.env.score, 12)

        self.env._step_delete_edge(6)
        self.assertEqual(self.env.face_score, 4)
        self.assertEqual(self.env.vertex_score, 8)
        self.assertEqual(self.env.score, 12)

    def test_vertex_delete_reward(self):
        self.env._step_insert_vertex(5)
        self.assertEqual(self.env.face_score, 4)
        self.assertEqual(self.env.vertex_score, 8)
        self.assertEqual(self.env.score, 12)

        self.env._step_insert_edge(5, 3)
        self.assertEqual(self.env.face_score, 3)
        self.assertEqual(self.env.vertex_score, 6)
        self.assertEqual(self.env.score, 9)

        self.env._step_delete_source_vertex(6)
        self.assertEqual(self.env.face_score, 2)
        self.assertEqual(self.env.vertex_score, 4)
        self.assertEqual(self.env.score, 6)

        self.env._step_delete_source_vertex(7)
        self.assertEqual(self.env.face_score, 2)
        self.assertEqual(self.env.vertex_score, 4)
        self.assertEqual(self.env.score, 6)

        self.env._step_insert_vertex(7)
        self.assertEqual(self.env.face_score, 4)
        self.assertEqual(self.env.vertex_score, 8)
        self.assertEqual(self.env.score, 12)

        self.env._step_insert_edge(8, 2)
        self.assertEqual(self.env.face_score, 3)
        self.assertEqual(self.env.vertex_score, 6)
        self.assertEqual(self.env.score, 9)

        self.env._step_delete_source_vertex(9)
        self.assertEqual(self.env.face_score, 3)
        self.assertEqual(self.env.vertex_score, 6)
        self.assertEqual(self.env.score, 9)

        self.env._step_delete_source_vertex(8)
        self.assertEqual(self.env.face_score, 3)
        self.assertEqual(self.env.vertex_score, 6)
        self.assertEqual(self.env.score, 9)

    def test_composite_reward(self):
        self.assertEqual(self.env.score, 9)
        self.assertEqual(self.env.initial_score, 9)

        self.env._step_insert_edge(5, 1)
        self.assertEqual(self.env.face_score, 2)
        self.assertEqual(self.env.vertex_score, 4)
        self.assertEqual(self.env.score, 6)

        self.env._step_insert_vertex(7)
        self.assertEqual(self.env.face_score, 4)
        self.assertEqual(self.env.vertex_score, 8)
        self.assertEqual(self.env.score, 12)

        self.env._step_insert_edge(6, 1)
        self.assertEqual(self.env.face_score, 3)
        self.assertEqual(self.env.vertex_score, 6)
        self.assertEqual(self.env.score, 9)

        self.env._step_insert_edge(2, 3)
        self.assertEqual(self.env.face_score, 2)
        self.assertEqual(self.env.vertex_score, 4)
        self.assertEqual(self.env.score, 6)

        self.env._step_insert_edge(12, 2)
        self.assertEqual(self.env.face_score, 1)
        self.assertEqual(self.env.vertex_score, 2)
        self.assertEqual(self.env.score, 3)

        self.env._step_insert_edge(3, 1)
        self.assertEqual(self.env.face_score, 0)
        self.assertEqual(self.env.vertex_score, 0)
        self.assertEqual(self.env.score, 0)

    def test_vertex_delete_reward2(self):
        self.env._step_insert_edge(2, 2)
        self.env._step_insert_vertex(6)
        self.env._step_insert_edge(8, 2)
        self.env._step_delete_edge(8)
        self.env._step_delete_source_vertex(11)

        self.assertEqual(self.env.face_score, 4)
        self.assertEqual(self.env.vertex_score, 4)
        self.assertEqual(self.env.score, 8)

    def test_feature_matrix(self):
        matrix = self.env._get_feature_matrix()
        test_matrix = np.zeros((self.env.template_size, 5), dtype=np.float32)
        test_matrix[:6, :] = [
            2 / 3,
            3,
            6 / 3,
            3,
            1
        ]
        self.assertTrue((test_matrix == matrix).all())


class TestHexEnvTemplate3(unittest.TestCase):
    def setUp(self) -> None:
        initializer = Hexagon(60)
        self.env = RandomPolygonEnv(
            3,
            initializer,
            template_size=3,
        )

        for vidx, _ in self.env.vertex_desired_degree.items():
            self.env.vertex_desired_degree[vidx] = 3
        self.env._update_scores_on_reset()

    def test_feature_matrix(self):
        matrix = self.env._get_feature_matrix()
        test_matrix = np.zeros((self.env.template_size, 5), dtype=np.float32)
        test_matrix[:6, :] = [
            2 / 3,
            3,
            6 / 3,
            3,
            1
        ]
        self.assertTrue((test_matrix == matrix).all())

    def test_next_edges(self):
        next_edges = self.env._get_next_edges()
        test_edges = [1, self.env.template_boundary_index, 0]
        self.assertTrue((next_edges == test_edges).all())

    def test_previous_edges(self):
        prev_edges = self.env._get_previous_edges()
        test_edges = [2, 0, self.env.template_boundary_index]
        self.assertTrue((prev_edges == test_edges).all())

    def test_twin_edges(self):
        twin_edges = self.env._get_twin_edges()
        test_edges = [self.env.geometric_boundary_index] * 3
        self.assertTrue((twin_edges == test_edges).all())


class TestObs4(unittest.TestCase):
    def setUp(self) -> None:
        initializer = Hexagon(60)
        env = RandomPolygonEnv(
            3,
            initializer,
            template_size=4,
        )
        for vidx, _ in env.vertex_desired_degree.items():
            env.vertex_desired_degree[vidx] = 3
        env._update_scores_on_reset()

        env._step_insert_edge(3, 2)
        env.template_center = (1, env.graph.half_edge_tag)
        env._build_template()
        self.env = env
        self.obs = env._get_obs()

    def test_template(self):
        htag = self.env.graph.half_edge_tag
        index_to_half_edge = [1, 2, 0, 7]
        index_to_half_edge = [(idx, htag) for idx in index_to_half_edge]
        self.assertTrue(index_to_half_edge == self.env.index_to_half_edge)

    def test_feature_matrix(self):
        test_matrix = np.zeros((self.env.template_size, 5), dtype=np.float32)
        test_matrix[:2, :] = [2 / 3, 3, 4 / 3, 3, 1]
        test_matrix[2:, :] = [3 / 3, 3, 4 / 3, 3, 1]
        matrix = self.obs["features"]

        self.assertTrue((matrix == test_matrix).all())

    def test_next_edges(self):
        test_next = [1, 3, 0, 2]
        self.assertTrue((test_next == self.obs["next"]).all())

    def test_prev_edges(self):
        test_prev = [2, 0, 3, 1]
        self.assertTrue((test_prev == self.obs["previous"]).all())

    def test_twin(self):
        t = self.env.template_boundary_index
        b = self.env.geometric_boundary_index
        test_edges = [b, b, b, t]
        self.assertTrue((test_edges == self.obs["twin"]).all())


class TestObs5(unittest.TestCase):
    def setUp(self) -> None:
        initializer = Hexagon(60)
        env = RandomPolygonEnv(
            3,
            initializer,
            template_size=5,
        )
        for vidx, _ in env.vertex_desired_degree.items():
            env.vertex_desired_degree[vidx] = 3
        env._update_scores_on_reset()

        env._step_insert_edge(3, 2)
        env.template_center = (1, env.graph.half_edge_tag)
        env._build_template()
        self.env = env
        self.obs = env._get_obs()

    def test_template(self):
        htag = self.env.graph.half_edge_tag
        index_to_half_edge = [1, 2, 0, 7, 6]
        index_to_half_edge = [(idx, htag) for idx in index_to_half_edge]
        self.assertTrue(index_to_half_edge == self.env.index_to_half_edge)

    def test_feature_matrix(self):
        test_matrix = np.zeros((self.env.template_size, 5), dtype=np.float32)
        test_matrix[:2, :] = [2 / 3, 3, 4 / 3, 3, 1]
        test_matrix[2:, :] = [3 / 3, 3, 4 / 3, 3, 1]
        matrix = self.obs["features"]

        self.assertTrue((matrix == test_matrix).all())

    def test_next_edges(self):
        t = self.env.template_boundary_index
        test_next = [1, 3, 0, 2, t]
        self.assertTrue((test_next == self.obs["next"]).all())

    def test_prev_edges(self):
        t = self.env.template_boundary_index
        test_prev = [2, 0, 3, 1, t]
        self.assertTrue((test_prev == self.obs["previous"]).all())

    def test_twin(self):
        t = self.env.template_boundary_index
        b = self.env.geometric_boundary_index
        test_edges = [b, b, b, 4, 3]
        self.assertTrue((test_edges == self.obs["twin"]).all())


class TestActionSequence(unittest.TestCase):
    def setUp(self) -> None:
        initializer = Hexagon(60)
        env = RandomPolygonEnv(
            3,
            initializer,
            template_size=18,
        )
        for vidx, _ in env.vertex_desired_degree.items():
            env.vertex_desired_degree[vidx] = 3
        env._update_scores_on_reset()

        env._step_insert_edge(5, 2)
        env._step_insert_vertex(3)
        env._step_insert_vertex(6)
        env._step_insert_edge(8, 2)
        env.template_center = (5, env.graph.half_edge_tag)
        env._build_template()
        self.env = env
        self.obs = env._get_obs()

    def test_template(self):
        htag = self.env.graph.half_edge_tag
        index_to_half_edge = [5, 0, 9, 1, 6, 10, 7, 11, 4, 2, 12, 8, 3]
        index_to_half_edge = [(idx, htag) for idx in index_to_half_edge]
        self.assertTrue(index_to_half_edge == self.env.index_to_half_edge)

    def test_feature_matrix(self):
        test_matrix = np.zeros((self.env.template_size, 5), dtype=np.float32)
        test_matrix[:13, 0] = [1, 2 / 3, 3 / 6, 2 / 3, 1, 1, 3 / 6, 3 / 6, 2 / 3, 1, 3 / 4, 3 / 4, 2 / 3]
        test_matrix[:13, 1] = [3, 3, 6, 3, 3, 3, 6, 6, 3, 3, 4, 4, 3]

        test_matrix[:5, 2] = 5 / 3
        test_matrix[5:13, 2] = 4 / 3

        test_matrix[:13, 3] = 3
        test_matrix[:13, 4] = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1]

        matrix = self.obs["features"]

        self.assertTrue((matrix == test_matrix).all())

    def test_next_edges(self):
        test_next_edges = np.arange(self.env.template_size)
        test_next_edges[:13] = [1, 3, 0, 4, 2, 7, 9, 11, 5, 12, 6, 8, 10]

        self.assertTrue((test_next_edges == self.obs["next"]).all())

    def test_previous_edges(self):
        test_prev_edges = np.arange(self.env.template_size)
        test_prev_edges[:13] = [2, 0, 4, 1, 3, 8, 10, 5, 11, 6, 12, 7, 9]

        self.assertTrue((test_prev_edges == self.obs["previous"]).all())

    def test_twin_edges(self):
        test_twin_edges = np.arange(self.env.template_size)
        b = self.env.geometric_boundary_index
        test_twin_edges[:13] = [b, b, 5, b, 6, 2, 4, 10, b, b, 7, b, b]

        self.assertTrue((test_twin_edges == self.obs["twin"]).all())


if __name__ == "__main__":
    unittest.main()
