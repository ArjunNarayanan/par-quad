"""The hole domains the certified generator cannot build.

`random_annulus` makes holes on the square lattice only and `SquareHole` is one
fixed shape, so this is what supplies hole domains with arbitrary corner angles
to the RL stage.
"""
import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd())

import envs.polygon_utils as utils
from envs.environment_initializers import RandomPolygonWithHole
from envs.environment_maker import get_env_initializer


def euler_characteristic(graph):
    seen, edges = set(), 0
    for h in graph.half_edge_list():
        if h in seen:
            continue
        seen.add(h)
        seen.add(graph.twin_half_edge(h))
        edges += 1
    return len(graph.vertex_list()) - edges + len(graph.face_list())


class TestRandomPolygonWithHole(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        self.init = RandomPolygonWithHole(range(6, 13), utils.average_face_angle(4))

    def test_is_a_one_face_annulus(self):
        """chi is 0, not 1: the domain has a hole. Square hole's par of 0 rests
        on exactly this -- its corner excess is 0, so par is |0 + 4*chi|."""
        for trial in range(15):
            with self.subTest(trial=trial):
                graph, _ = self.init()
                self.assertEqual(len(graph.face_list()), 1)
                self.assertEqual(euler_characteristic(graph), 0)

    def test_slit_vertices_are_visited_twice(self):
        """The hole is encoded by a slit, so exactly two vertices repeat."""
        for trial in range(10):
            with self.subTest(trial=trial):
                graph, _ = self.init()
                face = graph.face_list()[0]
                loop = graph.generate_half_edge_face_loop(
                    graph.first_face_halfedge(face))
                sources = [graph.source_vertex(h, tag=False) for h in loop]
                repeated = {v for v in sources if sources.count(v) > 1}
                self.assertEqual(len(repeated), 2)
                for vertex in repeated:
                    self.assertEqual(sources.count(vertex), 2)

    def test_every_vertex_has_a_desired_degree(self):
        for trial in range(10):
            with self.subTest(trial=trial):
                graph, desired = self.init()
                for vertex in graph.vertex_list(tag=False):
                    self.assertIn(vertex, desired)
                    self.assertGreaterEqual(desired[vertex], 2)

    def test_reachable_from_the_config_dispatch(self):
        init = get_env_initializer({
            "name": "RandomPolygonWithHole", "target_angle": 90,
            "min_polygon_degree": 6, "max_polygon_degree": 10})
        graph, desired = init()
        self.assertEqual(euler_characteristic(graph), 0)
        self.assertTrue(desired)


if __name__ == "__main__":
    unittest.main()
