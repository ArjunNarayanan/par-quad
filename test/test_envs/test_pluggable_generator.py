"""A shape generator from outside this repository must plug in via config alone.

The initializer contract is the whole interface: a callable returning
(Tiler, {untagged vertex id: desired degree}).
"""
import os
import sys
import textwrap
import unittest

sys.path.append(os.getcwd())

from envs.environment_maker import get_env_initializer


class TestCustomInitializer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_external_generator")
        os.makedirs(cls.directory, exist_ok=True)
        with open(os.path.join(cls.directory, "outside_shapes.py"), "w") as handle:
            handle.write(textwrap.dedent('''
                from envs.environment_initializers import RandomPolygon

                def make_initializer(target_angle=90, low=6, high=9, **kwargs):
                    """Stands in for a generator maintained elsewhere."""
                    return RandomPolygon(list(range(low, high + 1)), target_angle)
            '''))
        sys.path.insert(0, cls.directory)

    @classmethod
    def tearDownClass(cls):
        if cls.directory in sys.path:
            sys.path.remove(cls.directory)

    def test_external_factory_is_reachable_from_config(self):
        initializer = get_env_initializer({
            "name": "Custom", "factory": "outside_shapes:make_initializer",
            "target_angle": 90, "low": 6, "high": 8})
        graph, desired = initializer()
        self.assertEqual(len(graph.face_list()), 1)
        self.assertTrue(desired)
        for vertex in graph.vertex_list(tag=False):
            self.assertIn(vertex, desired)

    def test_extra_keys_reach_the_factory(self):
        initializer = get_env_initializer({
            "name": "Custom", "factory": "outside_shapes:make_initializer",
            "target_angle": 90, "low": 7, "high": 7})
        graph, _ = initializer()
        self.assertEqual(len(graph.vertex_list()), 7)

    def test_a_malformed_factory_path_is_rejected_clearly(self):
        with self.assertRaises(ValueError):
            get_env_initializer({"name": "Custom", "factory": "no_colon_here"})

    def test_it_composes_inside_a_mixture(self):
        initializer = get_env_initializer({
            "name": "Mixture", "weights": [0.5, 0.5], "components": [
                {"name": "Custom", "factory": "outside_shapes:make_initializer",
                 "target_angle": 90, "low": 6, "high": 6},
                {"name": "RandomPolygon", "target_angle": 90,
                 "min_polygon_degree": 8, "max_polygon_degree": 8}]})
        degrees = {len(initializer()[0].vertex_list()) for _ in range(30)}
        self.assertIn(6, degrees)
        self.assertIn(8, degrees)


if __name__ == "__main__":
    unittest.main()
