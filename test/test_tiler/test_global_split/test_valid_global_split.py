from src.tiler import Tiler
import unittest


def initialize_rectangle_graph():
    face_loops = [
        [0, 1, 6, 5],
        [1, 2, 7, 6],
        [2, 3, 8, 7],
        [3, 4, 9, 8],
        [5, 6, 11, 10],
        [6, 7, 12, 11],
        [7, 8, 13, 12],
        [8, 9, 14, 13]
    ]
    graph = Tiler.from_face_loops(face_loops)
    return graph


class TestRectangleGlobalSplit(unittest.TestCase):
    def setUp(self):
        self.graph = initialize_rectangle_graph()

    def test_max_steps(self):
        self.assertFalse(self.graph.is_valid_quad_global_split_source(3, 3))
        self.assertTrue(self.graph.is_valid_quad_global_split_source(3, 4))
        self.assertTrue(self.graph.is_valid_quad_global_split_source(8, 2))


class TestGraph2(unittest.TestCase):
    def setUp(self):
        self.graph = initialize_graph2()

    def test_valid_global_split(self):
        self.assertTrue(self.graph.is_valid_quad_global_split_source(3, 6))
        self.assertFalse(self.graph.is_valid_quad_global_split_source(3, 5))


def initialize_graph2():
    face_loops = [
        [0, 1, 3, 2],
        [2, 3, 5, 4],
        [4, 5, 6, 7],
        [1, 8, 5, 3],
        [5, 8, 7, 6]
    ]
    graph = Tiler.from_face_loops(face_loops)
    return graph


if __name__ == "__main__":
    unittest.main()
