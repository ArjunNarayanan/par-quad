from src.tiler import Tiler
import unittest


def initialize_rectangle_graph():
    face_loops = [
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [7, 3, 4, 8],
        [4, 1, 5, 8],
        [5, 6, 10, 9],
        [11, 10, 6, 7],
        [12, 11, 7, 8],
        [9, 12, 8, 5]
    ]
    graph = Tiler.from_face_loops(face_loops)
    return graph


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.graph = initialize_rectangle_graph()

    def test_invalid_global_split(self):
        self.assertFalse(self.graph.is_valid_quad_global_split_source(3, 10))
        self.assertFalse(self.graph.is_valid_quad_global_split_source(7, 10))


if __name__ == "__main__":
    unittest.main()
