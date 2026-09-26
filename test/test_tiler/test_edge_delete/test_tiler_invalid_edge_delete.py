from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loop = [0, 1, 5, 9, 8, 7, 6, 5, 1, 2, 3, 4]
    graph = Tiler.from_face_loops([face_loop])
    return graph


class TestDeleteEdge(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()

    def test_invalid_delete(self):
        self.assertFalse(self.graph.is_valid_delete_half_edge(1))
        self.assertFalse(self.graph.is_valid_delete_half_edge(7))


if __name__ == "__main__":
    unittest.main()
