from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5]
    ]
    graph = Tiler.from_face_loops(face_loops)
    return graph


class TestKNN(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()

    def test_knn_3(self):
        neighbors = self.graph.knn_half_edges_with_boundary(0, 3)
        h = self.graph.half_edge_tag

        test_neighbors = [(0, h), (1, h), (5, h)]
        self.assertEqual(len(neighbors), 3)
        self.assertTrue(
            all(n == t for n, t in zip(neighbors, test_neighbors))
        )

    def test_knn_15(self):
        neighbors = self.graph.knn_half_edges_with_boundary(2, 15)
        self.assertEqual(len(neighbors), 12)
        h = self.graph.half_edge_tag
        b = self.graph.boundary_tag

        test_neighbors = [2, 3, 1, 2, 4, 3, 0, 1, 5, 4, 0, 5]
        test_tags = [h, h, h, b, h, b, h, b, h, b, b, b]
        test_neighbors = list(zip(test_neighbors, test_tags))

        self.assertTrue(
            all(n == t for n, t in zip(neighbors, test_neighbors))
        )

    def test_knn_6(self):
        neighbors = self.graph.knn_half_edges_with_boundary(2, 6)
        self.assertEqual(len(neighbors), 6)
        h = self.graph.half_edge_tag
        b = self.graph.boundary_tag

        test_neighbors = [2, 3, 1, 2, 4, 3]
        test_tags = [h, h, h, b, h, b]
        test_neighbors = list(zip(test_neighbors, test_tags))

        self.assertTrue(
            all(n == t for n, t in zip(neighbors, test_neighbors))
        )
