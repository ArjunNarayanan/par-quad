from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loops = [
        [11, 12, 14, 15, 0, 1],
        [3, 10, 11, 1, 2],
        [3, 4, 5, 6, 7, 8, 9, 10],
        [11, 10, 9, 12],
        [12, 9, 13]
    ]
    graph = Tiler.from_face_loops(face_loops)
    return graph


class TestKNN(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()

    def test_knn_15_10(self):
        neighbors = self.graph.knn_half_edges_with_boundary(15, 10)
        self.assertEqual(len(neighbors), 10)
        h = self.graph.half_edge_tag
        b = self.graph.boundary_tag

        test_idx = [15, 16, 14, 10, 17, 11, 13, 9, 18, 20]
        test_tags = [h, h, h, b, h, b, h, b, h, h]
        test_neighbors = list(zip(test_idx, test_tags))

        self.assertEqual(len(test_neighbors), 10)
        self.assertTrue(
            all(n == t for n, t in zip(neighbors, test_neighbors))
        )

    def test_knn_19_12(self):
        neighbors = self.graph.knn_half_edges_with_boundary(19, 12)
        self.assertEqual(len(neighbors), 12)
        h = self.graph.half_edge_tag
        b = self.graph.boundary_tag

        test_idx = [19, 20, 22, 7, 21, 17, 0, 8, 6, 23, 18, 16]
        test_tags = [h, h, h, h, h, h, h, h, h, h, h, h]
        test_neighbors = list(zip(test_idx, test_tags))

        self.assertEqual(len(test_neighbors), 12)
        self.assertTrue(
            all(n == t for n, t in zip(neighbors, test_neighbors))
        )
