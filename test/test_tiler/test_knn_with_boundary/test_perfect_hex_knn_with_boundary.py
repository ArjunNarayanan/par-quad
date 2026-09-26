from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5]
    ]
    graph = Tiler.from_face_loops(face_loops)
    graph.insert_half_edge(0, 2)
    graph.insert_vertex(7)
    graph.insert_half_edge(8, 1)
    graph.insert_half_edge(5, 1)
    graph.insert_half_edge(6, 1)
    graph.insert_half_edge(15, 1)
    return graph


class TestKNN(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()

    def test_knn_13_8(self):
        neighbors = self.graph.knn_half_edges_with_boundary(13, 8)
        h = self.graph.half_edge_tag
        b = self.graph.boundary_tag

        test_idx = [13, 11, 4, 12, 10, 4, 5, 7]
        test_tags = [h, h, h, h, h, b, h, h]
        test_neighbors = list(zip(test_idx, test_tags))

        self.assertEqual(len(neighbors), 8)
        self.assertTrue(
            all(h == t for h, t in zip(neighbors, test_neighbors))
        )

    def test_knn_2_15(self):
        neighbors = self.graph.knn_half_edges_with_boundary(2, 15)
        h = self.graph.half_edge_tag
        b = self.graph.boundary_tag

        self.assertEqual(len(neighbors), 15)
        test_idx = [2, 9, 17, 2, 8, 16, 3, 10, 15, 1, 3, 11, 14, 1, 4]
        test_tags = [h, h, h, b, h, h, h, h, h, h, b, h, h, b, h]

        test_neighbors = list(zip(test_idx, test_tags))
        self.assertEqual(len(test_neighbors), 15)
        self.assertTrue(
            all(t == n for t, n in zip(test_neighbors, neighbors))
        )

    def test_knn_6_30(self):
        neighbors = self.graph.knn_half_edges_with_boundary(6, 30)
        self.assertEqual(len(neighbors), 24)
        h = self.graph.half_edge_tag
        b = self.graph.boundary_tag

        test_idx = [6, 0, 14, 7, 0, 15, 12, 5, 1, 16, 13, 5, 1, 17, 11, 4, 2, 9, 10, 4, 2, 8, 3, 3]
        test_tags = [h, h, h, h, b, h, h, h, h, h, h, b, b, h, h, h, h, h, h, b, b, h, h, b]
        test_neighbors = list(zip(test_idx, test_tags))

        self.assertEqual(len(test_neighbors), 24)
        self.assertTrue(
            all(t == n for t, n in zip(test_neighbors, neighbors))
        )
