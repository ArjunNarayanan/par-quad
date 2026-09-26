from src.tiler import Tiler
import unittest


def initialize_graph():
    graph = Tiler.from_face_loops([[0, 1, 2, 3]])
    graph.insert_half_edge(0, 1)
    graph.insert_vertex(4)
    graph.insert_vertex(6)
    graph.insert_half_edge(0, 2)
    graph.delete_half_edge(4)
    graph.delete_source_vertex(10)
    return graph


class TestDeleteVertex(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()

    def test_invalid_delete(self):
        self.assertFalse(self.graph.is_valid_delete_source_vertex(7))
        self.assertFalse(self.graph.is_valid_delete_source_vertex(8))
