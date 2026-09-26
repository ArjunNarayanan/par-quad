from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 2, 1, 3, 4, 5],
    ]
    graph = Tiler.from_face_loops(face_loops)
    return graph


class TestDeleteVertex(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        vidx = self.graph.source_vertex(2, tag=False)
        self.graph.user_defined_vertices.remove(vidx)

    def test_invalid_delete(self):
        self.assertFalse(self.graph.is_valid_delete_source_vertex(2))


if __name__ == "__main__":
    unittest.main()
