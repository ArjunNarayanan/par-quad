from src.tiler import Tiler
import unittest


def initialize_double_hex_graph():
    face_loops = [
        [0, 1, 2, 7, 8, 9],
        [2, 3, 4, 5, 6, 7]
    ]

    graph = Tiler.from_face_loops(face_loops)

    return graph


class TestActions(unittest.TestCase):
    def setUp(self):
        self.graph = initialize_double_hex_graph()
        self.graph.insert_half_edge(11, 2)
        self.graph.insert_vertex(12)
        self.graph.insert_vertex(4)

    def test_boundary_vertex(self):
        self.assertFalse(self.graph.is_boundary_vertex(10))
        self.assertTrue(self.graph.is_boundary_vertex(11))

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 12)
        self.assertEqual(self.graph.number_of_half_edges(), 17)
        self.assertEqual(self.graph.number_of_faces(), 3)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 11)
        self.assertEqual(self.graph.number_of_nodes(), 43)

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 28)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 28)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 17)
        self.assertEqual(self.graph.number_of_edges(), 73)

    def test_next_edges(self):
        next_edges = [1, 2, 3, 4, 16, 0, 7, 12, 9, 10, 15, 6, 14, 8, 11, 13, 5]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, htag) for (idx, ne) in enumerate(next_edges))
        )

    def test_previous_edges(self):
        next_edges = [1, 2, 3, 4, 16, 0, 7, 12, 9, 10, 15, 6, 14, 8, 11, 13, 5]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.previous_half_edge(ne) == (idx, htag) for (idx, ne) in enumerate(next_edges))
        )

    def test_twin_edges(self):
        twin_idx = [0, 1, 11, 2, 3, 4, 5, 6, 7, 8, 9, 2, 13, 12, 15, 14, 10]
        h = self.graph.half_edge_tag
        b = self.graph.boundary_tag
        twin_tags = [b, b, h] + 8 * [b] + 5 * [h] + [b]
        self.assertTrue(
            all(self.graph.twin_half_edge(idx) == (tidx, twin_tags[idx]) for idx, tidx in enumerate(twin_idx))
        )

        self.assertTrue(
            all(self.graph.twin_half_edge((hidx, twin_tags[idx])) == (idx, h) for idx, hidx in enumerate(twin_idx))
        )

    def test_source_vertices(self):
        source_vertices = [0, 1, 2, 7, 8, 9, 2, 3, 4, 5, 6, 7, 4, 10, 10, 7, 11]
        vtag = self.graph.vertex_tag
        self.assertTrue(
            all(self.graph.source_vertex(idx) == (vidx, vtag) for idx, vidx in enumerate(source_vertices))
        )

        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for hidx, vidx in enumerate(source_vertices))
        )

    def test_target_vertices(self):
        target_vertices = [1, 2, 7, 8, 11, 0, 3, 4, 5, 6, 7, 2, 10, 4, 7, 10, 9]
        vtag = self.graph.vertex_tag
        self.assertTrue(
            all(self.graph.target_vertex(idx) == (vidx, vtag) for idx, vidx in enumerate(target_vertices))
        )

        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for hidx, vidx in enumerate(target_vertices))
        )

    def test_boundary_source(self):
        source_vertices = [1, 2, 8, 11, 0, 3, 4, 5, 6, 7, 9]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag
        self.assertTrue(
            all(self.graph.source_vertex((bidx, btag)) == (vidx, vtag) for bidx, vidx in enumerate(source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = [0, 1, 7, 8, 9, 2, 3, 4, 5, 6, 11]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag
        self.assertTrue(
            all(self.graph.target_vertex((bidx, btag)) == (vidx, vtag) for bidx, vidx in enumerate(target_vertices))
        )

    def test_face_edges(self):
        face_indices = 6 * [0] + [2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 0]
        ftag = self.graph.face_tag
        htag = self.graph.half_edge_tag

        self.assertTrue(
            all(self.graph.face(idx, tag=False) == fidx for idx, fidx in enumerate(face_indices))
        )
        self.assertTrue(
            all(self.graph.has_edge((fidx, ftag), (hidx, htag)) for hidx, fidx in enumerate(face_indices))
        )

    def test_vertex_degree(self):
        degrees = [2, 2, 3, 2, 3, 2, 2, 4, 2, 2, 2, 2]
        self.assertTrue(
            all(self.graph.vertex_degree(vidx) == d for vidx, d in enumerate(degrees))
        )

    def test_face_degree(self):
        degrees = [7, 5, 5]
        self.assertTrue(
            all(self.graph.face_degree(fidx) == d for fidx, d in enumerate(degrees))
        )


if __name__ == "__main__":
    unittest.main()