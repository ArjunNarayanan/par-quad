from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loop = [
        [0, 1, 2, 6, 5],
        [6, 2, 3, 4, 5]
    ]
    graph = Tiler.from_face_loops(face_loop)
    return graph

"""
class TestDeleteEdge(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.graph.delete_half_edge(5)
        self.hidx = [0, 1, 6, 7, 8, 9, 3, 4]

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 7)
        self.assertEqual(self.graph.number_of_half_edges(), 8)
        self.assertEqual(self.graph.number_of_faces(), 1)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 6)
        self.assertEqual(self.graph.number_of_nodes(), 22)

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 14)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 14)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 8)

        self.assertEqual(self.graph.number_of_edges(), 36)

    def test_half_edges(self):
        self.assertTrue(len(self.graph.half_edges) == 14)
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all((hidx, htag) in self.graph.half_edges for hidx in self.hidx)
        )
        btag = self.graph.boundary_tag
        self.assertTrue(
            all((hidx, btag) in self.graph.half_edges for hidx in range(6))
        )

    def test_next_edges(self):
        next_idx = [1, 6, 7, 8, 9, 3, 4, 0]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, htag) for idx, ne in zip(self.hidx, next_idx))
        )

    def test_previous_edges(self):
        prev_idx = [4, 0, 1, 6, 7, 8, 9, 3]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.previous_half_edge(idx) == (pe, htag) for (idx, pe) in zip(self.hidx, prev_idx))
        )

    def test_twin_edges(self):
        b = self.graph.boundary_tag
        h = self.graph.half_edge_tag
        twin_idx = [0, 1, 3, 4, 5, 3, 9, 2]
        twin_tags = 5 * [b] + 2 * [h] + [b]

        self.assertTrue(
            all(self.graph.twin_half_edge(idx) == (tidx, tag) for idx, tidx, tag in zip(self.hidx, twin_idx, twin_tags))
        )
        self.assertTrue(all(
            self.graph.twin_half_edge((tidx, tag)) == (idx, h) for idx, tidx, tag in
            zip(self.hidx, twin_idx, twin_tags)))

    def test_source_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag
        source_vertices = [0, 1, 2, 3, 4, 5, 6, 5]

        self.assertTrue(
            all(self.graph.source_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, source_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, source_vertices))
        )

    def test_target_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag
        target_vertices = [1, 2, 3, 4, 5, 6, 5, 0]

        self.assertTrue(
            all(self.graph.target_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, target_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, target_vertices))
        )

    def test_boundary_source(self):
        source_vertices = [1, 2, 0, 3, 4, 5]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.source_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(source_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, btag)) for (idx, vidx) in enumerate(source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = [0, 1, 5, 2, 3, 4]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.target_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(target_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, btag)) for (idx, vidx) in enumerate(target_vertices))
        )

    def test_face_edges(self):
        face_idx = 8 * [1]

        self.assertTrue(
            all(self.graph.face(idx, tag=False) == fidx for idx, fidx in zip(self.hidx, face_idx))
        )
        ftag = self.graph.face_tag
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.has_edge((fidx, ftag), (idx, htag)) for idx, fidx in zip(self.hidx, face_idx))
        )

    def test_vertex_degree(self):
        degrees = [2, 2, 2, 2, 2, 3, 1]

        self.assertTrue(
            all(self.graph.vertex_degree(idx) == d for idx, d in enumerate(degrees))
        )

    def test_face_degree(self):
        self.assertEqual(self.graph.face_degree(1), 8)
"""

if __name__ == "__main__":
    unittest.main()
