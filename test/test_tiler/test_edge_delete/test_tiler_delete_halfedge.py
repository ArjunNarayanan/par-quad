from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 3],
        [1, 2, 3]
    ]
    graph = Tiler.from_face_loops(face_loops)
    return graph


class TestDeleteEdge(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.graph.delete_half_edge(1)
        self.hidx = [0, 3, 4, 2]

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 4)
        self.assertEqual(self.graph.number_of_half_edges(), 4)
        self.assertEqual(self.graph.number_of_faces(), 1)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 4)
        self.assertEqual(self.graph.number_of_nodes(), 13)

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 8)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 8)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 4)

        self.assertEqual(self.graph.number_of_edges(), 20)

    def test_half_edges(self):
        self.assertTrue(len(self.graph.half_edges) == 8)
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all((hidx, htag) in self.graph.half_edges for hidx in self.hidx)
        )
        btag = self.graph.boundary_tag
        self.assertTrue(
            all((hidx, btag) in self.graph.half_edges for hidx in range(4))
        )

    def test_next_edges(self):
        next_idx = [3, 4, 2, 0]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, htag) for idx, ne in zip(self.hidx, next_idx))
        )

    def test_previous_edges(self):
        next_idx = [3, 4, 2, 0]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.previous_half_edge(ne) == (idx, htag) for (idx, ne) in zip(self.hidx, next_idx))
        )

    def test_twin_edges(self):
        b = self.graph.boundary_tag
        h = self.graph.half_edge_tag
        twin_idx = [0, 2, 3, 1]
        twin_tags = 4 * [b]

        self.assertTrue(
            all(self.graph.twin_half_edge(idx) == (tidx, tag) for idx, tidx, tag in zip(self.hidx, twin_idx, twin_tags))
        )
        self.assertTrue(all(
            self.graph.twin_half_edge((tidx, tag)) == (idx, h) for idx, tidx, tag in
            zip(self.hidx, twin_idx, twin_tags)))

    def test_source_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag
        source_vertices = [0, 1, 2, 3]

        self.assertTrue(
            all(self.graph.source_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, source_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, source_vertices))
        )

    def test_target_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag
        target_vertices = [1, 2, 3, 0]

        self.assertTrue(
            all(self.graph.target_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, target_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, target_vertices))
        )

    def test_boundary_source(self):
        source_vertices = [1, 0, 2, 3]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.source_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(source_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, btag)) for (idx, vidx) in enumerate(source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = [0, 3, 1, 2]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.target_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(target_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, btag)) for (idx, vidx) in enumerate(target_vertices))
        )

    def test_face_edges(self):
        face_idx = 4 * [0]

        self.assertTrue(
            all(self.graph.face(idx, tag=False) == fidx for idx, fidx in zip(self.hidx, face_idx))
        )
        ftag = self.graph.face_tag
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.has_edge((fidx, ftag), (idx, htag)) for idx, fidx in zip(self.hidx, face_idx))
        )

    def test_vertex_degree(self):
        degrees = 4 * [2]

        self.assertTrue(
            all(self.graph.vertex_degree(idx) == d for idx, d in zip(range(4), degrees))
        )

    def test_face_degree(self):
        self.assertEqual(self.graph.face_degree(0), 4)


if __name__ == "__main__":
    unittest.main()
