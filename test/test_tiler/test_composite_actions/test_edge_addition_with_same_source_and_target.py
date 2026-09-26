from src.tiler import Tiler
import unittest


class TestInvalidInsert(unittest.TestCase):
    def setUp(self) -> None:
        graph = Tiler.from_face_loops([[0, 1, 2, 3]])
        graph.insert_half_edge(0, 1)
        graph.insert_vertex(4)
        graph.insert_vertex(6)
        graph.insert_half_edge(0, 2)
        graph.delete_half_edge(4)
        graph.insert_half_edge(9, 2)
        self.graph = graph
        self.hidx = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 6)
        self.assertEqual(self.graph.number_of_half_edges(), 12)
        self.assertEqual(self.graph.number_of_faces(), 3)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 4)
        self.assertEqual(self.graph.number_of_nodes(), 25)

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 16)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 16)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 12)

        self.assertEqual(self.graph.number_of_edges(), 44)

    def test_half_edges(self):
        self.assertEqual(
            len(self.graph.half_edges), self.graph.number_of_half_edges() + self.graph._number_of_nodes("boundary")
        )
        self.assertTrue(
            all(h in self.graph.half_edges for h in self.graph.half_edge_list())
        )
        self.assertTrue(
            all(h in self.graph.half_edges for h in self.graph._node_list_by_type("boundary", tag=True))
        )

    def test_next_edges(self):
        next_idx = [1, 2, 3, 13, 8, 10, 11, 7, 12, 6, 9, 0]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, htag) for idx, ne in zip(self.hidx, next_idx))
        )

    def test_previous_edges(self):
        next_idx = [1, 2, 3, 13, 8, 10, 11, 7, 12, 6, 9, 0]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.previous_half_edge(ne) == (idx, htag) for idx, ne in zip(self.hidx, next_idx))
        )

    def test_twin_edges(self):
        b = self.graph.boundary_tag
        h = self.graph.half_edge_tag
        twin_idx = [0, 1, 2, 3, 7, 6, 9, 8, 11, 10, 13, 12]
        twin_tags = [b, b, b, b, h, h, h, h, h, h, h, h]

        self.assertTrue(
            all(self.graph.twin_half_edge(idx) == (tidx, tag) for idx, tidx, tag in zip(self.hidx, twin_idx, twin_tags))
        )
        self.assertTrue(all(
            self.graph.twin_half_edge((tidx, tag)) == (idx, h) for idx, tidx, tag in
            zip(self.hidx, twin_idx, twin_tags)))

    def test_source_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag
        source_vertices = [0, 1, 2, 3, 4, 5, 5, 0, 4, 0, 0, 0]

        self.assertTrue(
            all(self.graph.source_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, source_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, source_vertices))
        )

    def test_target_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag
        target_vertices = [1, 2, 3, 0, 5, 4, 0, 5, 0, 4, 0, 0]

        self.assertTrue(
            all(self.graph.target_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.hidx, target_vertices)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.hidx, target_vertices))
        )

    def test_boundary_source(self):
        source_vertices = [1, 2, 3, 0]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.source_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(source_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, btag)) for (idx, vidx) in enumerate(source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = range(4)
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.target_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(target_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, btag)) for (idx, vidx) in enumerate(target_vertices))
        )

    def test_face_edges(self):
        face_idx = [2, 2, 2, 2, 1, 3, 1, 3, 3, 1, 3, 2]

        self.assertTrue(
            all(self.graph.face(idx, tag=False) == fidx for idx, fidx in zip(self.hidx, face_idx))
        )
        ftag = self.graph.face_tag
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.has_edge((fidx, ftag), (idx, htag)) for idx, fidx in zip(self.hidx, face_idx))
        )

    def test_vertex_degree(self):
        degrees = [6, 2, 2, 2, 2, 2]

        self.assertTrue(
            all(self.graph.vertex_degree(idx) == d for idx, d in zip(range(7), degrees))
        )

    def test_face_degree(self):
        degrees = [3, 5, 4]
        self.assertTrue(
            all(self.graph.face_degree(fidx) == d for fidx, d in zip(range(1, 4), degrees))
        )


if __name__ == "__main__":
    unittest.main()
