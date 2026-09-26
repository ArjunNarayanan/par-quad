from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 4],
        [0, 4, 3]
    ]
    graph = Tiler.from_face_loops(face_loops)
    return graph


class TestValidDelete(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        vidx = self.graph.source_vertex(4, tag=False)
        self.graph.user_defined_vertices.remove(vidx)

    def test_invalid_delete(self):
        self.assertTrue(self.graph.is_valid_delete_source_vertex(6))
        self.assertTrue(self.graph.is_valid_delete_source_vertex(4))


class TestDeleteVertex(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        vidx = self.graph.source_vertex(4, tag=False)
        self.graph.user_defined_vertices.remove(vidx)
        self.graph.delete_source_vertex(6)
        self.half_edges = [0, 1, 2, 4, 5, 7]

    def test_is_boundary_vertex(self):
        self.assertTrue(all(self.graph.is_boundary_vertex(vidx) for vidx in range(4)))
        self.assertTrue(4 not in self.graph.boundary_vertices)

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 4)
        self.assertEqual(self.graph.number_of_half_edges(), 6)
        self.assertEqual(self.graph.number_of_faces(), 2)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 4)
        self.assertEqual(self.graph.number_of_nodes(), 16)

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 10)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 10)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 6)

        self.assertEqual(self.graph.number_of_edges(), 26)

    def test_half_edges(self):
        self.assertTrue(len(self.graph.half_edges) == 10)
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all((hidx, htag) in self.graph.half_edges for hidx in self.half_edges)
        )
        btag = self.graph.boundary_tag
        self.assertTrue(
            all((hidx, btag) in self.graph.half_edges for hidx in range(4))
        )

    def test_next_edges(self):
        next_idx = [1, 2, 4, 0, 7, 5]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, htag) for idx, ne in zip(self.half_edges, next_idx))
        )

    def test_previous_edges(self):
        prev_edges = [4, 0, 1, 2, 7, 5]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.previous_half_edge(idx) == (ne, htag) for (idx, ne) in zip(self.half_edges, prev_edges))
        )

    def test_twin_edges(self):
        b = self.graph.boundary_tag
        h = self.graph.half_edge_tag
        twin_edges = [0, 1, 2, 5, 4, 3]
        twin_tags = [b, b, b, h, h, b]

        self.assertTrue(
            all(self.graph.twin_half_edge(idx) == (tidx, tag) for idx, tidx, tag in
                zip(self.half_edges, twin_edges, twin_tags))
        )

        self.assertTrue(
            all(self.graph.twin_half_edge((tidx, tag)) == (idx, h) for idx, tidx, tag in
                zip(self.half_edges, twin_edges, twin_tags))
        )

    def test_source_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag
        source = [0, 1, 2, 3, 0, 3]

        self.assertTrue(
            all(self.graph.source_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.half_edges, source)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.half_edges, source))
        )

    def test_target_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag
        target = [1, 2, 3, 0, 3, 0]

        self.assertTrue(
            all(self.graph.target_vertex(idx) == (sidx, vtag) for idx, sidx in zip(self.half_edges, target)))
        self.assertTrue(
            all(self.graph.has_edge((sidx, vtag), (idx, htag)) for idx, sidx in zip(self.half_edges, target))
        )

    def test_boundary_source(self):
        source_vertices = [1, 2, 3, 0]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.source_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = [0, 1, 2, 3]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.target_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(target_vertices))
        )

    def test_face_edges(self):
        faces = [0, 0, 0, 0, 1, 1]
        ftag = self.graph.face_tag

        self.assertTrue(
            all(self.graph.face(idx) == (fidx, ftag) for idx, fidx in zip(self.half_edges, faces))
        )
        ftag = self.graph.face_tag
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.has_edge((fidx, ftag), (idx, htag)) for idx, fidx in zip(self.half_edges, faces))
        )

    def test_vertex_degree(self):
        degrees = [3, 2, 2, 3]
        self.assertTrue(
            all(self.graph.vertex_degree(idx) == d for idx, d in enumerate(degrees))
        )

    def test_face_degree(self):
        degrees = [4, 2]
        self.assertTrue(
            all(self.graph.face_degree(fidx) == d for fidx, d in enumerate(degrees))
        )


if __name__ == "__main__":
    unittest.main()
