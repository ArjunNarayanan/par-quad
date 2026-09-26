from src.tiler import Tiler
import unittest


def initialize_tri_quad_graph():
    face_loops = [
        [0, 1, 4],
        [1, 2, 3, 4]
    ]

    graph = Tiler.from_face_loops(face_loops)

    return graph


class TestHalfEdgeConnectivity(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_tri_quad_graph()

    def test_num_graph_nodes(self):
        num_halfedge = self.graph.number_of_half_edges()
        self.assertEqual(num_halfedge, 7)

        num_verts = self.graph.number_of_vertices()
        self.assertEqual(num_verts, 5)

        num_faces = self.graph.number_of_faces()
        self.assertEqual(num_faces, 2)

        num_boundary = self.graph._number_of_nodes("boundary")
        self.assertEqual(num_boundary, 5)

        total_num_nodes = self.graph.number_of_nodes()
        self.assertEqual(total_num_nodes, 19)

    def test_num_edges(self):
        num_source_edges = self.graph.number_of_edges_of_type("source")
        self.assertEqual(num_source_edges, 12)

        num_target_edges = self.graph.number_of_edges_of_type("target")
        self.assertEqual(num_target_edges, 12)

        num_face_edges = self.graph.number_of_edges_of_type("face")
        self.assertEqual(num_face_edges, 7)

        self.assertEqual(len(self.graph.edges()), 31)

    def test_next_edges(self):
        next_edges = [1, 2, 0, 4, 5, 6, 3]
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, self.graph.half_edge_tag) for idx, ne in
                zip(range(7), next_edges))
        )

    def test_previous_edges(self):
        previous_edges = [2, 0, 1, 6, 3, 4, 5]
        self.assertTrue(
            all(self.graph.previous_half_edge(idx) == (pe, self.graph.half_edge_tag) for idx, pe in
                zip(range(7), previous_edges))
        )

    def test_source_vertices(self):
        source_vertices = [0, 1, 4, 1, 2, 3, 4]
        self.assertTrue(
            all(self.graph.source_vertex(h) == (v, self.graph.vertex_tag) for (h, v) in
                zip(range(7), source_vertices))
        )

        self.assertTrue(
            all(
                self.graph.has_edge((v, self.graph.vertex_tag), (h, self.graph.half_edge_tag)) for (v, h) in
                zip(source_vertices, range(7))
            )
        )

    def test_target_vertices(self):
        vertex_ids = [1, 4, 0, 2, 3, 4, 1]
        target_halfedge_ids = range(7)

        self.assertTrue(all(self.graph.target_vertex(h) == (v, self.graph.vertex_tag) for (h, v) in
                            zip(target_halfedge_ids, vertex_ids)))

        self.assertTrue(
            all(
                self.graph.has_edge((v, self.graph.vertex_tag), (h, self.graph.half_edge_tag)) for (v, h) in
                zip(vertex_ids, target_halfedge_ids)
            )
        )

    def test_boundary_source(self):
        boundary_ids = range(self.graph.next_boundary_index)
        source_ids = [1, 0, 2, 3, 4]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(all(self.graph.source_vertex((b, btag)) == (s, vtag) for b, s in zip(boundary_ids, source_ids)))
        self.assertTrue(all(self.graph.has_edge((s, vtag), (b, btag)) for s, b in zip(source_ids, boundary_ids)))

    def test_boundary_target(self):
        boundary_ids = range(self.graph.next_boundary_index)
        target_ids = [0, 4, 1, 2, 3]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(all(self.graph.target_vertex((b, btag)) == (s, vtag) for b, s in zip(boundary_ids, target_ids)))
        self.assertTrue(all(self.graph.has_edge((s, vtag), (b, btag)) for s, b in zip(target_ids, boundary_ids)))

    def test_face_edges(self):
        face_ids = 3 * [0] + 4 * [1]
        self.assertTrue(
            all(
                self.graph.face(half_edge) == (face, self.graph.face_tag) for half_edge, face in zip(range(7), face_ids)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((f, self.graph.face_tag), (h, self.graph.half_edge_tag))
                for (f, h) in zip(face_ids, range(7))
            )
        )

    def test_twin_edges(self):
        twin_id = [0, 6, 1, 2, 3, 4, 1]
        htag = self.graph.half_edge_tag
        btag = self.graph.boundary_tag
        twin_tags = [btag] + [htag] + 4 * [btag] + [htag]
        twin_nodes = [(node_id, node_tag) for node_id, node_tag in zip(twin_id, twin_tags)]
        self.assertTrue(
            all(self.graph.twin_half_edge(h) == t for h, t in zip(range(self.graph.next_half_edge_index), twin_nodes))
        )

        halfedge_id = [0, 2, 3, 4, 5]
        twin_tags = 5 * [htag]
        twin_nodes = [(h, t) for h, t in zip(halfedge_id, twin_tags)]
        self.assertTrue(all(self.graph.twin_half_edge((b, btag)) == t for (b, t) in zip(range(5), twin_nodes)))

    def test_vertex_degree(self):
        vertex_degree = [2, 3, 2, 2, 3]
        self.assertTrue(all(self.graph.vertex_degree(vid) == vd for vid, vd in zip(range(5), vertex_degree)))


if __name__ == "__main__":
    unittest.main()
