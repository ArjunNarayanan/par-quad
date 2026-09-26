from src.tiler import Tiler
import unittest


def initialize_tri_quad_graph():
    face_loops = [
        [0, 1, 4, 3],
        [4, 1, 6],
        [3, 4, 6, 5],
        [1, 2, 7, 6]
    ]

    graph = Tiler.from_face_loops(face_loops)

    return graph


class TestHalfEdgeConnectivity(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_tri_quad_graph()

    def test_num_graph_nodes(self):
        num_halfedge = self.graph.number_of_half_edges()
        self.assertEqual(num_halfedge, 15)

        num_verts = self.graph.number_of_vertices()
        self.assertEqual(num_verts, 8)

        num_faces = self.graph.number_of_faces()
        self.assertEqual(num_faces, 4)

        num_boundary = self.graph._number_of_nodes("boundary")
        self.assertEqual(num_boundary, 7)

        total_num_nodes = self.graph.number_of_nodes()
        self.assertEqual(total_num_nodes, 34)

    def test_num_edges(self):
        num_source_edges = self.graph.number_of_edges_of_type("source")
        self.assertEqual(num_source_edges, 22)

        target_edges = self.graph.number_of_edges_of_type("target")
        self.assertEqual(target_edges, 22)

        face_edges = self.graph.number_of_edges_of_type("face")
        self.assertEqual(face_edges, 15)

        self.assertEqual(len(self.graph.edges()), 59)

    def test_next_edges(self):
        next_edges = [1, 2, 3, 0, 5, 6, 4, 8, 9, 10, 7, 12, 13, 14, 11]
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, self.graph.half_edge_tag) for idx, ne in
                enumerate(next_edges))
        )

    def test_previous_edges(self):
        previous_edges = [3, 0, 1, 2, 6, 4, 5, 10, 7, 8, 9, 14, 11, 12, 13]
        self.assertTrue(
            all(self.graph.previous_half_edge(idx) == (pe, self.graph.half_edge_tag) for idx, pe in
                enumerate(previous_edges))
        )

    def test_source_vertices(self):
        source_vertices = [0, 1, 4, 3, 4, 1, 6, 3, 4, 6, 5, 1, 2, 7, 6]
        self.assertTrue(
            all(self.graph.source_vertex(h) == (v, self.graph.vertex_tag) for (h, v) in
                enumerate(source_vertices))
        )

        self.assertTrue(
            all(
                self.graph.has_edge((v, self.graph.vertex_tag), (h, self.graph.half_edge_tag)) for (h, v) in
                enumerate(source_vertices)
            )
        )

    def test_target_vertices(self):
        vertex_ids = [1, 4, 3, 0, 1, 6, 4, 4, 6, 5, 3, 2, 7, 6, 1]
        target_halfedge_ids = range(self.graph.next_half_edge_index)

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
        source_ids = [1, 0, 5, 3, 2, 7, 6]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(all(self.graph.source_vertex((b, btag)) == (s, vtag) for b, s in zip(boundary_ids, source_ids)))
        self.assertTrue(all(self.graph.has_edge((s, vtag), (b, btag)) for s, b in zip(source_ids, boundary_ids)))

    def test_boundary_target(self):
        boundary_ids = range(self.graph.next_boundary_index)
        target_ids = [0, 3, 6, 5, 1, 2, 7]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(all(self.graph.target_vertex((b, btag)) == (s, vtag) for b, s in zip(boundary_ids, target_ids)))
        self.assertTrue(all(self.graph.has_edge((s, vtag), (b, btag)) for s, b in zip(target_ids, boundary_ids)))

    def test_face_edges(self):
        face_ids = 4 * [0] + 3 * [1] + 4 * [2] + 4 * [3]
        self.assertTrue(
            all(
                self.graph.face(halfedge) == (face, self.graph.face_tag) for halfedge, face in
                enumerate(face_ids)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((f, self.graph.face_tag), (h, self.graph.half_edge_tag))
                for (h, f) in enumerate(face_ids)
            )
        )

    def test_twin_edges(self):
        twin_id = [0, 4, 7, 1, 1, 14, 8, 2, 6, 2, 3, 4, 5, 6, 5]
        htag = self.graph.half_edge_tag
        btag = self.graph.boundary_tag
        twin_tags = [btag] + 2 * [htag] + [btag] + 5 * [htag] + 5 * [btag] + [htag]

        twin_nodes = [(node_id, node_tag) for node_id, node_tag in zip(twin_id, twin_tags)]
        self.assertTrue(
            all(self.graph.twin_half_edge(h) == t for h, t in enumerate(twin_nodes))
        )

        halfedge_id = [0, 3, 9, 10, 11, 12, 13]
        self.assertTrue(all(self.graph.twin_half_edge((b, btag)) == (h, htag) for b, h in
                            zip(range(self.graph.next_boundary_index), halfedge_id)))

    def test_vertex_degree(self):
        vertex_degree = [2, 4, 2, 3, 3, 2, 4, 2]
        self.assertTrue(
            all(self.graph.vertex_degree(vid) == vd for vid, vd in
                zip(range(self.graph.next_vertex_index), vertex_degree)))

    def test_face_degree(self):
        face_degrees = [4, 3, 4, 4]
        num_faces = self.graph.number_of_faces()
        self.assertEqual(len(face_degrees), num_faces)
        self.assertTrue(all(self.graph.face_degree(fidx) == fd for (fidx, fd) in zip(range(num_faces), face_degrees)))

    def test_on_boundary(self):
        t = True
        f = False
        on_boundary = [t, f, f, t, f, f, f, f, f, t, t, t, t, t, f]
        num_halfedges = self.graph.number_of_half_edges()
        self.assertEqual(num_halfedges, len(on_boundary))

        self.assertTrue(
            all(
                self.graph.half_edge_on_boundary(hidx) == flag for hidx, flag in enumerate(on_boundary)
            )
        )


if __name__ == "__main__":
    unittest.main()
