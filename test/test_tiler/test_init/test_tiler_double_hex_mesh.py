from src.tiler import Tiler
import unittest


def initialize_double_hex_graph():
    face_loops = [
        [0, 1, 2, 7, 8, 9],
        [2, 3, 4, 5, 6, 7]
    ]

    graph = Tiler.from_face_loops(face_loops)

    return graph


class TestHalfEdgeConnectivity(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_double_hex_graph()

    def test_num_edges(self):
        self.assertEqual(len(self.graph.edges()), 56)

        source_edges = self.graph.number_of_edges_of_type("source")
        self.assertEqual(source_edges, 22)

        target_edges = self.graph.number_of_edges_of_type("target")
        self.assertEqual(target_edges, 22)

        face_edges = self.graph.number_of_edges_of_type("face")
        self.assertEqual(face_edges, 12)

    def test_num_graph_nodes(self):
        num_halfedge = self.graph.number_of_half_edges()
        self.assertEqual(num_halfedge, 12)

        num_verts = self.graph.number_of_vertices()
        self.assertEqual(num_verts, 10)

        num_faces = self.graph.number_of_faces()
        self.assertEqual(num_faces, 2)

        num_boundary = self.graph._number_of_nodes("boundary")
        self.assertEqual(num_boundary, 10)

        total_num_nodes = self.graph.number_of_nodes()
        self.assertEqual(total_num_nodes, 34)

    def test_next_edges(self):
        next_edges = list(range(1, 6)) + [0]
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, self.graph.half_edge_tag) for idx, ne in enumerate(next_edges))
        )
        next_edges = list(range(7, 12)) + [6]
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, self.graph.half_edge_tag) for idx, ne in
                zip(range(6, 12), next_edges))
        )

    def test_previous_edges(self):
        next_edges = list(range(1, 6)) + [0]
        self.assertTrue(
            all(self.graph.previous_half_edge(ne) == (idx, self.graph.half_edge_tag) for idx, ne in
                zip(range(6), next_edges))
        )
        next_edges = list(range(7, 12)) + [6]
        self.assertTrue(
            all(
                self.graph.previous_half_edge(ne) == (idx, self.graph.half_edge_tag) for idx, ne in
                zip(range(6, 12), next_edges)
            )
        )

    def test_source_vertices(self):
        source_vertices = [0, 1, 2, 7, 8, 9, 2, 3, 4, 5, 6, 7]
        self.assertTrue(
            all(self.graph.source_vertex(h) == (v, self.graph.vertex_tag) for (h, v) in
                zip(range(12), source_vertices))
        )

        vertex_ids = [0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9]
        source_halfedge_ids = [0, 1, 2, 6, 7, 8, 9, 10, 11, 3, 4, 5]
        self.assertTrue(
            all(
                self.graph.has_edge((v, self.graph.vertex_tag), (h, self.graph.half_edge_tag)) for (v, h) in
                zip(vertex_ids, source_halfedge_ids)
            )
        )

    def test_target_vertices(self):
        vertex_ids = [0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9]
        target_halfedge_ids = [5, 0, 1, 11, 6, 7, 8, 9, 10, 2, 3, 4]

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
        source_ids = [1, 2, 8, 9, 0, 3, 4, 5, 6, 7]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(all(self.graph.source_vertex((b, btag)) == (s, vtag) for b, s in zip(boundary_ids, source_ids)))
        self.assertTrue(all(self.graph.has_edge((s, vtag), (b, btag)) for s, b in zip(source_ids, boundary_ids)))

    def test_boundary_target(self):
        boundary_ids = range(self.graph.next_boundary_index)
        target_ids = [0, 1, 7, 8, 9, 2, 3, 4, 5, 6]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(all(self.graph.target_vertex((b, btag)) == (s, vtag) for b, s in zip(boundary_ids, target_ids)))
        self.assertTrue(all(self.graph.has_edge((s, vtag), (b, btag)) for s, b in zip(target_ids, boundary_ids)))

    def test_face_edges(self):
        face_ids = 6 * [0] + 6 * [1]
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
        twin_id = [0, 1, 11, 2, 3, 4, 5, 6, 7, 8, 9, 2]
        htag = self.graph.half_edge_tag
        btag = self.graph.boundary_tag
        twin_tags = 2 * [btag] + [htag] + 8 * [btag] + [htag]
        twin_nodes = [(node_id, node_tag) for node_id, node_tag in zip(twin_id, twin_tags)]
        self.assertTrue(
            all(self.graph.twin_half_edge(h) == t for h, t in enumerate(twin_nodes))
        )

        halfedge_id = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertTrue(all(self.graph.twin_half_edge((b, btag)) == (h, htag) for b, h in
                            enumerate(halfedge_id)))

    def test_vertex_degree(self):
        vertex_degree = [2, 2, 3, 2, 2, 2, 2, 3, 2, 2]
        self.assertTrue(
            all(self.graph.vertex_degree(vid) == vd for vid, vd in
                enumerate(vertex_degree)))

    def test_face_degree(self):
        self.assertTrue(all(self.graph.face_degree(fidx) == 6 for fidx in range(self.graph.number_of_faces())))

    def test_face_half_edges(self):
        halfedges = self.graph.face_half_edges(0)
        halfedges.sort()
        htag = self.graph.half_edge_tag
        test_halfedges = [(idx, htag) for idx in range(6)]
        self.assertEqual(len(halfedges), 6)
        self.assertTrue(all(he == te for he, te in zip(halfedges, test_halfedges)))

        halfedges = self.graph.face_half_edges(1)
        halfedges.sort()
        test_halfedges = [(idx, htag) for idx in range(6, 12)]
        self.assertEqual(len(halfedges), 6)
        self.assertTrue(all(he == te for he, te in zip(halfedges, test_halfedges)))

    def test_face_loop(self):
        face_loop = self.graph.generate_half_edge_face_loop(3)
        htag = self.graph.half_edge_tag
        test_loop = [(idx, htag) for idx in [3, 4, 5, 0, 1, 2]]
        self.assertTrue(all(he == te for he, te in zip(face_loop, test_loop)))

        face_loop = self.graph.generate_half_edge_face_loop(11)
        test_loop = [(idx, htag) for idx in [11, 6, 7, 8, 9, 10]]
        self.assertTrue(all(he == te for he, te in zip(face_loop, test_loop)))


if __name__ == "__main__":
    unittest.main()
