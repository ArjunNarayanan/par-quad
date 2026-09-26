from src.tiler import Tiler
import unittest


def initialize_hex_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5],
    ]

    graph = Tiler.from_face_loops(face_loops)

    return graph


class TestInsertHalfedge(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_hex_graph()
        self.graph.insert_half_edge(0, 2)
        self.graph.insert_half_edge(1, 1)
        self.graph.insert_half_edge(4, 1)

    def test_num_graph_nodes(self):
        num_half_edge = self.graph.number_of_half_edges()
        self.assertEqual(num_half_edge, 12)

        num_verts = self.graph.number_of_vertices()
        self.assertEqual(num_verts, 6)

        num_faces = self.graph.number_of_faces()
        self.assertEqual(num_faces, 4)

        num_boundary = self.graph._number_of_nodes("boundary")
        self.assertEqual(num_boundary, 6)

        total_num_nodes = self.graph.number_of_nodes()
        self.assertEqual(total_num_nodes, 28)

    def test_num_edges(self):
        source_edges = self.graph.number_of_edges_of_type("source")
        self.assertEqual(source_edges, 18)

        target_edges = self.graph.number_of_edges_of_type("target")
        self.assertEqual(target_edges, 18)

        face_edges = self.graph.number_of_edges_of_type("face")
        self.assertEqual(face_edges, 12)

        self.assertEqual(len(self.graph.edges()), 48)

    def test_next_edges(self):
        next_edges = [9, 2, 8, 11, 5, 10, 0, 3, 1, 6, 4, 7]
        htag = self.graph.half_edge_tag
        num_half_edges = self.graph.number_of_half_edges()

        self.assertTrue(
            all(self.graph.next_half_edge(hidx) == (n, htag) for (hidx, n) in zip(range(num_half_edges), next_edges))
        )

    def test_previous_edges(self):
        next_edges = [9, 2, 8, 11, 5, 10, 0, 3, 1, 6, 4, 7]
        htag = self.graph.half_edge_tag
        num_half_edges = self.graph.number_of_half_edges()

        self.assertTrue(
            all(self.graph.previous_half_edge(n) == (hidx, htag) for (n, hidx) in zip(next_edges, range(num_half_edges)))
        )

    def test_source_vertex(self):
        source_verts = [0, 1, 2, 3, 4, 5, 3, 0, 3, 1, 0, 4]
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag
        num_half_edges = self.graph.number_of_half_edges()

        self.assertEqual(num_half_edges, len(source_verts))
        self.assertTrue(
            all(self.graph.source_vertex(hidx) == (v, vtag) for (hidx, v) in zip(range(num_half_edges), source_verts))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for (hidx, vidx) in
                zip(range(num_half_edges), source_verts))
        )

    def test_target_vertex(self):
        target_verts = [1, 2, 3, 4, 5, 0, 0, 3, 1, 3, 4, 0]
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag
        num_half_edges = self.graph.number_of_half_edges()

        self.assertEqual(num_half_edges, len(target_verts))
        self.assertTrue(
            all(self.graph.target_vertex(hidx) == (v, vtag) for (hidx, v) in zip(range(num_half_edges), target_verts))
        )

        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for (hidx, vidx) in
                zip(range(num_half_edges), target_verts))
        )

    def test_boundary_source(self):
        source_verts = [1, 2, 3, 4, 5, 0]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag
        num_boundary = self.graph._number_of_nodes("boundary")

        self.assertEqual(num_boundary, len(source_verts))
        self.assertTrue(
            all(
                self.graph.source_vertex((bidx, btag)) == (vidx, vtag) for (bidx, vidx) in
                zip(range(num_boundary), source_verts)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((vidx, vtag), (bidx, btag)) for (bidx, vidx) in
                zip(range(num_boundary), source_verts)
            )
        )

    def test_boundary_target(self):
        target_verts = [0, 1, 2, 3, 4, 5]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag
        num_boundary = self.graph._number_of_nodes("boundary")

        self.assertEqual(num_boundary, len(target_verts))
        self.assertTrue(
            all(
                self.graph.target_vertex((bidx, btag)) == (vidx, vtag) for (bidx, vidx) in
                zip(range(num_boundary), target_verts)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((vidx, vtag), (bidx, btag)) for (bidx, vidx) in
                zip(range(num_boundary), target_verts)
            )
        )

    def test_face_edges(self):
        face_ids = [1, 2, 2, 0, 3, 3, 1, 0, 2, 1, 3, 0]
        ftag = self.graph.face_tag
        htag = self.graph.half_edge_tag
        num_half_edges = self.graph.number_of_half_edges()

        self.assertEqual(num_half_edges, len(face_ids))
        self.assertTrue(
            all(
                self.graph.face(hidx) == (f, ftag) for (hidx, f) in zip(range(num_half_edges), face_ids)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((fidx, ftag), (hidx, htag)) for (hidx, fidx) in zip(range(num_half_edges), face_ids)
            )
        )

    def test_twin_edge(self):
        twin_idx = [0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 11, 10]
        h, b = self.graph.half_edge_tag, self.graph.boundary_tag
        tags = [b, b, b, b, b, b, h, h, h, h, h, h]
        num_half_edges = self.graph.number_of_half_edges()

        self.assertEqual(num_half_edges, len(twin_idx))
        self.assertEqual(num_half_edges, len(tags))

        self.assertTrue(
            all(
                self.graph.twin_half_edge(hidx) == (tidx, tg) for (hidx, tidx, tg) in
                zip(range(num_half_edges), twin_idx, tags)
            )
        )

        self.assertTrue(
            all(
                self.graph.twin_half_edge((tidx, tag)) == (hidx, h) for (hidx, tidx, tag) in
                zip(range(num_half_edges), twin_idx, tags)
            )
        )

    def test_vertex_degree(self):
        vertex_degree = [4, 3, 2, 4, 3, 2]
        self.assertTrue(
            all(
                self.graph.vertex_degree(vid) == vd for vid, vd in
                zip(range(self.graph.number_of_vertices()), vertex_degree)
            )
        )

    def test_face_degree(self):
        self.assertTrue(all(self.graph.face_degree(fidx) == 3 for fidx in range(self.graph.number_of_faces())))


# graph = initialize_hex_graph()
if __name__ == "__main__":
    unittest.main()
