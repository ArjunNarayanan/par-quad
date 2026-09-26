from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 3, 2],
        [2, 3, 5, 4]
    ]
    coords = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 2],
        [1, 2]
    ]
    coords = dict(zip(range(6), coords))
    graph = Tiler.from_face_loops(face_loops, vertex_coordinates=coords)

    return graph


class TestGlobalSplit(unittest.TestCase):
    def setUp(self):
        self.graph = initialize_graph()
        self.graph.global_split_to_boundary(0, 3)

    def test_num_graph_nodes(self):
        num_halfedge = self.graph.number_of_half_edges()
        self.assertEqual(num_halfedge, 16)

        num_verts = self.graph.number_of_vertices()
        self.assertEqual(num_verts, 9)

        num_faces = self.graph.number_of_faces()
        self.assertEqual(num_faces, 4)

        num_boundary = self.graph._number_of_nodes("boundary")
        self.assertEqual(num_boundary, 8)

        total_num_nodes = self.graph.number_of_nodes()
        self.assertEqual(total_num_nodes, 37)

    def test_num_edges(self):
        source_edges = self.graph.number_of_edges_of_type("source")
        self.assertEqual(source_edges, 24)

        target_edges = self.graph.number_of_edges_of_type("target")
        self.assertEqual(target_edges, 24)

        face_edges = self.graph.number_of_edges_of_type("face")
        self.assertEqual(face_edges, 16)

        self.assertEqual(len(self.graph.edges()), 64)

    def test_next_prev_edges(self):
        next_edges = [12, 2, 11, 0, 5, 6, 14, 10, 1, 3, 15, 8, 9, 7, 4, 13]
        htag = self.graph.half_edge_tag

        self.assertTrue(self.graph.number_of_half_edges() == len(next_edges))
        self.assertTrue(all(self.graph.next_half_edge(hidx) == (n, htag) for (hidx, n) in enumerate(next_edges)))
        self.assertTrue(
            all(self.graph.previous_half_edge((n, htag)) == (hidx, htag) for (hidx, n) in enumerate(next_edges)))

    def test_source_vertex(self):
        source_verts = [0, 1, 3, 2, 7, 3, 5, 4, 6, 7, 2, 7, 6, 8, 8, 7]
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag

        self.assertTrue(self.graph.number_of_half_edges() == len(source_verts))
        self.assertTrue(
            all(self.graph.source_vertex(hidx) == (v, vtag) for (hidx, v) in enumerate(source_verts))
        )

        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for (hidx, vidx) in enumerate(source_verts))
        )

    def test_target_vertex(self):
        target_verts = [6, 3, 7, 0, 3, 5, 8, 2, 1, 2, 7, 6, 7, 4, 7, 8]
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag

        self.assertTrue(self.graph.number_of_half_edges() == len(target_verts))
        self.assertTrue(
            all(self.graph.target_vertex(hidx) == (v, vtag) for (hidx, v) in enumerate(target_verts))
        )

        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (hidx, htag)) for (hidx, vidx) in enumerate(target_verts))
        )

    def test_boundary_source(self):
        source_verts = [6, 3, 0, 5, 8, 2, 1, 4]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(self.graph._number_of_nodes("boundary"), len(source_verts))
        self.assertTrue(all(
            self.graph.source_vertex((bidx, btag)) == (vidx, vtag) for (bidx, vidx) in enumerate(source_verts)))

        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (bidx, btag)) for (bidx, vidx) in enumerate(source_verts)))

    def test_boundary_target(self):
        target_verts = [0, 1, 2, 3, 5, 4, 6, 8]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(self.graph._number_of_nodes("boundary"), len(target_verts))
        self.assertTrue(
            all(
                self.graph.target_vertex((bidx, btag)) == (vidx, vtag) for (bidx, vidx) in enumerate(target_verts)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((vidx, vtag), (bidx, btag)) for (bidx, vidx) in enumerate(target_verts)
            )
        )

    def test_face_edges(self):
        face_ids = [0, 2, 2, 0, 3, 3, 3, 1, 2, 0, 1, 2, 0, 1, 3, 1]
        ftag = self.graph.face_tag
        htag = self.graph.half_edge_tag

        self.assertEqual(self.graph.number_of_faces(), 4)
        self.assertTrue(
            all(
                self.graph.face(hidx) == (f, ftag) for (hidx, f) in enumerate(face_ids)
            )
        )

        self.assertTrue(
            all(
                self.graph.has_edge((fidx, ftag), (hidx, htag)) for (hidx, fidx) in enumerate(face_ids)
            )
        )

    def test_twin_edge(self):
        twin_idx = [0, 1, 4, 2, 2, 3, 4, 5, 6, 10, 9, 12, 11, 7, 15, 14]
        h, b = self.graph.half_edge_tag, self.graph.boundary_tag
        tags = [b, b, h, b, h, b, b, b, b, h, h, h, h, b, h, h]

        num_half_edge = self.graph.number_of_half_edges()
        self.assertEqual(num_half_edge, len(twin_idx))
        self.assertTrue(
            all(
                self.graph.twin_half_edge(hidx) == (tidx, tg) for (hidx, tidx, tg) in
                zip(range(num_half_edge), twin_idx, tags)
            )
        )

        self.assertTrue(
            all(
                self.graph.twin_half_edge((tidx, tag)) == (hidx, h) for (hidx, tidx, tag) in
                zip(range(num_half_edge), twin_idx, tags)
            )
        )

    def test_vertex_degree(self):
        vertex_degrees = [2, 2, 3, 3, 2, 2, 3, 4, 3]
        num_vertices = self.graph.number_of_vertices()
        self.assertEqual(num_vertices, len(vertex_degrees))
        self.assertTrue(all(self.graph.vertex_degree(vidx) == vd for (vidx, vd) in
                            zip(range(self.graph.number_of_vertices()), vertex_degrees)))

    def test_face_degree(self):
        face_degrees = [4, 4, 4, 4]
        num_faces = self.graph.number_of_faces()
        self.assertEqual(len(face_degrees), num_faces)
        self.assertTrue(all(self.graph.face_degree(fidx) == fd for (fidx, fd) in zip(range(num_faces), face_degrees)))


if __name__ == "__main__":
    unittest.main()
