from src.tiler import Tiler
import unittest


def initialize_triangle_graph():
    face_loops = [
        [0, 1, 2],
    ]
    vertex_coordinates = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, vertex_coordinates)
    graph.insert_vertex(1)
    return graph


def generate_coordinates():
    coords = [[-1, 0],
              [0, -1],
              [0, 1]]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestDeleteVertex(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_triangle_graph()
        self.graph._delete_boundary_vertex(3)

    def test_is_boundary_vertex(self):
        self.assertTrue(3 not in self.graph.boundary_vertices)

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 3)
        self.assertEqual(self.graph.number_of_half_edges(), 3)
        self.assertEqual(self.graph.number_of_faces(), 1)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 3)
        self.assertEqual(self.graph.number_of_nodes(), 10)

    def test_vertex_coordinate(self):
        self.assertFalse(3 in self.graph.vertex_coordinates)
        self.assertTrue(all(idx in self.graph.vertex_coordinates for idx in range(3)))

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 6)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 6)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 3)

        self.assertEqual(self.graph.number_of_edges(), 15)

    def test_half_edges(self):
        self.assertTrue(len(self.graph.half_edges) == 6)
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all((hidx, htag) in self.graph.half_edges for hidx in range(3))
        )
        btag = self.graph.boundary_tag
        self.assertTrue(
            all((hidx, btag) in self.graph.half_edges for hidx in range(3))
        )

    def test_next_edges(self):
        next_idx = [1, 2, 0]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, htag) for idx, ne in enumerate(next_idx))
        )

    def test_previous_edges(self):
        prev_edges = [2, 0, 1]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.previous_half_edge(idx) == (ne, htag) for (idx, ne) in enumerate(prev_edges))
        )

    def test_twin_edges(self):
        btag = self.graph.boundary_tag
        htag = self.graph.half_edge_tag

        self.assertTrue(
            all(self.graph.twin_half_edge(idx) == (idx, btag) for idx in range(3))
        )
        self.assertTrue(all(self.graph.twin_half_edge((idx, btag)) == (idx, htag) for idx in range(3)))

    def test_source_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag

        self.assertTrue(all(self.graph.source_vertex(idx) == (idx, vtag) for idx in range(3)))
        self.assertTrue(
            all(self.graph.has_edge((idx, vtag), (idx, htag)) for idx in range(3))
        )

    def test_target_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag

        target_vertices = [1, 2, 0]
        self.assertTrue(
            all(self.graph.target_vertex(idx) == (vidx, vtag) for idx, vidx in enumerate(target_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, htag)) for idx, vidx in enumerate(target_vertices))
        )

    def test_boundary_source(self):
        source_vertices = [1, 2, 0]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.source_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = [0, 1, 2]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.target_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in enumerate(target_vertices))
        )

    def test_face_edges(self):
        self.assertTrue(
            all(self.graph.face(idx, tag=False) == 0 for idx in range(3))
        )
        ftag = self.graph.face_tag
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.has_edge((0, ftag), (idx, htag)) for idx in range(3))
        )

    def test_vertex_degree(self):
        self.assertTrue(
            all(self.graph.vertex_degree(idx) == 2 for idx in range(3))
        )

    def test_face_degree(self):
        self.assertEqual(self.graph.face_degree(0), 3)


if __name__ == "__main__":
    unittest.main()
