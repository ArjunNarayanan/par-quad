from src.tiler import Tiler
import unittest


def initialize_triangle_graph():
    face_loops = [
        [0, 1, 2],
    ]
    vertex_coordinates = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, vertex_coordinates)
    return graph


def generate_coordinates():
    coords = [[0, 0],
              [1, 0],
              [0, 1], ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestInsertVertex(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_triangle_graph()
        vidx = self.graph._insert_boundary_vertex(0)
        self.inserted_vertex = vidx

    def test_inserted_vertex_id(self):
        self.assertEqual(self.inserted_vertex, (3, self.graph.vertex_tag))

    def test_is_boundary_vertex(self):
        self.assertTrue(self.graph.is_boundary_vertex(3))

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 4)
        self.assertEqual(self.graph.number_of_half_edges(), 4)
        self.assertEqual(self.graph.number_of_faces(), 1)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 4)
        self.assertEqual(self.graph.number_of_nodes(), 13)

    def test_vertex_coordinate(self):
        new_coord = self.graph.vertex_coordinate(3)
        self.assertTrue((new_coord == [0.5, 0]).all())

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 8)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 8)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 4)

        self.assertEqual(self.graph.number_of_edges(), 20)

    def test_next_edges(self):
        next_edges = [3, 2, 0, 1]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, htag) for (idx, ne) in zip(range(4), next_edges))
        )

    def test_previous_edges(self):
        prev_edges = [2, 3, 1, 0]
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.previous_half_edge(idx) == (ne, htag) for (idx, ne) in zip(range(4), prev_edges))
        )

    def test_twin_edges(self):
        btag = self.graph.boundary_tag
        htag = self.graph.half_edge_tag

        self.assertTrue(
            all(self.graph.twin_half_edge(idx) == (idx, btag) for idx in range(4))
        )
        self.assertTrue(all(self.graph.twin_half_edge((idx, btag)) == (idx, htag) for idx in range(4)))

    def test_source_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag

        self.assertTrue(all(self.graph.source_vertex(idx) == (idx, vtag) for idx in range(4)))
        self.assertTrue(
            all(self.graph.has_edge((idx, vtag), (idx, htag)) for idx in range(4))
        )

    def test_target_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag

        target_vertices = [3, 2, 0, 1]
        self.assertTrue(
            all(self.graph.target_vertex(idx) == (vidx, vtag) for idx, vidx in zip(range(4), target_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, htag)) for idx, vidx in zip(range(4), target_vertices))
        )

    def test_boundary_source(self):
        source_vertices = [3, 2, 0, 1]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.source_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in zip(range(4), source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = [0, 1, 2, 3]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag

        self.assertTrue(
            all(self.graph.target_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in zip(range(4), target_vertices))
        )

    def test_face_edges(self):
        self.assertTrue(
            all(self.graph.face(idx, tag=False) == 0 for idx in range(4))
        )
        ftag = self.graph.face_tag
        htag = self.graph.half_edge_tag
        self.assertTrue(
            all(self.graph.has_edge((0, ftag), (idx, htag), key="face") for idx in range(4))
        )

    def test_vertex_degree(self):
        self.assertTrue(
            all(self.graph.vertex_degree(idx) == 2 for idx in range(4))
        )

    def test_face_degree(self):
        self.assertEqual(self.graph.face_degree(0), 4)


if __name__ == "__main__":
    unittest.main()