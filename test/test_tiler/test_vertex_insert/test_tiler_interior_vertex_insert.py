from src.tiler import Tiler
import unittest


def initialize_triangle_graph():
    face_loops = [
        [0, 1, 3],
        [1, 2, 3]
    ]
    vertex_coordinates = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, vertex_coordinates)
    return graph


def generate_coordinates():
    coords = [[-1, 0],
              [0, -1],
              [1, 0],
              [0, 1]]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestInsertVertex(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_triangle_graph()
        self.vidx = self.graph._insert_interior_vertex(1)

    def test_is_boundary_vertex(self):
        self.assertFalse(self.graph.is_boundary_vertex(4))

    def test_inserted_vertex_id(self):
        self.assertEqual(self.vidx, (4, self.graph.vertex_tag))

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_vertices(), 5)
        self.assertEqual(self.graph.number_of_half_edges(), 8)
        self.assertEqual(self.graph.number_of_faces(), 2)
        self.assertEqual(self.graph._number_of_nodes("boundary"), 4)
        self.assertEqual(self.graph.number_of_nodes(), 19)

    def test_vertex_coordinate(self):
        new_coord = self.graph.vertex_coordinate(4)
        self.assertTrue((new_coord == [0, 0]).all())

    def test_number_of_edges(self):
        self.assertEqual(self.graph.number_of_edges_of_type("source"), 12)
        self.assertEqual(self.graph.number_of_edges_of_type("target"), 12)
        self.assertEqual(self.graph.number_of_edges_of_type("face"), 8)

        self.assertEqual(self.graph.number_of_edges(), 32)

    def test_next_edges(self):
        next_edges = [1, 6, 0, 4, 7, 3, 2, 5]
        htag = self.graph.half_edge_tag
        num_edges = len(next_edges)
        self.assertTrue(
            all(self.graph.next_half_edge(idx) == (ne, htag) for (idx, ne) in zip(range(num_edges), next_edges))
        )

    def test_previous_edges(self):
        prev_edges = [2, 0, 6, 5, 3, 7, 1, 4]
        htag = self.graph.half_edge_tag
        num_edges = len(prev_edges)
        self.assertTrue(
            all(self.graph.previous_half_edge(idx) == (ne, htag) for (idx, ne) in zip(range(num_edges), prev_edges))
        )

    def test_twin_edges(self):
        btag = self.graph.boundary_tag
        htag = self.graph.half_edge_tag

        twin_idx = [0, 5, 1, 2, 3, 1, 7, 6]
        twin_tag = [btag, htag] + 3 * [btag] + 3 * [htag]
        num_edges = len(twin_idx)

        self.assertTrue(
            all(self.graph.twin_half_edge(idx) == (tidx, tag) for idx, tidx, tag in
                zip(range(num_edges), twin_idx, twin_tag))
        )

        twin_idx = [0, 2, 3, 4]
        num_edges = len(twin_idx)
        self.assertTrue(
            all(self.graph.twin_half_edge((idx, btag)) == (tidx, htag) for idx, tidx in zip(range(num_edges), twin_idx))
        )

    def test_source_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag

        source_vertices = [0, 1, 3, 1, 2, 4, 4, 3]
        num_edges = len(source_vertices)

        self.assertTrue(
            all(self.graph.source_vertex(idx) == (vidx, vtag) for idx, vidx in zip(range(num_edges), source_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, htag)) for idx, vidx in zip(range(num_edges), source_vertices))
        )

    def test_target_vertices(self):
        vtag = self.graph.vertex_tag
        htag = self.graph.half_edge_tag

        target_vertices = [1, 4, 0, 2, 3, 1, 3, 4]
        num_edges = len(target_vertices)

        self.assertTrue(
            all(self.graph.target_vertex(idx) == (vidx, vtag) for idx, vidx in zip(range(num_edges), target_vertices))
        )
        self.assertTrue(
            all(self.graph.has_edge((vidx, vtag), (idx, htag)) for idx, vidx in zip(range(num_edges), target_vertices))
        )

    def test_boundary_source(self):
        source_vertices = [1, 0, 2, 3]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag
        num_edges = len(source_vertices)

        self.assertTrue(
            all(self.graph.source_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in
                zip(range(num_edges), source_vertices))
        )

    def test_boundary_target(self):
        target_vertices = [0, 3, 1, 2]
        btag = self.graph.boundary_tag
        vtag = self.graph.vertex_tag
        num_edges = len(target_vertices)

        self.assertTrue(
            all(self.graph.target_vertex((idx, btag)) == (vidx, vtag) for (idx, vidx) in
                zip(range(num_edges), target_vertices))
        )

    def test_face_edges(self):
        face_indices = [0, 0, 0, 1, 1, 1, 0, 1]
        ftag = self.graph.face_tag
        htag = self.graph.half_edge_tag
        num_edges = len(face_indices)

        self.assertTrue(
            all(self.graph.face(idx, tag=False) == fidx for idx, fidx in zip(range(num_edges), face_indices))
        )
        self.assertTrue(
            all(self.graph.has_edge((fidx, ftag), (hidx, htag)) for hidx, fidx in zip(range(num_edges), face_indices))
        )

    def test_vertex_degree(self):
        degrees = [2, 3, 2, 3]

        self.assertTrue(
            all(self.graph.vertex_degree(idx) == d for idx, d in enumerate(degrees))
        )

    def test_face_degree(self):
        self.assertTrue(all(self.graph.face_degree(idx) == 4 for idx in range(2)))


if __name__ == "__main__":
    unittest.main()
