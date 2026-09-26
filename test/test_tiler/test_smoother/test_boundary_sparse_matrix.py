from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3],
        [3, 2, 4]
    ]
    vertex_coordinates = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, vertex_coordinates)
    return graph


def generate_coordinates():
    coords = [[0, 0],
              [1, 0],
              [1, 1],
              [0, 0.1],
              [0, 1]]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestSmoother(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.graph.set_user_defined_vertex(3, False)

    def test_laplace_sparse_operator(self):
        index2vertex = self.graph.vertex_list(tag=False)
        num_verts = len(index2vertex)
        vertex2index = dict(zip(index2vertex, range(num_verts)))
        matrix = self.graph._construct_sparse_vertex_laplace_operator(vertex2index)

        self.assertTrue(matrix.shape == (num_verts, num_verts))
        self.assertEqual(matrix.count_nonzero(), 6)

        index2vertex = self.graph.vertex_list(tag=False)
        num_verts = len(index2vertex)
        vertex2index = dict(zip(index2vertex, range(num_verts)))

        self.assertEqual(matrix[vertex2index[0], vertex2index[0]], 1)
        self.assertEqual(matrix[vertex2index[1], vertex2index[1]], 1)
        self.assertEqual(matrix[vertex2index[2], vertex2index[2]], 1)
        self.assertEqual(matrix[vertex2index[4], vertex2index[4]], 1)
        self.assertEqual(matrix[vertex2index[3], vertex2index[0]], 1)
        self.assertEqual(matrix[vertex2index[3], vertex2index[4]], 1)

    def test_degrees(self):
        index2vertex = self.graph.vertex_list(tag=False)
        num_verts = len(index2vertex)
        vertex2index = dict(zip(index2vertex, range(num_verts)))

        degrees = self.graph._get_vertex_degrees_for_smoothing(index2vertex)

        test_degrees = {0: 1, 1: 1, 2: 1, 3: 2, 4: 1}
        self.assertTrue(all(test_degrees[vertex2index[vidx]] == degrees[vertex2index[vidx]] for vidx in index2vertex))


if __name__ == "__main__":
    unittest.main()
