import numpy as np
from src.tiler import Tiler
import unittest


def initialize_graph():
    face_loops = [
        [0, 1, 5, 4],
        [1, 2, 5],
        [4, 5, 2, 3],
        [4, 3, 0]
    ]
    vertex_coordinates = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, vertex_coordinates)
    return graph


def generate_coordinates():
    coords = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0.1, 0.1],
        [0.9, 0.9]
    ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestInteriorSmoother(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.graph.set_user_defined_vertex(4, False)
        self.graph.set_user_defined_vertex(5, False)

    def test_interior_vertex(self):
        boundary_flag = [True, True, True, True, False, False]
        self.assertTrue(all(self.graph.is_boundary_vertex(vidx) == boundary_flag[vidx] for vidx in range(6)))

    def test_laplace_operator(self):
        index2vertex = self.graph.vertex_list(tag=False)
        num_verts = len(index2vertex)
        vertex2index = dict(zip(index2vertex, range(num_verts)))
        matrix = self.graph._construct_sparse_vertex_laplace_operator(vertex2index).toarray()

        test_matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0.],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 1],
                [0, 1, 1, 0, 1, 0]
            ]
        )
        self.assertTrue((matrix == test_matrix).all())

    def test_laplace_degrees(self):
        index2vertex = self.graph.vertex_list(tag=False)
        degrees = self.graph._get_vertex_degrees_for_smoothing(index2vertex)
        test_degrees = [1, 1, 1, 1, 3, 3]
        self.assertTrue(degrees == test_degrees)

    def test_smooth_op(self):
        self.graph.smooth_vertices(num_iter=10)
        test_coords = {
            0: np.array([0., 0.]),
            1: np.array([1., 0.]),
            2: np.array([1., 1.]),
            3: np.array([0., 1.]),
            4: np.array([0.25, 0.5]),
            5: np.array([0.75, 0.5]),
        }
        self.assertTrue(
            np.isclose(test_coords[key], self.graph.vertex_coordinate(key), atol=1.e-2).all() for key in range(6)
        )


if __name__ == "__main__":
    unittest.main()