import numpy as np
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


class TestSmoothBoundary(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.graph.set_user_defined_vertex(3, False)

    def test_smooth1(self):
        self.graph.smooth_vertices(num_iter=1)

        coord = self.graph.vertex_coordinate(3)
        self.assertTrue((np.isclose(coord, [0.0, 0.5])).all())

    def test_smooth5(self):
        self.graph.smooth_vertices(num_iter=5)

        coord = self.graph.vertex_coordinate(3)
        self.assertTrue((np.isclose(coord, [0.0, 0.5])).all())

    def test_smooth_after_insert(self):
        self.graph.insert_vertex(6)
        self.graph.smooth_vertices(num_iter=5)

        c3 = self.graph.vertex_coordinate(3)
        self.assertTrue((np.isclose(c3, [0, 1 / 3], atol=1e-2)).all())

        c5 = self.graph.vertex_coordinate(5)
        self.assertTrue((np.isclose(c5, [0, 2 / 3], atol=1e-2)).all())


if __name__ == "__main__":
    unittest.main()
