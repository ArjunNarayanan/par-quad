from src.tiler import Tiler
import unittest
import numpy as np


def initialize_graph():
    face_loops = [
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7]
    ]
    coords = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, coords)
    return graph


def generate_coordinates():
    coords = [
        [0, 0],
        [0.1, 0],
        [1, 0],
        [0, 0.8],
        [0.7, 0.7],
        [1, 0.2],
        [0, 1],
        [0.4, 1],
        [1, 1]
    ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestHexAngles(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        vertices = [1, 3, 4, 5, 7]
        for vidx in vertices:
            self.graph.set_user_defined_vertex(vidx, False)

    def test_angles(self):
        angles = self.graph.half_edge_angles()
        angles = [v for v in angles.values()]
        self.assertFalse(np.isclose(angles, 90, atol=1e-2).all())

        self.graph.smooth_vertices(num_iter=10)
        angles = self.graph.half_edge_angles()
        angles = [v for v in angles.values()]
        self.assertTrue(np.isclose(angles, 90, atol=1e-3).all())


if __name__ == "__main__":
    unittest.main()
