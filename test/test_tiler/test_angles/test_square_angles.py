from src.tiler import Tiler
import unittest
import numpy as np


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3]
    ]
    coords = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, coords)
    return graph


def generate_coordinates():
    coords = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ]
    coords = dict(zip(range(6), coords))
    return coords


class TestHexAngles(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()

    def test_angles(self):
        angles = self.graph.half_edge_angles()
        self.assertTrue(all(a == 90 for a in angles.values()))


if __name__ == "__main__":
    unittest.main()
