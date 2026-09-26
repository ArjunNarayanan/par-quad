from src.tiler import Tiler
import unittest
import numpy as np


def initialize_graph():
    face_loops = [
        [0, 1, 2, 3, 4, 5]
    ]
    coords = generate_coordinates()
    graph = Tiler.from_face_loops(face_loops, coords)
    return graph


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    coords = dict(zip(range(6), coords))
    return coords


class TestHexAngles(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()

    def test_angles(self):
        h = self.graph.half_edge_tag

        angles = self.graph.half_edge_angles()
        angles = [angles[(idx, h)] for idx in range(6)]
        self.assertTrue(np.isclose(angles, 120).all())

        self.graph.insert_half_edge(0, 2)
        angles = self.graph.half_edge_angles()
        angles = [angles[(idx,h)] for idx in range(8)]
        test_angles = [60, 120, 120, 60, 120, 120, 60, 60]
        self.assertTrue((np.isclose(angles, test_angles)).all())

        self.graph.insert_vertex(7)
        angles = self.graph.half_edge_angles()
        angles = [angles[(idx, h)] for idx in range(10)]
        test_angles = [60, 120, 120, 60, 120, 120, 180, 60, 180, 60]
        self.assertTrue((np.isclose(angles, test_angles)).all())

        self.graph.insert_half_edge(8, 1)
        angles = self.graph.half_edge_angles()
        angles = [angles[(idx, h)] for idx in range(12)]
        test_angles = [60, 120, 120, 60, 60, 120, 180, 60, 60, 60, 60, 120]
        self.assertTrue((np.isclose(angles, test_angles)).all())

        self.graph.insert_half_edge(5, 1)
        angles = self.graph.half_edge_angles()
        angles = [angles[(idx, h)] for idx in range(14)]
        test_angles = [60, 120, 120, 60, 60, 60, 180, 60, 60, 60, 60, 60, 60, 60]
        self.assertTrue((np.isclose(angles, test_angles)).all())

        self.graph.insert_half_edge(6, 1)
        angles = self.graph.half_edge_angles()
        angles = [angles[(idx, h)] for idx in range(16)]
        test_angles = [60, 60, 120, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 120]
        self.assertTrue((np.isclose(angles, test_angles)).all())

        self.graph.insert_half_edge(15, 1)
        angles = self.graph.half_edge_angles()
        angles = [angles[(idx, h)] for idx in range(18)]
        self.assertTrue((np.isclose(angles, 60)).all())


if __name__ == "__main__":
    unittest.main()
