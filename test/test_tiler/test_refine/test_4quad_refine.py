import unittest
import numpy as np
from src.tiler import Tiler, quad_connectivity_representation, _refine_quads


def initialize_graph():
    coords = generate_coordinates()
    faces = [
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7]
    ]
    graph = Tiler.from_face_loops(faces, vertex_coordinates=coords)
    return graph


def generate_coordinates():
    coords = [
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [0, 2],
        [1, 2],
        [2, 2]
    ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestQuadRefine(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.representation = quad_connectivity_representation(self.graph)

    def test_refine(self):
        representation = self.representation
        coords = representation["coordinates"]
        vconn = representation["vertex connectivity"]
        edges = representation["edges"]
        econn = representation["edge connectivity"]

        new_coords, new_conn = _refine_quads(coords, edges, vconn, econn)

        test_coords = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
                [0.5, 0.],
                [1, 0.5],
                [0.5, 1],
                [0, 0.5],
                [1.5, 0.],
                [2, 0.5],
                [1.5, 1],
                [1, 1.5],
                [0.5, 2],
                [0, 1.5],
                [2, 1.5],
                [1.5, 2],
                [0.5, 0.5],
                [1.5, 0.5],
                [0.5, 1.5],
                [1.5, 1.5]
            ]
        )
        self.assertTrue(np.isclose(test_coords, new_coords).all())

        test_conn = np.array(
            [
                [0, 9, 21, 12],
                [1, 13, 22, 10],
                [3, 11, 23, 18],
                [4, 15, 24, 16],
                [9, 1, 10, 21],
                [13, 2, 14, 22],
                [11, 4, 16, 23],
                [15, 5, 19, 24],
                [21, 10, 4, 11],
                [22, 14, 5, 15],
                [23, 16, 7, 17],
                [24, 19, 8, 20],
                [12, 21, 11, 3],
                [10, 22, 15, 4],
                [18, 23, 17, 6],
                [16, 24, 20, 7]
            ]
        )
        self.assertTrue((test_conn == new_conn).all())
