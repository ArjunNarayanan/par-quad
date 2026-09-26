import unittest
import numpy as np
from src.tiler import Tiler, _refine_triangles, triangle_connectivity_representation


def initialize_graph():
    coords = np.array(
        [
            [0, 0.],
            [1, 0],
            [0.5, 1]
        ]
    )
    coords = dict(zip(range(3), coords))
    faces = [[0, 1, 2]]
    graph = Tiler.from_face_loops(faces, coords)
    return graph


class TestRefine(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.representation = triangle_connectivity_representation(self.graph)

    def test_refine(self):
        representation = self.representation
        coords = representation["coordinates"]
        vconn = representation["vertex connectivity"]
        edges = representation["edges"]
        econn = representation["edge connectivity"]

        new_coords, new_conn = _refine_triangles(coords, edges, vconn, econn)

        test_coords = np.array(
            [
                [0, 0.],
                [1, 0],
                [0.5, 1],
                [0.5, 0.],
                [0.75, 0.5],
                [0.25, 0.5]
            ]
        )
        self.assertTrue((new_coords == test_coords).all())

        test_conn = np.array(
            [
                [0, 3, 5],
                [3, 1, 4],
                [3, 4, 5],
                [5, 4, 2]
            ]
        )
        self.assertTrue((new_conn == test_conn).all())
