import unittest
import numpy as np
from src.tiler import Tiler, quad_connectivity_representation


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


class TestConnectivityPrimitives(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.graph.user_defined_vertices.discard(4)

    def test_representation(self):
        representation = quad_connectivity_representation(self.graph)

        test_vconn = np.array(
            [
                [0, 1, 4, 3],
                [1, 2, 5, 4],
                [3, 4, 7, 6],
                [4, 5, 8, 7]
            ]
        )
        vconn = representation["vertex connectivity"]
        self.assertTrue((vconn == test_vconn).all())

        test_edges = np.array(
            [
                [0, 1],
                [1, 4],
                [4, 3],
                [3, 0],
                [1, 2],
                [2, 5],
                [5, 4],
                [4, 7],
                [7, 6],
                [6, 3],
                [5, 8],
                [8, 7]
            ]
        )
        edges = representation["edges"]
        self.assertTrue((edges == test_edges).all())

        test_econn = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 1],
                [2, 7, 8, 9],
                [6, 10, 11, 7]
            ]
        )
        econn = representation["edge connectivity"]
        self.assertTrue((econn == test_econn).all())

        coords_dict = generate_coordinates()
        test_coords = np.array([coords_dict[idx] for idx in range(9)])
        coords = representation["coordinates"]
        self.assertTrue(np.isclose(coords, test_coords).all())

        test_user_vertices = [0, 1, 2, 3, 5, 6, 7, 8]
        user_vertices = representation["user vertices"]
        self.assertTrue(user_vertices == test_user_vertices)
