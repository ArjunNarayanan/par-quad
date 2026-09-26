import unittest

import numpy as np
from src.tiler import Tiler, _get_vertex_array, _get_edge_array, triangle_connectivity_representation


def initialize_graph():
    coords = generate_coordinates()
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 6],
        [0, 6, 1],
    ]
    graph = Tiler.from_face_loops(faces, vertex_coordinates=coords)
    return graph


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[0, 0],
              [-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    coords = dict(zip(range(len(coords)), coords))
    return coords


class TestConnectivityPrimitives(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = initialize_graph()
        self.graph.user_defined_vertices.discard(0)

    def test_vertex2index(self):
        coords, vertex2index = _get_vertex_array(self.graph)
        c = np.cos(np.pi / 3)
        s = np.sin(np.pi / 3)
        test_coords = np.array(
            [[0, 0],
             [-c, -s],
             [c, -s],
             [1, 0],
             [c, s],
             [-c, s],
             [-1, 0]]
        )
        self.assertTrue((coords == test_coords).all())
        self.assertTrue(len(vertex2index) == 7)
        self.assertTrue(all(vertex2index[idx] == idx for idx in range(7)))

    def test_edge2index(self):
        coords, vertex2index = _get_vertex_array(self.graph)
        edges, edge2index = _get_edge_array(self.graph, vertex2index)

        test_edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 0],
                [2, 3],
                [3, 0],
                [3, 4],
                [4, 0],
                [4, 5],
                [5, 0],
                [5, 6],
                [6, 0],
                [6, 1]
            ]
        )
        test_edge2index = {
            (0, 1): 0,
            (1, 0): 0,
            (1, 2): 1,
            (2, 1): 1,
            (2, 0): 2,
            (0, 2): 2,
            (2, 3): 3,
            (3, 2): 3,
            (3, 0): 4,
            (0, 3): 4,
            (3, 4): 5,
            (4, 3): 5,
            (0, 4): 6,
            (4, 0): 6,
            (4, 5): 7,
            (5, 4): 7,
            (0, 5): 8,
            (5, 0): 8,
            (5, 6): 9,
            (6, 5): 9,
            (0, 6): 10,
            (6, 0): 10,
            (6, 1): 11,
            (1, 6): 11
        }
        self.assertTrue((edges == test_edges).all())
        self.assertTrue(len(edge2index) == len(test_edge2index))
        self.assertTrue(all(edge2index[k] == v for k, v in test_edge2index.items()))

    def test_representation(self):
        representation = triangle_connectivity_representation(self.graph)

        test_vconn = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 5],
                [0, 5, 6],
                [0, 6, 1]
            ]
        )
        vconn = representation["vertex connectivity"]
        self.assertTrue((vconn == test_vconn).all())

        test_edges = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 0],
                [2, 3],
                [3, 0],
                [3, 4],
                [4, 0],
                [4, 5],
                [5, 0],
                [5, 6],
                [6, 0],
                [6, 1]
            ]
        )
        edges = representation["edges"]
        self.assertTrue((edges == test_edges).all())

        test_econn = np.array(
            [
                [0, 1, 2],
                [2, 3, 4],
                [4, 5, 6],
                [6, 7, 8],
                [8, 9, 10],
                [10, 11, 0]
            ]
        )
        econn = representation["edge connectivity"]
        self.assertTrue((econn == test_econn).all())

        coords_dict = generate_coordinates()
        test_coords = np.array([coords_dict[idx] for idx in range(7)])
        coords = representation["coordinates"]
        self.assertTrue(np.isclose(coords, test_coords).all())

        test_user_vertices = [1,2,3,4,5,6]
        user_vertices = representation["user vertices"]
        self.assertTrue(user_vertices == test_user_vertices)



if __name__ == "__main__":
    unittest.main()
