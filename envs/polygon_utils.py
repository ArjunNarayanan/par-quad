import numpy as np
from typing import Sequence


def average_face_angle(polygon_degree: int) -> float:
    """
    Calculate the average interior angle of a regular polygon given its degree (number of sides).

    Args:
        polygon_degree (int): Number of sides of the polygon.

    Returns:
        float: The average interior angle in degrees.
    """
    return (polygon_degree - 2) * 180 / polygon_degree


def angle_between(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Compute the angle in degrees between two 2D vectors.

    Args:
        v1 (Sequence[float]): The first 2D vector (list, tuple, or numpy array of floats).
        v2 (Sequence[float]): The second 2D vector (list, tuple, or numpy array of floats).

    Returns:
        float: The angle between v1 and v2 in degrees, in the range [0, 360).
    """
    assert len(v1) == 2 and len(v2) == 2, "Vectors must be 2D"
    
    dotp = v1[0] * v2[0] + v1[1] * v2[1]
    detp = (v1[0] * v2[1] - v2[0] * v1[1])
    angle = np.degrees(np.arctan2(detp, dotp))
    if angle < 0:
        angle = angle + 360
    return angle


def generate_coordinates(polygon_degree: int, scale: float = 0.5) -> list[list[float]]:
    """
    Generate random coordinates for the vertices of a polygon with a given degree and scale.

    Args:
        polygon_degree (int): Number of sides (vertices) of the polygon. Must be >= 3.
        scale (float, optional): Controls the randomness of the radii. Defaults to 0.5.

    Returns:
        list[list[float]]: List of [x, y] coordinates for each vertex.
    """
    assert polygon_degree >= 3
    angle = 2 * np.pi / polygon_degree
    angular_increments = angle * np.arange(polygon_degree)
    radii = (1 - scale) + scale * np.random.rand(polygon_degree)
    x_coord = np.cos(angular_increments) * radii
    y_coord = np.sin(angular_increments) * radii
    coords = [[x, y] for x, y in zip(x_coord, y_coord)]
    return coords


def get_polygon_interior_angles(face_loop: list[int], vertex_coords: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the interior angles at each vertex of a polygon.

    Args:
        face_loop (list[int]): Ordered list of vertex indices forming the polygon loop.
        vertex_coords (dict[int, list[float]]): Mapping from vertex index to [x, y] coordinates.

    Returns:
        dict[int, float]: Mapping from vertex index to its interior angle in degrees.
    """
    num_verts = len(face_loop)
    angles = {}
    for idx, vcurrent in enumerate(face_loop):
        vprev = face_loop[idx - 1]
        vnext = face_loop[(idx + 1) % num_verts]

        v1 = np.array(vertex_coords[vnext]) - np.array(vertex_coords[vcurrent])
        v2 = np.array(vertex_coords[vprev]) - np.array(vertex_coords[vcurrent])
        angles[vcurrent] = angle_between(v1, v2)
    return angles


def rounded_desired_degree(angle: float, target_angle: float) -> int:
    """
    Compute the desired degree (number of incident edges) for a vertex based on its angle and a target angle.

    Args:
        angle (float): The interior angle at the vertex (in degrees).
        target_angle (float): The target angle (in degrees) for rounding.

    Returns:
        int: The rounded desired degree, at least 2.
    """
    degree = max(int(np.round(angle / target_angle)) + 1, 2)
    return degree
