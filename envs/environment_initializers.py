import math
from src.tiler import Tiler
import numpy as np
import envs.polygon_utils as utils
from copy import deepcopy


class LEnv:
    def __init__(self, target_angle):
        self.target_angle = target_angle

    @staticmethod
    def generate_coordinates():
        coords = [
            [0, 0],
            [2, 0],
            [2, 1],
            [1, 1],
            [1, 2],
            [0, 2],
        ]
        coords = dict(zip(range(6), coords))
        return coords

    def __call__(self):
        coords = self.generate_coordinates()
        loop = [[0, 1, 2, 3, 4, 5]]
        graph = Tiler.from_face_loops(loop, coords)

        interior_angles = utils.get_polygon_interior_angles(
            loop[0], graph.vertex_coordinates
        )
        desired_degree = {
            vidx: utils.rounded_desired_degree(angle, self.target_angle)
            for vidx, angle in interior_angles.items()
        }

        return graph, desired_degree


class RandomPolygon:
    """Random star-shaped polygon on a jittered circle.

    `min_quality` rejects instances that no mesh can win. A pinned corner of
    interior angle `a` is split by its desired degree `d` into `d - 1` element
    corners of roughly `a / (d - 1)` degrees; if that lands outside
    (asin(min_quality), 180 - asin(min_quality)) then some element corner is
    always degenerate and par is unreachable however well the agent plays.
    Measured on `scale=0.8`, 44% of 20-gons and 8% of 12-gons were unwinnable
    this way before the filter.
    """

    def __init__(self, polygon_degree_range, target_angle, scale=0.5,
                 min_quality=0.4, max_rejection_tries=200):
        self.polygon_degree_range = list(polygon_degree_range)
        self.target_angle = target_angle
        self.scale = scale
        self.min_quality = min_quality
        self.max_rejection_tries = max_rejection_tries

    def set_degree_range(self, degree_range):
        self.polygon_degree_range = list(degree_range)

    def corner_angle_bounds(self):
        lower = np.degrees(np.arcsin(np.clip(self.min_quality, 0.0, 1.0)))
        return lower, 180.0 - lower

    def is_winnable(self, interior_angles, desired_degree):
        if self.min_quality <= 0:
            return True
        lower, upper = self.corner_angle_bounds()
        for vidx, angle in interior_angles.items():
            splits = max(desired_degree[vidx] - 1, 1)
            element_angle = angle / splits
            if element_angle < lower or element_angle > upper:
                return False
        return True

    def _sample(self):
        polygon_degree = int(np.random.choice(self.polygon_degree_range))
        coordinates = self.generate_random_coordinates(polygon_degree, self.scale)
        node_ids = list(range(polygon_degree))
        face_loop = [node_ids]
        coordinates = dict(zip(node_ids, coordinates))
        graph = Tiler.from_face_loops(face_loop, coordinates)
        interior_angles = utils.get_polygon_interior_angles(
            face_loop[0], graph.vertex_coordinates
        )
        desired_degree = {
            vidx: utils.rounded_desired_degree(angle, self.target_angle)
            for vidx, angle in interior_angles.items()
        }
        return graph, desired_degree, interior_angles

    def __call__(self):
        last = None
        for _ in range(self.max_rejection_tries):
            graph, desired_degree, interior_angles = self._sample()
            last = (graph, desired_degree)
            if self.is_winnable(interior_angles, desired_degree):
                return graph, desired_degree
        return last

    @staticmethod
    def generate_random_coordinates(polygon_degree, scale):
        assert polygon_degree >= 3

        angle = 2 * np.pi / polygon_degree
        angular_increments = angle * np.arange(polygon_degree)
        radii = (1 - scale) + scale * np.random.rand(polygon_degree)
        x_coord = np.cos(angular_increments) * radii
        y_coord = np.sin(angular_increments) * radii
        coords = [[x, y] for x, y in zip(x_coord, y_coord)]
        return coords


class RandomPolygonWithHole:
    """Random annular domain: a star-shaped outline with a star-shaped hole.

    The distribution the certified generator cannot reach. `random_annulus`
    builds holes on the square lattice only, and `SquareHole` is a single fixed
    shape, so nothing so far samples a hole domain with arbitrary corner angles.

    The hole is encoded the way the whole codebase encodes one: a single face
    loop with a SLIT joining the outline to the hole, so the two slit endpoints
    appear twice in the loop and the domain stays one face. The interior angle
    at such a vertex is the SUM over its occurrences, which is what decides the
    degree it wants -- compute it per occurrence and the slit vertices come out
    wrong.

    Containment is guaranteed rather than tested: both rings are star-shaped
    about the origin, so keeping every hole radius below every outline radius
    puts the hole strictly inside. The slit is still checked against both rings,
    because a non-convex hole can reach across the gap even when it is nested.
    """

    def __init__(self, polygon_degree_range, target_angle, scale=0.5,
                 hole_degree_range=(4, 8), hole_scale=0.35, hole_margin=0.9,
                 min_quality=0.4, max_rejection_tries=200):
        self.polygon_degree_range = list(polygon_degree_range)
        self.hole_degree_range = list(hole_degree_range)
        self.target_angle = target_angle
        self.scale = scale
        self.hole_scale = hole_scale
        self.hole_margin = hole_margin
        self.min_quality = min_quality
        self.max_rejection_tries = max_rejection_tries
        self._winnable = RandomPolygon(polygon_degree_range, target_angle,
                                       scale=scale, min_quality=min_quality)

    def set_degree_range(self, degree_range):
        self.polygon_degree_range = list(degree_range)

    @staticmethod
    def _segments_cross(p1, p2, p3, p4):
        """True when p1p2 and p3p4 properly cross (shared endpoints excluded)."""
        def side(a, b, c):
            return ((b[0] - a[0]) * (c[1] - a[1])
                    - (b[1] - a[1]) * (c[0] - a[0]))
        d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
        d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    def _slit_is_clear(self, start, end, rings):
        for ring in rings:
            count = len(ring)
            for index in range(count):
                a, b = ring[index], ring[(index + 1) % count]
                if a is start or b is start or a is end or b is end:
                    continue
                if self._segments_cross(start, end, a, b):
                    return False
        return True

    def _sample(self):
        outer_degree = int(np.random.choice(self.polygon_degree_range))
        hole_degree = int(np.random.choice(
            range(self.hole_degree_range[0], self.hole_degree_range[1] + 1)))

        outer = RandomPolygon.generate_random_coordinates(outer_degree, self.scale)
        # every hole radius stays below every outline radius, so nesting is exact
        ceiling = (1.0 - self.scale) * self.hole_margin
        hole = RandomPolygon.generate_random_coordinates(hole_degree, self.hole_scale)
        hole = [[x * ceiling, y * ceiling] for x, y in hole]

        # slit between the closest pair by polar angle, so it runs roughly radially
        def bearing(point):
            return math.atan2(point[1], point[0]) % (2 * math.pi)
        outer_index = 0
        hole_index = min(
            range(hole_degree),
            key=lambda j: abs((bearing(hole[j]) - bearing(outer[0]) + math.pi)
                              % (2 * math.pi) - math.pi))
        if not self._slit_is_clear(outer[outer_index], hole[hole_index], (outer, hole)):
            return None

        outer_ids = list(range(outer_degree))
        hole_ids = list(range(outer_degree, outer_degree + hole_degree))
        coordinates = dict(zip(outer_ids + hole_ids, outer + hole))

        # rotate each ring so the slit endpoints lead, then walk the hole backwards:
        # the outline is counter-clockwise and the hole must be clockwise inside it
        o = outer_ids[outer_index:] + outer_ids[:outer_index]
        i = hole_ids[hole_index:] + hole_ids[:hole_index]
        loop = o + [o[0], i[0]] + i[:0:-1] + [i[0]]

        graph = Tiler.from_face_loops([loop], coordinates)
        angles = self._summed_interior_angles(graph)
        desired = {vidx: utils.rounded_desired_degree(angle, self.target_angle)
                   for vidx, angle in angles.items()}
        return graph, desired, angles

    @staticmethod
    def _summed_interior_angles(graph):
        """Interior angle per vertex, summed over each visit the slit makes."""
        total = {}
        for hidx in graph.half_edge_list():
            vidx = graph.source_vertex(hidx, tag=False)
            nxt = graph.source_vertex(graph.next_half_edge(hidx), tag=False)
            prv = graph.source_vertex(graph.previous_half_edge(hidx), tag=False)
            here = np.asarray(graph.vertex_coordinate(vidx), dtype=float)
            first = np.asarray(graph.vertex_coordinate(nxt), dtype=float) - here
            second = np.asarray(graph.vertex_coordinate(prv), dtype=float) - here
            total[vidx] = total.get(vidx, 0.0) + utils.angle_between(first, second)
        return total

    def __call__(self):
        last = None
        for _ in range(self.max_rejection_tries):
            sampled = self._sample()
            if sampled is None:
                continue
            graph, desired, angles = sampled
            last = (graph, desired)
            if self._winnable.is_winnable(angles, desired):
                return graph, desired
        if last is None:
            raise RuntimeError(
                "RandomPolygonWithHole: no valid instance in "
                f"{self.max_rejection_tries} tries")
        return last


class FixedRandomPolygon:
    def __init__(self, polygon_degree, target_angle, scale=0.5):
        initializer = RandomPolygon([polygon_degree], target_angle, scale)
        graph, desired_degree = initializer()
        self.graph = graph
        self.desired_degree = desired_degree

    def __call__(self):
        return deepcopy(self.graph), deepcopy(self.desired_degree)


class Hexagon:
    """
    Represents a regular hexagon environment initializer for tiling and polygon experiments.

    This class generates the coordinates and desired vertex degrees for a regular hexagon,
    parameterized by a target angle. It can be used to create a graph representation of a hexagon
    with associated desired degrees for each vertex, suitable for mesh or tiling algorithms.

    Args:
        target_angle (float): The target angle (in degrees) to which the interior angles of the hexagon's vertices are rounded.

    Methods:
        generate_coordinates():
            Static method that returns the coordinates of the hexagon's vertices as a dictionary.
        __call__():
            Generates the hexagon graph and the desired degree dictionary for each vertex.
    """

    def __init__(self, target_angle):
        self.target_angle = target_angle

    @staticmethod
    def generate_coordinates():
        c = np.cos(np.pi / 3)
        s = np.sin(np.pi / 3)
        coords = [[-c, -s], [c, -s], [1, 0], [c, s], [-c, s], [-1, 0]]
        coords = dict(zip(range(6), coords))
        return coords

    def __call__(self):
        face_loops = [[0, 1, 2, 3, 4, 5]]
        coords = self.generate_coordinates()
        graph = Tiler.from_face_loops(face_loops, coords)

        interior_angles = utils.get_polygon_interior_angles(
            face_loops[0], graph.vertex_coordinates
        )
        desired_degree = {
            vidx: utils.rounded_desired_degree(angle, self.target_angle)
            for vidx, angle in interior_angles.items()
        }
        return graph, desired_degree


class CenterCrack:
    def __init__(self, target_angle):
        assert target_angle == 90
        self.target_angle = target_angle

    @staticmethod
    def generate_coordinates():
        coords = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0.0, 0.5],
            [0.25, 0.5],
            [0.5, 0.5 + 1e-9],
            [0.75, 0.5],
            [0.5, 0.5 - 1e-9],
        ]
        coords = dict(zip(range(len(coords)), coords))
        return coords

    def __call__(self):
        coords = self.generate_coordinates()
        loop = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 4]]
        graph = Tiler.from_face_loops(loop, coords)

        desired_degree = dict(zip(range(9), [2, 2, 2, 2, 3, 5, 3, 5, 3]))

        return graph, desired_degree


class SquareHole:
    def __init__(self, target_angle):
        # assert target_angle == 90
        self.target_angle = target_angle

    @staticmethod
    def generate_coordinates():
        coords = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
        ]
        coords = dict(zip(range(len(coords)), coords))
        return coords

    @staticmethod
    def generate_interior_angles():
        angles = {
            0: 90,
            1: 90,
            2: 90,
            3: 90,
            4: 270,
            5: 270,
            6: 270,
            7: 270,
        }
        return angles

    def __call__(self):
        coords = self.generate_coordinates()
        loop = [[0, 1, 2, 3, 0, 4, 7, 6, 5, 4]]
        graph = Tiler.from_face_loops(loop, coords)

        interior_angles = self.generate_interior_angles()
        desired_degree = {
            vidx: utils.rounded_desired_degree(angle, self.target_angle)
            for vidx, angle in interior_angles.items()
        }

        return graph, desired_degree


class MixedMesh:
    def __init__(self, target_angle):
        assert target_angle == 90
        self.target_angle = target_angle

    @staticmethod
    def generate_coordinates():
        coords = Hexagon.generate_coordinates()
        coords[6] = coords[0] - np.array([0, 1])
        coords[7] = coords[1] - np.array([0, 1])
        coords[8] = np.array([1.5, np.sin(np.pi / 3)])
        return coords

    @staticmethod
    def generate_angles():
        angles = {
            0: 210,
            1: 210,
            2: 180,
            3: 180,
            4: 120,
            5: 120,
            6: 90,
            7: 90,
            8: 60,
        }
        return angles

    def get_mesh(self):
        coords = self.generate_coordinates()
        loop = [[0, 1, 2, 3, 4, 5], [6, 7, 1, 0], [3, 2, 8]]
        graph = Tiler.from_face_loops(loop, coords)

        return graph

    def __call__(self):
        interior_angles = self.generate_angles()
        desired_degree = {
            vidx: utils.rounded_desired_degree(angle, self.target_angle)
            for vidx, angle in interior_angles.items()
        }

        graph = self.get_mesh()
        return graph, desired_degree


class SquareHole2:
    @staticmethod
    def generate_coordinates():
        coords = [
            [0, 0],
            [0.5, 0],
            [0.5, 0.25],
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
            [0.75, 0.25],
            [1, 0],
            [1, 1],
            [0, 1],
        ]
        coords = dict(zip(range(len(coords)), coords))
        return coords

    def __call__(self):
        coords = self.generate_coordinates()
        loop = [[0, 1, 2, 3, 4, 5, 6, 2, 1, 7, 8, 9]]
        graph = Tiler.from_face_loops(loop, coords)
        desired_degree = dict(zip(range(10), [2, 3, 3, 4, 4, 4, 4, 2, 2, 2]))

        return graph, desired_degree


class Arc:
    @staticmethod
    def generate_coordinates():
        coords = [
            [-1, 0],
            [-1, 1],
            [-0.8, 2],
            [0, 2.5],
            [0.8, 2],
            [1, 1],
            [1, 0],
            [2, 0],
            [2, 1],
            [1.56, 2.5],
            [0, 3.5],
            [-1.56, 2.5],
            [-2, 1],
            [-2, 0],
        ]
        coords = dict(zip(range(len(coords)), coords))
        return coords

    def __call__(self):
        coords = self.generate_coordinates()
        loop = [list(range(14))]
        graph = Tiler.from_face_loops(loop, coords)
        desired_degree = dict(zip(range(14), [2] + 5 * [3] + [2, 2] + 5 * [3] + [2]))

        return graph, desired_degree


class MixtureInitializer:
    """Draw from several initializers with adjustable weights.

    The curriculum controller owns the weights, so the mixture between
    reverse-curriculum start states and raw polygons can move during training
    without rebuilding the environments.
    """

    def __init__(self, initializers, weights=None):
        self.initializers = list(initializers)
        if weights is None:
            weights = [1.0] * len(self.initializers)
        self.set_weights(weights)

    def set_weights(self, weights):
        weights = np.asarray(weights, dtype=float)
        assert len(weights) == len(self.initializers)
        total = weights.sum()
        assert total > 0, "mixture weights must not be all zero"
        self.weights = weights / total

    def set_degree_range(self, degree_range):
        for initializer in self.initializers:
            setter = getattr(initializer, "set_degree_range", None)
            if setter is not None:
                setter(degree_range)

    def set_cost_to_go(self, cost_to_go):
        for initializer in self.initializers:
            setter = getattr(initializer, "set_cost_to_go", None)
            if setter is not None:
                setter(cost_to_go)

    def __call__(self):
        index = int(np.random.choice(len(self.initializers), p=self.weights))
        return self.initializers[index]()
