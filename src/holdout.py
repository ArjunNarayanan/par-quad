"""The held-out test set, at whichever meshing target is being asked for.

The fifteen Mesh Quest levels are domains, not quad problems: the same polygon
can be asked for an all-quad mesh or an all-triangle one, and only the desired
degrees change. Those are re-derived here from the outline's own angles at the
environment's target angle, rather than taken from the game, which bakes in 90
degrees. The derivation reproduces the game's own numbers exactly at 90,
including on the two slit domains, so nothing about the quad test set moves.

`EXTRA_LEVELS` adds a few domains that are diagnostic for triangles in a way
the game's set is not -- above all `Hexagon`, whose par is 0 and whose optimum
provably requires an interior vertex: all fourteen of its chord-only
triangulations score 4.

Nothing in training may touch any of these.
"""

import math
import os
import sys
from copy import deepcopy

import numpy as np

import envs.polygon_utils as utils
from src.tiler import Tiler

_GAME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "game")
if _GAME_DIR not in sys.path:
    sys.path.insert(0, _GAME_DIR)

from server import INITIALIZERS as _GAME_INITIALIZERS  # noqa: E402

LEVELS = dict(_GAME_INITIALIZERS)

# The held-out fifteen, FROZEN. Deriving this from the game's dictionary meant
# that adding a playable level silently changed the denominator and made every
# Mesh Quest score in the logs incomparable with every later one. New game levels
# are welcome; they join EXTRA_LEVELS below, not the fifteen.
LEVEL_NAMES = [
    "L-shape", "T-bracket", "I-bracket", "U-channel", "Z-shape", "Plus",
    "Staircase", "Triangle", "Pentagon", "Semicircle", "Pac-Man", "Star",
    "Square hole", "Triforce ring", "Gear",
]
_missing = [name for name in LEVEL_NAMES if name not in LEVELS]
assert not _missing, f"the game no longer defines held-out levels: {_missing}"


def _regular(n, phase=0.0):
    return [(math.cos(phase + 2 * math.pi * k / n), math.sin(phase + 2 * math.pi * k / n))
            for k in range(n)]


def _simple_polygon(points):
    """A held-out level from a counter-clockwise outline; desired degrees follow."""
    def make():
        coordinates = {index: list(point) for index, point in enumerate(points)}
        graph = Tiler.from_face_loops([list(range(len(points)))], coordinates)
        return graph, {}
    return make


# Domains the game's set does not cover well for triangles. Hexagon is the
# important one: par 0, and no chord-only triangulation reaches it, so an agent
# that has not learned to place an interior vertex cannot solve it at all.
EXTRA_LEVELS = {
    "Hexagon": _simple_polygon(_regular(6)),
    "Octagon": _simple_polygon(_regular(8)),
    "Chevron": _simple_polygon([(0, 0), (2, 1), (4, 0), (4, 1.2), (2, 2.2), (0, 1.2)]),
}
EXTRA_LEVEL_NAMES = list(EXTRA_LEVELS)
ALL_LEVELS = {**LEVELS, **EXTRA_LEVELS}
ALL_LEVEL_NAMES = LEVEL_NAMES + EXTRA_LEVEL_NAMES

# enough to build any level; the outline does not depend on the rest
DEFAULT_ENV_CONFIG = {
    "name": "AngleEnvWithLength",
    "face_desired_degree": 4,
    "template_size": 64,
    "max_edge_addition_steps": 3,
    "smooth_iterations": 5,
}


def target_angle_of(env_config):
    """The angle a single element corner wants, from the face target."""
    return utils.average_face_angle(env_config.get("face_desired_degree", 4))


def outline_angles(loop, coordinates):
    """Interior angle per distinct vertex, summed over its occurrences.

    A slit outline visits its two endpoints twice, and the domain's total angle
    at such a vertex is the sum of the two corners it turns through there --
    which is what decides the degree it wants.
    """
    count = len(loop)
    total = {}
    for index, vertex in enumerate(loop):
        previous, following = loop[index - 1], loop[(index + 1) % count]
        first = np.asarray(coordinates[following]) - np.asarray(coordinates[vertex])
        second = np.asarray(coordinates[previous]) - np.asarray(coordinates[vertex])
        total[vertex] = total.get(vertex, 0.0) + utils.angle_between(first, second)
    return total


def canonical_signature(degrees):
    """Cyclic sequence of desired degrees, smallest under rotation and reflection.

    This is what identifies a *domain*: two outlines with the same signature
    pose the agent the same problem, whatever their coordinates. Training on
    one and testing on the other is not a held-out measurement.
    """
    degrees = list(degrees)
    n = len(degrees)
    rotations = [tuple(degrees[i:] + degrees[:i]) for i in range(n)]
    backwards = list(reversed(degrees))
    rotations += [tuple(backwards[i:] + backwards[:i]) for i in range(n)]
    return min(rotations)


class FixedLevel:
    """Initializer wrapper: the same level every time, at the given target."""

    def __init__(self, name, target_angle=90):
        if name not in ALL_LEVELS:
            raise KeyError(f"Unknown held-out level: {name}")
        self.name = name
        self.target_angle = target_angle

    def __call__(self):
        graph, _ = ALL_LEVELS[self.name]()
        loop = graph.generate_half_edge_face_loop(
            graph.first_face_halfedge(graph.face_list()[0]))
        vertices = [graph.source_vertex(h, tag=False) for h in loop]
        angles = outline_angles(vertices, graph.vertex_coordinates)
        desired = {vertex: utils.rounded_desired_degree(angle, self.target_angle)
                   for vertex, angle in angles.items()}
        return graph, desired


def level_outline(level, env_config=None, env_maker=None):
    """The level's outline loop, its desired degrees at the target, and its par."""
    config = env_config or DEFAULT_ENV_CONFIG
    env = make_level_env(config, level, env_maker=env_maker)
    env.reset()
    graph = env.graph
    loop = graph.generate_half_edge_face_loop(
        graph.first_face_halfedge(graph.face_list()[0]))
    vertices = [graph.source_vertex(h, tag=False) for h in loop]
    return {
        "loop": vertices,
        "degrees": [env.vertex_desired_degree[v] for v in vertices],
        "coordinates": {v: graph.vertex_coordinate(v) for v in set(vertices)},
        "par": env.par,
        "solved_on_arrival": bool(env.is_at_par()),
    }


def holdout_signatures(levels=None, env_config=None):
    """Every held-out domain's signature, for excluding them from training."""
    return {level: canonical_signature(level_outline(level, env_config)["degrees"])
            for level in (levels or ALL_LEVEL_NAMES)}


def make_level_env(env_config, level, env_maker=None):
    """Build the training env, but with a held-out level as its initializer."""
    if env_maker is None:
        from envs.environment_maker import initialize_environment as env_maker
    config = deepcopy(env_config)
    config.pop("initializer", None)
    config["graph_initializer"] = FixedLevel(level, target_angle_of(config))
    # a held-out level is evaluated exactly as given, even where the domain is
    # already at par on arrival (a lone triangle, asked for triangles, is)
    config["resample_if_at_par"] = False
    return env_maker(config)
