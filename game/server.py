"""Interactive mesh-editing game server.

Copyright 2026 Arjun Narayanan.
Licensed under the MIT License (see LICENSE).

Wraps src.tiler.Tiler with a small JSON API and serves a single-page UI.
Run from the repo root (or anywhere):

    venv/bin/python game/server.py
"""

import json
import math
import sys
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import envs.polygon_utils as utils  # noqa: E402
from envs.environment_initializers import LEnv, SquareHole  # noqa: E402
import src.curved_levels as curved_levels  # noqa: E402
from src.boundary_arcs import edge_direction  # noqa: E402
from src.tiler import Tiler  # noqa: E402

# The element the player is cutting towards. `set_element_target` switches the
# whole module to triangles; every constant below is derived from it.
TARGET_ANGLE = 90
FACE_DESIRED = 4
# Minimum scaled Jacobian (sin of corner angle) required to win: rules out
# near-degenerate elements with corner angles outside roughly (24, 156) degrees.
QUALITY_THRESHOLD = 0.4
INTERIOR_DESIRED = utils.rounded_desired_degree(360, TARGET_ANGLE) - 1  # 4
BOUNDARY_DESIRED = utils.rounded_desired_degree(180, TARGET_ANGLE)  # 3


def set_element_target(face_desired):
    """Switch between quads (4) and triangles (3)."""
    global TARGET_ANGLE, FACE_DESIRED, INTERIOR_DESIRED, BOUNDARY_DESIRED
    FACE_DESIRED = face_desired
    TARGET_ANGLE = (face_desired - 2) * 180 / face_desired
    INTERIOR_DESIRED = utils.rounded_desired_degree(360, TARGET_ANGLE) - 1
    BOUNDARY_DESIRED = utils.rounded_desired_degree(180, TARGET_ANGLE)


def regular_polygon(n, phase=0.0):
    return [
        (math.cos(phase + 2 * math.pi * k / n), math.sin(phase + 2 * math.pi * k / n))
        for k in range(n)
    ]


def star_polygon():
    """Five-pointed star: tips at radius 1, reflex notches at pentagram radius."""
    inner = math.cos(math.radians(72)) / math.cos(math.radians(36))
    pts = []
    for k in range(5):
        tip = math.radians(90 + 72 * k)
        notch = math.radians(126 + 72 * k)
        pts.append((math.cos(tip), math.sin(tip)))
        pts.append((inner * math.cos(notch), inner * math.sin(notch)))
    return pts


def semicircle_polygon():
    """Base corners plus a 5-point arc (150-degree corners, demotable)."""
    pts = [(-1.0, 0.0), (1.0, 0.0)]
    for k in range(1, 6):
        t = math.radians(30 * k)
        pts.append((math.cos(t), math.sin(t)))
    return pts


def pacman_polygon():
    """Disk with a 90-degree wedge bite: reflex center + 7-point arc."""
    pts = [(0.0, 0.0)]
    for k in range(7):
        t = math.radians(45 + 45 * k)
        pts.append((math.cos(t), math.sin(t)))
    return pts


def gear_polygon(teeth=6, outer=1.0, root=0.62):
    """Square-wave gear: convex tooth corners and reflex roots."""
    pts = []
    w = 2 * math.pi / teeth
    for k in range(teeth):
        base = w * k
        for frac, rad in [(0.05, outer), (0.45, outer), (0.55, root), (0.95, root)]:
            t = base + frac * w
            pts.append((rad * math.cos(t), rad * math.sin(t)))
    return pts


def triforce_ring():
    """Triangle with a same-orientation triangular hole (chi = 0, par 0).

    The slit joining hole to rim runs between two mid-edge points, which both
    want degree 3 and so are already satisfied by the slit itself: unlike a
    corner-to-corner slit, this one is a legitimate edge of a perfect mesh and
    never has to be deleted.
    """
    outer = [
        (math.cos(math.radians(90 + 120 * k)), math.sin(math.radians(90 + 120 * k)))
        for k in range(3)
    ]
    hole = [
        (0.4 * math.cos(math.radians(90 + 120 * k)),
         0.4 * math.sin(math.radians(90 + 120 * k)))
        for k in range(3)
    ]
    coords = dict(enumerate([list(p) for p in outer] + [list(p) for p in hole]))
    # slit endpoints: midpoints of outer edge 0-1 and of inner edge 3-4
    coords[6] = [(outer[0][0] + outer[1][0]) / 2, (outer[0][1] + outer[1][1]) / 2]
    coords[7] = [(hole[0][0] + hole[1][0]) / 2, (hole[0][1] + hole[1][1]) / 2]
    loop = [0, 6, 7, 3, 5, 4, 7, 6, 1, 2]
    return slit_graph(loop, coords)


def square_hole_ring():
    """Square with a square hole; slit runs mid-edge to mid-edge (par 0).

    Same idea as triforce_ring: both slit endpoints are flat boundary points
    wanting degree 3, which the slit already provides.
    """
    coords = {
        0: [0.0, 0.0],
        1: [0.5, 0.0],
        2: [0.5, 0.25],
        3: [0.25, 0.25],
        4: [0.25, 0.75],
        5: [0.75, 0.75],
        6: [0.75, 0.25],
        7: [1.0, 0.0],
        8: [1.0, 1.0],
        9: [0.0, 1.0],
    }
    loop = [0, 1, 2, 3, 4, 5, 6, 2, 1, 7, 8, 9]
    return slit_graph(loop, coords)


def diamond_bracket():
    """A slotted plate with a diamond hole; the slit drops from the hole to the base.

    `straight-holes #18` of the geogen held-out set, the domain the agent could not
    solve after twelve million steps -- it reaches a full quad mesh carrying one
    vertex defect and then wanders. par is 0, so a perfect mesh exists. Lattice
    coordinates, so the outline is exactly the generator's.

    The slit runs from the hole's bottom corner (270 degrees, wants four) straight
    down to a flat point on the base edge (180 degrees, wants three), which is the
    mid-edge anchoring the other hole levels use: a slit endpoint is forced to degree
    three, so an anchor wanting less than that could never reach par.
    """
    coords = {
        0: [0, 0], 1: [8, 0], 2: [8, 4], 3: [7, 4], 4: [7, 1], 5: [6, 1],
        6: [6, 4], 7: [2, 4], 8: [2, 1], 9: [1, 1], 10: [1, 4], 11: [0, 4],
        12: [5, 2], 13: [4, 3], 14: [3, 2], 15: [4, 1], 16: [4, 0],
    }
    loop = [0, 16, 15, 14, 13, 12, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    return slit_graph(loop, coords)


def diamond_in_square():
    """A square plate with a diamond hole -- the primitive the diamond levels share.

    Topologically this is `Square hole`: four convex outer corners wanting two,
    four re-entrant hole corners wanting four, and a slit joining them. What
    differs is only the embedding -- the hole is turned forty-five degrees, so
    every edge a mesh draws between the hole and the plate runs at a slant to
    the boundary it starts from. Since the environment scores topology, the two
    levels have the same par and the same optimal connectivity; anything that
    makes this one harder is geometry, which is exactly what makes it a useful
    control against `Square hole`.

    The slit drops from the hole's bottom corner (270 degrees, wants four) to a
    flat point on the base edge (180 degrees, wants three). Both anchors can
    absorb the edge the slit forces on them, which is what keeps par at 0.
    """
    coords = {
        0: [0, 0], 1: [4, 0], 2: [4, 2], 3: [2, 4], 4: [4, 6], 5: [6, 4],
        6: [8, 0], 7: [8, 8], 8: [0, 8],
    }
    loop = [0, 1, 2, 3, 4, 5, 2, 1, 6, 7, 8]
    return slit_graph(loop, coords)


def diamond_tab():
    """A block with a thin tab and a diamond hole; the slit cuts right to the wall.

    `straight-holes #5` of the geogen held-out set, one of two domains the released
    agent does not solve. par is 0, so a perfect mesh exists; the agent gets to an
    all-quad mesh two vertices above the bound and stops.

    Worth playing with the slit in mind. The generator anchors it on the SHORTEST
    perpendicular cut, which runs one unit right from the hole's right corner
    (270 degrees, wants four) to a flat point on the right wall (180, wants three).
    That is legal -- neither anchor is forced off its target -- but the thin tab on
    the left sits at exactly the hole's height, so a longer cut that way may suit the
    domain better. Shortest and best are not the same thing, and only shortest has
    been tested.
    """
    coords = {
        0: [0, 2], 1: [2, 2], 2: [2, 0], 3: [7, 0], 4: [7, 5], 5: [2, 5],
        6: [2, 3], 7: [0, 3], 8: [6, 3], 9: [5, 4], 10: [4, 3], 11: [5, 2],
        12: [7, 3],
    }
    loop = [0, 1, 2, 3, 12, 8, 11, 10, 9, 8, 12, 4, 5, 6, 7]
    return slit_graph(loop, coords)


def desired_from_loop(loop, coords):
    """Desired degrees from the outline's own angles, at the current target.

    A slit outline visits its two endpoints twice, so the domain's total angle
    there is the sum of both corners it turns through. Deriving these instead of
    hardcoding them is what lets a hole shape be re-targeted; it reproduces the
    hand-written numbers exactly at 90 degrees.
    """
    count = len(loop)
    total = {}
    for index, vertex in enumerate(loop):
        previous, following = loop[index - 1], loop[(index + 1) % count]
        first = [coords[following][0] - coords[vertex][0],
                 coords[following][1] - coords[vertex][1]]
        second = [coords[previous][0] - coords[vertex][0],
                  coords[previous][1] - coords[vertex][1]]
        total[vertex] = total.get(vertex, 0.0) + utils.angle_between(first, second)
    return {v: utils.rounded_desired_degree(a, TARGET_ANGLE) for v, a in total.items()}


def slit_support_vertices(loop, coords, tolerance=1e-6):
    """Loop vertices that exist ONLY to carry the slit, not to shape the domain.

    A hole is represented by cutting the domain open along a zero-width slit, so
    the loop visits each of its two anchors twice. An anchor that lands on a
    genuine corner of the outline -- the diamond's 270-degree tip, say -- is part
    of the geometry and has to stay where it is. An anchor dropped mid-edge is
    not: it is a bookkeeping device for the representation, it happens to be
    flat, and pinning it costs real freedom. The Laplacian cannot open a slit
    whose ends are nailed down, which is why replayed hole instances come back
    folded: 62 percent of the folds sit on a pinned vertex.

    So the test is geometric, not structural: a doubled vertex whose material
    angle is a straight 180 degrees carries no corner and is returned here.
    """
    count = len(loop)
    seen = {}
    for index, vertex in enumerate(loop):
        seen.setdefault(vertex, []).append(index)
    support = set()
    for vertex, positions in seen.items():
        if len(positions) < 2:
            continue                      # visited once: an ordinary corner
        total = 0.0
        for index in positions:
            previous, following = loop[index - 1], loop[(index + 1) % count]
            first = [coords[following][0] - coords[vertex][0],
                     coords[following][1] - coords[vertex][1]]
            second = [coords[previous][0] - coords[vertex][0],
                      coords[previous][1] - coords[vertex][1]]
            total += utils.angle_between(first, second)
        # the slit contributes no angle of its own, so a mid-edge anchor sums to
        # a straight 180; anything else is a corner the outline actually turns
        if abs(total - 180.0) < 1e-6 or abs(total - 180.0) < tolerance:
            support.add(vertex)
    return support


def pinned_vertices(loop, coords):
    """Every loop vertex except the slit's own bookkeeping anchors."""
    return set(loop) - slit_support_vertices(loop, coords)


def slit_graph(loop, coords):
    """A `Tiler` for a slit outline, with only the real corners pinned."""
    graph = Tiler.from_face_loops([loop], coords,
                                  user_vertices=pinned_vertices(loop, coords))
    return graph, desired_from_loop(loop, coords)


def desired_options(angle, target_angle=None):
    """Every degree a corner of this angle would be content with.

    `rounded_desired_degree` is `round(a / theta) + 1`. When `a / theta` lands
    exactly on `k + 1/2` the two neighbouring corner treatments are equally
    admissible -- `k` elements of angle `a / k`, or `k + 1` of `a / (k+1)` --
    and rounding picks one by convention. Such a corner keeps both, so neither
    the score nor par depends on which integer happens to be even.
    """
    theta = TARGET_ANGLE if target_angle is None else target_angle
    ratio = angle / theta
    if abs(ratio - math.floor(ratio) - 0.5) > 1e-6:
        return (utils.rounded_desired_degree(angle, theta),)
    low = max(int(math.floor(ratio)) + 1, 2)
    high = max(int(math.ceil(ratio)) + 1, 2)
    return (low,) if low == high else (low, high)


def simple_polygon(coords):
    """Build an initializer from a CCW simple-polygon outline."""

    def make():
        coord_dict = dict(zip(range(len(coords)), [list(c) for c in coords]))
        loop = [list(range(len(coords)))]
        graph = Tiler.from_face_loops(loop, coord_dict)
        angles = utils.get_polygon_interior_angles(loop[0], graph.vertex_coordinates)
        desired = {
            v: utils.rounded_desired_degree(a, TARGET_ANGLE) for v, a in angles.items()
        }
        return graph, desired

    return make


def untangle_graph(graph, iters=8):
    """geo2d's optimiser on this mesh, in place. The port is game/js/untangle.js.

    A Laplacian sits at its own fixed point and cannot open a folded element,
    so it is the warm start rather than the answer. The domain's own corners are
    pinned; any other boundary vertex may only slide along the outline, and is
    projected back onto its arc afterwards where the outline is curved.

    geo2d is an optional dependency (`pip install -e ../geogen`). Without it the
    Laplacian result stands, which is strictly the harsher reading of the board.
    """
    try:
        import numpy as np

        from src.geo2d_bridge import import_geo2d, resmooth_mesh, tiler_faces
    except Exception:
        return False
    try:
        geo2d = import_geo2d()
        nodes, elements, index = tiler_faces(graph)
        corners = np.zeros(len(nodes), dtype=bool)
        for vertex, i in index.items():
            corners[i] = graph.is_user_defined_vertex(vertex)
        mesh = resmooth_mesh(geo2d.Mesh(nodes, elements), corners, iters=iters,
                             method="optimize", slide="optimize")
        before = _min_corner_quality(graph)
        restore = {vertex: graph.vertex_coordinates[vertex].copy() for vertex in index}
        for vertex, i in index.items():
            graph.set_vertex_coordinate(vertex, mesh.nodes[i])
        graph.snap_to_boundary_arcs()
        # `resmooth_mesh` guards on geo2d's own aspect-aware quality, which is
        # not the angle-based number this game scores with; a result can improve
        # one and hurt the other. Guard on the metric the player is judged by.
        if _min_corner_quality(graph) < before:
            for vertex, coordinate in restore.items():
                graph.set_vertex_coordinate(vertex, coordinate)
    except Exception:
        return False
    return True


def _min_corner_quality(graph):
    """Worst corner on the board, on the scale `_current_scores` reports."""
    angles = graph.half_edge_angles()
    if not angles:
        return 1.0
    ideal = math.sin(math.radians(TARGET_ANGLE))
    return min(math.sin(math.radians(a)) for a in angles.values()) / ideal


INITIALIZERS = {
    "L-shape": lambda: LEnv(TARGET_ANGLE)(),
    "T-bracket": simple_polygon(
        [(1, 0), (2, 0), (2, 2), (3, 2), (3, 3), (0, 3), (0, 2), (1, 2)]
    ),
    "I-bracket": simple_polygon(
        [(0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (3, 2),
         (3, 3), (0, 3), (0, 2), (1, 2), (1, 1), (0, 1)]
    ),
    "U-channel": simple_polygon(
        [(0, 0), (3, 0), (3, 2), (2, 2), (2, 1), (1, 1), (1, 2), (0, 2)]
    ),
    "Z-shape": simple_polygon(
        [(0, 0), (2, 0), (2, 1), (3, 1), (3, 2), (1, 2), (1, 1), (0, 1)]
    ),
    "Plus": simple_polygon(
        [(1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (2, 2),
         (2, 3), (1, 3), (1, 2), (0, 2), (0, 1), (1, 1)]
    ),
    "Staircase": simple_polygon(
        [(0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (1, 2), (1, 3), (0, 3)]
    ),
    "Triangle": simple_polygon([(0, 0), (1, 0), (0.5, math.sin(math.pi / 3))]),
    "Pentagon": simple_polygon(regular_polygon(5, math.pi / 2)),
    "Semicircle": simple_polygon(semicircle_polygon()),
    "Pac-Man": simple_polygon(pacman_polygon()),
    "Star": simple_polygon(star_polygon()),
    "Square hole": square_hole_ring,
    "Triforce ring": triforce_ring,
    "Gear": simple_polygon(gear_polygon()),
    "Diamond bracket": diamond_bracket,
    "Diamond tab": diamond_tab,
    "Diamond in square": diamond_in_square,
    # Curved levels. Carried natively -- one vertex per control point and the
    # arcs alongside, no discretisation -- so what each corner wants comes from
    # the tangent. A smooth boundary is the interesting case: every vertex of
    # Stadium wants three, so its par is 4 and the player has to discover that
    # a smooth disc cannot be meshed without four corners.
    "Plate with a hole": lambda: curved_levels.make_domain("Plate with a hole")(TARGET_ANGLE),
    "Notched plate": lambda: curved_levels.make_domain("Notched plate")(TARGET_ANGLE),
    "Snail": lambda: curved_levels.make_domain("Snail")(TARGET_ANGLE),
    # frozen out of geo2d (src/curved_domains.json): the curved levels that
    # are actually hard. The agent fails all three at best-of-25.
    "Tunnel": lambda: curved_levels.make_domain("Tunnel")(TARGET_ANGLE),
    "Fillet plate": lambda: curved_levels.make_domain("Fillet plate")(TARGET_ANGLE),
    "Hook": lambda: curved_levels.make_domain("Hook")(TARGET_ANGLE),
    "Bridge": lambda: curved_levels.make_domain("Bridge")(TARGET_ANGLE),
    "Clover": lambda: curved_levels.make_domain("Clover")(TARGET_ANGLE),
    "Boot": lambda: curved_levels.make_domain("Boot")(TARGET_ANGLE),
    "Skillet": lambda: curved_levels.make_domain("Skillet")(TARGET_ANGLE),
}

PORT = 8123
STATIC_DIR = Path(__file__).resolve().parent / "static"
# The game itself is the single-file build assembled from game/src by build.sh.
# `static/index.html` is the original thin client that drives the API routes
# below; it has had no feature since July and serving it was how `server.py`
# came to show a different game from the one everyone plays. It stays reachable
# at /api-client because the API is the way to play against this engine rather
# than the javascript port of it.
BUILT_PAGE = Path(__file__).resolve().parent / "mesh-quest.html"
BUILD_SCRIPT = Path(__file__).resolve().parent / "build.sh"


def built_page():
    """The current build, rebuilt first if game/src has moved on.

    Without this the server silently serves whatever was last built, which is
    the same failure as serving the old client -- just harder to spot.
    """
    sources = list((Path(__file__).resolve().parent / "src").glob("*")) + [
        Path(__file__).resolve().parent / "js" / "engine.js"]
    newest = max((path.stat().st_mtime for path in sources if path.is_file()),
                 default=0)
    if not BUILT_PAGE.exists() or BUILT_PAGE.stat().st_mtime < newest:
        import subprocess
        result = subprocess.run(["bash", str(BUILD_SCRIPT)],
                                capture_output=True, text=True)
        if result.returncode != 0 and not BUILT_PAGE.exists():
            raise RuntimeError(f"build.sh failed: {result.stderr[:300]}")
        print("rebuilt game/mesh-quest.html")
    return BUILT_PAGE.read_bytes()
RECORDS_PATH = Path(__file__).resolve().parent / "records.json"


def load_records():
    try:
        return json.loads(RECORDS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_records(records):
    RECORDS_PATH.write_text(json.dumps(records, indent=2))


class GameError(Exception):
    pass


class Game:
    def __init__(self, shape="L-shape"):
        self.shape = shape
        self.reset(shape)

    def reset(self, shape=None):
        shape = shape or self.shape
        if shape not in INITIALIZERS:
            raise GameError(f"Unknown shape: {shape}")
        graph, desired = INITIALIZERS[shape]()
        self.shape = shape
        self.graph = graph
        self.initial_desired = desired
        self.desired_sets = self._compute_desired_sets()
        self.undo_stack = []
        self.moves = 0
        self.records = load_records()
        self.record_just_set = False
        self.par = self._compute_par()

    def _compute_desired_sets(self):
        """Admissible degrees per original corner, from the start geometry.

        The start state is the raw outline: one face, so a half-edge's angle is
        its source corner's interior angle, summed where a slit revisits it.
        """
        g = self.graph
        total = {}
        for h in g.half_edge_list():
            v = g.source_vertex(h, tag=False)
            previous = g.source_vertex(g.previous_half_edge(h), tag=False)
            following = g.source_vertex(g.next_half_edge(h), tag=False)
            total[v] = total.get(v, 0.0) + utils.angle_between(
                edge_direction(g, v, following), edge_direction(g, v, previous))
        return {v: desired_options(a) for v, a in total.items()
                if v in self.initial_desired}

    def desired_degrees(self, vid):
        """Every degree this vertex would be content with."""
        options = self.desired_sets.get(vid)
        if options is not None:
            return options
        return (self.desired_degree(vid),)

    def vertex_defect(self, vid):
        degree = self.graph.vertex_degree(vid)
        return min(abs(degree - want) for want in self.desired_degrees(vid))

    def _current_scores(self):
        g = self.graph
        vertex_score = sum(self.vertex_defect(v) for v in g.vertex_list(tag=False))
        face_score = sum(
            abs(g.face_degree(f) - FACE_DESIRED) for f in g.face_list()
        )
        # measured against this element's ideal corner, not a right angle
        ideal = math.sin(math.radians(TARGET_ANGLE))
        min_quality = min(
            math.sin(math.radians(a)) / ideal for a in g.half_edge_angles().values()
        )
        return vertex_score, face_score, min_quality

    def _check_record(self):
        vertex_score, face_score, min_quality = self._current_scores()
        won = (
            vertex_score == self.par
            and face_score == 0
            and min_quality >= QUALITY_THRESHOLD
        )
        if not won:
            return
        best = self.records.get(self.shape)
        if best is None or self.moves < best["moves"]:
            self.records[self.shape] = {"moves": self.moves}
            self.record_just_set = True
            save_records(self.records)

    def _compute_par(self):
        """Topological lower bound on the vertex score for an all-quad mesh.

        For any all-quad mesh of this domain, discrete Gauss-Bonnet gives
        sum(generic_degree - degree) = 4*chi with generic degree 3 on the
        boundary and 4 in the interior. Hence sum(desired - degree) is the
        invariant C + 4*chi, and the L1 vertex score is at least its absolute
        value.
        """
        g = self.graph
        # the two options at a tie always differ by one, so taking `t` of them
        # high shifts the excess by exactly `t`: par is a scan over `t`
        corner_excess = 0
        num_ties = 0
        for vid in g.vertex_list(tag=False):
            generic = BOUNDARY_DESIRED if g.is_boundary_vertex(vid) else INTERIOR_DESIRED
            options = self.desired_degrees(vid)
            corner_excess += min(options) - generic
            num_ties += len(options) - 1
        num_edges = 0
        seen = set()
        for h in g.half_edge_list():
            if h in seen:
                continue
            seen.add(h)
            seen.add(g.twin_half_edge(h))
            num_edges += 1
        chi = len(g.vertex_list()) - num_edges + len(g.face_list())
        # the coefficient is the interior vertex's generic degree: for a mesh of
        # d-gons, sum_v (generic - degree) = 2*d*chi/(d-2)
        base = corner_excess + INTERIOR_DESIRED * chi
        return min(abs(base + taken) for taken in range(num_ties + 1))

    def desired_degree(self, vid):
        if vid in self.initial_desired:
            return self.initial_desired[vid]
        if self.graph.is_boundary_vertex(vid):
            return BOUNDARY_DESIRED
        return INTERIOR_DESIRED

    def halfedge(self, edge_id):
        h = (edge_id, self.graph.half_edge_tag)
        if not self.graph.is_half_edge(h):
            raise GameError(f"Unknown edge: {edge_id}")
        return h

    def delete_candidates(self, vid):
        g = self.graph
        vt = (vid, g.vertex_tag)
        for h in list(g._vertex_to_halfedge(vid)):
            if g.half_edges[h].source == vt and g.is_valid_delete_source_vertex(h):
                yield h

    def resolve_chord(self, a, b):
        """Find (halfedge, k) such that insert_half_edge connects vertex a to b."""
        g = self.graph
        for f in g.face_list():
            loop = g.generate_half_edge_face_loop(g.first_face_halfedge(f))
            srcs = [g.source_vertex(h, tag=False) for h in loop]
            n = len(loop)
            for i, h in enumerate(loop):
                if srcs[i] != a:
                    continue
                for k in range(n - 1):
                    if srcs[(i + k + 1) % n] == b and g.is_valid_edge_insert(h, k):
                        return h, k
        return None

    def apply_op(self, op, params, smooth_after=False):
        snapshot = deepcopy(self.graph)
        try:
            if op == "insert_vertex":
                h = self.halfedge(int(params["edge"]))
                self.graph.insert_vertex(h)
            elif op == "delete_edge":
                h = self.halfedge(int(params["edge"]))
                if not self.graph.is_valid_delete_half_edge(h):
                    raise GameError("This edge cannot be deleted")
                self.graph.delete_half_edge(h)
            elif op == "delete_vertex":
                vid = int(params["vertex"])
                h = next(self.delete_candidates(vid), None)
                if h is None:
                    raise GameError(
                        "Vertex not deletable: must be degree 2 and not an original corner"
                    )
                self.graph.delete_source_vertex(h)
            elif op == "insert_edge":
                a, b = int(params["a"]), int(params["b"])
                chord = self.resolve_chord(a, b) or self.resolve_chord(b, a)
                if chord is None:
                    raise GameError("Vertices must lie on the same face")
                self.graph.insert_half_edge(*chord)
            else:
                raise GameError(f"Unknown operation: {op}")

            if smooth_after:
                self.graph.smooth_vertices(num_iter=2)
        except GameError:
            self.graph = snapshot
            raise
        except Exception as exc:
            self.graph = snapshot
            raise GameError(str(exc) or "Operation not permitted")

        self.undo_stack.append(snapshot)
        self.moves += 1
        self.record_just_set = False
        self._check_record()

    def smooth(self, num_iter=3):
        self.undo_stack.append(deepcopy(self.graph))
        # The Laplacian is the untangler's warm start, not the whole of Smooth.
        # `game/js/engine.js` runs both, and the browser is what the player
        # sees, so a reference that stopped at the Laplacian would disagree with
        # it about quality -- and therefore about whether a board is WON.
        self.graph.smooth_vertices(num_iter=num_iter)
        untangle_graph(self.graph, iters=8)
        self.record_just_set = False
        self._check_record()

    def undo(self):
        if not self.undo_stack:
            raise GameError("Nothing to undo")
        self.graph = self.undo_stack.pop()
        self.moves = max(0, self.moves - 1)
        self.record_just_set = False

    def serialize(self):
        g = self.graph

        # scaled Jacobian per corner: sin of the interior angle at each
        # half-edge source. 1 = right angle, 0 = flat/collapsed, < 0 = concave.
        angles = g.half_edge_angles()
        corner_quality = {h: math.sin(math.radians(a)) for h, a in angles.items()}
        vertex_quality = {}
        face_quality = {}
        for h, q in corner_quality.items():
            vid = g.source_vertex(h, tag=False)
            fid = g.face(h, tag=False)
            vertex_quality[vid] = min(q, vertex_quality.get(vid, 1.0))
            face_quality[fid] = min(q, face_quality.get(fid, 1.0))
        min_quality = min(corner_quality.values())

        vertices = []
        for vid in g.vertex_list(tag=False):
            coord = g.vertex_coordinates[vid]
            vertices.append(
                {
                    "id": vid,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "degree": int(g.vertex_degree(vid)),
                    "desired": int(self.desired_degree(vid)),
                    "boundary": bool(g.is_boundary_vertex(vid)),
                    "user": bool(g.is_user_defined_vertex(vid)),
                    "deletable": next(self.delete_candidates(vid), None) is not None,
                    "quality": round(float(vertex_quality.get(vid, 1.0)), 3),
                }
            )

        edges = []
        seen = set()
        for h in g.half_edge_list():
            if h in seen:
                continue
            seen.add(h)
            seen.add(g.twin_half_edge(h))
            edges.append(
                {
                    "id": h[0],
                    "v1": g.source_vertex(h, tag=False),
                    "v2": g.target_vertex(h, tag=False),
                    "boundary": bool(g.half_edge_on_boundary(h)),
                    "deletable": bool(g.is_valid_delete_half_edge(h)),
                }
            )

        faces = []
        insertable_pairs = []
        for f in g.face_list():
            loop = g.generate_half_edge_face_loop(g.first_face_halfedge(f))
            srcs = [g.source_vertex(h, tag=False) for h in loop]
            faces.append(
                {
                    "id": f[0],
                    "degree": int(g.face_degree(f)),
                    "vertices": srcs,
                    "quality": round(float(face_quality.get(f[0], 1.0)), 3),
                }
            )
            n = len(loop)
            for i, h in enumerate(loop):
                for k in range(n - 1):
                    b = srcs[(i + k + 1) % n]
                    if srcs[i] != b and g.is_valid_edge_insert(h, k):
                        insertable_pairs.append([srcs[i], b])

        vertex_score = sum(abs(v["degree"] - v["desired"]) for v in vertices)
        face_score = sum(abs(f["degree"] - FACE_DESIRED) for f in faces)
        angle_score = float(
            sum(abs(a - TARGET_ANGLE) / TARGET_ANGLE for a in angles.values())
        )
        defect_sum = sum(v["desired"] - v["degree"] for v in vertices)

        return {
            "shape": self.shape,
            "shapes": list(INITIALIZERS.keys()),
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "insertable_pairs": insertable_pairs,
            "scores": {
                "vertex": int(vertex_score),
                "face": int(face_score),
                "angle": round(angle_score, 2),
                "defect_sum": int(defect_sum),
                "min_quality": round(float(min_quality), 3),
            },
            "par": self.par,
            "quality_threshold": QUALITY_THRESHOLD,
            "best": (self.records.get(self.shape) or {}).get("moves"),
            "new_record": self.record_just_set,
            "moves": self.moves,
            "can_undo": bool(self.undo_stack),
        }


GAME = Game()
LOCK = Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_state(self):
        self._send_json(200, GAME.serialize())

    def do_GET(self):
        if self.path in ("/", "/index.html", "/api-client"):
            body = ((STATIC_DIR / "index.html").read_bytes()
                    if self.path == "/api-client" else built_page())
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            with LOCK:
                self._send_state()
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        try:
            with LOCK:
                if self.path == "/api/op":
                    GAME.apply_op(
                        body.get("op"),
                        body.get("params", {}),
                        smooth_after=bool(body.get("smooth")),
                    )
                elif self.path == "/api/undo":
                    GAME.undo()
                elif self.path == "/api/smooth":
                    GAME.smooth(num_iter=int(body.get("iters", 3)))
                elif self.path == "/api/new":
                    GAME.reset(body.get("shape") or GAME.shape)
                else:
                    self._send_json(404, {"error": "not found"})
                    return
                self._send_state()
        except GameError as exc:
            self._send_json(400, {"error": str(exc)})


def main():
    built_page()                      # fail loudly here, not on the first request
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Mesh Quest running at http://127.0.0.1:{PORT}")
    print(f"  the API client lives at http://127.0.0.1:{PORT}/api-client")
    server.serve_forever()


if __name__ == "__main__":
    main()
