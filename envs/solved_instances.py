"""Instances that come with a certified solution, generated without search.

The generator runs the problem backwards. A polyomino is already a par-optimal
all-quad mesh -- every interior vertex has degree 4, every boundary vertex has
degree equal to the desired degree its 90/180/270 corner implies -- so instead
of searching for a solution we start from one and undo it, move by move, until
only the raw polygon is left. Reversing that walk gives a move sequence in the
environment's own action vocabulary that takes the polygon to par.

The polygon is then deformed (affine map plus per-corner jitter) as far as the
angle bins allow, which turns a rectilinear seed into a skewed, irregular
polygon whose known solution is still valid: the desired degree of a corner
only depends on which 90-degree bin its interior angle falls in.

Every instance is validated by replaying its moves through the real
environment, so nothing enters the dataset that the agent could not itself
have played.
"""

import math
import os
import pickle
from copy import deepcopy

import numpy as np

import envs.polygon_utils as utils
from src.tiler import Tiler

# grid-point neighbourhood: the four cells touching a lattice point
_CELL_OFFSETS = [(-1, -1), (0, -1), (0, 0), (-1, 0)]
_NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def random_polyomino(num_cells, rng, max_tries=200, spindliness=None):
    """A connected set of grid cells with no pinch points and no holes.

    `spindliness` biases growth towards cells that touch the shape only once,
    which grows combs and staircases rather than blobs. That matters because
    the outline degree, not the cell count, is what has to reach the held-out
    levels: Gear is a 24-gon whose solution is eleven quads, and a blobby
    eleven-cell polyomino has an outline half that size.
    """
    if spindliness is None:
        spindliness = float(rng.uniform(0.0, 3.0))
    for _ in range(max_tries):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < num_cells and frontier:
            candidates = sorted(frontier)
            contacts = np.array([
                sum((c[0] + dx, c[1] + dy) in cells for dx, dy in _NEIGHBOURS)
                for c in candidates], dtype=float)
            weights = np.exp(-spindliness * (contacts - 1.0))
            cell = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
            frontier.discard(cell)
            cells.add(cell)
            for dx, dy in _NEIGHBOURS:
                candidate = (cell[0] + dx, cell[1] + dy)
                if candidate not in cells:
                    frontier.add(candidate)
        if len(cells) == num_cells and _is_simple_polyomino(cells):
            return cells
    raise RuntimeError(f"could not build a simple polyomino of {num_cells} cells")


def random_annulus(num_cells, rng, max_tries=200):
    """A ring of grid cells around a single hole.

    Grown outward from the 3x3 ring, which is the smallest polyomino with an
    interior hole. The backward walk produces the slit form on its own: once
    every other interior edge is gone, the one edge joining the hole to the rim
    has the same face on both sides and so cannot be deleted, which is exactly
    the doubled edge the game's hole levels start from.
    """
    hole = (1, 1)
    cells = {(i, j) for i in range(3) for j in range(3)} - {hole}
    frontier = set()
    for cell in cells:
        for dx, dy in _NEIGHBOURS:
            candidate = (cell[0] + dx, cell[1] + dy)
            if candidate not in cells and candidate != hole:
                frontier.add(candidate)

    tries = 0
    while len(cells) < num_cells and frontier and tries < max_tries:
        tries += 1
        cell = tuple(rng.permutation(sorted(frontier))[0])
        frontier.discard(cell)
        if not _is_annulus(cells | {cell}):
            continue
        cells.add(cell)
        for dx, dy in _NEIGHBOURS:
            candidate = (cell[0] + dx, cell[1] + dy)
            if candidate not in cells and candidate != hole:
                frontier.add(candidate)

    if len(cells) >= 8 and _is_annulus(cells):
        return cells
    raise RuntimeError(f"could not build an annulus of {num_cells} cells")


def _is_annulus(cells):
    points = _grid_points(cells)
    edges = set()
    for x, y in cells:
        corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        for i in range(4):
            edges.add(frozenset((corners[i], corners[(i + 1) % 4])))
    if len(points) - len(edges) + len(cells) != 0:
        return False
    for point in points:
        incident = _incident_cells(point, cells)
        if len(incident) == 2:
            (ax, ay), (bx, by) = incident
            if abs(ax - bx) == 1 and abs(ay - by) == 1:
                return False
    return True


def _incident_cells(point, cells):
    x, y = point
    return [(x + dx, y + dy) for dx, dy in _CELL_OFFSETS if (x + dx, y + dy) in cells]


def _grid_points(cells):
    points = set()
    for x, y in cells:
        points.update({(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)})
    return points


def _is_simple_polyomino(cells):
    """No pinch points, and simply connected (Euler characteristic 1)."""
    points = _grid_points(cells)
    edges = set()
    for x, y in cells:
        corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        for i in range(4):
            edges.add(frozenset((corners[i], corners[(i + 1) % 4])))
    if len(points) - len(edges) + len(cells) != 1:
        return False
    for point in points:
        incident = _incident_cells(point, cells)
        if len(incident) == 2:
            (ax, ay), (bx, by) = incident
            if abs(ax - bx) == 1 and abs(ay - by) == 1:
                return False  # two cells meeting only at this corner
    return True


def polyomino_mesh(cells, target_angle=90, rng=None, flat_corner_probability=0.0):
    """Par-optimal all-quad mesh of a polyomino, plus its polygon corners.

    A lattice point touched by `k` cells has interior angle `90k` and mesh
    degree `k + 1`, and `rounded_desired_degree(90k, 90)` is exactly `k + 1`,
    so the mesh sits at par by construction. The polygon the agent starts from
    is the boundary cycle through the points that are not flat: the flat ones
    are vertices the agent inserts.
    """
    points = sorted(_grid_points(cells))
    point_id = {p: i for i, p in enumerate(points)}
    coordinates = {i: [float(p[0]), float(p[1])] for p, i in point_id.items()}

    loops = []
    for x, y in sorted(cells):
        corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        loops.append([point_id[c] for c in corners])

    corner_ids, desired = set(), {}
    for point in points:
        incident = len(_incident_cells(point, cells))
        vid = point_id[point]
        if incident == 4:
            desired[vid] = utils.rounded_desired_degree(360, target_angle) - 1
            continue
        angle = 90 * incident
        desired[vid] = utils.rounded_desired_degree(angle, target_angle)
        if incident != 2:
            corner_ids.add(vid)
        elif rng is not None and rng.random() < flat_corner_probability:
            # a flat boundary point promoted to a polygon corner: the agent no
            # longer inserts it, the polygon gains a vertex, and its degree can
            # now be odd -- which a polyomino outline never is on its own
            corner_ids.add(vid)

    graph = Tiler.from_face_loops(loops, coordinates, user_vertices=corner_ids)
    return graph, desired, corner_ids


def polyomino_triangle_mesh(cells, target_angle=60, rng=None,
                            flat_corner_probability=0.0):
    """Par-optimal all-triangle mesh of a polyomino: the grid, split one way.

    This is the family the generator could not reach. Every corner of a
    rectilinear domain is 90 or 270 degrees, and at the triangle target both are
    exact rounding ties, which `_angle_window` deliberately backs away from --
    so no amount of angle sampling produces a rectilinear domain at 60 degrees,
    and the held-out polyominoes were out of distribution entirely.

    Splitting every cell along the SAME diagonal is what makes the result
    optimal. An interior lattice point then keeps its four grid neighbours and
    gains exactly two diagonal ones -- the cell up-right of it and the cell
    down-left -- for degree 6, which is what an interior vertex wants at this
    target. On the boundary the diagonal reaches only one of the two cells at a
    flat point, giving degree 4; a convex corner comes out at 2 or 3 and a
    re-entrant one at 5 or 6, and both of those lie inside the tie set the
    corner admits. So every vertex is content and the mesh sits at par 0.
    """
    points = sorted(_grid_points(cells))
    point_id = {p: i for i, p in enumerate(points)}
    coordinates = {i: [float(p[0]), float(p[1])] for p, i in point_id.items()}

    loops = []
    for x, y in sorted(cells):
        lower_left, lower_right = (x, y), (x + 1, y)
        upper_right, upper_left = (x + 1, y + 1), (x, y + 1)
        # the shared diagonal runs lower-left to upper-right in every cell
        loops.append([point_id[lower_left], point_id[lower_right],
                      point_id[upper_right]])
        loops.append([point_id[lower_left], point_id[upper_right],
                      point_id[upper_left]])

    corner_ids, desired = set(), {}
    for point in points:
        incident = len(_incident_cells(point, cells))
        vid = point_id[point]
        if incident == 4:
            desired[vid] = utils.rounded_desired_degree(360, target_angle) - 1
            continue
        desired[vid] = utils.rounded_desired_degree(90 * incident, target_angle)
        if incident != 2:
            corner_ids.add(vid)
        elif rng is not None and rng.random() < flat_corner_probability:
            corner_ids.add(vid)

    graph = Tiler.from_face_loops(loops, coordinates, user_vertices=corner_ids)
    return graph, desired, corner_ids


# --------------------------------------------------------------------------
# triangular lattice: what the polyomino grid is to quads
# --------------------------------------------------------------------------

_ROOT3_2 = math.sqrt(3) / 2


def _tri_point(i, j):
    return (i + 0.5 * j, j * _ROOT3_2)


def _tri_corners(cell):
    kind, i, j = cell
    if kind == "u":
        return [(i, j), (i + 1, j), (i, j + 1)]
    return [(i + 1, j), (i + 1, j + 1), (i, j + 1)]


def _tri_neighbours(cell):
    kind, i, j = cell
    if kind == "u":
        return [("d", i, j), ("d", i - 1, j), ("d", i, j - 1)]
    return [("u", i, j), ("u", i + 1, j), ("u", i, j + 1)]


def _tri_cells_of(point):
    """The six cells around a lattice point, in counter-clockwise order."""
    i, j = point
    return [("u", i, j), ("d", i - 1, j), ("u", i - 1, j),
            ("d", i - 1, j - 1), ("u", i, j - 1), ("d", i, j - 1)]


def _tri_points(cells):
    return {p for c in cells for p in _tri_corners(c)}


def _tri_is_valid(cells, expect_euler=1):
    """Contiguous around every point, and of the expected topology."""
    points = _tri_points(cells)
    edges = set()
    for cell in cells:
        corners = _tri_corners(cell)
        for k in range(3):
            edges.add(frozenset((corners[k], corners[(k + 1) % 3])))
    if len(points) - len(edges) + len(cells) != expect_euler:
        return False
    for point in points:
        ring = [c in cells for c in _tri_cells_of(point)]
        count = sum(ring)
        if count == 6:
            continue
        # the incident cells must form one contiguous arc, or the point is a
        # pinch and the mesh is not a manifold there
        starts = sum(1 for k in range(6) if ring[k] and not ring[k - 1])
        if starts != 1:
            return False
    return True


def random_polyiamond(num_cells, rng, max_tries=200, spindliness=None, hole=False):
    """A connected patch of the triangular lattice, optionally with one hole.

    The direct analogue of `random_polyomino`: a lattice point touched by `k`
    triangles has interior angle `60k` and mesh degree `k + 1`, and
    `rounded_desired_degree(60k, 60)` is exactly `k + 1`, so the patch sits at
    par by construction.
    """
    if spindliness is None:
        # a hole needs a cell whose three neighbours are all present, which a
        # comb almost never has: grow compactly when one is wanted
        spindliness = float(rng.uniform(-1.5, -0.5) if hole else rng.uniform(0.0, 2.5))
    for _ in range(max_tries):
        cells = {("u", 0, 0)}
        frontier = set(_tri_neighbours(("u", 0, 0)))
        while len(cells) < num_cells and frontier:
            candidates = sorted(frontier)
            contacts = np.array([
                sum(n in cells for n in _tri_neighbours(c)) for c in candidates],
                dtype=float)
            weights = np.exp(-spindliness * (contacts - 1.0))
            cell = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
            frontier.discard(cell)
            cells.add(cell)
            frontier |= {n for n in _tri_neighbours(cell) if n not in cells}
        if len(cells) != num_cells or not _tri_is_valid(cells):
            continue
        if not hole:
            return cells
        # punch out one interior cell: all three neighbours present, and every
        # corner of it fully surrounded, so removing it leaves a clean ring
        interior = [c for c in cells if all(n in cells for n in _tri_neighbours(c))]
        for cell in (interior[k] for k in rng.permutation(len(interior))):
            remaining = cells - {cell}
            if _tri_is_valid(remaining, expect_euler=0):
                return remaining
    raise RuntimeError(f"could not build a polyiamond of {num_cells} cells")


def polyiamond_mesh(cells, target_angle=60, rng=None, flat_corner_probability=0.0):
    """Par-optimal all-triangle mesh of a lattice patch, plus its polygon corners."""
    points = sorted(_tri_points(cells))
    point_id = {p: k for k, p in enumerate(points)}
    coordinates = {k: list(_tri_point(*p)) for p, k in point_id.items()}
    loops = [[point_id[p] for p in _tri_corners(c)] for c in sorted(cells)]

    flat_count = int(round(180.0 / target_angle))   # 3 triangles make a flat corner
    corner_ids, desired = set(), {}
    for point in points:
        incident = sum(1 for c in _tri_cells_of(point) if c in cells)
        vid = point_id[point]
        if incident == 6:
            desired[vid] = utils.rounded_desired_degree(360, target_angle) - 1
            continue
        desired[vid] = utils.rounded_desired_degree(target_angle * incident, target_angle)
        if incident != flat_count:
            corner_ids.add(vid)
        elif rng is not None and rng.random() < flat_corner_probability:
            corner_ids.add(vid)

    graph = Tiler.from_face_loops(loops, coordinates, user_vertices=corner_ids)
    return graph, desired, corner_ids


def triangle_fan_mesh(num_faces, rng, target_angle=60, flat_corner_probability=0.0):
    """`m` triangles fanned from one hub: par is `|6 - m|`.

    The triangle counterpart of `polar_mesh`, and the family that supplies odd
    outlines and par above zero. Every boundary vertex is shared by two
    triangles, so it wants degree 3, which pins its angle to [90, 150) and caps
    the family at eleven triangles; the hub has degree `m` against a desired 6.
    """
    assert 4 <= num_faces <= 11, "corners would stop wanting degree 3"
    m = num_faces
    interior = 180.0 * (m - 2) / m
    low, high = _bin_bounds(3, target_angle)
    spread = min(interior - low, high - interior) * 0.6
    angles = interior + rng.uniform(-spread, spread, size=m)
    angles = angles - (angles.sum() - 180.0 * (m - 2)) / m
    outline = polygon_from_angles(angles, rng)
    if outline is None:
        return None

    centre = m
    coordinates = {k: outline[k] for k in range(m)}
    coordinates[centre] = outline.mean(axis=0)
    # a fan's boundary vertex has two incident faces, not the three a flat
    # boundary point has: its angle is 2 * target, so it wants degree 3
    desired = {k: utils.rounded_desired_degree(2 * target_angle, target_angle)
               for k in range(m)}
    desired[centre] = utils.rounded_desired_degree(360, target_angle) - 1
    loops = [[k, (k + 1) % m, centre] for k in range(m)]
    corner_ids = set(range(m))
    graph = Tiler.from_face_loops(loops, coordinates, user_vertices=corner_ids)
    return graph, desired, corner_ids


def polar_mesh(num_quads, rng, flat_corner_probability=0.35, target_angle=90):
    """`m` quads around one interior vertex: the seed family polyominoes miss.

    The boundary alternates between corners of a single quad (degree 2) and
    vertices shared by two (degree 3), and the hub has degree `m`, so the
    vertex score -- and therefore par -- is `|4 - m|`. That is how a domain
    with par above zero and an *odd* number of corners arises: at m = 3 the
    outline is a triangle with par 1, at m = 5 a pentagon with par 1, neither
    of which any polyomino outline can be.

    Corners must stay under 135 degrees to keep wanting degree 2, which caps
    the family at seven quads.
    """
    assert 3 <= num_quads <= 7, "corners would stop wanting degree 2"
    m = num_quads
    interior = 180.0 * (m - 2) / m
    spread = min(0.5 * (134.0 - interior), 25.0)
    angles = interior + rng.uniform(-spread, spread, size=m)
    angles = angles - (angles.sum() - 180.0 * (m - 2)) / m
    outline = polygon_from_angles(angles, rng)
    if outline is None:
        return None

    coordinates, loops = {}, []
    corner_ids = set()
    desired = {}
    centre = m * 2
    coordinates[centre] = outline.mean(axis=0)
    desired[centre] = utils.rounded_desired_degree(360, target_angle) - 1
    for k in range(m):
        corner, mid = 2 * k, 2 * k + 1
        coordinates[corner] = outline[k]
        coordinates[mid] = 0.5 * (outline[k] + outline[(k + 1) % m])
        desired[corner] = 2
        desired[mid] = utils.rounded_desired_degree(180, target_angle)
        corner_ids.add(corner)
    for k in range(m):
        corner, mid, previous_mid = 2 * k, 2 * k + 1, (2 * k - 1) % (2 * m)
        loops.append([corner, mid, centre, previous_mid])
    for k in range(m):
        if rng.random() < flat_corner_probability:
            corner_ids.add(2 * k + 1)

    graph = Tiler.from_face_loops(loops, coordinates, user_vertices=corner_ids)
    return graph, desired, corner_ids


# --------------------------------------------------------------------------
# backward walk
# --------------------------------------------------------------------------

def _face_sources(graph, half_edge):
    loop = graph.generate_half_edge_face_loop(half_edge)
    return loop, [graph.source_vertex(h, tag=False) for h in loop]


def _chord_between(graph, u, v, max_edge_addition_steps, chord_offset,
                   min_face_degree=3):
    """The `(half_edge, local_action)` whose chord joins vertices `u` and `v`.

    The same chord can be inserted from either end -- the two `k` values sum to
    the face degree minus two -- so try both and take the one the action space
    can express.
    """
    for face in graph.face_list():
        loop, sources = _face_sources(graph, graph.first_face_halfedge(face))
        if u not in sources or v not in sources:
            continue
        n = len(loop)
        i, j = sources.index(u), sources.index(v)
        for k, start in sorted([((j - i - 1) % n, loop[i]), ((i - j - 1) % n, loop[j])]):
            local = k - chord_offset
            if 0 <= local < max_edge_addition_steps and graph.is_valid_chord_insert(
                    start, k, min_face_degree):
                return start, local
    return None


def _half_edge_between(graph, u, v):
    for h in graph.half_edge_list():
        if graph.source_vertex(h, tag=False) == u and graph.target_vertex(h, tag=False) == v:
            return h
    for h in graph.half_edge_list():
        if graph.source_vertex(h, tag=False) == v and graph.target_vertex(h, tag=False) == u:
            return h
    return None


def accepts_as_solved(env, gate="laplacian", degeneracy_threshold=0.1,
                      untangle_iterations=12, warm_start_iterations=5):
    """Is this mesh a solution? Topology, plus whichever quality test is in force.

    `laplacian` is the historical gate: the env's own `is_at_par`, which demands
    `min_quality >= quality_threshold` on the LAPLACIAN-smoothed mesh. That is
    the same test the win condition used, and it discards certified instances
    whose optimal topology the Laplacian simply cannot draw -- on a hole-only
    run it rejects more than a third of all candidates, and it rejects the k=4
    pinwheel (0.196) while accepting k=5 (0.449) and k=6 (0.603), so the family
    it removes is not a random sample of the space.

    `untangle` matches the evaluator instead: topology at par, then geo2d's
    optimiser on a copy, then a low DEGENERACY threshold. A topology that admits
    no non-degenerate embedding is still refused; one that merely needs a better
    smoother is kept, because smoothing and refinement are downstream tools.
    """
    if env.global_face_score != 0 or env.global_vertex_score != env.par:
        return False
    if gate == "topology":
        return True
    if gate != "untangle":
        return env.is_at_par()
    try:
        from copy import deepcopy

        from src.geo2d_bridge import resmooth_env
        probe = deepcopy(env)
        if warm_start_iterations:
            probe.graph.smooth_vertices(num_iter=warm_start_iterations)
            probe._update_half_edge_angles()
        resmooth_env(probe, iters=untangle_iterations, method="optimize")
        return probe.min_element_quality() >= degeneracy_threshold
    except Exception:
        # geogen is optional and the optimiser can fail on a tangle; falling
        # back to the raw angles is the harsher test, never the laxer one
        return env.min_element_quality() >= degeneracy_threshold


def backward_walk(graph, desired, max_edge_addition_steps=3, chord_offset=1,
                  rng=None, max_moves=200, face_desired_degree=4, on_step=None):
    """Undo an at-par mesh down to its raw polygon.

    Returns `(polygon_graph, polygon_desired, moves)` where `moves` is the
    forward sequence -- the reverse of the deletions -- described in terms of
    vertex identities rather than half-edge ids, so it survives the rebuild of
    the polygon from scratch. Returns None if the walk stalls before it gets
    down to a single face.
    """
    rng = rng or np.random.default_rng()
    graph = deepcopy(graph)
    desired = dict(desired)
    moves = []
    if on_step is not None:
        on_step(graph, desired)          # the seed, before anything is undone

    for _ in range(max_moves):
        quad_merges, other_merges, vertex_moves = [], [], []
        for h in graph.half_edge_list():
            if graph.is_valid_delete_source_vertex(h):
                vertex_moves.append(h)
            if graph.is_valid_delete_half_edge(h):
                degree_a = graph.face_degree(graph.face(h))
                degree_b = graph.face_degree(graph.face(graph.twin_half_edge(h)))
                small, large = min(degree_a, degree_b), max(degree_a, degree_b)
                local = small - 2 - chord_offset
                if not (0 <= local < max_edge_addition_steps):
                    continue
                (quad_merges if small == face_desired_degree
                 else other_merges).append((h, large))

        # A merge is only expressible while one side is still small, so the walk
        # has to absorb target-sized faces into a single growing region: let two
        # mid-sized regions form and neither can ever take the other. Prefer
        # merging one into the largest face there is, and keep the deletions of
        # degree-2 vertices for when no such merge is left.
        if quad_merges:
            sizes = np.array([float(size) for _, size in quad_merges])
            weights = np.exp((sizes - sizes.max()) / 2.0)
            pick = int(rng.choice(len(quad_merges), p=weights / weights.sum()))
            kind, h = "edge", quad_merges[pick][0]
        elif vertex_moves:
            kind, h = "vertex", vertex_moves[int(rng.integers(len(vertex_moves)))]
        elif other_merges:
            kind, h = "edge", other_merges[int(rng.integers(len(other_merges)))][0]
        else:
            break

        if kind == "edge":
            u = graph.source_vertex(h, tag=False)
            v = graph.target_vertex(h, tag=False)
            graph.delete_half_edge(h)
            moves.append(("chord", u, v, None))
        else:
            previous = graph.previous_half_edge(h)
            deleted = graph.source_vertex(h, tag=False)
            u = graph.source_vertex(previous, tag=False)
            v = graph.target_vertex(h, tag=False)
            graph.delete_source_vertex(h)
            desired.pop(deleted, None)
            moves.append(("vertex", u, v, deleted))
        if on_step is not None:
            on_step(graph, desired)

    if len(graph.face_list()) != 1:
        return None
    moves.reverse()
    return graph, desired, moves


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def _bin_bounds(degree, target_angle=90.0):
    """Interior-angle interval that `rounded_desired_degree` maps to `degree`.

    `rounded_desired_degree` is `round(a / theta) + 1`, so the bin is one
    `theta` wide and centred on `theta * (degree - 1)`. Degree 2 absorbs
    everything below, because the function clamps there.
    """
    if degree <= 2:
        return 0.0, target_angle * 1.5
    return target_angle * (degree - 1.5), target_angle * (degree - 0.5)


def polygon_from_graph(graph):
    """The boundary cycle of a single-face graph, in loop order.

    A domain with a hole comes back with the slit vertices appearing twice,
    exactly as the game's hole levels are written.
    """
    faces = graph.face_list()
    assert len(faces) == 1, "expected the raw polygon"
    loop = graph.generate_half_edge_face_loop(graph.first_face_halfedge(faces[0]))
    return [graph.source_vertex(h, tag=False) for h in loop]


def deform_polygon(coordinates, loop, mesh_degree, rng, strength=0.35,
                   min_quality=0.4, max_tries=25, target_angle=90.0):
    """Affine map plus per-corner jitter; the desired degrees follow the angles.

    A corner's desired degree is a function of which 90-degree bin its interior
    angle falls in, so a rectilinear seed can be skewed and stretched a long
    way before anything changes -- and when a corner *does* cross into the next
    bin, the instance is still at par as long as every crossing goes the same
    way, because the vertex score and its Gauss-Bonnet bound then move
    together. That is what produces non-rectilinear instances with par above
    zero; the caller verifies which ones survived by replaying the solution.

    Quality is checked against the *mesh* degree, not the new desired degree:
    the number of elements meeting at a corner is fixed by the solution, so
    bumping what the corner wants does not change how finely its angle is cut.
    """
    base = np.array([coordinates[v] for v in loop], dtype=float)
    base = base - base.mean(axis=0)
    lower_quality = _element_angle_bound(min_quality, target_angle)

    for _ in range(max_tries):
        theta = rng.uniform(0, 2 * np.pi)
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        scale = np.diag(rng.uniform(1 - strength, 1 + strength, size=2))
        shear = np.array([[1.0, rng.uniform(-strength, strength)], [0.0, 1.0]])
        points = base @ (scale @ shear).T @ rotation.T
        points = points + rng.normal(scale=strength * 0.3, size=points.shape)

        candidate = {v: points[i] for i, v in enumerate(loop)}
        angles = utils.get_polygon_interior_angles(loop, candidate)
        desired, ok = {}, True
        for v, angle in angles.items():
            element_angle = angle / max(mesh_degree[v] - 1, 1)
            if element_angle < lower_quality or element_angle > 180 - lower_quality:
                ok = False
                break
            desired[v] = utils.rounded_desired_degree(angle, target_angle)
        if ok and _is_simple_polygon(points):
            return candidate, desired
    return None, None


def polygon_from_angles(interior_angles, rng, reference_lengths=None, max_tries=40):
    """Realise a simple polygon with exactly these interior angles.

    Interior angles fix the edge directions; what is left is a choice of
    positive edge lengths satisfying one closure equation in the plane. That is
    two linear constraints on `n` unknowns, so project a random positive length
    vector onto the closure subspace and retry until every length stays
    positive and the outline does not cross itself.
    """
    angles = np.asarray(interior_angles, dtype=float)
    n = len(angles)
    turns = 180.0 - angles
    if abs(turns.sum() - 360.0) > 1e-6:
        return None
    # the turn at vertex i separates edge i-1 from edge i, so edge 0 sets the
    # frame and the running heading picks up turns[1:]
    headings = np.radians(np.concatenate([[0.0], np.cumsum(turns[1:])]))
    directions = np.stack([np.cos(headings), np.sin(headings)], axis=1)  # (n, 2)

    basis = directions.T  # (2, n)
    projector = np.eye(n) - np.linalg.pinv(basis) @ basis
    if reference_lengths is None:
        reference = np.ones(n)
    else:
        reference = np.asarray(reference_lengths, dtype=float)
        reference = reference / max(reference.mean(), 1e-9)
    for attempt in range(max_tries):
        # stay near the seed's own edge lengths: the mesh was built for that
        # shape, and a side that should be three cells long inverts elements
        # if it is forced down to one. The nearest closing length vector to a
        # target is its projection onto the null space of the closure map.
        noise = 0.2 * (attempt < max_tries // 2)
        target = reference * (1.0 + rng.uniform(-noise, noise, size=n))
        lengths = projector @ target
        if lengths.min() <= 1e-3:
            continue
        points = np.zeros((n, 2))
        for i in range(1, n):
            points[i] = points[i - 1] + lengths[i - 1] * directions[i - 1]
        if np.linalg.norm(points[-1] + lengths[-1] * directions[-1] - points[0]) > 1e-6:
            continue
        if not _is_simple_polygon(points):
            continue
        scale = np.abs(points - points.mean(axis=0)).max()
        return (points - points.mean(axis=0)) / max(scale, 1e-9)
    return None


def _element_angle_bound(min_quality, target_angle):
    """Smallest element corner angle the quality threshold allows.

    Quality is `sin(angle) / sin(target)`, so the bound moves with the target:
    23.6 degrees for quads at 0.4, 20.3 for triangles.
    """
    ideal = np.sin(np.radians(target_angle))
    return np.degrees(np.arcsin(np.clip(min_quality * ideal, 0.0, 1.0)))


def _angle_window(mesh_degree, desired, min_quality=0.4, target_angle=90.0):
    """Interior angles that give `desired` at a corner of this mesh degree."""
    lower_quality = _element_angle_bound(min_quality, target_angle)
    low, high = _bin_bounds(desired, target_angle)
    low = max(low, 1.0)
    splits = max(mesh_degree - 1, 1)
    low = max(low, splits * lower_quality)
    high = min(high, splits * (180.0 - lower_quality), 359.0)
    if high - low < 6.0:
        return None
    return low + 2.0, high - 2.0


def sample_angles(mesh_degrees, bumps, rng, reference=None, max_deviation=None,
                  min_quality=0.4, max_tries=60, target_angle=90.0):
    """Angles in each corner's window, summing to what a polygon requires.

    `bumps[i]` shifts what corner `i` asks for by that many degrees of desired
    degree; every bump in one instance points the same way, which is what keeps
    the vertex score equal to its Gauss-Bonnet bound and so keeps the seed at
    par.
    """
    n = len(mesh_degrees)
    target_sum = 180.0 * (n - 2)
    if max_deviation is None:
        max_deviation = target_angle * 0.4
    windows = []
    for degree, bump in zip(mesh_degrees, bumps):
        window = _angle_window(degree, degree + bump, min_quality, target_angle)
        if window is None:
            return None
        windows.append(window)
    lows = np.array([w[0] for w in windows])
    highs = np.array([w[1] for w in windows])
    if reference is not None:
        # keep the outline near the seed's shape; a corner that has to cross
        # into the next bin gets the extra room that crossing needs
        reference = np.asarray(reference, dtype=float)
        room = np.array([max_deviation + (target_angle * 1.06 if bump else 0.0)
                         for bump in bumps])
        lows = np.maximum(lows, reference - room)
        highs = np.minimum(highs, reference + room)
        if (highs <= lows).any():
            return None
    if not (lows.sum() <= target_sum <= highs.sum()):
        return None

    for _ in range(max_tries):
        angles = lows + rng.random(n) * (highs - lows)
        for _ in range(200):
            residual = target_sum - angles.sum()
            if abs(residual) < 1e-7:
                break
            slack = (highs - angles) if residual > 0 else (angles - lows)
            total = slack.sum()
            if total < 1e-9:
                break
            angles = angles + np.sign(residual) * slack * min(abs(residual) / total, 1.0)
        if abs(angles.sum() - target_sum) < 1e-6 and (angles > lows - 1e-6).all() \
                and (angles < highs + 1e-6).all():
            return angles
    return None


def _is_simple_polygon(points):
    n = len(points)

    def crosses(p1, p2, p3, p4):
        def side(a, b, c):
            return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
        d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
        return d1 != d2 and d3 != d4

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if crosses(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                return False
    return True


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

class SolvedInstance:
    """A polygon, its desired degrees, and a certified solution.

    The moves are stored as vertex identities, not half-edge ids: polygon
    corners are numbered 0..n-1 in loop order and every vertex the solution
    inserts takes the next number in creation order, which is exactly how
    `Tiler` allocates them when the polygon is rebuilt. So the sequence
    replays in a fresh graph without any id bookkeeping surviving in the file.
    """

    __slots__ = ("loop", "coordinates", "desired", "moves", "par",
                 "face_desired_degree", "arcs")

    def __init__(self, loop, coordinates, desired, moves, par, face_desired_degree=4,
                 arcs=None):
        self.loop = loop
        self.coordinates = coordinates
        self.desired = desired
        self.moves = moves
        self.par = par
        self.face_desired_degree = face_desired_degree
        # Boundary edges bent into arcs: {(a, b): (Arc, start_vertex)}. Empty for
        # a straight-edged instance, and absent entirely from pickles written
        # before curved domains existed -- hence `curved` reads it defensively.
        self.arcs = arcs or {}

    @property
    def curved(self):
        return bool(getattr(self, "arcs", None))

    @property
    def target(self):
        """The face degree this instance is a solution for; 4 in older datasets."""
        return getattr(self, "face_desired_degree", 4)

    @property
    def polygon_degree(self):
        return len(self.loop)

    @property
    def has_hole(self):
        return len(set(self.loop)) != len(self.loop)

    @property
    def cost_to_go(self):
        return len(self.moves)

    def build(self):
        """A fresh `(Tiler, desired)` pair for the raw polygon.

        Curved instances hand their arcs to the graph, so a vertex inserted on a
        bent edge lands on the curve and smoothing keeps it there. Everything
        else is identical: the bend was chosen to leave every corner want alone,
        so the topology, the solution and par are exactly what they were.
        """
        coordinates = {i: list(coord) for i, coord in self.coordinates.items()}
        graph = Tiler.from_face_loops([list(self.loop)], coordinates)
        for (first, second), (arc, start) in getattr(self, "arcs", {}).items():
            graph.boundary_arcs.add(first, second, arc, start_vertex=start)
        return graph, dict(self.desired)


def _canonical_relabel(loop):
    """Distinct vertices numbered by first appearance in the loop.

    A slit visits its two endpoints twice, so the loop is longer than the
    vertex set and the two must be numbered separately.
    """
    relabel = {}
    for vertex in loop:
        if vertex not in relabel:
            relabel[vertex] = len(relabel)
    return relabel


def _canonical_moves(loop, moves):
    """Relabel a walk's moves onto 0..m-1 plus creation-ordered new vertices."""
    relabel = _canonical_relabel(loop)
    next_id = len(relabel)
    out = []
    for kind, u, v, created in moves:
        if u not in relabel or v not in relabel:
            return None
        new_id = None
        if kind == "vertex":
            new_id = next_id
            relabel[created] = new_id
            next_id += 1
        out.append((kind, relabel[u], relabel[v], new_id))
    return out


def resolve_move(env, move, alias):
    """Turn a stored move into the env's linear action index for this state."""
    kind, u, v, _ = move
    if u not in alias or v not in alias:
        return None
    graph = env.graph
    if kind == "chord":
        found = _chord_between(graph, alias[u], alias[v],
                               env.max_edge_addition_steps, env._chord_offset,
                               env.min_face_degree)
        if found is None:
            return None
        half_edge, local = found
    else:
        half_edge = _half_edge_between(graph, alias[u], alias[v])
        if half_edge is None:
            return None
        local = env._insert_vertex_action
    slot = env.half_edge_to_index.get(half_edge)
    if slot is None:
        return None
    return slot * env.num_actions_per_half_edge + local


def _play(env, move, alias, linear):
    """Step the env and, for a vertex insert, learn the new vertex's alias."""
    kind, u, v, created = move
    if kind == "vertex":
        half_edge = _half_edge_between(env.graph, alias[u], alias[v])
        env.step(linear)
        alias[created] = env.graph.target_vertex(half_edge, tag=False)
    else:
        env.step(linear)


def replay(env, instance, num_moves=None, rng=None, on_step=None):
    """Play a certified solution through the env.

    A solution is a *set* of insertions with a lot of ordering freedom, so at
    every state several of the remaining moves are playable and each leads to
    the same par mesh. `playable` records all of them, which is what the clone
    should be told: pushing it towards one arbitrary choice sets the replays
    fighting each other. With `rng` the order taken is drawn from that set,
    which also multiplies the states each instance covers.

    `on_step` is called with the env after the reset and after every move, for
    callers that want to watch the solution being played rather than collect it.

    Returns `(observations, actions, potentials, playable, played, ok)`; the
    observations are the env's own, so a trajectory recorded here is directly
    trainable, and the potentials give the exact return of the certified
    continuation from each state.
    """
    graph, desired = instance.build()
    obs, _ = env._reset_to_state(graph, desired)
    if on_step is not None:
        on_step(env)
    alias = {i: i for i in set(instance.loop)}
    limit = len(instance.moves) if num_moves is None else num_moves
    remaining = list(instance.moves)
    observations, actions, potentials, playable_sets = [], [], [], []

    for _ in range(limit):
        playable = []
        for candidate in remaining:
            index = resolve_move(env, candidate, alias)
            if index is not None and np.isfinite(obs["mask"][index]):
                playable.append((candidate, index))
        if not playable:
            return (observations, actions, potentials, playable_sets,
                    len(actions), False)
        pick = int(rng.integers(len(playable))) if rng is not None else 0
        move, linear = playable[pick]

        observations.append({k: np.array(value, copy=True) for k, value in obs.items()})
        actions.append(linear)
        potentials.append(env.potential)
        playable_sets.append([index for _, index in playable])
        _play(env, move, alias, linear)
        if on_step is not None:
            on_step(env)
        remaining.remove(move)
        obs = env._get_obs()
    return observations, actions, potentials, playable_sets, len(actions), True


def pinwheel_annulus_mesh(k=4, rng=None, hole_radius=2.0, rim_radius=5.657,
                          flat_fraction=0.5, untangle=True, handedness=None):
    """An annulus whose hole is turned half a step against its rim.

    `random_annulus` rings grid cells around a grid cell, so hole and rim are
    always axis-aligned and every hole corner faces a rim corner. A single ring
    of quads can then run corner to corner -- a radial mesh, balanced about the
    outward normal at every hole corner.

    Offset the two corner sets by half a step and no such pairing exists. The
    ring has to alternate: a quad straddling a rim corner, then a quad
    straddling a hole edge, all the way round, which puts every edge leaving a
    hole corner on the SAME side of the outward radial. That is the pinwheel,
    and the two ways of resolving it are mirror images, so reaching par means
    breaking a symmetry the domain still has.

        A_i = [C_i, F(i+1,-), H_(i+1), F(i,+)]     straddles rim corner i
        B_i = [H_(i+1), H_i, F(i,-), F(i,+)]       straddles hole edge i -> i+1

    Degrees fall out right by construction: k rim corners at 2, 2k rim flats at
    3, k hole corners at 4, so the seed is at par with every vertex content.

    Two things the construction has to get right, both learned the hard way.

    The flats are points ON the rim's straight edges, found by interpolating
    between consecutive rim corners rather than by placing them at some radius
    and hoping -- otherwise the rim is a 3k-gon that merely resembles a k-gon,
    the corner wants are wrong, and for k != 4 the domain is not the one
    intended at all.

    And only the k + k real corners are user-defined. The flats exist to carry
    the mesh, not to shape the domain: leaving them pinned would stop the
    backward walk deleting them, so the raw polygon would arrive with the
    supporting vertices already in place and the agent would never have to
    learn that inserting them is the move. Unpinned, the walk strips them and
    the domain that comes out is the bare plate-with-a-hole.

    `handedness` is +1 or -1, drawn evenly when it is None. The construction
    itself only ever produces one of the two, and a pinwheel's whole point is
    that both are valid and the agent has to pick: training on one handedness
    teaches a rule with a side baked into it. The mirror in `_rigid_transform`
    does happen to supply the other, measured at 23 against 21, but only for
    hole instances and only while that mirror exists, so the family carries its
    own balance rather than depending on a transform three steps downstream.

    Measured with `utilities/measure_chirality.py`, the generator's own hole
    solutions cap at 0.333 over hundreds of samples while these score 1.000 --
    the structure is not rare in training, it is absent, and the two held-out
    domains the released agent cannot solve need it (0.583 and 0.833).
    """
    rng = rng or np.random.default_rng()
    if handedness is None:
        handedness = 1 if rng.random() < 0.5 else -1
    step = 2 * math.pi / k
    coordinates, hole, corner = {}, {}, {}
    flat_minus, flat_plus = {}, {}

    def place(x, y):
        index = len(coordinates)
        coordinates[index] = [float(x), float(y)]
        return index

    for i in range(k):
        hole[i] = place(hole_radius * math.cos(i * step),
                        hole_radius * math.sin(i * step))
    rim = []
    for i in range(k):
        angle = (i + 0.5) * step
        rim.append((rim_radius * math.cos(angle), rim_radius * math.sin(angle)))
        corner[i] = place(*rim[-1])
    # the two flats either side of hole direction i lie on the rim EDGE joining
    # rim corners i-1 and i, so the outline is a genuine k-gon
    for i in range(k):
        a, b = rim[(i - 1) % k], rim[i]
        low, high = 0.5 - flat_fraction / 2, 0.5 + flat_fraction / 2
        flat_minus[i] = place(a[0] + low * (b[0] - a[0]), a[1] + low * (b[1] - a[1]))
        flat_plus[i] = place(a[0] + high * (b[0] - a[0]), a[1] + high * (b[1] - a[1]))

    loops = []
    for i in range(k):
        j = (i + 1) % k
        loops.append([corner[i], flat_minus[j], hole[j], flat_plus[i]])
        loops.append([hole[j], hole[i], flat_minus[i], flat_plus[i]])

    def signed_area(loop):
        points = [coordinates[v] for v in loop]
        n = len(points)
        return 0.5 * sum(points[a][0] * points[(a + 1) % n][1]
                         - points[(a + 1) % n][0] * points[a][1] for a in range(n))

    if handedness < 0:
        # the exact mirror: reflect in x and reverse every loop, which undoes
        # the orientation the reflection just flipped
        coordinates = {v: [-x, y] for v, (x, y) in coordinates.items()}
        loops = [loop[::-1] for loop in loops]

    # `from_face_loops` wants every loop counter-clockwise
    loops = [loop if signed_area(loop) > 0 else loop[::-1] for loop in loops]
    real_corners = set(hole.values()) | set(corner.values())
    graph = Tiler.from_face_loops(loops, coordinates, user_vertices=real_corners)

    if untangle:
        # every vertex here is on a boundary, so the only freedom is sliding the
        # flats along their own rim edge -- which is exactly the freedom the
        # hand-placed fractions were guessing at
        try:
            from src.geo2d_bridge import resmooth_env

            class _Holder:
                pass
            holder = _Holder()
            holder.graph = graph
            holder._update_half_edge_angles = lambda: None
            holder.min_element_quality = lambda: 1.0
            resmooth_env(holder, iters=12, method="optimize", slide="optimize")
        except Exception:
            pass                       # geogen is optional; the seed is usable as drawn

    desired = {v: graph.vertex_degree(v) for v in graph.vertex_list(tag=False)}
    return graph, desired, None


def _rigid_transform(coordinates, rng, scale_range=(0.7, 1.4)):
    """Rotate, mirror and uniformly scale. Returns `(coordinates, mirrored)`.

    Unsigned interior angles survive all three, which is what the corner-want
    bookkeeping needs. ORIENTATION does not: a reflection has determinant -1, so
    a face loop that was counter-clockwise comes back clockwise, and
    `Tiler.from_face_loops` assumes counter-clockwise. Every corner Jacobian
    then reads negative and the mesh scores as fully inverted while being
    perfectly drawn.

    That was silently costing most of the hole instances -- the transform is
    applied to holes and polar domains only, the mirror fires half the time, and
    the generator rejected 161 of 300 candidates with it against 23 without. So
    the flag comes back with the coordinates and the caller reverses the loop,
    which is the same reflection expressed combinatorially and restores the
    orientation the rest of the pipeline requires.
    """
    theta = rng.uniform(0, 2 * np.pi)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    mirrored = rng.random() < 0.5
    if mirrored:
        rotation = rotation @ np.array([[1.0, 0.0], [0.0, -1.0]])
    scale = rng.uniform(*scale_range)
    keys = list(coordinates)
    points = np.array([coordinates[k] for k in keys], dtype=float)
    points = (points - points.mean(axis=0)) @ rotation.T * scale
    return {k: points[i] for i, k in enumerate(keys)}, mirrored


def _build_from_angles(loop, mesh_degrees, seed_angles, seed_lengths, rng, max_bumps=4,
                       target_angle=90.0, parity_probability=0.0,
                       max_desired_degree=6):
    """Re-draw the polygon from prescribed corner angles.

    The mesh is a topological object, so re-siting the polygon changes only
    what each corner wants. Bumping `k` corners by one desired degree, all in
    the same direction, moves both the vertex score and its Gauss-Bonnet bound
    by `k`, so the recorded solution still lands exactly on par -- now on a
    domain whose par is `k` rather than zero.
    """
    degrees = [mesh_degrees[v] for v in loop]
    n = len(loop)
    direction = 1 if rng.random() < 0.5 else -1

    if rng.random() < parity_probability:
        # Bump every corner of one parity, so the outline ends up wanting a
        # single parity of degree throughout. A lattice outline mixes all of
        # {2,3,4,5,6}, but a rectilinear domain asked for triangles wants only
        # {3,5}, and a domain like the triforce ring only {2,4,6} -- degree sets
        # the seed families otherwise almost never produce. Every bump still
        # points the same way, so the instance stays at par.
        parity = int(rng.integers(0, 2))
        chosen = {i for i in range(n)
                  if degrees[i] % 2 == parity and degrees[i] + direction >= 2}
    else:
        limit = min(max_bumps, n)
        count = int(rng.integers(0, limit + 1))
        eligible = [i for i in range(n) if degrees[i] + direction >= 2]
        count = min(count, len(eligible))
        chosen = set(rng.permutation(eligible)[:count].tolist()) if count else set()

    bumps = [direction if i in chosen else 0 for i in range(n)]
    if max(d + b for d, b in zip(degrees, bumps)) > max_desired_degree:
        # a 300-degree corner bumped up wants seven edges, which drags the
        # outline's degree set out of the range real domains use and, for the
        # all-odd case, turns {3,5} into {3,5,7}
        return None
    reference_angles = [seed_angles[v] for v in loop]
    reference_lengths = [seed_lengths[v] for v in loop]
    angles = sample_angles(degrees, bumps, rng, reference=reference_angles,
                           target_angle=target_angle)
    if angles is None:
        return None
    points = polygon_from_angles(angles, rng, reference_lengths=reference_lengths)
    if points is None:
        return None
    coordinates = {v: points[i] for i, v in enumerate(loop)}
    desired = {v: degrees[i] + bumps[i] for i, v in enumerate(loop)}
    return coordinates, desired


def generate_instances(num_instances, cell_range=(2, 10), env=None, rng=None,
                       max_edge_addition_steps=3, chord_offset=1, strength=0.35,
                       deform_probability=0.4, angle_probability=0.5, max_bumps=4,
                       flat_corner_probability=0.3, hole_probability=0.15,
                       polar_probability=0.25, walk_attempts=6, parity_probability=0.0,
                       rectilinear_probability=0.35, pinwheel_probability=0.0,
                       curve_probability=0.0, curve_edge_probability=0.5,
                       gate="laplacian", degeneracy_threshold=0.1,
                       max_attempts_factor=12, face_desired_degree=4, verbose=False):
    """Build and validate solved instances. Only replay-verified ones are kept."""
    rng = rng or np.random.default_rng()
    target_angle = utils.average_face_angle(face_desired_degree)
    if env is None:
        env = default_scratch_env(max_edge_addition_steps,
                                  face_desired_degree=face_desired_degree)

    instances = []
    attempts = 0
    stats = {"walk_failed": 0, "relabel_failed": 0, "deform_failed": 0,
             "angle_failed": 0, "starts_solved": 0, "seed_failed": 0,
             "replay_failed": 0, "not_par": 0}
    while len(instances) < num_instances and attempts < num_instances * max_attempts_factor:
        attempts += 1
        num_cells = int(rng.integers(cell_range[0], cell_range[1] + 1))
        want_hole = rng.random() < hole_probability and num_cells >= 8
        want_polar = not want_hole and rng.random() < polar_probability
        triangles = face_desired_degree == 3
        # a hole turned against its rim: the one structure the grid annulus
        # cannot express, and the one the failing held-out domains need
        want_pinwheel = (want_hole and not triangles
                         and rng.random() < pinwheel_probability)
        if want_pinwheel:
            seed_graph, seed_desired, _ = pinwheel_annulus_mesh(
                k=int(rng.integers(4, 7)), rng=rng)
        elif want_polar:
            seed = (triangle_fan_mesh(int(rng.integers(4, 12)), rng, target_angle)
                    if triangles else
                    polar_mesh(int(rng.integers(3, 8)), rng,
                               flat_corner_probability=flat_corner_probability))
            if seed is None:
                stats["seed_failed"] += 1
                continue
            seed_graph, seed_desired, _ = seed
        elif triangles and rng.random() < rectilinear_probability:
            # a rectilinear domain asked for triangles: every corner is a
            # rounding tie, which the angle sampler cannot produce at all
            try:
                cells = (random_annulus(num_cells, rng) if want_hole
                         else random_polyomino(num_cells, rng))
            except RuntimeError:
                continue
            seed_graph, seed_desired, _ = polyomino_triangle_mesh(
                cells, target_angle=target_angle, rng=rng,
                flat_corner_probability=flat_corner_probability)
        elif triangles:
            try:
                cells = random_polyiamond(num_cells, rng, hole=want_hole)
            except RuntimeError:
                continue
            seed_graph, seed_desired, _ = polyiamond_mesh(
                cells, target_angle=target_angle, rng=rng,
                flat_corner_probability=flat_corner_probability)
        else:
            try:
                cells = (random_annulus(num_cells, rng) if want_hole
                         else random_polyomino(num_cells, rng))
            except RuntimeError:
                continue
            seed_graph, seed_desired, _ = polyomino_mesh(
                cells, rng=rng, flat_corner_probability=flat_corner_probability)
        walk = None
        for _ in range(walk_attempts):
            walk = backward_walk(seed_graph, seed_desired, max_edge_addition_steps,
                                 chord_offset, rng,
                                 face_desired_degree=face_desired_degree)
            if walk is not None:
                break
        if walk is None:
            stats["walk_failed"] += 1
            continue
        polygon_graph, polygon_desired, moves = walk
        loop = polygon_from_graph(polygon_graph)
        if len(loop) < 3:
            continue
        if want_hole and len(set(loop)) == len(loop):
            continue  # the walk did not leave a slit; not a hole instance
        coordinates = {v: polygon_graph.vertex_coordinate(v) for v in set(loop)}
        desired = {v: polygon_desired[v] for v in set(loop)}
        draw = 2.0 if (want_hole or want_polar) else rng.random()
        if want_hole or want_polar:
            # only angle-preserving maps on a slit outline: its repeated
            # vertices make the corner-angle bookkeeping above ill-defined
            coordinates, mirrored = _rigid_transform(coordinates, rng)
            if mirrored:
                # the coordinates are reflected, so the loop has to be reversed
                # or every face comes out clockwise
                loop = loop[::-1]

        canonical = _canonical_moves(loop, moves)
        if canonical is None:
            stats["relabel_failed"] += 1
            continue
        if draw < angle_probability:
            seed_angles = utils.get_polygon_interior_angles(loop, coordinates)
            seed_lengths = {}
            for index, vertex in enumerate(loop):
                nxt = loop[(index + 1) % len(loop)]
                seed_lengths[vertex] = float(np.linalg.norm(
                    np.asarray(coordinates[nxt]) - np.asarray(coordinates[vertex])))
            built = _build_from_angles(loop, polygon_desired, seed_angles,
                                       seed_lengths, rng, max_bumps,
                                       target_angle=target_angle,
                                       parity_probability=parity_probability)
            if built is None:
                stats["angle_failed"] += 1
                continue
            coordinates, desired = built
        elif draw < angle_probability + deform_probability:
            deformed, new_desired = deform_polygon(
                coordinates, loop, polygon_desired, rng,
                strength=float(rng.uniform(0.15, strength)),
                target_angle=target_angle)
            if deformed is None:
                stats["deform_failed"] += 1
                continue
            coordinates, desired = deformed, new_desired

        relabel = _canonical_relabel(loop)
        # Bend some boundary edges into arcs. Chosen so every corner stays
        # inside its own 90-degree bin, which leaves every WANT untouched and so
        # leaves par untouched -- the recorded solution is still exactly optimal
        # on a domain that is now curved. Verified on 28 of 28 instances.
        # Straight instances are the default; a slit outline is skipped because
        # a bend across a zero-width cut has no meaning.
        arcs = {}
        if curve_probability and not want_hole and rng.random() < curve_probability:
            from src.bend_outline import bend_outline
            outline_angles = utils.get_polygon_interior_angles(loop, coordinates)
            bent = bend_outline(loop, coordinates, outline_angles, rng,
                                target_angle=target_angle,
                                probability=curve_edge_probability)
            arcs = {(relabel[a], relabel[b]): (arc, relabel[start])
                    for (a, b), (arc, start) in bent.items()}

        instance = SolvedInstance(
            loop=[relabel[v] for v in loop],
            coordinates={relabel[v]: np.asarray(coordinates[v], dtype=float)
                         for v in relabel},
            desired={relabel[v]: desired[v] for v in relabel},
            moves=canonical,
            par=0,
            face_desired_degree=face_desired_degree,
            arcs=arcs,
        )
        graph_check, desired_check = instance.build()
        env._reset_to_state(graph_check, desired_check)
        instance.par = env.par
        if env.is_at_par():
            # a quadrilateral outline is already a solved mesh; its "solution"
            # would only teach the agent to spoil a finished domain
            stats["starts_solved"] += 1
            continue
        _, _, _, _, played, ok = replay(env, instance)
        if not ok or played != len(canonical):
            stats["replay_failed"] += 1
            continue
        if not accepts_as_solved(env, gate=gate,
                                 degeneracy_threshold=degeneracy_threshold):
            stats["not_par"] += 1
            continue
        instances.append(instance)
        if verbose and len(instances) % 250 == 0:
            print(f"  {len(instances)}/{num_instances} instances "
                  f"({attempts} attempts)", flush=True)
    return instances, stats


def default_scratch_env(max_edge_addition_steps=3, template_size=256,
                        face_desired_degree=4, tie_aware_scoring=True):
    from envs.angle_env_with_length import AngleEnvWithLength
    from envs.environment_initializers import LEnv
    return AngleEnvWithLength(
        face_desired_degree, LEnv(utils.average_face_angle(face_desired_degree)),
        template_size=template_size, max_steps_factor=8,
        max_edge_addition_steps=max_edge_addition_steps,
        resample_if_at_par=False, terminate_on_overflow=False,
        terminate_at_par=False, tie_aware_scoring=tie_aware_scoring)


class GeneratedInstanceInitializer:
    """Certified start states drawn from the generator, not from a fixed pickle.

    `SolvedInstanceInitializer` loads a pickle once and samples it forever. The
    geo2d side of the training mixture draws a NEW domain every episode, so the
    two halves of the distribution are not comparable: over a 4M-step run the
    random side sees about 124,000 domains it has never seen, while the
    certified side recycles 9,000 about seven times each. Every par > 0 domain
    the agent ever meets lives in that finite list, because geo2d builds
    rectilinear parts and those are all par 0.

    This draws batches from `generate_instances` instead and replaces the batch
    once it has been used up, so the certified half is as fresh as the random
    half. The generator makes roughly twenty verified instances a second and a
    batch lasts a few hundred episodes, which is a couple of percent of the
    step budget.

    The batch is built on FIRST USE, not in `__init__`, because each vectorised
    worker is its own process: constructing it eagerly would fork one pool into
    every worker and put us back where we started. Each worker seeds from its
    own pid unless told otherwise, so the streams differ.
    """

    REFRESH_MULTIPLE = 4          # draws per instance before a batch is replaced

    def __init__(self, batch_size=256, refresh_after=None, seed=None,
                 cost_to_go=None, degree_range=None, max_steps_factor=None,
                 min_max_steps=10, **generator_kwargs):
        self.batch_size = int(batch_size)
        # Default to reusing each batch REFRESH_MULTIPLE times. Discarding a
        # batch after a single pass costs a full generation per batch_size
        # draws, which is 25 s per 256 instances -- 98 ms amortised onto every
        # anchor episode against 74 ms of actual stepping, and measured at 256
        # fps against the baseline's 328. Four draws per instance is still far
        # fresher than the fixed pickle it replaces, which served 9,000
        # instances about seven times each for a whole run.
        self.refresh_after = int(refresh_after or batch_size * self.REFRESH_MULTIPLE)
        self.seed = seed
        self.cost_to_go = cost_to_go
        self.degree_range = list(degree_range) if degree_range else None
        self.max_steps_factor = max_steps_factor
        self.min_max_steps = min_max_steps
        self.generator_kwargs = dict(generator_kwargs)
        self._inner = None
        self._drawn = 0
        self._rng = None
        self._batches = 0

    @classmethod
    def from_config(cls, config):
        known = ("name", "batch_size", "refresh_after", "seed", "cost_to_go",
                 "min_polygon_degree", "max_polygon_degree", "max_steps_factor",
                 "min_max_steps")
        degree_range = None
        if "min_polygon_degree" in config:
            degree_range = list(range(config["min_polygon_degree"],
                                      config["max_polygon_degree"] + 1))
        generator_kwargs = {k: v for k, v in config.items() if k not in known}
        return cls(batch_size=config.get("batch_size", 256),
                   refresh_after=config.get("refresh_after"),
                   seed=config.get("seed"),
                   cost_to_go=config.get("cost_to_go"),
                   degree_range=degree_range,
                   max_steps_factor=config.get("max_steps_factor"),
                   min_max_steps=config.get("min_max_steps", 10),
                   **generator_kwargs)

    def _refill(self):
        if self._rng is None:
            base = os.getpid() if self.seed is None else self.seed
            self._rng = np.random.default_rng(base)
        instances, _ = generate_instances(self.batch_size, rng=self._rng,
                                          verbose=False, **self.generator_kwargs)
        if not instances:
            raise RuntimeError("the generator produced no instances; check its "
                               "settings against `generate_instances`")
        self._inner = SolvedInstanceInitializer(
            instances, cost_to_go=self.cost_to_go, degree_range=self.degree_range,
            max_steps_factor=self.max_steps_factor,
            min_max_steps=self.min_max_steps)
        self._drawn = 0
        self._batches += 1

    def set_degree_range(self, degree_range):
        self.degree_range = list(degree_range)
        if self._inner is not None:
            self._inner.set_degree_range(degree_range)

    def set_cost_to_go(self, cost_to_go):
        self.cost_to_go = cost_to_go
        if self._inner is not None:
            self._inner.set_cost_to_go(cost_to_go)

    def __call__(self):
        if self._inner is None or self._drawn >= self.refresh_after:
            self._refill()
        self._drawn += 1
        return self._inner()


class SolvedInstanceInitializer:
    """Sample start states at a chosen distance from a known solution.

    At cost-to-go `n` the episode begins `n` certified moves from par, so the
    agent learns to finish a global split long before it has to learn to start
    one. `cost_to_go = None` always starts from the raw polygon.
    """

    def __init__(self, instances, cost_to_go=None, env=None, degree_range=None,
                 max_steps_factor=None, min_max_steps=10):
        self.instances = list(instances)
        assert self.instances, "no solved instances supplied"
        self.cost_to_go = cost_to_go
        self.degree_range = list(degree_range) if degree_range else None
        self.max_steps_factor = max_steps_factor
        self.min_max_steps = min_max_steps
        if max_steps_factor is not None:
            # an instance whose solution is longer than the episode budget can
            # never be won inside one, so it is not a start state worth serving
            self.instances = [
                i for i in self.instances
                if i.cost_to_go <= max(min_max_steps,
                                       int(np.ceil(max_steps_factor * i.polygon_degree)))]
            assert self.instances, "no instance fits the episode budget"
        self._env = env
        self._pool = self._filtered()

    @classmethod
    def from_config(cls, config):
        path = config["dataset"]
        with open(path, "rb") as handle:
            instances = pickle.load(handle)
        degree_range = None
        if "min_polygon_degree" in config:
            degree_range = list(range(config["min_polygon_degree"],
                                      config["max_polygon_degree"] + 1))
        return cls(instances, cost_to_go=config.get("cost_to_go"),
                   degree_range=degree_range,
                   max_steps_factor=config.get("max_steps_factor"),
                   min_max_steps=config.get("min_max_steps", 10))

    def _filtered(self):
        if self.degree_range is None:
            return self.instances
        lo, hi = min(self.degree_range), max(self.degree_range)
        pool = [i for i in self.instances if lo <= len(i.loop) <= hi]
        return pool or self.instances

    def set_degree_range(self, degree_range):
        self.degree_range = list(degree_range)
        self._pool = self._filtered()

    def set_cost_to_go(self, cost_to_go):
        self.cost_to_go = cost_to_go

    def _scratch_env(self):
        if self._env is None:
            self._env = default_scratch_env()
        return self._env

    def __call__(self):
        instance = self._pool[np.random.randint(len(self._pool))]
        if self.cost_to_go is None or self.cost_to_go >= len(instance.moves):
            return instance.build()
        num_moves = len(instance.moves) - self.cost_to_go
        env = self._scratch_env()
        *_, ok = replay(env, instance, num_moves=num_moves)
        if not ok:
            return instance.build()
        return deepcopy(env.graph), dict(env.vertex_desired_degree)
