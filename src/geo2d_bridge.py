"""Bridge between the `geogen/geo2d` random-geometry generator and `AngleEnv`.

`geo2d.Geometry` is one counter-clockwise outer loop plus clockwise hole loops,
with edges that are straight, circular arcs or splines. The environment's
initializers return `(Tiler, vertex_desired_degree)`: a single-face half-edge
mesh whose face loop *is* the outline, and a desired degree per corner read
off the corner's interior angle at the target element angle.

This module does that translation for the first slice of the distribution --
single polygons with straight edges only (`preset="straight"`, `n_holes=0`) --
and offers the reverse translation, from the agent's final `Tiler` back into
a `geo2d.Mesh`, so the results can be scored with geo2d's own quality tools.

`geogen` is its own repository and is consumed here as an installed package
(`pip install -e /path/to/geogen`); see `import_geo2d` for the fallback.

    import geo2d
    from src.geo2d_bridge import make_geo2d_env, load_model, tiler_to_geo2d_mesh

    gs = geo2d.generate_many(0, 10, preset="straight", n_holes=0)
    env = make_geo2d_env(config["environment"], gs[0])
    model = load_model(checkpoint, config_fn, template_size=env.template_size)

Two conventions worth knowing. Coordinates are normalized the way the training
generator normalizes its outlines (centred, largest half-extent 1), because
`AngleEnvWithLength` feeds raw edge lengths to the network. Collinear vertices
(interior angle 180, which geo2d produces where two chamfers meet) are dropped
by default: they are not corners of the domain, and keeping one forces a
degree-3 mesh vertex at an arbitrary point of a straight edge.
"""

import os
import sys
from copy import deepcopy

import math
import numpy as np

import envs.polygon_utils as utils
from src.tiler import Tiler

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GEOGEN_PATH = os.path.normpath(os.path.join(_REPO_ROOT, "..", "geogen"))


def import_geo2d(path=None):
    """Import `geo2d`, preferring the installed `geogen` package.

    `geogen` is a separate repository, installed into the environment with
    `pip install -e /path/to/geogen`. If it is not installed, a checkout at
    `path`, `$GEOGEN_PATH` or the sibling directory `../geogen` is tried, so
    a fresh clone next to this one still works without any setup.
    """
    try:
        import geo2d
        return geo2d
    except ImportError:
        pass
    path = path or os.environ.get("GEOGEN_PATH", DEFAULT_GEOGEN_PATH)
    if os.path.isdir(os.path.join(path, "geo2d")):
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            import geo2d
            return geo2d
        except ImportError:
            pass
    raise ImportError(
        "geo2d is not available. Install the geogen package into this environment, e.g.\n"
        "    pip install -e /path/to/geogen\n"
        "or point GEOGEN_PATH at a geogen checkout.")


# --------------------------------------------------------------------------- geometry -> Tiler

def polygon_points(geometry, drop_collinear=True, normalize=True, tol_deg=1e-6):
    """The outline of a straight-edged, hole-free geometry as a CCW (n, 2) array.

    Raises `NotImplementedError` for holes or curved edges: those need a
    different start state (a slit for holes; arc corners for curves) and are
    deliberately not handled here yet.
    """
    if len(geometry.loops) != 1:
        raise NotImplementedError(
            f"geometry has {len(geometry.loops) - 1} hole(s); only single polygons are supported")
    loop = geometry.outer
    kinds = set(loop.kinds())
    if kinds != {"line"}:
        raise NotImplementedError(f"geometry has curved edges ({sorted(kinds - {'line'})})")

    points = np.asarray(loop.points, dtype=float)
    if not loop.is_ccw():
        points = points[::-1]

    if drop_collinear:
        angles = _interior_angles(points)
        keep = np.abs(angles - 180.0) > tol_deg
        points = points[keep]
        if len(points) < 3:
            raise ValueError("polygon degenerates after dropping collinear vertices")

    if normalize:
        points = normalize_points(points)
    return points


def normalize_points(points):
    """Centre on the centroid of the vertices and scale the largest half-extent to 1.

    This is the training generator's own normalization (see
    `envs/solved_instances.py`), so edge lengths land in the range the
    length-feature network was trained on.
    """
    points = np.asarray(points, dtype=float)
    centred = points - points.mean(axis=0)
    scale = np.abs(centred).max()
    return centred / max(scale, 1e-9)


def _interior_angles(points):
    """Interior angle at every vertex of a CCW polygon, in degrees."""
    n = len(points)
    angles = np.empty(n)
    for i in range(n):
        first = points[(i + 1) % n] - points[i]
        second = points[i - 1] - points[i]
        angles[i] = utils.angle_between(first, second)
    return angles


# --------------------------------------------------------------------- curved boundaries

def curved_loop_to_tiler(geometry, target_angle=90.0, normalize=True):
    """A geo2d geometry with ARCS, as a Tiler -- without discretising anything.

    A curve is not a different problem. Topology never asks what shape an edge
    is, and the corner want at a vertex is set by the TANGENT angle there, which
    `Loop.interior_angles()` already reports: a smooth arc joint reads 180 and
    wants degree three, exactly like a flat point on a straight edge. So the
    outline is taken as geo2d gives it, one vertex per control point, and the
    arcs are recorded on the graph so that inserting a vertex lands on the curve
    and smoothing keeps it there.

    Discretising instead would invent corners that are an artefact of the
    sampling, and would make par depend on how finely the curve was sampled --
    which is the opposite of a bound.

    Hole-free for now; a curved hole needs the slit construction to cut through
    an arc, which is the same idea and more bookkeeping.
    """
    from src.boundary_arcs import Arc

    if len(geometry.loops) != 1:
        raise NotImplementedError(
            f"geometry has {len(geometry.loops) - 1} hole(s); curved holes are not wired yet")
    loop = geometry.outer
    points = np.asarray(loop.points, dtype=float)
    angles = np.asarray(loop.interior_angles(), dtype=float)
    is_arc = np.asarray(loop.is_arc(), dtype=bool)
    count = len(points)

    # geo2d edge i runs from control point i to i+1. Reversing for CCW reverses
    # the vertex order, and edge i then joins the vertices that were i+1 and i.
    order = list(range(count))
    reversed_loop = not loop.is_ccw()
    if reversed_loop:
        order = order[::-1]
        points = points[::-1]
        angles = 360.0 - angles[::-1]     # see `_loop_arrays`: the side flips too

    coordinates = {i: points[i].astype(float) for i in range(count)}
    graph = Tiler.from_face_loops([list(range(count))], coordinates)

    for position, original in enumerate(order):
        if not is_arc[original]:
            continue
        edge = loop.edge(original)
        # geo2d edge i runs from control point i to i+1. Reversing puts original
        # point k at index n-1-k, so edge `original` joins THIS position and the
        # one BEFORE it, not the one after. geo2d hands back a counter-clockwise
        # outer loop every time (0 of 400 were clockwise), so this branch has
        # never run here -- but every hole it generates is clockwise, so curved
        # holes would meet it immediately.
        if reversed_loop:
            first, second = (position - 1) % count, position
        else:
            first, second = position, (position + 1) % count
        arc = Arc.from_geo2d(edge)
        # which endpoint the arc runs FROM, so a later split hands its halves to
        # the right pairs
        start = first if np.allclose(points[first], np.asarray(edge.p0, dtype=float),
                                     atol=1e-9) else second
        graph.boundary_arcs.add(first, second, arc, start_vertex=start)

    if normalize:
        _rescale_with_arcs(graph, points)

    desired = {i: utils.rounded_desired_degree(float(angles[i]), target_angle)
               for i in range(count)}
    return graph, desired


def curved_geometry_to_tiler(geometry, target_angle=90.0, normalize=True,
                             drop_collinear=True):
    """`(Tiler, desired_degree)` for a curved geo2d geometry, HOLES INCLUDED.

    Nothing about a hole changes because its boundary curves. The domain is still
    cut open by a slit into one face loop, par is still read off the corner
    wants, and the arcs are still carried as arcs -- keyed by vertex pair, so
    they survive the slit's renumbering and every later edit.

    A circular hole is the easy case, not the hard one. Every control point of a
    circle has tangent interior angle 180, so on the material side it also reads
    180 and wants degree three -- exactly what a slit endpoint is forced to. A
    SQUARE hole's corners want four, and a convex outer corner wants two and
    cannot carry a slit at all (see `_absorbs_a_slit`); a circle can be cut
    anywhere.

    What it costs is par. A circular hole has no corners to absorb its own
    curvature, so four defects have to go somewhere: a square plate with a
    square hole is par 0, and the same plate with a CIRCULAR hole is par 4,
    which the textbook O-grid hits exactly by putting one at each control point.
    """
    from src.boundary_arcs import Arc

    if len(geometry.loops) == 1:
        return curved_loop_to_tiler(geometry, target_angle=target_angle,
                                    normalize=normalize)

    outer, outer_angles, outer_arcs = _loop_arrays(
        geometry.outer, ccw=True, drop_collinear=drop_collinear)
    holes, hole_angles, hole_arcs = [], [], []
    for loop in geometry.loops[1:]:
        points, angles, arcs = _loop_arrays(loop, ccw=True,
                                            drop_collinear=drop_collinear)
        holes.append(points)
        hole_angles.append(angles)
        hole_arcs.append(arcs)

    coordinates, face_loop, angles, arcs = slit_outline(
        outer, holes, target_angle, outer_angles=outer_angles,
        hole_angles=hole_angles, outer_arcs=outer_arcs, hole_arcs=hole_arcs)

    graph = Tiler.from_face_loops([face_loop],
                                  {k: np.asarray(v, dtype=float)
                                   for k, v in coordinates.items()})
    for (first, second), (arc, start) in arcs.items():
        graph.boundary_arcs.add(first, second, arc, start_vertex=start)

    if normalize:
        _rescale_with_arcs(graph, np.asarray(outer, dtype=float))

    desired = {vertex: utils.rounded_desired_degree(float(angle), target_angle)
               for vertex, angle in angles.items()}
    return graph, desired


def _rescale_with_arcs(graph, points):
    """Centre and scale the graph, moving its arcs with it.

    The same normalization straight-edged domains get, except an arc carries a
    centre and a radius that have to travel too or the curve detaches from its
    own endpoints.
    """
    centre = points.mean(axis=0)
    scale = max(float(np.abs(points - centre).max()), 1e-9)
    for vertex, coordinate in list(graph.vertex_coordinates.items()):
        graph.vertex_coordinates[vertex] = (np.asarray(coordinate, dtype=float)
                                            - centre) / scale
    for _, (arc, _) in graph.boundary_arcs.items():
        arc.centre = (arc.centre - centre) / scale
        arc.radius = arc.radius / scale


def make_curved_env(env_config, geometry, template_size=None, target_angle=90.0):
    """An env on a curved geometry, via the `Custom` initializer hook."""
    graph, desired = curved_geometry_to_tiler(geometry, target_angle=target_angle)

    class _Fixed:
        def __init__(self):
            self.n = len(desired)

        def __call__(self):
            return deepcopy(graph), dict(desired)

    config = dict(env_config)
    config.pop("initializer", None)
    config["graph_initializer"] = _Fixed()
    config["resample_if_at_par"] = False
    if template_size is not None:
        config["template_size"] = template_size
    from envs.environment_maker import initialize_environment
    return initialize_environment(config)


# ---------------------------------------------------------------- holes: loops -> slit outline

def _loop_points(loop, drop_collinear=True, tol_deg=1e-6, ccw=True):
    """One straight-edged geo2d loop as an (n, 2) array, oriented CCW or CW."""
    kinds = set(loop.kinds())
    if kinds != {"line"}:
        raise NotImplementedError(f"loop has curved edges ({sorted(kinds - {'line'})})")
    points = np.asarray(loop.points, dtype=float)
    if loop.is_ccw() != bool(ccw):
        points = points[::-1]
    if drop_collinear:
        angles = _interior_angles(points)
        points = points[np.abs(angles - 180.0) > tol_deg]
        if len(points) < 3:
            raise ValueError("loop degenerates after dropping collinear vertices")
    return points


def geometry_loops(geometry, drop_collinear=True, tol_deg=1e-6):
    """`(outer, holes)` point arrays for a straight-edged geometry, all CCW.

    Holes come back counter-clockwise (the orientation in which their own
    interior angles are measured); `slit_outline` reverses them when it walks
    them from the material side.
    """
    outer = _loop_points(geometry.outer, drop_collinear, tol_deg, ccw=True)
    holes = [_loop_points(loop, drop_collinear, tol_deg, ccw=True)
             for loop in geometry.loops[1:]]
    return outer, holes



# ------------------------------------------------------- curved loops, holes included

def _loop_arrays(loop, ccw=True, drop_collinear=True, tol_deg=1e-6):
    """One geo2d loop as `(points, tangent angles, arcs)`, wound as asked.

    `arcs` is keyed by the local index pair the arc joins, carrying the arc and
    which of the two ends it runs FROM, exactly as `BoundaryArcs` wants.

    Two things differ from the straight-edged `_loop_points`. The angles are
    geo2d's own `interior_angles()`, which are TANGENT angles, so an arc joint
    reads 180 and wants three. And a flat vertex is only dropped when BOTH its
    edges are straight: a smooth arc junction also reads 180, and it is where
    the curve changes, so dropping it would throw the geometry away.
    """
    from src.boundary_arcs import Arc

    points = np.asarray(loop.points, dtype=float)
    angles = np.asarray(loop.interior_angles(), dtype=float)
    is_arc = np.asarray(loop.is_arc(), dtype=bool)
    count = len(points)

    order = list(range(count))
    reversed_loop = loop.is_ccw() != bool(ccw)
    if reversed_loop:
        order = order[::-1]
        points = points[::-1]
        # An angle is measured on the INSIDE of the traversal, so reversing the
        # winding swaps which side that is: geo2d reports 270 at the corner of a
        # clockwise square hole, which is the material side, and the same corner
        # read as a counter-clockwise polygon is 90. Reversing the order without
        # this leaves `slit_outline` to apply its own `360 -` to a number that
        # already had it, turning every square hole corner from want 4 into
        # want 2 -- and quietly wrecking par on every domain with a hole.
        angles = 360.0 - angles[::-1]

    arcs = {}
    for position, original in enumerate(order):
        if not is_arc[original]:
            continue
        edge = loop.edge(original)
        # see `curved_loop_to_tiler`: a reversed loop's edge joins this index
        # and the one BEFORE it
        if reversed_loop:
            first, second = (position - 1) % count, position
        else:
            first, second = position, (position + 1) % count
        arc = Arc.from_geo2d(edge)
        start = first if np.allclose(points[first], np.asarray(edge.p0, dtype=float),
                                     atol=1e-9) else second
        arcs[(first, second)] = (arc, start)

    if drop_collinear and count > 3:
        on_a_curve = {index for pair in arcs for index in pair}
        keep = [i for i in range(count)
                if i in on_a_curve or abs(angles[i] - 180.0) > tol_deg]
        if 3 <= len(keep) < count:
            remap = {old: new for new, old in enumerate(keep)}
            points, angles = points[keep], angles[keep]
            arcs = {(remap[a], remap[b]): (arc, remap[start])
                    for (a, b), (arc, start) in arcs.items()}
    return points, angles, arcs


def _sampled_loop(points, arcs, per_arc=24):
    """The loop as a dense polyline, following its arcs.

    Every geometric predicate below -- does the slit cross the boundary, is its
    midpoint inside the material -- is written for polygons. A chord through a
    bulging arc is not the boundary, so they get this instead.
    """
    pieces = []
    count = len(points)
    for i in range(count):
        j = (i + 1) % count
        entry = arcs.get((i, j)) or arcs.get((j, i))
        if entry is None:
            pieces.append(points[i][None, :])
            continue
        arc, start = entry
        sampled = arc.sample(per_arc)
        if start != i:
            sampled = sampled[::-1]
        pieces.append(sampled[:-1])
    return np.vstack(pieces)


def _obstacle_segments(coordinates, loop, arcs, per_arc=16):
    """The loop as segments for the crossing test, arcs sampled rather than chorded."""
    segments = []
    for index in range(len(loop)):
        first, second = loop[index], loop[(index + 1) % len(loop)]
        entry = arcs.get((first, second)) or arcs.get((second, first)) if arcs else None
        if entry is None:
            segments.append((coordinates[first], coordinates[second]))
            continue
        arc, start = entry
        sampled = arc.sample(per_arc)
        if start != first:
            sampled = sampled[::-1]
        segments += [(sampled[k], sampled[k + 1]) for k in range(len(sampled) - 1)]
    return segments


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p, tol=1e-12):
    """`p` lies on the closed segment `a`-`b`."""
    if abs(_orient(a, b, p)) > tol * max(1.0, abs(b[0] - a[0]) + abs(b[1] - a[1])):
        return False
    return (min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
            and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol)


def _segments_cross(a, b, c, d, tol=1e-12):
    """Proper crossing, or a collinear overlap, of segments `ab` and `cd`.

    Segments that merely share an endpoint do not count: a slit is allowed to
    start and end on boundary vertices.
    """
    d1, d2 = _orient(c, d, a), _orient(c, d, b)
    d3, d4 = _orient(a, b, c), _orient(a, b, d)
    if ((d1 > tol) != (d2 > tol)) and ((d3 > tol) != (d4 > tol)) \
            and min(abs(d1), abs(d2), abs(d3), abs(d4)) > tol:
        return True
    # collinear overlap, or an endpoint interior to the other segment
    for p, (u, v) in ((a, (c, d)), (b, (c, d)), (c, (a, b)), (d, (a, b))):
        if np.allclose(p, u) or np.allclose(p, v):
            continue
        if _on_segment(u, v, p):
            return True
    return False


def _point_in_polygon(point, polygon):
    """Ray casting; points on the boundary are undefined but never queried here."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def _loop_edges(points, loop):
    return [(points[loop[i]], points[loop[(i + 1) % len(loop)]]) for i in range(len(loop))]


def slit_outline(outer, holes, target_angle=90.0, outer_angles=None,
                 hole_angles=None, outer_arcs=None, hole_arcs=None):
    """One counter-clockwise face loop for a domain with holes, cut open by slits.

    A hole is joined to the boundary by a *slit*: a zero-width cut from a
    boundary vertex to a hole vertex, walked out and back, so the whole domain
    is a single face whose loop visits the two slit endpoints twice. This is
    the representation `SquareHole` and the certified annulus family already
    use, so `compute_par`, the action mask and the game engine all read it.

    Returns `(coordinates, loop, angles)`: vertex id -> point, the face loop as
    a list of vertex ids (with repeats at the slit endpoints), and vertex id ->
    the domain's interior angle there (a hole corner measures 360 minus its own
    interior angle, since the material is on the outside of the hole).
    """
    coordinates = {i: np.asarray(p, float) for i, p in enumerate(outer)}
    if outer_angles is None:
        outer_angles = _interior_angles(outer)
    angles = dict(zip(range(len(outer)), np.asarray(outer_angles, dtype=float)))
    loop = list(range(len(outer)))
    # arcs are carried in the SAME numbering as `coordinates`, so the caller can
    # hand them straight to `BoundaryArcs` once the Tiler exists
    arcs = dict(outer_arcs or {})
    hole_loops = []
    for index, hole in enumerate(holes):
        start = len(coordinates)
        ids = list(range(start, start + len(hole)))
        for i, point in zip(ids, hole):
            coordinates[i] = np.asarray(point, float)
        local = (_interior_angles(hole) if hole_angles is None
                 else np.asarray(hole_angles[index], dtype=float))
        for i, angle in zip(ids, local):
            # the material is OUTSIDE a hole, so the domain's angle there is the
            # reflex one. A circle reads 180 either way, which is exactly the
            # degree three a slit endpoint is forced to -- so a circular hole can
            # absorb a slit at any of its points.
            angles[i] = 360.0 - angle
        if hole_arcs and hole_arcs[index]:
            for (a, b), (arc, arc_start) in hole_arcs[index].items():
                arcs[(start + a, start + b)] = (arc, start + arc_start)
        hole_loops.append(ids)

    # merge the holes one at a time, right-most first: a slit only has to miss
    # the boundary as it stands plus the holes that have not been cut in yet
    # every predicate below is written for polygons, so give them the real
    # outline rather than a chord through each bulging arc
    outer_polygon = _sampled_loop(np.asarray(outer, dtype=float), outer_arcs or {})
    hole_polygons = [_sampled_loop(np.asarray(hole, dtype=float),
                                   (hole_arcs[i] if hole_arcs else {}) or {})
                     for i, hole in enumerate(holes)]

    order = sorted(range(len(hole_loops)),
                   key=lambda k: -max(coordinates[i][0] for i in hole_loops[k]))
    pending = {k: hole_loops[k] for k in order}
    for k in order:
        hole_ids = pending.pop(k)
        obstacles = _obstacle_segments(coordinates, loop, arcs)
        for other in pending.values():
            obstacles += _obstacle_segments(coordinates, other, arcs)
        obstacles += _obstacle_segments(coordinates, hole_ids, arcs)
        placed = _place_slit(coordinates, loop, hole_ids, outer_polygon, hole_polygons,
                             obstacles, angles=angles, target_angle=target_angle,
                             arcs=arcs)
        if placed is None:
            raise ValueError("no non-crossing slit reaches a hole")
        kind, position, j, extra = placed
        if kind == "split":
            # Anchor on a new flat point of the boundary edge. It wants degree
            # three, which is exactly what a slit endpoint is forced to, and the
            # split is par-neutral: a flat vertex adds nothing to the corner excess
            # and moves V and E together.
            new_id = len(coordinates)
            coordinates[new_id] = np.asarray(extra, dtype=float)
            angles[new_id] = 180.0
            before = loop[position]
            after = loop[(position + 1) % len(loop)]
            entry = arcs.get((before, after)) or arcs.get((after, before))
            if entry is not None:
                # the anchor lands ON the curve, so the arc has to be split with
                # it -- same bookkeeping as `BoundaryArcs.split`
                arc, arc_start = entry
                first_half, second_half = arc.split(arc.fraction_of(extra))
                arcs.pop((before, after), None)
                arcs.pop((after, before), None)
                finish = after if arc_start == before else before
                arcs[(arc_start, new_id)] = (first_half, arc_start)
                arcs[(new_id, finish)] = (second_half, new_id)
            loop = loop[:position + 1] + [new_id] + loop[position + 1:]
            position = position + 1
        rolled = hole_ids[j:] + hole_ids[:j]
        # the hole is walked clockwise, so the material stays on the left
        walk = [rolled[0]] + rolled[:0:-1]
        loop = loop[:position + 1] + walk + [rolled[0]] + loop[position:]
    return coordinates, loop, angles, arcs


def _loop_edges_by_id(coordinates, loop):
    return [(coordinates[loop[i]], coordinates[loop[(i + 1) % len(loop)]])
            for i in range(len(loop))]


def _absorbs_a_slit(angle, target_angle=90.0):
    """Can a corner of this interior angle carry a slit edge without a forced defect?

    A slit endpoint keeps its two boundary edges and gains the slit, so its degree
    can never fall below three. The agent's vocabulary is monotone -- it adds edges
    and never removes one -- so a vertex whose forced minimum sits ABOVE what it
    wants is stuck there for the whole episode and par becomes unreachable. A
    convex 90-degree corner wants degree two and is exactly that case; a flat point
    on an edge wants three and fits perfectly; a reflex corner wants four or more
    and the agent can add its way up.
    """
    want = corner_degree_options(angle, target_angle)
    return max(want) >= 3


def corner_degree_options(angle, target_angle=90.0, tie_tolerance=1e-6):
    """The degrees a corner of this interior angle is content with, ties included."""
    ratio = angle / target_angle
    if abs(ratio - np.floor(ratio) - 0.5) <= tie_tolerance:
        low = max(int(np.floor(ratio)) + 1, 2)
        high = max(int(np.ceil(ratio)) + 1, 2)
        if low != high:
            return (low, high)
    return (utils.rounded_desired_degree(angle, target_angle),)


def _foot_on_segment(point, a, b):
    """Nearest point to `point` on segment a-b, and how far along it sits."""
    direction = b - a
    length2 = float(direction @ direction)
    if length2 < 1e-18:
        return None, 0.0
    t = float((point - a) @ direction) / length2
    return a + t * direction, t


def _place_slit(coordinates, loop, hole_ids, outer, holes, obstacles, angles=None,
                target_angle=90.0, margin=0.08, arcs=None):
    """Where to cut, preferring the shortest cut PERPENDICULAR to a boundary edge.

    Two things decide a good slit. It must not strand a defect: a slit endpoint
    keeps its two boundary edges and gains the slit, so its degree can never fall
    below three, and an anchor wanting less than that is stuck above target for
    the whole episode. And it should be short and square to the boundary, because
    the slit is a line the mesh has to conform to -- a long diagonal one cuts
    across the domain's own structure and throws away whatever symmetry it had.

    So the candidates are, in cost order: the perpendicular foot from a hole
    vertex onto a boundary edge, which is the shortest cut to that edge and is
    axis-aligned wherever the boundary is, and which lands mid-edge on a flat
    point wanting exactly the degree three a slit forces; then existing corners
    that can absorb the edge, by length; then anything else. A foot too close to
    either end of its edge is dropped, since splitting there would leave a sliver.

    Returns `(kind, position, j, extra)` -- kind "vertex" anchors on the loop
    vertex at `position`, kind "split" cuts the loop edge after `position` at the
    point `extra` first.
    """
    seen = {}
    for position, vertex in enumerate(loop):
        seen.setdefault(vertex, []).append(position)

    candidates = []
    for position, vertex in enumerate(loop):
        after = loop[(position + 1) % len(loop)]
        a, b = coordinates[vertex], coordinates[after]
        entry = (arcs.get((vertex, after)) or arcs.get((after, vertex))) if arcs else None
        for j, hole_vertex in enumerate(hole_ids):
            point = coordinates[hole_vertex]
            if entry is None:
                foot, t = _foot_on_segment(point, a, b)
            else:
                # the nearest point on a CURVED edge is the radial projection,
                # which `Arc.project` already clamps to the arc's own span
                arc, start = entry
                foot = np.asarray(arc.project(point), dtype=float)
                t = arc.fraction_of(foot)
                if start != vertex:
                    t = 1.0 - t
            if foot is not None and margin < t < 1 - margin:
                candidates.append((0, float(np.linalg.norm(point - foot)),
                                   "split", position, j, foot))
        if len(seen[vertex]) > 1:
            continue                       # already a slit endpoint
        fits = True if angles is None else _absorbs_a_slit(angles[vertex], target_angle)
        for j, hole_vertex in enumerate(hole_ids):
            distance = float(np.linalg.norm(coordinates[vertex] - coordinates[hole_vertex]))
            candidates.append((1 if fits else 2, distance, "vertex", position, j, None))
    candidates.sort(key=lambda c: (c[0], c[1]))

    for tier, _, kind, position, j, extra in candidates:
        a = extra if kind == "split" else coordinates[loop[position]]
        b = coordinates[hole_ids[j]]
        if np.linalg.norm(a - b) < 1e-9:
            continue
        # A split anchor sits ON the boundary edge it splits, which the crossing
        # test would otherwise read as the cut running along an obstacle. Test a
        # fractionally shortened segment so the anchor end is clear of it; the
        # anchor itself is on the boundary by construction, which is the point.
        probe = a + (b - a) * 1e-6 if kind == "split" else a
        if any(_segments_cross(probe, b, c, d) for c, d in obstacles):
            continue
        midpoint = 0.5 * (a + b)
        if not _point_in_polygon(midpoint, outer):
            continue
        if any(_point_in_polygon(midpoint, hole) for hole in holes):
            continue
        return kind, position, j, extra
    return None


def geometry_to_tiler(geometry, target_angle=90, drop_collinear=True, normalize=True):
    """`(Tiler, desired_degree)` for a straight-edged geo2d geometry, holes included.

    Hole-free geometries take the original path (one face loop, one interior
    angle per corner). With holes the domain is cut open by `slit_outline` and
    the desired degrees are read off the *domain's* angle at each vertex, which
    for a hole corner is the reflex angle on the material side.
    """
    outer, holes = geometry_loops(geometry, drop_collinear=drop_collinear)
    if normalize:
        centre = outer.mean(axis=0)
        scale = max(float(np.abs(outer - centre).max()), 1e-9)
        outer = (outer - centre) / scale
        holes = [(hole - centre) / scale for hole in holes]
    if not holes:
        return polygon_to_tiler(outer, target_angle)
    coordinates, loop, angles, _ = slit_outline(outer, holes, target_angle)
    graph = Tiler.from_face_loops([loop], {v: list(p) for v, p in coordinates.items()})
    desired = {v: utils.rounded_desired_degree(angle, target_angle)
               for v, angle in angles.items()}
    return graph, desired


def geometry_corner_count(geometry, drop_collinear=True):
    """Distinct corners of the domain, holes included."""
    outer, holes = geometry_loops(geometry, drop_collinear=drop_collinear)
    return len(outer) + sum(len(hole) for hole in holes)


def polygon_to_tiler(points, target_angle=90):
    """A single-face `Tiler` from a CCW point array, with desired corner degrees."""
    points = np.asarray(points, dtype=float)
    node_ids = list(range(len(points)))
    coordinates = {i: points[i].copy() for i in node_ids}
    graph = Tiler.from_face_loops([node_ids], coordinates)
    interior_angles = utils.get_polygon_interior_angles(node_ids, graph.vertex_coordinates)
    desired = {v: utils.rounded_desired_degree(angle, target_angle)
               for v, angle in interior_angles.items()}
    return graph, desired


class Geo2DPolygon:
    """Initializer wrapper: the same geo2d polygon every time.

    Mirrors `src.holdout.FixedLevel` so the environment can be built with the
    stock `initialize_environment` and the `graph_initializer` hook.
    """

    def __init__(self, geometry, target_angle=90, drop_collinear=True, normalize=True):
        self.geometry = geometry
        self.target_angle = target_angle
        self.graph, self.desired_degree = geometry_to_tiler(
            geometry, target_angle, drop_collinear=drop_collinear, normalize=normalize)
        self.points = np.array([self.graph.vertex_coordinate(v)
                                for v in self.graph.vertex_list(tag=False)], dtype=float)

    @property
    def n(self):
        return len(self.points)

    def __call__(self):
        return deepcopy(self.graph), deepcopy(self.desired_degree)


def make_geo2d_env(env_config, geometry, template_size=None, drop_collinear=True,
                   normalize=True, env_maker=None):
    """The training env with a geo2d polygon as its (fixed) start state.

    `template_size` overrides the config's window; the network is
    template-agnostic, so a larger window lets bigger polygons finish without
    tripping `terminate_on_overflow` (see `load_model`).
    """
    if env_maker is None:
        from envs.environment_maker import initialize_environment as env_maker
    config = deepcopy(env_config)
    config.pop("initializer", None)
    if template_size is not None:
        config["template_size"] = int(template_size)
    target_angle = utils.average_face_angle(config.get("face_desired_degree", 4))
    config["graph_initializer"] = Geo2DPolygon(
        geometry, target_angle, drop_collinear=drop_collinear, normalize=normalize)
    # evaluate the polygon exactly as given, even if it happens to be at par
    config["resample_if_at_par"] = False
    return env_maker(config)


# --------------------------------------------------------------------------- model loading

def load_model(checkpoint, config_fn, template_size=None):
    """`src.utils.load_model_from_checkpoint`, at the config's or the given template size.

    The policy is a per-half-edge network with pooling, so nothing in the
    weights depends on `template_size`; only the stored observation space
    does. Passing the env's spaces through `custom_objects` is all it takes,
    so a checkpoint trained at template 64 evaluates at 128 unchanged.
    """
    from stable_baselines3 import PPO
    from envs.environment_maker import get_env_feature_size
    from src.feature_extractor import feature_extractor_initializer
    from src.utils import load_yaml_config

    config = load_yaml_config(config_fn)
    extractor_class, extractor_kwargs = feature_extractor_initializer(config)
    extractor_kwargs.update({"input_features": get_env_feature_size(config["environment"])})
    custom_objects = {
        "policy_kwargs": dict(features_extractor_class=extractor_class,
                              features_extractor_kwargs=extractor_kwargs),
    }
    # always re-declare the spaces: the checkpoint may have been trained at a
    # different template size than the config (or the override) asks for
    probe = make_geo2d_env(config["environment"], _unit_square_geometry(),
                           template_size=template_size)
    custom_objects["observation_space"] = probe.observation_space
    custom_objects["action_space"] = probe.action_space
    return PPO.load(checkpoint, custom_objects=custom_objects)


def _unit_square_geometry():
    geo2d = import_geo2d()
    return geo2d.Geometry([geo2d.Loop(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float))])


# --------------------------------------------------------------------------- Tiler -> geo2d

def tiler_faces(graph):
    """Node coordinates and CCW face loops of a `Tiler`, with a dense node relabelling."""
    vertices = graph.vertex_list(tag=False)
    index = {v: i for i, v in enumerate(vertices)}
    nodes = np.array([graph.vertex_coordinate(v) for v in vertices], dtype=float)
    elements = []
    for face in graph.face_list():
        loop = graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))
        elements.append([index[graph.source_vertex(h, tag=False)] for h in loop])
    return nodes, elements, index


def tiler_to_geo2d_mesh(graph):
    """The current mesh as a `geo2d.Mesh`, for geo2d's quality and plotting tools."""
    geo2d = import_geo2d()
    nodes, elements, _ = tiler_faces(graph)
    return geo2d.Mesh(nodes, elements)


def env_summary(env):
    """The win test's ingredients, read off the env."""
    return {
        "par": int(env.par),
        "face_score": int(env.global_face_score),
        "vertex_excess": int(env.global_vertex_score - env.par),
        "min_quality": float(env.min_element_quality()),
        "solved": bool(env.is_at_par()),
        "topologically_at_par": bool(env.is_topologically_at_par()),
        "overflowed": bool(env.template_overflowed()),
        "half_edges": int(env.graph.number_of_half_edges()),
        "faces": len(env.graph.face_list()),
        "vertices": len(env.graph.vertex_list()),
        "moves": int(env.num_steps),
    }


# --------------------------------------------------------------------------- re-smoothing

def _boundary_neighbours(mesh):
    """Per boundary node, its two neighbours along the boundary."""
    neighbours = {}
    for a, b in mesh.boundary_edges():
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))
    return neighbours


def resmooth_mesh(mesh, corners, iters=10, method="optimize", slide="optimize",
                  project=None, tangents=None):
    """Re-smooth a geo2d mesh of a straight-edged polygon with geo2d's smoothers.

    Alternates one sweep of geo2d's interior smoother (`optimize`, the
    untangling condition-number smoother, or `smart_laplacian`) with the
    boundary fixed, and one sweep that slides every non-corner boundary node
    along its straight edge: the target is the Laplacian mean of *all* its
    neighbours projected onto the segment between its two boundary
    neighbours, accepted only if the minimum corner quality of the adjacent
    elements does not drop (Freitag's rule, as in `smart_laplacian`).

    `corners` is a boolean mask of the nodes that are corners of the domain;
    those never move. `Tiler.smooth_vertices` differs in two ways: it is a
    plain Laplacian (it can fold or flatten an element the topology allows
    to be regular), and it moves boundary nodes only from their boundary
    neighbours, so the interior has no say in where they sit.

    `slide` picks the target the sliding node aims at. "laplacian" is the
    original: the neighbour mean, which is a proxy for the quantity actually
    being measured. "optimize" (the default) runs the same per-node distortion
    minimisation the interior sweep uses and projects THAT onto the chord, so
    the node aims at the objective itself. Both stay on the boundary and both
    keep Freitag's acceptance test, so neither can make an element worse; the
    optimised target is simply a better guess. Measured on three hand-played
    par solutions: 0.398 -> 0.623, 0.398 -> 0.461, and 0.140 -> 0.533, the last
    of which crosses the 0.4 gate. `False` disables sliding entirely.

    `project(node, point) -> point` bends the slide onto a CURVED boundary. The
    chord between a node's two boundary neighbours is the right track only on a
    straight edge; on an arc it cuts the corner, so the node leaves the domain
    and something has to put it back. Snapping afterwards was that something,
    and it is why refining a curved mesh used to invert elements: the smoother
    slid the boundary off the curve to open a fold, the snap put it straight
    back where it had folded, and the quality guard then reverted the whole
    sweep -- min quality +0.315 thrown away and -0.348 kept. Projecting INSIDE
    the loop means Freitag's acceptance test judges the position actually kept,
    which is the same rule the outer guard already follows.
    """
    geo2d = import_geo2d()
    from geo2d.smooth import _min_quality, _NodeCorners, _optimize_node  # noqa: E402

    corners = np.asarray(corners, dtype=bool)
    boundary = mesh.is_boundary()
    smoother = geo2d.optimize if method == "optimize" else geo2d.smart_laplacian
    sliders = np.nonzero(boundary & ~corners)[0]
    neighbours = _boundary_neighbours(mesh)
    # The boundary SLIDE is judged on corner quality too, so it wants the same
    # tangents the interior sweep does -- a node slid along its arc is accepted
    # or refused on how the corners read, and reading them off chords is the
    # thing being corrected.
    node_corners = _NodeCorners(mesh, tangents)
    local = {int(v): node_corners.of_node(int(v)) for v in sliders}
    offsets, all_neighbours = mesh.node_neighbors()

    out = mesh.copy()
    for _ in range(iters):
        out = (smoother(out, iters=1, fixed=boundary, tangents=tangents)
               if tangents else smoother(out, iters=1, fixed=boundary))
        if not slide:
            continue
        # the corner geometry moved with the interior sweep, so a node aiming at
        # the objective has to be re-read against the mesh as it now stands
        node_corners = _NodeCorners(out, tangents) if slide == "optimize" else node_corners
        X = out.nodes
        for v in sliders:
            v = int(v)
            tri, sn, dirs = (node_corners.of_node(v) if slide == "optimize"
                             else local[v])
            P = X[tri]
            a, b = (X[n] for n in neighbours[v])
            direction = b - a
            length = np.linalg.norm(direction)
            if length < 1e-12:
                continue
            direction /= length
            neighbour_mean = X[all_neighbours[offsets[v]:offsets[v + 1]]].mean(axis=0)
            if slide == "optimize":
                try:
                    target = _optimize_node(P, tri == v, sn, X[v].copy(), 6, dirs)
                except np.linalg.LinAlgError:
                    # geo2d's Newton step guards with `det > 0`, which a tiny
                    # positive determinant passes and `solve` still rejects. One
                    # node failing is not a reason to lose the whole mesh, and
                    # the neighbour mean is the target the Laplacian slide would
                    # have used anyway -- Freitag's acceptance test below still
                    # refuses it if it makes an element worse.
                    target = neighbour_mean
            else:
                target = neighbour_mean
            # stay strictly between the boundary neighbours, on their line
            s = float(np.clip((target - a) @ direction, 0.05 * length, 0.95 * length))
            candidate = a + s * direction
            if project is not None:
                moved = project(v, candidate)
                if moved is not None:
                    candidate = np.asarray(moved, dtype=float)
            before = _min_quality(P, sn, dirs)
            P_new = P.copy()
            P_new[tri == v] = candidate
            if _min_quality(P_new, sn, dirs) >= before - 1e-12:
                X[v] = candidate
    return out


def tangent_table(graph, index):
    """{(node, neighbour): unit tangent} for every CURVED side, for the smoother.

    geo2d's smoother builds each corner's Jacobian from the two chords meeting
    there. On a curved edge the element's side follows the arc, so the direction
    it leaves the corner is the tangent -- the same rule
    `Tiler.corner_shape_qualities` uses, and the same one the corner WANTS use.
    Passing this as `tangents=` makes the smoother optimise the quantity we
    score instead of a chord-based proxy for it.

    Returns {} on a straight domain, which leaves geo2d on its original path.
    """
    arcs = getattr(graph, "boundary_arcs", None)
    if not arcs:
        return {}
    from src.boundary_arcs import edge_direction
    table = {}
    for (first, second), (_, _) in arcs.items():
        if first not in index or second not in index:
            continue
        for here, there in ((first, second), (second, first)):
            direction = np.asarray(edge_direction(graph, here, there), dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                continue
            table[(int(index[here]), int(index[there]))] = direction / norm
    return table


def arc_projection(graph, index):
    """A `project` for `resmooth_mesh`: a boundary node onto its own arc.

    `index` maps the Tiler's vertices to geo2d node rows. A vertex can sit on
    two arcs -- at a joint between two of them, or as the midpoint of one that
    was split -- so the nearer projection wins; for a split arc the two agree.
    Returns None when the domain has no curved edges, which keeps the straight
    path exactly as it was.
    """
    if not graph.boundary_arcs:
        return None
    by_node = {}
    for (first, second), (arc, _) in graph.boundary_arcs.items():
        for vertex in (first, second):
            if vertex in index:
                by_node.setdefault(int(index[vertex]), []).append(arc)
    if not by_node:
        return None

    def project(node, point):
        arcs = by_node.get(int(node))
        if not arcs:
            return None
        point = np.asarray(point, dtype=float)
        best, distance = None, None
        for arc in arcs:
            candidate = np.asarray(arc.project(point), dtype=float)
            gap = float(np.linalg.norm(candidate - point))
            if distance is None or gap < distance:
                best, distance = candidate, gap
        return best

    return project


# ONE smoothing setting, shared by the training gate and every evaluator.
#
# They had drifted to three: the gate ran 4 iterations with the cheap laplacian
# slide, `score_quality_objective` ran 15 with the optimised slide, and
# `curved_report` ran 20. Same code path, different answers -- measured on 20
# real curved meshes, the gate and the scorer disagreed on 5 of them by up to
# 0.189, and the gate was the OPTIMISTIC one on 3, so the agent could be trained
# to accept meshes the scorer rejects. The original note claimed the cheap slide
# "errs toward refusing"; that is no longer true under the tangent metric.
#
# The SLIDE is what matters, not the iteration count. With `laplacian` the
# result sits 0.18 from converged however many iterations it gets (4, 8 and 15
# all give 0.18); with `optimize` it is within 0.0015 at FOUR and 0.0001 at
# eight. So four optimised iterations is the cheap converged setting: 126 ms a
# call against 439 for the scorer's old fifteen, and the gate fires about twice
# an episode.
SMOOTHING = dict(iters=4, method="optimize", slide="optimize", warm_start=5)


def resmooth_env(env, iters=None, method=None, slide=None, warm_start=None):
    """Re-smooth the env's current mesh in place with `resmooth_mesh`.

    The domain corners are the Tiler's user-defined vertices, exactly the
    ones its own smoother pins. Half-edge angles are refreshed so that
    `env.min_element_quality()` and `env.is_at_par()` describe the
    re-smoothed mesh. Call this only after an episode has ended; it changes
    the coordinates the policy would otherwise observe.
    """
    iters = SMOOTHING["iters"] if iters is None else iters
    method = SMOOTHING["method"] if method is None else method
    slide = SMOOTHING["slide"] if slide is None else slide
    warm_start = SMOOTHING["warm_start"] if warm_start is None else warm_start
    graph = env.graph
    if warm_start:
        # the untangler is a local optimiser and can start inside a tangle it
        # cannot escape; a few Laplacian sweeps cost about a millisecond. This
        # lives HERE, not in the caller, so every caller gets the same treatment
        graph.smooth_vertices(num_iter=warm_start)
        env._update_half_edge_angles()
    nodes, elements, index = tiler_faces(graph)
    corners = np.zeros(len(nodes), dtype=bool)
    for vertex, i in index.items():
        corners[i] = graph.is_user_defined_vertex(vertex)
    geo2d = import_geo2d()
    before = env.min_element_quality()
    # the snap below re-anchors arc spans; a revert must restore them with the
    # coordinates, or the kept mesh reads its tangents off spans that end where
    # the REJECTED vertices were (measured: -0.11 reported as -0.93)
    arcs_before = graph.boundary_arcs.copy() if graph.boundary_arcs else None
    mesh = resmooth_mesh(geo2d.Mesh(nodes, elements), corners, iters=iters,
                         method=method, slide=slide,
                         project=arc_projection(graph, index),
                         tangents=tangent_table(graph, index))
    for vertex, i in index.items():
        graph.set_vertex_coordinate(vertex, mesh.nodes[i])
    # The slide now lands ON the arc (see `arc_projection`), so this is
    # insurance rather than the fix it used to be. It stays because it is cheap
    # and because it is the last line of defence if any other path moves a
    # boundary vertex; it must stay BEFORE the quality guard below, so the guard
    # judges the mesh actually kept.
    graph.snap_to_boundary_arcs()
    env._update_half_edge_angles()
    # `optimize` is not monotone on meshes that still hold odd faces; never
    # report (or draw) a mesh the re-smoother made worse than the Laplacian left it
    if env.min_element_quality() < before - 1e-9:
        for vertex, i in index.items():
            graph.set_vertex_coordinate(vertex, nodes[i])
        if arcs_before is not None:
            graph.boundary_arcs = arcs_before
        env._update_half_edge_angles()
        mesh = geo2d.Mesh(nodes, elements)
    return mesh


def segments_for(sweep_degrees, per_segment=45.0):
    """Nearest power of two to `sweep / per_segment`, at least 1.

    A power of two because `insert_vertex` halves: 90 -> 2, 180 -> 4, and
    120 -> 2 rather than 3, which is the rounding.
    """
    want = abs(sweep_degrees) / per_segment
    if want <= 1.0:
        return 1
    return int(2 ** round(math.log2(want)))


def densify_arcs(graph, desired, per_segment=45.0):
    """Split every boundary arc until no segment sweeps more than ~per_segment.

    Not discretising: the arcs stay attached. It is the agent's own
    `insert_vertex` applied before the episode starts, so each new vertex lands
    ON the curve, the arc splits in two, its tangent reads 180 and it wants the
    generic degree three -- contributing nothing to the bound. par is preserved
    by construction (measured on 8 of 8 instances).

    Why it matters: an arc is ONE edge in the control polygon and several in
    any mesh that respects it. Left coarse, a 90-degree fillet is one element
    side whose curved block cannot score above ~0.7 and whose Laplacian-drawn
    stand-in is a chord a sagitta away from the truth; a circular hole is a
    4-gon whose O-grid quads carry 45-degree corners and fold under the
    tangent metric. At 45 degrees of sweep the block is nearly straight-sided,
    the joints read as the flat points the straight agent already knows want
    degree three, and `max_steps` -- scaled from the half-edge count -- grows
    exactly where the geometry demands more work.

    Returns the number of vertices added.
    """
    arcs = getattr(graph, "boundary_arcs", None)
    if not arcs:
        return 0
    added = 0
    while True:
        target = None
        for half_edge in graph.half_edge_list():
            if not graph.half_edge_on_boundary(half_edge):
                continue
            a = graph.source_vertex(half_edge, tag=False)
            b = graph.target_vertex(half_edge, tag=False)
            arc = graph.boundary_arcs.get(a, b)
            if arc is None:
                continue
            if segments_for(math.degrees(arc._sweep()), per_segment) > 1:
                target = half_edge
                break
        if target is None:
            return added
        graph.insert_vertex(target)
        new = graph.target_vertex(target, tag=False)
        desired[new] = 3              # interior of a smooth arc: tangent 180
        added += 1



# --------------------------------------------------------------------------- training initializer

class Geo2DRandomPolygon:
    """Initializer that draws a fresh geo2d polygon every episode.

    Plugs into the env through the `Custom` hook in `envs/environment_maker.py`:

        initializer:
          name: Custom
          factory: "src.geo2d_bridge:make_initializer"
          preset: straight        # any geo2d preset
          n_holes: 0              # single polygons only, for now
          ratio: 5                # optional geo2d overrides; ranges as [lo, hi]
          n_ops: [1, 2]
          max_corners: 24         # reject outlines the template cannot hold

    Each call draws a new geo2d seed from `rng`, so every env worker (which
    constructs its own initializer) sees its own stream; pass `seed` for a
    reproducible one. Outlines with more than `max_corners` corners are
    redrawn: an n-gon needs at least (n - 2) / 2 quads, i.e. 2(n - 2) half-edges,
    and the agent's solutions run well above that, so a corner cap is how the
    template size is respected. `set_degree_range` is the curriculum hook.
    """

    def __init__(self, preset="straight", n_holes=0, target_angle=90, seed=None,
                 drop_collinear=True, min_corners=3, max_corners=None,
                 max_rejection_tries=200, allow_curves=False, densify_sweep=None,
                 **overrides):
        self.geo2d = import_geo2d()
        # Degrees of sweep per boundary-arc segment; None leaves arcs as drawn.
        # Applied AFTER the corner filter, so which domains are drawn does not
        # change and a densified suite is paired with the undensified one.
        self.densify_sweep = None if densify_sweep is None else float(densify_sweep)
        self.preset = preset
        self.target_angle = target_angle
        self.drop_collinear = drop_collinear
        self.min_corners = int(min_corners)
        self.max_corners = None if max_corners is None else int(max_corners)
        self.max_rejection_tries = max_rejection_tries
        # Off by default so every existing config is unchanged. Turned on, a
        # curved draw is kept and carried natively instead of being redrawn.
        self.allow_curves = bool(allow_curves)
        self.rng = np.random.default_rng(seed)
        # YAML gives lists; geo2d wants (lo, hi) tuples for ranges
        self.overrides = {"n_holes": _as_range(n_holes)}
        self.overrides.update({key: _as_range(value) for key, value in overrides.items()})
        self.last_geometry = None
        self.generator_failures = 0
        self.last_failure = None

    def set_degree_range(self, degree_range):
        degree_range = list(degree_range)
        self.min_corners, self.max_corners = int(min(degree_range)), int(max(degree_range))

    def draw_geometry(self):
        seed = int(self.rng.integers(0, 2**31 - 1))
        return self.geo2d.generate(seed, preset=self.preset, **self.overrides)

    def __call__(self):
        for _ in range(self.max_rejection_tries):
            try:
                geometry = self.draw_geometry()
                # A curved outline does not need discretising and must not be
                # refused: the corner want at a vertex is its TANGENT angle, and
                # the arcs ride along on the graph so an inserted vertex lands on
                # the curve. PPO needs no solution, only an environment, so
                # geo2d's own curved presets can train directly.
                curved = any(any(loop.is_arc()) for loop in geometry.loops)
                if self.allow_curves and curved:
                    # holes included: `curved_geometry_to_tiler` cuts them open
                    # with the same slit a straight-edged hole gets, and the
                    # arcs survive the renumbering because BoundaryArcs is keyed
                    # by vertex pair
                    graph, desired = curved_geometry_to_tiler(
                        geometry, target_angle=self.target_angle,
                        drop_collinear=self.drop_collinear)
                else:
                    graph, desired = geometry_to_tiler(
                        geometry, self.target_angle, drop_collinear=self.drop_collinear)
            except (NotImplementedError, ValueError, ArithmeticError, IndexError) as error:
                # geo2d's own validity loop can still let a degenerate outline through
                # (seen: "a loop needs at least 3 points" from _assemble); an env
                # worker must never die on a bad draw, so redraw and count it
                self.generator_failures += 1
                self.last_failure = repr(error)
                continue
            n = len(desired)
            if n < self.min_corners or (self.max_corners is not None and n > self.max_corners):
                continue
            self.last_geometry = geometry
            if self.densify_sweep and getattr(graph, "boundary_arcs", None):
                densify_arcs(graph, desired, self.densify_sweep)
            return graph, desired
        raise RuntimeError(
            f"Geo2DRandomPolygon: no acceptable outline in {self.max_rejection_tries} draws "
            f"(preset={self.preset!r}, corners {self.min_corners}..{self.max_corners}, "
            f"{self.overrides})")


def _as_range(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value) if len(value) == 2 else tuple(value)
    return value


def make_initializer(**options):
    """Factory for the `Custom` initializer hook; see `Geo2DRandomPolygon`."""
    return Geo2DRandomPolygon(**options)
