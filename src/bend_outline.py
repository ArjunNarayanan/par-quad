"""Bend a certified instance's straight edges into arcs, without moving par.

par is a function of the corner WANTS, and a want is `round(angle / 90) + 1`.
Bending edge (a, b) into a circular arc of total turning `theta` rotates the
tangent at each endpoint by `theta/2`, so both interior angles move by that much
and nothing else changes. Keep every endpoint inside its existing 90-degree bin
and every want is untouched -- so par is untouched, and the instance's recorded
solution is still exactly optimal on a domain that is now curved.

That is the whole trick: geometric diversification that provably does not
change the answer. Measured over 20 certified instances, the median edge can take 54
degrees of bend and 92% can take at least 20, so there is room to make domains
that genuinely look curved.

What this does NOT do is change the TOPOLOGY of the training distribution. A
`rounded` geo2d domain replaces sharp re-entrant corners with runs of convex arc
control points, so its corner wants sit at 2:62% 3:22% 4:16% where a rectilinear
one is 2:49% 3:13% 4:38%. Bending preserves wants by construction, so it teaches
the agent that curvature is not a threat -- not what a fillet-heavy domain looks
like. Those are different gaps and this closes the first.

THE DEEPER OBJECTION, and it is the right one: because a bend preserves every
want by construction, the optimal TOPOLOGY on a bent domain is identical to the
straight one it came from. The recorded solution is bit-for-bit the same move
sequence. So the demonstration carries no information about how curvature should
change a decision -- the agent sees a curved outline in its features and the
correct answer never moves. Measured, 135 of 584 vertex insertions do land on a
bent edge and 103 of 159 instances touch an arc at least once, so it is not that
arcs go untouched; it is that touching one never costs or buys anything. This
teaches robustness to curvature and cannot teach anything else.

AND IT IS NOT A SUBSTITUTE FOR GEO2D'S CURVED GENERATOR. Drawn side by side the
two are plainly different animals: a bent certified instance is a small, mostly
convex polygon with a gentle bow -- median 7 corners, 2 arcs -- while a geo2d
`rounded` domain is a rectilinear part with fillets, notches and round cutouts,
median 15 corners and 5 arcs. Bending BOWS AN EDGE; geo2d FILLETS A CORNER.

A fillet cannot be applied to a certified instance for free. It deletes a want-2
corner and adds two want-3 smooth joints, which moves the Gauss-Bonnet sum by +1
per fillet -- so par changes -- and the recorded solution does not survive,
because the corner vertex it was built around is gone. Filleted training data
WITH answers needs a seed family that meshes a filleted domain at par by
construction, the way `pinwheel_annulus_mesh` does for turned holes. PPO is the
other half and needs none of this: it takes geo2d's curved domains directly, via
`Geo2DRandomPolygon(allow_curves=True)`.
"""

import numpy as np

from src.boundary_arcs import Arc


def bin_slack(angle, target_angle=90.0):
    """How far this interior angle can move before its desired degree changes."""
    scaled = angle / target_angle
    return target_angle * (0.5 - abs(scaled - round(scaled)))


def arc_through(start, end, turning_deg):
    """The circular arc from `start` to `end` whose total turning is `turning_deg`.

    Positive turning bulges to the LEFT of the chord, which for a
    counter-clockwise outline is outward, into the material's complement.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    theta = np.radians(float(turning_deg))
    if abs(theta) < 1e-9:
        return None

    chord = end - start
    length = float(np.linalg.norm(chord))
    if length < 1e-12:
        return None
    radius = length / (2.0 * np.sin(abs(theta) / 2.0))
    # centre sits off the chord midpoint, perpendicular, by the sagitta's complement
    midpoint = 0.5 * (start + end)
    normal = np.array([-chord[1], chord[0]]) / length
    offset = radius * np.cos(theta / 2.0)
    centre = midpoint - np.sign(theta) * offset * normal

    begin = float(np.arctan2(*(start - centre)[::-1]))
    finish = float(np.arctan2(*(end - centre)[::-1]))
    # Which way round the centre the traversal goes is NOT the sign of theta: a
    # left-bulging arc puts its centre to the RIGHT of the chord, so walking
    # start -> end goes clockwise around it. Pick the direction whose sweep has
    # the magnitude asked for, rather than assuming.
    anticlockwise = (finish - begin) % (2 * np.pi)
    return Arc(centre, radius, begin, finish,
               ccw=abs(anticlockwise - abs(theta)) < abs(anticlockwise - 2 * np.pi - -abs(theta)))


def tangent_turn(arc, at_start):
    """How far the arc's tangent is rotated from its chord, in degrees.

    `theta/2` by construction, but computing it from the arc keeps the two
    definitions from drifting apart.
    """
    point_a, point_b = arc.point(0.0), arc.point(1.0)
    chord = point_b - point_a
    chord /= max(float(np.linalg.norm(chord)), 1e-12)
    step = 1e-6
    if at_start:
        direction = (arc.point(step) - point_a)
    else:
        direction = (point_b - arc.point(1.0 - step))
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    cross = chord[0] * direction[1] - chord[1] * direction[0]
    dot = float(np.dot(chord, direction))
    return float(np.degrees(np.arctan2(cross, dot)))


def _segments_cross(p1, p2, p3, p4):
    def side(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (side(p1, p2, p3) != side(p1, p2, p4)
            and side(p3, p4, p1) != side(p3, p4, p2))


def _arc_stays_clear(arc, loop_points, skip, samples=9):
    """Does the bulge cross any edge of the outline it is not adjacent to?"""
    probe = [arc.point(t) for t in np.linspace(0.0, 1.0, samples)]
    count = len(loop_points)
    for i in range(count):
        if i in skip:
            continue
        a, b = loop_points[i], loop_points[(i + 1) % count]
        for first, second in zip(probe[:-1], probe[1:]):
            if _segments_cross(first, second, a, b):
                return False
    return True


def bend_outline(loop, coordinates, angles, rng, target_angle=90.0,
                 probability=0.5, margin=4.0, max_turning=70.0):
    """Bend some edges of a closed outline into arcs, leaving every want alone.

    Returns `{(a, b): (Arc, start_vertex)}` for the edges that were bent.
    `margin` is degrees of the bin held back as headroom, so an angle never
    lands exactly on a bin edge where the tie rule would take over.
    """
    count = len(loop)
    points = [np.asarray(coordinates[v], dtype=float) for v in loop]
    slack = {v: bin_slack(angles[v], target_angle) - margin for v in loop}
    arcs = {}
    # each vertex has ONE budget shared by its two edges, so spend half on each
    # and neither edge can push it out of its bin even if both bend hard
    for index in range(count):
        if rng.random() > probability:
            continue
        here, there = loop[index], loop[(index + 1) % count]
        room = min(slack[here], slack[there])
        if room <= 1.0:
            continue
        limit = min(2.0 * room, max_turning)
        turning = float(rng.uniform(-limit, limit))
        if abs(turning) < 2.0:
            continue
        # A bulge that crosses a neighbouring edge is the commonest failure --
        # 35% of attempts on concave outlines with short edges. Halving and
        # retrying recovers most of them, because the crossing is a function of
        # how far the arc bows out, not of the direction it bows in.
        arc = None
        for _ in range(4):
            candidate = arc_through(points[index], points[(index + 1) % count], turning)
            if candidate is not None and _arc_stays_clear(candidate, points, skip={index}):
                arc = candidate
                break
            turning *= 0.5
            if abs(turning) < 2.0:
                break
        if arc is None:
            continue
        arcs[(here, there)] = (arc, here)
        slack[here] -= abs(turning) / 2.0
        slack[there] -= abs(turning) / 2.0
    return arcs
