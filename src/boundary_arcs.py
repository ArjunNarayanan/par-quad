"""Curved boundaries, carried natively rather than discretised.

A curved domain is not a different problem. Topology does not know what shape an
edge is, and the corner want at a vertex comes from the TANGENT angle there --
which geo2d already computes: `Loop.interior_angles()` reports "the angle of the
material on the left, taking the edge tangents into account", so a smooth arc
joint reads 180 and wants degree three exactly like a flat point on a straight
edge. Discretising a curve would only manufacture corners that are an artefact
of the sampling, and would make par a function of how finely you sampled.

So the outline is used as geo2d gives it -- one vertex per control point -- and
the only thing the mesh has to remember is which boundary EDGES are arcs. Two
operations need it:

* inserting a vertex on a boundary edge, which must land ON the arc rather than
  on the chord between its endpoints, and splits the arc into two arcs;
* moving a boundary vertex during smoothing, which must slide it along the arc
  rather than off it.

Everything else -- the scores, par, the action space, the network -- is
unchanged, because none of it ever looks at edge geometry.

geo2d only builds circular arcs, so that is all this supports. A general curve
would need the same two operations and nothing more.

WHAT THIS DOES AND DOES NOT BUY. The representation works to machine precision:
vertices land on their arcs at 1e-16 and stay there through insertion,
normalization and smoothing. The straight-edge-trained policy still fails on
curved domains -- 1 of 16 on `rounded` -- and the reason is NOT the curvature.
Of twelve failures, eight missed on face score and eleven on vertex excess;
none missed on quality alone. The domains are simply out of distribution:

    straight/straight   want mix   2:49%  3:13%  4:38%
    curved/rounded      want mix   2:62%  3:22%  4:16%

A fillet replaces one sharp re-entrant corner with a run of convex arc control
points, so a curved domain is mostly want-2 and want-3 where a rectilinear one
is heavily want-4. The policy has never seen that shape of problem. This is a
training-data gap, not a representation gap, and closing it means teaching the
generator to build curved domains -- which it can, since par falls out of the
tangent angles exactly as it does for straight edges.
"""

import numpy as np


class Arc:
    """A circular arc between two points, carried by centre and orientation."""

    __slots__ = ("centre", "radius", "start", "end", "ccw")

    def __init__(self, centre, radius, start, end, ccw):
        self.centre = np.asarray(centre, dtype=float)
        self.radius = float(radius)
        self.start = float(start)
        self.end = float(end)
        self.ccw = bool(ccw)

    @classmethod
    def from_geo2d(cls, edge):
        """From a geo2d `Arc` edge, which carries centre, radius and endpoints."""
        centre = np.asarray(edge.center, dtype=float)
        radius = float(edge.radius)
        start = _angle_of(np.asarray(edge.p0, dtype=float) - centre)
        end = _angle_of(np.asarray(edge.p1, dtype=float) - centre)
        # which way round: ask the arc where its own midpoint is
        midpoint = np.asarray(edge.point(0.5), dtype=float)
        ccw = _sweeps_ccw(start, end, _angle_of(midpoint - centre))
        return cls(centre, radius, start, end, ccw)

    def point(self, t):
        """The point at arc-length fraction `t` along the arc."""
        return self.centre + self.radius * _unit(self.start + t * self._sweep())

    def project(self, point):
        """The nearest point ON the arc's circle, clamped to the arc's span.

        Used when a smoother has moved a boundary vertex off the curve: the
        radial projection is the nearest point on the circle, and clamping keeps
        it between the endpoints rather than letting it wrap round the far side.
        """
        offset = np.asarray(point, dtype=float) - self.centre
        norm = float(np.linalg.norm(offset))
        if norm < 1e-12:
            return self.point(0.5)
        angle = _angle_of(offset)
        fraction = _fraction_along(self.start, self._sweep(), angle)
        if 0.0 <= fraction <= 1.0:
            return self.point(fraction)
        # Outside the arc's span: clamp to the NEARER end. Clamping the fraction
        # instead picks whichever end the parameterisation happened to run past,
        # which for a point just short of the start is the wrong one.
        if abs(_wrapped(angle - self.start)) <= abs(_wrapped(angle - self.end)):
            return self.point(0.0)
        return self.point(1.0)

    def fraction_of(self, point):
        """Where `point` sits along the arc, as a fraction in [0, 1].

        The inverse of `point(t)` for anything on the arc, and the reason a slit
        can be anchored on a curve: the caller needs to know a foot is not sitting
        right on top of an endpoint before splitting there.
        """
        offset = np.asarray(point, dtype=float) - self.centre
        if float(np.linalg.norm(offset)) < 1e-12:
            return 0.5
        return _fraction_along(self.start, self._sweep(), _angle_of(offset))

    def sample(self, count=16):
        """`count` points along the arc, for geometric predicates.

        A slit must not cross the boundary, and a chord through a bulging arc is
        not the boundary. Sampling is the cheap, conservative way to give the
        crossing and inside tests the real outline.
        """
        return np.array([self.point(t) for t in np.linspace(0.0, 1.0, count)])

    def split(self, t=0.5):
        """Two arcs meeting at `t`, which is what inserting a vertex does."""
        middle = _angle_of(self.point(t) - self.centre)
        return (Arc(self.centre, self.radius, self.start, middle, self.ccw),
                Arc(self.centre, self.radius, middle, self.end, self.ccw))

    def _sweep(self):
        delta = (self.end - self.start) % (2 * np.pi)
        if not self.ccw:
            delta -= 2 * np.pi
        return delta

    def __repr__(self):
        return (f"Arc(centre={self.centre.round(3).tolist()}, r={self.radius:.3f}, "
                f"sweep={np.degrees(self._sweep()):.1f}deg)")


def _angle_of(vector):
    return float(np.arctan2(vector[1], vector[0]))


def _unit(angle):
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def _sweeps_ccw(start, end, through):
    """Does the arc from `start` to `end` passing `through` run anticlockwise?"""
    forward = (end - start) % (2 * np.pi)
    reached = (through - start) % (2 * np.pi)
    return reached <= forward


def _wrapped(angle):
    """`angle` folded into (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _fraction_along(start, sweep, angle):
    if abs(sweep) < 1e-12:
        return 0.0
    delta = (angle - start) % (2 * np.pi)
    if sweep < 0 and delta > 0:
        # A clockwise arc runs the other way round, so delta belongs in
        # (-2pi, 0]. Subtracting unconditionally moves an exact 0 -- the arc's
        # own START -- to -2pi, whose fraction is 4 for a quarter turn, clamps
        # to 1, and returns the END. Every pinned vertex on a concave arc then
        # teleports to the far corner the moment anything projects it.
        delta -= 2 * np.pi
    return delta / sweep


class BoundaryArcs:
    """Which boundary edges are arcs, keyed by their two endpoint vertices.

    Keyed by a vertex PAIR rather than a half-edge id, because half-edge ids do
    not survive the edit operations and vertex ids do. Each entry also records
    which of the two endpoints the arc runs FROM, so a split can hand the two
    halves to the right pairs without guessing from coordinates.
    """

    def __init__(self, arcs=None):
        self._arcs = dict(arcs or {})

    @staticmethod
    def key(a, b):
        return (a, b) if a <= b else (b, a)

    def __len__(self):
        return len(self._arcs)

    def __bool__(self):
        return bool(self._arcs)

    def get(self, a, b):
        entry = self._arcs.get(self.key(a, b))
        return None if entry is None else entry[0]

    def add(self, a, b, arc, start_vertex):
        """`arc` runs from `start_vertex`, which must be `a` or `b`."""
        assert start_vertex in (a, b), (start_vertex, a, b)
        self._arcs[self.key(a, b)] = (arc, start_vertex)

    def reanchor(self, a, b, vertex, point):
        """Move one END of the a-b arc onto `point`, keeping its circle.

        A boundary vertex is allowed to SLIDE along its own curve -- that is how
        the smoother opens a folded element without leaving the domain -- but
        the stored arc records the span by its two endpoint ANGLES, and sliding
        the vertex does not update them. The entry then describes an arc that no
        longer ends where its edge ends.

        That is not cosmetic. `edge_direction` finds the tangent by asking which
        END of the arc the vertex sits at; with a stale span neither end matches,
        it silently takes the far one, and returns a tangent for the wrong point.
        Corner WANTS and the corner QUALITY are both built on that tangent, so a
        stale span corrupts par and the reward together. Measured after the
        sliding smoother was introduced: a third to a half of all arcs had
        drifted, by up to 0.24 on a unit-scaled outline.

        Only the angle moves; centre and radius are untouched, so the vertex
        stays on exactly the curve the domain specified.
        """
        entry = self._arcs.get(self.key(a, b))
        if entry is None:
            return
        arc, start = entry
        offset = np.asarray(point, dtype=float) - arc.centre
        if float(np.linalg.norm(offset)) < 1e-12:
            return
        angle = _angle_of(offset)
        if vertex == start:
            moved = Arc(arc.centre, arc.radius, angle, arc.end, arc.ccw)
        else:
            moved = Arc(arc.centre, arc.radius, arc.start, angle, arc.ccw)
        # A slide SHRINKS the arc on one side of the vertex and GROWS the one on
        # the other, so refusing to grow -- which an earlier version of this did
        # -- repairs only half the drift and leaves the neighbour stale. What
        # must not happen is the span flipping round the far side of the circle,
        # which is a sign change, or wrapping the whole way round.
        sweep, was = moved._sweep(), arc._sweep()
        if sweep == 0.0 or np.sign(sweep) != np.sign(was):
            return
        if abs(sweep) >= 2 * np.pi - 1e-9:
            return
        self._arcs[self.key(a, b)] = (moved, start)

    def split(self, a, b, middle):
        """Replace the arc a-b with two halves meeting at vertex `middle`.

        Returns the point the new vertex should sit at, or None when a-b is
        straight and the caller should keep its own midpoint.
        """
        entry = self._arcs.pop(self.key(a, b), None)
        if entry is None:
            return None
        arc, start = entry
        first, second = arc.split(0.5)
        finish = b if start == a else a
        self._arcs[self.key(start, middle)] = (first, start)
        self._arcs[self.key(middle, finish)] = (second, middle)
        return arc.point(0.5)

    def drop(self, a, b):
        self._arcs.pop(self.key(a, b), None)

    def angle_at(self, a, b, vertex):
        """The arc's own angle at one of its endpoint vertices."""
        entry = self._arcs.get(self.key(a, b))
        if entry is None:
            return None
        arc, start = entry
        return arc.start if start == vertex else arc.end

    def merge(self, first, middle, second):
        """Undo a split: the arc `first`-`middle`-`second` becomes one again.

        Deleting a vertex that was inserted ON a curve has to put the curve
        back. Without this the two halves stay in the table keyed on a vertex
        that no longer exists, the edge that reappears is treated as STRAIGHT,
        and the domain quietly loses its curve under the player -- who can then
        insert on that edge and land on the chord instead of the circle.
        """
        left = self._arcs.get(self.key(first, middle))
        right = self._arcs.get(self.key(middle, second))
        if left is None or right is None:
            # not both curved: whatever is left refers to a dead vertex
            self.drop(first, middle)
            self.drop(middle, second)
            return None
        start_angle = self.angle_at(first, middle, first)
        end_angle = self.angle_at(middle, second, second)
        arc = left[0]
        self.drop(first, middle)
        self.drop(middle, second)
        merged = Arc(arc.centre, arc.radius, start_angle, end_angle, arc.ccw)
        self._arcs[self.key(first, second)] = (merged, first)
        return merged

    def copy(self):
        return BoundaryArcs(self._arcs)

    def items(self):
        return self._arcs.items()


def edge_direction(graph, here, there):
    """The direction the material leaves `here` along the edge towards `there`.

    On a straight edge this is the chord. On an arc it is the TANGENT, and that
    difference is the whole of curved-boundary support: the apex of a semicircle
    is a smooth point that wants three, while the chord through it reads as a
    sharp corner and would want two. `Tiler.half_edge_angles` stays on chords --
    it measures the element that is actually drawn, which is what the quality
    metric needs -- so anything asking what a CORNER WANTS comes here instead.

    `game/js/engine.js:edgeDirection` is the port; they must agree.
    """
    import numpy as np

    coordinates = graph.vertex_coordinates
    arcs = getattr(graph, "boundary_arcs", None)
    arc = arcs.get(here, there) if arcs else None
    if arc is None:
        return np.asarray(coordinates[there], float) - np.asarray(coordinates[here], float)
    # exactly the tangent, not a finite difference: on a circle the tangent is
    # the radius turned a quarter turn, in the direction the arc is travelled.
    # A difference quotient here is accurate to about 5e-5 degrees, which is
    # fine for a want but needlessly imprecise for a quantity with a closed form
    at_start = np.allclose(arc.point(0.0), coordinates[here], atol=1e-9)
    end = 0.0 if at_start else 1.0
    radial = np.asarray(arc.point(end), float) - arc.centre
    tangent = np.array([-radial[1], radial[0]])          # a quarter turn CCW
    if arc._sweep() < 0:
        tangent = -tangent
    # leaving `here` means travelling along the arc away from it
    return tangent if at_start else -tangent


def corner_angles(graph):
    """Interior angle at every vertex of the start outline, tangent-aware.

    Summed over the half-edges leaving a vertex, so a slit that revisits a
    corner accumulates there exactly as the chord-based version does.
    """
    import envs.polygon_utils as utils

    angles = {}
    for hidx in graph.half_edge_list():
        vertex = graph.source_vertex(hidx, tag=False)
        following = graph.target_vertex(hidx, tag=False)
        previous = graph.source_vertex(graph.previous_half_edge(hidx), tag=False)
        angles[vertex] = angles.get(vertex, 0.0) + utils.angle_between(
            edge_direction(graph, vertex, following),
            edge_direction(graph, vertex, previous))
    return angles
