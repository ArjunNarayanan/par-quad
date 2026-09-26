"""Loops of line and circular-arc edges, with geo2d's interface.

A stand-in for the parts of `geogen/geo2d` MeshRL touches, written for machines
where the private geogen repository is not available. Edge i of a loop runs from
control point i to control point i + 1. `interior_angles()` is the TANGENT angle
of the material on the left of the traversal, so a smooth arc joint reads 180.
"""

import numpy as np


def _rot(v, sign):
    return np.array([-sign * v[1], sign * v[0]], float)


class Line:
    kind = "line"

    def __init__(self, p0, p1):
        self.p0 = np.asarray(p0, float)
        self.p1 = np.asarray(p1, float)

    def point(self, t):
        return self.p0 + t * (self.p1 - self.p0)

    def tangent(self, t):
        d = self.p1 - self.p0
        return d / np.linalg.norm(d)

    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))


class Arc:
    kind = "arc"

    def __init__(self, p0, p1, center, ccw):
        self.p0 = np.asarray(p0, float)
        self.p1 = np.asarray(p1, float)
        self.center = np.asarray(center, float)
        self.radius = float(np.linalg.norm(self.p0 - self.center))
        self.ccw = bool(ccw)
        a0 = np.arctan2(*(self.p0 - self.center)[::-1])
        a1 = np.arctan2(*(self.p1 - self.center)[::-1])
        sweep = (a1 - a0) % (2 * np.pi) if self.ccw else -((a0 - a1) % (2 * np.pi))
        self._a0, self._sweep = a0, sweep

    def point(self, t):
        a = self._a0 + t * self._sweep
        return self.center + self.radius * np.array([np.cos(a), np.sin(a)])

    def tangent(self, t):
        radial = (self.point(t) - self.center) / self.radius
        return _rot(radial, 1 if self.ccw else -1)

    def length(self):
        return abs(self._sweep) * self.radius


class Loop:
    def __init__(self, points, edges=None):
        self.points = np.asarray(points, float)
        n = len(self.points)
        if n < 3:
            raise ValueError("a loop needs at least 3 points")
        if edges is None:
            edges = [Line(self.points[i], self.points[(i + 1) % n]) for i in range(n)]
        self.edges = list(edges)

    @property
    def n(self):
        return len(self.points)

    def edge(self, i):
        return self.edges[i]

    def kinds(self):
        return [e.kind for e in self.edges]

    def is_arc(self):
        return [e.kind == "arc" for e in self.edges]

    def sample(self, per_arc=16):
        out = []
        for e in self.edges:
            if e.kind == "arc":
                out.extend(e.point(t) for t in np.linspace(0, 1, per_arc, endpoint=False))
            else:
                out.append(e.p0)
        return np.array(out)

    def signed_area(self):
        p = self.sample()
        x, y = p[:, 0], p[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def is_ccw(self):
        return self.signed_area() > 0

    def interior_angles(self):
        n = self.n
        out = np.zeros(n)
        for i in range(n):
            t_in = self.edges[i - 1].tangent(1.0)
            t_out = self.edges[i].tangent(0.0)
            turn = np.degrees(np.arctan2(t_in[0] * t_out[1] - t_in[1] * t_out[0],
                                         float(t_in @ t_out)))
            out[i] = 180.0 - turn
        return out


class Geometry:
    def __init__(self, loops):
        self.loops = list(loops)

    @property
    def outer(self):
        return self.loops[0]

    @property
    def holes(self):
        return self.loops[1:]
