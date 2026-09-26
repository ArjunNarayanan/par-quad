"""Random rectilinear domains with circular features: a stand-in `rounded` preset.

Modelled on the ten geo2d `rounded` draws frozen in MeshRL's
`src/curved_domains.json`: a grid-aligned rectilinear outline (a union and
difference of rectangles), then features that are always quarter-circle arcs --
convex and concave corner fillets, semicircular scoops cut into an edge,
semicircular rounded tab ends -- and optional holes, circular (four quarter
arcs, clockwise) or rectangular. `ratio` is the outline's extent in grid units,
`n_ops` the number of rectangle operations, `n_mods` the number of curved
features, `n_holes` the number of holes. Ranges are `(lo, hi)` inclusive.

This is NOT geo2d: the draws differ seed for seed, so numbers measured on it are
comparable with each other, not with results measured on geogen's generator.
"""

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from .geometry import Arc, Geometry, Line, Loop


def _pick(rng, value):
    if isinstance(value, (tuple, list)):
        return int(rng.integers(int(value[0]), int(value[1]) + 1))
    return int(value)


def _rectilinear(rng, ratio, n_ops):
    """A simple grid polygon of extent ~ratio, from rectangle unions/differences."""
    W = int(rng.integers(max(3, ratio // 2), ratio + 1))
    H = int(rng.integers(max(3, ratio // 3), max(4, int(0.8 * ratio)) + 1))
    poly = box(0, 0, W, H)
    done, tries = 0, 0
    while done < n_ops and tries < 60 * (n_ops + 1):
        tries += 1
        w = int(rng.integers(1, max(2, W // 2) + 1))
        h = int(rng.integers(1, max(2, H // 2) + 1))
        minx, miny, maxx, maxy = (int(round(c)) for c in poly.bounds)
        x = int(rng.integers(minx - w + 1, maxx))
        y = int(rng.integers(miny - h + 1, maxy))
        r = box(x, y, x + w, y + h)
        new = poly.union(r) if rng.random() < 0.55 else poly.difference(r)
        new = new.buffer(0)
        if new.geom_type != "Polygon" or len(new.interiors) or new.area < 0.35 * W * H:
            continue
        # no pinch points: eroding by a hair must keep it one piece
        if new.buffer(-0.49).geom_type != "Polygon":
            continue
        b = new.bounds
        if max(b[2] - b[0], b[3] - b[1]) > ratio:
            continue
        poly, done = new, done + 1
    ring = np.asarray(poly.exterior.coords)[:-1]
    if Polygon(ring).exterior.is_ccw is False:
        ring = ring[::-1]
    return _drop_collinear(ring), poly


def _drop_collinear(ring):
    keep = []
    n = len(ring)
    for i in range(n):
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-9:
            keep.append(b)
    return np.asarray(keep, float)


class _Outline:
    """A CCW loop being edited: points plus per-edge arc data (None = line)."""

    def __init__(self, ring):
        self.pts = [np.asarray(p, float) for p in ring]
        self.arc = [None] * len(ring)        # arc on edge i -> (center, ccw)
        self.locked = [False] * len(ring)    # vertex used by a feature
        self.elocked = [False] * len(ring)   # edge used by a feature

    def n(self):
        return len(self.pts)

    def seg(self, i):
        return self.pts[i], self.pts[(i + 1) % self.n()]

    def replace_vertex(self, i, new_pts, new_arcs):
        """Vertex i -> new_pts; edges between them carry new_arcs."""
        n = self.n()
        pts = self.pts[:i] + new_pts + self.pts[i + 1:]
        arc = self.arc[:i] + new_arcs + [self.arc[i]] + self.arc[i + 1:]
        lk = self.locked[:i] + [True] * len(new_pts) + self.locked[i + 1:]
        elk = self.elocked[:i] + [True] * len(new_arcs) + [self.elocked[i]] + self.elocked[i + 1:]
        self.pts, self.arc, self.locked, self.elocked = pts, arc, lk, elk
        del n

    def replace_edge(self, i, inner_pts, inner_arcs):
        """Edge i (pts[i] -> pts[i+1]) gains inner points; inner_arcs has len(inner)+1."""
        pts = self.pts[:i + 1] + inner_pts + self.pts[i + 1:]
        arc = self.arc[:i] + inner_arcs + self.arc[i + 1:]
        lk = self.locked[:i + 1] + [True] * len(inner_pts) + self.locked[i + 1:]
        elk = self.elocked[:i] + [True] * len(inner_arcs) + self.elocked[i + 1:]
        self.pts, self.arc, self.locked, self.elocked = pts, arc, lk, elk

    def to_loop(self):
        n = self.n()
        edges = []
        for i in range(n):
            a, b = self.seg(i)
            if self.arc[i] is None:
                edges.append(Line(a, b))
            else:
                c, ccw = self.arc[i]
                edges.append(Arc(a, b, c, ccw))
        return Loop(np.asarray(self.pts), edges)


def _turn(a, b, c):
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


def _other_edges(outline, skip):
    segs = []
    for j in range(outline.n()):
        if j in skip:
            continue
        a, b = outline.seg(j)
        segs.append(LineString([a, b]))
    return unary_union(segs) if segs else None


def _fillet(rng, o):
    cands = []
    n = o.n()
    for i in range(n):
        if o.locked[i] or o.elocked[i - 1] or o.elocked[i]:
            continue
        a, b, c = o.pts[i - 1], o.pts[i], o.pts[(i + 1) % n]
        if o.arc[i - 1] is not None or o.arc[i] is not None:
            continue
        cands.append(i)
    if not cands:
        return False
    i = cands[int(rng.integers(len(cands)))]
    a, b, c = o.pts[i - 1], o.pts[i], o.pts[(i + 1) % n]
    l1, l2 = np.linalg.norm(b - a), np.linalg.norm(c - b)
    d_in, d_out = (b - a) / l1, (c - b) / l2
    full = min(l1, l2)
    # half-grid radii; the whole side only when the neighbouring corner is free
    choices = [r for r in np.arange(0.5, full + 1e-9, 0.5) if r <= 0.5 * full + 1e-9]
    if (not o.locked[i - 1] and not o.locked[(i + 1) % n]) and rng.random() < 0.3:
        choices = choices + [full]
    choices = [r for r in choices if r < full - 1e-9 or not np.isclose(l1, l2)] or choices
    if not choices:
        return False
    r = float(choices[int(rng.integers(len(choices)))])
    if r >= l1 - 1e-9 or r >= l2 - 1e-9:
        return False                    # never swallow a whole side
    p0, p1 = b - r * d_in, b + r * d_out
    centre = p0 + r * d_out
    ccw = _turn(a, b, c) > 0
    o.replace_vertex(i, [p0, p1], [(centre, ccw)])
    return True


def _scoop(rng, o, poly, bump=False):
    n = o.n()
    cands = []
    for i in range(n):
        if o.elocked[i] or o.arc[i] is not None:
            continue
        a, b = o.seg(i)
        if np.linalg.norm(b - a) >= (2.0 if bump else 2.0):
            cands.append(i)
    if bump:
        # a rounded tab end: an edge between two convex corners, replaced whole
        cands = [i for i in range(n)
                 if o.arc[i] is None and not o.elocked[i]
                 and not o.locked[i] and not o.locked[(i + 1) % n]
                 and o.arc[i - 1] is None and o.arc[(i + 1) % n] is None
                 and _turn(o.pts[i - 1], o.pts[i], o.pts[(i + 1) % n]) > 0
                 and _turn(o.pts[i], o.pts[(i + 1) % n], o.pts[(i + 2) % n]) > 0]
    if not cands:
        return False
    i = cands[int(rng.integers(len(cands)))]
    a, b = o.seg(i)
    L = np.linalg.norm(b - a)
    d = (b - a) / L
    n_in = np.array([-d[1], d[0]])
    if bump:
        r = L / 2
        c = (a + b) / 2
        apex = c - r * n_in
        # the tab's side edges arrive perpendicular, so both old corners become
        # smooth joints; clearance is checked on the whole outline afterwards
        o.replace_edge(i, [apex], [(c, True), (c, True)])
        return True
    rmax = min(L / 2 - 0.5, 0.45 * max(1.0, L))
    radii = [r for r in np.arange(0.5, rmax + 1e-9, 0.5)]
    if not radii:
        return False
    r = float(radii[int(rng.integers(len(radii)))])
    s = float(rng.uniform(r + 0.25, L - r - 0.25)) if L - 2 * r > 0.5 else L / 2
    s = round(s * 2) / 2
    s = min(max(s, r + 0.25), L - r - 0.25)
    c = a + s * d
    apex = c + r * n_in
    others = _other_edges(o, {i})
    if others is not None and Point(c).distance(others) < 1.3 * r:
        return False
    o.replace_edge(i, [c - r * d, apex, c + r * d], [None, (c, False), (c, False), None])
    return True


def _holes(rng, outer_loop, count, ratio, unit=1.0):
    region = Polygon(outer_loop.sample(24))
    loops, taken = [], []
    for _ in range(count):
        for _try in range(80):
            minx, miny, maxx, maxy = region.bounds
            if rng.random() < 0.65:
                r = unit * float(rng.choice(np.arange(0.5, max(0.6, ratio / unit / 6) + 1e-9, 0.5)))
                c = np.array([rng.integers(int(minx), int(maxx) + 1),
                              rng.integers(int(miny), int(maxy) + 1)], float)
                c += unit * rng.choice([0.0, 0.5], size=2)
                shape = Point(c).buffer(r, 64)
                if not region.buffer(-0.6 * unit).contains(shape):
                    continue
                if any(shape.buffer(0.6 * unit).intersects(t) for t in taken):
                    continue
                pts = [c + r * np.array([np.cos(t), np.sin(t)])
                       for t in (0.0, -np.pi / 2, -np.pi, -1.5 * np.pi)]
                edges = [Arc(pts[k], pts[(k + 1) % 4], c, False) for k in range(4)]
                loops.append(Loop(np.asarray(pts), edges))
                taken.append(shape)
                break
            m = max(2, int(ratio / unit) // 4)
            w, h = unit * int(rng.integers(1, m + 1)), unit * int(rng.integers(1, m + 1))
            x = unit * int(rng.integers(int(minx / unit), max(int(minx / unit) + 1, int(maxx / unit))))
            y = unit * int(rng.integers(int(miny / unit), max(int(miny / unit) + 1, int(maxy / unit))))
            shape = box(x, y, x + w, y + h)
            if not region.buffer(-0.6 * unit).contains(shape) or any(shape.buffer(0.6 * unit).intersects(t) for t in taken):
                continue
            pts = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]], float)   # clockwise
            loops.append(Loop(pts))
            taken.append(shape)
            break
    return loops


def generate_rounded(seed, n_holes=0, ratio=8, n_ops=(1, 3), n_mods=(1, 4), **_):
    """Grid features stay at ratio-8 scale: a larger `ratio` means a larger outline
    built from more operations, not finer features (which would only raise the
    boundary edge-length ratio the transfer suites cap at 10)."""
    rng = np.random.default_rng(seed)
    unit = max(1.0, float(ratio) / 8.0)
    for _attempt in range(50):
        ops = _pick(rng, n_ops)
        mods = _pick(rng, n_mods)
        ring, poly = _rectilinear(rng, int(round(ratio / unit)), ops)
        o = _Outline(ring)
        made = 0
        for _k in range(6 * max(mods, 1)):
            if made >= mods:
                break
            kind = rng.random()
            ok = (_fillet(rng, o) if kind < 0.55 else
                  _scoop(rng, o, poly) if kind < 0.8 else
                  _scoop(rng, o, poly, bump=True))
            made += int(bool(ok))
        outer = o.to_loop()
        sampled = Polygon(outer.sample(24))
        if not sampled.is_valid or sampled.area <= 0:
            continue
        # no near-pinches: a quarter-cell erosion must leave one piece
        if sampled.buffer(-0.24).geom_type != "Polygon":
            continue
        if unit != 1.0:
            o.pts = [p * unit for p in o.pts]
            o.arc = [None if a is None else (a[0] * unit, a[1]) for a in o.arc]
            outer = o.to_loop()
        count = _pick(rng, n_holes)
        holes = _holes(rng, outer, count, int(ratio), unit) if count else []
        return Geometry([outer] + holes)
    raise ValueError("no valid rounded outline")


def generate_rectilinear(seed, n_holes=0, ratio=8, n_ops=(1, 3), **_):
    rng = np.random.default_rng(seed)
    ring, _ = _rectilinear(rng, int(ratio), _pick(rng, n_ops))
    outer = Loop(ring)
    holes = _holes(rng, outer, _pick(rng, n_holes), int(ratio))
    holes = [h for h in holes if not any(h.is_arc())]
    return Geometry([outer] + holes)


PRESETS = {"rounded": generate_rounded, "polycube": generate_rectilinear,
           "straight": generate_rectilinear}


def generate(seed, preset="rounded", **options):
    return PRESETS[preset](int(seed), **options)


def generate_many(seed, count, **options):
    return [generate(seed + k, **options) for k in range(count)]
