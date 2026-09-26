"""Simultaneous untangling and smoothing (Escobar et al.), after geo2d.smooth.

Ported back from MeshRL's `game/js/untangle.js`, which was itself ported from
geo2d and agreed with it to three decimals. One free node at a time minimises

    sum_k (|a_k|^2 + |b_k|^2) * sin(theta_n) / (2 h(J_k)),   h(J) = (J + sqrt(J^2 + d^2)) / 2

over the corners of its elements, with Newton steps on finite differences and
backtracking. `tangents` {(node, neighbour): unit vector} replaces the direction
of a curved side by its tangent at the corner, keeping the chord length.
"""

import numpy as np

from .mesh import Mesh, corner_quality


class _NodeCorners:
    def __init__(self, mesh, tangents=None):
        tri, sn = mesh.corners()
        self.tri, self.sn = tri, sn
        dirs = np.full((len(tri), 2, 2), np.nan)
        if tangents:
            for k, (p, v, q) in enumerate(tri):
                t = tangents.get((int(v), int(q)))
                if t is not None:
                    dirs[k, 0] = t
                t = tangents.get((int(v), int(p)))
                if t is not None:
                    dirs[k, 1] = t
        self.dirs = dirs
        self.by_node = [[] for _ in range(len(mesh.nodes))]
        elements = mesh.elements
        start = 0
        for e in elements:
            n = len(e)
            ks = list(range(start, start + n))
            for v in set(e):
                self.by_node[v].extend(ks)
            start += n
        self.by_node = [np.asarray(ks, int) for ks in self.by_node]

    def of_node(self, v):
        ks = self.by_node[int(v)]
        return self.tri[ks], self.sn[ks], self.dirs[ks]


def _min_quality(P, sn, dirs=None):
    if len(P) == 0:
        return 1.0
    return float(corner_quality(P, sn, dirs).min())


def _terms(P, dirs):
    a = P[..., 2, :] - P[..., 1, :]
    b = P[..., 0, :] - P[..., 1, :]
    if dirs is not None:
        la = np.linalg.norm(a, axis=-1, keepdims=True)
        lb = np.linalg.norm(b, axis=-1, keepdims=True)
        da, db = dirs[..., 0, :], dirs[..., 1, :]
        a = np.where(np.isfinite(da), la * np.nan_to_num(da), a)
        b = np.where(np.isfinite(db), lb * np.nan_to_num(db), b)
    J = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    L2 = (a * a).sum(-1) + (b * b).sum(-1)
    return J, L2


def _optimize_node(P, mask, sn, x0, newton_iters=6, dirs=None):
    """New position for the node that sits at `mask` inside the corner stack P."""
    P = np.asarray(P, float)
    mask = np.asarray(mask, bool)
    x0 = np.asarray(x0, float)
    if len(P) == 0:
        return x0

    def batch(xs):
        xs = np.atleast_2d(xs)                       # (m, 2)
        Q = np.broadcast_to(P, (len(xs),) + P.shape).copy()
        Q[:, mask] = xs[:, None, :].repeat(mask.sum(), axis=1)
        J, L2 = _terms(Q, dirs)                      # (m, k)
        jmin = J.min(axis=1, keepdims=True)
        jmean = np.abs(J).mean(axis=1, keepdims=True)
        delta2 = np.where(jmin > 0, 0.0, 0.1 * jmean * (0.1 * jmean - jmin))
        h = 0.5 * (J + np.sqrt(J * J + delta2))
        with np.errstate(divide="ignore", invalid="ignore"):
            f = (L2 * sn / (2 * h)).sum(axis=1)
        bad = ~(h > 0).all(axis=1)
        f[bad] = np.inf
        return f

    _, L2 = _terms(np.where(mask[..., None], x0, P), dirs)
    scale = np.sqrt(max(float(L2.mean()), 1e-300) / 2)
    hstep = 1e-6 * scale
    x = x0.copy()
    f = batch(x)[0]
    E = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]], float)
    for _ in range(newton_iters):
        fx = batch(x + hstep * E)
        if not np.isfinite(fx).all():
            hstep *= 0.1
            continue
        fxp, fxm, fyp, fym, fpp, fmm = fx
        gx, gy = (fxp - fxm) / (2 * hstep), (fyp - fym) / (2 * hstep)
        hxx = (fxp - 2 * f + fxm) / hstep ** 2
        hyy = (fyp - 2 * f + fym) / hstep ** 2
        hxy = (fpp - fxp - fyp + 2 * f - fxm - fym + fmm) / (2 * hstep ** 2)
        det = hxx * hyy - hxy * hxy
        if det > 0 and hxx > 0:
            d = -np.array([hyy * gx - hxy * gy, -hxy * gx + hxx * gy]) / det
        else:
            gn = np.hypot(gx, gy)
            if gn == 0 or not np.isfinite(gn):
                break
            d = -np.array([gx, gy]) / gn * scale * 0.2
        alphas = 0.5 ** np.arange(12)
        cands = x + alphas[:, None] * d
        fc = batch(cands)
        better = np.nonzero(fc < f)[0]
        if len(better) == 0:
            break
        k = better[0]
        step = np.linalg.norm(alphas[k] * d)
        x, f = cands[k], fc[k]
        if step < 1e-12 * scale:
            break
    return x


def _mask(fixed, n):
    """`fixed` as a boolean mask; geo2d accepts a mask or an index list."""
    mask = np.zeros(n, bool)
    if fixed is None:
        return mask
    fixed = np.asarray(fixed)
    if fixed.dtype == bool and len(fixed) == n:
        return fixed.copy()
    mask[fixed.astype(int)] = True
    return mask


def optimize(mesh, iters=8, fixed=None, tangents=None, newton_iters=6):
    out = mesh.copy()
    fixed = _mask(fixed, len(out.nodes))
    corners = _NodeCorners(out, tangents)
    X = out.nodes
    free = [v for v in range(len(X)) if not fixed[v] and len(corners.by_node[v])]
    for _ in range(iters):
        for v in free:
            tri, sn, dirs = corners.of_node(v)
            X[v] = _optimize_node(X[tri], tri == v, sn, X[v].copy(), newton_iters, dirs)
    return out


def smart_laplacian(mesh, iters=8, fixed=None, tangents=None):
    out = mesh.copy()
    fixed = _mask(fixed, len(out.nodes))
    corners = _NodeCorners(out, tangents)
    offsets, nbr = out.node_neighbors()
    X = out.nodes
    for _ in range(iters):
        for v in range(len(X)):
            if fixed[v] or offsets[v + 1] == offsets[v]:
                continue
            tri, sn, dirs = corners.of_node(v)
            before = _min_quality(X[tri], sn, dirs)
            target = X[nbr[offsets[v]:offsets[v + 1]]].mean(axis=0)
            P = X[tri].copy()
            P[tri == v] = target
            if _min_quality(P, sn, dirs) >= before - 1e-12:
                X[v] = target
    return out
