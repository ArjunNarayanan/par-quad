"""geo2d's Mesh: nodes plus polygonal elements, and the shape quality."""

import numpy as np


def corner_quality(P, sn, dirs=None):
    """2J / (|a|^2 + |b|^2) / sin(ideal angle) per corner; P is (k, 3, 2) = prev, node, next.

    `dirs` (k, 2, 2) optionally replaces the DIRECTION of side a (node->next) and
    side b (node->prev) by a unit tangent, keeping the chord length; NaN rows
    leave the chord alone.
    """
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
    return 2.0 * J / np.maximum(L2, 1e-300) / sn


class Mesh:
    def __init__(self, nodes, elements):
        self.nodes = np.asarray(nodes, float).copy()
        self._elements = [[int(v) for v in e] for e in elements]

    # geo2d exposes both `mesh.elements` (a list) and, in one caller, `mesh.elements()`
    @property
    def elements(self):
        return _Elements(self._elements)

    def copy(self):
        return Mesh(self.nodes.copy(), [list(e) for e in self._elements])

    def boundary_edges(self):
        uses = {}
        for e in self._elements:
            n = len(e)
            for i in range(n):
                a, b = e[i], e[(i + 1) % n]
                key = (min(a, b), max(a, b))
                uses.setdefault(key, []).append((a, b))
        return [v[0] for v in uses.values() if len(v) == 1]

    def is_boundary(self):
        mask = np.zeros(len(self.nodes), bool)
        for a, b in self.boundary_edges():
            mask[a] = mask[b] = True
        return mask

    def node_neighbors(self):
        nbr = [set() for _ in range(len(self.nodes))]
        for e in self._elements:
            n = len(e)
            for i in range(n):
                a, b = e[i], e[(i + 1) % n]
                nbr[a].add(b)
                nbr[b].add(a)
        offsets = np.zeros(len(self.nodes) + 1, int)
        flat = []
        for i, s in enumerate(nbr):
            flat.extend(sorted(s))
            offsets[i + 1] = len(flat)
        return offsets, np.asarray(flat, int)

    def corners(self):
        tri, sn = [], []
        for e in self._elements:
            n = len(e)
            s = np.sin(np.pi * (n - 2) / n)
            for j in range(n):
                tri.append((e[j - 1], e[j], e[(j + 1) % n]))
                sn.append(s)
        return np.asarray(tri, int), np.asarray(sn, float)

    def quality(self):
        """Per element, the minimum corner shape quality."""
        out = []
        for e in self._elements:
            n = len(e)
            s = np.sin(np.pi * (n - 2) / n)
            tri = np.array([(e[j - 1], e[j], e[(j + 1) % n]) for j in range(n)])
            out.append(float(corner_quality(self.nodes[tri], s).min()))
        return np.asarray(out)


class _Elements(list):
    def __call__(self):
        return self
