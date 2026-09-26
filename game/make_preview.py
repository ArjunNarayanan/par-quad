"""Render docs/preview.png — the social-card image for the Pages site.

Uses the real game engine: each shape shown is an actual solved mesh, drawn in
the game's own palette. Run: venv/bin/python game/make_preview.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "game"))
from server import Game  # noqa: E402

PAPER = "#f3f5f7"
INK = "#17242e"
MUTED = "#5d6b77"
EDGE = "#46545f"
GOOD = "#2b8a4d"
WARN = "#d9962e"
# the light-theme mosaic palette from game/src/style.css, picked per face by
# the same face-id hash the game uses, so the card matches the live board
TILES = ["#f0a08c", "#f0c47a", "#a8d18a", "#7fcfc4",
         "#93bfe8", "#b3aae0", "#e7a6c4", "#ddd08a"]
TILE_ALPHA = 0.82


def tile_glass(face_id):
    """(colour, alpha) for a face — the tileClass() hash from game/src/ui.js."""
    h = ((face_id + 1) * 0x9E3779B1) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    return TILES[h % len(TILES)], TILE_ALPHA - (0.22 if h & 8 else 0)


def eid(g, a, b):
    for e in g.serialize()["edges"]:
        if {e["v1"], e["v2"]} == {a, b}:
            return e["id"]
    raise SystemExit(f"no edge {a}-{b}")


def solved_l():
    g = Game("L-shape")
    g.apply_op("insert_vertex", {"edge": eid(g, 0, 1)})
    g.apply_op("insert_vertex", {"edge": eid(g, 5, 0)})
    g.apply_op("insert_edge", {"a": 6, "b": 3})
    g.apply_op("insert_edge", {"a": 7, "b": 3})
    g.smooth(4)
    return g


def solved_star():
    g = Game("Star")
    g.apply_op("insert_edge", {"a": 3, "b": 7})
    g.apply_op("insert_vertex", {"edge": eid(g, 3, 7)})
    g.apply_op("insert_edge", {"a": 5, "b": 10})
    g.apply_op("insert_edge", {"a": 3, "b": 7})
    g.apply_op("insert_vertex", {"edge": eid(g, 3, 7)})
    g.apply_op("insert_edge", {"a": 11, "b": 9})
    g.apply_op("insert_edge", {"a": 11, "b": 1})
    g.smooth(5)
    return g


def solved_square_hole():
    g = Game("Square hole")
    for a, b in [(0, 1), (1, 7), (7, 8)]:
        g.apply_op("insert_vertex", {"edge": eid(g, a, b)})
    g.apply_op("insert_vertex", {"edge": eid(g, 7, 12)})
    g.apply_op("insert_vertex", {"edge": eid(g, 8, 9)})
    g.apply_op("insert_vertex", {"edge": eid(g, 8, 14)})
    g.apply_op("insert_vertex", {"edge": eid(g, 9, 0)})
    g.apply_op("insert_vertex", {"edge": eid(g, 16, 0)})
    for a, b in [(10, 3), (11, 6), (13, 6), (12, 5),
                 (15, 5), (14, 4), (16, 4), (17, 3)]:
        g.apply_op("insert_edge", {"a": a, "b": b})
    g.smooth(6)
    return g


def draw(ax, game, scale=1.0, dx=0.0, dy=0.0):
    s = game.serialize()
    pos = {v["id"]: (v["x"], v["y"]) for v in s["vertices"]}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    k = scale / span

    def T(p):
        return (dx + (p[0] - cx) * k, dy + (p[1] - cy) * k)

    for f in s["faces"]:
        pts = [T(pos[v]) for v in f["vertices"]]
        color, alpha = tile_glass(f["id"])
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=alpha,
                             edgecolor="none", zorder=1))
    for e in s["edges"]:
        (x1, y1), (x2, y2) = T(pos[e["v1"]]), T(pos[e["v2"]])
        ax.plot([x1, x2], [y1, y2], color=INK if e["boundary"] else EDGE,
                lw=3.0 if e["boundary"] else 1.9, zorder=2,
                solid_capstyle="round")
    for v in s["vertices"]:
        x, y = T(pos[v["id"]])
        ok = v["degree"] == v["desired"]
        ax.plot(x, y, "o", ms=9.5, color=GOOD if ok else WARN,
                markeredgecolor="white", markeredgewidth=1.6, zorder=3)


def main():
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 630)
    ax.axis("off")

    draw(ax, solved_l(), scale=210, dx=250, dy=250)
    draw(ax, solved_star(), scale=300, dx=610, dy=250)
    draw(ax, solved_square_hole(), scale=250, dx=985, dy=250)

    ax.text(60, 545, "MESH QUEST", color=INK, fontsize=54, fontweight="bold",
            family="monospace", va="center")
    ax.text(64, 487, "slice shapes into blocks", color=MUTED, fontsize=26,
            va="center")

    out = ROOT / "docs" / "preview.png"
    fig.savefig(out, facecolor=PAPER)
    print("wrote", out)


if __name__ == "__main__":
    main()
