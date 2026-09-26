"""The large-domain transfer set: does the policy hold above its training range?

Training and the golden benchmark both cap at 24 corners (`MIN_CORNERS = 8`,
`-max_corners 24`), so this set is the generalisation axis, and it is bucketed by size because the
question is not "does it still work" but "where does it start to degrade".

`max_corners` is NOT the knob. It is a rejection filter on the draw, so raising
it past 24 changes nothing: the `straight` preset's own `ratio`/`n_ops`/`n_mods`
are what set the domain's size. Measured over 120 seeds, `ratio=24` with
`n_ops=(6,12)` and `n_mods=(4,8)` gives a corner-count median of 40 and puts 83
of 120 draws inside 25-50.

    venv/bin/python utilities/transfer_set.py -gallery out/transfer/domains.png
"""

import argparse
import os
import sys

sys.path.append(os.getcwd())

# One generator, three windows on it. Using one generator for all three buckets
# is deliberate: a difference between buckets is then SIZE and not a change of
# distribution, which is the whole point of bucketing.
GENERATOR = dict(preset="straight", n_holes=0, ratio=24, n_ops=(6, 12), n_mods=(4, 8))
HOLED = dict(preset="straight", n_holes=(1, 2), ratio=24, n_ops=(6, 12), n_mods=(4, 8))

# The ratio cap is the whole reason this set can be READ. Drawn without it the
# large domains carry boundary edge-length ratios of 11.5x median and 20x max
# against the golden set's 5.8x and 10x -- so a drop on them would confound
# SIZE with geometric extremity, and a forced
# ratio near 20x is where a par-0 mesh can stop being embeddable at all (Hook
# is degenerate at 20x). Capping at 10x holds extremity at the benchmark's own
# worst case and leaves corner count as the only thing that moves. There are
# 51/98/101 qualifying draws per bucket, so 16 each is comfortable.
MAX_RATIO = 10.0

BUCKETS = {
    "transfer-25-32": (GENERATOR, 25, 32),
    "transfer-33-40": (GENERATOR, 33, 40),
    "transfer-41-50": (GENERATOR, 41, 50),
    "transfer-holes": (HOLED, 25, 50),
}


def edge_length_ratio(geometry):
    """Longest over shortest boundary edge, on the Tiler the agent is handed."""
    import numpy as np
    import src.geo2d_bridge as bridge
    graph, _ = (bridge.curved_geometry_to_tiler(geometry) if len(geometry.loops) > 1
                else bridge.geometry_to_tiler(geometry))
    lengths = []
    for half_edge in graph.boundary_half_edge_list():
        p = np.asarray(graph.vertex_coordinate(graph.source_vertex(half_edge)), float)
        q = np.asarray(graph.vertex_coordinate(graph.target_vertex(half_edge)), float)
        lengths.append(float(np.linalg.norm(p - q)))
    lengths = [x for x in lengths if x > 1e-12]
    return max(lengths) / min(lengths)


def transfer_geometries(bucket, seed=7, n=16, max_ratio=MAX_RATIO):
    """`n` geometries in a bucket's corner window, drawn like `score_suites`.

    `max_ratio=None` drops the extremity cap, which gives the STRESS version of
    the set -- useful, but not a size experiment.
    """
    from src.geo2d_bridge import import_geo2d, geometry_corner_count
    geo2d = import_geo2d()
    kwargs, lo, hi = BUCKETS[bucket]
    out, draw = [], seed
    while len(out) < n and draw < seed + 2000 * n:
        try:
            geometry = geo2d.generate(draw, **kwargs)
            if lo <= geometry_corner_count(geometry) <= hi:
                if max_ratio is None or edge_length_ratio(geometry) <= max_ratio:
                    out.append((draw, geometry))
        except Exception:
            pass
        draw += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", default=7, type=int)
    parser.add_argument("-n", default=16, type=int)
    parser.add_argument("-gallery", default="out/transfer/domains.png")
    parser.add_argument("-max_ratio", default=MAX_RATIO, type=float,
                        help="0 disables the extremity cap (the stress set)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from src.geo2d_bridge import geometry_corner_count

    names = list(BUCKETS)
    cols = min(args.n, 8)
    rows = sum(int(np.ceil(args.n / cols)) for _ in names)
    figure, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.25 * rows))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")

    row = 0
    for name in names:
        drawn = transfer_geometries(name, args.seed, args.n,
                                    args.max_ratio or None)
        print(f"{name:16s} {len(drawn):2d} domains", end="")
        corners = []
        for k, (draw, geometry) in enumerate(drawn):
            axis = axes[row + k // cols][k % cols]
            n_corners = geometry_corner_count(geometry)
            corners.append(n_corners)
            for index, loop in enumerate(geometry.loops):
                pts = np.asarray(loop.points, dtype=float)
                closed = np.vstack([pts, pts[:1]])
                axis.fill(closed[:, 0], closed[:, 1],
                          facecolor="#e9eef4" if index == 0 else "#ffffff",
                          edgecolor="none", zorder=1 + index)
                axis.plot(closed[:, 0], closed[:, 1], color="#17242e",
                          linewidth=1.3, zorder=3)
            axis.set_aspect("equal")
            axis.axis("off")
            axis.set_title(f"seed {draw} · {n_corners} corners", fontsize=6.5)
        ratios = [edge_length_ratio(g) for _, g in drawn]
        print(f"   corners {min(corners)}-{max(corners)} median {int(np.median(corners))}"
              f"   edge ratio median {np.median(ratios):.1f}x max {max(ratios):.1f}x")
        axes[row][0].text(-0.15, 0.5, name, rotation=90, fontsize=8,
                          transform=axes[row][0].transAxes,
                          va="center", ha="center")
        row += int(np.ceil(args.n / cols))

    os.makedirs(os.path.dirname(args.gallery) or ".", exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.gallery, dpi=170, bbox_inches="tight")
    print(f"\nwrote {args.gallery}")


if __name__ == "__main__":
    main()
