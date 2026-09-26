"""Per-mesh distributions of minimum quality and of the excess over par, I(M) - par:
the agent against Gmsh, one mesh per domain, every domain of both evaluation sets.

    venv/bin/python utilities/paper_distributions.py \
        -agent out/paper/e2e-final -gmsh out/paper/gmsh-e2e -seed 0 \
        -out figures/fig-distributions.pdf

Reads the per-domain records only (the agent's at one rollout seed, and Gmsh's rows,
which were searched to that seed's element counts and store the agent's values beside
their own). A mesh that is not all-quadrilateral is its own bar, so every method is
counted on every domain and no subset is compared.

Form: small multiples, one row per method, one column per metric and set, the same
bins in every row and a shared y-scale per column. The agent is the accent; Gmsh rows
are neutral and identified by their row label, never by colour.
"""
import argparse
import json
import os

import numpy as np

IN_DIST = ["straight", "polycube", "straight-holes", "polycube-holes"]
LARGER = ["straight-transfer", "polycube-transfer", "straight-holes-transfer", "polycube-holes-transfer"]

# validated (dataviz validate_palette.js, light surface): contrast >= 3:1, normal-vision
# dE 16.8, CVD dE 14.7; the grey fails the chroma floor on purpose (it is a de-emphasis,
# not a category -- rows are identified by label)
ACCENT = "#2a78d6"
NEUTRAL = "#8a939c"
INK = "#1f2933"
MUTED = "#52606d"
RULE = "#c9ced4"

METHODS = [("agent", "Ours"), ("blossom", "Gmsh blossom,\nmatched count"),
           ("frontal-quad", "Gmsh frontal-quad,\nmatched count"),
           ("quasi-structured", "Gmsh quasi-structured,\n3$\\times$ elements")]

# bins: index 0 is "not all-quad"; the rest are labelled by their lower edge
Q_EDGES = [-np.inf, 0.2, 0.3, 0.4, 0.5, 0.6, np.inf]
Q_LABELS = ["not\nquad", "<.2", ".2", ".3", ".4", ".5", "≥.6"]
R_EDGES = [0.0, 0.5, 1.5, 4.5, 9.5, 19.5, np.inf]              # counts: 0, 1, 2-4, 5-9, 10-19, >=20
R_LABELS = ["0", "1", "2", "5", "10", "≥20"]             # all-quad meshes only; lower edges


def _records(agent_dir, gmsh_dir, suites, seed):
    """{method: [(all_quad, min quality, excess over par), ...]}, one entry per domain."""
    out = {m: [] for m, _ in METHODS}
    for suite in suites:
        agent = json.load(open(os.path.join(agent_dir, f"{suite}.{seed}.json")))["suites"][suite]["domains"]
        by_draw = {d["draw"]: d for d in agent}
        rows = json.load(open(os.path.join(gmsh_dir, f"{suite}.json")))["rows"]
        for d in agent:
            out["agent"].append((d["face"] == 0, d["quality"], d["irregular"] - d["par"]))
        for method, _ in METHODS[1:]:
            for r in rows[method]:
                assert r["draw"] in by_draw, (suite, method, r["draw"])
                ok = bool(r.get("all_quad")) and not r.get("failed")
                out[method].append((ok, r.get("quality") or 0.0,
                                    (r.get("vertex_score") or 0) - (r.get("par") or 0)))
    return out


def _histogram(values, edges, index):
    counts = np.zeros(len(edges), dtype=int)          # slot 0: not all-quad
    for ok, *metrics in values:
        if not ok:
            counts[0] += 1
            continue
        v = metrics[index]
        k = int(np.searchsorted(edges, v, side="right"))
        counts[min(max(k, 1), len(edges) - 1)] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-agent", required=True)
    parser.add_argument("-gmsh", required=True)
    parser.add_argument("-seed", default=0, type=int)
    parser.add_argument("-bar", default=0.3, type=float)
    parser.add_argument("-width", default=5.5, type=float)
    parser.add_argument("-out", required=True)
    parser.add_argument("-png", default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": 0.5, "axes.linewidth": 0.5,
    })
    import matplotlib.pyplot as plt

    sets = [("in-distribution", IN_DIST), ("larger boundaries", LARGER)]
    data = [(name, _records(args.agent, args.gmsh, suites, args.seed)) for name, suites in sets]
    columns = []
    for name, records in data:
        total = len(records["agent"])
        columns.append((f"{name} ({total})", "minimum quality $q$", records, 0, Q_EDGES, Q_LABELS, total))
        columns.append((f"{name} ({total})", "excess over par", records, 1, R_EDGES, R_LABELS, total))

    figure, axes = plt.subplots(len(METHODS), len(columns), figsize=(args.width, 3.55), sharex="col",
                                squeeze=False, gridspec_kw=dict(wspace=0.38, hspace=0.18))
    x = np.arange(len(Q_LABELS), dtype=float)
    x[1:] += 0.45                                        # a gap after the "not quad" category
    for c, (set_name, metric, records, index, edges, labels, total) in enumerate(columns):
        for r, (method, label) in enumerate(METHODS):
            axis = axes[r][c]
            counts = _histogram(records[method], edges, index)
            colour = ACCENT if method == "agent" else NEUTRAL
            if index == 0:
                # minimum quality: counts over every domain, "not quad" as its own category
                axis.bar(x[1:], counts[1:], width=0.8, color=colour, linewidth=0, zorder=2)
                axis.bar(x[:1], counts[:1], width=0.8, color="white", edgecolor=colour, hatch="////",
                         linewidth=0.6, zorder=2)
                axis.set_ylim(0, total * 1.02)
                axis.set_yticks([0, total // 2, total])
            else:
                # regularity is only defined on an all-quad mesh: the share of this method's
                # all-quad meshes per bin (completion is the quality panel's "not quad" bar)
                n_quad = int(counts[1:].sum())
                share = counts[1:] / max(n_quad, 1)
                xr = np.arange(len(share), dtype=float)
                axis.bar(xr, share, width=0.8, color=colour, linewidth=0, zorder=2)
                axis.set_ylim(0, 1.02)
                axis.set_yticks([0, 0.5, 1])
                axis.set_yticklabels(["0", "50%", "100%"])
            axis.tick_params(axis="y", labelsize=6, colors=MUTED, length=2, width=0.4)
            axis.tick_params(axis="x", length=0, labelsize=6, colors=MUTED, pad=2)
            axis.grid(axis="y", color=RULE, linewidth=0.4, zorder=0)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color(RULE)
            # the one number per panel the tables quote, in ink (never the series colour)
            if index == 0:
                usable = sum(1 for ok, q, _ in records[method] if ok and q >= args.bar)
                note = f"usable {usable}"
                boundary = (x[2] + x[3]) / 2        # between the ".2" and ".3" bins
                axis.axvline(boundary, color=INK, linewidth=0.5, linestyle=(0, (2, 2)), zorder=3)
            else:
                note = f"{int(counts[1])}/{n_quad} at 0"
            if index == 0 and counts[0]:
                axis.text(x[0], counts[0] + total * 0.03, str(counts[0]), ha="center", va="bottom",
                          fontsize=6, color=INK)
            # the regularity note sits mid-panel: Gmsh's mass is in the right-most bins
            axis.text(0.98 if index == 0 else 0.62, 0.9, note, transform=axis.transAxes,
                      ha="right", va="top", fontsize=6, color=INK)
            if c == 0:
                axis.set_ylabel(label, fontsize=6.8, color=INK, rotation=0, ha="right", va="center",
                                labelpad=4, fontweight="bold" if method == "agent" else "normal")
            if r == 0:
                axis.set_title(metric, fontsize=7, color=INK, pad=3)
            if r == len(METHODS) - 1:
                axis.set_xticks(x if index == 0 else np.arange(len(labels), dtype=float))
                axis.set_xticklabels(labels)
    # set headers over each pair of columns
    for c in range(0, len(columns), 2):
        left = axes[0][c].get_position()
        right = axes[0][c + 1].get_position()
        figure.text((left.x0 + right.x1) / 2, left.y1 + 0.055, columns[c][0], ha="center", va="bottom",
                    fontsize=7.5, color=INK)
    figure.text(0.5, 0.005, f"quality: domains per bin, dashed line the usability bar $q = {args.bar:g}$; "
                "excess over par: share of each method's all-quad meshes; bins labelled by lower edge",
                ha="center", va="bottom", fontsize=6, color=MUTED)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    figure.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    if args.png:
        figure.savefig(args.png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
