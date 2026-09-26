"""Head-to-head win rate, agent against each Gmsh variant, domain by domain.

    venv/bin/python utilities/head_to_head.py -tag mit-2M [-gmsh out/paper/gmsh] [-suites ...]

Arjun's rule (2026-09-23): on a domain, a method WINS if it produces an all-quad mesh and the
other does not; if both do, the one with the better minimum shape quality (after each side's
own smoothing) wins; if neither does, the domain is a tie. Counting wins avoids the bias of
comparing medians over different completed subsets (Gmsh's median is over the 15-27 of 64 it
completes; the agent's over 61). Agent meshes are the best-of-5 selections of the paper
protocol, one pass per rollout seed, so the counts are averaged over passes; Gmsh rows come
from `gmsh_variants_suite.py`, paired by `draw`.
"""
import argparse
import glob
import json
import os
import statistics
import sys

sys.path.append(os.getcwd())

IN_DIST = ["straight", "polycube", "straight-holes", "polycube-holes"]
LARGE = ["straight-transfer", "polycube-transfer", "straight-holes-transfer", "polycube-holes-transfer"]
VARIANTS = ["blossom", "frontal-quad", "packing", "simple-recombine", "quasi-structured", "subdivide", "blossom-full"]


def _excess_over_par(record):
    """Excess over par of one mesh, I(M) - par: the agent's records carry `irregular`
    (par + excess), Gmsh's rows `vertex_score`; both are sums of |deg - want|. Per domain
    both sides share par, so this compares irregularity without dividing by a vertex count
    that a method can raise by adding elements."""
    score = record.get("irregular", record.get("vertex_score"))
    return float(score) - float(record.get("par") or 0)


def outcomes(agent, gmsh, criterion="quality"):
    """(wins, losses, ties) from the agent's side over the paired domains.

    Completion decides first under either criterion. When both meshes are all-quad,
    `quality` compares minimum shape quality (higher wins) and `regularity` compares
    excess over par (lower wins); equal values are a tie."""
    w = l = t = 0
    for draw, a in agent.items():
        g = gmsh.get(draw)
        if g is None:
            continue
        a_ok = a["face"] == 0
        g_ok = (not g["failed"]) and g["all_quad"]
        if a_ok and not g_ok:
            w += 1
        elif g_ok and not a_ok:
            l += 1
        elif a_ok and g_ok:
            if criterion == "regularity":
                a_v, g_v = -_excess_over_par(a), -_excess_over_par(g)
            else:
                a_v, g_v = a["quality"], g["quality"]
            if abs(a_v - g_v) < 1e-12:
                t += 1
            elif a_v > g_v:
                w += 1
            else:
                l += 1
        else:
            t += 1
    return w, l, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-tag", required=True); p.add_argument("-gmsh", default="out/paper/gmsh")
    p.add_argument("-suites", default=",".join(IN_DIST + LARGE))
    p.add_argument("-variants", default=",".join(VARIANTS))
    p.add_argument("-criterion", default="quality", choices=["quality", "regularity"],
                   help="when both are all-quad: better minimum quality, or lower excess over par")
    p.add_argument("-agent_dir", default=None, help="directory of agent JSONs (default out/paper/<tag>); "
                   "files <suite>.<seed>.json or any name containing <suite>.json")
    p.add_argument("-gmsh_curved", default=None, help="curved-suite Gmsh files, <dir>/<prefix>.<suite>.json "
                   "(gmsh_curved_suite.py format: rows with draw / all_quad / fixed); one variant")
    args = p.parse_args()
    suites = args.suites.split(","); variants = args.variants.split(",")
    if args.gmsh_curved:
        variants = ["blossom (curved file)"]
    print(f"\n**Head-to-head ({args.criterion}), agent `{args.tag}` against each Gmsh variant** (wins / losses / ties per suite, "
          f"mean over the best-of-5 passes; win rate = wins / (wins + losses))\n")
    print("| suite | n | " + " | ".join(variants) + " |")
    print("|---|---|" + "---|" * len(variants))
    totals = {}
    for suite in suites:
        adir = args.agent_dir or f"out/paper/{args.tag}"
        files = sorted(glob.glob(f"{adir}/{suite}.[0-9].json")) or sorted(glob.glob(f"{adir}/*{suite}.json"))
        if args.gmsh_curved:
            gpath = args.gmsh_curved.replace("<suite>", suite)
            if not os.path.exists(gpath) or not files:
                continue
            crows = json.load(open(gpath))["rows"]
            rows = {variants[0]: [dict(draw=r["draw"], failed=not r.get("all_quad", False) and r.get("fixed") is None,
                                       all_quad=bool(r.get("all_quad", False)), quality=float(r.get("fixed") or -9)) for r in crows]}
        else:
            gpath = f"{args.gmsh}/{suite}.json"
            if not os.path.exists(gpath) or not files:
                continue
            rows = json.load(open(gpath))["rows"]
        cells = []
        for v in variants:
            gm = {r["draw"]: r for r in rows[v]}
            per_pass = []
            for f in files:
                ag = {d["draw"]: d for d in json.load(open(f))["suites"][suite]["domains"]}
                per_pass.append(outcomes(ag, gm, args.criterion))
            w = statistics.mean(x[0] for x in per_pass); l = statistics.mean(x[1] for x in per_pass); t = statistics.mean(x[2] for x in per_pass)
            n = w + l + t
            tv = totals.setdefault(v, {"in": [0, 0, 0], "large": [0, 0, 0]})
            key = "in" if ("transfer" not in suite) else "large"
            tv[key][0] += w; tv[key][1] += l; tv[key][2] += t
            rate = w / (w + l) if (w + l) > 0 else float("nan")
            cells.append(f"{w:.1f} / {l:.1f} / {t:.1f} ({100 * rate:.0f}%)")
        print(f"| {suite} | {int(round(n))} | " + " | ".join(cells) + " |")
    print("\n| set | " + " | ".join(variants) + " |")
    print("|---|" + "---|" * len(variants))
    for key, label in (("in", "in distribution"), ("large", "larger boundaries")):
        cells = []
        for v in variants:
            w, l, t = totals.get(v, {}).get(key, (0, 0, 0))
            if w + l + t == 0:
                cells.append("-"); continue
            rate = w / (w + l) if (w + l) > 0 else float("nan")
            cells.append(f"{w:.1f} / {l:.1f} / {t:.1f} ({100 * rate:.0f}%)")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
