"""The paper's Tables 1-3 from the scoring queue and the Gmsh variant runs.

    venv/bin/python utilities/paper_tables.py -tag pinwheel [-gmsh out/paper/gmsh]

Table 1 / 3 (agent): per suite, from out/paper/<tag>/<suite>.<rollout_seed>.json:
  all-quad, usable (>= 0.3), at par on the selected mesh, par reached on any attempt,
  median quality, p10, median excess over par, irregular vertices per vertex;
  best-of-5 as mean +- sd over the rollout seeds, and the greedy pass beside it.
Table 2 (Gmsh): per suite and variant, from out/paper/gmsh/<suite>.json, with the
  mean element count so "matched" can be read for what it is (blossom-full and
  quasi-structured cannot get down to the agent's count and are scored at 4-20x it).
"""
import argparse
import glob
import json
import os
import statistics
import sys

sys.path.append(os.getcwd())
from utilities.score_quality_objective import _percentile  # noqa: E402

IN_DIST = ["straight", "polycube", "straight-holes", "polycube-holes"]
LARGE = ["straight-transfer", "polycube-transfer", "straight-holes-transfer", "polycube-holes-transfer"]


def suite_stats(path, suite):
    doms = json.load(open(path))["suites"][suite]["domains"]
    quad = [d for d in doms if d["face"] == 0]
    q = [d["quality"] for d in quad]
    exc = [d["excess"] for d in quad]
    return dict(n=len(doms), quad=len(quad), usable=sum(v >= 0.3 for v in q), usable_01=sum(v >= 0.2 for v in q),
                at_par=sum(e == 0 for e in exc), par_any=sum(d.get("par_reached", False) for d in doms),
                par_any_topo=sum(d.get("par_reached_topological", False) for d in doms),
                q_med=statistics.median(q) if q else float("nan"), p10=_percentile(q, 10) if q else float("nan"),
                exc_med=statistics.median(exc) if exc else float("nan"),
                irr=statistics.median([(d["par"] + d["excess"]) / max(d["vertices"], 1) for d in quad]) if quad else float("nan"),
                elements=statistics.mean(d["elements"] for d in quad) if quad else float("nan"))


def msd(values, fmt):
    if not values:
        return "-"
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.mean(values))} ± {statistics.pstdev(values):.1f}" if "d" in fmt or ".0f" in fmt \
        else f"{fmt.format(statistics.mean(values))} ± {statistics.pstdev(values):.2f}"


def agent_table(tag, suites, title):
    print(f"\n**{title}** (`out/paper/{tag}/`; best of 5 as mean ± sd over rollout seeds, greedy beside it)\n")
    # column order (Arjun, 2026-09-24): all-quad, usable, irregular vertices per vertex, at par
    print("| suite | n | protocol | all-quad | usable (q >= 0.3) | irregular / vertex | at par (selected) | par on any attempt | q med | q p10 | excess med | elements |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    totals = {}
    for suite in suites:
        files = sorted(glob.glob(f"out/paper/{tag}/{suite}.[0-9].json"))
        greedy = f"out/paper/{tag}/{suite}.greedy.json"
        if files:
            st = [suite_stats(f, suite) for f in files]
            n = st[0]["n"]
            row = dict(quad=msd([s["quad"] for s in st], "{:.1f}"), usable=msd([s["usable"] for s in st], "{:.1f}"),
                       at_par=msd([s["at_par"] for s in st], "{:.1f}"), par_any=msd([s["par_any"] for s in st], "{:.1f}"),
                       q=msd([s["q_med"] for s in st], "{:+.2f}"), p10=msd([s["p10"] for s in st], "{:+.2f}"),
                       exc=msd([s["exc_med"] for s in st], "{:.1f}"), irr=msd([s["irr"] for s in st], "{:.3f}"),
                       el=msd([s["elements"] for s in st], "{:.1f}"))
            print(f"| {suite} | {n} | best of 5, {len(st)} passes | {row['quad']} | {row['usable']} | {row['irr']} | {row['at_par']} | {row['par_any']} | {row['q']} | {row['p10']} | {row['exc']} | {row['el']} |")
            for k in ("quad", "usable", "at_par", "par_any"):
                totals.setdefault(k, []).append(statistics.mean(s[k] for s in st))
        if os.path.exists(greedy):
            g = suite_stats(greedy, suite)
            print(f"| {suite} | {g['n']} | greedy | {g['quad']} | {g['usable']} | {g['irr']:.3f} | {g['at_par']} | {g['par_any']} | {g['q_med']:+.2f} | {g['p10']:+.2f} | {g['exc_med']:.1f} | {g['elements']:.1f} |")
    if totals:
        print(f"\nTotals over the suites (best of 5, mean over passes): all-quad {sum(totals['quad']):.1f}, usable {sum(totals['usable']):.1f}, "
              f"at par (selected) {sum(totals['at_par']):.1f}, par on any attempt {sum(totals['par_any']):.1f}.")


def gmsh_table(gdir, suites, tag):
    print(f"\n**Gmsh, seven variants, searched to the agent's element count domain by domain** (`{gdir}/`)\n")
    print("| suite | variant | all-quad | usable (q >= 0.3) | irregular / vertex | at par | excess med | q med | q p10 | mean elements (agent) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for suite in suites:
        path = f"{gdir}/{suite}.json"
        if not os.path.exists(path):
            continue
        d = json.load(open(path)); rows = d["rows"]
        agent_el = statistics.mean(r["agent_elements"] for r in next(iter(rows.values())) if not r["failed"])
        for v, s in d["summary"].items():
            f = lambda x, fmt: "-" if x is None else fmt.format(x)
            gq = [r["quality"] for r in rows[v] if not r["failed"] and r["all_quad"]]
            s = dict(s, usable=sum(x >= 0.3 for x in gq))   # recounted at the 0.3 bar from the rows
            print(f"| {suite} | {v} | {s['all_quad']} | {s['usable']} | {f(s['irregular_per_vertex_median'], '{:.3f}')} | {s['at_par']} | {f(s['excess_median'], '{:.1f}')} | {f(s['quality_median'], '{:+.2f}')} | {f(s['quality_p10'], '{:+.2f}')} | {f(s['elements_mean'], '{:.1f}')} ({agent_el:.1f}) |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tag", default="pinwheel")
    parser.add_argument("-gmsh", default="out/paper/gmsh")
    args = parser.parse_args()
    agent_table(args.tag, IN_DIST, "Table 1: in-distribution (8-24 corners)")
    agent_table(args.tag, LARGE, "Table 3: larger boundaries (25-50 total corners)")
    gmsh_table(args.gmsh, IN_DIST + LARGE, args.tag)


if __name__ == "__main__":
    main()
