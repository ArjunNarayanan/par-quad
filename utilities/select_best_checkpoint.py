"""Pick an arm's best checkpoint on the scorer, in the order a mesher is judged.

    venv/bin/python utilities/select_best_checkpoint.py <arm> [-suites a,b,c] [-min_steps 1]

Reads `out/<arm>/scores/<steps>.<suite>.json` for every checkpoint scored on ALL
the given suites and ranks by (usable at 0.3, all-quad, -median excess over par,
median quality), summed / pooled over the suites. Writes `out/<arm>/best.json`
and copies the winning checkpoint to `experiments/.../<arm>/best_on_scorer.zip`.
Selection suites should be IN-distribution ones; transfer suites are for reporting.
"""
import argparse
import glob
import json
import os
import shutil
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("-suites", default="straight,polycube,curved,curved-holes")
    parser.add_argument("-min_steps", default=1, type=int, help="ignore the warm start (step 0) by default")
    args = parser.parse_args()
    suites = args.suites.split(",")
    scores = f"out/{args.arm}/scores"
    steps = sorted({os.path.basename(f).split(".")[0] for f in glob.glob(f"{scores}/[0-9]*.json")})
    rows = []
    for st in steps:
        if int(st) < args.min_steps:
            continue
        files = [f"{scores}/{st}.{s}.json" for s in suites]
        if not all(os.path.exists(f) for f in files):
            continue
        usable = quad = 0; excess, quality = [], []
        for f, s in zip(files, suites):
            doms = json.load(open(f))["suites"][s]["domains"]
            q = [d for d in doms if d["face"] == 0]
            quad += len(q); usable += sum(d["quality"] >= 0.3 for d in q)
            excess += [d["excess"] for d in q]; quality += [d["quality"] for d in q]
        rows.append(dict(steps=int(st), usable=usable, quad=quad,
                         excess_median=statistics.median(excess) if excess else 99,
                         quality_median=statistics.median(quality) if quality else -1))
    if not rows:
        print("nothing scored on all suites yet"); return
    best = max(rows, key=lambda r: (r["usable"], r["quad"], -r["excess_median"], r["quality_median"], r["steps"]))
    out = dict(arm=args.arm, suites=suites, best=best, rows=rows)
    json.dump(out, open(f"out/{args.arm}/best.json", "w"), indent=1)
    src = f"experiments/self-play/quad/{args.arm}/warm_ppo_{best['steps']}_steps.zip"
    dst = f"experiments/self-play/quad/{args.arm}/best_on_scorer.zip"
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    print(f"best on scorer: {best['steps']/1e6:.2f}M  usable {best['usable']}  quad {best['quad']}  "
          f"exc med {best['excess_median']}  q med {best['quality_median']:+.3f}  ({len(rows)} checkpoints ranked)")


if __name__ == "__main__":
    main()
