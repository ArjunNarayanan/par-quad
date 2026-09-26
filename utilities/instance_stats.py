"""Summarise a behaviour-cloning instance set.

Quality is always reported on the FINAL geometry -- after the untangler, not
after the Laplacian. The Laplacian is what the environment runs between moves,
so it is the right thing for the agent to observe, but it is the wrong thing to
judge a mesh by: it sits at its own fixed point and cannot open a folded
element, so it reports a topology as bad when only its drawing was. Every
number here that says "quality" means what a downstream mesher would actually
get.

    venv/bin/python utilities/instance_stats.py -instances data/foo.pkl
    venv/bin/python utilities/instance_stats.py -n 600 -hole_probability 0.6 \\
        -pinwheel_probability 0.4 -gate untangle
"""

import argparse
import collections
import os
import pickle
import sys
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())

from envs.solved_instances import (default_scratch_env, generate_instances,  # noqa: E402
                                   replay)
from utilities.measure_chirality import chirality  # noqa: E402


def final_quality(env, untangle_iterations=20, warm_start=5):
    """Worst element quality after the untangler, which is what downstream sees."""
    laplacian = env.min_element_quality()
    try:
        from src.geo2d_bridge import resmooth_env
        probe = deepcopy(env)
        if warm_start:
            probe.graph.smooth_vertices(num_iter=warm_start)
            probe._update_half_edge_angles()
        resmooth_env(probe, iters=untangle_iterations, method="optimize",
                     slide="optimize")
        return probe.min_element_quality(), laplacian
    except Exception:
        return laplacian, laplacian


def summarise(instances, env=None, quality=True, label=""):
    env = env or default_scratch_env(3, face_desired_degree=4)
    par = collections.Counter()
    moves, corners, final, lap, chir = [], [], [], [], []
    holes = circulating = 0
    for instance in instances:
        par[int(instance.par)] += 1
        moves.append(len(instance.moves))
        corners.append(len(set(instance.loop)))
        if instance.has_hole:
            holes += 1
        if not quality:
            continue
        *_, played, ok = replay(env, instance)
        if not ok:
            continue
        good, raw = final_quality(env)
        final.append(good)
        lap.append(raw)
        if instance.has_hole:
            value = chirality(env.graph, env.vertex_desired_degree)
            if value is not None:
                chir.append(value)
                circulating += int(value >= 0.15)

    total = max(len(instances), 1)
    print(f"\n=== {label or 'instances'}: {len(instances)}")
    print("  par        " + "  ".join(f"{k}:{v}" for k, v in sorted(par.items())))
    print(f"  corners    min {min(corners)}  median {int(np.median(corners))}  "
          f"max {max(corners)}")
    print(f"  moves      min {min(moves)}  median {int(np.median(moves))}  "
          f"max {max(moves)}")
    print(f"  holes      {holes}/{total} ({100 * holes / total:.1f}%)")
    if chir:
        print(f"  circulating {circulating}/{total} ({100 * circulating / total:.1f}%"
              f" of all, {100 * circulating / max(holes, 1):.0f}% of holes)")
        values = np.array(chir)
        print(f"  chirality  mean {values.mean():.3f}  max {values.max():.3f}")
    if final:
        good, raw = np.array(final), np.array(lap)
        print(f"  quality (final, untangled)  mean {good.mean():+.3f}  "
              f"min {good.min():+.3f}  below 0.4: {int((good < 0.4).sum())}  "
              f"below 0.1: {int((good < 0.1).sum())}")
        print(f"    for reference, laplacian    mean {raw.mean():+.3f}  "
              f"min {raw.min():+.3f}  below 0.4: {int((raw < 0.4).sum())}")
    return {"holes": holes / total,
            "circulating": (circulating / total) if chir else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-instances", default=None, help="an existing .pkl to read")
    parser.add_argument("-n", default=400, type=int)
    parser.add_argument("-min_cells", default=2, type=int)
    parser.add_argument("-max_cells", default=14, type=int)
    parser.add_argument("-hole_probability", default=0.15, type=float)
    parser.add_argument("-pinwheel_probability", default=0.0, type=float)
    parser.add_argument("-gate", default="laplacian",
                        choices=("laplacian", "untangle", "topology"))
    parser.add_argument("-seed", default=17, type=int)
    parser.add_argument("-no_quality", action="store_true",
                        help="skip the replay pass; counts only")
    args = parser.parse_args()

    if args.instances:
        with open(args.instances, "rb") as handle:
            instances = pickle.load(handle)
        label = args.instances
    else:
        instances, stats = generate_instances(
            args.n, cell_range=(args.min_cells, args.max_cells),
            rng=np.random.default_rng(args.seed),
            hole_probability=args.hole_probability,
            pinwheel_probability=args.pinwheel_probability,
            gate=args.gate, verbose=False)
        label = (f"hole {args.hole_probability} pinwheel "
                 f"{args.pinwheel_probability} gate {args.gate}")
        print(f"rejections: {stats}")
    summarise(instances, quality=not args.no_quality, label=label)


if __name__ == "__main__":
    main()
