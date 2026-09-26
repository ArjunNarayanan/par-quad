"""A held-out test set of domains whose Gauss-Bonnet bound is NOT zero.

Every geo2d domain is par 0 -- all eleven presets, all 120 test domains -- because
geo2d builds rectilinear mechanical parts and par is the total rounding error of
the corner angles. So the benchmark has never tested the case the bound is
actually interesting in: a domain where a perfectly regular mesh is IMPOSSIBLE
and the agent has to find the least irregular one instead.

The certified-instance generator makes them, by bumping k corners' desired
degree in a consistent direction, and about 35% of training instances are par >
0. This draws the same families at a held-out seed and keeps only the domains,
not the solutions -- the agent has to find its own.

Disjointness from training is by seed, and `-exclude` additionally drops any
domain whose outline signature appears in a training pickle, matching
`check_leakage.py`'s rule so the *domain* goes rather than just the pose.

    venv/bin/python utilities/make_par_set.py -n 48 -seed 4242 \\
        -exclude data/instances_arm_pinwheel.pkl -out data/par_test_set.pkl
"""

import argparse
import collections
import os
import pickle
import sys

import numpy as np

sys.path.append(os.getcwd())

from envs.solved_instances import generate_instances  # noqa: E402


def outline_signature(instance):
    """What `check_leakage` matches on: the domain, not its pose.

    Corner turn angles in loop order, rounded, canonicalised over rotations and
    reflections, so a domain that is the same shape differently placed collides.
    """
    import envs.polygon_utils as utils
    loop = list(instance.loop)
    coordinates = {v: np.asarray(c, dtype=float) for v, c in instance.coordinates.items()}
    angles = utils.get_polygon_interior_angles(loop, coordinates)
    sequence = tuple(int(round(angles[v])) for v in loop)
    rotations = [sequence[i:] + sequence[:i] for i in range(len(sequence))]
    flipped = tuple(reversed(sequence))
    rotations += [flipped[i:] + flipped[:i] for i in range(len(flipped))]
    return min(rotations)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", default=48, type=int, help="domains to keep")
    parser.add_argument("-seed", default=4242, type=int, help="held out from training")
    parser.add_argument("-min_corners", default=8, type=int)
    parser.add_argument("-max_corners", default=24, type=int)
    parser.add_argument("-min_par", default=1, type=int)
    parser.add_argument("-draw", default=4000, type=int,
                        help="candidates to generate before filtering")
    parser.add_argument("-exclude", default=None,
                        help="a training pickle whose domains must not appear")
    parser.add_argument("-out", default="data/par_test_set.pkl")
    args = parser.parse_args()

    banned = set()
    if args.exclude and os.path.exists(args.exclude):
        with open(args.exclude, "rb") as handle:
            for instance in pickle.load(handle):
                banned.add(outline_signature(instance))
        print(f"excluding {len(banned)} outline signatures from {args.exclude}")

    instances, _ = generate_instances(args.draw, cell_range=(2, 14),
                                      hole_probability=0.30,
                                      pinwheel_probability=0.40, gate="untangle",
                                      rng=np.random.default_rng(args.seed),
                                      verbose=False)
    kept, seen, leaked = [], set(), 0
    for instance in instances:
        if instance.par < args.min_par:
            continue
        corners = len(set(instance.loop))
        if not (args.min_corners <= corners <= args.max_corners):
            continue
        signature = outline_signature(instance)
        if signature in banned:
            leaked += 1
            continue
        if signature in seen:          # no duplicate domains within the set either
            continue
        seen.add(signature)
        kept.append(instance)
        if len(kept) >= args.n:
            break

    if not kept:
        raise SystemExit("no domains matched; widen -draw or the corner window")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(kept, handle)

    par = collections.Counter(int(i.par) for i in kept)
    corners = [len(set(i.loop)) for i in kept]
    holes = sum(1 for i in kept if i.has_hole)
    print(f"\n{len(kept)} domains -> {args.out}")
    print("  par      " + "  ".join(f"{k}:{v}" for k, v in sorted(par.items())))
    print(f"  corners  min {min(corners)}  median {int(np.median(corners))}  max {max(corners)}")
    print(f"  holes    {holes}/{len(kept)}")
    print(f"  dropped for colliding with training: {leaked}")


if __name__ == "__main__":
    main()
