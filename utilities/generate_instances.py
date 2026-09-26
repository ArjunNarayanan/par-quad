"""Build a dataset of instances that come with a certified solution.

Run as a script so the pickle records `envs.solved_instances.SolvedInstance`
rather than a `__main__` class, which would only unpickle back inside this
same script.

    venv/bin/python utilities/generate_instances.py -n 12000 -max_cells 16 \
        -out data/solved_instances.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.append(os.getcwd())

from envs.solved_instances import generate_instances  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate solved instances")
    parser.add_argument("-n", default=2000, type=int)
    parser.add_argument("-min_cells", default=2, type=int)
    parser.add_argument("-max_cells", default=16, type=int)
    parser.add_argument("-hole_probability", default=0.15, type=float)
    parser.add_argument("-polar_probability", default=0.25, type=float)
    parser.add_argument("-pinwheel_probability", default=0.0, type=float,
                        help="share of hole instances with the hole turned against "
                             "its rim; needs -gate untangle to survive")
    parser.add_argument("-gate", default="laplacian",
                        choices=("laplacian", "untangle", "topology"))
    parser.add_argument("-parity_probability", default=0.0, type=float,
                        help="share of outlines re-drawn to want a single parity of degree")
    parser.add_argument("-max_bumps", default=4, type=int)
    parser.add_argument("-face_desired_degree", default=4, type=int,
                        help="4 for quads, 3 for triangles")
    parser.add_argument("-seed", default=0, type=int)
    parser.add_argument("-out", default="data/solved_instances.pkl")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    instances, stats = generate_instances(
        args.n, cell_range=(args.min_cells, args.max_cells), rng=rng,
        hole_probability=args.hole_probability,
        polar_probability=args.polar_probability,
        pinwheel_probability=args.pinwheel_probability, gate=args.gate,
        parity_probability=args.parity_probability, max_bumps=args.max_bumps,
        face_desired_degree=args.face_desired_degree, verbose=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(instances, handle)

    degrees = np.array([i.polygon_degree for i in instances])
    moves = np.array([i.cost_to_go for i in instances])
    pars = np.array([i.par for i in instances])
    print(f"\n{len(instances)} instances -> {args.out}")
    print(f"  face target    : {args.face_desired_degree}")
    print(f"  with holes     : {sum(i.has_hole for i in instances)}")
    print(f"  polygon degree : min {degrees.min()} median {int(np.median(degrees))} "
          f"max {degrees.max()}")
    print(f"  solution length: min {moves.min()} median {int(np.median(moves))} "
          f"max {moves.max()}")
    print(f"  par            : {dict(zip(*np.unique(pars, return_counts=True)))}")
    print(f"  rejections     : {stats}")


if __name__ == "__main__":
    main()
