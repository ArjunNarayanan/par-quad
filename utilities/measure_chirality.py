"""Does a solution circulate around its hole, and does the generator ever build one?

A domain whose hole is turned relative to its outer boundary -- a diamond inside
a square -- cannot be meshed radially. Each re-entrant hole corner has to send
its edges consistently to one side, clockwise or counter-clockwise, and the two
choices are mirror images: reaching par means breaking a symmetry the domain
still has. A domain whose hole is aligned with its rim needs no such choice.

The statistic is the mean sign of the angle from the outward radial, over every
edge leaving a re-entrant boundary corner, in absolute value. 0 is a balanced
radial mesh; 1 is a pinwheel with every edge on the same side.

This matters because the certified-instance generator builds its holes with
`random_annulus`, a ring of grid cells around one grid cell -- hole and rim
always axis-aligned, never rotated. So the pinwheel is not a rare case in
training. It is absent.

    venv/bin/python utilities/measure_chirality.py -n 250
    venv/bin/python utilities/measure_chirality.py -logs out/failures-best
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "game"))


def _neighbours(graph, vertex):
    """Every vertex joined to this one by an edge, each counted once."""
    found = set()
    for half_edge in graph.half_edge_list():
        source = graph.source_vertex(half_edge, tag=False)
        target = graph.target_vertex(half_edge, tag=False)
        if source == vertex:
            found.add(target)
        elif target == vertex:
            found.add(source)
    return sorted(found)


def chirality(graph, desired, min_edges=4):
    """|mean sign of edge angle from the outward radial| at re-entrant corners."""
    points = np.array([graph.vertex_coordinate(v) for v in graph.vertex_list(tag=False)])
    centre = points.mean(axis=0)
    signs = []
    for vertex in graph.vertex_list(tag=False):
        want = desired.get(vertex)
        if want is None:
            continue
        want = max(want) if isinstance(want, (set, tuple, list)) else want
        # a re-entrant corner is a boundary vertex wanting four or more
        if not graph.is_boundary_vertex(vertex) or want < 4:
            continue
        origin = np.array(graph.vertex_coordinate(vertex), dtype=float)
        outward = origin - centre
        norm = np.linalg.norm(outward)
        if norm < 1e-9:
            continue
        outward /= norm
        # NEIGHBOURS, not outgoing half-edges. On the boundary only the interior
        # half-edge of an edge exists, so counting by source makes the measure
        # depend on the loop's orientation: the same mesh reflected scored
        # -1.000 one way and +0.333 the other. An edge is an edge either way.
        for target_vertex in _neighbours(graph, vertex):
            target = np.array(graph.vertex_coordinate(target_vertex), dtype=float)
            direction = target - origin
            length = np.linalg.norm(direction)
            if length < 1e-9:
                continue
            direction /= length
            cross = outward[0] * direction[1] - outward[1] * direction[0]
            if abs(cross) > 1e-6:
                signs.append(math.copysign(1.0, cross))
    if len(signs) < min_edges:
        return None
    return abs(sum(signs) / len(signs))


def generator_distribution(count, hole_probability=1.0):
    from envs.solved_instances import (default_scratch_env, generate_instances,
                                       replay)
    env = default_scratch_env(3, face_desired_degree=4)
    instances, _ = generate_instances(count, hole_probability=hole_probability,
                                      verbose=False)
    values = []
    for instance in instances:
        if not instance.has_hole:
            continue
        *_, played, ok = replay(env, instance)
        if not ok or not env.is_at_par():
            continue
        value = chirality(env.graph, env.vertex_desired_degree)
        if value is not None:
            values.append(value)
    return np.array(values)


def log_values(directory):
    sys.path.insert(0, os.path.join(os.getcwd(), "utilities"))
    import server
    from replay_game_log import boards, replay as replay_log
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        payload = json.load(open(path))
        seen = set()
        for record in boards(payload, None, False):
            level = record["shape"]
            if level in seen:
                continue
            game, _, failure = replay_log(record)
            if failure is not None:
                continue
            vertex, face, _ = game._current_scores()
            if face != 0 or vertex != record.get("par"):
                continue          # only par solutions say anything about structure
            seen.add(level)
            desired = {v: max(server.Game.desired_degrees(game, v))
                       for v in game.graph.vertex_list(tag=False)}
            value = chirality(game.graph, desired)
            if value is not None:
                out.append((level, os.path.basename(path), value))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=250,
                        help="certified instances to draw for the reference distribution")
    parser.add_argument("-logs", default=None,
                        help="directory of exported Mesh Quest logs to score as well")
    args = parser.parse_args()

    values = generator_distribution(args.n)
    print(f"certified hole solutions from the generator: {len(values)}")
    print(f"  chirality  mean {values.mean():.3f}   median {np.median(values):.3f}"
          f"   max {values.max():.3f}")
    for low, high in ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
        n = int(((values >= low) & (values < high)).sum())
        bar = "#" * (n * 40 // max(len(values), 1))
        print(f"  {low:.1f}-{high:.1f}  {n:>4}  ({100 * n / len(values):5.1f}%)  {bar}")

    if args.logs:
        print(f"\npar solutions played by hand in {args.logs}:")
        for level, source, value in log_values(args.logs):
            beyond = "  <- beyond anything the generator builds" if value > values.max() else ""
            print(f"  {level:<22} {value:.3f}   ({source}){beyond}")


if __name__ == "__main__":
    main()
