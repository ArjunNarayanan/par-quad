"""What a par-0 solution must look like, for a domain that has one.

    venv/bin/python utilities/par0_structure.py -level "Hook"

A vertex score of 0 means every vertex sits exactly on its want, so the mesh has
no singularities at all. Walking the boundary, a want-2 vertex turns +90, want-3
goes straight and want-4 turns -90, so the mesh boundary is a RECTILINEAR
polygon whose corners are the domain's want-not-3 vertices and whose sides are
the runs between them. Each domain edge contributes a length of at least one to
its side, and any edge may be subdivided to contribute more.

So finding the solution is, first, an integer problem: pick a length for each
side, at or above its lower bound, so the loop closes and stays simple. That
fixes how many vertices to insert and exactly where. The quads then follow.

This is the part that has no local cue on a curved boundary. A smooth arc joint
wants three and looks like any other flat point, so the counting has to be done
over the whole outline -- which is why these domains are hard for a policy
reading a local window, and fiddly for a human.
"""

import argparse
import itertools
import os
import sys

sys.path.append(os.getcwd())

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
NAMES = ["east", "north", "west", "south"]


def sides_of(wants):
    """The rectilinear sides a zero-score mesh must have: (direction, lower bound).

    Also returns, for each side, the domain vertices it runs between.
    """
    count = len(wants)
    turns = [3 - w for w in wants]
    if sum(turns) != 4:
        return None
    corners = [i for i, t in enumerate(turns) if t]
    sides, direction = [], 0
    for index, corner in enumerate(corners):
        following = corners[(index + 1) % len(corners)]
        direction = (direction + turns[corner]) % 4
        length = (following - corner) % count or count
        sides.append({"from": corner, "to": following, "dir": direction,
                      "min": length})
    return sides


def is_simple(sides, lengths):
    """Does walking these sides trace a non-self-intersecting rectilinear loop?

    Walk it in UNIT steps and require every lattice point to be visited once.
    Checking only whether segments share an edge is not enough: two sides can
    cross at a point they both merely pass through, which is what a bare
    edge-overlap test lets past and what produces a zero-area answer.
    """
    point, seen = (0, 0), []
    for side, length in zip(sides, lengths):
        dx, dy = DIRECTIONS[side["dir"]]
        for _ in range(length):
            seen.append(point)
            point = (point[0] + dx, point[1] + dy)
    if point != (0, 0):
        return False
    return len(set(seen)) == len(seen)


def area_of(sides, lengths):
    point, shoelace = (0, 0), 0
    for side, length in zip(sides, lengths):
        dx, dy = DIRECTIONS[side["dir"]]
        nxt = (point[0] + dx * length, point[1] + dy * length)
        shoelace += point[0] * nxt[1] - nxt[0] * point[1]
        point = nxt
    return abs(shoelace) // 2


def solve(sides, budget=6):
    """The cheapest set of side lengths that closes the loop and stays simple."""
    best = None
    n = len(sides)
    for extra_total in range(budget + 1):
        for split in itertools.combinations_with_replacement(range(n), extra_total):
            extras = [0] * n
            for k in split:
                extras[k] += 1
            lengths = [s["min"] + e for s, e in zip(sides, extras)]
            if not is_simple(sides, lengths):
                continue
            best = (extras, lengths, area_of(sides, lengths))
            return best
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-level", required=True, help="a name in src/curved_domains.json")
    parser.add_argument("-budget", default=6, type=int)
    args = parser.parse_args()

    from src.curved_levels import from_spec, load_domains
    spec = load_domains()[args.level]
    _, wants = from_spec(spec)
    order = sorted(wants)
    sequence = [wants[v] for v in order]

    sides = sides_of(sequence)
    if sides is None:
        print(f"{args.level}: not a par-0 domain (turning number is not 4)")
        return
    found = solve(sides, args.budget)
    print(f"{args.level}: {len(sequence)} control points, wants {sequence}")
    print(f"  a zero-score mesh is a rectilinear {len(sides)}-gon.\n")
    if found is None:
        print(f"  NO simple closure within {args.budget} extra edges -- par 0 may be"
              " unreachable here, or needs a bigger budget.")
        return
    extras, lengths, area = found
    print(f"  {'side':>4} {'runs between':>14} {'direction':>10} {'min':>4} "
          f"{'use':>4}  insert")
    for side, length, extra in zip(sides, lengths, extras):
        print(f"  {'':>4} v{side['from']:<3} -> v{side['to']:<6} "
              f"{NAMES[side['dir']]:>10} {side['min']:>4} {length:>4}"
              + (f"  {extra} vertex(es) on this run" if extra else ""))
    print(f"\n  total insertions: {sum(extras)}   quads in the finished mesh: {area}")
    print(f"  boundary vertices: {sum(lengths)}   interior vertices: "
          f"{area - sum(lengths) // 2 + 1}")


if __name__ == "__main__":
    main()
