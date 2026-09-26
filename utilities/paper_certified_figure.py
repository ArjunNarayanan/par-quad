"""The certified families, for the paper: one instance per family, four stages each.

    venv/bin/python utilities/paper_certified_figure.py -out out/certified/fig-certified.pdf

Each row runs the REAL generator (`generate_instances`) restricted to one family by
its own probabilities, and records, through the same spies as
`animate_generator_fold.py`, what happened to the first instance it accepted:

  certified seed   the mesh the generator built, at par by construction
  backward walk    that mesh part-way undone (the middle frame of the walk)
  raw polygon      the single face the walk ends on: the training start state
  training instance  the outline after the generator's transform, with the recorded
                   moves replayed on it and accepted by the solved test. Polyominoes
                   are skewed and their corners jittered; the other families only get
                   `_rigid_transform` (rotate, scale, mirror half the time), and since
                   the features are normalised lengths and unsigned angles only the
                   mirror is visible to the agent -- so that column is drawn with the
                   rotation and scale undone and the mirror kept.

Nothing is picked by hand: the instance is the first accepted at a fixed seed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

FILL = "#eef1f4"
NONQUAD = "#c9c3e6"
EDGE = "#7b8794"
BOUNDARY = "#1f2933"
INK = "#1f2933"
MUTED = "#52606d"

FAMILIES = [
    ("polyomino", "polyomino", dict(hole_probability=0.0, polar_probability=0.0,
                                    cell_range=(6, 8), deform_probability=1.0,
                                    angle_probability=0.0), 1),
    ("annulus", "annulus", dict(hole_probability=1.0, pinwheel_probability=0.0,
                                cell_range=(8, 10)), 1),
    ("polar", "polar mesh", dict(hole_probability=0.0, polar_probability=1.0), 6),
    ("pinwheel", "pinwheel annulus", dict(hole_probability=1.0, pinwheel_probability=1.0,
                                          cell_range=(8, 10)), 1),
]
COLUMNS = ["certified seed", "backward walk", "raw polygon", "training instance"]


def snapshot(graph):
    faces = []
    for face in graph.face_list():
        loop = graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))
        points = np.array([graph.vertex_coordinate(graph.source_vertex(h, tag=False))
                           for h in loop], dtype=float)
        faces.append(points)
    return faces


def capture(family_options, seed, gate):
    """The first accepted instance of one family: seed, walk frames, final state."""
    import envs.solved_instances as si
    pending, found = {}, []

    real_walk, real_replay, real_accept = si.backward_walk, si.replay, si.accepts_as_solved
    real_rigid = si._rigid_transform

    def rigid(coordinates, *args, **kwargs):
        out, mirrored = real_rigid(coordinates, *args, **kwargs)
        keys = list(coordinates)
        x = np.array([coordinates[k] for k in keys], float); x -= x.mean(axis=0)
        y = np.array([out[k] for k in keys], float)
        m = np.linalg.lstsq(x, y, rcond=None)[0]          # y = x @ m, m = scale * R^T
        scale = float(np.sqrt(abs(np.linalg.det(m))))
        flip = np.diag([1.0, -1.0]) if mirrored else np.eye(2)
        rotation = (m.T / scale) @ flip                    # R = Rot(theta) @ flip
        # undo Rot(theta) and the scale, keep the flip: y -> y @ Rot(theta) / scale
        pending["undo"] = rotation / scale
        pending["mirrored"] = mirrored
        return out, mirrored

    def walk(graph, desired, *args, **kwargs):
        frames = []
        kwargs["on_step"] = lambda g, d: frames.append(snapshot(g))
        pending.clear()
        pending["seed"] = snapshot(graph)
        result = real_walk(graph, desired, *args, **kwargs)
        pending["walk"] = frames + [snapshot(result[0])] if result and result[0] is not None else frames
        return result

    def replay_spy(env, instance, *args, **kwargs):
        result = real_replay(env, instance, *args, **kwargs)
        pending["par"] = int(env.par)
        return result

    def accept(env, **kwargs):
        verdict = real_accept(env, **kwargs)
        if verdict and not found and pending.get("walk"):
            final = snapshot(env.graph)
            if "undo" in pending:
                final = [f @ pending["undo"] for f in final]
            found.append({**pending, "final": final,
                          "quality": float(env.min_element_quality())})
        return verdict

    si.backward_walk, si.replay, si.accepts_as_solved = walk, replay_spy, accept
    si._rigid_transform = rigid
    try:
        si.generate_instances(1, gate=gate, rng=np.random.default_rng(seed), **family_options)
    finally:
        si.backward_walk, si.replay, si.accepts_as_solved = real_walk, real_replay, real_accept
        si._rigid_transform = real_rigid
    return found[0] if found else None


def _limits(faces, pad=0.08):
    points = np.vstack(faces)
    low, high = points.min(axis=0), points.max(axis=0)
    centre, half = (low + high) / 2, float((high - low).max()) * (0.5 + pad)
    return (centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half)


def _draw(axis, faces, scale):
    from collections import Counter
    from matplotlib.patches import Polygon
    uses = Counter()
    for points in faces:
        keys = [tuple(np.round(p, 9)) for p in points]
        for i in range(len(keys)):
            a, b = keys[i], keys[(i + 1) % len(keys)]
            uses[(min(a, b), max(a, b))] += 1
    for points in faces:
        axis.add_patch(Polygon(points, closed=True, linewidth=0.5 * scale, edgecolor=EDGE,
                               facecolor=FILL if len(points) == 4 else NONQUAD, zorder=1))
    for (a, b), count in uses.items():
        if count == 1:
            axis.plot([a[0], b[0]], [a[1], b[1]], color=BOUNDARY, linewidth=1.1 * scale,
                      solid_capstyle="round", zorder=3)
    xl, yl = _limits(faces)
    axis.set_xlim(*xl)
    axis.set_ylim(*yl)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for side in axis.spines.values():
        side.set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-seed", default=3, type=int)
    parser.add_argument("-gate", default="untangle")
    parser.add_argument("-width", default=5.5, type=float)
    parser.add_argument("-out", required=True)
    parser.add_argument("-png", default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    import matplotlib.pyplot as plt

    rows = []
    for key, label, options, min_faces in FAMILIES:
        # the first accepted instance at the first seed whose certified mesh has
        # at least `min_faces` elements (a three-quad polar triangle says little)
        for seed in range(args.seed, args.seed + 50):
            case = capture(options, seed, args.gate)
            if case is not None and len(case["seed"]) >= min_faces:
                break
        else:
            raise SystemExit(f"no accepted {key} instance from seed {args.seed}")
        middle = case["walk"][len(case["walk"]) // 3]
        rows.append((label, case, [case["seed"], middle, case["walk"][-1], case["final"]]))
        print(f"{key}: {len(case['seed'])} faces, {len(case['walk'])} walk steps, "
              f"seed {seed}, par {case.get('par')}, mirrored {case.get('mirrored')}, final min quality {case['quality']:+.2f}", flush=True)

    figure, axes = plt.subplots(len(rows), 4, figsize=(args.width, 1.25 * len(rows) + 0.25),
                                squeeze=False, gridspec_kw=dict(wspace=0.08, hspace=0.12))
    for r, (label, case, stages) in enumerate(rows):
        for c, faces in enumerate(stages):
            _draw(axes[r][c], faces, 1.0)
            if r == 0:
                axes[r][c].set_title(COLUMNS[c], fontsize=8, color=INK, pad=4)
        axes[r][0].set_ylabel(f"{label}\npar {case.get('par', '?')}", fontsize=7.5, color=INK, labelpad=4)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    figure.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    if args.png:
        figure.savefig(args.png, dpi=250, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
