"""Animate the released agent building a block decomposition, move by move.

    venv/bin/python utilities/animate_rollout.py -suite straight-holes -draw 10 -n 24 \
        -out docs/media -name indist

Rolls out the agent greedily on one domain of a scorer suite (the same seeded draw the
paper's tables use) at the evaluation budget, and writes `<name>.mp4` and `<name>.gif`:
one frame per move, drawn as the paper's figures are (quadrilaterals light, faces that
are not yet quadrilaterals shaded, elements below the usability bar hatched). Irregular
vertices are marked once the mesh is all-quadrilateral, and the last frame -- the mesh
after the untangling smoother, as it is scored -- is held.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())

CONFIG = "models/paper-2026-09-24/pinwheel-mitq-e2e-v1.config.yml"
CHECKPOINT = "models/paper-2026-09-24/pinwheel-mitq-e2e-v1-4M.zip"


def draw_domain(suite, draw, n, seed, densify):
    from src.geo2d_bridge import make_initializer
    from utilities.score_quality_objective import (MAX_EDGE_RATIO, MIN_CORNERS, SUITES,
                                                   boundary_edge_ratio)
    options = dict(SUITES[suite])
    capped = "min_corners" in options
    options.setdefault("max_corners", 24)
    options.setdefault("min_corners", MIN_CORNERS)
    source = make_initializer(seed=seed, densify_sweep=densify or None, **options)
    index, kept = -1, 0
    while kept < n and index < 40 * n:
        index += 1
        try:
            graph, desired = source()
        except Exception:
            continue
        if capped and boundary_edge_ratio(graph) > MAX_EDGE_RATIO:
            continue
        kept += 1
        if index == draw:
            return graph, desired
    raise SystemExit(f"draw {draw} is not among the first {n} kept draws of {suite}")


def snapshot(env):
    """The current mesh in `paper_figure`'s drawing format."""
    from src.geo2d_bridge import tiler_faces
    g = env.graph
    nodes, elements, index = tiler_faces(g)
    corner = g.corner_shape_qualities()
    face_quality = []
    for face in g.face_list():
        loop = g.generate_half_edge_face_loop(g.first_face_halfedge(face))
        values = [corner[h] for h in loop if h in corner]
        face_quality.append(float(min(values)) if values else 1.0)
    all_quad = all(len(e) == 4 for e in elements)
    vertices = [dict(node=i, degree=int(g.vertex_degree(v)),
                     wants=tuple(int(w) for w in env.desired_degrees(v)) if all_quad else (int(g.vertex_degree(v)),))
                for v, i in index.items()]
    return dict(nodes=np.asarray(nodes, float), elements=[list(e) for e in elements],
                face_quality=face_quality, vertices=vertices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-suite", required=True)
    p.add_argument("-draw", required=True, type=int)
    p.add_argument("-n", default=24, type=int, help="the suite size the draw index comes from")
    p.add_argument("-seed", default=7, type=int)
    p.add_argument("-densify", default=0.0, type=float, help="arc discretisation in degrees (45 for curved)")
    p.add_argument("-config", default=CONFIG)
    p.add_argument("-checkpoint", default=CHECKPOINT)
    p.add_argument("-max_steps_factor", default=6.0, type=float)
    p.add_argument("-fps", default=6, type=int)
    p.add_argument("-hold", default=2.0, type=float, help="seconds the final mesh is held")
    p.add_argument("-size", default=4.0, type=float, help="inches")
    p.add_argument("-out", default="docs/media")
    p.add_argument("-name", required=True)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from envs.environment_maker import initialize_environment
    from src.geo2d_bridge import load_model, resmooth_env
    from src.utils import load_yaml_config
    from utilities.paper_figure import _domain_arcs, _draw, _style

    _style()
    graph, desired = draw_domain(args.suite, args.draw, args.n, args.seed, args.densify)
    config = load_yaml_config(args.config)
    cfg = dict(config["environment"])
    cfg["max_steps_factor"] = args.max_steps_factor
    cfg["quality_metric"] = "shape"

    class Fixed:
        def __call__(self):
            return deepcopy(graph), dict(desired)
    cfg.pop("initializer", None)
    cfg["graph_initializer"] = Fixed()
    cfg["resample_if_at_par"] = False
    env = initialize_environment(cfg)
    model = load_model(args.checkpoint, args.config, template_size=cfg.get("template_size", 200))
    observation, _ = env.reset()

    frames, captions = [snapshot(env)], ["start: one face"]
    done = truncated = False
    moves = 0
    while not (done or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, truncated, _ = env.step(action)
        moves += 1
        mesh = snapshot(env)
        open_faces = sum(len(e) != 4 for e in mesh["elements"])
        frames.append(mesh)
        captions.append(f"move {moves}: {open_faces} open face{'s' if open_faces != 1 else ''}"
                        if open_faces else f"move {moves}: all quadrilateral")
        if not open_faces and env.global_vertex_score == env.par:
            break
    if all(len(e) == 4 for e in frames[-1]["elements"]):
        try:
            resmooth_env(env)
        except Exception:
            pass
        excess = int(env.global_vertex_score - env.par)
        frames.append(snapshot(env))
        captions.append(f"{moves} moves, excess over par {excess}" + (" (optimal)" if excess == 0 else ""))

    arcs = _domain_arcs(tempfile.gettempdir(), args.suite, args.draw, args.seed) if args.suite.startswith("curved") else ()
    points = np.vstack([m["nodes"] for m in frames])
    low, high = points.min(axis=0), points.max(axis=0)
    pad = 0.04 * float((high - low).max())
    count = max(len(frames[-1]["elements"]), 1)
    scale = float(np.clip((20.0 / count) ** 0.25, 0.55, 1.0))
    work = tempfile.mkdtemp()
    for k, (mesh, caption) in enumerate(zip(frames, captions)):
        fig, ax = plt.subplots(figsize=(args.size, args.size * float((high - low)[1] + 2 * pad) / float((high - low)[0] + 2 * pad) + 0.35))
        _draw(ax, mesh, 0.3, scale, arcs=arcs)
        ax.set_xlim(low[0] - pad, high[0] + pad)
        ax.set_ylim(low[1] - pad, high[1] + pad)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel(caption, fontsize=10, color="#3b4252")
        fig.savefig(os.path.join(work, f"f{k:04d}.png"), dpi=120, facecolor="white")
        plt.close(fig)
    last = os.path.join(work, f"f{len(frames) - 1:04d}.png")
    for extra in range(int(args.hold * args.fps)):
        shutil.copy(last, os.path.join(work, f"f{len(frames) + extra:04d}.png"))
    os.makedirs(args.out, exist_ok=True)
    mp4 = os.path.join(args.out, f"{args.name}.mp4")
    gif = os.path.join(args.out, f"{args.name}.gif")
    even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i",
                    os.path.join(work, "f%04d.png"), "-vf", even, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", mp4], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i",
                    os.path.join(work, "f%04d.png"), "-vf",
                    "scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse",
                    gif], check=True)
    shutil.rmtree(work)
    print(f"{args.name}: {moves} moves, {len(frames)} frames -> {mp4}, {gif}")


if __name__ == "__main__":
    main()
