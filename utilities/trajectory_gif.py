"""Animated GIF of one greedy rollout per failing transfer domain.

    venv/bin/python utilities/trajectory_gif.py -config <cfg> -checkpoint <zip> \
        -suite straight-transfer -failures out/failures/mit2M.straight-transfer.json \
        -per_mode 2 -out out/trajectories

Reads the failure diagnostic (`utilities/transfer_failures.py`) to pick domains that end
INCOMPLETE or BELOW THE BAR, replays the greedy rollout on each with the same seeded draw,
and writes one frame per move: quads shaded by their worst corner's shape quality (the
gallery's colour map), faces that are not yet quads hatched, the half-edge the move was
applied to in orange, and a caption with the step count against the budget, open faces,
excess over par and the current minimum quality. The last frame holds; for an all-quad
ending it is the untangled mesh the scorer would judge.
"""
import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())
import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from PIL import Image                                    # noqa: E402
from utilities.score_quality_objective import SUITES, MAX_EDGE_RATIO, boundary_edge_ratio  # noqa: E402
from utilities.curved_report import shade                # noqa: E402
from utilities.gallery_curved import boundary_path       # noqa: E402


def frame(env, caption, action_half_edge=None, size=480):
    g = env.graph
    fig, ax = plt.subplots(figsize=(size / 96, size / 96 + 0.35), dpi=96)
    corner = g.corner_shape_qualities()
    for face in g.face_list():
        loop = g.generate_half_edge_face_loop(g.first_face_halfedge(face))
        pts = np.array([g.vertex_coordinate(g.source_vertex(h, tag=False)) for h in loop], float)
        if len(loop) == 4:
            worst = min(corner[h] for h in loop if h in corner)
            ax.fill(pts[:, 0], pts[:, 1], facecolor=shade(worst), edgecolor="#37474f", linewidth=0.6, zorder=1)
        else:
            ax.fill(pts[:, 0], pts[:, 1], facecolor="#ffffff", edgecolor="#b71c1c", linewidth=0.8, hatch="///", zorder=1)
    for piece in boundary_path(g):
        ax.plot(piece[:, 0], piece[:, 1], color="#11181c", linewidth=1.6, zorder=3)
    if action_half_edge is not None and g.is_half_edge(action_half_edge):
        a = np.array(g.vertex_coordinate(g.source_vertex(action_half_edge, tag=False)), float)
        b = np.array(g.vertex_coordinate(g.target_vertex(action_half_edge, tag=False)), float)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#ff6f00", linewidth=3.0, zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(caption, fontsize=8, family="monospace")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(img)


def caption_of(env, step, note=""):
    return (f"step {step:3d}/{env.max_steps}  open {int(env.global_face_score):2d}  "
            f"excess {abs(int(env.global_vertex_score) - env.par):2d}  min q {env.min_element_quality():+.2f}{note}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-config", required=True); p.add_argument("-checkpoint", required=True)
    p.add_argument("-suite", required=True); p.add_argument("-failures", default=None)
    p.add_argument("-draws", default="", help="alternative to -failures: comma list of draw:mode, e.g. 76:incomplete,24:below_bar")
    p.add_argument("-per_mode", default=2, type=int); p.add_argument("-n", default=16, type=int)
    p.add_argument("-seed", default=7, type=int); p.add_argument("-out", default="out/trajectories")
    p.add_argument("-ms", default=160, type=int, help="milliseconds per frame")
    args = p.parse_args()

    from envs.environment_maker import initialize_environment
    from src.geo2d_bridge import make_initializer, resmooth_env, load_model
    from src.utils import load_yaml_config
    wanted = {}
    if args.failures:
        recs = json.load(open(args.failures))["records"]
        for mode in ("incomplete", "below_bar"):
            for r in [x for x in recs if x["mode"] == mode][:args.per_mode]:
                wanted[r["draw"]] = r
    for item in [x for x in args.draws.split(",") if x.strip()]:
        d, mode = item.split(":")
        wanted[int(d)] = dict(draw=int(d), mode=mode, corners=0)
    if not wanted:
        print("nothing to render"); return
    config = load_yaml_config(args.config); env_config = dict(config["environment"]); env_config["quality_metric"] = "shape"
    model = load_model(args.checkpoint, args.config, template_size=env_config.get("template_size", 200))
    options = dict(SUITES[args.suite]); options.setdefault("max_corners", 24); options.setdefault("min_corners", 8)
    source = make_initializer(seed=args.seed, **options)
    outdir = os.path.join(args.out, args.suite); os.makedirs(outdir, exist_ok=True)

    built, draw, done_draws = 0, -1, 0
    while built < args.n and draw < 40 * args.n and done_draws < len(wanted):
        draw += 1
        try:
            graph, desired = source()
        except Exception:
            continue
        if "min_corners" in SUITES[args.suite] and boundary_edge_ratio(graph) > MAX_EDGE_RATIO:
            continue
        built += 1
        if draw not in wanted:
            continue
        done_draws += 1
        rec = wanted[draw]

        class Fixed:
            def __init__(self): self.n = len(desired)
            def __call__(self): return deepcopy(graph), dict(desired)

        cfg = dict(env_config); cfg.pop("initializer", None); cfg["graph_initializer"] = Fixed(); cfg["resample_if_at_par"] = False
        env = initialize_environment(cfg)
        obs, _ = env.reset()
        frames = [frame(env, caption_of(env, 0))]
        done, step = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            half_edge, local = env._linear_action_index_to_half_edge_and_action(int(action))
            obs, _, done, trunc, _ = env.step(action); done = done or trunc; step += 1
            kind = "insert" if local == env._insert_vertex_action else f"chord k={env.chord_steps(local)}"
            frames.append(frame(env, caption_of(env, step, f"  {kind}"), half_edge))
        note = "  END: incomplete" if env.global_face_score > 0 else ""
        if env.global_face_score == 0:
            try:
                resmooth_env(env)
            except Exception:
                pass
            note = f"  END untangled: min q {env.min_element_quality():+.2f}"
        frames.append(frame(env, caption_of(env, step, note)))
        name = f"{rec['mode']}-draw{draw:03d}-corners{len(desired)}.gif"
        durations = [args.ms] * (len(frames) - 1) + [2500]
        frames[0].save(os.path.join(outdir, name), save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
        print(f"{name}: {len(frames)} frames, greedy end open {int(env.global_face_score)}, min q {env.min_element_quality():+.3f} (diagnostic: {rec['mode']}, q {rec.get('quality', float('nan')):+.3f})", flush=True)


if __name__ == "__main__":
    main()
