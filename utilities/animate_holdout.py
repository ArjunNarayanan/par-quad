"""Animate the agent solving the held-out Mesh Quest levels.

Renders a greedy rollout per level and writes an animated GIF: one panel per
level, all fifteen advancing together, with each panel holding its final mesh
once solved so the grid stays in step.

Faces are coloured by how far they are from the element the run asks for
(quads or triangles) and vertices by how far their degree is from what the
domain wants, so "solved" is visible rather than asserted -- at par every face
is pale blue and every vertex is grey. The edge inserted by the most recent
move is drawn heavier.

    venv/bin/python utilities/animate_holdout.py -input models/mesh-quest-15of15 \\
        -out solve.gif
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

sys.path.append(os.getcwd())

from src.holdout import ALL_LEVEL_NAMES, LEVEL_NAMES, make_level_env  # noqa: E402
from src.utils import load_yaml_config, load_model_from_checkpoint  # noqa: E402

DONE_FILL = "#dfe7f2"
WORK_FILL = "#f6d9b0"
BAD_FILL = "#eab8b8"
EDGE_COLOUR = "#3c4a5a"
NEW_EDGE_COLOUR = "#c1440e"
REGULAR_VERTEX = "#8b97a6"
IRREGULAR_VERTEX = "#c1440e"
SOLVED_TITLE = "#1d7a46"


def snapshot(env, previous_edges=None):
    """Everything a frame needs, read off the env's current mesh."""
    graph = env.graph
    faces = []
    for face in graph.face_list():
        loop = graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))
        points = [graph.vertex_coordinate(graph.source_vertex(h, tag=False)) for h in loop]
        faces.append((np.asarray(points, dtype=float), graph.face_degree(face)))

    edges = {}
    for half_edge in graph.half_edge_list():
        source = graph.source_vertex(half_edge, tag=False)
        target = graph.target_vertex(half_edge, tag=False)
        key = frozenset((source, target))
        if key not in edges:
            edges[key] = (np.asarray(graph.vertex_coordinate(source), dtype=float),
                          np.asarray(graph.vertex_coordinate(target), dtype=float))

    vertices = []
    for vertex in graph.vertex_list(tag=False):
        # tie-aware: a corner on a rounding tie is regular at either admissible degree
        defect = env.vertex_defect(vertex)
        vertices.append((np.asarray(graph.vertex_coordinate(vertex), dtype=float), defect))

    new_keys = set(edges) - set(previous_edges or {})
    return {
        "faces": faces,
        "edges": edges,
        "new_edges": new_keys,
        "vertices": vertices,
        "moves": env.num_steps,
        "face_score": env.global_face_score,
        "vertex_excess": env.global_vertex_score - env.par,
        "par": env.par,
        "solved": bool(env.is_at_par()),
    }


def _one_rollout(model, env, deterministic):
    obs, _ = env.reset()
    frames = [snapshot(env)]
    for _ in range(env.max_steps):
        if frames[-1]["solved"]:
            break
        batched = {key: np.expand_dims(value, 0) for key, value in obs.items()}
        action, _ = model.predict(batched, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(int(np.asarray(action).ravel()[0]))
        frames.append(snapshot(env, previous_edges=frames[-1]["edges"]))
        if terminated or truncated:
            break
    return frames


def rollout_frames(model, env, samples=0):
    """One rollout to animate.

    `samples` above zero runs that many sampled episodes and keeps the shortest
    solved one, falling back to the closest miss -- the best-of-N protocol,
    animated. Zero is a single greedy rollout.
    """
    if samples <= 0:
        return _one_rollout(model, env, deterministic=True)

    best, best_key = None, None
    for _ in range(samples):
        frames = _one_rollout(model, env, deterministic=False)
        final = frames[-1]
        key = (not final["solved"], final["face_score"],
               abs(final["vertex_excess"]), len(frames))
        if best_key is None or key < best_key:
            best, best_key = frames, key
            if final["solved"] and len(frames) <= 4:
                break
    return best


def face_colour(degree, target):
    """Pale blue once a face is the element we asked for, warm while it is not.

    Red is reserved for a face no chord can finish. At the quad target that is
    every odd face -- a chord splits one face into two whose degrees sum to
    degree + 2, so parity is invariant and an odd face needs a vertex insert
    before it can ever become quads. At the triangle target no such obstruction
    exists: any face of degree 3 or more triangulates by chords alone, so only
    a degenerate face (degree below the target) is red.
    """
    if degree == target:
        return DONE_FILL
    if degree < target:
        return BAD_FILL
    if target == 4 and degree % 2 == 1:
        return BAD_FILL
    return WORK_FILL


def draw(axis, frame, level, limits, target=4, position=None):
    axis.clear()
    if position is not None:
        axis.set_position(position)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    # square data limits plus a fixed axes rectangle: the drawn box, and so the
    # pixel each mesh coordinate lands on, is identical in every frame
    axis.set_aspect("equal", adjustable="box", anchor="C")
    axis.axis("off")

    for points, degree in frame["faces"]:
        colour = face_colour(degree, target)
        axis.fill(points[:, 0], points[:, 1], facecolor=colour, edgecolor="none", zorder=1)

    for key, (start, end) in frame["edges"].items():
        is_new = key in frame["new_edges"]
        axis.plot([start[0], end[0]], [start[1], end[1]],
                  color=NEW_EDGE_COLOUR if is_new else EDGE_COLOUR,
                  linewidth=2.6 if is_new else 1.1,
                  zorder=3 if is_new else 2, solid_capstyle="round")

    for point, defect in frame["vertices"]:
        axis.plot(point[0], point[1], "o", markersize=3.4 if defect else 2.4,
                  color=IRREGULAR_VERTEX if defect else REGULAR_VERTEX, zorder=4)

    if frame["solved"]:
        status = f"solved in {frame['moves']}"
        colour = SOLVED_TITLE
    else:
        status = f"move {frame['moves']}   faces {frame['face_score']}   " \
                 f"vertices {frame['vertex_excess']:+d}"
        colour = "#55606d"
    axis.set_title(f"{level}\n{status}", fontsize=9, color=colour, pad=4,
                   loc="center")


def write_gif(path, fps, pause_seconds, pause_at=None):
    """Re-encode a written GIF so it is stable, small, and pauses where asked.

    Three things the plain encoder gets wrong for this use. A duplicated
    trailing frame is collapsed -- identical frames are the point -- which
    silently removes an end pause, so the pause is a duration instead. Frames
    are quantised against one shared palette, or a panel that has finished
    shimmers as the palette is re-derived around it. And disposal is "leave in
    place": disposing to background erases whatever the optimiser left out of
    a partial frame.
    """
    from PIL import Image, ImageSequence

    with Image.open(path) as handle:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(handle)]

    stack = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for index, frame in enumerate(frames):
        stack.paste(frame, (0, index * frame.height))
    master = stack.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    paletted = [frame.quantize(palette=master, dither=Image.Dither.NONE)
                for frame in frames]

    step_ms = int(round(1000 / fps))
    durations = [step_ms] * len(paletted)
    holds = [len(paletted) - 1] if pause_at is None else list(pause_at)
    for index in holds:
        if 0 <= index < len(durations):
            durations[index] = step_ms + int(round(max(pause_seconds, 0.0) * 1000))

    # optimize=True re-derives per-frame palettes and crops frames, which undoes
    # the shared palette; full frames are a little larger and exactly reproducible.
    paletted[0].save(path, save_all=True, append_images=paletted[1:],
                     duration=durations, loop=0, disposal=1, optimize=False)
    return len(paletted)


def level_limits(frames, margin=0.12):
    points = np.concatenate([np.concatenate([p for p, _ in frame["faces"]])
                             for frame in frames])
    low, high = points.min(axis=0), points.max(axis=0)
    span = max((high - low).max(), 1e-6)
    centre = (high + low) / 2
    half = span * (0.5 + margin)
    return ((centre[0] - half, centre[0] + half),
            (centre[1] - half, centre[1] + half))


def main():
    parser = argparse.ArgumentParser(description="Animate the held-out rollouts")
    parser.add_argument("-input", default=None, help="directory holding config.yml and agent.zip")
    parser.add_argument("-config", default=None)
    parser.add_argument("-checkpoint", default=None)
    parser.add_argument("-levels", default=None, help="comma-separated subset")
    parser.add_argument("-extra", action="store_true",
                        help="include the extra held-out domains")
    parser.add_argument("-face_desired_degree", default=None, type=int,
                        help="override the meshing target: 4 for quads, 3 for triangles")
    parser.add_argument("-samples", default=0, type=int,
                        help="sampled rollouts to draw from; 0 is a single greedy rollout")
    parser.add_argument("-title", default=None)
    parser.add_argument("-out", default="solve.gif")
    parser.add_argument("-fps", default=1.8, type=float)
    parser.add_argument("-dpi", default=72, type=int)
    parser.add_argument("-columns", default=5, type=int)
    parser.add_argument("-end_pause", default=3.0, type=float,
                        help="seconds to hold the final frame before looping")
    args = parser.parse_args()

    config_fn = args.config or os.path.join(args.input, "config.yml")
    checkpoint = args.checkpoint
    if checkpoint is None:
        for name in ("agent.zip", "best_holdout_model.zip", "bc_model.zip"):
            candidate = os.path.join(args.input, name)
            if os.path.isfile(candidate):
                checkpoint = candidate
                break
    if config_fn is None or checkpoint is None:
        parser.error("need -input <dir>, or both -config and -checkpoint")

    config = load_yaml_config(config_fn)
    model = load_model_from_checkpoint(checkpoint, config_fn)
    env_config = dict(config["environment"])
    if args.face_desired_degree is not None:
        env_config["face_desired_degree"] = args.face_desired_degree
    levels = (args.levels.split(",") if args.levels
              else (ALL_LEVEL_NAMES if args.extra else LEVEL_NAMES))

    rollouts, limits = {}, {}
    for level in levels:
        env = make_level_env(env_config, level)
        frames = rollout_frames(model, env, samples=args.samples)
        rollouts[level] = frames
        limits[level] = level_limits(frames)
        outcome = f"solved in {frames[-1]['moves']}" if frames[-1]["solved"] else "not solved"
        print(f"{level:14s} {len(frames):3d} frames  {outcome}")

    length = max(len(frames) for frames in rollouts.values())
    columns = min(args.columns, len(levels))
    rows = int(np.ceil(len(levels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.7 * columns, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[len(levels):]:
        axis.axis("off")
    figure.patch.set_facecolor("white")

    # Fixed margins rather than tight_layout: the titles change width from frame
    # to frame ("move 7  faces 10  vertices +10" against "solved in 4"), and a
    # layout re-solved per frame moves every panel a little, which reads as the
    # geometry drifting when it is really the axes underneath it.
    figure.subplots_adjust(left=0.015, right=0.985, top=0.895, bottom=0.075,
                           wspace=0.06, hspace=0.30)
    positions = [axis.get_position() for axis in axes]

    solved = sum(rollouts[level][-1]["solved"] for level in levels)
    protocol = f"best of {args.samples} sampled rollouts" if args.samples else "greedy rollouts"
    target = env_config.get("face_desired_degree", 4)
    element = "triangles" if target == 3 else "quads"
    figure.suptitle(args.title or
                    f"One policy asked for {element} — {protocol} on the held-out "
                    f"levels — {solved}/{len(levels)} solved", fontsize=13, y=0.985)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    figure.legend(
        handles=[
            Patch(facecolor=DONE_FILL, edgecolor=EDGE_COLOUR, label=element[:-1]),
            Patch(facecolor=WORK_FILL, edgecolor=EDGE_COLOUR,
                  label="even, wants a chord" if target == 4 else "wants a chord"),
            Patch(facecolor=BAD_FILL, edgecolor=EDGE_COLOUR,
                  label="odd, wants a vertex" if target == 4 else "degenerate"),
            Line2D([], [], color=NEW_EDGE_COLOUR, linewidth=2.6, label="the move just made"),
            Line2D([], [], color=IRREGULAR_VERTEX, marker="o", linestyle="none",
                   markersize=5, label="vertex off its desired degree"),
        ],
        loc="lower center", ncol=5, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 0.0))

    writer = PillowWriter(fps=args.fps)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with writer.saving(figure, args.out, dpi=args.dpi):
        for step in range(length):
            for axis, level, position in zip(axes, levels, positions):
                frames = rollouts[level]
                draw(axis, frames[min(step, len(frames) - 1)], level, limits[level],
                     target=target, position=position)
            writer.grab_frame(facecolor="white")
    length = write_gif(args.out, args.fps, args.end_pause)

    print(f"\nwrote {args.out} ({length} frames, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
