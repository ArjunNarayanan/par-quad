"""Run and evaluate a checkpoint on random geo2d polygons.

The geometries come from the sibling `geogen/geo2d` generator (straight-edged,
hole-free for now -- see `src/geo2d_bridge.py`), each one becomes the start
state of the training env, and the agent plays it greedily and, optionally,
as a best-of-N of sampled rollouts. The win test is the game's: vertex score
at par, every face a quad, min scaled Jacobian at or above the threshold.

Outputs, in `-out`:

    results.csv          one row per geometry (par, corners, both protocols)
    summary.json         solve counts and the run's settings
    geometries.jsonl     the geo2d geometries, so the run is reproducible
    meshes/geom_XX.json  the best final mesh per geometry as a geo2d Mesh
    gallery_greedy.png   the greedy final meshes, coloured as in solve.gif
    gallery_best.png     the best mesh found across all rollouts
    summary.png          solve rate and move count against polygon size
    smoothing.png        Tiler's Laplacian against the geo2d re-smoother, side by side
    solve.gif            (with -gif) the greedy rollouts, animated

Final meshes are re-smoothed with geo2d's untangling optimizer plus sliding of
the inserted boundary nodes along their edge (`-resmooth`, see
`src.geo2d_bridge.resmooth_env`). That happens after the episode, so the
agent's own win test (`*_solved`, `*_quality`, Laplacian coordinates) is
untouched; the re-smoothed quality and the win test re-run on it are reported
alongside (`*_quality_resmoothed`, `*_solved_resmoothed`), and the galleries
and saved meshes show the re-smoothed coordinates.

    venv/bin/python utilities/evaluate_geo2d.py -input models/mesh-quest-15of15 \\
        -seed 0 -n 24 -preset straight -n_samples 16 -template_size 128 -out out/geo2d
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from copy import deepcopy

import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.append(os.getcwd())

from src.geo2d_bridge import (env_summary, import_geo2d, load_model,  # noqa: E402
                              make_geo2d_env, resmooth_env, tiler_to_geo2d_mesh)
from src.utils import load_yaml_config  # noqa: E402
from utilities.animate_holdout import (BAD_FILL, DONE_FILL, EDGE_COLOUR,  # noqa: E402
                                       IRREGULAR_VERTEX, NEW_EDGE_COLOUR, WORK_FILL,
                                       draw, level_limits, snapshot, write_gif)

GREEDY_COLOUR = "#2a78d6"
BEST_COLOUR = "#eb6834"
TEXT_COLOUR = "#52514e"


# --------------------------------------------------------------------------- rollouts

def _batch(obs):
    return {key: np.expand_dims(value, 0) for key, value in obs.items()}


def _state_rank(env):
    """How good the mesh is right now: all-quad first, then close to par."""
    return (env.global_face_score, abs(env.global_vertex_score - env.par))


def rollout(model, env, deterministic, keep_frames=False, keep_best=True):
    """One episode. Returns (summary, frames); frames is empty unless asked for.

    The episode is scored on the BEST state it passed through, not the last one.
    A winning state always ends the episode -- `terminate_at_par` sees to that --
    so this never changes whether a domain counts as solved. What it changes is
    the mesh we keep from a domain that was not solved: the agent regularly walks
    through an all-quad mesh carrying one defect and then wanders off it, and the
    terminal state throws that away. Ranked `(face_score, distance from par)`, so
    an all-quad mesh wins over a closer-to-par one with a stray triangle.
    """
    obs, _ = env.reset()
    frames = [snapshot(env)] if keep_frames else []
    best = None
    if keep_best:
        best = (_state_rank(env), deepcopy(env.graph), dict(env.vertex_desired_degree))
    for _ in range(env.max_steps):
        if env.is_at_par():
            break
        action, _ = model.predict(_batch(obs), deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(int(np.asarray(action).ravel()[0]))
        if keep_frames:
            frames.append(snapshot(env, previous_edges=frames[-1]["edges"]))
        if keep_best and _state_rank(env) < best[0]:
            best = (_state_rank(env), deepcopy(env.graph), dict(env.vertex_desired_degree))
        if terminated or truncated:
            break
    if keep_best and _state_rank(env) > best[0]:
        env.graph = best[1]
        env.vertex_desired_degree = best[2]
        # the angle cache is keyed by half-edge and belongs to the graph we just
        # replaced, so it has to be rebuilt before any score touches it
        env._update_half_edge_angles()
        env._update_scores_on_reset()
    if not keep_frames:
        # a lone final snapshot has no previous frame, so nothing counts as "new"
        frames = [snapshot(env)]
        frames[-1]["new_edges"] = set()
    return env_summary(env), frames


def finish(env, frames, resmooth, resmooth_iters):
    """Re-smooth the final mesh and append it as one more frame.

    Returns (summary_after, frame_after, mesh_after). With `resmooth="none"`
    the final frame is simply the last one and the summary the env's own.
    """
    before = frames[-1]
    if resmooth == "none":
        return env_summary(env), before, tiler_to_geo2d_mesh(env.graph)
    mesh = resmooth_env(env, iters=resmooth_iters, method=resmooth)
    after = snapshot(env, previous_edges=before["edges"])
    after["new_edges"] = set()
    frames.append(after)
    return env_summary(env), after, mesh


def _rank(summary):
    """Smaller is better: solved first, then closest to par, then quality, then short."""
    return (not summary["solved"], summary["face_score"], abs(summary["vertex_excess"]),
            -summary["min_quality"], summary["moves"])


def evaluate_geometry(model, env_config, geometry, index, n_samples, template_size,
                      drop_collinear, keep_frames, resmooth="optimize", resmooth_iters=10):
    env = make_geo2d_env(env_config, geometry, template_size=template_size,
                         drop_collinear=drop_collinear)
    greedy, frames = rollout(model, env, deterministic=True, keep_frames=keep_frames)
    greedy_before = frames[-1]
    greedy_after, greedy_frame, greedy_mesh = finish(env, frames, resmooth, resmooth_iters)

    best, best_after, best_frame, best_mesh, hits = greedy, greedy_after, greedy_frame, greedy_mesh, 0
    for _ in range(n_samples):
        sampled, sampled_frames = rollout(model, env, deterministic=False)
        hits += int(sampled["solved"])
        if _rank(sampled) < _rank(best):
            best = sampled
            best_after, best_frame, best_mesh = finish(env, sampled_frames, resmooth, resmooth_iters)

    row = {
        "index": index,
        "seed": geometry.meta.get("seed", -1),
        "corners": env.graph_initializer.n,
        "raw_corners": geometry.outer.n,
        "par": greedy["par"],
        "max_steps": env.max_steps,
        "greedy_solved": int(greedy["solved"]),
        "greedy_moves": greedy["moves"],
        "greedy_face": greedy["face_score"],
        "greedy_vertex_excess": greedy["vertex_excess"],
        "greedy_quality": round(greedy["min_quality"], 3),
        "greedy_overflow": int(greedy["overflowed"]),
        "greedy_faces": greedy["faces"],
        "greedy_quality_resmoothed": round(greedy_after["min_quality"], 3),
        "greedy_solved_resmoothed": int(greedy_after["solved"]),
        "bestN_solved": int(best["solved"]),
        "bestN_hits": hits,
        "bestN_moves": best["moves"] if best["solved"] else -1,
        "bestN_face": best["face_score"],
        "bestN_vertex_excess": best["vertex_excess"],
        "bestN_quality": round(best["min_quality"], 3),
        "bestN_faces": best["faces"],
        "bestN_quality_resmoothed": round(best_after["min_quality"], 3),
        "bestN_solved_resmoothed": int(best_after["solved"]),
    }
    return row, frames, greedy_before, greedy_frame, best_frame, greedy_mesh, best_mesh


# --------------------------------------------------------------------------- plots

def panel_title(row, protocol):
    return f"#{row['index']}  n={row['corners']}  par={row['par']}"


def _legend(figure, ncol=5):
    figure.legend(
        handles=[
            Patch(facecolor=DONE_FILL, edgecolor=EDGE_COLOUR, label="quad"),
            Patch(facecolor=WORK_FILL, edgecolor=EDGE_COLOUR, label="even, wants a chord"),
            Patch(facecolor=BAD_FILL, edgecolor=EDGE_COLOUR, label="odd, wants a vertex"),
            Line2D([], [], color=NEW_EDGE_COLOUR, linewidth=2.6, label="the move just made"),
            Line2D([], [], color=IRREGULAR_VERTEX, marker="o", linestyle="none",
                   markersize=5, label="vertex off its desired degree"),
        ],
        loc="lower center", ncol=ncol, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))


def gallery(rows, frames, limits, title, path, columns=6, dpi=110):
    """One panel per geometry, drawn in the solve.gif visual language."""
    count = len(rows)
    columns = min(columns, count)
    rows_n = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows_n, columns, figsize=(2.7 * columns, 3.0 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[count:]:
        axis.axis("off")
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(left=0.015, right=0.985, top=1 - 0.55 / (3.0 * rows_n),
                           bottom=0.45 / (3.0 * rows_n), wspace=0.06, hspace=0.30)
    for axis, row, frame in zip(axes, rows, frames):
        draw(axis, frame, panel_title(row, None), limits[row["index"]], target=4)
    figure.suptitle(title, fontsize=13, y=0.995)
    _legend(figure)
    figure.savefig(path, dpi=dpi, facecolor="white")
    plt.close(figure)


def summary_plot(rows, n_samples, title, path, dpi=130):
    """Solve rate per corner-count bin, and moves against corners for the solved ones."""
    corners = np.array([r["corners"] for r in rows])
    greedy = np.array([r["greedy_solved"] for r in rows], dtype=bool)
    best = np.array([r["bestN_solved"] for r in rows], dtype=bool)

    edges = np.arange(corners.min() // 4 * 4, corners.max() + 5, 4)
    bins = np.digitize(corners, edges) - 1
    labels = [f"{edges[i]}–{edges[i + 1] - 1}" for i in range(len(edges) - 1)]
    present = [i for i in range(len(labels)) if (bins == i).any()]

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    figure.patch.set_facecolor("white")
    for axis in (left, right):
        axis.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#c9c8c3")
        axis.tick_params(colors=TEXT_COLOUR, labelsize=9)
        axis.yaxis.grid(True, color="#e6e5e1", linewidth=0.8)
        axis.set_axisbelow(True)

    x = np.arange(len(present))
    width = 0.38
    for offset, mask, colour, label in ((-width / 2, greedy, GREEDY_COLOUR, "greedy"),
                                        (width / 2, best, BEST_COLOUR, f"best of {n_samples}")):
        rates, counts = [], []
        for i in present:
            sel = bins == i
            counts.append(int(sel.sum()))
            rates.append(100.0 * mask[sel].mean())
        bars = left.bar(x + offset, rates, width * 0.94, color=colour, label=label, linewidth=0)
        for bar, rate, count in zip(bars, rates, counts):
            left.text(bar.get_x() + bar.get_width() / 2, rate + 1.5,
                      f"{rate:.0f}%", ha="center", va="bottom", fontsize=8, color=TEXT_COLOUR)
    left.set_xticks(x)
    left.set_xticklabels([f"{labels[i]}\n(n={int((bins == i).sum())})" for i in present], fontsize=8)
    left.set_ylim(0, 112)
    left.set_ylabel("solved (%)", color=TEXT_COLOUR)
    left.set_xlabel("corners in the polygon", color=TEXT_COLOUR)
    left.set_title("Solve rate by polygon size", fontsize=11, color="#0b0b0b", loc="left")
    left.legend(frameon=False, fontsize=9, loc="upper right")

    # an n-gon needs at least (n - 2) / 2 quads; with chords alone that is (n - 4) / 2 moves
    right.plot([corners.min(), corners.max()],
               [(corners.min() - 4) / 2, (corners.max() - 4) / 2],
               color="#c9c8c3", linewidth=1.2, linestyle="--",
               label="fewest possible moves, (n − 4) / 2")
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(rows))
    for mask, moves_key, colour, label, marker in (
            (greedy, "greedy_moves", GREEDY_COLOUR, "greedy, solved", "o"),
            (best & ~greedy, "bestN_moves", BEST_COLOUR, f"only best of {n_samples}", "o")):
        if mask.any():
            right.scatter(corners[mask] + jitter[mask], [rows[i][moves_key] for i in np.where(mask)[0]],
                          s=42, color=colour, edgecolor="white", linewidth=1.2, label=label,
                          marker=marker, zorder=3)
    failed = ~best
    if failed.any():
        right.scatter(corners[failed] + jitter[failed], [rows[i]["greedy_moves"] for i in np.where(failed)[0]],
                      s=42, facecolor="none", edgecolor="#8b97a6", linewidth=1.2,
                      label="not solved (moves used)", zorder=2)
    right.set_xlabel("corners in the polygon", color=TEXT_COLOUR)
    right.set_ylabel("moves", color=TEXT_COLOUR)
    right.set_title("Moves to par", fontsize=11, color="#0b0b0b", loc="left")
    right.legend(frameon=False, fontsize=9, loc="upper left")
    right.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    figure.suptitle(title, fontsize=12, y=1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def smoothing_figure(rows, before, after, limits, title, path, count=6, dpi=110):
    """Tiler's Laplacian (top) against the geo2d re-smoother (bottom), for the
    geometries whose minimum quality changed the most."""
    gain = [abs(r["greedy_quality_resmoothed"] - r["greedy_quality"]) for r in rows]
    picks = sorted(range(len(rows)), key=lambda i: -gain[i])[:min(count, len(rows))]
    picks.sort()
    figure, axes = plt.subplots(2, len(picks), figsize=(2.7 * len(picks), 6.8))
    axes = np.atleast_2d(axes)
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(left=0.015, right=0.985, top=0.86, bottom=0.08, wspace=0.06, hspace=0.35)
    for column, i in enumerate(picks):
        row = rows[i]
        for line, frame, label, quality in (
                (0, before[i], "Tiler Laplacian", row["greedy_quality"]),
                (1, after[i], "geo2d optimize + slide", row["greedy_quality_resmoothed"])):
            axis = axes[line, column]
            draw(axis, frame, f"#{row['index']}  n={row['corners']}", limits[row["index"]], target=4)
            status = axis.get_title().split("\n")[1]
            axis.set_title(f"#{row['index']}  {label}\n{status}\nmin quality {quality:.2f}",
                           fontsize=8.5, color=axis.title.get_color(), pad=4)
    figure.suptitle(title, fontsize=13, y=0.975)
    _legend(figure)
    figure.savefig(path, dpi=dpi, facecolor="white")
    plt.close(figure)


def animate(rows, rollouts, limits, title, path, fps=1.8, dpi=72, columns=6, end_pause=3.0):
    """The greedy rollouts animated together, as `animate_holdout.py` does for the levels."""
    count = len(rows)
    length = max(len(frames) for frames in rollouts)
    columns = min(columns, count)
    rows_n = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows_n, columns, figsize=(2.7 * columns, 3.0 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[count:]:
        axis.axis("off")
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(left=0.015, right=0.985, top=1 - 0.55 / (3.0 * rows_n),
                           bottom=0.45 / (3.0 * rows_n), wspace=0.06, hspace=0.30)
    positions = [axis.get_position() for axis in axes]
    figure.suptitle(title, fontsize=13, y=0.995)
    _legend(figure)

    writer = PillowWriter(fps=fps)
    with writer.saving(figure, path, dpi=dpi):
        for step in range(length):
            for axis, row, frames, position in zip(axes, rows, rollouts, positions):
                draw(axis, frames[min(step, len(frames) - 1)], panel_title(row, None),
                     limits[row["index"]], target=4, position=position)
            writer.grab_frame(facecolor="white")
    plt.close(figure)
    return write_gif(path, fps, end_pause)


# --------------------------------------------------------------------------- main

def _range(text):
    parts = [int(p) for p in text.split(",")]
    return parts[0] if len(parts) == 1 else (parts[0], parts[1])


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on random geo2d polygons")
    parser.add_argument("-input", default=None, help="directory holding config.yml and agent.zip")
    parser.add_argument("-config", default=None)
    parser.add_argument("-checkpoint", default=None)
    parser.add_argument("-seed", default=0, type=int)
    parser.add_argument("-n", default=24, type=int, help="number of geometries")
    parser.add_argument("-preset", default="straight", help="geo2d preset")
    parser.add_argument("-n_holes", default="0", type=_range)
    parser.add_argument("-ratio", default=None, type=int, help="geo2d lattice ratio")
    parser.add_argument("-n_ops", default=None, type=_range, help="e.g. 1,2")
    parser.add_argument("-n_mods", default=None, type=_range, help="e.g. 0,1")
    parser.add_argument("-n_samples", default=16, type=int, help="sampled rollouts for best-of-N")
    parser.add_argument("-template_size", default=None, type=int,
                        help="override the config's half-edge window (the network is size-agnostic)")
    parser.add_argument("-keep_collinear", action="store_true",
                        help="keep 180-degree vertices instead of dropping them")
    parser.add_argument("-gif", action="store_true", help="also animate the greedy rollouts")
    parser.add_argument("-resmooth", default="optimize", choices=["none", "optimize", "smart"],
                        help="geo2d smoother applied to the final mesh, after the episode")
    parser.add_argument("-resmooth_iters", default=10, type=int)
    parser.add_argument("-columns", default=6, type=int)
    parser.add_argument("-title", default=None)
    parser.add_argument("-rng_seed", default=0, type=int,
                        help="seed for the sampled rollouts (torch and numpy)")
    parser.add_argument("-out", default="out/geo2d")
    args = parser.parse_args()
    torch.manual_seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    config_fn = args.config or os.path.join(args.input, "config.yml")
    checkpoint = args.checkpoint or os.path.join(args.input, "agent.zip")
    config = load_yaml_config(config_fn)
    env_config = dict(config["environment"])
    template_size = args.template_size or env_config["template_size"]
    model = load_model(checkpoint, config_fn, template_size=template_size)

    geo2d = import_geo2d()
    overrides = {"n_holes": args.n_holes}
    for key in ("ratio", "n_ops", "n_mods"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    geometries = geo2d.generate_many(args.seed, args.n, preset=args.preset, **overrides)

    os.makedirs(os.path.join(args.out, "meshes"), exist_ok=True)
    geo2d.save_jsonl(geometries, os.path.join(args.out, "geometries.jsonl"))

    rows, rollouts, before_frames, greedy_frames, best_frames, limits = [], [], [], [], [], {}
    header = f"{'#':>3} {'n':>3} {'par':>3} {'greedy':>7} {'moves':>5} {'faces':>5} {'vex':>4} " \
             f"{'q':>5} {'q_re':>5} {'bestN':>6} {'hits':>4} {'moves':>5} {'q_re':>5}"
    print(header)
    print("-" * len(header))
    for index, geometry in enumerate(geometries):
        row, frames, before, greedy_frame, best_frame, greedy_mesh, best_mesh = evaluate_geometry(
            model, env_config, geometry, index, args.n_samples, template_size,
            drop_collinear=not args.keep_collinear, keep_frames=args.gif,
            resmooth=args.resmooth, resmooth_iters=args.resmooth_iters)
        rows.append(row)
        rollouts.append(frames)
        before_frames.append(before)
        greedy_frames.append(greedy_frame)
        best_frames.append(best_frame)
        limits[index] = level_limits(frames + [best_frame])
        best_mesh.save(os.path.join(args.out, "meshes", f"geom_{index:02d}.json"))
        greedy_mesh.save(os.path.join(args.out, "meshes", f"geom_{index:02d}_greedy.json"))
        print(f"{index:3d} {row['corners']:3d} {row['par']:3d} "
              f"{'WIN' if row['greedy_solved'] else 'fail':>7} {row['greedy_moves']:5d} "
              f"{row['greedy_face']:5d} {row['greedy_vertex_excess']:+4d} {row['greedy_quality']:5.2f} "
              f"{row['greedy_quality_resmoothed']:5.2f} "
              f"{'WIN' if row['bestN_solved'] else 'fail':>6} {row['bestN_hits']:4d} "
              f"{row['bestN_moves']:5d} {row['bestN_quality_resmoothed']:5.2f}")

    greedy_total = sum(r["greedy_solved"] for r in rows)
    best_total = sum(r["bestN_solved"] for r in rows)
    print("-" * len(header))
    print(f"greedy {greedy_total}/{len(rows)}   best-of-{args.n_samples} {best_total}/{len(rows)}")
    if args.resmooth != "none":
        re_greedy = sum(r["greedy_solved_resmoothed"] for r in rows)
        re_best = sum(r["bestN_solved_resmoothed"] for r in rows)
        print(f"win test on the re-smoothed meshes ({args.resmooth}): "
              f"greedy {re_greedy}/{len(rows)}   best-of-{args.n_samples} {re_best}/{len(rows)}")

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "greedy": greedy_total, "bestN": best_total, "total": len(rows),
        "n_samples": args.n_samples, "template_size": template_size,
        "preset": args.preset, "seed": args.seed, "overrides": {k: str(v) for k, v in overrides.items()},
        "drop_collinear": not args.keep_collinear, "checkpoint": checkpoint,
        "resmooth": args.resmooth, "resmooth_iters": args.resmooth_iters,
        "greedy_resmoothed": sum(r["greedy_solved_resmoothed"] for r in rows),
        "bestN_resmoothed": sum(r["bestN_solved_resmoothed"] for r in rows),
        "rng_seed": args.rng_seed,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    stem = args.title or f"geo2d '{args.preset}' polygons, seed {args.seed}"
    gallery(rows, greedy_frames, limits,
            f"{stem} — greedy rollouts — {greedy_total}/{len(rows)} solved",
            os.path.join(args.out, "gallery_greedy.png"), columns=args.columns)
    gallery(rows, best_frames, limits,
            f"{stem} — best of greedy + {args.n_samples} sampled rollouts — {best_total}/{len(rows)} solved",
            os.path.join(args.out, "gallery_best.png"), columns=args.columns)
    summary_plot(rows, args.n_samples, stem, os.path.join(args.out, "summary.png"))
    if args.resmooth != "none":
        smoothing_figure(rows, before_frames, greedy_frames, limits,
                         f"{stem} — greedy final meshes: Tiler's Laplacian (top) "
                         f"against geo2d {args.resmooth} + boundary sliding (bottom)",
                         os.path.join(args.out, "smoothing.png"))
    if args.gif:
        animate(rows, rollouts, limits,
                f"{stem} — greedy rollouts — {greedy_total}/{len(rows)} solved",
                os.path.join(args.out, "solve.gif"), columns=args.columns)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
