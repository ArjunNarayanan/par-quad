"""Gmsh on the curved Mesh Quest levels, scored by the agent's own rule.

    venv/bin/python utilities/compare_gmsh_curved.py

`compare_gmsh.py` refuses a curved domain: it reads corner wants through
`geometry_loops`, which is straight-edged only. The fix is small -- the wants at
a control point come from the TANGENT angle, which `_loop_arrays` already
reports -- and everything downstream (the element count match, the scoreboard)
is reused unchanged.

Two readings per domain, as in the straight comparison: Gmsh at its own natural
coarseness for the requested size, and Gmsh searched to the element count the
agent produced, since a block decomposition and a general mesh are otherwise not
comparable.
"""

import argparse
import os
import sys
from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())

from utilities.compare_gmsh import (VARIANTS, corner_wants, gmsh_quad_mesh,  # noqa: E402
                                    mesh_at_element_count, score_mesh)


def curved_corner_wants(geometry, target_angle=90.0, tol=1e-6):
    """Corner coordinate -> degrees wanted, with the angles read off TANGENTS."""
    from src.geo2d_bridge import _loop_arrays
    wants = {}
    for index, loop in enumerate(geometry.loops):
        points, angles, _ = _loop_arrays(loop, ccw=True, drop_collinear=True)
        for point, angle in zip(points, angles):
            # material is outside a hole, so its angle is the reflex one
            value = 360.0 - float(angle) if index else float(angle)
            key = (round(float(point[0]) / tol), round(float(point[1]) / tol))
            wants[key] = corner_wants(value, target_angle)
    return wants, tol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-levels", default="")
    parser.add_argument("-variants", default="blossom,frontal-quad,quasi-structured,packing")
    parser.add_argument("-out", default="out/gmsh_curved")
    args = parser.parse_args()

    import yaml
    from envs.environment_maker import initialize_environment
    from src.curved_levels import from_spec, load_domains
    from src.geo2d_bridge import import_geo2d
    geo2d = import_geo2d()

    base = yaml.safe_load(open(
        "experiments/self-play/quad/curved-overfit-v1/config.yml"))["environment"]
    domains = load_domains()
    chosen = [n.strip() for n in args.levels.split(",") if n.strip()] or list(domains)

    print(f"{'level':18} {'par':>4} {'quads':>6}  {'best gmsh variant':>18} "
          f"{'excess':>7} {'face':>5} {'q':>7}")
    for name in chosen:
        spec = domains[name]
        source = spec.get("source", "")
        if "generate(" not in source:
            print(f"{name:18}   -- not a geo2d domain, skipped")
            continue
        seed = int(source.split("generate(")[1].split(",")[0])
        geometry = geo2d.generate(seed, preset="rounded")

        graph, desired = from_spec(spec)
        class Fixed:
            def __init__(self):
                self.n = len(desired)
            def __call__(self):
                return deepcopy(graph), dict(desired)
        cfg = dict(base)
        cfg.pop("initializer", None)
        cfg["graph_initializer"] = Fixed()
        cfg["resample_if_at_par"] = False
        env = initialize_environment(cfg)
        par = env.par

        wants, tol = curved_corner_wants(geometry)
        best = None
        for variant in args.variants.split(","):
            options = VARIANTS[variant.strip()]
            for target in (4, 6, 8, 12, 20, 40):
                # `mesh_at_element_count` hands back (count, lc, nodes, elements)
                found = mesh_at_element_count(geometry, target, options)
                if found is None:
                    continue
                _, _, nodes, elements = found
                result = score_mesh(nodes, elements, wants, tol, par)
                key = (result["face_score"], abs(result["vertex_score"] - par),
                       -result.get("min_quality", 0))
                if best is None or key < best[0]:
                    best = (key, variant.strip(), len(elements), result)
        if best is None:
            print(f"{name:18} {par:4d}      -   gmsh failed on every setting")
            continue
        _, variant, count, result = best
        print(f"{name:18} {par:4d} {count:6d}  {variant:>18} "
              f"{abs(result['vertex_score'] - par):7d} {result['face_score']:5d} "
              f"{result.get('min_quality', float('nan')):7.3f}")


if __name__ == "__main__":
    main()
