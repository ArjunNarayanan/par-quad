"""geo2d-lite: a stand-in for the private `geogen/geo2d` package (see generate.py)."""

from . import export, smooth  # noqa: F401
from .generate import PRESETS, generate, generate_many  # noqa: F401
from .geometry import Arc, Geometry, Line, Loop  # noqa: F401
from .mesh import Mesh  # noqa: F401
from .smooth import optimize, smart_laplacian  # noqa: F401

LITE = True


def save_jsonl(geometries, path):
    import json
    with open(path, "w") as handle:
        for g in geometries:
            handle.write(json.dumps({"loops": [
                {"points": l.points.tolist(),
                 "arcs": [None if e.kind == "line" else
                          {"center": e.center.tolist(), "ccw": e.ccw} for e in l.edges]}
                for l in g.loops]}) + "\n")
