"""The ported untangler must agree with geo2d's own, and beat the Laplacian.

`game/js/untangle.js` is a port of `geo2d.smooth.optimize` so the browser can
run the same acceptance test the environment's evaluator does. A Laplacian
cannot open a folded element -- it is already at its fixed point -- so a board
that is topologically perfect can sit unwinnable for a reason the player has no
move against. These pin the port to the original.

Skipped where node or geogen is unavailable.
"""

import json
import os
import shutil
import subprocess
import sys
import unittest

import numpy as np

NODE = shutil.which("node")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNTANGLE = os.path.join(ROOT, "game", "js", "untangle.js")

RUN = """
const u = require(process.argv[1]);
const d = JSON.parse(process.argv[2]);
const out = u.optimize(d.elements, d.coords, d.fixed, {iters: d.iters, newtonIters: 6});
console.log(JSON.stringify({q: u.minQuality(d.elements, out), nodes: out}));
"""


def _js(elements, coords, fixed, iters=8):
    payload = json.dumps({"elements": [[int(v) for v in e] for e in elements],
                          "coords": [[float(x), float(y)] for x, y in coords],
                          "fixed": [int(v) for v in fixed], "iters": iters})
    result = subprocess.run([NODE, "-e", RUN, UNTANGLE, payload],
                            capture_output=True, text=True, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(result.stderr[:400])
    return json.loads(result.stdout)


@unittest.skipIf(NODE is None, "node is not available")
class TestUntanglerPort(unittest.TestCase):
    def test_it_untangles_a_folded_quad(self):
        """The case the Laplacian cannot touch: one node across the diagonal."""
        elements = [[0, 1, 2, 3]]
        coords = [[0, 0], [1, 0], [0.2, 0.2], [0, 1]]
        before = _js(elements, coords, [0, 1, 2, 3], iters=1)["q"]
        self.assertLess(before, 0, "the fixture is supposed to start inverted")
        after = _js(elements, coords, [0, 1, 3], iters=20)["q"]
        self.assertGreater(after, 0.9, "a free node should recover the square")

    def test_a_fully_pinned_mesh_does_not_move(self):
        elements = [[0, 1, 2, 3]]
        coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
        out = _js(elements, coords, [0, 1, 2, 3], iters=5)
        np.testing.assert_allclose(out["nodes"], coords, atol=0)

    def test_it_matches_geo2d_on_the_hole_levels(self):
        try:
            from geo2d.smooth import optimize as reference

            from src.geo2d_bridge import import_geo2d, tiler_faces
            geo2d = import_geo2d()
        except Exception as exc:                       # pragma: no cover
            raise unittest.SkipTest(f"geogen unavailable: {exc}")
        sys.path.insert(0, os.path.join(ROOT, "game"))
        from server import INITIALIZERS, Game

        for level in ("Square hole", "Triforce ring", "Diamond in square"):
            with self.subTest(level=level):
                game = Game(level)
                game.graph.smooth_vertices(num_iter=3)
                nodes, elements, index = tiler_faces(game.graph)
                fixed = [i for v, i in index.items()
                         if game.graph.is_user_defined_vertex(v)]
                python = reference(geo2d.Mesh(np.array(nodes, float), elements),
                                   iters=8, fixed=np.array(fixed), newton_iters=6)
                q_python = float(np.min(python.quality()))
                q_js = float(np.min(geo2d.Mesh(
                    np.array(_js(elements, nodes, fixed)["nodes"]), elements).quality()))
                self.assertAlmostEqual(q_python, q_js, places=3,
                                       msg=f"{level}: python {q_python} vs js {q_js}")


if __name__ == "__main__":
    unittest.main()
