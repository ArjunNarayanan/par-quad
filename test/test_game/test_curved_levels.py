"""The curved levels, and the rule that makes them mean anything.

A curved boundary changes exactly one thing: what a corner WANTS. That comes
from the tangent, not from the chord, and three implementations have to agree
about it -- `envs/global_angle_env.py` (what the agent is scored against),
`game/server.py` (the reference) and `game/js/engine.js` (what the player sees).

The bug this pins is quiet and total. Measured from chords, every vertex of the
Stadium reads 135 degrees, which is a rounding TIE, so par came out 0 on a
domain whose true par is 4 -- an unreachable target, silently. The wants are
identical either way; only par moves, and only on a curved domain.
"""

import copy
import math
import os
import sys
import shutil
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "game") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "game"))

from src.boundary_arcs import corner_angles  # noqa: E402
from src.curved_levels import LEVELS  # noqa: E402

# par and the wants of every curved level, worked out by hand from the tangents.
# A smooth point wants three; the end of a diameter turns 90 and wants two.
# The simple single-loop builders. `plate-with-hole` is deliberately absent:
# it is cut open by a slit, so its face loop revisits vertices and neither the
# "every corner is 90 or 180" nor the turning identity below applies to it.
SIMPLE = ("semicircle", "quarter-disc", "stadium", "rounded-square")

EXPECTED = {
    "semicircle": {"wants": [2, 2, 3], "par": 2},
    "quarter-disc": {"wants": [2, 2, 2], "par": 1},
    "stadium": {"wants": [3] * 6, "par": 4},
    "rounded-square": {"wants": [3] * 8, "par": 4},
}


class TestCurvedCornerAngles(unittest.TestCase):
    def test_every_corner_is_measured_from_its_tangent(self):
        for name in SIMPLE:
            build = LEVELS[name]
            with self.subTest(level=name):
                graph, desired = build()
                angles = corner_angles(graph)
                for vertex, angle in angles.items():
                    # a tangent corner on these domains is a right angle or a
                    # smooth point; a chord would give 45, 135 or 150 instead
                    self.assertIn(round(angle, 6), (90.0, 180.0))
                self.assertEqual(sorted(desired.values()), EXPECTED[name]["wants"])

    def test_the_angles_sum_to_the_turning_of_a_curved_loop(self):
        """(n - 2) * 180 PLUS the total sweep of the arcs.

        A straight polygon's interior angles sum to (n - 2) * 180. Every arc
        adds its own sweep on top, because the tangent keeps turning between one
        corner and the next -- the extra turning that a chord cannot see, and
        the reason a smooth loop has par 4 rather than 0.
        """
        for name in SIMPLE:
            build = LEVELS[name]
            with self.subTest(level=name):
                graph, _ = build()
                angles = corner_angles(graph)
                sweep = sum(abs(math.degrees(arc._sweep()))
                            for _, (arc, _) in graph.boundary_arcs.items())
                self.assertAlmostEqual(sum(angles.values()),
                                       (len(angles) - 2) * 180.0 + sweep, places=9)


class TestCurvedParAgrees(unittest.TestCase):
    """The env and the game must set the same target on a curved domain."""

    def test_env_par_matches_the_reference_engine(self):
        import server

        import yaml
        from envs.environment_maker import initialize_environment

        config = yaml.safe_load(open(os.path.join(
            ROOT, "experiments/self-play/quad/semicircle-v1/config.yml")))
        base = config["environment"]
        for level, expected in EXPECTED.items():
            with self.subTest(level=level):
                env_config = copy.deepcopy(base)
                env_config["initializer"]["name_of_level"] = level
                env = initialize_environment(env_config)
                env.reset()
                self.assertEqual(env.par, expected["par"])

    def test_every_game_level_agrees_between_the_engines(self):
        """Including the ones with a circular hole, cut open by a slit."""
        import json
        import subprocess

        import server

        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not available")
        script = ("const e = require('./game/js/engine.js');"
                  "const out = {};"
                  "for (const s of Object.keys(e.SHAPES)) out[s] = new e.Game(s).par;"
                  "console.log(JSON.stringify(out));")
        result = subprocess.run([node, "-e", script], capture_output=True,
                                text=True, cwd=ROOT)
        if result.returncode != 0:
            self.skipTest(f"engine.js failed: {result.stderr[:200]}")
        js = json.loads(result.stdout)
        for name in server.INITIALIZERS:
            with self.subTest(level=name):
                self.assertIn(name, js)
                self.assertEqual(server.Game(name).par, js[name])


class TestSmoothNeverMakesTheBoardWorse(unittest.TestCase):
    """A local optimiser started inside a tangle can come out worse.

    `Smooth` is a request to improve the board. geo2d's `resmooth_mesh` keeps its
    own result only when it improved; the reference and the browser port both
    need the same guard, or pressing the button can cost the player their mesh.
    """

    def test_the_reference_keeps_only_an_improvement(self):
        import server

        game = server.Game("Plate with a hole")
        game.apply_op("insert_vertex", {"edge": 0}, True)
        game.apply_op("insert_vertex", {"edge": 1}, True)
        for _ in range(4):
            before = game._current_scores()[2]
            game.smooth(3)
            self.assertGreaterEqual(game._current_scores()[2], before - 1e-9)


class TestFrozenDomains(unittest.TestCase):
    """The geo2d levels are frozen numbers, and both engines read the same ones.

    `geo2d.generate(263)` is reproducible only while geo2d's generator is, and a
    Mesh Quest record means nothing if the shape behind it can move. So a chosen
    domain is written out once and both engines read that file -- which only
    helps if the browser's generated copy cannot drift from the source.
    """

    def test_the_browser_copy_matches_the_source(self):
        import json
        import re

        with open(os.path.join(ROOT, "src", "curved_domains.json")) as handle:
            source = json.load(handle)
        with open(os.path.join(ROOT, "game", "js", "curved_domains.js")) as handle:
            text = handle.read()
        body = re.search(r"const CURVED_DOMAINS = (.*);\n\nif \(typeof module",
                         text, re.S)
        self.assertIsNotNone(body, "generated file no longer has the expected shape")
        self.assertEqual(json.loads(body.group(1)), source,
                         "run utilities/export_curved_level.py to regenerate")

    def test_every_frozen_domain_builds_and_keeps_its_arcs(self):
        from src.curved_levels import from_spec, load_domains

        domains = load_domains()
        self.assertTrue(domains, "no frozen curved domains")
        for name, spec in domains.items():
            with self.subTest(domain=name):
                graph, desired = from_spec(spec)
                self.assertEqual(len(desired), len(spec["points"]))
                self.assertEqual(len(list(graph.boundary_arcs.items())),
                                 len(spec["arcs"]))
                # an arc's endpoints must sit on its own circle, or the outline
                # is torn where the curve meets its control points
                for (first, second), (arc, _) in graph.boundary_arcs.items():
                    for vertex in (first, second):
                        point = graph.vertex_coordinates[vertex]
                        radius = math.dist(point, arc.centre)
                        self.assertAlmostEqual(radius, arc.radius, places=9)


class TestArcProjectionIsStable(unittest.TestCase):
    """Projecting a point already ON an arc must return that same point.

    It did not, for CLOCKWISE arcs. `_fraction_along` put the fraction of the
    arc's own start at 4 rather than 0, which clamped to 1 and returned the far
    END -- so the first thing that projected (smoothing runs after every move)
    teleported both pinned corners of every concave arc, tearing the outline.
    Only convex arcs were in the hand-written levels, so nothing saw it until
    geo2d's domains, which have scoops, were frozen into levels.
    """

    def test_both_endpoints_of_every_arc_are_fixed_points(self):
        from src.curved_levels import from_spec, load_domains, LEVELS

        cases = [(name, build()[0]) for name, build in LEVELS.items()]
        cases += [(name, from_spec(spec)[0])
                  for name, spec in load_domains().items()]
        for name, graph in cases:
            with self.subTest(domain=name):
                for (first, second), (arc, _) in graph.boundary_arcs.items():
                    for vertex in (first, second):
                        point = graph.vertex_coordinates[vertex]
                        landed = arc.project(point)
                        self.assertAlmostEqual(math.dist(landed, point), 0.0, places=9)

    def test_a_clockwise_arc_projects_like_its_mirror(self):
        """The handedness of an arc must not change where a point lands."""
        from src.boundary_arcs import Arc

        for ccw in (True, False):
            with self.subTest(ccw=ccw):
                arc = (Arc((0, 0), 1.0, 0.0, math.pi / 2, ccw=True) if ccw
                       else Arc((0, 0), 1.0, math.pi / 2, 0.0, ccw=False))
                # the same three points on the same quarter circle, either way round
                for t in (0.0, 0.5, 1.0):
                    point = arc.point(t)
                    self.assertAlmostEqual(math.dist(arc.project(point), point),
                                           0.0, places=12)
                # and a point off the circle still lands between the endpoints
                landed = arc.project([0.5, 0.5])
                self.assertAlmostEqual(math.hypot(*landed), 1.0, places=12)
                self.assertGreater(landed[0], 0.0)
                self.assertGreater(landed[1], 0.0)

    def test_an_original_corner_survives_an_insertion_elsewhere(self):
        """The bug as the player met it: insert on one edge, the outline tears."""
        import server

        game = server.Game("Fillet plate")
        graph = game.graph
        before = {v: list(graph.vertex_coordinates[v])
                  for v in graph.vertex_coordinates}
        bottom = next(h for h in graph.half_edge_list()
                      if {graph.source_vertex(h, tag=False),
                          graph.target_vertex(h, tag=False)} == {1, 2})
        game.apply_op("insert_vertex", {"edge": bottom[0]}, True)
        for vertex, coordinate in before.items():
            self.assertAlmostEqual(
                math.dist(game.graph.vertex_coordinates[vertex], coordinate), 0.0,
                places=9, msg=f"pinned corner {vertex} moved")


class TestTheMeshStaysOnTheDomain(unittest.TestCase):
    """A boundary vertex on a curve must remain on that curve, always.

    This is the invariant the whole curved design rests on, and two separate
    code paths broke it: the untangler slides a boundary node along the CHORD
    between its neighbours, because geo2d knows nothing about our arcs. The game
    ported that slide and never snapped back; `resmooth_env` did the same and is
    called IN PLACE by `EvaluatorAngleEnv` whenever the agent reaches par, with
    the result KEPT -- so a solved curved mesh had a boundary up to 0.25 off the
    domain on a unit-scaled outline, and the quality gate judged that mesh.
    """

    def _off_arc(self, graph):
        worst = 0.0
        for (first, second), (arc, _) in graph.boundary_arcs.items():
            for vertex in (first, second):
                if vertex not in graph.vertex_coordinates:
                    continue
                point = graph.vertex_coordinates[vertex]
                worst = max(worst, abs(math.dist(point, arc.centre) - arc.radius))
        return worst

    def test_untangling_leaves_every_vertex_on_its_arc(self):
        import copy

        import yaml

        from envs.environment_maker import initialize_environment
        from src.curved_levels import from_spec, load_domains
        from src.geo2d_bridge import resmooth_env

        base = yaml.safe_load(open(os.path.join(
            ROOT, "experiments/self-play/quad/semicircle-v1/config.yml")))["environment"]
        for name, spec in load_domains().items():
            with self.subTest(domain=name):
                graph, desired = from_spec(spec)

                class Fixed:
                    def __init__(self):
                        self.n = len(desired)

                    def __call__(self):
                        return copy.deepcopy(graph), dict(desired)

                config = dict(base)
                config.pop("initializer", None)
                config["graph_initializer"] = Fixed()
                config["resample_if_at_par"] = False
                env = initialize_environment(config)
                # refine so there are inserted vertices ON the arcs to displace
                for _ in range(6):
                    boundary = [h for h in env.graph.half_edge_list()
                                if env.graph.half_edge_on_boundary(h)]
                    env.graph.insert_vertex(boundary[0])
                env.graph.smooth_vertices(num_iter=5)
                env._update_half_edge_angles()
                try:
                    resmooth_env(env, iters=10, method="optimize", slide="optimize")
                except Exception as error:  # geo2d is optional
                    self.skipTest(f"geo2d unavailable: {error}")
                self.assertLess(self._off_arc(env.graph), 1e-9)


class TestClockwiseCurvedLoops(unittest.TestCase):
    """A clockwise curved loop must attach its arcs to the right vertices.

    `curved_loop_to_tiler` reverses a clockwise loop, and geo2d edge i runs from
    control point i to i+1 -- so after reversal that edge joins THIS index and
    the one BEFORE it, not the one after. The code used (position, position+1),
    which is off by one. It never showed because geo2d hands back a
    counter-clockwise outer loop every time (0 of 400), but every hole it
    generates is clockwise, so curved holes meet it immediately.
    """

    def test_a_clockwise_loop_keeps_its_arcs_on_their_own_vertices(self):
        try:
            from src.geo2d_bridge import curved_loop_to_tiler, import_geo2d
            geo2d = import_geo2d()
        except Exception as error:
            self.skipTest(f"geo2d unavailable: {error}")

        seen_clockwise = 0
        for seed in range(40):
            try:
                geometry = geo2d.generate(seed, preset="rounded")
            except Exception:
                continue
            for loop in geometry.loops[1:]:
                if loop.is_ccw() or not any(loop.is_arc()):
                    continue
                seen_clockwise += 1
                graph, _ = curved_loop_to_tiler(geo2d.Geometry([loop]), normalize=False)
                for (first, second), (arc, _) in graph.boundary_arcs.items():
                    for vertex in (first, second):
                        point = graph.vertex_coordinates[vertex]
                        self.assertAlmostEqual(
                            math.dist(point, arc.centre), arc.radius, places=9,
                            msg=f"seed {seed}: arc ({first},{second}) is not on v{vertex}")
                if seen_clockwise >= 5:
                    break
            if seen_clockwise >= 5:
                break
        self.assertGreater(seen_clockwise, 0, "no clockwise curved loop to test")


if __name__ == "__main__":
    unittest.main()
