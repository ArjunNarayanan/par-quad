"""The geo2d bridge: random geo2d polygons must build through the training env.

Skipped when the `geogen` package is not installed (and no sibling checkout
is found). The checks guard
the translation, not the agent: orientation, collinear-vertex removal, the
desired degrees against geo2d's own vertex angles, the env's par, and the
round trip back into a `geo2d.Mesh`.
"""

import unittest

import numpy as np

try:
    from src.geo2d_bridge import (Geo2DPolygon, import_geo2d, make_geo2d_env, polygon_points,
                                  resmooth_env, tiler_to_geo2d_mesh)
    geo2d = import_geo2d()
except ImportError:  # pragma: no cover - geogen is a separate, optional install
    geo2d = None

ENV_CONFIG = {
    "name": "AngleEnvWithLength",
    "face_desired_degree": 4,
    "template_size": 64,
    "max_edge_addition_steps": 3,
    "max_steps_factor": 2.0,
    "smooth_iterations": 5,
}


@unittest.skipIf(geo2d is None, "geogen/geo2d not available")
class TestGeo2DBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometries = geo2d.generate_many(0, 12, preset="straight", n_holes=0)

    def test_points_are_ccw_and_normalized(self):
        for geometry in self.geometries:
            points = polygon_points(geometry)
            x, y = points[:, 0], points[:, 1]
            area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
            self.assertGreater(area, 0)
            self.assertAlmostEqual(np.abs(points).max(), 1.0)

    def test_collinear_vertices_are_dropped(self):
        for geometry in self.geometries:
            kept = polygon_points(geometry, drop_collinear=True)
            raw = polygon_points(geometry, drop_collinear=False)
            flat = np.sum(np.abs(geometry.outer.interior_angles() - 180.0) < 1e-6)
            self.assertEqual(len(raw) - len(kept), flat)

    def test_desired_degrees_match_geo2d_angles(self):
        for geometry in self.geometries:
            initializer = Geo2DPolygon(geometry, target_angle=90, drop_collinear=False)
            angles = geometry.outer.interior_angles()
            for vertex, angle in enumerate(angles):
                expected = max(int(np.round(angle / 90.0)) + 1, 2)
                self.assertEqual(initializer.desired_degree[vertex], expected)

    def test_env_builds_with_par_zero_for_lattice_polygons(self):
        # every corner of a 45/90-degree polygon is on a bin or a tie, so the
        # rounding error that par measures vanishes
        for geometry in self.geometries:
            env = make_geo2d_env(ENV_CONFIG, geometry)
            obs, _ = env.reset()
            self.assertEqual(obs["features"].shape, (64, 10))
            self.assertEqual(env.par, 0)
            self.assertFalse(env.is_at_par())
            self.assertEqual(len(env.graph.face_list()), 1)

    def test_template_override(self):
        env = make_geo2d_env(ENV_CONFIG, self.geometries[0], template_size=128)
        obs, _ = env.reset()
        self.assertEqual(obs["features"].shape, (128, 10))
        self.assertEqual(obs["mask"].shape, (128 * 4,))

    def test_round_trip_to_geo2d_mesh(self):
        env = make_geo2d_env(ENV_CONFIG, self.geometries[1])
        env.reset()
        # take the first valid action so the mesh has more than one face
        action = int(np.flatnonzero(np.isfinite(env._get_obs()["mask"]))[0])
        env.step(action)
        mesh = tiler_to_geo2d_mesh(env.graph)
        self.assertEqual(mesh.nodes.shape[0], len(env.graph.vertex_list()))
        self.assertEqual(len(mesh.quality()), len(env.graph.face_list()))
        self.assertEqual(sorted(mesh.degrees()), sorted(
            env.graph.vertex_degree(v) for v in env.graph.vertex_list(tag=False)))

    def test_resmooth_keeps_corners_and_boundary(self):
        rng = np.random.default_rng(0)
        env = make_geo2d_env(ENV_CONFIG, self.geometries[2])
        obs, _ = env.reset()
        for _ in range(8):
            valid = np.flatnonzero(np.isfinite(obs["mask"]))
            obs, _, terminated, _, _ = env.step(int(rng.choice(valid)))
            if terminated:
                break
        graph = env.graph
        corners = {v: graph.vertex_coordinate(v).copy()
                   for v in graph.vertex_list(tag=False) if graph.is_user_defined_vertex(v)}
        points = env.graph_initializer.points
        polygon = geo2d.Geometry([geo2d.Loop(points)])
        before = env.min_element_quality()
        resmooth_env(env, iters=5)
        for v, coordinate in corners.items():
            np.testing.assert_allclose(graph.vertex_coordinate(v), coordinate)
        for v in graph.vertex_list(tag=False):
            if graph.is_boundary_vertex(v):
                distance = np.linalg.norm(polygon.project(graph.vertex_coordinate(v))[3]
                                          - graph.vertex_coordinate(v))
                self.assertLess(distance, 1e-9)
        # a resmoothed mesh is reported through the env's own metric
        self.assertGreaterEqual(env.min_element_quality(), min(before, 0.0) - 1e-9)

    def test_rejects_holes_and_arcs(self):
        with_hole = geo2d.generate_many(3, 8, preset="polycube", n_holes=(1, 1))
        holed = [g for g in with_hole if len(g.loops) > 1]
        if holed:
            with self.assertRaises(NotImplementedError):
                polygon_points(holed[0])
        curved = [g for g in geo2d.generate_many(3, 8, preset="rounded", n_holes=0)
                  if "arc" in g.outer.kinds()]
        if curved:
            with self.assertRaises(NotImplementedError):
                polygon_points(curved[0])


if __name__ == "__main__":
    unittest.main()
