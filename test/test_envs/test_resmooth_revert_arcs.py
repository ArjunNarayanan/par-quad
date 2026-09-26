"""`resmooth_env` must restore arc spans when it reverts a worse result.

The snap after the untangler re-anchors every arc span to where its vertices
now sit. When the guard then finds the result worse and restores the old
coordinates, the spans have to go back too, or the kept mesh reads its tangents
off spans ending where the REJECTED vertices were -- its quality is then
misreported (measured: -0.11 read as -0.93) and every later want and tangent is
computed at the wrong point.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.append(os.getcwd())

try:
    from src.geo2d_bridge import import_geo2d
    geo2d = import_geo2d()
except ImportError:
    geo2d = None


def _stale(graph):
    count = 0
    for (a, b), (arc, _) in graph.boundary_arcs.items():
        for v in (a, b):
            p = np.asarray(graph.vertex_coordinate(v), float)
            if np.linalg.norm(np.asarray(arc.project(p)) - p) > 1e-9:
                count += 1
    return count


@unittest.skipIf(geo2d is None, "geogen/geo2d not available")
class TestRevertRestoresArcs(unittest.TestCase):
    def test_revert_keeps_spans_consistent(self):
        import src.geo2d_bridge as bridge
        from src.curved_levels import from_spec, load_domains
        from src.surgery import domain_env

        graph, desired = from_spec(load_domains()["Fillet plate"])
        bridge.densify_arcs(graph, desired, 45.0)
        config = {"name": "EvaluatorAngleEnv", "template_size": 64, "max_edge_addition_steps": 3,
                  "quality_metric": "shape", "face_desired_degree": 4}
        env = domain_env(config, graph, desired)
        env.reset()
        g = env.graph
        # one boundary vertex strictly inside an arc (a densified joint)
        (a, b), (arc, _) = next(iter(g.boundary_arcs.items()))
        joint = b if not g.is_user_defined_vertex(b) else a
        self.assertEqual(_stale(g), 0)

        real = bridge.resmooth_mesh

        def slid(mesh, corners, **kwargs):
            # slide the joint a little along its own circle: the snap re-anchors
            # both spans at it
            out = mesh.copy()
            _, _, index = bridge.tiler_faces(g)
            i = index[joint]
            offset = out.nodes[i] - arc.centre
            angle = np.arctan2(offset[1], offset[0]) + 0.05 * np.sign(arc._sweep())
            out.nodes[i] = arc.centre + arc.radius * np.array([np.cos(angle), np.sin(angle)])
            return out

        # the guard sees the slid mesh as WORSE, so it must revert
        readings = iter([1.0, 0.0])
        real_quality = type(env).min_element_quality

        def quality(self):
            try:
                return next(readings)
            except StopIteration:
                return real_quality(self)

        with mock.patch.object(bridge, "resmooth_mesh", side_effect=slid), \
                mock.patch.object(type(env), "min_element_quality", quality):
            bridge.resmooth_env(env, warm_start=0)
        self.assertEqual(_stale(env.graph), 0)


if __name__ == "__main__":
    unittest.main()
