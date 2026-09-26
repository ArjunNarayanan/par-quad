"""The baseline comparison must score everyone with the environment's own rule.

`utilities/compare_gmsh.score_mesh` re-implements the vertex defect on a bare
(nodes, elements) mesh, because Gmsh's output never becomes a `Tiler`. A
re-implementation that drifts from `AngleEnv` turns the whole comparison into two
different metrics wearing one name, and the first version of it did exactly that:
it used `rounded_desired_degree` alone, which resolves a rounding tie by numpy's
round-half-to-even instead of keeping both admissible corner treatments, so a
legal 135-degree corner scored as a defect and every method's excess over par came
out inflated.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.getcwd())

geo2d = pytest.importorskip("geo2d", reason="geogen is not installed")

from src.geo2d_bridge import (env_summary, geometry_to_tiler, make_geo2d_env,  # noqa: E402
                              tiler_faces)
from src.utils import load_yaml_config  # noqa: E402
from utilities.compare_gmsh import corner_wants, domain_corner_wants, score_mesh  # noqa: E402

CONFIG = "experiments/self-play/quad/geo2d-holes-push-v1/config-sticky.yml"


def test_corner_wants_keeps_both_treatments_at_a_tie():
    # 135 / 90 = 1.5 and 225 / 90 = 2.5 are exact ties: both corner treatments
    # are geometrically legal, so both degrees have to be admissible
    assert corner_wants(135.0) == (2, 3)
    assert corner_wants(225.0) == (3, 4)
    assert corner_wants(315.0) == (4, 5)
    # and a corner that is not on a tie still resolves to exactly one degree
    assert corner_wants(90.0) == (2,)
    assert corner_wants(180.0) == (3,)
    assert corner_wants(270.0) == (4,)


@pytest.mark.parametrize("preset,n_holes", [("straight", 0), ("straight", (1, 2))])
def test_score_mesh_agrees_with_the_environment(preset, n_holes):
    """The standalone scorer and `AngleEnv` must agree on the same mesh.

    Scored on the start state itself -- no policy needed, so the test does not
    depend on a checkpoint -- and after a handful of scripted edits, so the
    comparison covers meshes the agent actually builds and not just polygons.
    """
    if not os.path.exists(CONFIG):
        pytest.skip("run config not present")
    env_config = load_yaml_config(CONFIG)["environment"]
    checked = 0
    for seed in range(40):
        try:
            geometry = geo2d.generate(seed, preset=preset, n_holes=n_holes)
            geometry_to_tiler(geometry, 90)
        except Exception:
            continue
        env = make_geo2d_env(env_config, geometry, template_size=160)
        observation, _ = env.reset()
        rng = np.random.default_rng(seed)
        for _ in range(rng.integers(3, 12)):
            valid = np.nonzero(np.isfinite(observation["mask"]))[0]
            if not len(valid):
                break
            observation, _, terminated, truncated, _ = env.step(int(rng.choice(valid)))
            if terminated or truncated:
                break

        summary = env_summary(env)
        nodes, elements, _ = tiler_faces(env.graph)
        wants, tol = domain_corner_wants(geometry, normalize=True)
        score = score_mesh(nodes, elements, wants, tol, int(env.par))

        assert score["face_score"] == summary["face_score"], f"seed {seed}: face score"
        assert score["excess_over_par"] == summary["vertex_excess"], f"seed {seed}: vertex excess"
        checked += 1
        if checked >= 8:
            break
    assert checked >= 4, "not enough geometries were generated to make the test meaningful"


def test_the_optimised_slide_survives_a_singular_newton_step():
    """geo2d's `_optimize_node` guards with `det > 0`, which `solve` can reject.

    A tiny positive determinant passes the guard and raises `LinAlgError`, which
    used to take the whole re-smooth with it -- and therefore every scoring run,
    at the end of hours of training. One node falling back to the neighbour mean
    is the right answer; Freitag's acceptance test still refuses a move that
    makes an element worse.
    """
    import numpy as np
    import pytest

    pytest.importorskip("geo2d")
    from src.geo2d_bridge import import_geo2d, resmooth_mesh
    geo2d = import_geo2d()

    # `geo2d.smooth` the FUNCTION shadows `geo2d.smooth` the module, so the
    # module is only reachable through sys.modules -- which is also what
    # `from geo2d.smooth import _optimize_node` resolves against
    import sys as _sys
    smooth = _sys.modules["geo2d.smooth"]
    real = smooth._optimize_node
    calls = {"n": 0}

    def exploding(*args, **kwargs):
        calls["n"] += 1
        raise np.linalg.LinAlgError("Singular matrix")

    # a ring of quads round a square hole, so there are boundary nodes to slide
    nodes = np.array([[0., 0.], [1., 0.], [2., 0.], [2., 1.], [2., 2.],
                      [1., 2.], [0., 2.], [0., 1.],
                      [0.7, 0.7], [1.3, 0.7], [1.3, 1.3], [0.7, 1.3]], dtype=float)
    elements = [[0, 1, 9, 8], [1, 2, 3, 9], [3, 4, 10, 9], [4, 5, 11, 10],
                [5, 6, 7, 11], [7, 0, 8, 11], [8, 9, 10, 11]]
    corners = np.zeros(len(nodes), dtype=bool)
    corners[[0, 2, 4, 6]] = True

    smooth._optimize_node = exploding
    try:
        mesh = resmooth_mesh(geo2d.Mesh(nodes.copy(), elements), corners,
                             iters=2, method="smart", slide="optimize")
    finally:
        smooth._optimize_node = real
    assert calls["n"] > 0, "the fixture never reached the optimised slide"
    assert np.isfinite(mesh.nodes).all()
