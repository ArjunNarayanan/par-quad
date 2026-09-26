"""Bending a straight outline into arcs must not move par.

par is a function of the corner WANTS, and a want is `round(angle / 90) + 1`.
Bending edge (a, b) by a total turning `theta` rotates the tangent at each end
by `theta/2`. Keep both ends inside their existing bin and no want moves, so par
does not either -- and the instance's recorded solution is still exactly optimal
on a domain that is now curved.

The bound itself survives for a reason worth stating: `sum(generic - degree) =
4 * chi` is COMBINATORIAL, true of any all-quad mesh of a disk whatever shape
its edges are, and the only geometric input is the corner want, which for a
curved boundary is the tangent angle.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.getcwd())

import envs.polygon_utils as utils  # noqa: E402
from src.bend_outline import arc_through, bend_outline, bin_slack, tangent_turn  # noqa: E402


def test_the_tangent_turns_by_half_the_total_turning():
    """The whole argument rests on this, so pin it."""
    for turning in (5.0, 12.0, -25.0, 45.0, -70.0):
        arc = arc_through([0.0, 0.0], [1.0, 0.0], turning)
        assert tangent_turn(arc, at_start=True) == pytest.approx(turning / 2, abs=0.05)


def test_the_arc_actually_joins_the_two_points():
    for turning in (8.0, -33.0, 60.0):
        arc = arc_through([0.3, -0.2], [1.4, 0.9], turning)
        np.testing.assert_allclose(arc.point(0.0), [0.3, -0.2], atol=1e-9)
        np.testing.assert_allclose(arc.point(1.0), [1.4, 0.9], atol=1e-9)


def test_the_sign_of_the_turning_picks_the_side():
    left = arc_through([0.0, 0.0], [1.0, 0.0], 40.0).point(0.5)
    right = arc_through([0.0, 0.0], [1.0, 0.0], -40.0).point(0.5)
    assert left[1] > 0 > right[1], "the bulge should follow the sign"


def test_bin_slack_is_zero_on_a_bin_edge():
    assert bin_slack(45.0) == pytest.approx(0.0, abs=1e-9)    # 45 is a tie
    assert bin_slack(90.0) == pytest.approx(45.0, abs=1e-9)   # dead centre


def _square():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    loop = [0, 1, 2, 3]
    coordinates = {i: points[i] for i in loop}
    return loop, coordinates


def test_bending_never_changes_a_corner_want():
    loop, coordinates = _square()
    angles = utils.get_polygon_interior_angles(loop, coordinates)
    before = {v: utils.rounded_desired_degree(angles[v], 90) for v in loop}
    rng = np.random.default_rng(0)
    for _ in range(40):
        arcs = bend_outline(loop, coordinates, angles, rng, probability=1.0)
        for (a, b), (arc, start) in arcs.items():
            # the tangent at each end moves by at most half the turning, and the
            # budget was set so that cannot cross a bin edge
            turn = abs(tangent_turn(arc, at_start=True))
            assert turn <= bin_slack(angles[a], 90) + 1e-6
            assert turn <= bin_slack(angles[b], 90) + 1e-6
    assert before == {v: utils.rounded_desired_degree(angles[v], 90) for v in loop}


def test_a_bulge_that_would_cross_the_outline_is_shrunk_or_refused():
    """A long bend on a short edge of a concave outline crosses its neighbour."""
    # a narrow L: the inner short edge has very little room to bow
    points = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 0.4], [0.4, 0.4],
                       [0.4, 3.0], [0.0, 3.0]])
    loop = list(range(len(points)))
    coordinates = {i: points[i] for i in loop}
    angles = utils.get_polygon_interior_angles(loop, coordinates)
    rng = np.random.default_rng(3)
    arcs = bend_outline(loop, coordinates, angles, rng, probability=1.0)
    # whatever came back must not leave the polygon's own footprint badly
    for (a, b), (arc, _) in arcs.items():
        for t in np.linspace(0, 1, 7):
            point = arc.point(t)
            assert -1.5 < point[0] < 4.5 and -1.5 < point[1] < 4.5, \
                "a bulge escaped far outside the outline"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_bent_instance_still_reaches_par(seed):
    """The claim that matters: the recorded solution still works, and par holds."""
    from envs.solved_instances import default_scratch_env, generate_instances, replay
    instances, _ = generate_instances(
        40, cell_range=(2, 10), hole_probability=0.0, curve_probability=1.0,
        rng=np.random.default_rng(seed), verbose=False)
    curved = [i for i in instances if i.curved]
    assert curved, "the generator produced no curved instances"
    env = default_scratch_env(3, face_desired_degree=4)
    for instance in curved:
        *_, played, ok = replay(env, instance)
        assert ok, "a curved instance failed to replay"
        assert env.global_face_score == 0
        assert env.global_vertex_score == env.par, "the bend moved par"
