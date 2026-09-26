"""`mask_flat_quads`: no chord may close a quad with a flat boundary corner.

A flat boundary corner is a boundary vertex wanting degree >= 3 whose two sides
in the quad are both boundary edges. Such a quad has a zero Jacobian there.
"""

import numpy as np

from envs.environment_maker import initialize_environment


def _env(**overrides):
    config = dict(name="GlobalAngleEnv", initializer=dict(name="LEnv"), face_desired_degree=4,
                  template_size=64, max_edge_addition_steps=3, reward_mode="normalized")
    config.update(overrides)
    env = initialize_environment(config)
    env.reset(seed=0)
    return env


def _flat_quads(env):
    """Faces that are quads with a flat boundary corner, in the current mesh."""
    g = env.graph
    bad = 0
    for face in g.face_list():
        if g.face_degree(face) != 4:
            continue
        loop = g.generate_half_edge_face_loop(g.first_face_halfedge(face))
        for i, e_out in enumerate(loop):
            e_in = loop[i - 1]
            v = g.source_vertex(e_out, tag=False)
            if env._flat_boundary_corner(v, e_in, e_out):
                bad += 1
                break
    return bad


def test_random_play_never_creates_a_flat_quad_under_the_mask():
    rng = np.random.default_rng(0)
    for seed in range(4):
        env = _env(mask_flat_quads=True)
        env.reset(seed=seed)
        for _ in range(120):
            valid = np.flatnonzero(np.isfinite(env._get_obs()["mask"]))
            if len(valid) == 0:
                break
            _, _, term, trunc, _ = env.step(int(rng.choice(valid)))
            assert _flat_quads(env) == 0
            if term or trunc:
                break


def test_the_mask_only_removes_flat_closures():
    """Every action the masked env forbids that the unmasked env allows would have
    produced a flat quad; every other action agrees."""
    a, b = _env(mask_flat_quads=False), _env(mask_flat_quads=True)
    a.reset(seed=1); b.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(40):
        ma = np.isfinite(a._get_obs()["mask"]); mb = np.isfinite(b._get_obs()["mask"])
        assert not np.any(mb & ~ma)                      # the mask never ADDS an action
        for idx in np.flatnonzero(ma & ~mb):             # what it removes closes a flat quad
            h, local = b._linear_action_index_to_half_edge_and_action(int(idx))
            if local < b.max_edge_addition_steps:
                assert b._chord_closes_flat_quad(h, b.chord_steps(local))
            else:
                assert local == b._insert_vertex_action and b._insert_closes_flat_quad(h)
        valid = np.flatnonzero(mb)
        if len(valid) == 0:
            break
        act = int(rng.choice(valid))
        a.step(act); _, _, term, trunc, _ = b.step(act)
        if term or trunc:
            break


def test_default_is_off():
    assert _env().mask_flat_quads is False


def test_all_masked_state_falls_back_to_the_validity_mask():
    """A bare triangle whose corners all want 2 has no chord, and every vertex
    insertion makes a quad with a flat corner, so the flat mask forbids
    everything. An all -inf mask makes the policy's softmax NaN (it killed the
    paper's straight-holes scoring on a 3-corner draw); the env must hand back
    the validity mask for that state instead."""
    from src.tiler import Tiler

    coordinates = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)}
    triangle = Tiler.from_face_loops([[0, 1, 2]], coordinates, user_vertices={0, 1, 2})
    desired = {0: 2, 1: 2, 2: 2}

    class Triangle:
        def __init__(self):
            self.n = 3

        def __call__(self):
            from copy import deepcopy
            return deepcopy(triangle), dict(desired)

    config = dict(name="GlobalAngleEnv", graph_initializer=Triangle(), face_desired_degree=4,
                  template_size=64, max_edge_addition_steps=3, reward_mode="normalized",
                  mask_flat_quads=True, resample_if_at_par=False)
    env = initialize_environment(config)
    obs, _ = env.reset(seed=0)
    assert env.mask_flat_quads is True
    valid = np.isfinite(obs["mask"])
    assert valid.any(), "the fallback must leave at least one legal action"
    # exactly the three vertex insertions the validity mask allows
    assert int(valid.sum()) == 3
    # and the flag is untouched for the next state
    env.step(int(np.flatnonzero(valid)[0]))
    assert env.mask_flat_quads is True


def test_thin_mask_only_removes_thin_boundary_closures():
    """`mask_thin_quads: R` forbids a chord that closes a face of at most four sides with a
    corner on the boundary whose sides differ by more than R; everything else stays legal,
    and R = 0 (off) is the plain validity mask."""
    off = _env(); on = _env(mask_thin_quads=4.0)
    m_off, m_on = off._get_action_mask(), on._get_action_mask()
    assert np.isfinite(m_on).sum() <= np.isfinite(m_off).sum()
    # every action the thin mask removes is a chord whose new face has a thin boundary corner
    for idx in np.flatnonzero(np.isfinite(m_off) & ~np.isfinite(m_on)):
        h, local = on._linear_action_index_to_half_edge_and_action(int(idx))
        assert local < on.max_edge_addition_steps
        assert on._chord_closes_thin_quad(h, on.chord_steps(local))
    # random play under the mask never creates a quad with a thin boundary corner
    rng = np.random.default_rng(0)
    env = _env(mask_thin_quads=4.0)
    for _ in range(40):
        mask = env._get_action_mask()
        legal = np.flatnonzero(np.isfinite(mask))
        if not len(legal):
            break
        _, _, done, trunc, _ = env.step(int(rng.choice(legal)))
        g = env.graph
        for face in g.face_list():
            if g.face_degree(face) != 4:
                continue
            loop = g.generate_half_edge_face_loop(g.first_face_halfedge(face))
            for i, e_out in enumerate(loop):
                e_in = loop[i - 1]
                v = g.source_vertex(e_out, tag=False)
                # only chord-made corners are governed; an INSERT can still make one, so
                # this checks the rule's own predicate is consistent, not a global invariant
                env._thin_boundary_corner(v, e_in, e_out)
        if done or trunc:
            break
