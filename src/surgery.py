"""Surgery-and-continue on curved domains: regularity first, then shape.

The straight-trained agent reaches meshes close to par on curved domains; what it
cannot see is that a topology which is fine on a polygon can be badly drawn when
one side of an element is an arc. The hypothesis this module tests is that the
bad drawing is an INTERMEDIATE state, not a dead end: from a regular but badly
shaped mesh there are downstream states that are both regular and well shaped,
and the policy can find them if it is handed the right broken state.

So the loop is

    candidates  <- the agent's own rollouts (every all-quad state it passes)
    repeat:
        seed    <- the most regular candidates whose shape is below the bar
        face    <- the worst element of the seed, after untangling
        surgery <- one local edit at that element
                   * `sheet`   split the quad sheet through it (deterministic;
                                all-quad in, all-quad out, irregularity EXACTLY
                                unchanged -- every new vertex lands on its want)
                   * `insert`  a vertex on one of its sides, then the agent
                   * `chord`   a diagonal of it, then the agent
        candidates += untangled results

and the answer is chosen REGULARITY FIRST among usable meshes: the fewest
irregular vertices among meshes whose minimum shape quality clears the bar, and
only then quality. `utilities/score_quality_objective.py` chooses quality first,
which is why its curved rows sit at 0.2 irregular per vertex however well the
repair works: it spends regularity to buy shape whenever it can.

Irregularity is `sum_v |deg(v) - want(v)|` (the env's vertex score); per vertex
it is that over the vertex count. Wants are the domain's own for its original
vertices and 3 on the boundary / 4 inside for every vertex added since, which is
also what the env registers on insertion.
"""

from copy import deepcopy

import numpy as np

from src.geo2d_bridge import resmooth_env


# ----------------------------------------------------------------------------- env glue

def domain_env(env_config, graph, desired, max_steps_factor=None):
    """An env pinned to one domain."""
    from envs.environment_maker import initialize_environment

    class Fixed:
        n = len(desired)

        def __call__(self):
            return deepcopy(graph), dict(desired)

    cfg = dict(env_config)
    cfg.pop("initializer", None)
    cfg["graph_initializer"] = Fixed()
    cfg["resample_if_at_par"] = False
    cfg["quality_metric"] = "shape"
    if max_steps_factor is not None:
        cfg["max_steps_factor"] = max_steps_factor
    return initialize_environment(cfg)


def wants_for(env, graph, base):
    table = dict(base)
    for v in graph.vertex_list(tag=False):
        if v not in table:
            table[v] = (env.boundary_vertex_desired_degree if graph.is_boundary_vertex(v)
                        else env.interior_vertex_desired_degree)
    return table


def load_state(env, graph, base):
    """Point the env at `graph` (no copy) with a consistent want table."""
    env.graph = graph
    env.vertex_desired_degree = wants_for(env, graph, base)
    env._update_half_edge_angles()
    env.vertex_desired_options = env._compute_desired_options()


def fingerprint(graph):
    degrees = tuple(sorted(graph.vertex_degree(v) for v in graph.vertex_list()))
    return (graph.number_of_vertices(), len(graph.face_list()), degrees)


class Candidate:
    __slots__ = ("graph", "irr", "q", "vertices", "faces", "origin", "worst", "parent", "cut")

    def __init__(self, graph, irr, q, origin, worst):
        self.graph = graph
        self.irr = int(irr)
        self.q = float(q)
        self.vertices = graph.number_of_vertices()
        self.faces = len(graph.face_list())
        self.origin = origin
        self.worst = worst
        # provenance, for figures: the seed Candidate it came from and the state
        # right after the surgery, before the agent closed it (None for rollouts)
        self.parent = None
        self.cut = None

    @property
    def irr_per_vertex(self):
        return self.irr / max(self.vertices, 1)


def measure(env, graph, base, origin=""):
    """Untangle a copy of an all-quad `graph`; return a Candidate."""
    g = deepcopy(graph)
    load_state(env, g, base)
    irr = env.global_l1_vertex_score()
    try:
        resmooth_env(env)
    except Exception:
        pass
    q = float(env.min_element_quality())
    return Candidate(g, irr, q, origin, worst_faces(g, 3))


def face_qualities(graph):
    """{face: min corner shape quality} under the tangent-aware metric."""
    corners = graph.corner_shape_qualities()
    out = {}
    for h, q in corners.items():
        f = graph.face(h)
        out[f] = min(out.get(f, np.inf), q)
    return out


def worst_faces(graph, k=3):
    fq = face_qualities(graph)
    return sorted(fq, key=fq.get)[:k]


# ----------------------------------------------------------------------------- rollouts

def rollouts(env, model, base, start=None, n=4, budget=None, deterministic_first=True,
             keep_per_level=3):
    """All-quad states the policy passes through, as {fingerprint: (irr, graph)}.

    `start` is a graph to resume from (surgery); None resets to the domain. Per
    rollout and per irregularity level at most `keep_per_level` states are kept
    (first, last, and one in between), since consecutive all-quad states on one
    trajectory differ by a closing pair of moves and are near-duplicates.
    """
    found = {}
    for trial in range(n):
        if start is None:
            obs, _ = env.reset()
        else:
            s = deepcopy(start)
            obs, _ = env._reset_to_state(s, wants_for(env, s, base))
        if budget is not None:
            env.max_steps = min(env.max_steps, int(budget))
        per_level = {}
        done = False
        while True:
            if int(env.global_face_score) == 0:
                irr = int(env.global_vertex_score)
                per_level.setdefault(irr, []).append(deepcopy(env.graph))
            if done:
                break
            action, _ = model.predict(obs, deterministic=(deterministic_first and trial == 0))
            obs, _, done, truncated, _ = env.step(action)
            done = done or truncated
        for irr, graphs in per_level.items():
            if len(graphs) > keep_per_level:
                idx = np.unique(np.linspace(0, len(graphs) - 1, keep_per_level).round().astype(int))
                graphs = [graphs[i] for i in idx]
            for g in graphs:
                found.setdefault(fingerprint(g), (irr, g))
    return found


# ----------------------------------------------------------------------------- surgeries

def _face_loop(graph, face):
    return graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))


def sheet_edges(graph, half_edge):
    """The edges a quad sheet crosses, starting across `half_edge`'s edge.

    Walks from the quad of `half_edge` through opposite sides in both
    directions until the boundary (or back to the start: a closed sheet).
    Returns a list of half-edges, one per crossed edge, each in a face of the
    sheet, or None if the sheet meets a non-quad or crosses itself.
    """
    g = graph
    if g.face_degree(g.face(half_edge)) != 4:
        return None
    crossed, faces = [], set()

    def walk(h, forward_list):
        # h is a half-edge in a quad; step to its opposite side and across
        while True:
            f = g.face(h)
            if f in faces:
                return "loop"
            faces.add(f)
            opp = g.next_half_edge(g.next_half_edge(h))
            forward_list.append(opp)
            twin = g.twin_half_edge(opp)
            if g.is_boundary_half_edge(twin):
                return "boundary"
            if g.face_degree(g.face(twin)) != 4:
                return "odd"
            h = twin

    ahead = []
    end = walk(half_edge, ahead)
    if end == "odd":
        return None
    if end == "loop":
        # closed only if we came back through the start edge itself
        if g.twin_half_edge(ahead[-1]) != half_edge:
            return None
        return [half_edge] + ahead[:-1]
    behind = []
    twin0 = g.twin_half_edge(half_edge)
    if not g.is_boundary_half_edge(twin0):
        if g.face_degree(g.face(twin0)) != 4:
            return None
        end2 = walk(twin0, behind)
        if end2 != "boundary":
            return None
    crossed = [half_edge] + ahead + behind
    return crossed


def split_sheet(graph, half_edge):
    """Refine the sheet through `half_edge`'s edge; returns True on success.

    Inserts a vertex on every crossed edge and joins the two new vertices of
    each quad. Degrees of existing vertices do not change; each new vertex ends
    at degree 4 inside or 3 on the boundary, which is its want, so the vertex
    score is EXACTLY preserved while the element count grows by the sheet length.
    """
    edges = sheet_edges(graph, half_edge)
    if not edges:
        return False
    g = graph
    # the quads to cut, identified by their two crossed half-edges; inserting a
    # vertex keeps the half-edge ids of the first half, so record faces first
    faces = []
    for h in edges:
        faces.append(g.face(h))
        tw = g.twin_half_edge(h)
        if not g.is_boundary_half_edge(tw):
            faces.append(g.face(tw))
    new_vertices = set()
    for h in edges:
        g.insert_vertex(h)
        new_vertices.add(g.target_vertex(h, tag=False))
    for f in set(faces):
        loop = _face_loop(g, f)
        starts = [h for h in loop if g.source_vertex(h, tag=False) in new_vertices]
        if len(starts) != 2 or len(loop) != 6:
            return False
        h = starts[0]
        g.insert_half_edge(h, 2)
    return True


def surgery_moves(graph, face):
    """(name, function(graph) -> bool) edits at one face, applied to a copy."""
    loop = _face_loop(graph, face)
    moves = []
    idx = {h: i for i, h in enumerate(loop)}
    for i, h in enumerate(loop):
        # sheet splits: two directions per quad (sides 0/2 and 1/3)
        if i < 2 and len(loop) == 4:
            moves.append((f"sheet{i}", lambda g, h=h: split_sheet(g, h)))
        moves.append((f"insert{i}", lambda g, h=h: (g.insert_vertex(h), True)[1]))
    if len(loop) == 4:
        for i in (0, 1):
            h = loop[i]
            moves.append((f"chord{i}", lambda g, h=h: (g.insert_half_edge(h, 1), True)[1]
                          if g.is_valid_chord_insert(h, 1, 3) else False))
    del idx
    return moves


def defect_faces(env, graph, base, k=4):
    """Faces touching irregular vertices, worst-shaped first, at most `k`."""
    load_state(env, graph, base)
    fq = face_qualities(graph)
    faces = []
    for f in graph.face_list():
        loop = _face_loop(graph, f)
        for h in loop:
            v = graph.source_vertex(h, tag=False)
            if env.vertex_defect(v):
                faces.append(f)
                break
    faces.sort(key=lambda f: fq.get(f, 1.0))
    return faces[:k]


# ----------------------------------------------------------------------------- search

def key_regular_first(c, bar):
    """Sort key: usable meshes by (irr, elements, -q); unusable ones after, by (-q, irr).

    Elements break ties among equally regular usable meshes, so refinement is
    kept only when it buys regularity or clears the bar -- never to lower the
    irregular-per-vertex ratio by inflating the vertex count."""
    if c.q >= bar:
        return (0, c.irr, c.faces, -c.q)
    return (1, -c.q, c.irr)


def surgery_search(env, model, base, bar=0.3, n_initial=5, n_continue=3, rounds=6,
                   beam=3, faces_per_seed=2, continue_budget=40, max_measure=60,
                   moves=("sheet", "insert", "chord"), log=None, initial=None, pool=None,
                   polish=0, time_limit=None):
    """Run the loop; returns (best Candidate or None, pool list, stats dict).

    `pool` ({fingerprint: Candidate}, already measured) replaces the initial
    rollouts, so a caller can score the agent alone on the same candidates.

    `polish` rounds run once no below-bar candidate can beat the best usable
    one: the seeds are then the best USABLE meshes still above par, and the
    surgery goes at the elements around their irregular vertices rather than
    at the worst element -- cutting next to a defect and letting the policy
    re-close the region is how a defect pair gets cancelled. `time_limit`
    (seconds) stops expanding once spent.
    """
    import time
    started = time.time()
    polished = 0
    pool = dict(pool) if pool else {}
    stats = dict(measured=0, rollouts=0, rounds=0)

    def best_usable_irr():
        irs = [c.irr for c in pool.values() if c.q >= bar]
        return min(irs) if irs else None

    def absorb(found, origin, limit=max_measure, parent=None, cut=None):
        """Measure new all-quad states that could still matter."""
        items = sorted(found.items(), key=lambda kv: kv[1][0])
        taken = 0
        for fp, (irr, g) in items:
            if fp in pool:
                continue
            cap = best_usable_irr()
            if cap is not None and irr > cap:
                continue
            if taken >= limit:
                break
            pool[fp] = measure(env, g, base, origin)
            pool[fp].parent, pool[fp].cut = parent, cut
            stats["measured"] += 1
            taken += 1

    if not pool:
        if initial is None:
            found = rollouts(env, model, base, None, n=n_initial)
            stats["rollouts"] += n_initial
        else:
            found = initial
        absorb(found, "agent")
    if not pool:
        return None, [], stats

    expanded = set()
    for r in range(rounds):
        best = min(pool.values(), key=lambda c: key_regular_first(c, bar))
        par = int(env.par)
        if best.q >= bar and best.irr <= par:
            break
        cap = best_usable_irr()
        # seeds: the most regular BELOW-bar candidates that could beat the best
        # usable one on regularity, then by quality
        seeds = [c for fp, c in pool.items()
                 if c.q < bar and (cap is None or c.irr < cap) and fp not in expanded]
        seeds.sort(key=lambda c: (c.irr, -c.q))
        seeds = seeds[:beam]
        target = {}
        if not seeds and polished < polish and cap is not None and cap > par:
            polished += 1
            seeds = [c for fp, c in pool.items()
                     if c.q >= bar and c.irr == cap and fp not in expanded]
            seeds.sort(key=lambda c: -c.q)
            seeds = seeds[:beam]
            for c in seeds:
                target[id(c)] = defect_faces(env, c.graph, base, 2 * faces_per_seed)
        if not seeds:
            break
        if time_limit is not None and time.time() - started > time_limit:
            break
        stats["rounds"] += 1
        for seed in seeds:
            expanded.add(fingerprint(seed.graph))
            for face in target.get(id(seed), seed.worst[:faces_per_seed]):
                for name, move in surgery_moves(seed.graph, face):
                    if not any(name.startswith(m) for m in moves):
                        continue
                    g = deepcopy(seed.graph)
                    try:
                        ok = move(g)
                    except Exception:
                        ok = False
                    if not ok:
                        continue
                    origin = f"{seed.origin}>{name}"
                    load_state(env, g, base)
                    if env.global_l1_face_score() == 0:
                        # still all-quad: measure it directly
                        fp = fingerprint(g)
                        if fp not in pool:
                            pool[fp] = measure(env, g, base, origin)
                            pool[fp].parent, pool[fp].cut = seed, deepcopy(g)
                            stats["measured"] += 1
                        continue
                    try:
                        found = rollouts(env, model, base, g, n=n_continue,
                                         budget=continue_budget)
                    except Exception:
                        continue
                    stats["rollouts"] += n_continue
                    absorb(found, origin, limit=8, parent=seed, cut=g)
        if log:
            b = min(pool.values(), key=lambda c: key_regular_first(c, bar))
            log(f"    round {r}: pool {len(pool)} best irr {b.irr} q {b.q:+.3f} "
                f"V {b.vertices} ({b.origin})")
    best = min(pool.values(), key=lambda c: key_regular_first(c, bar))
    return best, list(pool.values()), stats
