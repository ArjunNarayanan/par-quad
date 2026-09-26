"""Search over the environment's own action vocabulary.

The puzzle is deterministic, fully observable and has a provable goal test, so
a search can certify solutions the policy only guesses at. Everything here
speaks the environment's action language -- `(half_edge, local_action)` pairs
with the same chord offset and the same layout the env uses -- so a trajectory
found by search is directly replayable and directly trainable.

`beam_search` is policy-guided and is what the held-out evaluation and the
expert-iteration loop call; `ida_star` is the exact fallback for small
instances.
"""

import heapq
import math
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from src.canonical import certificate, weisfeiler_lehman_hash


class SearchState:
    """A mesh plus the desired-degree bookkeeping the score needs."""

    __slots__ = ("graph", "desired", "spec", "_cert")

    def __init__(self, graph, desired, spec):
        self.graph = graph
        self.desired = dict(desired)
        self.spec = spec
        self._cert = None

    def copy(self):
        return SearchState(deepcopy(self.graph), self.desired, self.spec)

    def desired_of(self, vidx):
        if vidx in self.desired:
            return self.desired[vidx]
        return (self.spec.boundary_desired if self.graph.is_boundary_vertex(vidx)
                else self.spec.interior_desired)

    def vertex_score(self):
        return sum(abs(self.graph.vertex_degree(v) - self.desired_of(v))
                   for v in self.graph.vertex_list(tag=False))

    def face_score(self):
        return sum(abs(self.graph.face_degree(f) - self.spec.face_desired)
                   for f in self.graph.face_list())

    def min_quality(self):
        angles = self.graph.half_edge_angles()
        if not angles:
            return 1.0
        return min(math.sin(math.radians(a)) for a in angles.values())

    def fingerprint(self, exact=False):
        if self._cert is None:
            fn = certificate if exact else weisfeiler_lehman_hash
            self._cert = fn(self.graph, self.desired,
                            boundary_desired=self.spec.boundary_desired,
                            interior_desired=self.spec.interior_desired)
        return self._cert


class ActionSpec:
    """The action vocabulary and scoring constants, mirrored from the env."""

    def __init__(self, max_edge_addition_steps=3, chord_offset=1, allow_delete=False,
                 min_face_degree=3,
                 face_desired=4, boundary_desired=3, interior_desired=4,
                 quality_threshold=0.4, smooth_iterations=5):
        self.max_edge_addition_steps = max_edge_addition_steps
        self.chord_offset = chord_offset
        self.allow_delete = allow_delete
        self.min_face_degree = min_face_degree
        self.face_desired = face_desired
        self.boundary_desired = boundary_desired
        self.interior_desired = interior_desired
        self.quality_threshold = quality_threshold
        self.smooth_iterations = smooth_iterations

        self.insert_vertex_action = (
            max_edge_addition_steps + 1 if allow_delete else max_edge_addition_steps)
        self.delete_edge_action = max_edge_addition_steps if allow_delete else None
        self.delete_vertex_action = (
            max_edge_addition_steps + 2 if allow_delete else None)
        self.num_actions_per_half_edge = (
            max_edge_addition_steps + (3 if allow_delete else 1))

    @classmethod
    def from_env(cls, env):
        return cls(
            max_edge_addition_steps=env.max_edge_addition_steps,
            chord_offset=env._chord_offset,
            allow_delete=env.allow_delete,
            min_face_degree=env.min_face_degree,
            face_desired=env.face_desired_degree,
            boundary_desired=env.boundary_vertex_desired_degree,
            interior_desired=env.interior_vertex_desired_degree,
            quality_threshold=env.quality_threshold,
            smooth_iterations=env.smooth_iterations,
        )

    def chord_steps(self, local):
        return local + self.chord_offset


def legal_actions(state):
    """Every `(half_edge, local_action)` the env would leave unmasked."""
    spec = state.spec
    graph = state.graph
    out = []
    for h in graph.half_edge_list():
        for local in range(spec.max_edge_addition_steps):
            if graph.is_valid_chord_insert(h, spec.chord_steps(local), spec.min_face_degree):
                out.append((h, local))
        out.append((h, spec.insert_vertex_action))
        if spec.allow_delete:
            if graph.is_valid_delete_half_edge(h):
                out.append((h, spec.delete_edge_action))
            if graph.is_valid_delete_source_vertex(h):
                out.append((h, spec.delete_vertex_action))
    return out


def apply_action(state, action, smooth=True):
    """A new state with `action` applied, mirroring the env step exactly."""
    spec = state.spec
    child = state.copy()
    graph = child.graph
    h, local = action
    if local < spec.max_edge_addition_steps:
        graph.insert_half_edge(h, spec.chord_steps(local))
    elif local == spec.insert_vertex_action:
        on_boundary = graph.half_edge_on_boundary(h)
        graph.insert_vertex(h)
        new_v = graph.target_vertex(h, tag=False)
        child.desired[new_v] = (spec.boundary_desired if on_boundary
                                else spec.interior_desired)
    elif local == spec.delete_edge_action:
        graph.delete_half_edge(h)
    elif local == spec.delete_vertex_action:
        source = graph.source_vertex(h, tag=False)
        graph.delete_source_vertex(h)
        child.desired.pop(source, None)
    else:
        raise ValueError("Unexpected local action index: " + str(local))
    if smooth and spec.smooth_iterations:
        graph.smooth_vertices(num_iter=spec.smooth_iterations)
    return child


def heuristic(state, par):
    """Admissible lower bound on the number of moves still required."""
    # a chord changes at most two vertex degrees by one
    h1 = math.ceil(max(state.vertex_score() - par, 0) / 2)
    # a d-gon needs at least ceil((d-4)/2) more chords; an odd d also needs a vertex
    h2 = 0
    for f in state.graph.face_list():
        d = state.graph.face_degree(f)
        if d > 4:
            h2 += math.ceil((d - 4) / 2)
        if d % 2 == 1:
            h2 += 1
    return max(h1, h2)


def is_goal(state, par):
    if state.vertex_score() != par or state.face_score() != 0:
        return False
    return state.min_quality() >= state.spec.quality_threshold


def state_from_env(env):
    """Snapshot the env's current mesh as a search state."""
    return SearchState(deepcopy(env.graph), env.vertex_desired_degree,
                       ActionSpec.from_env(env))


def render_observation(env, state, depth, max_steps=None):
    """Render a search state through the env's own observation pipeline."""
    env.graph = state.graph
    env.vertex_desired_degree = dict(state.desired)
    env.num_steps = depth
    if max_steps is not None:
        env.max_steps = max_steps
    env._update_half_edge_angles()
    env._update_global_face_score()
    env._update_global_angle_score()
    env._update_global_vertex_score()
    env._set_half_edge_template_center(env.graph.half_edge_list())
    env._build_template()
    return env._get_obs(), list(env.index_to_half_edge)


def _stack_observations(obs_list, keys):
    batch = {}
    for key in keys:
        arr = np.stack([o[key] for o in obs_list])
        dtype = torch.int64 if key in ("next", "previous", "twin") else torch.float32
        batch[key] = torch.as_tensor(arr, dtype=dtype)
    return batch


def policy_log_probs(policy, obs_batch):
    with torch.no_grad():
        distribution = policy.get_distribution(obs_batch)
    return distribution.distribution.logits.numpy()


def decode_linear_action(linear, index_to_half_edge, num_actions_per_half_edge):
    slot, local = divmod(int(linear), num_actions_per_half_edge)
    if slot >= len(index_to_half_edge):
        return None
    return index_to_half_edge[slot], local


def beam_search(policy, env, state=None, par=None, beam=32, top_k=6,
                max_depth=20, budget_s=120.0, heuristic_weight=1.5,
                max_expansions=None):
    """Policy-guided beam search returning env-replayable actions.

    Ranks a child by the policy's log-probability of the move that produced it
    plus an admissible lower bound on what remains, so the prior proposes and
    the bound keeps the beam honest.
    """
    scratch = deepcopy(env)
    if state is None:
        state = state_from_env(scratch)
    if par is None:
        par = scratch.par
    spec = state.spec
    obs_keys = list(env.observation_space.spaces.keys())
    n_actions = spec.num_actions_per_half_edge

    if is_goal(state, par):
        return [], {"expanded": 0, "seconds": 0.0, "status": "already-solved"}

    frontier = [(0.0, 0, state, [])]
    seen = {state.fingerprint()}
    t0 = time.time()
    expanded = 0
    tie = 0
    # the best all-quad mesh seen anywhere in the search, not just in the final
    # frontier: a complete mesh is often found and then pruned for having a poor
    # prior, and it is exactly what a near-miss is worth keeping for
    best_complete = None

    for depth in range(max_depth):
        if not frontier or time.time() - t0 > budget_s:
            break
        if max_expansions is not None and expanded >= max_expansions:
            break

        obs_list, decoders = [], []
        for _, _, st, _ in frontier:
            obs, idx2h = render_observation(scratch, st, depth)
            obs_list.append(obs)
            decoders.append(idx2h)
        batch = _stack_observations(obs_list, obs_keys)
        logits = policy_log_probs(policy, batch)
        logp = logits - torch.logsumexp(torch.as_tensor(logits), dim=1, keepdim=True).numpy()

        children = []
        for i, (score, _, st, path) in enumerate(frontier):
            order = np.argsort(-logp[i])[: top_k * 6]
            taken = 0
            for linear in order:
                if not np.isfinite(logp[i][linear]):
                    continue
                action = decode_linear_action(linear, decoders[i], n_actions)
                if action is None:
                    continue
                h, local = action
                graph = st.graph
                if local < spec.max_edge_addition_steps:
                    ok = graph.is_valid_chord_insert(
                        h, spec.chord_steps(local), spec.min_face_degree)
                elif local == spec.insert_vertex_action:
                    ok = graph.is_half_edge(h)
                elif local == spec.delete_edge_action:
                    ok = graph.is_valid_delete_half_edge(h)
                else:
                    ok = graph.is_valid_delete_source_vertex(h)
                if not ok:
                    continue

                child = apply_action(st, action)
                expanded += 1
                fingerprint = child.fingerprint()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                if (child.face_score() == 0
                        and child.min_quality() >= child.spec.quality_threshold):
                    excess = child.vertex_score() - par
                    if best_complete is None or excess < best_complete[0]:
                        best_complete = (excess, path + [action])
                if is_goal(child, par):
                    return path + [action], {
                        "expanded": expanded, "seconds": time.time() - t0,
                        "status": "solved", "depth": depth + 1,
                        "best_excess": 0, "best_actions": path + [action]}
                tie += 1
                f = score + logp[i][linear] - heuristic_weight * heuristic(child, par)
                children.append((f, tie, child, path + [action]))
                taken += 1
                if taken >= top_k:
                    break
        children.sort(key=lambda x: -x[0])
        frontier = children[:beam]

    return None, {"expanded": expanded, "seconds": time.time() - t0,
                  "status": "not-found",
                  "best_excess": None if best_complete is None else best_complete[0],
                  "best_actions": None if best_complete is None else best_complete[1]}


def ida_star(state, par, max_depth=10, budget_s=60.0, exact_dedup=True):
    """Iterative-deepening A*. Exact but only tractable on small instances."""
    t0 = time.time()
    nodes = [0]

    def dfs(node, g_cost, limit, path, seen):
        if time.time() - t0 > budget_s:
            raise TimeoutError
        f = g_cost + heuristic(node, par)
        if f > limit:
            return None, f
        if is_goal(node, par):
            return list(path), f
        next_limit = math.inf
        for action in legal_actions(node):
            child = apply_action(node, action)
            nodes[0] += 1
            key = child.fingerprint(exact=exact_dedup)
            if key in seen and seen[key] <= g_cost + 1:
                continue
            seen[key] = g_cost + 1
            path.append(action)
            found, child_f = dfs(child, g_cost + 1, limit, path, seen)
            path.pop()
            if found is not None:
                return found, child_f
            next_limit = min(next_limit, child_f)
        return None, next_limit

    limit = heuristic(state, par)
    while limit <= max_depth:
        try:
            found, nxt = dfs(state, 0, limit, [], {})
        except TimeoutError:
            return None, {"nodes": nodes[0], "seconds": time.time() - t0,
                          "status": "timeout"}
        if found is not None:
            return found, {"nodes": nodes[0], "seconds": time.time() - t0,
                           "status": "optimal", "depth": len(found)}
        if nxt == math.inf:
            break
        limit = nxt
    return None, {"nodes": nodes[0], "seconds": time.time() - t0,
                  "status": "exhausted"}
