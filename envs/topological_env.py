"""The same problem with the geometry taken out of the loop.

Quad block decomposition is a combinatorial problem wearing a geometric coat. Par
is a count. Face score and vertex score are counts. Every edit operation is
topological. The only geometry in the MDP was a Laplacian smoother run after every
move, whose output fed two of ten per-half-edge features and one global scalar --
and which was also what the win test measured element quality on.

That arrangement cost more than it bought:

* **It made the MDP non-stationary.** Smoothing moves every angle every step, so
  the same topological state carries different observations depending on how it
  was reached.
* **It made the win test measure the smoother.** A Laplacian folds elements that
  geo2d's untangler recovers from -1.000 to 0.613 on the SAME topology. Domains
  were failing on geometry their topology admitted.
* **It gave the agent a signal it could not act on.** At a topologically perfect
  mesh no legal move preserves `face_score == 0 and vertex_score == par` -- a chord
  through a quad leaves a triangle, a vertex insert leaves a pentagon -- so fixing
  geometry means a multi-move detour that a purely topological potential punishes
  at every step. Measured: the agent reaches par topology at quality 0.257, cannot
  act, and wanders off it.

So the smoother comes out of the loop, the features become counts, and quality
moves to an **evaluator** that runs only when the agent claims a solution. The
evaluator untangles a copy of the candidate and asks one question: does this
topology admit a non-degenerate embedding? A quad with a 180-degree corner -- a
vertex inserted on the edge of a triangle -- never will, and is rejected. A mesh at
quality 0.25 will, and is accepted; smoothing and refinement are downstream tools,
not the agent's problem.

Rejection hands the candidate back. The observation carries a bit saying so,
because `at par and accepted` and `at par and rejected` are otherwise the same
state with opposite correct actions, and the template centres on the offending
element so the agent is at least standing in the right place.

KNOWN GAP, deliberately left open: the degeneracy test is angle-based, so it is
blind to aspect ratio -- a 1x10 rectangle meshed as one quad scores 1.0. Nothing
here discourages combs of thin quads, which refinement cannot repair. The fixes if
it shows up are an aspect-ratio feature, a reward term, and centering on bad
elements; topology is the harder problem and comes first.
"""

from copy import deepcopy

import numpy as np

from envs.global_angle_env import AngleEnv


class TopologicalEnv(AngleEnv):
    """`AngleEnv` with counts for features and an evaluator for the quality gate."""

    def __init__(self, *args, degeneracy_threshold=0.2, evaluator="untangle",
                 untangle_iterations=12, warm_start_iterations=5, **kwargs):
        # geometry never enters an observation, so smoothing the mesh between moves
        # only costs time; the coordinates still exist for the evaluator and for
        # anything that wants to draw the result
        kwargs.setdefault("smooth_iterations", 0)
        self.degeneracy_threshold = degeneracy_threshold
        self.evaluator = evaluator
        self.untangle_iterations = untangle_iterations
        self.warm_start_iterations = warm_start_iterations
        self._rejected = False
        self._rejected_face = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_feature_size():
        return 7

    @classmethod
    def from_config(cls, config):
        """`AngleEnv.from_config` plus this env's own three settings.

        The parent builds its keyword arguments explicitly, so anything new has to
        be threaded through here rather than picked up from the dictionary.
        """
        env = super().from_config(config)
        # geometry is out of the loop unless a config explicitly asks otherwise,
        # so a config written for AngleEnv does not silently smooth every step
        env.smooth_iterations = config.get("smooth_iterations", 0)
        env.degeneracy_threshold = config.get("degeneracy_threshold", 0.2)
        env.evaluator = config.get("evaluator", "untangle")
        env.untangle_iterations = config.get("untangle_iterations", 12)
        env.warm_start_iterations = config.get("warm_start_iterations", 5)
        return env

    # ------------------------------------------------------------------ features

    def _get_topological_context(self):
        """Four counts about the far end of the half-edge.

        `AngleEnv._get_context_features` is the same list plus the target corner's
        angle, which is the one entry that needed the smoother.
        """
        n = len(self.index_to_half_edge)
        out = np.zeros((n, 4), dtype=np.float32)
        for idx, hidx in enumerate(self.index_to_half_edge):
            source = self.graph.source_vertex(hidx, tag=False)
            target = self.graph.source_vertex(self.graph.next_half_edge(hidx), tag=False)
            out[idx, 0] = float(self.graph.is_boundary_vertex(source))
            out[idx, 1] = float(self.graph.is_user_defined_vertex(source))
            out[idx, 2] = (self.effective_desired_degree(target)
                           if target in self.vertex_desired_degree
                           else self.interior_vertex_desired_degree)
            out[idx, 3] = min(self.graph.vertex_degree(target), self.vertex_degree_threshold)
        return out

    def _get_feature_matrix(self):
        sources = self.template_vertices
        faces = self.template_faces
        n = len(self.index_to_half_edge)

        matrix = np.zeros((self.template_size, self.num_features), dtype=np.float32)
        matrix[:n, 0] = [self.effective_desired_degree(v) for v in sources]
        matrix[:n, 1] = [min(self.graph.vertex_degree(v), self.vertex_degree_threshold)
                         for v in sources]
        matrix[:n, 2] = [min(self.graph.face_degree(f), self.face_degree_threshold)
                         for f in faces]
        matrix[:n, 3:7] = self._get_topological_context()
        return matrix

    def _get_global_features(self):
        """The same global vector with quality replaced by the rejection bit.

        Element quality was a single scalar saying how bad the worst element was
        and nothing about where, which the agent could not act on. The bit it gets
        instead is actionable: it says the evaluator has already refused this
        topology, so being at par is not the end of the episode.
        """
        vertex_defect = abs(self.global_vertex_score - self.par)
        half_edges = self.graph.number_of_half_edges()
        features = [
            min(vertex_defect, 20) / 5.0,
            min(self.global_face_score, 40) / 5.0,
            min(self.number_of_odd_faces(), 20) / 5.0,
            self.num_steps / max(self.max_steps, 1),
            min(half_edges / max(self.template_size, 1), 2.0),
            float(self._rejected),
            min(self.par, 10) / 5.0,
            min(len(self.graph.face_list()), 40) / 10.0,
        ]
        if self.target_in_observation:
            features += [self.face_desired_degree / 4.0,
                         self.interior_vertex_desired_degree / 6.0]
        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------ evaluator

    def is_topologically_at_par(self):
        return self.global_face_score == 0 and self.global_vertex_score == self.par

    def _worst_element_quality(self, graph=None):
        """Worst scaled Jacobian, optionally of some other graph."""
        graph = graph or self.graph
        angles = graph.half_edge_angles()
        if not angles:
            return 1.0
        ideal = np.sin(np.radians(self.desired_angle))
        return min(np.sin(np.radians(a)) for a in angles.values()) / ideal

    def _untangled_quality(self):
        """Worst element quality after the untangler, on a COPY of the mesh.

        The env itself cannot be deepcopied: `graph_initializer` may be a
        geo2d-backed Mixture holding a module reference, and copy raises
        `TypeError: cannot pickle 'module' object`. That was silently caught by
        the except below and the evaluator degraded to raw angles -- the
        untangler never ran. So swap in a copy of the GRAPH, which does copy,
        and put the original back in a finally: the coordinates the agent
        observes must not move.
        """
        from src.geo2d_bridge import resmooth_env
        original_graph = self.graph
        original_angles = dict(self.half_edge_angles)
        try:
            self.graph = deepcopy(original_graph)
            if self.warm_start_iterations:
                self.graph.smooth_vertices(num_iter=self.warm_start_iterations)
                self._update_half_edge_angles()
            resmooth_env(self, iters=self.untangle_iterations, method="optimize",
                         slide="optimize")
            return self._worst_element_quality()
        finally:
            self.graph = original_graph
            self.half_edge_angles = original_angles

    def evaluate_candidate(self):
        """Does this topology admit a non-degenerate embedding?

        Untangles a COPY so the episode's own state is untouched, then asks whether
        the worst element clears `degeneracy_threshold`. The threshold is low on
        purpose: it is there to reject a topology that forces a flat or folded
        corner, not to demand a good mesh. Anything above it is smoothing and
        refinement's job, downstream of the agent.
        """
        if self.evaluator == "none":
            return True, None
        if self.evaluator == "raw":
            quality = self._worst_element_quality()
            return quality >= self.degeneracy_threshold, self._worst_face()
        try:
            # Laplacian sweeps FIRST. The untangler is a local optimiser, and from
            # the raw midpoint embedding a no-smoothing episode leaves behind it
            # starts inside a tangle it cannot always escape: on 114 certified
            # solutions -- optimal by construction -- untangling alone left 8 below
            # 0.1, while five Laplacian sweeps beforehand left none, worst case
            # 0.507. The warm start costs about a millisecond.
            #
            # NOT on a deepcopy of the env: `graph_initializer` may hold a geo2d
            # module reference, copy raises, the except below swallowed it and
            # this degraded to raw angles without ever saying so. Copy the GRAPH.
            quality = self._untangled_quality()
        except Exception:
            # geogen is an optional install and the untangler can fail on a
            # degenerate mesh; falling back to the raw angles is strictly the
            # harsher test, so a failure here never passes something it should not
            quality = self._worst_element_quality()
        return quality >= self.degeneracy_threshold, self._worst_face()

    def _worst_face(self):
        """The face carrying the worst corner, for the template to centre on."""
        worst, chosen = None, None
        for hidx in self.graph.half_edge_list():
            angle = self.half_edge_angles.get(hidx)
            if angle is None:
                continue
            value = float(np.sin(np.radians(angle)))
            if worst is None or value < worst:
                worst, chosen = value, self.graph.face(hidx)
        return chosen

    def is_at_par(self):
        if not self.is_topologically_at_par():
            return False
        accepted, face = self.evaluate_candidate()
        if not accepted:
            self._rejected = True
            self._rejected_face = face
        return accepted

    # ------------------------------------------------------------------ centering

    def _sticky_center(self):
        """As `AngleEnv`, but a rejected candidate centres on the bad element.

        Every integer key ties once the topology is perfect, so the parent would
        pick the lowest half-edge id -- an arbitrary place, rarely anywhere near the
        element the evaluator objected to.
        """
        if self._rejected and self._rejected_face is not None:
            live = [h for h in self.graph.half_edge_list()
                    if self.graph.face(h) == self._rejected_face]
            if live:
                return self._worst_vertex_half_edge(live)
            self._rejected_face = self._worst_face()
        return super()._sticky_center()

    def reset(self, *args, **kwargs):
        self._rejected = False
        self._rejected_face = None
        return super().reset(*args, **kwargs)
