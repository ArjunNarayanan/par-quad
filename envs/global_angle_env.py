import os
from src.tiler import Tiler
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
from copy import deepcopy
import pickle
import uuid
import numpy as np
import envs.polygon_utils as utils
from src.canonical import weisfeiler_lehman_hash

# Number of entries in obs["global"]; see `_get_global_features`.
GLOBAL_FEATURE_SIZE = 7
# Two more when the meshing target itself is in the observation, so one policy
# can be asked for quads or triangles; see `target_in_observation`.
TARGET_FEATURE_SIZE = 2


class AngleEnv(gym.Env):
    """Quad block decomposition as a finite-horizon puzzle over a half-edge mesh.

    The default action vocabulary is *monotone*: chord insertion and vertex
    insertion only. Every expert solution of the fifteen Mesh Quest levels uses
    those two operations and nothing else, so dropping the deletes costs no
    reachable goal state while removing the move/undo two-cycle that otherwise
    absorbs a deterministic policy. Set `allow_delete=True` for the full
    four-operation vocabulary.
    """

    def __init__(
            self,
            face_desired_degree,
            graph_initializer,
            template_size=32,
            max_steps_factor=1.5,
            min_max_steps=8,
            logdir=None,
            max_edge_addition_steps=3,
            face_reward_weight=0.5,
            angle_reward_weight=0.25,
            vertex_reward_weight=0.25,
            use_boundary=False,
            smooth_iterations=5,
            template_seed_face=False,
            vertex_degree_threshold=10,
            face_degree_threshold=10,
            terminate_at_par=True,
            par_bonus=1.0,
            quality_threshold=0.4,
            # --- WP1 switches: new behaviour on by default -------------------
            allow_digon=False,
            allow_delete=False,
            reward_mode="potential",
            step_cost=0.05,
            vertex_potential_weight=1.0,
            face_potential_weight=1.0,
            boundary_underdegree_weight=0.0,
            mask_flat_quads=False,
            mask_thin_quads=0.0,
            quality_penalty=1.0,
            quality_potential_weight=0.0,
            quality_metric="angle",
            quality_aggregate="min",
            clearance_potential_weight=0.0,
            clearance_target=4.0,
            best_so_far_reward=False,
            best_so_far_all_quad=False,
            terminate_when_usable=False,
            usable_quality=0.2,
            mask_inverse=True,
            revisit_penalty=0.2,
            deterministic_center=True,
            global_observation=True,
            global_features="raw",
            resample_if_at_par=True,
            max_reset_tries=50,
            terminate_on_overflow=False,
            target_in_observation=False,
            tie_aware_scoring=True,
            tie_tolerance=1e-6,
            center_mode="global_worst",
    ):
        super().__init__()
        self.graph_initializer = graph_initializer

        # An all-quad mesh of this domain cannot score better than `par`, so
        # that -- not zero -- is the finish line. Terminating there stops the
        # agent being forced to keep editing a mesh it has already perfected.
        self.terminate_at_par = terminate_at_par
        self.par_bonus = par_bonus
        self.quality_threshold = quality_threshold
        self.par = 0

        self.template_size = template_size
        self.max_steps_factor = max_steps_factor
        self.min_max_steps = min_max_steps

        self.num_steps = 0
        self.smooth_iterations = smooth_iterations
        # seed the template walk from the centre's whole FACE rather than one
        # of its half-edges; see Tiler._template_seeds. Off: it changes which
        # half-edges a truncated window holds.
        self.template_seed_face = template_seed_face

        if logdir is None:
            logdir = os.path.join(os.getcwd(), "experiments")
        self.logdir = logdir

        self.allow_digon = allow_digon
        self.allow_delete = allow_delete
        self.reward_mode = reward_mode
        self.step_cost = step_cost
        self.vertex_potential_weight = vertex_potential_weight
        # A non-quad face costs this much potential. At 1 a quad that closes a
        # face while dropping the worst corner by more than 1/(0.7 w_q) is a net
        # LOSS to the reward, so the policy is paid to leave the face open; above
        # 0.7 * 0.4 * w_q closing a face is always worth more than any quality it
        # could cost, which is "all-quad first" written into the potential.
        self.face_potential_weight = face_potential_weight
        # A boundary vertex BELOW its want is a flat or reflex corner owned by one
        # face -- a geometric degeneracy whatever par says. The vertex term is
        # |score - par|, so on a par-6 domain six such joints cost it nothing,
        # and that is where the hole-free tail lives: every zero-quality corner
        # on the stubborn domains is a boundary vertex at degree 2 wanting 3,
        # with the mesh at or near par. This charges each unit of boundary
        # under-degree regardless of the budget; it is extensive like the face
        # and vertex terms and enters d0 the same way.
        self.boundary_underdegree_weight = boundary_underdegree_weight
        # Forbid any chord that would close a QUAD with a flat boundary corner: a
        # boundary vertex wanting degree >= 3 whose two edges in that quad are both
        # boundary edges (an arc joint or a mid-edge vertex, interior angle ~180,
        # or a reflex corner). Such a quad has a zero or negative Jacobian at that
        # corner by construction and is never a face of a good final mesh, and
        # since a union of final quads has >= 6 sides no construction order of a
        # good mesh passes through one -- so the mask costs no reachable good
        # mesh; it only removes the detour through a degenerate state that the
        # policy otherwise takes on half the curved transfer domains.
        self.mask_flat_quads = mask_flat_quads
        self.mask_thin_quads = float(mask_thin_quads)
        self.quality_penalty = quality_penalty
        # weight on the quality shortfall INSIDE the potential; 0 keeps the old
        # behaviour, where quality is charged once at the end and nowhere else
        self.quality_potential_weight = quality_potential_weight
        self.clearance_potential_weight = clearance_potential_weight
        self.clearance_target = clearance_target
        self.best_so_far_reward = best_so_far_reward
        # Only an ALL-QUAD state may set the running best. Under plain best-so-far
        # an open-face state with a high potential is a candidate like any other,
        # so leaving the last face open is paid exactly like closing it and every
        # quality-holding arm leaked all-quad over training; a face WEIGHT that
        # stops that makes every face worth closing at any quality. Eligibility
        # is the lexicographic rule itself: nothing is ever paid for a mesh that
        # is not all-quad, and within all-quad meshes quality and regularity
        # trade at their weights. The running best starts at the START state's
        # score, so the first all-quad state is paid for all the face and vertex
        # progress it took to get there.
        self.best_so_far_all_quad = best_so_far_all_quad
        self.best_candidate = None
        # for the logs: whether the trajectory EVER reached par, and how many
        # moves it took to reach its best candidate. Both are the trajectory's
        # properties; reading them off the final state says nothing once the
        # episode runs past its own goal to max_steps.
        self.ever_at_par = False
        self.best_step = 0
        # "angle" is sin(corner) and cannot see aspect ratio, so a comb of
        # paper-thin quads scores perfectly; "shape" is geo2d's own metric,
        # the same Jacobian over the SUM of the squared edge lengths instead of
        # their product, and agrees with geo2d exactly on an all-quad mesh.
        self.quality_metric = quality_metric
        # "min" is the worst corner in the mesh -- right for a GATE, sparse for a
        # reward: fixing any element but the worst earns nothing, and the signal
        # jumps as which corner is worst changes. "mean" counts every poor
        # element and gives a dense gradient but lets one inverted element be
        # averaged away. "min+mean" keeps the hard constraint and adds density.
        self.quality_aggregate = quality_aggregate
        # Stop when the mesh is ALL-QUAD and well enough shaped, not only at par.
        # The agent cannot stop on its own -- the vocabulary is monotone, every
        # action adds -- so if the ending condition is par and par is out of
        # reach, it must keep going and can only break what it built. Traced on
        # curved domains: face score 22 -> 0 by move 34, then 226 further moves
        # ending at 63. Off by default; `usable_quality` is the bar.
        self.terminate_when_usable = terminate_when_usable
        self.usable_quality = usable_quality
        self.mask_inverse = mask_inverse and allow_delete
        self.revisit_penalty = revisit_penalty if allow_delete else 0.0
        self.deterministic_center = deterministic_center
        self.global_observation = global_observation
        # "raw": counts capped at a ceiling tuned for <= 24 corners; they pin at
        # the cap on large domains (5 of 7 entries for 28-93% of steps at 60-100
        # corners). "intensive": the same quantities per face, bounded on any size.
        self.global_features = global_features
        self.resample_if_at_par = resample_if_at_par
        self.max_reset_tries = max_reset_tries
        self.terminate_on_overflow = terminate_on_overflow
        self.target_in_observation = target_in_observation
        self.tie_aware_scoring = tie_aware_scoring
        self.tie_tolerance = tie_tolerance
        self.center_mode = center_mode
        self.vertex_desired_options = {}
        self.global_feature_size = (
            GLOBAL_FEATURE_SIZE + (TARGET_FEATURE_SIZE if target_in_observation else 0)
            # one more for the gap to the best candidate banked so far
            + (1 if best_so_far_reward else 0))

        self.max_edge_addition_steps = max_edge_addition_steps
        self.num_actions_per_half_edge = self.get_num_actions_per_half_edge(
            max_edge_addition_steps, allow_delete=allow_delete
        )
        self.num_actions_per_halfedge = self.num_actions_per_half_edge
        self.total_num_actions_in_template = self.template_size * self.num_actions_per_half_edge

        # local action indices
        self._insert_vertex_action = (
            self.max_edge_addition_steps + 1 if allow_delete else self.max_edge_addition_steps
        )
        self._delete_edge_action = self.max_edge_addition_steps if allow_delete else None
        self._delete_vertex_action = (
            self.max_edge_addition_steps + 2 if allow_delete else None
        )
        self._chord_offset = 0 if allow_digon else 1
        self.min_face_degree = 2 if allow_digon else 3

        self.face_desired_degree = face_desired_degree
        self.desired_angle = utils.average_face_angle(self.face_desired_degree)
        self.interior_vertex_desired_degree = utils.rounded_desired_degree(360, self.desired_angle) - 1
        self.boundary_vertex_desired_degree = utils.rounded_desired_degree(180, self.desired_angle)

        graph, vertex_desired_degree = self.graph_initializer()

        self.graph = graph
        self.vertex_desired_degree = vertex_desired_degree
        self.max_steps = self._compute_max_steps()

        self.face_reward_weight = face_reward_weight
        self.angle_reward_weight = angle_reward_weight
        self.vertex_reward_weight = vertex_reward_weight

        self.use_boundary = use_boundary
        self._inverse_block = None
        self._visited = set()
        self._update_half_edge_angles()
        self.vertex_desired_options = self._compute_desired_options()
        self._set_half_edge_template_center(self.graph.half_edge_list())
        self._build_template()
        self._update_scores_on_reset()

        self.exception_occurred = False
        self.par = self.compute_par()
        self.potential = self._compute_potential()
        self.initial_defect = self._capture_initial_defect()
        self.terminated = self.is_terminated()

        # Attributes for excpetion handling
        self.initial_graph = deepcopy(self.graph)
        self.initial_vertex_desired_degree = self.vertex_desired_degree.copy()
        self.exception_count = 0
        self.action_sequence = []

        self.template_boundary_index = -2
        self.geometric_boundary_index = -1
        self.num_features = self.get_feature_size()

        self.action_space = Discrete(self.num_actions_per_half_edge)
        self.vertex_degree_threshold = vertex_degree_threshold
        self.face_degree_threshold = face_degree_threshold
        features_max_val = max(self.vertex_degree_threshold, self.face_degree_threshold)

        spaces = {
            "features": Box(low=0, high=features_max_val, shape=(self.template_size, self.num_features)),
            "next": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
            "previous": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
            "twin": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
            "mask": Box(low=-np.inf, high=0, shape=(self.total_num_actions_in_template,)),
            "progress": Box(low=0, high=1, shape=(1,)),
        }
        if self.global_observation:
            spaces["global"] = Box(low=-2.0, high=10.0, shape=(self.global_feature_size,))
        self.observation_space = Dict(spaces)

    @classmethod
    def from_config(cls, config):
        face_desired_degree = config["face_desired_degree"]
        graph_initializer = config["graph_initializer"]

        template_size = config["template_size"]
        max_steps_factor = config.get("max_steps_factor", 1.5)
        min_max_steps = config.get("min_max_steps", 8)
        logdir = config.get("logdir", None)
        max_edge_addition_steps = config.get("max_edge_addition_steps", 3)

        face_reward_weight = config.get("face_reward_weight", 1 / 3)
        angle_reward_weight = config.get("angle_reward_weight", 1 / 3)
        vertex_reward_weight = config.get("vertex_reward_weight", 1 / 3)

        use_boundary = config.get("use_boundary", False)
        smooth_iterations = config.get("smooth_iterations", 5)
        template_seed_face = config.get("template_seed_face", False)

        vertex_degree_threshold = config.get("vertex_degree_threshold", 10)
        face_degree_threshold = config.get("face_degree_threshold", 10)

        tie_aware_scoring = config.get("tie_aware_scoring", True)
        tie_tolerance = config.get("tie_tolerance", 1e-6)
        center_mode = config.get("center_mode", "global_worst")

        terminate_at_par = config.get("terminate_at_par", True)
        par_bonus = config.get("par_bonus", 1.0)
        quality_threshold = config.get("quality_threshold", 0.4)

        return cls(
            face_desired_degree,
            graph_initializer,
            template_size=template_size,
            max_steps_factor=max_steps_factor,
            min_max_steps=min_max_steps,
            logdir=logdir,
            max_edge_addition_steps=max_edge_addition_steps,
            face_reward_weight=face_reward_weight,
            angle_reward_weight=angle_reward_weight,
            vertex_reward_weight=vertex_reward_weight,
            use_boundary=use_boundary,
            smooth_iterations=smooth_iterations,
            template_seed_face=template_seed_face,
            vertex_degree_threshold=vertex_degree_threshold,
            face_degree_threshold=face_degree_threshold,
            terminate_at_par=terminate_at_par,
            par_bonus=par_bonus,
            quality_threshold=quality_threshold,
            allow_digon=config.get("allow_digon", False),
            allow_delete=config.get("allow_delete", False),
            reward_mode=config.get("reward_mode", "potential"),
            step_cost=config.get("step_cost", 0.05),
            vertex_potential_weight=config.get("vertex_potential_weight", 1.0),
            face_potential_weight=config.get("face_potential_weight", 1.0),
            boundary_underdegree_weight=config.get("boundary_underdegree_weight", 0.0),
            mask_flat_quads=config.get("mask_flat_quads", False),
            mask_thin_quads=config.get("mask_thin_quads", 0.0),
            quality_penalty=config.get("quality_penalty", 1.0),
            quality_potential_weight=config.get("quality_potential_weight", 0.0),
            quality_metric=config.get("quality_metric", "angle"),
            quality_aggregate=config.get("quality_aggregate", "min"),
            clearance_potential_weight=config.get("clearance_potential_weight", 0.0),
            clearance_target=config.get("clearance_target", 4.0),
            best_so_far_reward=config.get("best_so_far_reward", False),
            best_so_far_all_quad=config.get("best_so_far_all_quad", False),
            terminate_when_usable=config.get("terminate_when_usable", False),
            usable_quality=config.get("usable_quality", 0.2),
            mask_inverse=config.get("mask_inverse", True),
            revisit_penalty=config.get("revisit_penalty", 0.2),
            deterministic_center=config.get("deterministic_center", True),
            global_observation=config.get("global_observation", True),
            global_features=config.get("global_features", "raw"),
            resample_if_at_par=config.get("resample_if_at_par", True),
            terminate_on_overflow=config.get("terminate_on_overflow", False),
            target_in_observation=config.get("target_in_observation", False),
            tie_aware_scoring=tie_aware_scoring,
            tie_tolerance=tie_tolerance,
            center_mode=center_mode,
        )

    @staticmethod
    def get_num_actions_per_half_edge(max_edge_addition_steps, allow_delete=False):
        # chords, plus insert-vertex, plus (optionally) the two deletes
        return max_edge_addition_steps + (3 if allow_delete else 1)

    @staticmethod
    def get_num_actions_per_halfedge(max_edge_addition_steps, allow_delete=False):
        return AngleEnv.get_num_actions_per_half_edge(max_edge_addition_steps, allow_delete)

    @staticmethod
    def get_feature_size():
        return 11

    def _compute_max_steps(self):
        return max(
            self.min_max_steps,
            int(np.ceil(self.max_steps_factor * self.graph.number_of_half_edges())),
        )

    @staticmethod
    def _get_feature_rotation_matrix(v1, v2):
        """
        return a rotation matrix that will align the vector from v1 to v2 with the x-axis
        """
        v = v2 - v1
        theta = np.arctan2(v[1], v[0])
        c = np.cos(theta)
        s = np.sin(theta)
        matrix = np.array([
            [c, s],
            [-s, c]
        ])
        return matrix

    def _update_half_edge_angles(self):
        self.half_edge_angles = self.graph.half_edge_angles()

    def global_l1_face_score(self):
        score = sum(abs(self.graph.face_degree(fidx) - self.face_desired_degree) for fidx in self.graph.face_list())
        return score

    def _update_global_face_score(self):
        self.global_face_score = self.global_l1_face_score()

    def global_l1_angle_score(self):
        score = sum(
            abs(self.half_edge_angles[hidx] - self.desired_angle) / self.desired_angle for hidx
            in self.graph.half_edge_list()
        )
        return score

    def _update_global_angle_score(self):
        self.global_angle_score = self.global_l1_angle_score()

    def global_l1_vertex_score(self):
        score = sum(self.vertex_defect(vidx)
                    for vidx in self.graph.vertex_list(tag=False))
        return score

    def _update_global_vertex_score(self):
        self.global_vertex_score = self.global_l1_vertex_score()

    def number_of_odd_faces(self):
        return sum(1 for fidx in self.graph.face_list() if self.graph.face_degree(fidx) % 2 == 1)

    def get_face_score(self):
        return self.global_face_score

    def get_vertex_score(self):
        return self.global_vertex_score

    def get_angle_score(self):
        return self.global_angle_score

    def get_score(self):
        return self.score

    def get_initial_score(self):
        return self.initial_score

    def _compute_potential(self):
        """Negative distance-to-goal in units of "number of defects".

        With `quality_potential_weight` set, the element-quality shortfall joins
        it and the objective changes from "reach par" to "get as close to par as
        you can while keeping the mesh usable". That matters on domains where par
        is not reachable at all: the par-0 mesh of some curved outlines inverts
        (see `utilities/build_par0_mesh.py`), so chasing par there is chasing a
        mesh no smoother can draw.

        Putting the shortfall in the POTENTIAL rather than charging it once at
        the end is the whole point. At gamma 1 the two give the same return --
        the sum telescopes to the terminal value either way -- but in the
        potential every move that degrades quality is paid for when it happens,
        instead of the blame landing on whichever move ended the episode. The
        vocabulary is monotone, so an agent that cannot stop keeps refining;
        this is what makes over-refinement cost something immediately.
        """
        vertex_defect = abs(self.global_vertex_score - self.par)
        potential = -(self.face_potential_weight * self.global_face_score
                      + self.vertex_potential_weight * vertex_defect)
        if self.quality_potential_weight:
            potential -= self.quality_potential_weight * self.quality_shortfall()
        if self.boundary_underdegree_weight:
            potential -= self.boundary_underdegree_weight * self.boundary_underdegree()
        if self.clearance_potential_weight:
            potential -= self.clearance_potential_weight * self.clearance_shortfall()
        return potential

    def clearance_shortfall(self):
        """How thin the elements are against the curvature they sit on.

        The quality terms judge the mesh AS DRAWN, with every boundary edge a
        straight chord. That is not the mesh anyone uses: a block decomposition
        is refined to the target size, and on a curved boundary refinement puts
        the new vertices ON the curve, so the boundary bulges outward by the
        arc's sagitta while the interior stays where the chord left it. The
        element in between absorbs the displacement, and a shallow one has less
        to give. Measured over 55 arc edges, the ratio of an element's depth to
        its arc's sagitta -- its CLEARANCE -- orders the quality of what that
        element becomes, monotonically across quartiles: +0.344, +0.350, +0.505,
        +0.586 (Spearman +0.475).

        So a thin block against a tightly curved edge is a defect the agent
        cannot currently see, because on the chord it scores perfectly well.
        This charges for it directly, at one sagitta and one point-to-line
        distance per arc edge -- cheap enough for the step loop, where actually
        refining is not.

        Zero when the domain has no arcs, so straight training is untouched.
        """
        graph = self.graph
        arcs = getattr(graph, "boundary_arcs", None)
        if not arcs or self.clearance_target <= 0:
            return 0.0
        # One pass over the faces, not one per arc: the same walk was O(arcs x
        # faces) and this runs on every step of every episode.
        by_edge = {}
        for face in graph.face_list():
            loop = graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))
            points = [graph.source_vertex(h, tag=False) for h in loop]
            for where, vertex in enumerate(points):
                ahead = points[(where + 1) % len(points)]
                by_edge.setdefault(frozenset((vertex, ahead)), []).append(points)

        worst = 0.0
        for (first, second), (arc, _) in arcs.items():
            faces = by_edge.get(frozenset((first, second)))
            if not faces:
                continue
            start = np.asarray(graph.vertex_coordinate(first), dtype=float)
            finish = np.asarray(graph.vertex_coordinate(second), dtype=float)
            sagitta = float(np.linalg.norm(arc.point(0.5) - 0.5 * (start + finish)))
            if sagitta < 1e-12:
                continue
            along = finish - start
            length = float(np.linalg.norm(along))
            if length < 1e-12:
                continue
            normal = np.array([-along[1], along[0]], dtype=float) / length
            depth = None
            for points in faces:
                here = [abs(float((np.asarray(graph.vertex_coordinate(v), dtype=float)
                                   - start) @ normal))
                        for v in points if v not in (first, second)]
                if here:
                    depth = min(here) if depth is None else min(depth, min(here))
            if depth is None:
                continue
            worst = max(worst, 1.0 - min(depth / sagitta, self.clearance_target)
                        / self.clearance_target)
        return worst

    def _depth_across_edge(self, first, second):
        """How deep the face on this boundary edge reaches away from it.

        The distance from the edge's line to the nearest vertex of its face that
        is not an endpoint of the edge. None when no face carries the edge,
        which happens while the outline is still one unmeshed loop. Kept for
        tests and inspection; `clearance_shortfall` walks the faces once itself.
        """
        graph = self.graph
        start = np.asarray(graph.vertex_coordinate(first), dtype=float)
        finish = np.asarray(graph.vertex_coordinate(second), dtype=float)
        along = finish - start
        length = float(np.linalg.norm(along))
        if length < 1e-12:
            return None
        normal = np.array([-along[1], along[0]], dtype=float) / length
        best = None
        for face in graph.face_list():
            loop = graph.generate_half_edge_face_loop(graph.first_face_halfedge(face))
            points = [graph.source_vertex(h, tag=False) for h in loop]
            if first not in points or second not in points:
                continue
            count = len(points)
            where = points.index(first)
            if points[(where + 1) % count] != second and points[where - 1] != second:
                continue
            depths = [abs(float((np.asarray(graph.vertex_coordinate(v), dtype=float)
                                 - start) @ normal))
                      for v in points if v not in (first, second)]
            if not depths:
                continue
            here = min(depths)
            best = here if best is None else min(best, here)
        return best

    def candidate_score(self):
        """What this state would be worth if the episode were cut off here.

        The episode is run to `max_steps` and the mesh that counts is the BEST
        state along the way, not the last one -- so the quantity to maximise is
        the score of that best state. Writing it out:

            score(s_k) = potential(s_k) + par_bonus * [at par] - step_cost * k

        The step charge is inside the max, not outside it, and that is the whole
        reason this works. Charged at the END it is `step_cost * max_steps`,
        which is fixed at reset and therefore a constant offset per episode --
        it cancels out of the policy gradient and the agent has no reason to be
        brief. Charged against each CANDIDATE, a mesh reached in twenty moves
        beats an equally good one reached in forty, which is the frugality that
        took mean episode length 38 -> 21 and the score 125.8 -> 130.0 when it
        was first introduced.

        With this, `max_k score(s_k) - score(s_0)` is exactly the return that
        cutting the trajectory at its best candidate would give, including the
        charge for the steps up to it -- so the post-hoc truncation this
        replaces is not approximated, it is reproduced.
        """
        score = self.potential
        if self.reward_mode == "normalized":
            score = score / self.initial_defect
            charge = self.step_cost * self.num_steps / max(self.max_steps, 1)
        else:
            charge = self.step_cost * self.num_steps
        if not self.exception_occurred and self.is_at_par():
            score += self.par_bonus
        # Quality has to be IN the candidate, whichever way it is configured.
        # With `quality_potential_weight` set it already is, carried in the
        # potential. Without it, it is otherwise charged once at the end against
        # the FINAL state -- and a penalty on the final state is exactly what
        # running to max_steps is meant to stop mattering, so it would punish
        # the agent for moves it has no way to decline.
        if not self.quality_potential_weight and self.quality_penalty > 0:
            score -= self.quality_penalty * max(
                0.0, self.quality_threshold - self.min_element_quality())
        return score - charge

    def _best_so_far_reward(self):
        """Reward only for beating your own best: `max(0, score - best)`.

        Never negative, so wrecking the mesh after a good candidate costs
        nothing -- which is the point. The monotone vocabulary gives the agent
        no way to stop, so under any other rule it is punished for moves it
        cannot decline to make.
        """
        if self.best_so_far_all_quad and self.global_face_score > 0 and self.best_candidate is not None:
            return 0.0
        score = self.candidate_score()
        if self.best_candidate is None:
            self.best_candidate = score
            return 0.0
        gain = max(0.0, score - self.best_candidate)
        if score > self.best_candidate:
            self.best_step = self.num_steps
        self.best_candidate = max(self.best_candidate, score)
        return gain

    def quality_shortfall(self):
        """How far the FINISHED part of the mesh falls short of the quality bar.

        Measured over the corners of faces that are ALREADY the target element,
        and over nothing else. The earlier version gated the whole term on
        `face_score == 0`, which was meant to avoid judging a raw polygon (a
        reflex corner scores -1, a 1.4 shortfall on the start state) but had a
        worse consequence: quality contributed exactly zero until the mesh was
        already complete, so the agent got no guidance on HOW to finish, only
        that finishing was worth four-plus units. Measured on curved-holes, it
        bought that: all-quad went 9/16 -> 13/16 while the tenth percentile of
        per-mesh quality went +0.16 -> -0.33.

        Restricting to quad faces gives the signal from the first quad onwards
        and still never judges an unfinished face. Early on few faces qualify,
        so the term is naturally small and grows as the mesh comes together.
        """
        qualities = self._target_face_qualities()
        if not qualities:
            return 0.0
        ideal = np.sin(np.radians(self.desired_angle))
        if self.quality_aggregate == "sum":
            # EXTENSIVE: one shortfall per finished face, its worst corner, summed.
            # The other aggregates are one bounded number per mesh, and against
            # the face term (one charge per open face, 10-30 of them on the way)
            # that let the policy amortise a single bad element against many
            # closings: face weight 5 took min+mean to a median quality of 0.00
            # with every face closed. Summed per face, each bad element is
            # charged in full, one flat quad costs 0.4 * w whether it is alone
            # or one of twenty, and the trade "close this face with this quad"
            # is the same on every domain size. Keep 0.4 * w at or below the
            # face weight so closing a face is never a net loss.
            return sum(max(0.0, self.quality_threshold - q / ideal)
                       for q in self._target_face_worst_corners())
        shortfalls = [max(0.0, self.quality_threshold - q / ideal) for q in qualities]
        worst = max(shortfalls)
        average = sum(shortfalls) / len(shortfalls)
        if self.quality_aggregate == "mean":
            return average
        if self.quality_aggregate == "min+mean":
            return 0.7 * worst + 0.3 * average
        return worst

    def _target_face_worst_corners(self):
        """The worst raw corner quality of each face that is already a quad."""
        if self.quality_metric == "shape":
            corners = self.graph.corner_shape_qualities()
        else:
            corners = {h: np.sin(np.radians(a))
                       for h, a in self.half_edge_angles.items()}
        out = []
        for face in self.graph.face_list():
            if self.graph.face_degree(face) != self.face_desired_degree:
                continue
            own = [corners[h] for h in self.graph.face_half_edges(face) if h in corners]
            if own:
                out.append(min(own))
        return out

    def _target_face_qualities(self):
        """Raw corner qualities, for corners of faces that are already quads."""
        if self.quality_metric == "shape":
            corners = self.graph.corner_shape_qualities()
        else:
            corners = {h: np.sin(np.radians(a))
                       for h, a in self.half_edge_angles.items()}
        out = []
        for face in self.graph.face_list():
            if self.graph.face_degree(face) != self.face_desired_degree:
                continue
            for half_edge in self.graph.face_half_edges(face):
                if half_edge in corners:
                    out.append(corners[half_edge])
        return out

    def _capture_initial_defect(self):
        """How far from par this episode starts, for the normalized reward.

        `reward_mode="normalized"` divides the potential difference by this, so
        the return telescopes to the FRACTION of the initial defect recovered
        rather than its absolute size. Without it a 16-gon starting 20 defects
        from par earns four times a 6-gon's return for the same quality of play,
        and across a degree curriculum the critic has to fit both while the
        advantages are dominated by the large instances.

        Floored at one, not asserted: `reset` resamples away from an already
        solved start, but a start that is topologically at par and only fails
        the quality test (a 4-gon with a collinear corner, which the certified
        generator does produce) is not resampled, and `_reset_to_state` and
        search replay never resample. The defect is a sum of integers, so a
        floor of 1 changes nothing for any episode that has work to do and
        keeps a zero-defect start's rewards O(1); a floor of 1e-6 turned them
        into rewards of 1e6 and value losses of 1e11.
        """
        return max(-self._compute_potential(), 1.0)

    def _update_scores_on_reset(self):
        self._update_global_face_score()
        self._update_global_angle_score()
        self._update_global_vertex_score()

        self.score = self.face_reward_weight * self.global_face_score + \
                     self.angle_reward_weight * self.global_angle_score + \
                     self.vertex_reward_weight * self.global_vertex_score
        self.min_score = self.score

        self.initial_score = self.score
        self.initial_face_score = max(self.global_face_score, 1)
        self.initial_angle_score = max(self.global_angle_score, 1)
        self.initial_vertex_score = max(self.global_vertex_score, 1)

        self.reward = 0

    def _half_edge_angle_score(self, hidx):
        angle = self.half_edge_angles[hidx]
        score = abs(angle - self.desired_angle) / self.desired_angle
        return score

    def _half_edge_face_score(self, hidx):
        fidx = self.graph.face(hidx)
        fdegree = self.graph.face_degree(fidx)
        score = abs(fdegree - self.face_desired_degree)
        return score

    def _half_edge_vertex_score(self, hidx):
        vidx = self.graph.source_vertex(hidx, tag=False)
        return self.vertex_defect(vidx)

    def _half_edge_score(self, hidx):
        angle_score = self._half_edge_angle_score(hidx)
        face_score = self._half_edge_face_score(hidx)
        vertex_score = self._half_edge_vertex_score(hidx)
        score = self.face_reward_weight * face_score + \
                self.angle_reward_weight * angle_score + \
                self.vertex_reward_weight * vertex_score

        return score

    def _face_irregularity(self, fidx):
        return abs(self.graph.face_degree(fidx) - self.face_desired_degree)

    def _worst_vertex_half_edge(self, half_edges):
        """Of these, the one whose source vertex is furthest from what it wants.

        Lowest half-edge id breaks a tie, so a state always maps to one centre.
        """
        if not half_edges:
            return None
        best = max(self.vertex_defect(self.graph.source_vertex(h, tag=False))
                   for h in half_edges)
        return min(h for h in half_edges
                   if self.vertex_defect(self.graph.source_vertex(h, tag=False)) == best)

    def _surviving_center_face(self):
        """The face the centre belongs to, after whatever the last move did.

        A delete can take the centre half-edge with it, and the face it was on
        may or may not be the one that survives: `delete_half_edge` keeps the
        deleted edge's OWN face and absorbs its twin's into that, so a centre
        sitting on the twin side loses its face id even though the region is
        still there, merged. Recording the centre's face id would miss that,
        which is why witnesses from its face loop are kept instead -- a move
        deletes at most one half-edge pair, so on a face of degree three or more
        at least one witness survives and names the merged face.
        """
        centre = getattr(self, "template_center", None)
        if centre is not None and self.graph.is_half_edge(centre):
            return self.graph.face(centre)
        for witness in getattr(self, "_center_face_witnesses", ()):
            if self.graph.is_half_edge(witness):
                return self.graph.face(witness)
        return None

    def _center_anchor(self):
        """Where the window currently sits, whether or not its centre survived."""
        centre = getattr(self, "template_center", None)
        if centre is not None and self.graph.is_half_edge(centre):
            return centre
        for witness in getattr(self, "_center_face_witnesses", ()):
            if self.graph.is_half_edge(witness):
                return witness
        return None

    def _template_neighbourhood(self, anchor):
        """The window around `anchor` on the CURRENT topology.

        Deliberately not `self.index_to_half_edge`: the centre is chosen before
        the template is rebuilt, so that list is a step stale -- it lacks the
        half-edges the last move created, which are exactly at the edge of the
        work, and may still name ones a delete removed.
        """
        if anchor is None or not self.graph.is_half_edge(anchor):
            return []
        if self.use_boundary:
            found = self.graph.knn_half_edges_with_boundary(
                anchor, self.template_size, seed_face=self.template_seed_face)
        else:
            found = self.graph.knn_half_edges(
                anchor, self.template_size, seed_face=self.template_seed_face)
        return [h for h in found if self.graph.is_half_edge(h)]

    def _remember_center_face(self):
        centre = getattr(self, "template_center", None)
        if centre is None or not self.graph.is_half_edge(centre):
            self._center_face_witnesses = ()
            return
        self._center_face_witnesses = tuple(
            self.graph.generate_half_edge_face_loop(centre))

    def _sticky_center(self):
        """Work one face until it is finished, then move to the worst one left.

        The global-argmax rule re-picks a centre from scratch every step, so a
        single chord can teleport the window to an unrelated part of the mesh
        and consecutive observations describe different workspaces. It also lets
        geometry decide a topological question: its angle term is a function of
        `half_edge_angles`, which Laplacian smoothing moves on every step.

        This keeps the centre while its face is still unfinished -- only the
        template around it is rebuilt -- and re-picks it, by face irregularity
        then vertex defect, once that face is the element we asked for. Both of
        those are integers, so smoothing cannot move the decision.

        When a delete removes the centre itself, the window stays on the
        surviving merged face rather than starting over globally: the region is
        still the one being worked, and re-picking from scratch there would give
        back exactly the teleporting this rule exists to stop.
        """
        face = self._surviving_center_face()
        if face is not None and self._face_irregularity(face) != 0:
            centre = getattr(self, "template_center", None)
            if centre is not None and self.graph.is_half_edge(centre):
                return centre      # still work to do here; do not move
            return self._worst_vertex_half_edge(self._half_edges_of(face))

        # The face is finished. Prefer unfinished work still inside the window
        # over starting again somewhere else: the agent keeps the context it has
        # just built up, and the centre hops to an adjacent piece of the problem
        # instead of teleporting across the mesh.
        nearby = [h for h in self._template_neighbourhood(self._center_anchor())
                  if self._face_irregularity(self.graph.face(h)) != 0]
        if nearby:
            worst = max(self._face_irregularity(self.graph.face(h)) for h in nearby)
            return self._worst_vertex_half_edge(
                [h for h in nearby
                 if self._face_irregularity(self.graph.face(h)) == worst])

        # Nothing unfinished is in view, so the window has to move somewhere it
        # cannot see. This is the only case that looks at the whole mesh.
        faces = self.graph.face_list()
        if not faces:
            return None
        worst = max(self._face_irregularity(f) for f in faces)
        if worst == 0:
            # every face is the element we asked for, so no face is irregular;
            # fall back to the worst vertex anywhere
            candidates = self.graph.half_edge_list()
        else:
            candidates = [h for h in self.graph.half_edge_list()
                          if self._face_irregularity(self.graph.face(h)) == worst]
        return self._worst_vertex_half_edge(candidates)

    def _half_edges_of(self, fidx):
        return [h for h in self.graph.half_edge_list()
                if self.graph.face(h) == fidx]

    def _set_half_edge_template_center(self, half_edges):
        if self.center_mode == "sticky_face":
            centre = self._sticky_center()
            if centre is not None:
                self.template_center = centre
                self._remember_center_face()
                return
        half_edge_scores = np.array([self._half_edge_score(hidx) for hidx in half_edges])
        max_indices = np.nonzero(half_edge_scores == half_edge_scores.max())[0]
        if self.deterministic_center:
            # lowest half-edge id among the worst, so the window does not jump
            # around under a tie-break coin flip
            template_center_idx = min(max_indices, key=lambda i: half_edges[i])
        else:
            template_center_idx = np.random.choice(max_indices)
        self.template_center = half_edges[template_center_idx]

    def _build_template(self):
        if self.use_boundary:
            self.index_to_half_edge = self.graph.knn_half_edges_with_boundary(
                self.template_center, self.template_size,
                seed_face=self.template_seed_face)
        else:
            self.index_to_half_edge = self.graph.knn_half_edges(
                self.template_center, self.template_size,
                seed_face=self.template_seed_face)

        self.half_edge_to_index = {half_edge: idx for idx, half_edge in enumerate(self.index_to_half_edge)}

        self.template_faces = [self.graph.face(hidx) for hidx in self.index_to_half_edge]
        self.template_vertices = [self.graph.source_vertex(hidx, tag=False) for hidx in self.index_to_half_edge]

    def _compute_desired_options(self):
        """Corners whose angle leaves the desired degree genuinely undecided.

        `rounded_desired_degree` is `round(a / theta) + 1`, and when `a / theta`
        lands exactly on `k + 1/2` the two neighbouring corner treatments are
        both admissible: `k` elements of angle `a / k` or `k + 1` of `a / (k+1)`
        both sit the same distance either side of the target. Rounding picks one
        of them by a convention -- numpy's round-half-to-even, which does not
        even break consecutive ties in the same direction -- and that arbitrary
        choice then shows up in both the vertex score and par.

        So a tie corner carries a SET of admissible degrees rather than one.
        Irregularity is the distance to the nearer member, which keeps the score
        separable, and par is the smallest bound over the assignments, which
        keeps it a bound: the assignment minimising the score already has its own
        par below its own score, so `score >= par` survives.

        Only the domain's original corners can tie. Vertices the agent creates
        take the generic degree and are never ambiguous.
        """
        options = {}
        if not self.tie_aware_scoring:
            return options
        # the start state is the raw polygon: one face, so a half-edge's angle
        # is its source corner's interior angle, summed where a slit revisits.
        # On a CURVED outline the corner is measured from the tangents, not the
        # chords: every vertex of a filleted domain reads 180 and wants three,
        # where the chords would read 135, call it a tie, and hand par a bound
        # that no mesh of that domain can reach.
        if getattr(self.graph, "boundary_arcs", None):
            from src.boundary_arcs import corner_angles
            angles = corner_angles(self.graph)
        else:
            angles = {}
            for hidx in self.graph.half_edge_list():
                vidx = self.graph.source_vertex(hidx, tag=False)
                angles[vidx] = angles.get(vidx, 0.0) + self.half_edge_angles[hidx]
        for vidx, angle in angles.items():
            if vidx not in self.vertex_desired_degree:
                continue
            ratio = angle / self.desired_angle
            if abs(ratio - np.floor(ratio) - 0.5) > self.tie_tolerance:
                continue
            low = max(int(np.floor(ratio)) + 1, 2)
            high = max(int(np.ceil(ratio)) + 1, 2)
            if low != high:
                options[vidx] = (low, high)
        return options

    def desired_degrees(self, vidx):
        """Every degree this vertex would be content with."""
        options = self.vertex_desired_options.get(vidx)
        if options is not None:
            return options
        return (self.vertex_desired_degree[vidx],)

    def _flat_boundary_corner(self, vidx, edge_in, edge_out):
        """Is `vidx` a boundary vertex wanting >= 3 whose two sides in a face are
        both boundary edges? (`edge_in` ends at it, `edge_out` leaves it.)"""
        g = self.graph
        if not (g.is_boundary_half_edge(g.twin_half_edge(edge_in))
                and g.is_boundary_half_edge(g.twin_half_edge(edge_out))):
            return False
        return min(self.desired_degrees(vidx)) >= 3

    def _insert_closes_flat_quad(self, half_edge):
        """Would inserting a vertex on this edge turn a triangle into a flat quad?
        Either the new vertex is the flat corner (a GEOMETRIC boundary edge: the
        vertex wants 3 and is pinned between two boundary edges, which the smoother
        can never open) or the triangle's apex already is. An interior insertion's
        new vertex is not flagged, since the smoother moves it off the line. Both
        faces of an interior edge are checked."""
        g = self.graph
        h = g._ensure_tag_form(half_edge, g.half_edge_tag)
        twin = g.twin_half_edge(h)
        on_boundary = g.is_boundary_half_edge(twin)
        for side in ((h,) if on_boundary else (h, twin)):
            face = g.face(side)
            if g.face_degree(face) != 3:
                continue
            if on_boundary:
                return True
            loop = g.generate_half_edge_face_loop(side)
            i = loop.index(side)
            e_in, e_out = loop[(i + 1) % 3], loop[(i + 2) % 3]     # the two edges at the apex
            if self._flat_boundary_corner(g.target_vertex(e_in, tag=False), e_in, e_out):
                return True
        return False

    def _chord_closes_flat_quad(self, half_edge, k):
        """Would the chord from source(half_edge) to the vertex k+1 ahead leave either
        resulting face a quad with a flat boundary corner? See `mask_flat_quads`."""
        g = self.graph
        h = g._ensure_tag_form(half_edge, g.half_edge_tag)
        loop = g.generate_half_edge_face_loop(h)
        n = len(loop)
        i0 = loop.index(h)
        # A face of FOUR sides with a flat corner is a degenerate quad; a face of
        # THREE sides with a flat apex is a dead end, since every insertion that
        # finishes it makes that quad. Faces of five or more sides are fine: the
        # flat vertex can still take a chord. So both new faces are checked when
        # they have at most four sides.
        # The cut-off face: loop edges i0 .. i0+k (k+1 edges) plus the chord, k+2
        # sides. Its interior vertices are the targets of edges i0 .. i0+k-1.
        if k + 2 <= 4:
            for j in range(k):
                e_in, e_out = loop[(i0 + j) % n], loop[(i0 + j + 1) % n]
                if self._flat_boundary_corner(g.target_vertex(e_in, tag=False), e_in, e_out):
                    return True
        # the remaining face: the other n-k-1 loop edges plus the chord, n-k sides
        # (the Tiler sets the old face's degree to degree - k). Its interior
        # vertices are the targets of edges i0+k+1 .. i0+n-2.
        if n - k <= 4:
            for j in range(k + 1, n - 1):
                e_in, e_out = loop[(i0 + j) % n], loop[(i0 + j + 1) % n]
                if self._flat_boundary_corner(g.target_vertex(e_in, tag=False), e_in, e_out):
                    return True
        return False

    def _thin_boundary_corner(self, vidx, edge_in, edge_out, chord_length=None):
        """Would the corner at `vidx`, between `edge_in` (ending at it) and `edge_out`
        (leaving it), have side lengths in a ratio above `mask_thin_quads`, with at
        least one side on the boundary? `chord_length` replaces the length of the
        side that is the chord about to be inserted (passed as None edge)."""
        g = self.graph
        c = np.asarray(g.vertex_coordinate(vidx), float)

        def length(edge):
            other = g.source_vertex(edge, tag=False) if g.target_vertex(edge, tag=False) == vidx \
                else g.target_vertex(edge, tag=False)
            return float(np.linalg.norm(np.asarray(g.vertex_coordinate(other), float) - c))

        sides = []
        on_boundary = False
        for edge in (edge_in, edge_out):
            if edge is None:
                sides.append(chord_length)
            else:
                sides.append(length(edge))
                on_boundary = on_boundary or g.is_boundary_half_edge(g.twin_half_edge(edge))
        if not on_boundary or min(sides) <= 1e-12:
            return False
        return max(sides) / min(sides) > self.mask_thin_quads

    def _chord_closes_thin_quad(self, half_edge, k):
        """Would the chord from source(half_edge) to the vertex k+1 ahead leave either
        resulting face a quad with a THIN boundary corner -- a corner with one side on
        the boundary whose two sides differ in length by more than `mask_thin_quads`?
        The sliver the transfer diagnostic found: one short boundary side against one
        long chord, aspect 4.5-11.7. Checked at every corner of each new face of at
        most four sides, including the two corners the chord itself makes."""
        g = self.graph
        h = g._ensure_tag_form(half_edge, g.half_edge_tag)
        loop = g.generate_half_edge_face_loop(h)
        n = len(loop)
        i0 = loop.index(h)
        start = g.source_vertex(h, tag=False)
        end = g.target_vertex(loop[(i0 + k) % n], tag=False)
        chord = float(np.linalg.norm(np.asarray(g.vertex_coordinate(end), float)
                                     - np.asarray(g.vertex_coordinate(start), float)))
        if k + 2 <= 4:
            # cut-off face: edges i0..i0+k then the chord back to start
            for j in range(k):
                e_in, e_out = loop[(i0 + j) % n], loop[(i0 + j + 1) % n]
                if self._thin_boundary_corner(g.target_vertex(e_in, tag=False), e_in, e_out):
                    return True
            if self._thin_boundary_corner(end, loop[(i0 + k) % n], None, chord):
                return True
            if self._thin_boundary_corner(start, None, loop[i0], chord):
                return True
        if n - k <= 4:
            for j in range(k + 1, n - 1):
                e_in, e_out = loop[(i0 + j) % n], loop[(i0 + j + 1) % n]
                if self._thin_boundary_corner(g.target_vertex(e_in, tag=False), e_in, e_out):
                    return True
            if self._thin_boundary_corner(start, loop[(i0 + n - 1) % n], None, chord):
                return True
            if self._thin_boundary_corner(end, None, loop[(i0 + k + 1) % n], chord):
                return True
        return False

    def boundary_underdegree(self):
        """Sum over boundary vertices of how far the degree sits BELOW the want it
        is measured against (0 when at or above). See `boundary_underdegree_weight`."""
        total = 0
        for vidx in self.graph.vertex_list(tag=False):
            if not self.graph.is_boundary_vertex(vidx):
                continue
            gap = self.effective_desired_degree(vidx) - self.graph.vertex_degree(vidx)
            if gap > 0:
                total += gap
        return total

    def vertex_defect(self, vidx):
        """Distance from this vertex's degree to the nearest degree it wants."""
        degree = self.graph.vertex_degree(vidx)
        return min(abs(degree - want) for want in self.desired_degrees(vidx))

    def effective_desired_degree(self, vidx):
        """The degree this vertex is currently being measured against."""
        degree = self.graph.vertex_degree(vidx)
        return min(self.desired_degrees(vidx), key=lambda want: abs(degree - want))

    def compute_par(self):
        """Lower bound on the vertex score for any all-quad mesh of this domain.

        Discrete Gauss-Bonnet: summing (desired - degree) over every vertex of
        an all-quad mesh always gives 4 * euler_characteristic. Boundary corners
        fix part of that sum in advance, so whatever the corners cannot supply
        must be carried by irregular vertices. The gap is the smallest vertex
        score any mesh of this domain can reach.

        The coefficient is the interior vertex's generic degree -- 4 for quads,
        6 for triangles -- not the constant 4 it looks like. For a mesh of
        d-gons, with f_b = d/(d-2) faces around a flat boundary vertex and
        f_i = 2*f_b around an interior one, d*F = 2E - B and Euler give
        sum_v (generic - degree) = 2*d*chi/(d-2) = f_i * chi.

        The quantity is invariant under all four edit operations, so it only
        needs computing once per episode.
        """
        corner_excess = 0
        num_ties = 0
        for vidx in self.graph.vertex_list(tag=False):
            generic = (self.boundary_vertex_desired_degree
                       if self.graph.is_boundary_vertex(vidx)
                       else self.interior_vertex_desired_degree)
            options = self.desired_degrees(vidx)
            # the two options at a tie always differ by one, so taking `t` of
            # them high shifts the excess by exactly `t`: par is a scan over
            # `t`, never an enumeration of the assignments
            corner_excess += min(options) - generic
            num_ties += len(options) - 1

        seen = set()
        num_edges = 0
        for hidx in self.graph.half_edge_list():
            if hidx in seen:
                continue
            seen.add(hidx)
            seen.add(self.graph.twin_half_edge(hidx))
            num_edges += 1

        euler = (len(self.graph.vertex_list()) - num_edges
                 + len(self.graph.face_list()))
        base = corner_excess + self.interior_vertex_desired_degree * euler
        return min(abs(base + taken) for taken in range(num_ties + 1))

    def min_element_quality(self):
        """Worst scaled Jacobian in the mesh, against *this* element's ideal.

        1.0 at the target angle, 0.0 at a collapsed or flat corner, negative
        when an element folds over itself. Dividing by sin of the desired angle
        is what makes it an element quality rather than a quad quality: it is
        the identity for quads (sin 90 = 1) and the standard triangle scaled
        Jacobian (2/sqrt3) sin(theta) for triangles. Every triangle has a corner
        at or below 60 degrees, so at an all-triangle goal state the value still
        lands in [0, 1].
        """
        if not self.half_edge_angles:
            return 1.0
        ideal = np.sin(np.radians(self.desired_angle))
        if self.quality_metric == "shape":
            qualities = self.graph.corner_shape_qualities()
            if not qualities:
                return 1.0
            return float(min(qualities.values())) / ideal
        return min(np.sin(np.radians(angle))
                   for angle in self.half_edge_angles.values()) / ideal

    def next_demo_action_for_self(self):
        """The forced action when this episode is replaying a demonstration.

        Named for reachability: a vectorised env can only call a method on its
        workers by name, and PPO's rollout collection asks each worker whether
        it is mid-demonstration before committing to a sampled action.
        """
        driver = getattr(self, "demo_driver", None)
        return None if driver is None else driver.next_action(self)

    def is_at_par(self):
        """True when the mesh is as good as this domain topologically allows."""
        if self.global_face_score != 0:
            return False
        if self.global_vertex_score != self.par:
            return False
        return self.min_element_quality() >= self.quality_threshold

    def is_topologically_at_par(self):
        return self.global_face_score == 0 and self.global_vertex_score == self.par

    def template_overflowed(self):
        """Is the mesh larger than the observation window?

        Not an error, and by design: the template is the `template_size`
        half-edges NEAREST the centre, so it is a fixed-size view of a mesh that
        may be any size, out-of-window neighbours already have their own learned
        embedding (the -2 sentinel), and the scores, `par` and the win test are
        all computed over the whole graph rather than the window. The one real
        consequence is that the action space only reaches half-edges currently in
        view, which is a per-step restriction, not a reason to end the episode.

        `terminate_on_overflow` therefore defaults to False. It was True when the
        template was 64 slots and running past it mostly meant the agent had lost
        the plot, where ending the episode kept noise out of the rollout buffer.
        Measured at template 160 on the straight-holes suite, turning it off
        changes nothing that can be measured -- 19 of 24 solved against 20, well
        inside the metric's own spread -- and removes an arbitrary ending.
        """
        return self.graph.number_of_half_edges() > self.template_size

    def is_terminated(self):
        if self.exception_occurred:
            return True

        if self.num_steps >= self.max_steps:
            return True

        if self.terminate_at_par and self.is_at_par():
            return True

        # An all-quad mesh whose elements are good enough IS the goal when the
        # objective is "lowest irregularity at acceptable quality". Without this
        # the episode runs on past it and the monotone vocabulary destroys it.
        if self.terminate_when_usable and self.global_face_score == 0 \
                and self.min_element_quality() >= self.usable_quality:
            return True

        if self.terminate_on_overflow and self.template_overflowed():
            return True

        if self.reward_mode == "legacy" and self.score == 0:
            return True

        return False

    def _normalize_coordinates(self, coordinates):
        rot = self._get_feature_rotation_matrix(coordinates[0], coordinates[1])
        centered_coordinates = (coordinates - coordinates[0])
        scale = max(np.linalg.norm(centered_coordinates, axis=1).max(), 0.1)
        scaled_coordinates = centered_coordinates / scale
        normalized_coordinates = (rot @ scaled_coordinates.transpose()).transpose()
        return normalized_coordinates

    def _get_coordinate_features(self, vertices):
        coordinates = np.array([self.graph.vertex_coordinate(v) for v in vertices])
        coordinates = self._normalize_coordinates(coordinates)
        return coordinates

    def _get_context_features(self):
        """Five cheap per-half-edge features about the ends of the half-edge.

        The chord actions land on vertices a few steps ahead in the face loop,
        so the target end deserves the same description the source end gets;
        recovering it through ten message-passing hops is wasted capacity.
        """
        n = len(self.index_to_half_edge)
        out = np.zeros((n, 5), dtype=np.float32)
        for idx, hidx in enumerate(self.index_to_half_edge):
            source = self.graph.source_vertex(hidx, tag=False)
            next_hidx = self.graph.next_half_edge(hidx)
            target = self.graph.source_vertex(next_hidx, tag=False)
            out[idx, 0] = float(self.graph.is_boundary_vertex(source))
            out[idx, 1] = float(self.graph.is_user_defined_vertex(source))
            out[idx, 2] = (self.effective_desired_degree(target)
                           if target in self.vertex_desired_degree
                           else self.interior_vertex_desired_degree)
            out[idx, 3] = min(self.graph.vertex_degree(target), self.vertex_degree_threshold)
            out[idx, 4] = self.half_edge_angles[next_hidx] / self.desired_angle
        return out

    def _get_feature_matrix(self):
        source_vertices = self.template_vertices
        faces = self.template_faces

        coordinate_features = self._get_coordinate_features(source_vertices)
        vertex_desired_degree = [self.effective_desired_degree(vidx)
                                 for vidx in source_vertices]
        vertex_irregularities = [
            min(self.graph.vertex_degree(vidx), self.vertex_degree_threshold) for vidx in source_vertices
        ]

        face_irregularities = [min(self.graph.face_degree(fidx), self.face_degree_threshold) for fidx in faces]

        angle_irregularities = [(self.half_edge_angles[hidx]) / self.desired_angle for hidx in self.index_to_half_edge]

        num_half_edges = len(self.index_to_half_edge)

        matrix = np.zeros((self.template_size, self.num_features), dtype=np.float32)
        matrix[:num_half_edges, [0, 1]] = coordinate_features
        matrix[:num_half_edges, 2] = vertex_desired_degree
        matrix[:num_half_edges, 3] = vertex_irregularities
        matrix[:num_half_edges, 4] = face_irregularities
        matrix[:num_half_edges, 5] = angle_irregularities
        matrix[:num_half_edges, 6:11] = self._get_context_features()

        return matrix

    def _get_global_features(self):
        vertex_defect = abs(self.global_vertex_score - self.par)
        num_half_edges = self.graph.number_of_half_edges()
        if self.global_features == "intensive":
            # per face, so every entry stays in its training range whatever the
            # domain size; the size itself is one bounded entry (tanh of faces/40)
            faces = max(len(self.graph.face_list()), 1)
            features = np.array([
                min(vertex_defect / faces, 2.0),
                min(self.global_face_score / faces, 1.0),
                min(self.number_of_odd_faces() / faces, 1.0),
                min(num_half_edges / max(self.template_size, 1), 2.0),
                float(np.clip(self.min_element_quality(), -1.0, 1.0)),
                min(self.par / faces, 2.0),
                float(np.tanh(faces / 40.0)),
            ], dtype=np.float32)
        else:
            features = np.array([
                min(vertex_defect, 20) / 5.0,
                min(self.global_face_score, 40) / 5.0,
                min(self.number_of_odd_faces(), 20) / 5.0,
                min(num_half_edges / max(self.template_size, 1), 2.0),
                float(np.clip(self.min_element_quality(), -1.0, 1.0)),
                min(self.par, 10) / 5.0,
                min(len(self.graph.face_list()), 40) / 10.0,
            ], dtype=np.float32)
        if self.best_so_far_reward:
            gap = 0.0 if self.best_candidate is None \
                else float(self.candidate_score() - self.best_candidate)
            features = np.concatenate([features, np.array(
                [float(np.clip(gap, -2.0, 2.0))], dtype=np.float32)])
        if self.target_in_observation:
            features = np.concatenate([features, np.array([
                self.face_desired_degree / 4.0,
                self.interior_vertex_desired_degree / 6.0,
            ], dtype=np.float32)])
        return features

    def _get_next_edges(self):
        next_edges = np.arange(self.template_size)
        for (idx, hidx) in enumerate(self.index_to_half_edge):
            if self.graph.is_half_edge(hidx):
                next_hidx = self.graph.next_half_edge(hidx)
                next_idx = self.half_edge_to_index.get(next_hidx, self.template_boundary_index)
                next_edges[idx] = next_idx
            else:
                next_edges[idx] = self.geometric_boundary_index

        return next_edges

    def _get_previous_edges(self):
        prev_edges = np.arange(self.template_size)
        for (idx, hidx) in enumerate(self.index_to_half_edge):
            if self.graph.is_half_edge(hidx):
                prev_hidx = self.graph.previous_half_edge(hidx)
                prev_idx = self.half_edge_to_index.get(prev_hidx, self.template_boundary_index)
                prev_edges[idx] = prev_idx
            else:
                prev_edges[idx] = self.geometric_boundary_index

        return prev_edges

    def _get_twin_edges(self):
        twin_edges = np.arange(self.template_size)
        for idx, hidx in enumerate(self.index_to_half_edge):
            if self.graph.is_half_edge(hidx):
                if self.graph.half_edge_on_boundary(hidx):
                    twin_edges[idx] = self.geometric_boundary_index
                else:
                    twin_hidx = self.graph.twin_half_edge(hidx)
                    twin_idx = self.half_edge_to_index.get(twin_hidx, self.template_boundary_index)
                    twin_edges[idx] = twin_idx
            else:
                twin_edges[idx] = self.geometric_boundary_index

        return twin_edges

    def _linear_action_index_to_half_edge_and_action(self, linear_action_index):
        hidx = linear_action_index // self.num_actions_per_half_edge
        local_action_index = linear_action_index % self.num_actions_per_half_edge

        if hidx < len(self.index_to_half_edge):
            half_edge = self.index_to_half_edge[hidx]
        else:
            half_edge = None

        return half_edge, local_action_index

    def chord_steps(self, local_action_index):
        """`k` handed to `insert_half_edge` for this local chord action."""
        return local_action_index + self._chord_offset

    def is_valid_action(self, linear_action_index):
        half_edge, local_action_index = self._linear_action_index_to_half_edge_and_action(linear_action_index)

        if not self.graph.is_half_edge(half_edge):
            return False

        if local_action_index < self.max_edge_addition_steps:
            if not self.graph.is_valid_chord_insert(
                    half_edge, self.chord_steps(local_action_index), self.min_face_degree):
                return False
            if self.mask_flat_quads and self._chord_closes_flat_quad(
                    half_edge, self.chord_steps(local_action_index)):
                return False
            if self.mask_thin_quads and self._chord_closes_thin_quad(
                    half_edge, self.chord_steps(local_action_index)):
                return False
            if self._inverse_block is not None:
                kind, payload = self._inverse_block
                if kind == "chord":
                    tagged = self.graph._ensure_tag_form(half_edge, self.graph.half_edge_tag)
                    source = self.graph.source_vertex(tagged, tag=False)
                    loop = self.graph.generate_half_edge_face_loop(tagged)
                    j = (loop.index(tagged) + self.chord_steps(local_action_index) + 1) % len(loop)
                    target = self.graph.source_vertex(loop[j], tag=False)
                    if frozenset((source, target)) == payload:
                        return False
            return True

        if local_action_index == self._insert_vertex_action:
            if self._inverse_block is not None:
                kind, payload = self._inverse_block
                if kind == "vertex" and half_edge == payload:
                    return False
            if self.mask_flat_quads and self._insert_closes_flat_quad(half_edge):
                return False
            return True

        if local_action_index == self._delete_edge_action:
            if not self.graph.is_valid_delete_half_edge(half_edge):
                return False
            if self._inverse_block is not None:
                kind, payload = self._inverse_block
                if kind == "delete_edge" and half_edge in payload:
                    return False
            return True

        if local_action_index == self._delete_vertex_action:
            if not self.graph.is_valid_delete_source_vertex(half_edge):
                return False
            if self._inverse_block is not None:
                kind, payload = self._inverse_block
                if kind == "delete_vertex":
                    if self.graph.source_vertex(half_edge, tag=False) == payload:
                        return False
            return True

        return True

    def _get_action_mask(self):
        action_mask = self._build_action_mask()
        # The flat-corner mask is a PREFERENCE layered on the validity mask, and on a
        # face with no other way forward it can forbid everything: a bare triangle
        # whose corners all want 2 has no chord, and every vertex insertion makes a
        # quad with a flat corner -- the intermediate state the mask exists to
        # refuse. An all -inf mask makes the policy's softmax NaN and kills the
        # process (seen on the paper's straight-holes suite). Fall back to the
        # validity mask for that state, so the episode continues and the reward,
        # not the mask, judges the flat quad it now has to make.
        if (self.mask_flat_quads or self.mask_thin_quads) and not np.isfinite(action_mask).any():
            flat, thin = self.mask_flat_quads, self.mask_thin_quads
            self.mask_flat_quads, self.mask_thin_quads = False, 0.0
            try:
                action_mask = self._build_action_mask()
            finally:
                self.mask_flat_quads, self.mask_thin_quads = flat, thin
        return action_mask

    def _build_action_mask(self):
        num_slots = len(self.index_to_half_edge)
        n_actions = self.num_actions_per_half_edge
        action_mask = np.full(self.total_num_actions_in_template, -np.inf, dtype=np.float32)
        for slot in range(num_slots):
            base = slot * n_actions
            for local in range(n_actions):
                if self.is_valid_action(base + local):
                    action_mask[base + local] = 0.0
        return action_mask

    def _get_progress(self):
        return np.array([self.num_steps / self.max_steps], dtype=np.float32)

    def _get_blank_obs(self):
        features = np.zeros((self.template_size, self.num_features), dtype=np.float32)
        next_edges = np.arange(self.template_size)
        prev_edges = np.arange(self.template_size)
        twin_edges = np.arange(self.template_size)
        mask = np.zeros(self.total_num_actions_in_template, dtype=np.float32)
        obs = {
            "features": features,
            "next": next_edges,
            "previous": prev_edges,
            "twin": twin_edges,
            "mask": mask,
            "progress": np.array([0.0], dtype=np.float32)
        }
        if self.global_observation:
            obs["global"] = np.zeros(self.global_feature_size, dtype=np.float32)
        return obs

    def _get_obs(self):
        if self.exception_occurred:
            return self._get_blank_obs()
        obs = {
            "features": self._get_feature_matrix(),
            "next": self._get_next_edges(),
            "previous": self._get_previous_edges(),
            "twin": self._get_twin_edges(),
            "mask": self._get_action_mask(),
            "progress": self._get_progress()
        }
        if self.global_observation:
            obs["global"] = self._get_global_features()
        return obs

    def _step_insert_edge(self, hidx, num_steps):
        if self.graph.is_valid_chord_insert(hidx, num_steps, self.min_face_degree):
            if self.mask_inverse:
                first_new = (self.graph.next_half_edge_index, self.graph.half_edge_tag)
                second_new = (self.graph.next_half_edge_index + 1, self.graph.half_edge_tag)
                self.graph.insert_half_edge(hidx, num_steps)
                self._inverse_block = ("delete_edge", frozenset((first_new, second_new)))
            else:
                self.graph.insert_half_edge(hidx, num_steps)

    def _step_insert_vertex(self, hidx):
        if self.graph.is_half_edge(hidx):
            self.graph.insert_vertex(hidx)
            new_vertex_idx = self.graph.target_vertex(hidx, tag=False)
            if self.graph.half_edge_on_boundary(hidx):
                self.vertex_desired_degree[new_vertex_idx] = self.boundary_vertex_desired_degree
            else:
                self.vertex_desired_degree[new_vertex_idx] = self.interior_vertex_desired_degree
            if self.mask_inverse:
                self._inverse_block = ("delete_vertex", new_vertex_idx)

    def _step_delete_edge(self, hidx):
        if self.graph.is_valid_delete_half_edge(hidx):
            source = self.graph.source_vertex(hidx, tag=False)
            target = self.graph.target_vertex(hidx, tag=False)
            # (endpoints are only used when mask_inverse is on)
            self.graph.delete_half_edge(hidx)
            if self.mask_inverse:
                self._inverse_block = ("chord", frozenset((source, target)))

    def _step_delete_source_vertex(self, hidx):
        if self.graph.is_valid_delete_source_vertex(hidx):
            source_vertex = self.graph.source_vertex(hidx, tag=False)
            self.graph.delete_source_vertex(hidx)
            self.vertex_desired_degree.pop(source_vertex)
            if self.mask_inverse:
                self._inverse_block = ("vertex", hidx)

    def _step_half_edge_action(self, half_edge, action):
        assert 0 <= action < self.num_actions_per_half_edge
        self._inverse_block = None

        if action < self.max_edge_addition_steps:
            self._step_insert_edge(half_edge, self.chord_steps(action))
        elif action == self._insert_vertex_action:
            self._step_insert_vertex(half_edge)
        elif action == self._delete_edge_action:
            self._step_delete_edge(half_edge)
        elif action == self._delete_vertex_action:
            self._step_delete_source_vertex(half_edge)
        else:  # pragma: no cover - guarded by the assert above
            raise ValueError("Unexpected local action index: " + str(action))

    def _get_reward(self):
        if self.exception_occurred:
            return 0
        else:
            return self.reward

    def _update_scores_on_step(self):
        prev_face_score = self.global_face_score
        prev_angle_score = self.global_angle_score
        prev_vertex_score = self.global_vertex_score
        prev_potential = self.potential

        self._update_global_face_score()
        self._update_global_angle_score()
        self._update_global_vertex_score()

        self.score = self.face_reward_weight * self.global_face_score + \
                     self.angle_reward_weight * self.global_angle_score + \
                     self.vertex_reward_weight * self.global_vertex_score

        if self.reward_mode == "legacy":
            face_reward = (prev_face_score - self.global_face_score) / self.initial_face_score
            angle_reward = (prev_angle_score - self.global_angle_score) / self.initial_angle_score
            vertex_reward = (prev_vertex_score - self.global_vertex_score) / self.initial_vertex_score
            self.reward = (self.face_reward_weight * face_reward +
                           self.angle_reward_weight * angle_reward +
                           self.vertex_reward_weight * vertex_reward)
        elif self.reward_mode == "normalized":
            # telescopes over the episode to 1 - final_defect/initial_defect,
            # so a return is the fraction of the starting defect recovered and
            # is comparable across polygon sizes. Delivered per step rather than
            # at the end: identical advantages at gamma = lambda = 1, and a
            # usable signal if either ever moves off 1.
            self.potential = self._compute_potential()
            if self.best_so_far_reward:
                self.reward = self._best_so_far_reward()
            else:
                self.reward = ((self.potential - prev_potential) / self.initial_defect
                               - self.step_cost / max(self.max_steps, 1))
        else:
            self.potential = self._compute_potential()
            if self.best_so_far_reward:
                self.reward = self._best_so_far_reward()
            else:
                self.reward = (self.potential - prev_potential) - self.step_cost

        self.min_score = min(self.score, self.min_score)

    def _apply_revisit_penalty(self):
        if self.revisit_penalty <= 0:
            return False
        fingerprint = weisfeiler_lehman_hash(
            self.graph, self.vertex_desired_degree,
            boundary_desired=self.boundary_vertex_desired_degree,
            interior_desired=self.interior_vertex_desired_degree,
        )
        revisited = fingerprint in self._visited
        self._visited.add(fingerprint)
        if revisited:
            self.reward -= self.revisit_penalty
        return revisited

    def step(self, linear_action_index):

        if self.num_steps >= self.max_steps:
            print("WARNING : NUM STEPS > MAX STEPS!!")  # this should not happen

        half_edge, local_action = self._linear_action_index_to_half_edge_and_action(linear_action_index)
        self.action_sequence.append((half_edge, local_action))

        revisited = False
        try:
            self._step_half_edge_action(half_edge, local_action)
            self.num_steps += 1

            # update the template center after step
            self.graph.smooth_vertices(num_iter=self.smooth_iterations)
            self._update_half_edge_angles()
            self._update_scores_on_step()
            revisited = self._apply_revisit_penalty()
            self._set_half_edge_template_center(self.graph.half_edge_list())
            self._build_template()

            self.terminated = self.is_terminated()
            observation = self._get_obs()

        except Exception as e:
            self.exception_occurred = True
            print("\n\n\tENCOUNTERED ENVIRONMENT EXCPETION\n\n")
            print(e)
            print("\n\n")
            self._log_exception()
            self.terminated = True
            observation = self._get_obs()

        reward = self._get_reward()

        at_par = (not self.exception_occurred) and self.is_at_par()
        self.ever_at_par = self.ever_at_par or at_par
        # `candidate_score` already carries par_bonus, so paying it again here
        # would double it and reinstate the "stop at the first par mesh" pull
        # that running to max_steps exists to remove
        if at_par and not self.best_so_far_reward:
            reward += self.par_bonus
        # The quality shortfall is charged on EVERY ending, not only on a
        # time-out. While it sat behind an `elif` the agent paid it only when it
        # failed, so the winning path carried no quality signal at all: reach
        # par, take the bonus, terminate, and how the mesh was drawn never
        # entered the return. Measured consequence, training under a 0.1
        # degeneracy gate: laplacian quality mean 0.696 against a baseline's
        # 0.799, with eight domains of 120 below 0.4 and one inverted.
        #
        # Solving stays strictly better than not: the same mesh timed out pays
        # the same shortfall AND forgoes par_bonus.
        # ... unless the shortfall is already in the potential, where it has been
        # charged step by step and telescopes to the same terminal value
        if (self.reward_mode not in ("legacy",) and self.terminated
                and not self.exception_occurred and self.quality_penalty > 0
                and not self.quality_potential_weight
                and not self.best_so_far_reward):
            shortfall = max(0.0, self.quality_threshold - self.min_element_quality())
            reward -= self.quality_penalty * shortfall

        return observation, reward, self.terminated, False, {
            "score": self.score,
            "par": self.par,
            "at_par": at_par,
            # what the trajectory achieved, as against where it happened to stop
            "best_at_par": self.ever_at_par,
            "moves_to_best": self.best_step,
            "revisited": revisited,
            "num_steps": self.num_steps,
            "polygon_degree": getattr(self, "polygon_degree", 0),
            "cost_to_go": getattr(self, "start_cost_to_go", -1),
        }

    def _log_exception(self):
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)

        exception_filename = str(uuid.uuid4()) + ".pkl"
        self.exception_count += 1
        exception_filepath = os.path.join(self.logdir, exception_filename)
        output_data = {
            "graph": self.initial_graph,
            "actions": self.action_sequence,
        }
        with open(exception_filepath, "wb") as output_file:
            pickle.dump(output_data, output_file)

        print("\n\n\tLOGGED EXCEPTED ENV TO : ", exception_filepath, "\n\n\t")

    def _reset_to_state(self, graph: Tiler, vertex_desired_degree):
        self.graph = graph
        self.vertex_desired_degree = vertex_desired_degree

        self._inverse_block = None
        self._visited = set()
        self._update_half_edge_angles()
        self.vertex_desired_options = self._compute_desired_options()
        self._set_half_edge_template_center(self.graph.half_edge_list())
        self._build_template()
        self._update_scores_on_reset()
        self.par = self.compute_par()
        self.potential = self._compute_potential()
        self.initial_defect = self._capture_initial_defect()
        # the start state is the first candidate, so the return measures the
        # improvement over it rather than its absolute value
        self.best_candidate = self.candidate_score() if self.best_so_far_reward else None
        self.ever_at_par = False
        self.best_step = 0

        num_half_edges = self.graph.number_of_half_edges()
        self.polygon_degree = num_half_edges
        self.max_steps = self._compute_max_steps()
        self.num_steps = 0
        self.terminated = self.is_terminated()

        self.initial_graph = deepcopy(self.graph)
        self.action_sequence = []
        self.exception_occurred = False

        obs = self._get_obs()

        return obs, {"score": self.score, "par": self.par}

    def _draw_initial_state(self):
        """Draw a start state, rejecting ones that are already solved.

        A start state at par teaches nothing: SB3 steps the env anyway, and the
        only available moves make the mesh worse, so the agent learns that
        acting on a good mesh is catastrophic.
        """
        graph, desired = self.graph_initializer()
        if not self.resample_if_at_par:
            return graph, desired
        for _ in range(self.max_reset_tries):
            probe = (deepcopy(self.graph), self.vertex_desired_degree,
                     self.vertex_desired_options)
            self.graph, self.vertex_desired_degree = graph, desired
            self._update_half_edge_angles()
            self.vertex_desired_options = self._compute_desired_options()
            self._update_global_face_score()
            self._update_global_vertex_score()
            self.par = self.compute_par()
            solved = self.is_at_par()
            (self.graph, self.vertex_desired_degree,
             self.vertex_desired_options) = probe
            if not solved:
                return graph, desired
            graph, desired = self.graph_initializer()
        raise RuntimeError(
            "graph_initializer produced only already-solved states in "
            f"{self.max_reset_tries} tries"
        )

    def reset(self, seed=None, options=None):
        graph, vertex_desired_degree = self._draw_initial_state()
        return self._reset_to_state(graph, vertex_desired_degree)

    def configure_initializer(self, **options):
        """Push curriculum settings into the initializer, from a VecEnv call.

        Each option `foo=value` is delivered to `initializer.set_foo(value)` if
        that setter exists and is ignored otherwise, so one controller can drive
        a mixture of initializers that support different knobs.
        """
        applied = {}
        for key, value in options.items():
            setter = getattr(self.graph_initializer, "set_" + key, None)
            if setter is not None:
                setter(value)
                applied[key] = value
        return applied

#######################################################################################################################
#######################################################################################################################
