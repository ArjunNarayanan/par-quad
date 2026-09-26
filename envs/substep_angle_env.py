import os
from src.tiler import Tiler
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
from copy import deepcopy
import pickle
import uuid
import numpy as np
import envs.polygon_utils as utils


class AngleEnv(gym.Env):
    def __init__(
            self,
            face_desired_degree,
            graph_initializer,
            template_size=20,
            max_steps_factor=2,
            num_substeps=10,
            logdir=None,
            max_edge_addition_steps=3,
            face_reward_weight=1 / 3,
            angle_reward_weight=1 / 3,
            vertex_reward_weight=1 / 3,
            use_boundary=False,
            smooth_iterations=5,
    ):
        super().__init__()
        self.graph_initializer = graph_initializer

        self.template_size = template_size
        self.max_steps_factor = max_steps_factor
        self.num_substeps = num_substeps
        self.num_steps = 0
        self.smooth_iterations = smooth_iterations

        if logdir is None:
            logdir = os.path.join(os.getcwd(), "experiments")

        self.logdir = logdir

        self.max_edge_addition_steps = max_edge_addition_steps
        self.num_actions_per_half_edge = self.get_num_actions_per_half_edge(max_edge_addition_steps)
        self.num_actions_per_halfedge = self.num_actions_per_half_edge
        self.total_num_actions_in_template = self.template_size * self.num_actions_per_half_edge

        self.face_desired_degree = face_desired_degree
        self.desired_angle = utils.average_face_angle(self.face_desired_degree)
        self.interior_vertex_desired_degree = utils.rounded_desired_degree(360, self.desired_angle) - 1
        self.boundary_vertex_desired_degree = utils.rounded_desired_degree(180, self.desired_angle)

        graph, vertex_desired_degree = self.graph_initializer()
        self.graph = graph
        self.vertex_desired_degree = vertex_desired_degree
        self.max_steps = int(self.max_steps_factor * self.graph.number_of_half_edges())

        self.face_reward_weight = face_reward_weight
        self.angle_reward_weight = angle_reward_weight
        self.vertex_reward_weight = vertex_reward_weight

        self.use_boundary = use_boundary
        self._update_half_edge_angles()
        self._set_half_edge_template_center(self.graph.half_edge_list())
        self._build_template()
        self._update_scores_on_reset()

        self.exception_occurred = False
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
        self.vertex_degree_threshold = 3
        self.face_degree_threshold = 10

        self.observation_space = Dict(
            {
                "features": Box(low=0, high=10, shape=(self.template_size, self.num_features)),
                "next": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "previous": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "twin": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "mask": Box(low=-np.inf, high=0, shape=(self.total_num_actions_in_template,)),
                "progress": Box(low=0, high=1, shape=(1,))
            }
        )

    @classmethod
    def from_config(cls, config):
        face_desired_degree = config["face_desired_degree"]
        graph_initializer = config["graph_initializer"]

        template_size = config["template_size"]
        max_steps_factor = config.get("max_steps_factor", 2)
        num_substeps = config.get("num_substeps", 10)
        logdir = config.get("logdir", None)
        max_edge_addition_steps = config.get("max_edge_addition_steps", 3)

        face_reward_weight = config.get("face_reward_weight", 1 / 3)
        angle_reward_weight = config.get("angle_reward_weight", 1 / 3)
        vertex_reward_weight = config.get("vertex_reward_weight", 1 / 3)

        use_boundary = config.get("use_boundary", False)
        smooth_iterations = config.get("smooth_iterations", 5)

        return cls(
            face_desired_degree,
            graph_initializer,
            template_size=template_size,
            max_steps_factor=max_steps_factor,
            num_substeps=num_substeps,
            logdir=logdir,
            max_edge_addition_steps=max_edge_addition_steps,
            face_reward_weight=face_reward_weight,
            angle_reward_weight=angle_reward_weight,
            vertex_reward_weight=vertex_reward_weight,
            use_boundary=use_boundary,
            smooth_iterations=smooth_iterations
        )

    @staticmethod
    def get_num_actions_per_half_edge(max_edge_addition_steps):
        # 3 more actions are - delete edge, insert vertex, delete vertex
        return max_edge_addition_steps + 3

    @staticmethod
    def get_num_actions_per_halfedge(max_edge_addition_steps):
        # 3 more actions are - delete edge, insert vertex, delete vertex
        return max_edge_addition_steps + 3

    @staticmethod
    def get_feature_size():
        return 6

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

    def template_l1_face_score(self):
        score = sum(abs(self.graph.face_degree(fidx) - self.face_desired_degree) for fidx in self.unique_template_faces)
        return score

    def _update_template_face_scores(self):
        self.template_face_score = self.template_l1_face_score()

    def get_total_face_score(self):
        score = sum(abs(self.graph.face_degree(fidx) - self.face_desired_degree) for fidx in self.graph.face_list())
        return score

    def template_l1_angle_score(self):
        score = sum(
            abs(self.half_edge_angles[hidx] - self.desired_angle) / self.desired_angle for hidx
            in self.half_edge_to_index
        )
        return score

    def _update_template_angle_scores(self):
        self.template_angle_score = self.template_l1_angle_score()

    def template_l1_vertex_score(self):
        score = sum(
            abs(self.graph.vertex_degree(vidx) - self.vertex_desired_degree[vidx]) for vidx in
            self.unique_template_vertices
        )
        return score

    def _update_template_vertex_scores(self):
        self.template_vertex_score = self.template_l1_vertex_score()

    def is_terminated(self):
        if self.exception_occurred:
            return True

        if self.num_steps % self.num_substeps == 0:
            return True

        if self.num_steps >= self.max_steps:
            return True

        return False

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
        vdegree = self.graph.vertex_degree(vidx)
        vdesired = self.vertex_desired_degree[vidx]
        score = abs(vdesired - vdegree)
        return score

    def _half_edge_score(self, hidx):
        angle_score = self._half_edge_angle_score(hidx)
        face_score = self._half_edge_face_score(hidx)
        vertex_score = self._half_edge_vertex_score(hidx)
        score = self.face_reward_weight * face_score + \
                self.angle_reward_weight * angle_score + \
                self.vertex_reward_weight * vertex_score

        return score

    def _set_half_edge_template_center(self, half_edges):
        half_edge_scores = np.array([self._half_edge_score(hidx) for hidx in half_edges])
        max_indices = np.nonzero(half_edge_scores == half_edge_scores.max())[0]
        template_center_idx = np.random.choice(max_indices)
        self.template_center = half_edges[template_center_idx]

    def _build_template(self):
        if self.use_boundary:
            self.index_to_half_edge = self.graph.knn_half_edges_with_boundary(self.template_center, self.template_size)
        else:
            self.index_to_half_edge = self.graph.knn_half_edges(self.template_center, self.template_size)

        self.template_faces = [self.graph.face(hidx) for hidx in self.index_to_half_edge]
        self.unique_template_faces = set(self.template_faces)

        self.template_vertices = [self.graph.source_vertex(hidx, tag=False) for hidx in self.index_to_half_edge]
        self.unique_template_vertices = set(self.template_vertices)

        self.half_edge_to_index = {half_edge: idx for idx, half_edge in enumerate(self.index_to_half_edge)}

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

    def _get_feature_matrix(self):
        source_vertices = self.template_vertices
        faces = self.template_faces

        coordinate_features = self._get_coordinate_features(source_vertices)
        vertex_desired_degree = [self.vertex_desired_degree[vidx] for vidx in source_vertices]
        vertex_irregularities = [
            min((self.graph.vertex_degree(vidx) - vdesired) / vdesired, self.vertex_degree_threshold) for
            vidx, vdesired in zip(source_vertices, vertex_desired_degree)
        ]

        fdesired = self.face_desired_degree
        face_irregularities = [
            min((self.graph.face_degree(fidx) - fdesired) / fdesired, self.face_degree_threshold) for
            fidx in faces
        ]

        angle_irregularities = [
            (self.half_edge_angles[hidx] - self.desired_angle) / self.desired_angle for
            hidx in self.index_to_half_edge
        ]

        num_half_edges = len(self.index_to_half_edge)

        matrix = np.zeros((self.template_size, self.num_features), dtype=np.float32)
        matrix[:num_half_edges, [0, 1]] = coordinate_features
        matrix[:num_half_edges, 2] = vertex_desired_degree
        matrix[:num_half_edges, 3] = vertex_irregularities
        matrix[:num_half_edges, 4] = face_irregularities
        matrix[:num_half_edges, 5] = angle_irregularities

        return matrix

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

    def is_valid_action(self, linear_action_index):
        half_edge, local_action_index = self._linear_action_index_to_half_edge_and_action(linear_action_index)

        if not self.graph.is_half_edge(half_edge):
            return False

        if local_action_index < self.max_edge_addition_steps:
            if not self.graph.is_valid_edge_insert(half_edge, local_action_index):
                return False

        if local_action_index == self.max_edge_addition_steps:
            if not self.graph.is_valid_delete_half_edge(half_edge):
                return False

        if local_action_index == self.num_actions_per_half_edge - 1:
            if not self.graph.is_valid_delete_source_vertex(half_edge):
                return False

        return True

    def _get_action_mask(self):
        total_num_actions = self.total_num_actions_in_template
        action_mask = np.array([0 if self.is_valid_action(idx) else -np.inf for idx in range(total_num_actions)],
                               dtype=np.float32)
        return action_mask

    def _get_progress(self):
        return np.array([(self.num_steps % self.num_substeps) / self.num_substeps], dtype=np.float32)

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
        return obs

    def _get_obs(self):
        if self.exception_occurred:
            return self._get_blank_obs()
        else:
            obs = {
                "features": self._get_feature_matrix(),
                "next": self._get_next_edges(),
                "previous": self._get_previous_edges(),
                "twin": self._get_twin_edges(),
                "mask": self._get_action_mask(),
                "progress": self._get_progress()
            }
            return obs

    def _step_insert_edge(self, hidx, num_steps):
        if self.graph.is_valid_edge_insert(hidx, num_steps):
            self.graph.insert_half_edge(hidx, num_steps)

    def _step_insert_vertex(self, hidx):
        if self.graph.is_half_edge(hidx):
            self.graph.insert_vertex(hidx)
            new_vertex_idx = self.graph.target_vertex(hidx, tag=False)
            if self.graph.half_edge_on_boundary(hidx):
                self.vertex_desired_degree[new_vertex_idx] = self.boundary_vertex_desired_degree
            else:
                self.vertex_desired_degree[new_vertex_idx] = self.interior_vertex_desired_degree

    def _step_delete_edge(self, hidx):
        if self.graph.is_valid_delete_half_edge(hidx):
            self.graph.delete_half_edge(hidx)

    def _step_delete_source_vertex(self, hidx):
        if self.graph.is_valid_delete_source_vertex(hidx):
            source_vertex = self.graph.source_vertex(hidx, tag=False)
            self.graph.delete_source_vertex(hidx)
            self.vertex_desired_degree.pop(source_vertex)

    def _step_half_edge_action(self, half_edge, action):
        assert 0 <= action < self.num_actions_per_half_edge

        if action < self.max_edge_addition_steps:
            self._step_insert_edge(half_edge, action)
        elif action == self.max_edge_addition_steps:
            self._step_delete_edge(half_edge)
        elif action == self.max_edge_addition_steps + 1:
            self._step_insert_vertex(half_edge)
        else:  # delete vertex
            self._step_delete_source_vertex(half_edge)

    def _local_reset_template_center(self):
        # set the first valid half-edge in the existing template as the new template center
        self.template_center = None
        for hidx in self.index_to_half_edge:
            if self.graph.is_half_edge(hidx):
                self.template_center = hidx
                break

    def _get_reward(self):
        return self.reward

    def step(self, linear_action_index):

        if self.num_steps >= self.max_steps:
            print("WARNING : NUM STEPS > MAX STEPS!!")  # this should not happen

        half_edge, local_action = self._linear_action_index_to_half_edge_and_action(linear_action_index)
        self.action_sequence.append((half_edge, local_action))

        try:
            self._step_half_edge_action(half_edge, local_action)
            self.num_steps += 1

            # update the template center after step
            self._local_reset_template_center()
            self._build_template()
            self._update_scores_on_step()

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

        return observation, reward, self.terminated, False, {"score": self.score}

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

    def _update_scores_on_step(self):
        prev_score = self.score
        self.graph.smooth_vertices(num_iter=self.smooth_iterations)

        self._update_half_edge_angles()
        self._update_template_face_scores()
        self._update_template_angle_scores()
        self._update_template_vertex_scores()

        self.score = self.face_reward_weight * self.template_face_score + \
                     self.angle_reward_weight * self.template_angle_score + \
                     self.vertex_reward_weight * self.template_vertex_score

        self.reward = prev_score - self.score
        self.min_score = min(self.score, self.min_score)

    def _update_scores_on_reset(self):
        self._update_template_face_scores()
        self._update_template_angle_scores()
        self._update_template_vertex_scores()

        self.score = self.face_reward_weight * self.template_face_score + \
                     self.angle_reward_weight * self.template_angle_score + \
                     self.vertex_reward_weight * self.template_vertex_score
        self.min_score = self.score

        self.initial_score = self.score
        self.initial_face_score = self.template_face_score
        self.initial_angle_score = self.template_angle_score
        self.initial_vertex_score = self.template_vertex_score

        self.reward = 0

    def _reset_to_state(self, graph: Tiler, vertex_desired_degree):
        self.graph = graph
        self.vertex_desired_degree = vertex_desired_degree

        self._update_half_edge_angles()
        self._set_half_edge_template_center(self.graph.half_edge_list())
        self._build_template()
        self._update_scores_on_reset()

        num_half_edges = self.graph.number_of_half_edges()
        self.max_steps = int(self.max_steps_factor * num_half_edges)
        self.num_steps = 0
        self.terminated = self.is_terminated()

        self.initial_graph = deepcopy(self.graph)
        self.action_sequence = []
        self.exception_occurred = False

        obs = self._get_obs()

        return obs, {"score": self.score}

    def hard_reset(self):
        graph, vertex_desired_degree = self.graph_initializer()
        return self._reset_to_state(graph, vertex_desired_degree)

    def soft_reset(self):
        self._update_half_edge_angles()
        self._set_half_edge_template_center(self.graph.half_edge_list())
        self._build_template()
        self._update_scores_on_reset()

        obs = self._get_obs()
        return obs, {"score": self.score}

    def reset(self, seed=None, options=None):
        if self.exception_occurred:
            return self.hard_reset()

        if self.num_steps >= self.max_steps:
            return self.hard_reset()

        return self.soft_reset()

#######################################################################################################################
#######################################################################################################################
