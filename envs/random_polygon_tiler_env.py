import os
from src.tiler import Tiler
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
from copy import deepcopy
import pickle
import uuid
import numpy as np
import envs.polygon_utils as utils


class RandomPolygonEnv(gym.Env):
    def __init__(
            self,
            face_desired_degree,
            graph_initializer,
            template_size=20,
            max_steps_factor=3,
            logdir=None,
            max_edge_addition_steps=3,
            face_reward_weight=0.5,
            vertex_reward_weight=0.5,
            incremental_reward=True,
            scale_reward=True,
            use_progress=True,
            use_boundary=False,
            eval_mode=False,
            fixed_reset=False
    ):
        super().__init__()
        self.graph_initializer = graph_initializer

        self.template_size = template_size
        self.max_steps_factor = max_steps_factor

        self.incremental_reward = incremental_reward
        self.scale_reward = scale_reward

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

        graph, desired_degree = self.graph_initializer()

        self.graph = graph
        self.vertex_desired_degree = desired_degree
        self.max_steps = int(self.max_steps_factor * self.graph.number_of_half_edges())

        self.num_steps = 0

        self.face_reward_weight = face_reward_weight
        self.vertex_reward_weight = vertex_reward_weight

        self.face_score = self.global_face_score()
        self.vertex_score = self.global_vertex_score()
        self.score = face_reward_weight * self.face_score + vertex_reward_weight * self.vertex_score
        self.initial_score = self.score
        self.min_score = self.score
        self.initial_face_score = self.face_score
        self.initial_vertex_score = self.vertex_score
        self.reward = 0

        self.eval_mode = eval_mode
        self.fixed_reset = fixed_reset

        self.exception_occurred = False
        self.terminated = self.is_terminated()

        # Attributes for excpetion handling
        self.initial_graph = deepcopy(self.graph)
        self.initial_vertex_desired_degree = self.vertex_desired_degree.copy()
        self.exception_count = 0
        self.action_sequence = []

        self.use_boundary = use_boundary
        self.template_center = self._select_half_edge_template_center(self.graph.half_edge_list())
        self._build_template()

        self.template_boundary_index = -2
        self.geometric_boundary_index = -1
        self.num_features = self.get_feature_size()
        self.use_progress = use_progress

        self.action_space = Discrete(self.num_actions_per_half_edge)
        self.clamp_features_max = 5
        self.observation_space = Dict(
            {
                "features": Box(low=0, high=self.clamp_features_max, shape=(self.template_size, self.num_features)),
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
        max_steps_factor = config["max_steps_factor"]
        max_edge_addition_steps = config.get("max_edge_addition_steps", 3)
        logdir = config.get("logdir", None)
        face_reward_weight = config.get("face_reward_weight", 0.5)
        vertex_reward_weight = config.get("vertex_reward_weight", 0.5)
        incremental_reward = config.get("incremental_reward", False)
        scale_reward = config["scale_reward"]
        use_boundary = config.get("use_boundary", False)
        eval_mode = config.get("eval_mode", False)
        fixed_reset = config.get("fixed_reset", False)

        return cls(
            face_desired_degree,
            graph_initializer,
            template_size=template_size,
            max_steps_factor=max_steps_factor,
            logdir=logdir,
            max_edge_addition_steps=max_edge_addition_steps,
            face_reward_weight=face_reward_weight,
            vertex_reward_weight=vertex_reward_weight,
            incremental_reward=incremental_reward,
            scale_reward=scale_reward,
            use_boundary=use_boundary,
            eval_mode=eval_mode,
            fixed_reset=fixed_reset
        )

    def _face_score(self, fidx):
        face_degree = self.graph.face_degree(fidx)
        return abs(face_degree - self.face_desired_degree)

    def global_face_score(self):
        faces = self.graph.face_list(tag=False)
        face_score = sum(self._face_score(fidx) for fidx in faces)
        return face_score

    def _vertex_score(self, vidx):
        vertex_degree = self.graph.vertex_degree(vidx)
        return abs(vertex_degree - self.vertex_desired_degree[vidx])

    def global_vertex_score(self):
        vertices = self.graph.vertex_list(tag=False)
        vertex_score = sum(self._vertex_score(vidx) for vidx in vertices)
        return vertex_score

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
        return 5

    def _half_edge_score(self, hidx):
        source_vertex = self.graph.source_vertex(hidx, tag=False)
        vertex_score = self._vertex_score(source_vertex)
        face = self.graph.face(hidx)
        face_score = self._face_score(face)
        return vertex_score + face_score

    def _select_half_edge_template_center(self, half_edges):
        half_edge_score = np.array([self._half_edge_score(hidx) for hidx in half_edges])
        max_indices = np.nonzero(half_edge_score == half_edge_score.max())[0]
        template_center_idx = np.random.choice(max_indices)
        return half_edges[template_center_idx]

    def _build_template(self):
        if self.use_boundary:
            self.index_to_half_edge = self.graph.knn_half_edges_with_boundary(self.template_center, self.template_size)
        else:
            self.index_to_half_edge = self.graph.knn_half_edges(self.template_center, self.template_size)

        self.half_edge_to_index = {half_edge: idx for idx, half_edge in enumerate(self.index_to_half_edge)}

    def _get_feature_matrix_archive(self):
        matrix = np.zeros((self.template_size, self.num_features), dtype=np.float32)
        for order, hidx in enumerate(self.index_to_half_edge):
            vidx = self.graph.source_vertex(hidx, tag=False)
            vertex_degree = self.graph.vertex_degree(vidx)
            vertex_desired_degree = self.vertex_desired_degree[vidx]
            is_user_defined_vertex = 1.0 if self.graph.is_user_defined_vertex(vidx) else 0.0

            fidx = self.graph.face(hidx)
            face_degree = self.graph.face_degree(fidx)

            matrix[order, :] = [
                min(vertex_degree / vertex_desired_degree, self.clamp_features_max),
                vertex_desired_degree,
                min(face_degree / self.face_desired_degree, self.clamp_features_max),
                self.face_desired_degree,
                is_user_defined_vertex
            ]

        return matrix

    def _get_feature_matrix(self):
        matrix = np.zeros((self.template_size, self.num_features), dtype=np.float32)
        for order, hidx in enumerate(self.index_to_half_edge):
            is_half_edge = self.graph.is_half_edge(hidx)

            vidx = self.graph.source_vertex(hidx, tag=False)
            vertex_degree = self.graph.vertex_degree(vidx)
            vertex_desired_degree = self.vertex_desired_degree[vidx]
            is_user_defined_vertex = 1.0 if self.graph.is_user_defined_vertex(vidx) else 0.0

            fidx = self.graph.face(hidx)
            face_degree = self.graph.face_degree(fidx) if is_half_edge else 0
            face_desired_degree = self.face_desired_degree if is_half_edge else 0

            matrix[order, :] = [
                min(vertex_degree / vertex_desired_degree, self.clamp_features_max),
                vertex_desired_degree,
                min(face_degree / self.face_desired_degree, self.clamp_features_max),
                face_desired_degree,
                is_user_defined_vertex
            ]

        return matrix

    def _get_user_defined_vertex_flag(self, vidx):
        if self.graph.is_user_defined_vertex(vidx):
            return 1.0
        else:
            return 0.0

    def _get_next_edges_archive(self):
        next_edges = np.arange(self.template_size)
        for (idx, hidx) in enumerate(self.index_to_half_edge):
            next_hidx = self.graph.next_half_edge(hidx)
            next_idx = self.half_edge_to_index.get(next_hidx, self.template_boundary_index)
            next_edges[idx] = next_idx
        return next_edges

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

    def _get_previous_edges_archive(self):
        prev_edges = np.arange(self.template_size)
        for (idx, hidx) in enumerate(self.index_to_half_edge):
            prev_hidx = self.graph.previous_half_edge(hidx)
            prev_idx = self.half_edge_to_index.get(prev_hidx, self.template_boundary_index)
            prev_edges[idx] = prev_idx
        return prev_edges

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

    def _get_twin_edges_archive(self):
        twin_edges = np.arange(self.template_size)
        for idx, hidx in enumerate(self.index_to_half_edge):
            if self.graph.half_edge_on_boundary(hidx):
                twin_edges[idx] = self.geometric_boundary_index
            else:
                twin_hidx = self.graph.twin_half_edge(hidx)
                twin_idx = self.half_edge_to_index.get(twin_hidx, self.template_boundary_index)
                twin_edges[idx] = twin_idx
        return twin_edges

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
        if self.use_progress:
            return np.array([self.num_steps / self.max_steps], dtype=np.float32)
        else:
            return np.array([0.0], dtype=np.float32)

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

    def _update_scores_and_reward(self, face_reward, vertex_reward):
        self.face_score -= face_reward
        self.vertex_score -= vertex_reward
        self.score = self.face_reward_weight * self.face_score + self.vertex_reward_weight * self.vertex_score
        self.min_score = min(self.min_score, self.score)

        if self.scale_reward:
            self.reward = self.face_reward_weight * face_reward / self.initial_face_score + \
                          self.vertex_reward_weight * vertex_reward / self.initial_vertex_score
        else:
            self.reward = self.face_reward_weight * face_reward + self.vertex_reward_weight * vertex_reward

    def _step_insert_edge(self, hidx, num_steps):
        if self.graph.is_valid_edge_insert(hidx, num_steps):
            original_face_idx = self.graph.face(hidx)
            original_face_degree = self.graph.face_degree(original_face_idx)

            self.graph.insert_half_edge(hidx, num_steps)
            new_face_idx = self.graph.face(hidx)

            original_face_new_degree = self.graph.face_degree(original_face_idx)
            new_face_degree = self.graph.face_degree(new_face_idx)
            face_reward = abs(original_face_degree - self.face_desired_degree) - \
                          (abs(original_face_new_degree - self.face_desired_degree) +
                           abs(new_face_degree - self.face_desired_degree))

            source_vertex = self.graph.source_vertex(hidx, tag=False)
            prev_half_edge = self.graph.previous_half_edge(hidx)
            prev_source_vertex = self.graph.source_vertex(prev_half_edge, tag=False)

            source_vertex_degree = self.graph.vertex_degree(source_vertex)
            prev_source_vertex_degree = self.graph.vertex_degree(prev_source_vertex)

            source_vertex_desired_degree = self.vertex_desired_degree[source_vertex]
            prev_source_vertex_desired_degree = self.vertex_desired_degree[prev_source_vertex]
            vertex_reward = (abs(source_vertex_degree - 1 - source_vertex_desired_degree) +
                             abs(prev_source_vertex_degree - 1 - prev_source_vertex_desired_degree)) - \
                            (abs(source_vertex_degree - source_vertex_desired_degree) +
                             abs(prev_source_vertex_degree - prev_source_vertex_desired_degree))

            self._update_scores_and_reward(face_reward, vertex_reward)

        return

    def _step_insert_vertex(self, hidx):
        if self.graph.is_half_edge(hidx):
            if self.graph.half_edge_on_boundary(hidx):
                face_idx = self.graph.face(hidx)
                face_degree = self.graph.face_degree(face_idx)
                self.graph.insert_vertex(hidx)

                new_vertex_idx = self.graph.target_vertex(hidx, tag=False)
                self.vertex_desired_degree[new_vertex_idx] = self.boundary_vertex_desired_degree

                face_reward = abs(face_degree - self.face_desired_degree) - abs(
                    face_degree + 1 - self.face_desired_degree)
                vertex_reward = 0 - abs(2 - self.boundary_vertex_desired_degree)
            else:
                face_idx = self.graph.face(hidx)
                face_degree = self.graph.face_degree(face_idx)

                twin_hidx = self.graph.twin_half_edge(hidx)
                twin_face = self.graph.face(twin_hidx)
                twin_face_degree = self.graph.face_degree(twin_face)

                self.graph.insert_vertex(hidx)

                new_vertex_idx = self.graph.target_vertex(hidx, tag=False)
                self.vertex_desired_degree[new_vertex_idx] = self.interior_vertex_desired_degree

                face_reward = (abs(face_degree - self.face_desired_degree) +
                               abs(twin_face_degree - self.face_desired_degree)) - \
                              (abs(face_degree + 1 - self.face_desired_degree) +
                               abs(twin_face_degree + 1 - self.face_desired_degree))
                vertex_reward = 0 - abs(2 - self.interior_vertex_desired_degree)

            self._update_scores_and_reward(face_reward, vertex_reward)

        return

    def _step_delete_edge(self, hidx):
        if self.graph.is_valid_delete_half_edge(hidx):
            face = self.graph.face(hidx)
            twin_hidx = self.graph.twin_half_edge(hidx)
            twin_face = self.graph.face(twin_hidx)

            source_vertex = self.graph.source_vertex(hidx, tag=False)
            target_vertex = self.graph.target_vertex(hidx, tag=False)

            current_face_degree = self.graph.face_degree(face)
            twin_face_degree = self.graph.face_degree(twin_face)

            source_vertex_degree = self.graph.vertex_degree(source_vertex)
            target_vertex_degree = self.graph.vertex_degree(target_vertex)

            source_desired_degree = self.vertex_desired_degree[source_vertex]
            target_desired_degree = self.vertex_desired_degree[target_vertex]

            self.graph.delete_half_edge(hidx)

            face_reward = (abs(current_face_degree - self.face_desired_degree) +
                           abs(twin_face_degree - self.face_desired_degree)) - \
                          (abs(current_face_degree + twin_face_degree - 2 - self.face_desired_degree))

            vertex_reward = (abs(source_vertex_degree - source_desired_degree) +
                             abs(target_vertex_degree - target_desired_degree)) - \
                            (abs(source_vertex_degree - 1 - source_desired_degree) +
                             abs(target_vertex_degree - 1 - target_desired_degree))
            self._update_scores_and_reward(face_reward, vertex_reward)

        return

    def _step_delete_source_vertex(self, hidx):
        if self.graph.is_valid_delete_source_vertex(hidx):
            if self.graph.half_edge_on_boundary(hidx):
                face_idx = self.graph.face(hidx)
                source_vertex = self.graph.source_vertex(hidx, tag=False)

                face_degree = self.graph.face_degree(face_idx)
                source_vertex_degree = self.graph.vertex_degree(source_vertex)
                source_vertex_desired_degree = self.vertex_desired_degree[source_vertex]

                self.graph.delete_source_vertex(hidx)

                face_reward = abs(face_degree - self.face_desired_degree) - (
                    abs(face_degree - 1 - self.face_desired_degree))
                vertex_reward = abs(source_vertex_degree - source_vertex_desired_degree)

            else:
                face_idx = self.graph.face(hidx)
                twin_half_edge = self.graph.twin_half_edge(hidx)
                source_vertex = self.graph.source_vertex(hidx, tag=False)
                twin_face = self.graph.face(twin_half_edge)

                face_degree = self.graph.face_degree(face_idx)
                twin_face_degree = self.graph.face_degree(twin_face)
                source_vertex_degree = self.graph.vertex_degree(source_vertex)
                source_vertex_desired_degree = self.vertex_desired_degree[source_vertex]

                self.graph.delete_source_vertex(hidx)

                face_reward = (abs(face_degree - self.face_desired_degree) +
                               abs(twin_face_degree - self.face_desired_degree)) - \
                              (abs(face_degree - 1 - self.face_desired_degree) +
                               abs(twin_face_degree - 1 - self.face_desired_degree))
                vertex_reward = abs(source_vertex_degree - source_vertex_desired_degree)

            self._update_scores_and_reward(face_reward, vertex_reward)
            self.vertex_desired_degree.pop(source_vertex)

    def _update_scores_on_reset(self):
        self.face_score = self.global_face_score()
        self.vertex_score = self.global_vertex_score()
        self.score = self.face_reward_weight * self.face_score + self.vertex_reward_weight * self.vertex_score
        self.min_score = self.score
        self.initial_score = self.score
        self.initial_face_score = self.face_score
        self.initial_vertex_score = self.vertex_score

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

    def _update_half_edge_template_center(self):
        self.template_center = self._select_half_edge_template_center(self.graph.half_edge_list())

    def is_terminated(self):
        if self.num_steps >= self.max_steps:
            return True

        if self.score == 0:
            return True

        if self.exception_occurred:
            return True

        return False

    def _get_terminal_reward(self):
        if self.terminated:
            face_reward = (self.initial_face_score - self.face_score) / self.initial_face_score
            vertex_reward = (self.initial_vertex_score - self.vertex_score) / self.initial_vertex_score
            reward = self.face_reward_weight * face_reward + self.vertex_reward_weight * vertex_reward
            return reward
        else:
            return 0

    def _get_incremental_reward(self):
        return self.reward

    def _get_eval_mode_reward(self):
        if self.terminated:
            reward = (self.initial_score - self.min_score) / self.initial_score
            return reward
        else:
            return 0

    def _get_reward(self):
        # if environment is in eval_mode, return best normalized reward
        if self.eval_mode:
            return self._get_eval_mode_reward()

        if self.incremental_reward:
            return self._get_incremental_reward()
        else:
            return self._get_terminal_reward()

    def step(self, linear_action_index):

        if self.num_steps >= self.max_steps:
            print("WARNING : NUM STEPS > MAX STEPS!!")  # this should not happen

        half_edge, local_action = self._linear_action_index_to_half_edge_and_action(linear_action_index)
        self.action_sequence.append((half_edge, local_action))

        try:
            self._step_half_edge_action(half_edge, local_action)
            self.num_steps += 1
            # update the template center after step
            self._update_half_edge_template_center()
            self._build_template()
            self.terminated = self.is_terminated()
            observation = self._get_obs()
        except Exception as e:
            self.exception_occurred = True
            print("\n\n\tENCOUNTERED ENVIRONMENT EXCPETION\n\n")
            self._log_exception()
            self.terminated = True
            observation = self._get_obs()

        reward = self._get_reward()

        return observation, reward, self.terminated, False, {"score": self.score}

    def _log_exception(self):
        exception_filename = str(uuid.uuid4()) + ".pkl"
        self.exception_count += 1
        exception_filepath = os.path.join(self.logdir, exception_filename)
        output_data = {
            "graph": self.initial_graph,
            "actions": self.action_sequence,
            "vertex_desired_degree": self.initial_vertex_desired_degree
        }
        with open(exception_filepath, "wb") as output_file:
            pickle.dump(output_data, output_file)

        print("\n\n\tLOGGED EXCEPTED ENV TO : ", exception_filepath, "\n\n\t")

    def _hard_reset(self):
        graph, desired_degree = self.graph_initializer()
        self.graph = graph
        self.vertex_desired_degree = desired_degree

        self.initial_graph = deepcopy(self.graph)
        self.initial_vertex_desired_degree = self.vertex_desired_degree.copy()
        self.action_sequence = []
        self.exception_occurred = False

        self.template_center = self._select_half_edge_template_center(self.graph.half_edge_list())
        self._build_template()

        num_halfedges = self.graph.number_of_half_edges()
        self.max_steps = int(self.max_steps_factor * num_halfedges)
        self.num_steps = 0

        self._update_scores_on_reset()

        self.terminated = self.is_terminated()

        obs = self._get_obs()
        return obs, {"score": self.score}

    def _reset_to_initial_state(self):
        self.graph = deepcopy(self.initial_graph)
        self.vertex_desired_degree = self.initial_vertex_desired_degree.copy()

        self.action_sequence = []
        self.exception_occurred = False

        self.template_center = self._select_half_edge_template_center(self.graph.half_edge_list())
        self._build_template()

        self.num_steps = 0

        self._update_scores_on_reset()

        self.terminated = self.is_terminated()

        obs = self._get_obs()
        return obs, {"score": self.score}

    def reset(self, seed=None, options=None):
        if self.fixed_reset:
            return self._reset_to_initial_state()
        else:
            return self._hard_reset()

#######################################################################################################################
#######################################################################################################################
