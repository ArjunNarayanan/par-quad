from src.polygraph import PolyGraph
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np


def initialize_graph():
    node_ids = [0, 1, 2, 3, 4, 5]
    assert len(node_ids) == 6
    face_loops = [node_ids]
    graph = PolyGraph.from_face_loops(face_loops)

    return graph


def env_score(graph: PolyGraph, face_desired_degree: int):
    score = 0
    for face in range(graph.number_of_faces()):
        score += abs(graph.face_degree(face) - face_desired_degree)
    return score


class HexEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.max_num_halfedges = 12
        self.num_actions_per_halfedge = 3

        self.graph = initialize_graph()
        self.index_to_halfedge = self._get_index_to_halfedge()
        self.halfedge_to_index = self._get_halfedge_to_index(self.index_to_halfedge)

        # self.action_space = Discrete(self.max_num_halfedges * self.num_actions_per_halfedge)
        self.action_space = Discrete(self.num_actions_per_halfedge)

        self.observation_space = Dict(
            {
                "features": Box(low=0, high=2, shape=(self.max_num_halfedges, 4)),
                "next": Box(low=0, high=35, shape=(self.max_num_halfedges,), dtype=np.int64),
                "previous": Box(low=0, high=35, shape=(self.max_num_halfedges,), dtype=np.int64),
                "twin": Box(low=-1, high=35, shape=(self.max_num_halfedges,), dtype=np.int64)
            }
        )

        self.max_actions = 5
        self.face_desired_degree = 3
        self.vertex_desired_degree = 3

        self.num_actions = 0
        self.score = env_score(self.graph, self.face_desired_degree)
        self.reward = 0
        self.no_action_reward = -1

    def _get_index_to_halfedge(self):
        hidx = [hidx for hidx, data in self.graph.nodes(data=True) if data.get("type") == "halfedge"]
        return hidx

    @staticmethod
    def _get_halfedge_to_index(index_to_halfedge):
        return {halfedge: idx for idx, halfedge in enumerate(index_to_halfedge)}

    def _get_feature_matrix(self):
        matrix = np.zeros((self.max_num_halfedges, 4))
        for order, hidx in enumerate(self.index_to_halfedge):
            vidx = self.graph.source_vertex(hidx)
            vertex_degree = self.graph.vertex_degree(vidx)
            fidx = self.graph.face(hidx)
            face_degree = self.graph.face_degree(fidx)

            matrix[order, :] = [
                vertex_degree / self.vertex_desired_degree,
                self.vertex_desired_degree,
                face_degree / self.face_desired_degree,
                self.face_desired_degree
            ]
        return matrix

    def _get_next_edges(self):
        next_edges = np.arange(self.max_num_halfedges)
        for idx, hidx in enumerate(self.index_to_halfedge):
            next_hidx = self.graph.next_halfedge(hidx)
            next_idx = self.halfedge_to_index[next_hidx]
            next_edges[idx] = next_idx
        return next_edges

    def _get_previous_edges(self):
        previous_edges = np.arange(self.max_num_halfedges)
        for idx, hidx in enumerate(self.index_to_halfedge):
            prev_hidx = self.graph.previous_halfedge(hidx)
            prev_idx = self.halfedge_to_index[prev_hidx]
            previous_edges[idx] = prev_idx
        return previous_edges

    def _get_twin_edges(self):
        twin_edges = np.repeat([-1], self.max_num_halfedges)
        for idx, hidx in enumerate(self.index_to_halfedge):
            if not self.graph.halfedge_on_boundary(hidx):
                twin_hidx = self.graph.twin_halfedge(hidx)
                twin_idx = self.halfedge_to_index[twin_hidx]
                twin_edges[idx] = twin_idx
        return twin_edges

    def _get_obs(self):
        obs = {
            "features": self._get_feature_matrix(),
            "next": self._get_next_edges(),
            "previous": self._get_previous_edges(),
            "twin": self._get_twin_edges()
        }
        return obs

    def env_score(self):
        score = 0
        for face in range(self.graph.number_of_faces()):
            score += abs(self.graph.face_degree(face) - self.face_desired_degree)
        return score

    def reset(self, seed=None, options=None):
        self.graph = initialize_graph()
        self.index_to_halfedge = self._get_index_to_halfedge()
        self.halfedge_to_index = self._get_halfedge_to_index(self.index_to_halfedge)

        self.num_actions = 0
        self.score = env_score(self.graph, self.face_desired_degree)
        self.reward = 0

        observation = self._get_obs()
        return observation, {"score": self.score}

    def is_valid_action(self, halfedge_idx, num_edge_insert_steps):
        if halfedge_idx >= len(self.index_to_halfedge):
            return False
        hidx = self.index_to_halfedge[halfedge_idx]
        if not self.graph.is_valid_edge_insert(hidx, num_edge_insert_steps):
            return False
        return True

    def is_terminated(self):
        if self.num_actions >= self.max_actions or self.score == 0:
            return True
        else:
            return False

    def step(self, linear_action_index):
        if self.num_actions >= self.max_actions:
            print("WARNING : NUM ACTIONS > MAX ACTIONS!!")

        halfedge_idx = linear_action_index // self.num_actions_per_halfedge
        num_edge_insert_steps = linear_action_index % self.num_actions_per_halfedge + 1

        # if halfedge is valid, perform action
        if self.is_valid_action(halfedge_idx, num_edge_insert_steps):
            prev_score = self.score
            hidx = self.index_to_halfedge[halfedge_idx]
            self.graph.insert_halfedge(hidx, num_edge_insert_steps)
            self.index_to_halfedge = self._get_index_to_halfedge()
            self.halfedge_to_index = self._get_halfedge_to_index(self.index_to_halfedge)

            self.score = env_score(self.graph, self.face_desired_degree)
            self.reward = prev_score - self.score
        # if the selected halfedge is not valid, just increment number of steps and provide a negative reward
        else:
            self.reward = self.no_action_reward

        self.num_actions += 1
        terminated = self.is_terminated()
        observation = self._get_obs()
        return observation, self.reward, terminated, False, {"score": self.score}
