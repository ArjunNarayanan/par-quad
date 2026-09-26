import numpy as np
from envs.global_angle_env import AngleEnv


class AngleEnvWithLength(AngleEnv):
    """`AngleEnv` with edge lengths in place of vertex coordinates.

    Lengths are invariant under the rigid motions that the coordinate features
    only approximately normalise away, so the same local configuration looks
    the same wherever it sits in the mesh.
    """

    def __init__(self, *args, normalize_lengths=None, **kwargs):
        # None keeps the raw absolute length, which is what every checkpoint to
        # date was trained on. "median" divides by the template's own median,
        # which is what makes the feature scale-free.
        self.normalize_lengths = normalize_lengths
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        """The base builds every kwarg explicitly, so the extra one is added here."""
        env = super().from_config(config)
        env.normalize_lengths = config.get("normalize_lengths", None)
        return env

    @staticmethod
    def get_feature_size():
        return 10

    def _get_length_features(self):
        num_halfedges = len(self.index_to_half_edge)
        features = np.zeros(num_halfedges, dtype=np.float32)
        for idx, hidx in enumerate(self.index_to_half_edge):
            next_hidx = self.graph.next_half_edge(hidx)
            src = self.graph.source_vertex(hidx)
            next_src = self.graph.source_vertex(next_hidx)

            src_coords = self.graph.vertex_coordinate(src)
            next_coords = self.graph.vertex_coordinate(next_src)

            features[idx] = np.linalg.norm(next_coords - src_coords)
        if self.normalize_lengths == "median":
            # `normalize_points` scales every domain to the same box, so a
            # domain with more corners has proportionally shorter edges and the
            # RAW feature drifts with size: median 0.378 at 8-24 corners, 0.155
            # at 25-50, 0.120 at 60-100 -- a 3.2x shift into a range the
            # network never trained on. Dividing by the template's own median
            # removes it exactly (measured 1.00x across all three tiers) and
            # keeps the feature local: it says "how long is this edge compared
            # with its neighbours", which is what the decision actually needs.
            #
            # The median, not the centre half-edge. The centre re-selects every
            # step (it tracks the worst half-edge), so normalising by it makes
            # the denominator non-stationary WITHIN an episode, and a
            # near-zero centre edge would blow up the whole vector. The median
            # is stable against both and still removes the drift.
            positive = features[features > 0]
            if positive.size:
                features = features / max(float(np.median(positive)), 1e-9)
        return features

    def _get_feature_matrix(self):
        source_vertices = self.template_vertices
        faces = self.template_faces

        length_features = self._get_length_features()
        vertex_desired_degree = [self.vertex_desired_degree[vidx] for vidx in source_vertices]
        vertex_irregularities = [
            min(self.graph.vertex_degree(vidx), self.vertex_degree_threshold) for vidx in source_vertices
        ]

        face_irregularities = [min(self.graph.face_degree(fidx), self.face_degree_threshold) for fidx in faces]

        angle_irregularities = [self.half_edge_angles[hidx] / self.desired_angle for hidx in self.index_to_half_edge]

        num_half_edges = len(self.index_to_half_edge)

        matrix = np.zeros((self.template_size, self.num_features), dtype=np.float32)
        matrix[:num_half_edges, 0] = length_features
        matrix[:num_half_edges, 1] = vertex_desired_degree
        matrix[:num_half_edges, 2] = vertex_irregularities
        matrix[:num_half_edges, 3] = face_irregularities
        matrix[:num_half_edges, 4] = angle_irregularities
        matrix[:num_half_edges, 5:10] = self._get_context_features()

        return matrix
