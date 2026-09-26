import networkx as nx
import itertools
from collections import deque
import numpy as np
import scipy as sp
import warnings
from typing import Optional, Sequence

UNTAGGED_INDEX_TYPE = int
TAG_TYPE = str
TAGGED_INDEX_TYPE = tuple[UNTAGGED_INDEX_TYPE, TAG_TYPE]
UNION_INDEX_TYPE = TAGGED_INDEX_TYPE | UNTAGGED_INDEX_TYPE


def induced_angle(v1: np.ndarray, v2: np.ndarray):
    """
    Compute the induced angle(s) in degrees between two sets of 2D vectors.

    Parameters
    ----------
    v1 : np.ndarray
        An (N, 2) array of N 2D vectors.
    v2 : np.ndarray
        An (N, 2) array of N 2D vectors.

    Returns
    -------
    np.ndarray
        An array of angles in degrees, in the range [0, 360).
    """
    dotp = (v1 * v2).sum(axis=1)
    detp = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    angle = np.degrees(np.arctan2(detp, dotp))
    angle[angle < 0] += 360
    return angle


class HalfEdge:
    def __init__(
        self,
        id: TAGGED_INDEX_TYPE,
        face: TAGGED_INDEX_TYPE | None = None,
        next: TAGGED_INDEX_TYPE | None = None,
        previous: TAGGED_INDEX_TYPE | None = None,
        twin: TAGGED_INDEX_TYPE | None = None,
        source: TAGGED_INDEX_TYPE | None = None,
        target: TAGGED_INDEX_TYPE | None = None,
    ):
        self.id = id
        self.face = face
        self.next = next
        self.previous = previous
        self.twin = twin
        self.source = source
        self.target = target


class Tiler(nx.MultiGraph):
    def __init__(self):
        super().__init__()
        self.next_half_edge_index: int = 0
        self.next_vertex_index: int = 0
        self.next_face_index: int = 0
        self.next_boundary_index: int = 0

        self.half_edge_tag: str = "h"
        self.vertex_tag: str = "v"
        self.face_tag: str = "f"
        self.boundary_tag: str = "b"

        self.vertex_coordinates: dict[TAGGED_INDEX_TYPE, np.ndarray] = None

        # TODO: would be nice to have this in tagged form for consistency
        self.user_defined_vertices: set[UNTAGGED_INDEX_TYPE] = set()
        # Which boundary edges are arcs. Empty unless a curved geometry filled
        # it in, and every code path below is a no-op when it is empty.
        from src.boundary_arcs import BoundaryArcs
        self.boundary_arcs = BoundaryArcs()
        # TODO: would be nice to have this in tagged form for consistency
        self.boundary_vertices: set[UNTAGGED_INDEX_TYPE] = set()

        self.half_edges: dict[TAGGED_INDEX_TYPE, HalfEdge] = dict()
        self.vertex_degrees: dict[TAGGED_INDEX_TYPE, int] = dict()
        self.face_degrees: dict[TAGGED_INDEX_TYPE, int] = dict()

    @classmethod
    def from_face_loops(
        cls,
        face_loops: list[list[int]],
        vertex_coordinates: dict[int, Sequence[float]] = None,
        user_vertices: set[
            int
        ] = None,  # TODO: Change this to any Sequence and cast type appropriately
    ):
        graph = cls()

        num_faces = len(face_loops)
        num_half_edges = sum(len(l) for l in face_loops)
        vertex_ids = set(itertools.chain.from_iterable(face_loops))
        num_vertices = len(vertex_ids)

        if vertex_coordinates is not None:
            assert all(
                v in vertex_coordinates for v in vertex_ids
            ), "Some vertices were not found in vertex_coordinates"
            for k, v in vertex_coordinates.items():
                vertex_coordinates[k] = np.array(v, dtype=float)

        graph.vertex_coordinates = vertex_coordinates

        if not user_vertices:
            graph.user_defined_vertices = vertex_ids.copy()
        else:
            graph.user_defined_vertices = set(user_vertices)

        graph.next_half_edge_index = num_half_edges
        graph.next_vertex_index = num_vertices
        graph.next_face_index = num_faces
        graph.next_boundary_index = 0

        half_edge_ids = [
            (idx, graph.half_edge_tag) for idx in range(graph.next_half_edge_index)
        ]
        graph.add_nodes_from(half_edge_ids, type="half_edge")
        half_edges = [HalfEdge(hidx) for hidx in half_edge_ids]
        graph.half_edges = dict(zip(half_edge_ids, half_edges))

        vertex_ids = [(idx, graph.vertex_tag) for idx in vertex_ids]
        graph.add_nodes_from(vertex_ids, type="vertex")

        face_ids = [(idx, graph.face_tag) for idx in range(graph.next_face_index)]
        graph.add_nodes_from(face_ids, type="face")

        graph.initialize_half_edges_from_face_loops(face_loops)
        graph.initialize_half_edge_source_associations(face_loops)
        graph.initialize_half_edge_target_associations(face_loops)
        graph.initialize_half_edge_face_associations(face_loops)
        graph.initialize_twin_associations()
        graph.initialize_boundary_source_target_associations()
        graph.initialize_vertex_degrees()
        graph.initialize_face_degrees()
        graph.initialize_boundary_vertices()

        return graph

    def is_user_defined_vertex(self, vidx: UNION_INDEX_TYPE) -> bool:
        vidx = self._ensure_untagged_form(vidx)
        return vidx in self.user_defined_vertices

    def associate_previous_next_half_edge(
        self, previous_hidx: UNION_INDEX_TYPE, next_hidx: UNION_INDEX_TYPE
    ) -> None:
        previous_hidx = self._ensure_tag_form(previous_hidx, self.half_edge_tag)
        next_hidx = self._ensure_tag_form(next_hidx, self.half_edge_tag)

        self.half_edges[previous_hidx].next = next_hidx
        self.half_edges[next_hidx].previous = previous_hidx

    def associate_half_edge_source_vertex(
        self, hidx: UNION_INDEX_TYPE, vidx: UNION_INDEX_TYPE
    ) -> None:
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)

        self.half_edges[hidx].source = vidx
        self.add_edge(hidx, vidx, key="source")

    def associate_half_edge_target_vertex(
        self, hidx: UNION_INDEX_TYPE, vidx: UNION_INDEX_TYPE
    ) -> None:
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)

        self.half_edges[hidx].target = vidx
        self.add_edge(hidx, vidx, key="target")

    def associate_half_edge_face(
        self, hidx: UNION_INDEX_TYPE, fidx: UNION_INDEX_TYPE
    ) -> None:
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        fidx = self._ensure_tag_form(fidx, self.face_tag)

        self.half_edges[hidx].face = fidx
        self.add_edge(hidx, fidx, key="face")

    def associate_half_edge_twin(
        self, hidx: UNION_INDEX_TYPE, twin_hidx: UNION_INDEX_TYPE
    ) -> None:
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        twin_hidx = self._ensure_tag_form(twin_hidx, self.half_edge_tag)

        self.half_edges[hidx].twin = twin_hidx
        self.half_edges[twin_hidx].twin = hidx

    def _add_face_loop(
        self,
        source_half_edges: list[UNTAGGED_INDEX_TYPE],
        next_half_edges: list[UNTAGGED_INDEX_TYPE],
    ) -> None:
        assert len(source_half_edges) == len(next_half_edges)
        assert next_half_edges[-1] == source_half_edges[0]

        # TODO: we should probably check whether the inputs are in tagged or untagged form
        source_edges_with_tags = [
            (idx, self.half_edge_tag) for idx in source_half_edges
        ]
        next_edges_with_tags = [(idx, self.half_edge_tag) for idx in next_half_edges]

        for src_hidx, next_hidx in zip(source_edges_with_tags, next_edges_with_tags):
            self.associate_previous_next_half_edge(src_hidx, next_hidx)

    # TODO: make these private methods since it makes strong assumptions
    def add_sequential_face_loop(
        self, half_edge_start: UNTAGGED_INDEX_TYPE, half_edge_stop: UNTAGGED_INDEX_TYPE
    ) -> None:
        # !ASSUMES THAT HALF EDGES IN A FACE LOOP ARE INDEXED SEQUENTIALLY!
        assert half_edge_stop - half_edge_start + 1 > 1
        source_indices = list(range(half_edge_start, half_edge_stop + 1))
        next_indices = source_indices[1:] + [source_indices[0]]
        self._add_face_loop(source_indices, next_indices)

    # TODO: make these private methods since it makes strong assumptions
    def initialize_half_edges_from_face_loops(
        self, face_loops: list[list[UNTAGGED_INDEX_TYPE]]
    ) -> None:
        start = 0
        for loop in face_loops:
            stop = start + len(loop) - 1
            self.add_sequential_face_loop(start, stop)
            start = stop + 1

    # TODO: make these private methods since it makes strong assumptions
    def initialize_half_edge_source_associations(
        self, face_loops: list[list[UNTAGGED_INDEX_TYPE]]
    ) -> None:
        source_vertices = [
            (v, self.vertex_tag) for v in itertools.chain.from_iterable(face_loops)
        ]
        half_edge_ids = [(h, self.half_edge_tag) for h in range(len(source_vertices))]

        for hidx, vidx in zip(half_edge_ids, source_vertices):
            self.associate_half_edge_source_vertex(hidx, vidx)

    # TODO: make these private methods since it makes strong assumptions
    def initialize_half_edge_target_associations(
        self, face_loops: list[list[UNTAGGED_INDEX_TYPE]]
    ) -> None:
        target_vertices = []
        for loop in face_loops:
            rotated_loop = loop[1:] + [loop[0]]
            target_vertices += rotated_loop

        target_vertex_ids = [(v, self.vertex_tag) for v in target_vertices]
        half_edge_ids = [(h, self.half_edge_tag) for h in range(len(target_vertex_ids))]

        for hidx, vidx in zip(half_edge_ids, target_vertices):
            self.associate_half_edge_target_vertex(hidx, vidx)

    # TODO: make these private methods since it makes strong assumptions
    def initialize_half_edge_face_associations(
        self, face_loops: list[list[UNTAGGED_INDEX_TYPE]]
    ) -> None:
        face_ids = []
        for face_idx, loop in enumerate(face_loops):
            loop_face_ids = len(loop) * [face_idx]
            face_ids += loop_face_ids

        face_nodes = [(f, self.face_tag) for f in face_ids]
        half_edge_nodes = [(h, self.half_edge_tag) for h in range(len(face_nodes))]

        for hidx, fidx in zip(half_edge_nodes, face_nodes):
            self.associate_half_edge_face(hidx, fidx)

    @staticmethod
    def _ensure_tag_form(idx: UNION_INDEX_TYPE, tag: TAG_TYPE) -> TAGGED_INDEX_TYPE:
        if isinstance(idx, tuple):
            # assert len(idx) == 2
            return idx
        else:
            return idx, tag

    @staticmethod
    def _ensure_untagged_form(idx: UNION_INDEX_TYPE) -> UNTAGGED_INDEX_TYPE:
        if isinstance(idx, tuple):
            return idx[0]
        else:
            return idx

    def initialize_twin_associations(self) -> None:
        half_edge_nodes = [
            (h, self.half_edge_tag) for h in range(self.next_half_edge_index)
        ]
        src_target = [
            (self.source_vertex(h), self.target_vertex(h)) for h in half_edge_nodes
        ]

        # TODO: check that no duplicates
        src_target_to_half_edge = dict(zip(src_target, half_edge_nodes))
        boundary_nodes = []
        boundary_count = 0

        for src_target, hidx in src_target_to_half_edge.items():
            src, target = src_target
            if (target, src) in src_target_to_half_edge:
                twin_hidx = src_target_to_half_edge[(target, src)]
                self.associate_half_edge_twin(hidx, twin_hidx)
            else:
                boundary_hidx = (boundary_count, self.boundary_tag)
                boundary_nodes.append(boundary_hidx)
                boundary_half_edge = HalfEdge(boundary_hidx)
                self.half_edges[boundary_hidx] = boundary_half_edge
                self.associate_half_edge_twin(hidx, boundary_hidx)
                boundary_count += 1

        self.next_boundary_index = boundary_count
        self.add_nodes_from(boundary_nodes, type="boundary")

    def initialize_boundary_source_target_associations(self) -> None:
        boundary_nodes = [node for node in self.nodes() if node[1] == self.boundary_tag]
        boundary_source = []
        boundary_target = []
        for bnode in boundary_nodes:
            twin_edge = self.twin_half_edge(bnode)
            twin_source = self.source_vertex(twin_edge)
            twin_target = self.target_vertex(twin_edge)

            self.associate_half_edge_source_vertex(bnode, twin_target)
            self.associate_half_edge_target_vertex(bnode, twin_source)

            boundary_source.append(twin_target)
            boundary_target.append(twin_source)

    def initialize_vertex_degrees(self) -> None:
        vertices = self.vertex_list(tag=True)
        degrees = self.vertex_degree_of_list(vertices)
        self.vertex_degrees.update(zip(vertices, degrees))

    def initialize_boundary_vertices(self):
        for hidx in self.half_edge_list():
            if self.half_edge_on_boundary(hidx):
                src = self.source_vertex(hidx, tag=False)
                self.boundary_vertices.add(src)

    def initialize_face_degrees(self):
        faces = self.face_list(tag=True)
        degrees = self.face_degree_of_list(faces)
        self.face_degrees.update(zip(faces, degrees))

    def _number_of_nodes(self, type):
        return sum(
            1 for vert, data in self.nodes(data=True) if data.get("type") == type
        )

    def number_of_half_edges(self):
        return self._number_of_nodes("half_edge")

    def number_of_vertices(self):
        return self._number_of_nodes("vertex")

    def number_of_faces(self):
        return self._number_of_nodes("face")

    def number_of_edges_of_type(self, type=None):
        return sum(1 for src, dst, key in self.edges(keys=True) if key == type)

    def _node_list_by_type(self, node_type, tag):
        nodes = [
            node
            for node, data in self.nodes(data=True)
            if data.get("type") == node_type
        ]
        if not tag:
            nodes = [self._ensure_untagged_form(idx) for idx in nodes]
        return nodes

    def half_edge_list(self, tag=True):
        return self._node_list_by_type("half_edge", tag)

    def boundary_half_edge_list(self, tag=True):
        return self._node_list_by_type("boundary", tag)

    def halfedge_list(self, tag=True):
        return self.half_edge_list(tag)

    def face_list(self, tag=True):
        return self._node_list_by_type("face", tag)

    def vertex_list(self, tag=True):
        return self._node_list_by_type("vertex", tag)

    def source_vertex(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self.half_edges[hidx].source
        if tag:
            return vidx
        else:
            return self._ensure_untagged_form(vidx)

    def target_vertex(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self.half_edges[hidx].target
        if tag:
            return vidx
        else:
            return self._ensure_untagged_form(vidx)

    def twin_half_edge(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        twin_idx = self.half_edges[hidx].twin
        if tag:
            return twin_idx
        else:
            return self._ensure_untagged_form(twin_idx)

    def next_half_edge(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        next_idx = self.half_edges[hidx].next
        if tag:
            return next_idx
        else:
            return self._ensure_untagged_form(next_idx)

    def previous_half_edge(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        prev_idx = self.half_edges[hidx].previous
        if tag:
            return prev_idx
        else:
            return self._ensure_untagged_form(prev_idx)

    def face(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        fidx = self.half_edges[hidx].face
        if tag:
            return fidx
        else:
            return self._ensure_untagged_form(fidx)

    def vertex_degree(self, vidx):
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)

        # degree = self.degree(vidx) // 2
        degree = self.vertex_degrees[vidx]

        return degree

    def vertex_degree_of_list(self, vertex_list):
        vertex_list = [
            self._ensure_tag_form(vidx, self.vertex_tag) for vidx in vertex_list
        ]
        # every mesh edge incident on a vertex will correspond to two half-edges in the 
        # half-edge graph structure. So find the graph degree and divide by two
        degree_dict = self.degree(vertex_list)
        # TODO: check that the degrees are all even
        degrees = [degree_dict[vidx] // 2 for vidx in vertex_list]
        return degrees

    def face_degree(self, fidx):
        fidx = self._ensure_tag_form(fidx, self.face_tag)
        return self.face_degrees[fidx]

    def face_degree_of_list(self, face_list):
        face_list = [self._ensure_tag_form(fidx, self.face_tag) for fidx in face_list]
        degree_dict = self.degree(face_list)
        degrees = [degree_dict[fidx] for fidx in face_list]
        return degrees

    def half_edge_on_boundary(self, hidx):
        tidx = self.twin_half_edge(hidx)
        return self.half_edges[tidx].face is None

    def _vertex_to_halfedge(self, vidx):
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)
        return (hidx for vidx, hidx in self.edges(vidx) if self.is_half_edge(hidx))

    def is_boundary_vertex(self, vidx):
        vidx = self._ensure_untagged_form(vidx)
        return vidx in self.boundary_vertices

    def face_half_edges(self, face_idx, tag=True):
        """return all half_edges connected to a face"""
        face_idx = self._ensure_tag_form(face_idx, self.face_tag)
        half_edges = [h for f, h in self.edges(face_idx)]
        if tag:
            return half_edges
        else:
            half_edges = [self._ensure_untagged_form(h) for h in half_edges]
            return half_edges

    def first_face_halfedge(self, face_idx, tag=True):
        face_idx = self._ensure_tag_form(face_idx, self.face_tag)
        halfedge = next(
            h for f, h, key in self.edges(face_idx, keys=True) if key == "face"
        )
        if tag:
            return halfedge
        else:
            return halfedge[0]

    def generate_half_edge_face_loop(self, half_edge_idx):
        """generate list of half_edges in a face loop"""
        half_edge_idx = self._ensure_tag_form(half_edge_idx, self.half_edge_tag)
        loop = [half_edge_idx]
        next_half_edge = self.next_half_edge(half_edge_idx)
        while next_half_edge != half_edge_idx:
            loop.append(next_half_edge)
            next_half_edge = self.next_half_edge(next_half_edge)

        return loop

    def generate_halfedge_face_loop(self, halfedge_idx):
        return self.generate_half_edge_face_loop(halfedge_idx)

    def is_half_edge(self, hidx):
        if hidx is None:
            return False

        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        return self.has_node(hidx) and self.nodes[hidx].get("type") == "half_edge"

    def is_boundary_half_edge(self, hidx):
        if hidx is None:
            return False

        hidx = self._ensure_tag_form(hidx, self.boundary_tag)
        return self.has_node(hidx) and self.nodes[hidx].get("type") == "boundary"

    def create_face(self):
        new_face_idx = (self.next_face_index, self.face_tag)
        self.add_node(new_face_idx, type="face")
        self.next_face_index += 1
        return new_face_idx

    def set_face_degree(self, fidx, degree):
        fidx = self._ensure_tag_form(fidx, self.face_tag)
        self.face_degrees[fidx] = degree

    def create_half_edge(self, next_half_edge_idx, prev_half_edge_idx):
        next_half_edge_idx = self._ensure_tag_form(
            next_half_edge_idx, self.half_edge_tag
        )
        prev_half_edge_idx = self._ensure_tag_form(
            prev_half_edge_idx, self.half_edge_tag
        )

        next_half_edge = self.half_edges[next_half_edge_idx]
        prev_half_edge = self.half_edges[prev_half_edge_idx]
        assert (
            next_half_edge.face == prev_half_edge.face
        ), "next/previous edges must share face"

        source_vertex = prev_half_edge.target
        target_vertex = next_half_edge.source
        face_idx = next_half_edge.face

        new_half_edge_idx = (self.next_half_edge_index, self.half_edge_tag)
        self.next_half_edge_index += 1

        new_half_edge = HalfEdge(
            new_half_edge_idx,
            face=face_idx,
            next=next_half_edge_idx,
            previous=prev_half_edge_idx,
            source=source_vertex,
            target=target_vertex,
        )
        self.half_edges[new_half_edge_idx] = new_half_edge
        self.add_node(new_half_edge_idx, type="half_edge")

        next_half_edge.previous = new_half_edge_idx
        prev_half_edge.next = new_half_edge_idx

        self.add_edge(new_half_edge_idx, source_vertex, key="source")
        self.add_edge(new_half_edge_idx, target_vertex, key="target")
        self.add_edge(new_half_edge_idx, face_idx, key="face")

        return new_half_edge_idx

    def create_boundary_half_edge(self, target_vertex, source_vertex):
        target_vertex = self._ensure_tag_form(target_vertex, self.vertex_tag)
        source_vertex = self._ensure_tag_form(source_vertex, self.vertex_tag)

        boundary_edge_idx = (self.next_boundary_index, self.boundary_tag)
        self.next_boundary_index += 1

        self.add_node(boundary_edge_idx, type="boundary")
        boundary_edge = HalfEdge(
            boundary_edge_idx, source=source_vertex, target=target_vertex
        )
        self.half_edges[boundary_edge_idx] = boundary_edge

        self.add_edge(boundary_edge_idx, source_vertex, key="source")
        self.add_edge(boundary_edge_idx, target_vertex, key="target")

        return boundary_edge_idx

    def is_valid_edge_insert(self, hidx, k):
        if not self.is_half_edge(hidx):
            return False

        face_idx = self.face(hidx)
        if not (0 <= k < self.face_degree(face_idx) - 1):
            return False

        return True

    def is_valid_chord_insert(self, hidx, k, min_face_degree=3):
        """Chord insert that leaves both new faces at least this many sides.

        `is_valid_edge_insert` allows `k = face_degree - 2`, which splits off a
        2-gon on the far side. Quad meshing never wants one, so the environment
        asks for a floor of three.
        """
        if not self.is_valid_edge_insert(hidx, k):
            return False
        degree = self.face_degree(self.face(hidx))
        return (k + 2) >= min_face_degree and (degree - k) >= min_face_degree

    def insert_half_edge(self, hidx, k):
        """
        Insert edge from source of hidx to target of half edge that is
        k next steps ahead.
        """
        assert self.is_valid_edge_insert(hidx, k), "Invalid edge insert encountered"

        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        face_idx = self.face(hidx)
        face_degree = self.face_degree(face_idx)
        self.set_face_degree(face_idx, face_degree - k)

        new_face_idx = self.create_face()
        self.set_face_degree(new_face_idx, k + 2)

        # remove edges to current face and add edge to new face
        current_half_edge = hidx
        for count in range(k + 1):
            self.remove_edge(current_half_edge, face_idx, key="face")
            self.associate_half_edge_face(current_half_edge, new_face_idx)
            current_half_edge = self.next_half_edge(current_half_edge)

        new_face_next_idx = hidx
        new_face_prev_idx = self.previous_half_edge(current_half_edge)
        old_face_next_idx = current_half_edge
        old_face_prev_idx = self.previous_half_edge(hidx)

        new_face_source_vertex = self.target_vertex(new_face_prev_idx)
        new_face_target_vertex = self.source_vertex(hidx)
        self.vertex_degrees[new_face_source_vertex] += 1
        self.vertex_degrees[new_face_target_vertex] += 1

        new_face_new_idx = self.create_half_edge(new_face_next_idx, new_face_prev_idx)
        old_face_new_idx = self.create_half_edge(old_face_next_idx, old_face_prev_idx)

        self.associate_half_edge_twin(new_face_new_idx, old_face_new_idx)

    def vertex_coordinate(self, vertex_idx):
        vertex_idx = self._ensure_untagged_form(vertex_idx)
        return self.vertex_coordinates[vertex_idx]

    def get_new_vertex_coordinate(self, next_vertex, previous_vertex, on_arc=None):
        """Where a vertex inserted on this edge goes.

        The midpoint of the chord, unless the edge is a boundary ARC -- then the
        midpoint of the arc, so a curved boundary stays curved instead of being
        chipped into a polygon one insertion at a time.

        `on_arc` is passed by the BOUNDARY insertion path only. It cannot be
        looked up here from the vertex pair alone: `boundary_arcs` is keyed by
        pair, and a chord can join the same two vertices through the interior,
        which would put an interior vertex out on the boundary curve.
        """
        if self.vertex_coordinates is None:
            return None
        if on_arc is not None:
            return on_arc.point(0.5)
        return 0.5 * (self.vertex_coordinate(next_vertex)
                      + self.vertex_coordinate(previous_vertex))

    def snap_to_boundary_arcs(self):
        """Put every vertex that has left its arc back onto it.

        Smoothing can move a boundary vertex off its curve; this is the last
        line of defence for any path that does. It is a no-op on a straight
        domain, and a no-op on a vertex already sitting on one of its arcs.

        A vertex owns TWO arcs wherever two of them meet -- at a joint, and at
        every midpoint where a refinement split one. Projecting onto each in
        turn and keeping the last was a bug with a large blast radius: a vertex
        slid legitimately ALONG one arc is outside the span of its other one, so
        the second projection clamped it back to the shared endpoint and undid
        the slide. On a refined curved mesh that dragged the boundary back onto
        the folds the smoother had just opened -- min quality +0.440 discarded
        and -0.034 kept, which then failed the caller's own quality guard and
        reverted the whole sweep. The NEAREST projection is the right one: it is
        zero when the vertex is already on the boundary, whichever of its arcs
        it happens to be on.
        """
        if not self.boundary_arcs or self.vertex_coordinates is None:
            return
        owners = {}
        for (first, second), (arc, _) in self.boundary_arcs.items():
            for vertex in (first, second):
                if vertex in self.vertex_coordinates and self.is_boundary_vertex(vertex):
                    owners.setdefault(vertex, []).append(arc)
        for vertex, arcs in owners.items():
            point = np.asarray(self.vertex_coordinates[vertex], dtype=float)
            best, distance = None, None
            for arc in arcs:
                candidate = np.asarray(arc.project(point), dtype=float)
                gap = float(np.linalg.norm(candidate - point))
                if distance is None or gap < distance:
                    best, distance = candidate, gap
            self.vertex_coordinates[vertex] = best
        # A vertex that slid ALONG its curve is still on the boundary, but the
        # arcs either side of it now record a span that no longer ends where it
        # does -- and `edge_direction` reads the tangent off those ends, so both
        # the corner want and the corner quality would be computed at the wrong
        # point. Re-anchor after placing, never before.
        for (first, second), (_, _) in list(self.boundary_arcs.items()):
            for vertex in (first, second):
                if vertex in owners:
                    self.boundary_arcs.reanchor(
                        first, second, vertex, self.vertex_coordinates[vertex])

    def create_vertex(
        self, coord: Optional[np.ndarray] = None, on_boundary: bool = False
    ) -> tuple[int, str]:
        new_vertex_idx = (self.next_vertex_index, self.vertex_tag)
        self.add_node(new_vertex_idx, type="vertex")

        if coord is not None and self.vertex_coordinates is not None:
            # assert len(coord) == 2
            self.vertex_coordinates[self.next_vertex_index] = coord

        if on_boundary:
            self.boundary_vertices.add(self.next_vertex_index)

        self.next_vertex_index += 1
        return new_vertex_idx

    def set_vertex_degree(self, vidx, degree):
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)
        self.vertex_degrees[vidx] = degree

    def set_vertex_coordinate(self, vidx, coord):
        vidx = self._ensure_untagged_form(vidx)
        self.vertex_coordinates[vidx] = np.array(coord)

    def set_user_defined_vertex(self, vidx, flag):
        vidx = self._ensure_untagged_form(vidx)
        if flag:
            self.user_defined_vertices.add(vidx)
        else:
            self.user_defined_vertices.discard(vidx)

    def set_target_vertex(self, hidx, vidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)

        current_target = self.target_vertex(hidx)
        twin_edge = self.twin_half_edge(hidx)
        self.remove_edge(hidx, current_target, key="target")
        self.remove_edge(twin_edge, current_target, key="source")

        self.half_edges[hidx].target = vidx
        self.half_edges[twin_edge].source = vidx
        self.add_edge(hidx, vidx, key="target")
        self.add_edge(twin_edge, vidx, key="source")

    def _insert_boundary_vertex(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.is_half_edge(hidx)
        assert self.half_edge_on_boundary(hidx)

        current_target_vertex = self.target_vertex(hidx)
        current_source_vertex = self.source_vertex(hidx)

        source = self._ensure_untagged_form(current_source_vertex)
        target = self._ensure_untagged_form(current_target_vertex)
        arc = self.boundary_arcs.get(source, target) if self.boundary_arcs else None
        new_vertex_coord = self.get_new_vertex_coordinate(
            current_target_vertex, current_source_vertex, on_arc=arc
        )
        new_vertex_idx = self.create_vertex(new_vertex_coord, on_boundary=True)
        self.set_vertex_degree(new_vertex_idx, 2)
        if arc is not None:
            # the edge became two edges, so the arc becomes two arcs
            self.boundary_arcs.split(
                source, target, self._ensure_untagged_form(new_vertex_idx))

        next_half_edge = self.next_half_edge(hidx)
        self.set_target_vertex(hidx, new_vertex_idx)

        new_half_edge = self.create_half_edge(next_half_edge, hidx)
        new_boundary_edge = self.create_boundary_half_edge(
            new_vertex_idx, current_target_vertex
        )
        self.associate_half_edge_twin(new_half_edge, new_boundary_edge)

        face_idx = self.face(hidx)
        self.face_degrees[face_idx] += 1

        return new_vertex_idx

    def _insert_interior_vertex(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.is_half_edge(hidx)
        assert not self.half_edge_on_boundary(hidx)

        next_hidx = self.next_half_edge(hidx)
        twin_hidx = self.twin_half_edge(hidx)
        twin_prev_hidx = self.previous_half_edge(twin_hidx)

        current_target_vertex = self.target_vertex(hidx)
        current_source_vertex = self.source_vertex(hidx)
        new_vertex_coord = self.get_new_vertex_coordinate(
            current_target_vertex, current_source_vertex
        )
        new_vertex_idx = self.create_vertex(new_vertex_coord, on_boundary=False)
        self.set_vertex_degree(new_vertex_idx, 2)

        self.set_target_vertex(hidx, new_vertex_idx)

        new_next_hidx = self.create_half_edge(next_hidx, hidx)
        new_twin_prev_hidx = self.create_half_edge(twin_hidx, twin_prev_hidx)
        self.associate_half_edge_twin(new_next_hidx, new_twin_prev_hidx)

        face_idx = self.face(hidx)
        self.face_degrees[face_idx] += 1
        twin_face_idx = self.face(twin_hidx)
        self.face_degrees[twin_face_idx] += 1

        return new_vertex_idx

    def insert_vertex(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.is_half_edge(hidx)
        if self.half_edge_on_boundary(hidx):
            new_vertex = self._insert_boundary_vertex(hidx)
        else:
            new_vertex = self._insert_interior_vertex(hidx)

    def _delete_vertex_metadata(self, vidx):
        vidx = self._ensure_untagged_form(vidx)
        if self.vertex_coordinates is not None:
            coords = self.vertex_coordinates.pop(vidx, None)
            if coords is None:
                print("Warning: deleted vertex not found in vertex coordinates")
        self.boundary_vertices.discard(vidx)

    def _delete_half_edge(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        twin = self.twin_half_edge(hidx)

        self.remove_node(hidx)
        self.remove_node(twin)
        self.half_edges.pop(hidx)
        self.half_edges.pop(twin)

    def _delete_vertex(self, vidx):
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)
        self.remove_node(vidx)
        self.vertex_degrees.pop(vidx)
        self._delete_vertex_metadata(vidx)

    def _delete_boundary_vertex(self, hidx):
        """deletes vertex at source of hidx"""
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.half_edge_on_boundary(hidx)
        source_vertex = self.source_vertex(hidx)
        assert self.vertex_degree(source_vertex) == 2
        assert not self.is_user_defined_vertex(source_vertex)

        next_half_edge = self.next_half_edge(hidx)
        prev_half_edge = self.previous_half_edge(hidx)

        target_vertex = self.target_vertex(hidx)
        # This vertex may have been inserted ON a curve, splitting one arc in
        # two. Putting the halves back is the inverse of that split; skipping it
        # leaves them keyed on a vertex about to disappear and the edge that
        # reappears is silently straight.
        if self.boundary_arcs:
            before = self.source_vertex(prev_half_edge, tag=False)
            gone = self._ensure_tag_form(source_vertex, self.vertex_tag)[0]
            after = self.target_vertex(hidx, tag=False)
            self.boundary_arcs.merge(before, gone, after)

        self.set_target_vertex(prev_half_edge, target_vertex)
        self.associate_previous_next_half_edge(prev_half_edge, next_half_edge)

        face_idx = self.face(hidx)
        self.face_degrees[face_idx] -= 1

        self._delete_half_edge(hidx)
        self._delete_vertex(source_vertex)

    def _delete_interior_vertex(self, hidx):
        """deletes vertex at source of hidx"""
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert not self.half_edge_on_boundary(hidx)
        source_vertex = self.source_vertex(hidx)
        # assert self.vertex_degree(source_vertex) == 2

        next_half_edge = self.next_half_edge(hidx)
        prev_half_edge = self.previous_half_edge(hidx)
        twin_half_edge = self.twin_half_edge(hidx)

        next_twin_half_edge = self.next_half_edge(twin_half_edge)
        prev_twin_half_edge = self.previous_half_edge(twin_half_edge)

        target_vertex = self.target_vertex(hidx)
        self.set_target_vertex(prev_half_edge, target_vertex)
        self.associate_previous_next_half_edge(prev_half_edge, next_half_edge)
        self.associate_previous_next_half_edge(prev_twin_half_edge, next_twin_half_edge)

        face_idx = self.face(hidx)
        self.face_degrees[face_idx] -= 1
        twin_face_idx = self.face(twin_half_edge)
        self.face_degrees[twin_face_idx] -= 1

        self._delete_half_edge(hidx)
        self._delete_vertex(source_vertex)

    def is_valid_delete_source_vertex(self, hidx):
        """Check if vertex at source of hidx can be deleted"""
        if not self.is_half_edge(hidx):
            return False

        vidx = self.source_vertex(hidx)
        if self.is_user_defined_vertex(vidx):
            return False

        if self.vertex_degree(vidx) != 2:
            return False

        fidx = self.face(hidx)
        if self.face_degree(fidx) < 3:
            return False

        if not self.half_edge_on_boundary(hidx):
            twin_edge = self.twin_half_edge(hidx)
            twin_face = self.face(twin_edge)
            if self.face_degree(twin_face) < 3:
                return False

        return True

    def delete_source_vertex(self, hidx):
        assert self.is_valid_delete_source_vertex(
            hidx
        ), "cannot delete vertex at source of : " + str(hidx)
        if self.half_edge_on_boundary(hidx):
            self._delete_boundary_vertex(hidx)
        else:
            self._delete_interior_vertex(hidx)

    def is_valid_delete_half_edge(self, hidx):
        if not self.is_half_edge(hidx):
            return False

        if self.half_edge_on_boundary(hidx):
            return False

        source_vertex = self.source_vertex(hidx)
        target_vertex = self.target_vertex(hidx)
        if (
            self.vertex_degree(source_vertex) <= 2
            or self.vertex_degree(target_vertex) <= 2
        ):
            return False

        # Deleting a half-edge results in merging of faces on either side.
        # If these faces are already the same, this is an invalid delete
        twin_edge = self.twin_half_edge(hidx)
        current_face = self.face(hidx)
        twin_face = self.face(twin_edge)
        if current_face == twin_face:
            return False

        return True

    def _half_edges_of_face(self, fidx):
        fidx = self._ensure_tag_form(fidx, self.face_tag)
        edges = (
            target for src, target, key in self.edges(fidx, keys=True) if key == "face"
        )
        return edges

    def delete_half_edge(self, hidx):
        assert self.is_valid_delete_half_edge(hidx)

        next_edge = self.next_half_edge(hidx)
        prev_edge = self.previous_half_edge(hidx)
        twin_edge = self.twin_half_edge(hidx)
        prev_twin = self.previous_half_edge(twin_edge)
        next_twin = self.next_half_edge(twin_edge)
        source_vertex = self.source_vertex(hidx)
        target_vertex = self.target_vertex(hidx)

        self.vertex_degrees[source_vertex] -= 1
        self.vertex_degrees[target_vertex] -= 1

        current_face = self.face(hidx)
        twin_face = self.face(twin_edge)

        current_face_degree = self.face_degree(current_face)
        twin_face_degree = self.face_degree(twin_face)
        self.set_face_degree(current_face, current_face_degree + twin_face_degree - 2)

        # Associate all edges from neighboring face with current face and delete neighboring face
        twin_face_edges = self._half_edges_of_face(twin_face)
        for half_edge in twin_face_edges:
            self.associate_half_edge_face(half_edge, current_face)

        # delete the twin face
        self.remove_node(twin_face)
        self.face_degrees.pop(twin_face)

        self.associate_previous_next_half_edge(prev_edge, next_twin)
        self.associate_previous_next_half_edge(prev_twin, next_edge)
        self._delete_half_edge(hidx)

    def _finite_quad_global_split_to_boundary(self, hidx, max_steps):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        seen = set()
        seen.add(hidx)

        current_half_edge = hidx
        for step in range(max_steps):
            current_half_edge = self.next_half_edge(
                self.next_half_edge(current_half_edge)
            )
            if self.half_edge_on_boundary(current_half_edge):
                return True
            current_half_edge = self.twin_half_edge(current_half_edge)
            if current_half_edge in seen:
                return False
            seen.add(current_half_edge)

        return False

    def is_valid_quad_global_split_source(self, hidx, max_steps):
        if not self.is_half_edge(hidx):
            return False
        if not self.half_edge_on_boundary(hidx):
            return False

        return self._finite_quad_global_split_to_boundary(hidx, max_steps)

    def global_split_to_boundary(self, hidx, max_steps):
        assert self.is_valid_quad_global_split_source(hidx, max_steps)
        original_hidx = hidx

        self.insert_vertex(hidx)
        hidx = self.next_half_edge(hidx)

        num_steps = 0
        while num_steps < max_steps and not self.is_boundary_half_edge(hidx):
            next_half_edge = self.next_half_edge(self.next_half_edge(hidx))
            self.insert_vertex(next_half_edge)
            self.insert_half_edge(hidx, 2)
            hidx = self.twin_half_edge(next_half_edge)
            num_steps += 1

        if not self.is_boundary_half_edge(hidx):
            message = "Global split from " + str(
                original_hidx
            ), " terminated in interior at " + str(hidx)
            warnings.warn(message)

    def _get_global_line_half_edges(self, hidx, max_steps):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        half_edges = [hidx]

        def increment_half_edge(hidx):
            next_half_edge = self.twin_half_edge(
                self.next_half_edge(self.next_half_edge(hidx))
            )
            return next_half_edge

        num_steps = 0
        current_half_edge = increment_half_edge(hidx)

        while num_steps < max_steps and not self.is_boundary_half_edge(
            current_half_edge
        ):
            half_edges.append(current_half_edge)
            current_half_edge = increment_half_edge(current_half_edge)
            num_steps += 1

        if not self.is_boundary_half_edge(current_half_edge):
            warnings.warn("Global split failed to terminate from hidx : ", hidx)

        return half_edges

    def _template_seeds(self, hidx, seed_face):
        """Where the template's breadth-first walk starts.

        From a single half-edge the walk reaches `twin(hidx)` -- a half-edge of
        the NEIGHBOURING face -- at depth one, while the far side of hidx's own
        face waits until depth two. The window therefore leaks across hidx
        before it has closed the face it is supposed to be centred on, and which
        way it leaks depends on which half-edge of that face happened to be
        chosen as the centre.

        Seeding with the whole face loop puts the face in the window first and
        makes the expansion symmetric about it: on a 9x9 grid at k=16 the
        window's centroid moves from 0.319 off the centre face to exactly on it.

        It does not make the window independent of which half-edge was named.
        The seeds are the loop rotated to start at hidx, so a `k` that falls
        mid-frontier still cuts a start-dependent set -- on a 7x7 grid the four
        half-edges of a face give one window at k=4, 8 and 16 and four different
        ones at k=12, 20 and 24. Making it fully independent would need the
        frontier sorted by something other than arrival order, which is a larger
        change for a bias that vanishes once the window holds a few rings.
        """
        if not seed_face:
            return [hidx]
        try:
            loop = self.generate_half_edge_face_loop(hidx)
        except Exception:
            return [hidx]
        if hidx not in loop:
            return [hidx]
        # keep hidx first: callers index the template and the centre is index 0
        start = loop.index(hidx)
        return list(loop[start:]) + list(loop[:start])

    def knn_half_edges(self, hidx, k, seed_face=False):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.is_half_edge(hidx)
        assert k >= 1

        seen = set()
        queue = deque()

        for seed in self._template_seeds(hidx, seed_face):
            if seed not in seen:
                seen.add(seed)
                queue.append(seed)
        neighbors = []

        def append_if_not_seen(edge):
            if edge not in seen:
                seen.add(edge)
                queue.append(edge)

        while queue:
            edge = queue.popleft()
            neighbors.append(edge)
            if len(neighbors) >= k:
                return neighbors

            next_edge = self.next_half_edge(edge)
            append_if_not_seen(next_edge)

            prev_edge = self.previous_half_edge(edge)
            append_if_not_seen(prev_edge)

            if not self.half_edge_on_boundary(edge):
                twin_edge = self.twin_half_edge(edge)
                append_if_not_seen(twin_edge)

        return neighbors

    def knn_half_edges_with_boundary(self, hidx, k, seed_face=False):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.is_half_edge(hidx)
        assert k >= 1

        seen = set()
        queue = deque()

        for seed in self._template_seeds(hidx, seed_face):
            if seed not in seen:
                seen.add(seed)
                queue.append(seed)
        neighbors = []

        def append_if_not_seen(edge):
            if edge not in seen:
                seen.add(edge)
                queue.append(edge)

        while queue:
            edge = queue.popleft()
            neighbors.append(edge)
            if len(neighbors) >= k:
                return neighbors

            if self.is_half_edge(edge):
                next_edge = self.next_half_edge(edge)
                append_if_not_seen(next_edge)

                prev_edge = self.previous_half_edge(edge)
                append_if_not_seen(prev_edge)

                twin_edge = self.twin_half_edge(edge)
                append_if_not_seen(twin_edge)

        return neighbors

    def _construct_sparse_vertex_laplace_operator(self, vertex2index):
        num_verts = len(vertex2index)

        row_indices = []
        col_indices = []
        values = []

        def update_row_col_val(row, col, val):
            row_indices.append(vertex2index[row])
            col_indices.append(vertex2index[col])
            values.append(val)

        for hidx in self.half_edge_list():
            src = self.source_vertex(hidx, tag=False)
            dst = self.target_vertex(hidx, tag=False)

            if self.half_edge_on_boundary(hidx):
                if not self.is_user_defined_vertex(src):
                    update_row_col_val(src, dst, 1.0)
                if not self.is_user_defined_vertex(dst):
                    update_row_col_val(dst, src, 1.0)
            else:
                if (not self.is_boundary_vertex(src)) and (
                    not self.is_user_defined_vertex(src)
                ):
                    update_row_col_val(src, dst, 1.0)

        # Set all user defined vertices to have same coords
        for vidx, idx in vertex2index.items():
            if self.is_user_defined_vertex(vidx):
                update_row_col_val(idx, idx, 1.0)

        matrix = sp.sparse.csr_matrix(
            (values, (row_indices, col_indices)), shape=(num_verts, num_verts)
        )
        return matrix

    def _get_vertex_degrees_for_smoothing(self, index2vertex):
        num_verts = len(index2vertex)
        degrees = num_verts * [1]
        for idx, vidx in enumerate(index2vertex):
            if not self.is_user_defined_vertex(vidx):
                if self.is_boundary_vertex(vidx):
                    degrees[idx] = 2
                else:
                    degrees[idx] = self.vertex_degree(vidx)

        return degrees

    def smooth_vertices(self, num_iter=3):
        index2vertex = self.vertex_list(tag=False)
        num_verts = len(index2vertex)
        vertex2index = dict(zip(index2vertex, range(num_verts)))

        matrix = self._construct_sparse_vertex_laplace_operator(vertex2index)
        coordinates = np.array([self.vertex_coordinate(vidx) for vidx in index2vertex])
        degrees = np.array(
            self._get_vertex_degrees_for_smoothing(index2vertex)
        ).reshape(-1, 1)

        for step in range(num_iter):
            coordinates = matrix @ coordinates
            coordinates /= degrees

        for idx, vidx in enumerate(index2vertex):
            self.set_vertex_coordinate(vidx, coordinates[idx])

        # A boundary vertex on a curve is smoothed along the CHORD between its
        # neighbours, which walks it off the curve -- measured at 0.2 of a unit
        # radius after five sweeps. Snapping here rather than at the call sites
        # keeps the invariant with the object that owns it, and is a no-op on a
        # straight-edged domain.
        self.snap_to_boundary_arcs()

    def corner_shape_qualities(self):
        """Per half-edge: `2J / (|a|^2 + |b|^2)`, the ASPECT-AWARE corner quality.

        `half_edge_angles` sees only the angle, so a corner of two edges meeting
        squarely scores 1 whether they are the same length or one is a hundred
        times the other. That is the metric the gate and the reward have been
        using, and it is blind to exactly the failure it most needs to catch --
        a comb of paper-thin quads has perfect angles. Measured on one
        checkpoint's own meshes, the angle metric reads 0.707 on a mesh geo2d
        scores 0.056.

        This is geo2d's `corner_quality` before its ideal-angle division: the
        same J the angle uses, divided by the sum of the squared edge lengths
        rather than by their product, which is 1 when they are equal and falls
        away as they diverge. Identical to geo2d on an all-quad mesh.

        **The ANGLE comes from the TANGENT, the LENGTHS from the chord.** On a
        curved edge the element's side follows the arc, so the direction it
        leaves the corner is the arc's tangent; building the Jacobian from the
        chord measures an element nobody has. It is the same correction the
        corner WANTS already use -- `boundary_arcs.edge_direction` is the shared
        rule, and both now agree, which is the point: a smooth arc joint reads
        180 degrees, wants degree three, AND scores as the flat corner it is if
        one face owns both of its sides. The chord reading saw an angle there
        and called it fine. Aspect ratio stays on the straight-sided polygonal
        patch of the block, which is what `|a|` and `|b|` are.

        Measured on the released curved meshes, the chord reading overstates
        quality by about 3x and hides folded elements.
        Exactly unchanged on a domain with no arcs, since there the tangent IS
        the chord -- every straight-domain result and checkpoint is untouched.
        """
        coordinates = np.array([v for k, v in self.vertex_coordinates.items()])
        index = {k: i for i, (k, _) in enumerate(self.vertex_coordinates.items())}
        half_edges = self.half_edge_list()
        source = [index[self.source_vertex(h, tag=False)] for h in half_edges]
        target = [index[self.target_vertex(h, tag=False)] for h in half_edges]
        previous = [index[self.source_vertex(self.previous_half_edge(h), tag=False)]
                    for h in half_edges]
        centre = coordinates[source]
        a = coordinates[target] - centre
        b = coordinates[previous] - centre
        # Turn the direction of any side that is an ARC onto its tangent, while
        # keeping the chord LENGTH. Only arc-carrying edges are touched, so the
        # straight path stays the vectorised one it was.
        arcs = getattr(self, "boundary_arcs", None)
        if arcs and len(list(arcs.items())):
            from src.boundary_arcs import edge_direction
            corner_of = {h: i for i, h in enumerate(half_edges)}
            for half_edge in half_edges:
                row = corner_of[half_edge]
                here = self.source_vertex(half_edge, tag=False)
                ahead = self.target_vertex(half_edge, tag=False)
                behind = self.source_vertex(
                    self.previous_half_edge(half_edge), tag=False)
                for vector, there in ((a, ahead), (b, behind)):
                    if arcs.get(here, there) is None:
                        continue
                    direction = np.asarray(
                        edge_direction(self, here, there), dtype=float)
                    norm = float(np.linalg.norm(direction))
                    if norm < 1e-12:
                        continue
                    vector[row] = np.linalg.norm(vector[row]) * direction / norm
        jacobian = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        lengths = (a * a).sum(axis=1) + (b * b).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            quality = np.where(lengths > 0, 2.0 * jacobian / lengths, 0.0)
        return dict(zip(half_edges, quality))

    def half_edge_angles(self):
        coordinates = np.array([v for k, v in self.vertex_coordinates.items()])
        index2vertex = [k for k, _ in self.vertex_coordinates.items()]
        num_verts = len(index2vertex)
        vertex2index = dict(zip(index2vertex, range(num_verts)))

        half_edges = self.half_edge_list()
        num_half_edges = len(half_edges)

        half_edge_source = num_half_edges * [0]
        half_edge_next = num_half_edges * [0]
        half_edge_prev = num_half_edges * [0]

        for idx, hidx in enumerate(half_edges):
            half_edge_source[idx] = vertex2index[self.source_vertex(hidx, tag=False)]
            half_edge_next[idx] = vertex2index[self.target_vertex(hidx, tag=False)]
            half_edge_prev[idx] = vertex2index[
                self.source_vertex(self.previous_half_edge(hidx), tag=False)
            ]

        C = coordinates[half_edge_source]
        N = coordinates[half_edge_next]
        P = coordinates[half_edge_prev]

        v1 = N - C
        v2 = P - C
        angles = induced_angle(v1, v2)
        angles_dict = dict(zip(half_edges, angles))

        return angles_dict


def _get_vertex_array(tiler: Tiler):
    num_verts = tiler.number_of_vertices()
    coordinates = np.array([v for k, v in tiler.vertex_coordinates.items()])
    vertices = [k for k, v in tiler.vertex_coordinates.items()]
    vertex2index = dict(zip(vertices, range(num_verts)))
    return coordinates, vertex2index


def _get_edge_array(tiler: Tiler, vertex2index):
    edge_count = 0
    edge2index = {}
    edges = []

    # sorting is not necessary, but it makes it easier to check with test cases
    halfedges = sorted(tiler.half_edge_list())

    for hidx in halfedges:
        src = tiler.source_vertex(hidx, tag=False)
        dst = tiler.target_vertex(hidx, tag=False)
        if (src, dst) not in edge2index:
            edge2index[(src, dst)] = edge_count
            edge2index[(dst, src)] = edge_count
            edges.append([vertex2index[src], vertex2index[dst]])
            edge_count += 1

    return np.array(edges), edge2index


def _get_connectivity_representation(tiler: Tiler, face_degree):
    num_faces = tiler.number_of_faces()
    coordinates, vertex2index = _get_vertex_array(tiler)
    edges, edge2index = _get_edge_array(tiler, vertex2index)

    connectivity = np.zeros((num_faces, face_degree), dtype=int)
    face2edge = np.zeros((num_faces, face_degree), dtype=int)

    for idx, face_index in enumerate(tiler.face_list()):
        hidx = tiler.first_face_halfedge(face_index)
        for step in range(face_degree):
            src = tiler.source_vertex(hidx, tag=False)
            dst = tiler.target_vertex(hidx, tag=False)
            connectivity[idx, step] = vertex2index[src]
            face2edge[idx, step] = edge2index[(src, dst)]
            hidx = tiler.next_half_edge(hidx)

    user_vertices = [vertex2index[vidx] for vidx in tiler.user_defined_vertices]

    representation = {
        "vertex connectivity": connectivity,
        "coordinates": coordinates,
        "edges": edges,
        "edge connectivity": face2edge,
        "user vertices": user_vertices,
    }
    return representation


def triangle_connectivity_representation(tiler: Tiler):
    assert all(tiler.face_degree(fidx) == 3 for fidx in tiler.face_list())
    return _get_connectivity_representation(tiler, 3)


def quad_connectivity_representation(tiler: Tiler):
    assert all(tiler.face_degree(fidx) == 4 for fidx in tiler.face_list())
    return _get_connectivity_representation(tiler, 4)


def _refine_triangles(
    vertex_coordinates, edges, vertex_connectivity, edge_connectivity
):
    assert vertex_coordinates.shape[1] == 2
    assert edges.shape[1] == 2
    assert vertex_connectivity.shape[1] == 3
    assert edge_connectivity.shape[1] == 3

    # get mid-point of edges
    midpoint_coords = 0.5 * (
        vertex_coordinates[edges[:, 0]] + vertex_coordinates[edges[:, 1]]
    )
    vertex_offset = vertex_coordinates.shape[0]

    v0 = vertex_connectivity[:, 0]
    v1 = vertex_connectivity[:, 1]
    v2 = vertex_connectivity[:, 2]

    v01 = edge_connectivity[:, 0] + vertex_offset
    v12 = edge_connectivity[:, 1] + vertex_offset
    v20 = edge_connectivity[:, 2] + vertex_offset

    # construct new vertex connectivity
    block1 = np.column_stack([v0, v01, v20])
    block2 = np.column_stack([v01, v1, v12])
    block3 = np.column_stack([v01, v12, v20])
    block4 = np.column_stack([v20, v12, v2])
    new_vertex_connectivity = np.vstack([block1, block2, block3, block4])

    new_coordinates = np.vstack([vertex_coordinates, midpoint_coords])

    return new_coordinates, new_vertex_connectivity


def _refine_quads(vertex_coordinates, edges, vertex_connectivity, edge_connectivity):
    assert vertex_coordinates.shape[1] == 2
    assert edges.shape[1] == 2
    assert vertex_connectivity.shape[1] == 4
    assert edge_connectivity.shape[1] == 4

    # get mid-point of edges
    edge_midpoint_coords = 0.5 * (
        vertex_coordinates[edges[:, 0]] + vertex_coordinates[edges[:, 1]]
    )

    # get face centroids
    face_centroids = 0.25 * (
        vertex_coordinates[vertex_connectivity[:, 0]]
        + vertex_coordinates[vertex_connectivity[:, 1]]
        + vertex_coordinates[vertex_connectivity[:, 2]]
        + vertex_coordinates[vertex_connectivity[:, 3]]
    )

    # edge midpoint vertices and face centroid vertices' indices need to be offset
    # when we combine all the vertices together
    edge_vertices_offset = vertex_coordinates.shape[0]
    face_vertices_offset = vertex_coordinates.shape[0] + edges.shape[0]

    v0 = vertex_connectivity[:, 0]
    v1 = vertex_connectivity[:, 1]
    v2 = vertex_connectivity[:, 2]
    v3 = vertex_connectivity[:, 3]

    v01 = edge_connectivity[:, 0] + edge_vertices_offset
    v12 = edge_connectivity[:, 1] + edge_vertices_offset
    v23 = edge_connectivity[:, 2] + edge_vertices_offset
    v30 = edge_connectivity[:, 3] + edge_vertices_offset

    vf = np.arange(vertex_connectivity.shape[0]) + face_vertices_offset

    block1 = np.column_stack([v0, v01, vf, v30])
    block2 = np.column_stack([v01, v1, v12, vf])
    block3 = np.column_stack([vf, v12, v2, v23])
    block4 = np.column_stack([v30, vf, v23, v3])

    new_vertex_connectivity = np.vstack([block1, block2, block3, block4])
    new_coordinates = np.vstack(
        [vertex_coordinates, edge_midpoint_coords, face_centroids]
    )

    return new_coordinates, new_vertex_connectivity


def refine(tiler: Tiler, face_degree):
    if face_degree == 3:
        representation = triangle_connectivity_representation(tiler)
    elif face_degree == 4:
        representation = quad_connectivity_representation(tiler)
    else:
        raise ValueError(
            "Expected face degree in {3,4} got face degree = ", face_degree
        )

    coords = representation["coordinates"]
    vconn = representation["vertex connectivity"]
    edges = representation["edges"]
    econn = representation["edge connectivity"]

    if face_degree == 3:
        new_coords, new_conn = _refine_triangles(coords, edges, vconn, econn)
    elif face_degree == 4:
        new_coords, new_conn = _refine_quads(coords, edges, vconn, econn)
    else:
        raise ValueError(
            "Expected face degree in {3,4} got face degree = ", face_degree
        )

    num_coords = new_coords.shape[0]
    new_coords = dict(zip(range(num_coords), new_coords))
    new_conn = [c.tolist() for c in new_conn]

    user_vertices = representation["user vertices"]
    new_tiler = Tiler.from_face_loops(
        new_conn, vertex_coordinates=new_coords, user_vertices=user_vertices
    )
    return new_tiler
