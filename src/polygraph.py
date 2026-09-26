import networkx as nx
import itertools
from collections import deque


class PolyGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.next_halfedge_index = 0
        self.next_vertex_index = 0
        self.next_face_index = 0
        self.next_boundary_index = 0

        self.halfedge_tag = "h"
        self.vertex_tag = "v"
        self.face_tag = "f"
        self.boundary_tag = "b"

        self.vertex_coordinates = None
        self.user_defined_vertices = set()

    @classmethod
    def from_face_loops(cls, face_loops, vertex_coordinates=None):
        graph = cls()

        num_faces = len(face_loops)
        num_halfedges = sum(len(l) for l in face_loops)
        vertex_ids = set(itertools.chain.from_iterable(face_loops))
        num_vertices = len(vertex_ids)

        if vertex_coordinates is not None:
            assert all(
                v in vertex_coordinates for v in vertex_ids), "Some vertices were not found in vertex_coordinates"

        graph.vertex_coordinates = vertex_coordinates
        graph.user_defined_vertices = vertex_ids
        graph.next_halfedge_index = num_halfedges
        graph.next_vertex_index = num_vertices
        graph.next_face_index = num_faces
        graph.next_boundary_index = 0

        halfedge_ids = [(idx, graph.halfedge_tag) for idx in range(graph.next_halfedge_index)]
        graph.add_nodes_from(halfedge_ids, type="halfedge")

        vertex_ids = [(idx, graph.vertex_tag) for idx in vertex_ids]
        graph.add_nodes_from(vertex_ids, type="vertex")

        face_ids = [(idx, graph.face_tag) for idx in range(graph.next_face_index)]
        graph.add_nodes_from(face_ids, type="face")

        graph.initialize_half_edges_from_face_loops(face_loops)
        graph.add_halfedge_to_vertex_edges(face_loops)
        graph.add_halfedge_to_face_edges(face_loops)
        graph._add_twin_edges()
        graph._add_twin_source_target_edges()

        return graph

    def _add_face_loop(self, source_half_edges, next_half_edges):
        assert len(source_half_edges) == len(next_half_edges)
        assert next_half_edges[-1] == source_half_edges[0]

        source_edges_with_tags = [(idx, self.halfedge_tag) for idx in source_half_edges]
        next_edges_with_tags = [(idx, self.halfedge_tag) for idx in next_half_edges]

        next_edges = zip(source_edges_with_tags, next_edges_with_tags)
        self.add_edges_from(next_edges, type="next")
        prev_edges = zip(next_edges_with_tags, source_edges_with_tags)
        self.add_edges_from(prev_edges, type="previous")

    def add_face_loop(self, face_loop):
        next_indices = face_loop[1:] + [face_loop[0]]
        self._add_face_loop(face_loop, next_indices)

    def add_sequential_face_loop(self, halfedge_start, halfedge_stop):
        # !ASSUMES THAT HALF EDGES IN A FACE LOOP ARE INDEXED SEQUENTIALLY!
        assert halfedge_stop - halfedge_start + 1 >= 3
        source_indices = list(range(halfedge_start, halfedge_stop + 1))
        next_indices = source_indices[1:] + [source_indices[0]]
        self._add_face_loop(source_indices, next_indices)

    def initialize_half_edges_from_face_loops(self, face_loops):
        start = 0
        for loop in face_loops:
            stop = start + len(loop) - 1
            self.add_sequential_face_loop(start, stop)
            start = stop + 1

    def add_undirected_edge(self, source, dst, name):
        self.add_edge(source, dst, type=name)
        self.add_edge(dst, source, type=name)

    def add_undirected_edges(self, source, dst, name):
        # adds edges from source to dst and vice verse
        # both sets of edges are assigned the same name
        self.add_edges_from(zip(source, dst), type=name)
        self.add_edges_from(zip(dst, source), type=name)

    def remove_undirected_edge(self, source, dst):
        self.remove_edge(source, dst)
        self.remove_edge(dst, source)

    def add_edges_to_halfedge_source(self, face_loops):
        source_vertices = [(v, self.vertex_tag) for v in itertools.chain.from_iterable(face_loops)]
        halfedge_ids = [(h, self.halfedge_tag) for h in range(len(source_vertices))]
        self.add_undirected_edges(halfedge_ids, source_vertices, "source")

    def add_edges_to_halfedge_target(self, face_loops):
        target_vertices = []
        for loop in face_loops:
            rotated_loop = loop[1:] + [loop[0]]
            target_vertices += rotated_loop

        target_vertex_ids = [(v, self.vertex_tag) for v in target_vertices]
        halfedge_ids = [(h, self.halfedge_tag) for h in range(len(target_vertex_ids))]
        self.add_undirected_edges(halfedge_ids, target_vertex_ids, "target")

    def add_halfedge_to_vertex_edges(self, face_loops):
        self.add_edges_to_halfedge_source(face_loops)
        self.add_edges_to_halfedge_target(face_loops)

    def add_halfedge_to_face_edges(self, face_loops):
        face_ids = []
        for face_idx, loop in enumerate(face_loops):
            loop_face_ids = len(loop) * [face_idx]
            face_ids += loop_face_ids

        face_nodes = [(f, self.face_tag) for f in face_ids]
        halfedge_nodes = [(h, self.halfedge_tag) for h in range(len(face_nodes))]
        self.add_undirected_edges(halfedge_nodes, face_nodes, "face")

    @staticmethod
    def _ensure_tag_form(idx, tag):
        if isinstance(idx, tuple):
            # assert len(idx) == 2
            return idx
        else:
            return (idx, tag)

    @staticmethod
    def _ensure_untagged_form(idx):
        if isinstance(idx, tuple):
            return idx[0]
        else:
            return idx

    def _add_twin_edges(self):
        halfedge_nodes = [(h, self.halfedge_tag) for h in range(self.next_halfedge_index)]
        src_target = [(self.source_vertex(h), self.target_vertex(h)) for h in halfedge_nodes]
        src_target_to_halfedge = dict(zip(src_target, halfedge_nodes))
        twins = []
        boundary_nodes = []
        boundary_count = 0

        for idx in range(len(src_target)):
            src, target = src_target[idx]
            if (target, src) in src_target_to_halfedge:
                twin_edge = src_target_to_halfedge[(target, src)]
                twins.append(twin_edge)
            else:
                boundary_node = (boundary_count, self.boundary_tag)
                boundary_nodes.append(boundary_node)
                twins.append(boundary_node)
                boundary_count += 1

        self.next_boundary_index = boundary_count
        self.add_nodes_from(boundary_nodes, type="boundary")
        self.add_undirected_edges(halfedge_nodes, twins, "twin")

    def _add_twin_source_target_edges(self):
        boundary_nodes = [node for node in self.nodes() if node[1] == self.boundary_tag]
        boundary_source = []
        boundary_target = []
        for bnode in boundary_nodes:
            twin_edge = self.twin_halfedge(bnode)
            twin_source = self.source_vertex(twin_edge)
            twin_target = self.target_vertex(twin_edge)
            boundary_source.append(twin_target)
            boundary_target.append(twin_source)

        self.add_undirected_edges(boundary_nodes, boundary_source, "source")
        self.add_undirected_edges(boundary_nodes, boundary_target, "target")

    def _number_of_nodes(self, type):
        return sum(1 for vert, data in self.nodes(data=True) if data.get("type") == type)

    def number_of_edges_of_type(self, type=None):
        return sum(1 for src, dst, data in self.edges(data=True) if data.get("type") == type)

    def number_of_vertices(self):
        return self._number_of_nodes("vertex")

    def number_of_halfedges(self):
        return self._number_of_nodes("halfedge")

    def number_of_faces(self):
        return self._number_of_nodes("face")

    def vertex_coordinate(self, vertex_idx):
        vertex_idx = self._ensure_untagged_form(vertex_idx)
        return self.vertex_coordinates[vertex_idx]

    def source_vertex(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        vertex_id = next(dst for _, dst, data in self.edges(halfedge_index, data=True) if data.get("type") == "source")
        if tag:
            return vertex_id
        else:
            return vertex_id[0]

    def target_vertex(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        vertex_id = next(dst for _, dst, data in self.edges(halfedge_index, data=True) if data.get("type") == "target")
        if tag:
            return vertex_id
        else:
            return vertex_id[0]

    def twin_halfedge(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        twin = next(target for src, target, data in self.edges(halfedge_index, data=True) if data.get("type") == "twin")
        if tag:
            return twin
        else:
            return twin[0]

    def next_halfedge(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        next_edge = next(
            target for src, target, data in self.edges(halfedge_index, data=True) if data.get("type") == "next"
        )
        if tag:
            return next_edge
        else:
            return next_edge[0]

    def previous_halfedge(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        prev_edge = next(
            target for src, target, data in self.edges(halfedge_index, data=True) if data.get("type") == "previous"
        )
        if tag:
            return prev_edge
        else:
            return prev_edge[0]

    def face(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        face = next(
            target for src, target, data in self.edges(halfedge_index, data=True) if data.get("type") == "face"
        )
        if tag:
            return face
        else:
            return face[0]

    def face_halfedges(self, face_idx, tag=True):
        """return all halfedges connected to a face"""
        face_idx = self._ensure_tag_form(face_idx, self.face_tag)
        halfedges = [h for f, h, data in self.edges(face_idx, data=True) if data.get("type") == "face"]
        if tag:
            return halfedges
        else:
            halfedges = [h[0] for h in halfedges]
            return halfedges

    def first_face_halfedge(self, face_idx, tag=True):
        face_idx = self._ensure_tag_form(face_idx, self.face_tag)
        halfedge = next(h for f, h, data in self.edges(face_idx, data=True) if data.get("type") == "face")
        if tag:
            return halfedge
        else:
            return halfedge[0]

    def generate_halfedge_face_loop(self, halfedge_idx):
        """generate list of halfedges in a face loop"""
        halfedge_idx = self._ensure_tag_form(halfedge_idx, self.halfedge_tag)
        loop = [halfedge_idx]
        next_halfedge = self.next_halfedge(halfedge_idx)
        while next_halfedge != halfedge_idx:
            loop.append(next_halfedge)
            next_halfedge = self.next_halfedge(next_halfedge)

        return loop

    def halfedge_on_boundary(self, hidx):
        twin = self.twin_halfedge(hidx)
        return self.nodes[twin]["type"] == "boundary"

    def is_halfedge(self, hidx):
        if hidx is None:
            return False

        hidx = self._ensure_tag_form(hidx, self.halfedge_tag)
        return self.has_node(hidx) and self.nodes[hidx].get("type") == "halfedge"

    def _vertex_to_halfedge(self, vidx):
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)
        return (hidx for vidx, hidx in self.edges(vidx) if self.is_halfedge(hidx))

    def vertex_on_boundary(self, vidx):
        return any(self.halfedge_on_boundary(hidx) for hidx in self._vertex_to_halfedge(vidx))

    def _node_list_by_type(self, node_type, tag):
        nodes = [node for node, data in self.nodes(data=True) if data.get("type") == node_type]
        if not tag:
            nodes = [self._ensure_untagged_form(idx) for idx in nodes]
        return nodes

    def halfedge_list(self, tag=True):
        return self._node_list_by_type("halfedge", tag)

    def face_list(self, tag=True):
        return self._node_list_by_type("face", tag)

    def vertex_list(self, tag=True):
        return self._node_list_by_type("vertex", tag)

    def is_user_defined_vertex(self, vidx):
        vidx = self._ensure_untagged_form(vidx)
        return vidx in self.user_defined_vertices

    def vertex_degree(self, vertex_index):
        vertex_index = self._ensure_tag_form(vertex_index, self.vertex_tag)
        d = self.in_degree(vertex_index) // 2
        return d

    def face_degree(self, face_index):
        face_index = self._ensure_tag_form(face_index, self.face_tag)
        d = self.in_degree(face_index)
        return d

    def create_face(self):
        new_face_idx = (self.next_face_index, self.face_tag)
        self.add_node(new_face_idx, type="face")
        self.next_face_index += 1
        return new_face_idx

    def create_halfedge(self, next_halfedge, prev_halfedge):
        source_vertex = self.target_vertex(prev_halfedge)
        target_vertex = self.source_vertex(next_halfedge)
        face_idx = self.face(next_halfedge)
        halfedge_idx = (self.next_halfedge_index, self.halfedge_tag)
        self.add_node(halfedge_idx, type="halfedge")
        self.add_undirected_edge(halfedge_idx, source_vertex, "source")
        self.add_undirected_edge(halfedge_idx, target_vertex, "target")
        self.add_undirected_edge(halfedge_idx, face_idx, "face")
        self.associate_previous_next(prev_halfedge, halfedge_idx)
        self.associate_previous_next(halfedge_idx, next_halfedge)
        self.next_halfedge_index += 1

        return halfedge_idx

    def create_boundary_halfedge(self, target_vertex, source_vertex):
        boundary_edge = (self.next_boundary_index, self.boundary_tag)
        self.add_node(boundary_edge, type="boundary")
        self.add_undirected_edge(boundary_edge, target_vertex, "target")
        self.add_undirected_edge(boundary_edge, source_vertex, "source")
        self.next_boundary_index += 1
        return boundary_edge

    def create_vertex(self, coord=None):
        new_vertex_idx = (self.next_vertex_index, self.vertex_tag)
        self.add_node(new_vertex_idx, type="vertex")
        if coord is not None and self.vertex_coordinates is not None:
            # assert len(coord) == 2
            self.vertex_coordinates[self.next_vertex_index] = coord
        self.next_vertex_index += 1
        return new_vertex_idx

    def associate_previous_next(self, prev_halfedge, next_halfedge):
        self.add_edge(prev_halfedge, next_halfedge, type="next")
        self.add_edge(next_halfedge, prev_halfedge, type="previous")

    def disassociate_next_halfedge(self, halfedge):
        next_halfedge = self.next_halfedge(halfedge)
        self.remove_edge(halfedge, next_halfedge)
        self.remove_edge(next_halfedge, halfedge)

    def disassociate_previous_halfedge(self, halfedge):
        prev_halfedge = self.previous_halfedge(halfedge)
        self.remove_edge(halfedge, prev_halfedge)
        self.remove_edge(prev_halfedge, halfedge)

    def is_valid_edge_insert(self, hidx, k):
        if not self.is_halfedge(hidx):
            return False

        face_idx = self.face(hidx)
        if not (1 <= k <= self.face_degree(face_idx) - 3):
            return False

        # If the target vertex to connect to is the same as the source vertex,
        # this is an invalid action
        current_halfedge = hidx
        current_source = self.source_vertex(hidx)
        for steps in range(k):
            current_halfedge = self.next_halfedge(current_halfedge)
        target_vertex = self.target_vertex(current_halfedge)
        if current_source == target_vertex:
            return False

        return True

    def insert_halfedge(self, hidx, k):
        """insert an edge from source of `hidx` to target of halfedge that is k `next` steps away"""
        assert self.is_valid_edge_insert(hidx, k), "Invalid edge insert encountered"
        hidx = self._ensure_tag_form(hidx, self.halfedge_tag)
        face_idx = self.face(hidx)

        new_face_idx = self.create_face()
        # remove edges to current face and add edge to new face
        current_halfedge = hidx
        for count in range(k + 1):
            self.remove_undirected_edge(current_halfedge, face_idx)
            self.add_undirected_edge(current_halfedge, new_face_idx, "face")
            current_halfedge = self.next_halfedge(current_halfedge)

        new_face_next_halfedge = hidx
        new_face_prev_halfedge = self.previous_halfedge(current_halfedge)
        old_face_next_halfedge = current_halfedge
        old_face_prev_halfedge = self.previous_halfedge(hidx)

        self.remove_undirected_edge(new_face_prev_halfedge, old_face_next_halfedge)
        self.remove_undirected_edge(new_face_next_halfedge, old_face_prev_halfedge)

        new_face_new_halfedge = self.create_halfedge(new_face_next_halfedge, new_face_prev_halfedge)
        old_face_new_halfedge = self.create_halfedge(old_face_next_halfedge, old_face_prev_halfedge)
        self.add_undirected_edge(new_face_new_halfedge, old_face_new_halfedge, "twin")

        return new_face_idx

    def set_target_vertex(self, hidx, vidx):
        """
        delete edges between hidx and its current target vertex
        delete edges between twin of hidx and current target of hidx
        insert target edges between hidx and vidx
        insert source edges between twin of hidx and vidx

        :param hidx: halfedge index
        :param vidx: vertex index
        :return: None
        """
        current_target = self.target_vertex(hidx)
        twin_edge = self.twin_halfedge(hidx)
        self.remove_undirected_edge(hidx, current_target)
        self.remove_undirected_edge(twin_edge, current_target)
        self.add_undirected_edge(hidx, vidx, "target")
        self.add_undirected_edge(twin_edge, vidx, "source")

    def get_new_vertex_coordinate(self, next_vertex, previous_vertex):
        if self.vertex_coordinates is not None:
            coord1 = self.vertex_coordinate(next_vertex)
            coord2 = self.vertex_coordinate(previous_vertex)
            new_coord = [(c1 + c2) / 2 for c1, c2 in zip(coord1, coord2)]
            return new_coord
        else:
            return None

    def _insert_boundary_vertex(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.halfedge_tag)
        assert self.is_halfedge(hidx)
        assert self.halfedge_on_boundary(hidx)

        current_target_vertex = self.target_vertex(hidx)
        current_source_vertex = self.source_vertex(hidx)

        new_vertex_coord = self.get_new_vertex_coordinate(current_target_vertex, current_source_vertex)
        new_vertex_idx = self.create_vertex(new_vertex_coord)

        next_halfedge = self.next_halfedge(hidx)
        self.disassociate_next_halfedge(hidx)

        self.set_target_vertex(hidx, new_vertex_idx)

        new_halfedge = self.create_halfedge(next_halfedge, hidx)
        new_boundary_edge = self.create_boundary_halfedge(new_vertex_idx, current_target_vertex)
        self.add_undirected_edge(new_halfedge, new_boundary_edge, "twin")

        return new_vertex_idx

    def _insert_interior_vertex(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.halfedge_tag)
        assert self.is_halfedge(hidx)
        assert not self.halfedge_on_boundary(hidx)

        next_hidx = self.next_halfedge(hidx)
        twin_hidx = self.twin_halfedge(hidx)
        twin_prev_hidx = self.previous_halfedge(twin_hidx)

        current_target_vertex = self.target_vertex(hidx)
        current_source_vertex = self.source_vertex(hidx)
        new_vertex_coord = self.get_new_vertex_coordinate(current_target_vertex, current_source_vertex)
        new_vertex_idx = self.create_vertex(new_vertex_coord)

        self.disassociate_next_halfedge(hidx)
        self.disassociate_previous_halfedge(twin_hidx)

        self.set_target_vertex(hidx, new_vertex_idx)

        new_next_hidx = self.create_halfedge(next_hidx, hidx)
        new_twin_prev_hidx = self.create_halfedge(twin_hidx, twin_prev_hidx)
        self.add_undirected_edge(new_next_hidx, new_twin_prev_hidx, "twin")

        return new_vertex_idx

    def insert_vertex(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.halfedge_tag)
        assert self.is_halfedge(hidx)
        if self.halfedge_on_boundary(hidx):
            new_vertex = self._insert_boundary_vertex(hidx)
        else:
            new_vertex = self._insert_interior_vertex(hidx)

        if tag:
            return new_vertex
        else:
            return self._ensure_untagged_form(new_vertex)

    def _delete_vertex_coordinates(self, vidx):
        vidx = self._ensure_untagged_form(vidx)
        if self.vertex_coordinates is not None:
            coords = self.vertex_coordinates.pop(vidx, None)
            if coords is None:
                print("Warning: deleted vertex not found in vertex coordinates")

    def _delete_halfedge(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.halfedge_tag)
        twin = self.twin_halfedge(hidx)
        self.remove_node(hidx)
        self.remove_node(twin)

    def _delete_vertex(self, vidx):
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)
        self.remove_node(vidx)
        self._delete_vertex_coordinates(vidx)

    def _delete_boundary_vertex(self, hidx):
        """deletes vertex at source of hidx"""
        hidx = self._ensure_tag_form(hidx, self.halfedge_tag)
        assert self.halfedge_on_boundary(hidx)
        source_vertex = self.source_vertex(hidx)
        assert self.vertex_degree(source_vertex) == 2
        assert not self.is_user_defined_vertex(source_vertex)

        next_halfedge = self.next_halfedge(hidx)
        prev_halfedge = self.previous_halfedge(hidx)

        target_vertex = self.target_vertex(hidx)
        self.set_target_vertex(prev_halfedge, target_vertex)
        self.associate_previous_next(prev_halfedge, next_halfedge)

        self._delete_halfedge(hidx)
        self._delete_vertex(source_vertex)

    def _delete_interior_vertex(self, hidx):
        """deletes vertex at source of hidx"""
        hidx = self._ensure_tag_form(hidx, self.halfedge_tag)
        assert not self.halfedge_on_boundary(hidx)
        source_vertex = self.source_vertex(hidx)
        assert self.vertex_degree(source_vertex) == 2

        next_halfedge = self.next_halfedge(hidx)
        prev_halfedge = self.previous_halfedge(hidx)
        twin_halfedge = self.twin_halfedge(hidx)

        next_twin_halfedge = self.next_halfedge(twin_halfedge)
        prev_twin_halfedge = self.previous_halfedge(twin_halfedge)

        target_vertex = self.target_vertex(hidx)
        self.set_target_vertex(prev_halfedge, target_vertex)
        self.associate_previous_next(prev_halfedge, next_halfedge)
        self.associate_previous_next(prev_twin_halfedge, next_twin_halfedge)

        self._delete_halfedge(hidx)
        self._delete_vertex(source_vertex)

    def is_valid_delete_source_vertex(self, hidx):
        """Check if vertex at source of hidx can be deleted"""
        if not self.is_halfedge(hidx):
            return False

        vidx = self.source_vertex(hidx)
        if self.is_user_defined_vertex(vidx):
            return False

        if self.vertex_degree(vidx) > 2:
            return False

        prev_hidx = self.previous_halfedge(hidx)
        prev_src = self.source_vertex(prev_hidx)
        curr_target = self.target_vertex(hidx)
        if prev_src == curr_target:
            return False

        fidx = self.face(hidx)
        if self.face_degree(fidx) < 4:
            return False

        if not self.halfedge_on_boundary(hidx):
            twin_edge = self.twin_halfedge(hidx)
            twin_face = self.face(twin_edge)
            if self.face_degree(twin_face) < 4:
                return False

        return True

    def delete_source_vertex(self, hidx):
        assert self.is_valid_delete_source_vertex(hidx), "cannot delete vertex at source of : " + str(hidx)
        if self.halfedge_on_boundary(hidx):
            self._delete_boundary_vertex(hidx)
        else:
            self._delete_interior_vertex(hidx)

    def is_valid_delete_halfedge(self, hidx):
        if not self.is_halfedge(hidx):
            return False

        if self.halfedge_on_boundary(hidx):
            return False

        source_vertex = self.source_vertex(hidx)
        target_vertex = self.target_vertex(hidx)
        if self.vertex_degree(source_vertex) <= 2 or self.vertex_degree(target_vertex) <= 2:
            return False

        # Deleting a half-edge results in merging of faces on either side.
        # If these faces are already the same, this is an invalid delete
        twin_edge = self.twin_halfedge(hidx)
        current_face = self.face(hidx)
        twin_face = self.face(twin_edge)
        if current_face == twin_face:
            return False

        return True

    def _halfedges_of_face(self, fidx):
        fidx = self._ensure_tag_form(fidx, self.face_tag)
        edges = (target for src, target, data in self.edges(fidx, data=True) if data.get("type") == "face")
        return edges

    def delete_halfedge(self, hidx):
        assert self.is_valid_delete_halfedge(hidx)

        next_edge = self.next_halfedge(hidx)
        prev_edge = self.previous_halfedge(hidx)
        twin_edge = self.twin_halfedge(hidx)
        prev_twin = self.previous_halfedge(twin_edge)
        next_twin = self.next_halfedge(twin_edge)

        current_face = self.face(hidx)
        twin_face = self.face(twin_edge)
        # Associate all edges from neighboring face with current face and delete neighboring face
        twin_face_edges = self._halfedges_of_face(twin_face)
        for halfedge in twin_face_edges:
            self.add_undirected_edge(halfedge, current_face, "face")
        self.remove_node(twin_face)

        self.associate_previous_next(prev_edge, next_twin)
        self.associate_previous_next(prev_twin, next_edge)
        self._delete_halfedge(hidx)

        # return the deleted face
        return twin_face

    def knn_halfedges(self, hidx, k):
        hidx = self._ensure_tag_form(hidx, self.halfedge_tag)
        assert self.is_halfedge(hidx)
        assert k >= 1

        seen = set()
        queue = deque()

        seen.add(hidx)
        queue.append(hidx)
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

            next_edge = self.next_halfedge(edge)
            append_if_not_seen(next_edge)

            prev_edge = self.previous_halfedge(edge)
            append_if_not_seen(prev_edge)

            if not self.halfedge_on_boundary(edge):
                twin_edge = self.twin_halfedge(edge)
                append_if_not_seen(twin_edge)

        return neighbors

    def _accumulate_neighbor_coordinates(self):
        accumulated_coords = {k: 2 * [0] for k in self.vertex_coordinates.keys()}
        for hidx in self.halfedge_list():
            src_vidx = self.source_vertex(hidx, tag=False)
            src_coords = self.vertex_coordinates[src_vidx]
            dst_vidx = self.target_vertex(hidx, tag=False)
            dst_coords = accumulated_coords[dst_vidx]

            dst_coords[0] += src_coords[0]
            dst_coords[1] += src_coords[1]

        return accumulated_coords

    def laplace_smooth_vertices(self):
        accumulated_coords = self._accumulate_neighbor_coordinates()

        # average the coordinates
        for vidx, coords in accumulated_coords.items():
            degree = self.vertex_degree(vidx)

            coords[0] /= degree
            coords[1] /= degree

        # set boundary coordinates to their original values
        for vidx in accumulated_coords.keys():
            if self.vertex_on_boundary(vidx):
                accumulated_coords[vidx] = self.vertex_coordinates[vidx]

        self.vertex_coordinates = accumulated_coords
        return
