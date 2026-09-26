"""Canonical fingerprints for mesh states.

Two flavours, both invariant under the (index, tag) labelling that `Tiler`
happens to hand out:

`certificate` is exact -- the lexicographically smallest DFS encoding of the
combinatorial map, so two states share it iff they are isomorphic. It costs
O(darts^2) and is what the search uses for dedup.

`weisfeiler_lehman_hash` is a cheap invariant: a few rounds of label
refinement over `next`/`previous`/`twin`, then a hash of the sorted multiset
of final labels. Collisions are possible in principle and vanishingly rare on
meshes this size, which is fine for an in-episode revisit check.
"""

import hashlib


def _desired_of(graph, desired, vidx, boundary_desired, interior_desired):
    if vidx in desired:
        return desired[vidx]
    return boundary_desired if graph.is_boundary_vertex(vidx) else interior_desired


def certificate(graph, desired, boundary_desired=3, interior_desired=4):
    """Lexicographically smallest DFS encoding of the combinatorial map."""
    darts = graph.half_edge_list()
    best = None
    for root in darts:
        order, index = [], {}
        stack = [root]
        while stack:
            d = stack.pop()
            if d in index:
                continue
            index[d] = len(order)
            order.append(d)
            nxt = graph.next_half_edge(d)
            twin = graph.twin_half_edge(d)
            for nb in (twin, nxt):
                if graph.is_half_edge(nb) and nb not in index:
                    stack.append(nb)
        if len(order) != len(darts):
            continue
        cert = []
        for d in order:
            nxt = graph.next_half_edge(d)
            twin = graph.twin_half_edge(d)
            src = graph.source_vertex(d, tag=False)
            cert.append((
                index.get(nxt, -1),
                index.get(twin, -1),
                graph.vertex_degree(src),
                _desired_of(graph, desired, src, boundary_desired, interior_desired),
                graph.face_degree(graph.face(d)),
            ))
        cert = tuple(cert)
        if best is None or cert < best:
            best = cert
    return best


def weisfeiler_lehman_hash(graph, desired, rounds=3,
                           boundary_desired=3, interior_desired=4):
    """A cheap isomorphism-invariant fingerprint of the mesh state."""
    darts = graph.half_edge_list()
    labels = {}
    for d in darts:
        src = graph.source_vertex(d, tag=False)
        labels[d] = hash((
            graph.vertex_degree(src),
            _desired_of(graph, desired, src, boundary_desired, interior_desired),
            graph.face_degree(graph.face(d)),
            graph.half_edge_on_boundary(d),
        ))

    for _ in range(rounds):
        new_labels = {}
        for d in darts:
            nxt = graph.next_half_edge(d)
            prv = graph.previous_half_edge(d)
            twin = graph.twin_half_edge(d)
            new_labels[d] = hash((
                labels[d],
                labels.get(nxt, 0),
                labels.get(prv, 0),
                labels.get(twin, 0),
            ))
        labels = new_labels

    digest = hashlib.blake2b(digest_size=16)
    for value in sorted(labels.values()):
        digest.update(value.to_bytes(8, "little", signed=True))
    return digest.hexdigest()
