"""A Gmsh .geo writer for line / circular-arc geometries."""


def to_geo(geometry, lc=0.1, recombine=True):
    lines = [f"lc = {float(lc)!r};"]
    point_id, curve_id, loop_ids = 0, 0, []
    for loop in geometry.loops:
        ids = []
        for p in loop.points:
            point_id += 1
            lines.append(f"Point({point_id}) = {{{float(p[0])!r}, {float(p[1])!r}, 0, lc}};")
            ids.append(point_id)
        curves = []
        n = loop.n
        for i, edge in enumerate(loop.edges):
            a, b = ids[i], ids[(i + 1) % n]
            curve_id += 1
            if edge.kind == "arc":
                point_id += 1
                c = edge.center
                lines.append(f"Point({point_id}) = {{{float(c[0])!r}, {float(c[1])!r}, 0, lc}};")
                lines.append(f"Circle({curve_id}) = {{{a}, {point_id}, {b}}};")
            else:
                lines.append(f"Line({curve_id}) = {{{a}, {b}}};")
            curves.append(curve_id)
        curve_id += 1
        lines.append(f"Curve Loop({curve_id}) = {{{', '.join(map(str, curves))}}};")
        loop_ids.append(curve_id)
    lines.append(f"Plane Surface(1) = {{{', '.join(map(str, loop_ids))}}};")
    if recombine:
        lines += ["Recombine Surface{1};", "Mesh.Algorithm = 8;", "Mesh.RecombinationAlgorithm = 1;"]
    return "\n".join(lines) + "\n"
