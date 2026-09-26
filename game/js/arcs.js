// Curved boundaries in the browser, carried natively rather than discretised.
//
// The port of src/boundary_arcs.py. A curve is not a different problem:
// topology never asks what shape an edge is, and the corner want at a vertex is
// its TANGENT angle -- so a smooth arc joint reads 180 and wants degree three,
// exactly like a flat point on a straight edge. Only two operations need to
// know an edge is curved: inserting a vertex on it, which must land ON the arc
// and splits it in two, and smoothing a vertex that sits on it, which walks off
// the curve otherwise.
//
// Keyed by vertex PAIR rather than half-edge id, because ids do not survive the
// edits and vertex ids do. Each entry records which endpoint the arc runs FROM,
// so a split hands its halves to the right pairs without guessing.

"use strict";

function angleOf(v) { return Math.atan2(v[1], v[0]); }
function unit(a) { return [Math.cos(a), Math.sin(a)]; }

class Arc {
  constructor(centre, radius, start, end, ccw) {
    this.centre = [centre[0], centre[1]];
    this.radius = radius;
    this.start = start;
    this.end = end;
    this.ccw = !!ccw;
  }

  sweep() {
    let d = (this.end - this.start) % (2 * Math.PI);
    if (d < 0) d += 2 * Math.PI;
    return this.ccw ? d : d - 2 * Math.PI;
  }

  point(t) {
    const a = this.start + t * this.sweep();
    const u = unit(a);
    return [this.centre[0] + this.radius * u[0], this.centre[1] + this.radius * u[1]];
  }

  project(p) {
    const off = [p[0] - this.centre[0], p[1] - this.centre[1]];
    const n = Math.hypot(off[0], off[1]);
    if (n < 1e-12) return this.point(0.5);
    const sweep = this.sweep();
    if (Math.abs(sweep) < 1e-12) return this.point(0);
    const angle = angleOf(off);
    let d = (angle - this.start) % (2 * Math.PI);
    if (d < 0) d += 2 * Math.PI;
    // A clockwise arc runs the other way round, so d belongs in (-2pi, 0].
    // Subtracting unconditionally moves an exact 0 -- the arc's own START --
    // to -2pi, whose fraction is 4 for a quarter turn, clamps to 1, and returns
    // the END. Every pinned vertex on a concave arc then teleports to the far
    // corner the moment anything projects it.
    if (sweep < 0 && d > 0) d -= 2 * Math.PI;
    const t = d / sweep;
    if (t >= 0 && t <= 1) return this.point(t);
    // Outside the span: clamp to the NEARER end, not to whichever end the
    // parameterisation happened to run past.
    const wrap = (x) => ((x + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return this.point(Math.abs(wrap(angle - this.start)) <= Math.abs(wrap(angle - this.end)) ? 0 : 1);
  }

  split(t) {
    const mid = angleOf([this.point(t)[0] - this.centre[0],
                         this.point(t)[1] - this.centre[1]]);
    return [new Arc(this.centre, this.radius, this.start, mid, this.ccw),
            new Arc(this.centre, this.radius, mid, this.end, this.ccw)];
  }

  clone() { return new Arc(this.centre, this.radius, this.start, this.end, this.ccw); }
}

class BoundaryArcs {
  constructor() { this.map = new Map(); }
  static key(a, b) { return a <= b ? a + ":" + b : b + ":" + a; }
  get size() { return this.map.size; }
  get(a, b) { const e = this.map.get(BoundaryArcs.key(a, b)); return e ? e.arc : null; }
  add(a, b, arc, startVertex) { this.map.set(BoundaryArcs.key(a, b), { arc, start: startVertex }); }

  angleAt(a, b, vertex) {
    const e = this.map.get(BoundaryArcs.key(a, b));
    if (!e) return null;
    return e.start === vertex ? e.arc.start : e.arc.end;
  }

  // Undo a split: deleting a vertex that was inserted ON a curve has to put the
  // curve back. Without this the two halves stay keyed on a vertex that no
  // longer exists and the edge that reappears is silently STRAIGHT.
  merge(first, middle, second) {
    const left = this.map.get(BoundaryArcs.key(first, middle));
    const right = this.map.get(BoundaryArcs.key(middle, second));
    if (!left || !right) {
      this.map.delete(BoundaryArcs.key(first, middle));
      this.map.delete(BoundaryArcs.key(middle, second));
      return null;
    }
    const startAngle = this.angleAt(first, middle, first);
    const endAngle = this.angleAt(middle, second, second);
    const arc = left.arc;
    this.map.delete(BoundaryArcs.key(first, middle));
    this.map.delete(BoundaryArcs.key(middle, second));
    const merged = new Arc(arc.centre, arc.radius, startAngle, endAngle, arc.ccw);
    this.map.set(BoundaryArcs.key(first, second), { arc: merged, start: first });
    return merged;
  }

  split(a, b, middle) {
    const k = BoundaryArcs.key(a, b);
    const e = this.map.get(k);
    if (!e) return null;
    this.map.delete(k);
    const [first, second] = e.arc.split(0.5);
    const finish = e.start === a ? b : a;
    this.map.set(BoundaryArcs.key(e.start, middle), { arc: first, start: e.start });
    this.map.set(BoundaryArcs.key(middle, finish), { arc: second, start: middle });
    return e.arc.point(0.5);
  }

  clone() {
    const out = new BoundaryArcs();
    for (const [k, v] of this.map) out.map.set(k, { arc: v.arc.clone(), start: v.start });
    return out;
  }

  entries() { return this.map.values(); }
}

// One arc per edge of a circle sector: helper for building the levels below.
function arcBetween(centre, radius, startAngle, endAngle) {
  return new Arc(centre, radius, startAngle, endAngle, true);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { Arc, BoundaryArcs, arcBetween };
}
if (typeof globalThis !== "undefined") {
  globalThis.MeshQuestArcs = { Arc, BoundaryArcs, arcBetween };
}
