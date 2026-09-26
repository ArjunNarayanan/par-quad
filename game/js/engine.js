/* Mesh Quest — Copyright 2026 Arjun Narayanan.
 * Licensed under the MIT License (see LICENSE).
 * Released under the MIT License (see LICENSE).
 */
/* Mesh Quest engine: JavaScript port of the game subset of src/tiler.py.
 *
 * Faithful to the Python implementation including ID assignment order, so
 * states can be diffed against the Python engine op-for-op. Verified by
 * replaying identical op traces through both engines (see game/js/tests).
 *
 * IDs are tagged strings: h3 (half-edge), v2 (vertex), f1 (face), b0 (boundary).
 */
"use strict";

// The element the player is cutting towards. Quads by default; `setElementTarget`
// switches the whole engine to triangles, which changes what every corner wants
// and what the par bound is, but nothing about the moves.
let TARGET_ANGLE = 90;
let FACE_DESIRED = 4;
const QUALITY_THRESHOLD = 0.4;

// numpy-style round-half-to-even (np.round); angles here are non-negative
function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// sin of the ideal corner: 1 for quads, sin 60 for triangles. Dividing by it
// makes the quality an element quality rather than a quad quality.
function idealSin() {
  return Math.sin((TARGET_ANGLE * Math.PI) / 180);
}

function desiredFromAngle(angle, target = TARGET_ANGLE) {
  // Semicircle's 150 degree corners give exactly 2.5 at the triangle target,
  // and atan2 lands a hair either side of it in the two implementations --
  // enough to move a corner into the next bin and change par. Quantise away
  // the sub-nano noise so a tie is a tie on both sides, then round half to
  // even as numpy does.
  return Math.max(roundHalfEven(Math.round((angle / target) * 1e9) / 1e9) + 1, 2);
}

// Every degree a corner of this angle would be content with. When a / theta
// lands exactly on k + 1/2 the two neighbouring corner treatments are equally
// admissible -- k elements of angle a / k, or k + 1 of a / (k+1) -- so rounding
// picks one of them by convention, and half-to-even does not even break
// consecutive ties the same way. Such a corner keeps both, so neither the score
// nor par depends on which integer happens to be even.
function desiredOptions(angle, target = TARGET_ANGLE) {
  const ratio = Math.round((angle / target) * 1e9) / 1e9;
  if (Math.abs(ratio - Math.floor(ratio) - 0.5) > 1e-6) {
    return [desiredFromAngle(angle, target)];
  }
  const low = Math.max(Math.floor(ratio) + 1, 2);
  const high = Math.max(Math.ceil(ratio) + 1, 2);
  return low === high ? [low] : [low, high];
}

let INTERIOR_DESIRED = desiredFromAngle(360) - 1; // 4 for quads, 6 for triangles
let BOUNDARY_DESIRED = desiredFromAngle(180); // 3 for quads, 4 for triangles

// The element currently being cut towards. A real top-level function, not just
// an export: the browser build concatenates these files with no module system,
// so the UI can only see what is declared at this level.
function elementTarget() {
  return FACE_DESIRED;
}

// Switch the engine between quads (4) and triangles (3). Every constant below
// is derived, so callers only ever set this one number.
function setElementTarget(faceDesired) {
  FACE_DESIRED = faceDesired;
  TARGET_ANGLE = ((faceDesired - 2) * 180) / faceDesired;
  INTERIOR_DESIRED = desiredFromAngle(360) - 1;
  BOUNDARY_DESIRED = desiredFromAngle(180);
}

function num(id) {
  return parseInt(id.slice(1), 10);
}

class GameError extends Error {}

class Mesh {
  constructor() {
    this.arcs = null;   // set by a curved level; every path below no-ops without it
    this.he = new Map(); // tagged id -> {face,next,prev,twin,source,target}
    this.nodeType = new Map(); // tagged id -> "half_edge"|"vertex"|"face"|"boundary"
    this.coords = new Map(); // numeric vid -> [x, y]
    this.userVerts = new Set(); // numeric
    this.boundaryVerts = new Set(); // numeric
    this.vertexDeg = new Map(); // vertex id -> degree
    this.faceDeg = new Map(); // face id -> degree
    this.faceHE = new Map(); // face id -> Set of half-edge ids (insertion order)
    this.nextH = 0;
    this.nextV = 0;
    this.nextF = 0;
    this.nextB = 0;
  }

  static fromFaceLoops(loops, coords, userVertices = null, arcs = null) {
    const m = new Mesh();
    if (arcs) m.arcs = arcs;
    const vertexIds = new Set();
    for (const loop of loops) for (const v of loop) vertexIds.add(v);
    const pinned = userVertices === null ? vertexIds : new Set(userVertices);

    m.nextH = loops.reduce((s, l) => s + l.length, 0);
    m.nextV = vertexIds.size;
    m.nextF = loops.length;

    for (let i = 0; i < m.nextH; i++) {
      m.he.set("h" + i, { face: null, next: null, prev: null, twin: null, source: null, target: null });
      m.nodeType.set("h" + i, "half_edge");
    }
    for (const v of vertexIds) {
      m.nodeType.set("v" + v, "vertex");
      m.coords.set(v, [coords[v][0], coords[v][1]]);
      if (pinned.has(v)) m.userVerts.add(v);
    }
    for (let f = 0; f < m.nextF; f++) {
      m.nodeType.set("f" + f, "face");
      m.faceHE.set("f" + f, new Set());
    }

    // next/prev, source/target, face associations, loop by loop
    let start = 0;
    for (let fi = 0; fi < loops.length; fi++) {
      const loop = loops[fi];
      const n = loop.length;
      for (let i = 0; i < n; i++) {
        const h = "h" + (start + i);
        const hn = "h" + (start + ((i + 1) % n));
        const rec = m.he.get(h);
        rec.next = hn;
        m.he.get(hn).prev = h;
        rec.source = "v" + loop[i];
        rec.target = "v" + loop[(i + 1) % n];
        rec.face = "f" + fi;
        m.faceHE.get("f" + fi).add(h);
      }
      start += n;
    }

    // twins (iteration in half-edge id order, matching the Python dict order)
    const pair = new Map(); // "src|dst" -> hid
    for (let i = 0; i < m.nextH; i++) {
      const h = "h" + i;
      const r = m.he.get(h);
      pair.set(r.source + "|" + r.target, h);
    }
    for (const [key, h] of pair) {
      const [src, dst] = key.split("|");
      const rev = pair.get(dst + "|" + src);
      if (rev !== undefined) {
        m.he.get(h).twin = rev;
        m.he.get(rev).twin = h;
      } else {
        const b = "b" + m.nextB;
        m.nextB += 1;
        m.nodeType.set(b, "boundary");
        m.he.set(b, { face: null, next: null, prev: null, twin: h, source: dst, target: src });
        m.he.get(h).twin = b;
      }
    }

    // degrees: every mesh edge is represented by two directed records
    // (half-edge/half-edge in the interior, half-edge/boundary at the rim),
    // so a vertex's degree is half its incidence count over all records
    for (const v of vertexIds) m.vertexDeg.set("v" + v, 0);
    for (const [id, type] of m.nodeType) {
      if (type !== "half_edge" && type !== "boundary") continue;
      const r = m.he.get(id);
      m.vertexDeg.set(r.source, m.vertexDeg.get(r.source) + 1);
      m.vertexDeg.set(r.target, m.vertexDeg.get(r.target) + 1);
    }
    for (const [v, d] of m.vertexDeg) m.vertexDeg.set(v, d / 2);

    for (let f = 0; f < m.nextF; f++) m.faceDeg.set("f" + f, loops[f].length);

    for (const [id, type] of m.nodeType) {
      if (type === "half_edge" && m.onBoundary(id)) {
        m.boundaryVerts.add(num(m.he.get(id).source));
      }
    }
    return m;
  }

  clone() {
    const m = new Mesh();
    for (const [k, v] of this.he) m.he.set(k, { ...v });
    for (const [k, v] of this.nodeType) m.nodeType.set(k, v);
    for (const [k, v] of this.coords) m.coords.set(k, [v[0], v[1]]);
    m.userVerts = new Set(this.userVerts);
    m.boundaryVerts = new Set(this.boundaryVerts);
    m.vertexDeg = new Map(this.vertexDeg);
    m.faceDeg = new Map(this.faceDeg);
    for (const [k, v] of this.faceHE) m.faceHE.set(k, new Set(v));
    m.nextH = this.nextH; m.nextV = this.nextV; m.nextF = this.nextF; m.nextB = this.nextB;
    // arcs travel with the copy, or undo quietly straightens the boundary
    if (this.arcs) m.arcs = this.arcs.clone();
    return m;
  }

  isHalfEdge(h) { return this.nodeType.get(h) === "half_edge"; }
  isBoundaryHalfEdge(h) { return this.nodeType.get(h) === "boundary"; }
  onBoundary(h) { return this.he.get(this.he.get(h).twin).face === null; }

  halfEdgeList() {
    const out = [];
    for (const [id, t] of this.nodeType) if (t === "half_edge") out.push(id);
    return out;
  }
  vertexList() {
    const out = [];
    for (const [id, t] of this.nodeType) if (t === "vertex") out.push(id);
    return out;
  }
  faceList() {
    const out = [];
    for (const [id, t] of this.nodeType) if (t === "face") out.push(id);
    return out;
  }

  faceLoop(h) {
    const loop = [h];
    let cur = this.he.get(h).next;
    while (cur !== h) {
      loop.push(cur);
      cur = this.he.get(cur).next;
    }
    return loop;
  }

  firstFaceHalfEdge(f) {
    for (const h of this.faceHE.get(f)) return h;
    throw new GameError("face has no half-edges: " + f);
  }

  // --- construction helpers (ID order matches Python) ---------------------

  createFace() {
    const f = "f" + this.nextF;
    this.nextF += 1;
    this.nodeType.set(f, "face");
    this.faceHE.set(f, new Set());
    return f;
  }

  createVertex(coord, onBdry) {
    const v = "v" + this.nextV;
    this.nodeType.set(v, "vertex");
    if (coord) this.coords.set(this.nextV, [coord[0], coord[1]]);
    if (onBdry) this.boundaryVerts.add(this.nextV);
    this.nextV += 1;
    return v;
  }

  createHalfEdge(nextH, prevH) {
    const nr = this.he.get(nextH);
    const pr = this.he.get(prevH);
    const h = "h" + this.nextH;
    this.nextH += 1;
    this.nodeType.set(h, "half_edge");
    this.he.set(h, {
      face: nr.face, next: nextH, prev: prevH,
      twin: null, source: pr.target, target: nr.source,
    });
    nr.prev = h;
    pr.next = h;
    if (nr.face !== null) this.faceHE.get(nr.face).add(h);
    return h;
  }

  createBoundaryHalfEdge(targetV, sourceV) {
    const b = "b" + this.nextB;
    this.nextB += 1;
    this.nodeType.set(b, "boundary");
    this.he.set(b, { face: null, next: null, prev: null, twin: null, source: sourceV, target: targetV });
    return b;
  }

  associateTwin(a, b) {
    this.he.get(a).twin = b;
    this.he.get(b).twin = a;
  }

  setTargetVertex(h, v) {
    const r = this.he.get(h);
    const twin = this.he.get(r.twin);
    r.target = v;
    twin.source = v;
  }

  // --- edit operations -----------------------------------------------------

  isValidEdgeInsert(h, k) {
    if (!this.isHalfEdge(h)) return false;
    const f = this.he.get(h).face;
    return k >= 0 && k < this.faceDeg.get(f) - 1;
  }

  insertHalfEdge(h, k) {
    if (!this.isValidEdgeInsert(h, k)) throw new GameError("Invalid edge insert");
    const f = this.he.get(h).face;
    this.faceDeg.set(f, this.faceDeg.get(f) - k);
    const nf = this.createFace();
    this.faceDeg.set(nf, k + 2);

    let cur = h;
    for (let c = 0; c < k + 1; c++) {
      this.faceHE.get(f).delete(cur);
      this.he.get(cur).face = nf;
      this.faceHE.get(nf).add(cur);
      cur = this.he.get(cur).next;
    }

    const newFaceNext = h;
    const newFacePrev = this.he.get(cur).prev;
    const oldFaceNext = cur;
    const oldFacePrev = this.he.get(h).prev;

    const srcV = this.he.get(newFacePrev).target;
    const dstV = this.he.get(h).source;
    this.vertexDeg.set(srcV, this.vertexDeg.get(srcV) + 1);
    this.vertexDeg.set(dstV, this.vertexDeg.get(dstV) + 1);

    const a = this.createHalfEdge(newFaceNext, newFacePrev);
    const b = this.createHalfEdge(oldFaceNext, oldFacePrev);
    this.associateTwin(a, b);
  }

  midpoint(v1, v2) {
    const a = this.coords.get(num(v1));
    const b = this.coords.get(num(v2));
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  }

  snapToArcs() {
    // A boundary vertex on a curve is smoothed along the CHORD between its
    // neighbours, which walks it off the curve. Projecting afterwards keeps the
    // invariant with the object that owns it, and is a no-op without arcs.
    //
    // A vertex owns TWO arcs wherever two of them meet, so take the NEAREST
    // projection rather than each in turn: a vertex slid legitimately along one
    // arc is outside the span of its neighbour, and projecting onto that one
    // clamps it back to the shared joint and undoes the slide. See
    // Tiler.snap_to_boundary_arcs for the measured damage on the Python side.
    if (!this.arcs || this.arcs.size === 0) return;
    const owners = new Map();
    for (const [k, e] of this.arcs.map) {
      const [a, b] = k.split(":").map(Number);
      for (const v of [a, b]) {
        if (!this.coords.has(v)) continue;
        if (!this.boundaryVerts.has(v)) continue;
        if (!owners.has(v)) owners.set(v, []);
        owners.get(v).push(e.arc);
      }
    }
    for (const [v, arcs] of owners) {
      const p = this.coords.get(v);
      let best = null, distance = Infinity;
      for (const arc of arcs) {
        const q = arc.project(p);
        const gap = Math.hypot(q[0] - p[0], q[1] - p[1]);
        if (gap < distance) { best = q; distance = gap; }
      }
      this.coords.set(v, best);
    }
  }

  insertVertex(h) {
    if (!this.isHalfEdge(h)) throw new GameError("Unknown edge");
    if (this.onBoundary(h)) this._insertBoundaryVertex(h);
    else this._insertInteriorVertex(h);
  }

  _insertBoundaryVertex(h) {
    const r = this.he.get(h);
    const curTarget = r.target;
    const curSource = r.source;
    // On a curved edge the new vertex belongs ON the arc, not on the chord, or
    // the boundary gets chipped into a polygon one insertion at a time.
    const a = num(curSource), b = num(curTarget);
    const arc = this.arcs ? this.arcs.get(a, b) : null;
    const where = arc ? arc.point(0.5) : this.midpoint(curTarget, curSource);
    const nv = this.createVertex(where, true);
    this.vertexDeg.set(nv, 2);
    if (arc) this.arcs.split(a, b, num(nv));

    const nextH = r.next;
    this.setTargetVertex(h, nv);
    const nh = this.createHalfEdge(nextH, h);
    const nb = this.createBoundaryHalfEdge(nv, curTarget);
    this.associateTwin(nh, nb);
    this.faceDeg.set(r.face, this.faceDeg.get(r.face) + 1);
  }

  _insertInteriorVertex(h) {
    const r = this.he.get(h);
    const nextH = r.next;
    const twinH = r.twin;
    const twinPrev = this.he.get(twinH).prev;

    const curTarget = r.target;
    const curSource = r.source;
    const nv = this.createVertex(this.midpoint(curTarget, curSource), false);
    this.vertexDeg.set(nv, 2);

    this.setTargetVertex(h, nv);
    const a = this.createHalfEdge(nextH, h);
    const b = this.createHalfEdge(twinH, twinPrev);
    this.associateTwin(a, b);

    this.faceDeg.set(r.face, this.faceDeg.get(r.face) + 1);
    const tf = this.he.get(twinH).face;
    this.faceDeg.set(tf, this.faceDeg.get(tf) + 1);
  }

  isUserVertex(v) { return this.userVerts.has(num(v)); }

  isValidDeleteSourceVertex(h) {
    if (!this.isHalfEdge(h)) return false;
    const v = this.he.get(h).source;
    if (this.isUserVertex(v)) return false;
    if (this.vertexDeg.get(v) !== 2) return false;
    const f = this.he.get(h).face;
    if (this.faceDeg.get(f) < 3) return false;
    if (!this.onBoundary(h)) {
      const tf = this.he.get(this.he.get(h).twin).face;
      if (this.faceDeg.get(tf) < 3) return false;
    }
    return true;
  }

  _removeHalfEdgePair(h) {
    const twin = this.he.get(h).twin;
    for (const id of [h, twin]) {
      const r = this.he.get(id);
      if (r.face !== null && this.faceHE.has(r.face)) this.faceHE.get(r.face).delete(id);
      this.he.delete(id);
      this.nodeType.delete(id);
    }
  }

  _removeVertex(v) {
    this.nodeType.delete(v);
    this.vertexDeg.delete(v);
    this.coords.delete(num(v));
    this.boundaryVerts.delete(num(v));
  }

  deleteSourceVertex(h) {
    if (!this.isValidDeleteSourceVertex(h)) throw new GameError("Vertex not deletable");
    if (this.onBoundary(h)) this._deleteBoundaryVertex(h);
    else this._deleteInteriorVertex(h);
  }

  _deleteBoundaryVertex(h) {
    const r = this.he.get(h);
    const src = r.source;
    const nextH = r.next;
    const prevH = r.prev;
    // this vertex may sit ON a curve, splitting one arc in two; put it back
    if (this.arcs) {
      this.arcs.merge(num(this.he.get(prevH).source), num(src), num(r.target));
    }
    this.setTargetVertex(prevH, r.target);
    this.he.get(prevH).next = nextH;
    this.he.get(nextH).prev = prevH;
    this.faceDeg.set(r.face, this.faceDeg.get(r.face) - 1);
    this._removeHalfEdgePair(h);
    this._removeVertex(src);
  }

  _deleteInteriorVertex(h) {
    const r = this.he.get(h);
    const src = r.source;
    const nextH = r.next;
    const prevH = r.prev;
    const twinH = r.twin;
    const nextTwin = this.he.get(twinH).next;
    const prevTwin = this.he.get(twinH).prev;

    this.setTargetVertex(prevH, r.target);
    this.he.get(prevH).next = nextH;
    this.he.get(nextH).prev = prevH;
    this.he.get(prevTwin).next = nextTwin;
    this.he.get(nextTwin).prev = prevTwin;

    this.faceDeg.set(r.face, this.faceDeg.get(r.face) - 1);
    const tf = this.he.get(twinH).face;
    this.faceDeg.set(tf, this.faceDeg.get(tf) - 1);

    this._removeHalfEdgePair(h);
    this._removeVertex(src);
  }

  isValidDeleteHalfEdge(h) {
    if (!this.isHalfEdge(h)) return false;
    if (this.onBoundary(h)) return false;
    const r = this.he.get(h);
    if (this.vertexDeg.get(r.source) <= 2 || this.vertexDeg.get(r.target) <= 2) return false;
    const tf = this.he.get(r.twin).face;
    if (r.face === tf) return false;
    return true;
  }

  deleteHalfEdge(h) {
    if (!this.isValidDeleteHalfEdge(h)) throw new GameError("This edge cannot be deleted");
    const r = this.he.get(h);
    const nextH = r.next;
    const prevH = r.prev;
    const twinH = r.twin;
    const prevTwin = this.he.get(twinH).prev;
    const nextTwin = this.he.get(twinH).next;

    this.vertexDeg.set(r.source, this.vertexDeg.get(r.source) - 1);
    this.vertexDeg.set(r.target, this.vertexDeg.get(r.target) - 1);

    const curFace = r.face;
    const twinFace = this.he.get(twinH).face;
    this.faceDeg.set(curFace, this.faceDeg.get(curFace) + this.faceDeg.get(twinFace) - 2);

    for (const he of Array.from(this.faceHE.get(twinFace))) {
      this.he.get(he).face = curFace;
      this.faceHE.get(curFace).add(he);
    }
    this.nodeType.delete(twinFace);
    this.faceHE.delete(twinFace);
    this.faceDeg.delete(twinFace);

    this.he.get(prevH).next = nextTwin;
    this.he.get(nextTwin).prev = prevH;
    this.he.get(prevTwin).next = nextH;
    this.he.get(nextH).prev = prevTwin;
    this._removeHalfEdgePair(h);
  }

  // --- geometry -----------------------------------------------------------

  halfEdgeAngles() {
    const out = new Map();
    for (const [id, t] of this.nodeType) {
      if (t !== "half_edge") continue;
      const r = this.he.get(id);
      const c = this.coords.get(num(r.source));
      const n = this.coords.get(num(r.target));
      const p = this.coords.get(num(this.he.get(r.prev).source));
      const v1 = [n[0] - c[0], n[1] - c[1]];
      const v2 = [p[0] - c[0], p[1] - c[1]];
      const dot = v1[0] * v2[0] + v1[1] * v2[1];
      const det = v1[0] * v2[1] - v1[1] * v2[0];
      let ang = (Math.atan2(det, dot) * 180) / Math.PI;
      if (ang < 0) ang += 360;
      out.set(id, ang);
    }
    return out;
  }

  // Simultaneous untangling and smoothing (game/js/untangle.js), the same
  // optimiser the environment's evaluator runs. The Laplacian below cannot open
  // a folded element -- it sits at its own fixed point -- so a board that is
  // topologically perfect can stay unwinnable for a reason the player cannot
  // act on. This is what the win test should be measuring.
  untangle(iters = 8) {
    const U = (typeof require !== "undefined" && typeof module !== "undefined")
      ? require("./untangle.js")
      : (typeof MeshQuestUntangle !== "undefined" ? MeshQuestUntangle : null);
    if (!U) { this.smooth(iters); return; }
    const verts = this.vertexList().map(num);
    const idx = new Map(verts.map((v, i) => [v, i]));
    const coords = verts.map((v) => {
      const c = this.coords.get(v);
      return [c[0], c[1]];
    });
    const elements = [];
    for (const f of this.faceList()) {
      elements.push(this.faceLoop(this.firstFaceHalfEdge(f))
        .map((h) => idx.get(num(this.he.get(h).source))));
    }
    // pinned: the domain's real corners. An unpinned BOUNDARY vertex may only
    // travel along the outline, so it is confined to the segment between its
    // two boundary neighbours rather than set free into the interior.
    const fixed = [], slide = new Map();
    for (const v of verts) {
      if (this.userVerts.has(v)) { fixed.push(idx.get(v)); continue; }
      if (!this.boundaryVerts.has(v)) continue;
      const nbrs = [];
      for (const [id, t] of this.nodeType) {
        if (t !== "half_edge" || !this.onBoundary(id)) continue;
        const r = this.he.get(id);
        if (num(r.source) === v) nbrs.push(idx.get(num(r.target)));
        else if (num(r.target) === v) nbrs.push(idx.get(num(r.source)));
      }
      if (nbrs.length >= 2) slide.set(idx.get(v), [coords[nbrs[0]], coords[nbrs[1]]]);
      else fixed.push(idx.get(v));
    }
    const out = U.optimize(elements, coords, fixed, { iters, slide });
    const before = this.minCornerQuality();
    const restore = verts.map((v) => { const c = this.coords.get(v); return [c[0], c[1]]; });
    verts.forEach((v, i) => this.coords.set(v, [out[i][0], out[i][1]]));
    // The slide constraint above confines a boundary vertex to the CHORD
    // between its neighbours, which on a curved level pulls it off the outline
    // and into the domain. Project it back, exactly as smooth() does.
    this.snapToArcs();
    // A local optimiser started inside a tangle can come out worse than it went
    // in, and on a degenerate board it does. geo2d's `resmooth_mesh` keeps its
    // own result only when it improved; without the same guard, pressing Smooth
    // can make the board worse, which is never what the player asked for.
    if (this.minCornerQuality() < before) {
      verts.forEach((v, i) => this.coords.set(v, restore[i]));
    }
  }

  // The worst corner on the board, on the same angle-based scale the win test
  // uses. `Game.serialize` reports this number; `untangle` needs it to decide
  // whether to keep what the optimiser handed back.
  minCornerQuality() {
    let worst = 1;
    for (const [, a] of this.halfEdgeAngles()) {
      worst = Math.min(worst, Math.sin((a * Math.PI) / 180) / idealSin());
    }
    return worst;
  }

  smooth(numIter) {
    // mirrors Tiler.smooth_vertices: fixed 0/1 neighbor structure, then
    // numIter rounds of averaging with per-vertex divisors
    const verts = this.vertexList().map(num);
    const idx = new Map(verts.map((v, i) => [v, i]));
    const nbrs = verts.map(() => []);

    for (const [id, t] of this.nodeType) {
      if (t !== "half_edge") continue;
      const r = this.he.get(id);
      const s = num(r.source);
      const d = num(r.target);
      if (this.onBoundary(id)) {
        if (!this.userVerts.has(s)) nbrs[idx.get(s)].push(idx.get(d));
        if (!this.userVerts.has(d)) nbrs[idx.get(d)].push(idx.get(s));
      } else if (!this.boundaryVerts.has(s) && !this.userVerts.has(s)) {
        nbrs[idx.get(s)].push(idx.get(d));
      }
    }

    const div = verts.map((v) => {
      if (this.userVerts.has(v)) return 1;
      if (this.boundaryVerts.has(v)) return 2;
      return this.vertexDeg.get("v" + v);
    });

    let cur = verts.map((v) => {
      const c = this.coords.get(v);
      return [c[0], c[1]];
    });
    for (let it = 0; it < numIter; it++) {
      const next = verts.map((v, i) => {
        if (this.userVerts.has(v)) return [cur[i][0], cur[i][1]];
        let x = 0, y = 0;
        for (const j of nbrs[i]) { x += cur[j][0]; y += cur[j][1]; }
        return [x / div[i], y / div[i]];
      });
      cur = next;
    }
    verts.forEach((v, i) => this.coords.set(v, cur[i]));
    this.snapToArcs();
  }
}

// --- shapes -----------------------------------------------------------------

function regularPolygon(n, phase = 0) {
  const pts = [];
  for (let k = 0; k < n; k++) {
    const a = phase + (2 * Math.PI * k) / n;
    pts.push([Math.cos(a), Math.sin(a)]);
  }
  return pts;
}

function starPolygon() {
  const inner = Math.cos((72 * Math.PI) / 180) / Math.cos((36 * Math.PI) / 180);
  const pts = [];
  for (let k = 0; k < 5; k++) {
    const tip = ((90 + 72 * k) * Math.PI) / 180;
    const notch = ((126 + 72 * k) * Math.PI) / 180;
    pts.push([Math.cos(tip), Math.sin(tip)]);
    pts.push([inner * Math.cos(notch), inner * Math.sin(notch)]);
  }
  return pts;
}

function semicirclePolygon() {
  const pts = [[-1, 0], [1, 0]];
  for (let k = 1; k <= 5; k++) {
    const t = (30 * k * Math.PI) / 180;
    pts.push([Math.cos(t), Math.sin(t)]);
  }
  return pts;
}

function pacmanPolygon() {
  const pts = [[0, 0]];
  for (let k = 0; k < 7; k++) {
    const t = ((45 + 45 * k) * Math.PI) / 180;
    pts.push([Math.cos(t), Math.sin(t)]);
  }
  return pts;
}

function gearPolygon(teeth = 6, outer = 1.0, root = 0.62) {
  const pts = [];
  const w = (2 * Math.PI) / teeth;
  for (let k = 0; k < teeth; k++) {
    const base = w * k;
    for (const [frac, rad] of [[0.05, outer], [0.45, outer], [0.55, root], [0.95, root]]) {
      const t = base + frac * w;
      pts.push([rad * Math.cos(t), rad * Math.sin(t)]);
    }
  }
  return pts;
}

// Both rings put the hole-to-rim slit between two mid-edge points, which want
// degree 3 and are already satisfied by the slit: it belongs to a perfect mesh
// and never has to be deleted.
function triforceRing() {
  const coords = {};
  for (let k = 0; k < 3; k++) {
    const t = ((90 + 120 * k) * Math.PI) / 180;
    coords[k] = [Math.cos(t), Math.sin(t)];
    coords[k + 3] = [0.4 * Math.cos(t), 0.4 * Math.sin(t)];
  }
  coords[6] = [(coords[0][0] + coords[1][0]) / 2, (coords[0][1] + coords[1][1]) / 2];
  coords[7] = [(coords[3][0] + coords[4][0]) / 2, (coords[3][1] + coords[4][1]) / 2];
  const loop = [0, 6, 7, 3, 5, 4, 7, 6, 1, 2];
  return { loops: [loop], coords, desired: desiredFromLoop(loop, coords),
           userVertices: pinnedVertices(loop, coords) };
}

function interiorAngles(loop, coords) {
  const n = loop.length;
  const out = {};
  for (let i = 0; i < n; i++) {
    const c = coords[loop[i]];
    const p = coords[loop[(i - 1 + n) % n]];
    const nx = coords[loop[(i + 1) % n]];
    const v1 = [nx[0] - c[0], nx[1] - c[1]];
    const v2 = [p[0] - c[0], p[1] - c[1]];
    const dot = v1[0] * v2[0] + v1[1] * v2[1];
    const det = v1[0] * v2[1] - v1[1] * v2[0];
    let ang = (Math.atan2(det, dot) * 180) / Math.PI;
    if (ang < 0) ang += 360;
    out[loop[i]] = ang;
  }
  return out;
}

function summedInteriorAngles(loop, coords) {
  const n = loop.length;
  const out = {};
  for (let i = 0; i < n; i++) {
    const c = coords[loop[i]];
    const p = coords[loop[(i - 1 + n) % n]];
    const nx = coords[loop[(i + 1) % n]];
    const v1 = [nx[0] - c[0], nx[1] - c[1]];
    const v2 = [p[0] - c[0], p[1] - c[1]];
    const dot = v1[0] * v2[0] + v1[1] * v2[1];
    const det = v1[0] * v2[1] - v1[1] * v2[0];
    let ang = (Math.atan2(det, dot) * 180) / Math.PI;
    if (ang < 0) ang += 360;
    out[loop[i]] = (out[loop[i]] || 0) + ang;
  }
  return out;
}

// A slit outline visits its two endpoints twice, so the domain's total angle
// there is the sum of both corners. Deriving the desired degrees this way
// instead of hardcoding them is what lets these shapes be re-targeted; it
// reproduces the hand-written numbers exactly at 90 degrees.
function slitSupportVertices(loop, coords) {
  // A hole is cut open along a zero-width slit, so the loop visits each anchor
  // twice. An anchor on a real corner shapes the domain and stays pinned; one
  // dropped mid-edge is bookkeeping, sums to a straight 180 degrees, and
  // pinning it stops the smoother ever opening the slit.
  const counts = new Map();
  for (const v of loop) counts.set(v, (counts.get(v) || 0) + 1);
  const angles = summedInteriorAngles(loop, coords);
  const support = new Set();
  for (const [v, n] of counts) {
    if (n < 2) continue;
    if (Math.abs(angles[v] - 180) < 1e-6) support.add(v);
  }
  return support;
}

function pinnedVertices(loop, coords) {
  const support = slitSupportVertices(loop, coords);
  return new Set(loop.filter((v) => !support.has(v)));
}

function desiredFromLoop(loop, coords) {
  const angles = summedInteriorAngles(loop, coords);
  const desired = {};
  for (const v of Object.keys(angles)) desired[v] = desiredFromAngle(angles[v]);
  return desired;
}

function simplePolygon(pts) {
  return () => {
    const coords = {};
    pts.forEach((p, i) => (coords[i] = [p[0], p[1]]));
    const loop = pts.map((_, i) => i);
    const angles = interiorAngles(loop, coords);
    const desired = {};
    for (const v of loop) desired[v] = desiredFromAngle(angles[v]);
    return { loops: [loop], coords, desired };
  };
}

function squareHole() {
  const coords = {
    0: [0, 0], 1: [0.5, 0], 2: [0.5, 0.25], 3: [0.25, 0.25], 4: [0.25, 0.75],
    5: [0.75, 0.75], 6: [0.75, 0.25], 7: [1, 0], 8: [1, 1], 9: [0, 1],
  };
  const loop = [0, 1, 2, 3, 4, 5, 6, 2, 1, 7, 8, 9];
  return { loops: [loop], coords, desired: desiredFromLoop(loop, coords),
           userVertices: pinnedVertices(loop, coords) };
}

function diamondBracket() {
  // straight-holes #18 of the geogen held-out set: a slotted plate with a diamond
  // hole, par 0. The slit drops from the hole's bottom corner (270 degrees, wants
  // four) straight down to a flat point on the base edge (180 degrees, wants three).
  const coords = {
    0: [0, 0], 1: [8, 0], 2: [8, 4], 3: [7, 4], 4: [7, 1], 5: [6, 1],
    6: [6, 4], 7: [2, 4], 8: [2, 1], 9: [1, 1], 10: [1, 4], 11: [0, 4],
    12: [5, 2], 13: [4, 3], 14: [3, 2], 15: [4, 1], 16: [4, 0],
  };
  const loop = [0, 16, 15, 14, 13, 12, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
  return { loops: [loop], coords, desired: desiredFromLoop(loop, coords),
           userVertices: pinnedVertices(loop, coords) };
}

function diamondInSquare() {
  // The primitive the diamond levels share: topologically this IS `Square hole`
  // -- four convex outer corners wanting two, four re-entrant hole corners
  // wanting four, par 0 -- with the hole turned forty-five degrees, so every
  // edge between hole and plate runs at a slant. Scoring is topological, so any
  // extra difficulty here is geometry, which makes it a control on `Square hole`.
  const coords = {
    0: [0, 0], 1: [4, 0], 2: [4, 2], 3: [2, 4], 4: [4, 6], 5: [6, 4],
    6: [8, 0], 7: [8, 8], 8: [0, 8],
  };
  const loop = [0, 1, 2, 3, 4, 5, 2, 1, 6, 7, 8];
  return { loops: [loop], coords, desired: desiredFromLoop(loop, coords),
           userVertices: pinnedVertices(loop, coords) };
}

function diamondTab() {
  // straight-holes #5 of the geogen held-out set, par 0. The slit is the SHORTEST
  // perpendicular cut: one unit right from the hole's right corner to the wall.
  // The thin tab on the left sits at the hole's height, so a longer cut that way
  // may suit the domain better -- shortest and best are not the same thing.
  const coords = {
    0: [0, 2], 1: [2, 2], 2: [2, 0], 3: [7, 0], 4: [7, 5], 5: [2, 5],
    6: [2, 3], 7: [0, 3], 8: [6, 3], 9: [5, 4], 10: [4, 3], 11: [5, 2],
    12: [7, 3],
  };
  const loop = [0, 1, 2, 3, 12, 8, 11, 10, 9, 8, 12, 4, 5, 6, 7];
  return { loops: [loop], coords, desired: desiredFromLoop(loop, coords),
           userVertices: pinnedVertices(loop, coords) };
}

// --- curved levels ----------------------------------------------------------
// Carried natively: one vertex per control point, arcs recorded alongside. The
// corner want comes from the TANGENT, so the apex of a semicircle reads 180 and
// wants three exactly like a flat point, while each end of the diameter reads
// 90 and wants two.

function arcsFrom(list) {
  const A = (typeof require !== "undefined" && typeof module !== "undefined")
    ? require("./arcs.js")
    : globalThis.MeshQuestArcs;
  const out = new A.BoundaryArcs();
  for (const [a, b, centre, radius, start, end] of list) {
    out.add(a, b, new A.Arc(centre, radius, start, end, true), a);
  }
  return out;
}

// The direction the material leaves `u` along the edge to `w`. On a straight
// edge that is the chord; on an arc it is the TANGENT, which is the whole point
// -- the apex of a semicircle is a smooth point and wants three, while the
// chord would read it as a sharp corner. Mirrors `_tangent_angles` in
// src/curved_levels.py, and is used only for what a corner WANTS. Live angles
// and the quality metric stay on chords, as Tiler.half_edge_angles does.
function edgeDirection(arcs, at, u, w) {
  const arc = arcs && arcs.get(u, w);
  if (!arc) return [at(w)[0] - at(u)[0], at(w)[1] - at(u)[1]];
  // exactly the tangent: on a circle it is the radius turned a quarter turn,
  // in the direction the arc is travelled. Mirrors src/boundary_arcs.py.
  const head = arc.point(0);
  const atStart = Math.hypot(head[0] - at(u)[0], head[1] - at(u)[1]) < 1e-9;
  const p = arc.point(atStart ? 0 : 1);
  const rx = p[0] - arc.centre[0], ry = p[1] - arc.centre[1];
  const turn = arc.sweep() < 0 ? -1 : 1;
  const sign = (atStart ? 1 : -1) * turn;
  return [-ry * sign, rx * sign];
}

function curvedDesired(loop, coords, arcs) {
  const n = loop.length;
  const at = (k) => coords[k];
  // SUM over positions, never overwrite: a hole is carried as a slit, so the
  // loop revisits each slit endpoint and the material angle there is the two
  // visits added together. Matches `corner_angles` in src/boundary_arcs.py.
  const total = {};
  for (let i = 0; i < n; i++) {
    const v = loop[i], prev = loop[(i - 1 + n) % n], next = loop[(i + 1) % n];
    total[v] = (total[v] || 0) + angleBetweenVectors(
      edgeDirection(arcs, at, v, next), edgeDirection(arcs, at, v, prev));
  }
  const desired = {};
  for (const v of Object.keys(total)) desired[v] = desiredFromAngle(total[v]);
  return desired;
}

function angleBetweenVectors(a, b) {
  let d = Math.atan2(b[1], b[0]) - Math.atan2(a[1], a[0]);
  while (d < 0) d += 2 * Math.PI;
  return (d * 180) / Math.PI;
}

// The frozen geo2d domains (src/curved_domains.json). These are the curved
// levels that are actually HARD: realistic many-sided outlines the agent fails
// on, where Semicircle and Stadium fall to one or two moves.
function curvedDomains() {
  return (typeof require !== "undefined" && typeof module !== "undefined")
    ? require("./curved_domains.js")
    : globalThis.MeshQuestCurvedDomains;
}

function frozenCurved(name) {
  return function () {
    const spec = curvedDomains()[name];
    const coords = {};
    spec.points.forEach((p, i) => { coords[i] = [p[0], p[1]]; });
    // a domain with a hole is cut open by a slit, so its loop revisits the two
    // slit endpoints rather than being range(n)
    const loop = spec.loop ? spec.loop.slice() : spec.points.map((_, i) => i);
    const A = (typeof require !== "undefined" && typeof module !== "undefined")
      ? require("./arcs.js") : globalThis.MeshQuestArcs;
    const arcs = new A.BoundaryArcs();
    for (const e of spec.arcs) {
      arcs.add(e.edge[0], e.edge[1],
               new A.Arc(e.centre, e.radius, e.from, e.to, e.ccw), e.start);
    }
    return { loops: [loop], coords, desired: curvedDesired(loop, coords, arcs), arcs };
  };
}

function semicircleCurved() {
  const coords = { 0: [-1, 0], 1: [1, 0], 2: [0, 1] };
  const loop = [0, 1, 2];
  const arcs = arcsFrom([[1, 2, [0, 0], 1, 0, Math.PI / 2],
                         [2, 0, [0, 0], 1, Math.PI / 2, Math.PI]]);
  return { loops: [loop], coords, desired: curvedDesired(loop, coords, arcs), arcs };
}

function stadiumCurved() {
  const coords = { 0: [-1, -1], 1: [1, -1], 2: [2, 0], 3: [1, 1], 4: [-1, 1], 5: [-2, 0] };
  const loop = [0, 1, 2, 3, 4, 5];
  const arcs = arcsFrom([
    [1, 2, [1, 0], 1, -Math.PI / 2, 0],
    [2, 3, [1, 0], 1, 0, Math.PI / 2],
    [4, 5, [-1, 0], 1, Math.PI / 2, Math.PI],
    [5, 0, [-1, 0], 1, Math.PI, (3 * Math.PI) / 2],
  ]);
  return { loops: [loop], coords, desired: curvedDesired(loop, coords, arcs), arcs };
}

function quarterDiscCurved() {
  const coords = { 0: [0, 0], 1: [1, 0], 2: [0, 1] };
  const loop = [0, 1, 2];
  const arcs = arcsFrom([[1, 2, [0, 0], 1, 0, Math.PI / 2]]);
  return { loops: [loop], coords, desired: curvedDesired(loop, coords, arcs), arcs };
}

const SHAPES = {
  "L-shape": simplePolygon([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]),
  "T-bracket": simplePolygon([[1, 0], [2, 0], [2, 2], [3, 2], [3, 3], [0, 3], [0, 2], [1, 2]]),
  "I-bracket": simplePolygon([[0, 0], [3, 0], [3, 1], [2, 1], [2, 2], [3, 2], [3, 3], [0, 3], [0, 2], [1, 2], [1, 1], [0, 1]]),
  "U-channel": simplePolygon([[0, 0], [3, 0], [3, 2], [2, 2], [2, 1], [1, 1], [1, 2], [0, 2]]),
  "Z-shape": simplePolygon([[0, 0], [2, 0], [2, 1], [3, 1], [3, 2], [1, 2], [1, 1], [0, 1]]),
  "Plus": simplePolygon([[1, 0], [2, 0], [2, 1], [3, 1], [3, 2], [2, 2], [2, 3], [1, 3], [1, 2], [0, 2], [0, 1], [1, 1]]),
  "Staircase": simplePolygon([[0, 0], [3, 0], [3, 1], [2, 1], [2, 2], [1, 2], [1, 3], [0, 3]]),
  "Triangle": simplePolygon([[0, 0], [1, 0], [0.5, Math.sin(Math.PI / 3)]]),
  "Pentagon": simplePolygon(regularPolygon(5, Math.PI / 2)),
  "Semicircle": simplePolygon(semicirclePolygon()),
  "Pac-Man": simplePolygon(pacmanPolygon()),
  "Star": simplePolygon(starPolygon()),
  "Square hole": squareHole,
  "Triforce ring": triforceRing,
  "Gear": simplePolygon(gearPolygon()),
  "Diamond bracket": diamondBracket,
  "Diamond tab": diamondTab,
  "Diamond in square": diamondInSquare,
  "Plate with a hole": frozenCurved("Plate with a hole"),
  "Notched plate": frozenCurved("Notched plate"),
  "Snail": frozenCurved("Snail"),
  "Tunnel": frozenCurved("Tunnel"),
  "Fillet plate": frozenCurved("Fillet plate"),
  "Hook": frozenCurved("Hook"),
  "Bridge": frozenCurved("Bridge"),
  "Clover": frozenCurved("Clover"),
  "Boot": frozenCurved("Boot"),
  "Skillet": frozenCurved("Skillet"),
};

// --- game -------------------------------------------------------------------

class Game {
  constructor(shape = "L-shape", storage = null) {
    this.storage = storage; // {get(shape), set(shape, moves)} or null
    this.reset(shape);
  }

  reset(shape) {
    shape = shape || this.shape;
    if (!SHAPES[shape]) throw new GameError("Unknown shape: " + shape);
    const shape_ = SHAPES[shape]();
    const { loops, coords, desired } = shape_;
    this.shape = shape;
    this.mesh = Mesh.fromFaceLoops(loops, coords, shape_.userVertices || null,
                                   shape_.arcs || null);
    this.initialDesired = {};
    for (const k of Object.keys(desired)) this.initialDesired[k] = desired[k];
    this.desiredSets = this._computeDesiredSets();
    this.undoStack = [];
    this.moves = 0;
    this.recordJustSet = false;
    this.par = this._computePar();
  }

  desiredDegree(vid) {
    if (vid in this.initialDesired) return this.initialDesired[vid];
    return this.mesh.boundaryVerts.has(vid) ? BOUNDARY_DESIRED : INTERIOR_DESIRED;
  }

  // The start state is the raw outline: one face, so a half-edge's angle is its
  // source corner's interior angle, summed where a slit revisits it.
  _computeDesiredSets() {
    const m = this.mesh;
    const at = (v) => m.coords.get(v);   // m.coords is keyed by the bare id
    const total = new Map();
    for (const h of m.halfEdgeList()) {
      const v = num(m.he.get(h).source);
      const prev = num(m.he.get(m.he.get(h).prev).source);
      const next = num(m.he.get(h).target);
      const a = angleBetweenVectors(edgeDirection(m.arcs, at, v, next),
                                    edgeDirection(m.arcs, at, v, prev));
      total.set(v, (total.get(v) || 0) + a);
    }
    const sets = {};
    for (const [v, angle] of total) {
      if (v in this.initialDesired) sets[v] = desiredOptions(angle);
    }
    return sets;
  }

  desiredDegrees(vid) {
    return this.desiredSets[vid] || [this.desiredDegree(vid)];
  }

  vertexDefect(vid) {
    const degree = this.mesh.vertexDeg.get("v" + vid);
    return Math.min(...this.desiredDegrees(vid).map(want => Math.abs(degree - want)));
  }

  _computePar() {
    const m = this.mesh;
    // the two options at a tie always differ by one, so taking `t` of them high
    // shifts the excess by exactly `t`: par is a scan over `t`
    let cornerExcess = 0;
    let numTies = 0;
    for (const v of m.vertexList()) {
      const vid = num(v);
      const generic = m.boundaryVerts.has(vid) ? BOUNDARY_DESIRED : INTERIOR_DESIRED;
      const options = this.desiredDegrees(vid);
      cornerExcess += Math.min(...options) - generic;
      numTies += options.length - 1;
    }
    let numEdges = 0;
    const seen = new Set();
    for (const h of m.halfEdgeList()) {
      if (seen.has(h)) continue;
      seen.add(h);
      seen.add(m.he.get(h).twin);
      numEdges += 1;
    }
    const chi = m.vertexList().length - numEdges + m.faceList().length;
    // For a mesh of d-gons, sum_v (generic - degree) = 2*d*chi/(d-2), which is
    // the interior vertex's generic degree: 4 for quads, 6 for triangles.
    const base = cornerExcess + INTERIOR_DESIRED * chi;
    let par = Math.abs(base);
    for (let taken = 1; taken <= numTies; taken++) {
      par = Math.min(par, Math.abs(base + taken));
    }
    return par;
  }

  resolveChord(a, b) {
    const m = this.mesh;
    for (const f of m.faceList()) {
      const loop = m.faceLoop(m.firstFaceHalfEdge(f));
      const srcs = loop.map((h) => num(m.he.get(h).source));
      const n = loop.length;
      for (let i = 0; i < n; i++) {
        if (srcs[i] !== a) continue;
        for (let k = 0; k < n - 1; k++) {
          if (srcs[(i + k + 1) % n] === b && m.isValidEdgeInsert(loop[i], k)) {
            return [loop[i], k];
          }
        }
      }
    }
    return null;
  }

  deleteCandidate(vid) {
    const m = this.mesh;
    const v = "v" + vid;
    for (const [id, t] of m.nodeType) {
      if (t !== "half_edge") continue;
      if (m.he.get(id).source === v && m.isValidDeleteSourceVertex(id)) return id;
    }
    return null;
  }

  applyOp(op, params, smoothAfter = false) {
    const snapshot = this.mesh.clone();
    try {
      if (op === "insert_vertex") {
        const h = "h" + params.edge;
        if (!this.mesh.isHalfEdge(h)) throw new GameError("Unknown edge");
        this.mesh.insertVertex(h);
      } else if (op === "delete_edge") {
        const h = "h" + params.edge;
        this.mesh.deleteHalfEdge(h);
      } else if (op === "delete_vertex") {
        const h = this.deleteCandidate(params.vertex);
        if (h === null) {
          throw new GameError("Vertex not deletable: must be degree 2 and not an original corner");
        }
        this.mesh.deleteSourceVertex(h);
      } else if (op === "insert_edge") {
        const chord = this.resolveChord(params.a, params.b) || this.resolveChord(params.b, params.a);
        if (chord === null) throw new GameError("Vertices must lie on the same face");
        this.mesh.insertHalfEdge(chord[0], chord[1]);
      } else {
        throw new GameError("Unknown operation: " + op);
      }
      if (smoothAfter) this.mesh.smooth(2);
    } catch (e) {
      this.mesh = snapshot;
      throw e;
    }
    this.undoStack.push(snapshot);
    this.moves += 1;
    this.recordJustSet = false;
    this._checkRecord();
  }

  smooth(numIter = 3) {
    this.undoStack.push(this.mesh.clone());
    // the untangler, with the Laplacian as its own warm start: from the raw
    // midpoint layout a local optimiser can begin inside a tangle it cannot
    // escape, which is the same warm start the evaluator uses
    this.mesh.smooth(numIter);
    this.mesh.untangle(8);
    this.recordJustSet = false;
    this._checkRecord();
  }

  undo() {
    if (!this.undoStack.length) throw new GameError("Nothing to undo");
    this.mesh = this.undoStack.pop();
    this.moves = Math.max(0, this.moves - 1);
    this.recordJustSet = false;
  }

  // Every board this game has passed through, oldest first. The undo stack
  // already holds a clone from before each move, so it doubles as a replay —
  // and because undo pops it, this is the path the player actually kept.
  history() {
    const live = this.mesh;
    const frames = [];
    try {
      for (const snapshot of this.undoStack) {
        this.mesh = snapshot;
        frames.push({ ...this.serialize(), moves: frames.length });
      }
    } finally {
      this.mesh = live;
    }
    frames.push({ ...this.serialize(), moves: frames.length });
    return frames;
  }

  _currentScores() {
    const m = this.mesh;
    let vertexScore = 0;
    for (const v of m.vertexList()) vertexScore += this.vertexDefect(num(v));
    let faceScore = 0;
    for (const f of m.faceList()) faceScore += Math.abs(m.faceDeg.get(f) - FACE_DESIRED);
    let minQ = 1;
    for (const a of m.halfEdgeAngles().values()) {
      minQ = Math.min(minQ, Math.sin((a * Math.PI) / 180) / idealSin());
    }
    return [vertexScore, faceScore, minQ];
  }

  _checkRecord() {
    const [vs, fs, mq] = this._currentScores();
    const won = vs === this.par && fs === 0 && mq >= QUALITY_THRESHOLD;
    if (!won || !this.storage) return;
    const best = this.storage.get(this.shape);
    if (best == null || this.moves < best) {
      this.storage.set(this.shape, this.moves);
      this.recordJustSet = true;
    }
  }

  serialize() {
    const m = this.mesh;
    const angles = m.halfEdgeAngles();
    const cornerQ = new Map();
    for (const [h, a] of angles) cornerQ.set(h, Math.sin((a * Math.PI) / 180) / idealSin());
    const vertexQ = new Map();
    const faceQ = new Map();
    let minQuality = 1;
    for (const [h, q] of cornerQ) {
      const v = num(m.he.get(h).source);
      const f = num(m.he.get(h).face);
      vertexQ.set(v, Math.min(q, vertexQ.has(v) ? vertexQ.get(v) : 1));
      faceQ.set(f, Math.min(q, faceQ.has(f) ? faceQ.get(f) : 1));
      minQuality = Math.min(minQuality, q);
    }

    const vertices = m.vertexList().map((v) => {
      const vid = num(v);
      const c = m.coords.get(vid);
      return {
        id: vid, x: c[0], y: c[1],
        degree: m.vertexDeg.get(v),
        desired: this.desiredDegree(vid),
        desiredOptions: this.desiredDegrees(vid),
        boundary: m.boundaryVerts.has(vid),
        user: m.userVerts.has(vid),
        deletable: this.deleteCandidate(vid) !== null,
        quality: vertexQ.has(vid) ? vertexQ.get(vid) : 1,
      };
    });

    const edges = [];
    const seen = new Set();
    for (const h of m.halfEdgeList()) {
      if (seen.has(h)) continue;
      seen.add(h);
      seen.add(m.he.get(h).twin);
      const r = m.he.get(h);
      // `bow` is the arc's midpoint, present only on a curved boundary edge.
      // The UI draws every edge as a quadratic bezier already, so one point is
      // all it needs to follow the real boundary rather than the chord -- and
      // it keeps arc geometry out of the renderer entirely.
      const arc = m.arcs && m.arcs.get(num(r.source), num(r.target));
      edges.push({
        id: num(h), v1: num(r.source), v2: num(r.target),
        boundary: m.onBoundary(h),
        deletable: m.isValidDeleteHalfEdge(h),
        bow: arc ? arc.point(0.5) : null,
      });
    }

    const faces = [];
    const insertablePairs = [];
    for (const f of m.faceList()) {
      const loop = m.faceLoop(m.firstFaceHalfEdge(f));
      const srcs = loop.map((h) => num(m.he.get(h).source));
      faces.push({
        id: num(f), degree: m.faceDeg.get(f), vertices: srcs,
        quality: faceQ.has(num(f)) ? faceQ.get(num(f)) : 1,
      });
      const n = loop.length;
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < n - 1; k++) {
          const b = srcs[(i + k + 1) % n];
          if (srcs[i] !== b && m.isValidEdgeInsert(loop[i], k)) {
            insertablePairs.push([srcs[i], b]);
          }
        }
      }
    }

    let vertexScore = 0;
    let defectSum = 0;
    for (const v of vertices) {
      vertexScore += Math.min(...v.desiredOptions.map(w => Math.abs(v.degree - w)));
      defectSum += v.desired - v.degree;
    }
    let faceScore = 0;
    for (const f of faces) faceScore += Math.abs(f.degree - FACE_DESIRED);
    let angleScore = 0;
    for (const a of angles.values()) angleScore += Math.abs(a - TARGET_ANGLE) / TARGET_ANGLE;

    return {
      shape: this.shape,
      shapes: Object.keys(SHAPES),
      vertices, edges, faces,
      insertable_pairs: insertablePairs,
      scores: {
        vertex: vertexScore, face: faceScore,
        angle: angleScore, defect_sum: defectSum, min_quality: minQuality,
      },
      par: this.par,
      quality_threshold: QUALITY_THRESHOLD,
      best: this.storage ? this.storage.get(this.shape) ?? null : null,
      new_record: this.recordJustSet,
      moves: this.moves,
      can_undo: this.undoStack.length > 0,
    };
  }
}

if (typeof module !== "undefined") {
  module.exports = {
    Game, Mesh, SHAPES, GameError, QUALITY_THRESHOLD, setElementTarget,
    elementTarget, desiredOptions,
  };
}
