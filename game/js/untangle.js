// Simultaneous untangling and smoothing, ported from geo2d's `smooth.optimize`
// (Escobar et al.): minimise, over one free node at a time,
//
//     sum_k (|a_k|^2 + |b_k|^2) * sin(theta_n) / (2 h(J_k))
//
// over the corners of the elements touching it, where a = next - node,
// b = prev - node, J = a x b, and h(J) = (J + sqrt(J^2 + delta^2)) / 2
// regularises the corner Jacobian so an INVERTED corner is pushed back out
// instead of blowing the objective up. Newton steps with finite-difference
// derivatives and backtracking, swept Gauss-Seidel.
//
// This is what the environment's evaluator runs, so the game agrees with the
// agent's acceptance test rather than with a Laplacian, which cannot open a
// folded element at all: it sits at its own fixed point and stays there.
//
// The port is deliberately scalar -- a Mesh Quest board is tens of elements,
// not thousands -- and takes plain arrays so it has no dependency on the DCEL.

"use strict";

function cornerTerms(P) {
  // P: [prev, node, next] as [x, y] pairs -> { J, L2 }
  const ax = P[2][0] - P[1][0], ay = P[2][1] - P[1][1];
  const bx = P[0][0] - P[1][0], by = P[0][1] - P[1][1];
  return { J: ax * by - ay * bx, L2: ax * ax + ay * ay + bx * bx + by * by };
}

function objective(corners, x, delta) {
  // Each corner is a (prev, node, next) triple of POSITIONS plus slots saying
  // which of the three is the node being moved. The node can appear in a corner
  // as any of the three, which is why the whole adjacent element contributes:
  // sliding a vertex changes the corners at its neighbours too.
  let total = 0;
  for (const c of corners) {
    const p = c.slots[0] ? x : c.P[0];
    const q = c.slots[1] ? x : c.P[1];
    const r = c.slots[2] ? x : c.P[2];
    const { J, L2 } = cornerTerms([p, q, r]);
    const h = 0.5 * (J + Math.sqrt(J * J + delta * delta));
    if (!(h > 0)) return Infinity;
    total += (L2 * c.sin) / (2 * h);
  }
  return total;
}

function cornerJacobians(corners, x) {
  let jmin = Infinity, jabs = 0;
  for (const c of corners) {
    const p = c.slots[0] ? x : c.P[0];
    const q = c.slots[1] ? x : c.P[1];
    const r = c.slots[2] ? x : c.P[2];
    const { J } = cornerTerms([p, q, r]);
    if (J < jmin) jmin = J;
    jabs += Math.abs(J);
  }
  return { jmin, jmean: jabs / Math.max(corners.length, 1) };
}

function optimizeNode(corners, x0, newtonIters) {
  let meanL2 = 0;
  for (const c of corners) {
    const p = c.slots[0] ? x0 : c.P[0];
    const q = c.slots[1] ? x0 : c.P[1];
    const r = c.slots[2] ? x0 : c.P[2];
    meanL2 += cornerTerms([p, q, r]).L2;
  }
  meanL2 /= Math.max(corners.length, 1);
  let hstep = 1e-6 * Math.sqrt(meanL2 / 2);
  let x = [x0[0], x0[1]];

  const F = (xx) => {
    const { jmin, jmean } = cornerJacobians(corners, xx);
    // delta follows the worst corner, so the regularisation switches itself off
    // once every corner is positive and the tangle is gone
    const delta = jmin > 0 ? 0 : Math.sqrt(0.1 * jmean * (0.1 * jmean - jmin));
    return objective(corners, xx, delta);
  };

  let f = F(x);
  for (let it = 0; it < newtonIters; it++) {
    const fxp = F([x[0] + hstep, x[1]]), fxm = F([x[0] - hstep, x[1]]);
    const fyp = F([x[0], x[1] + hstep]), fym = F([x[0], x[1] - hstep]);
    if (![fxp, fxm, fyp, fym].every(Number.isFinite)) { hstep *= 0.1; continue; }
    const gx = (fxp - fxm) / (2 * hstep), gy = (fyp - fym) / (2 * hstep);
    const hxx = (fxp - 2 * f + fxm) / (hstep * hstep);
    const hyy = (fyp - 2 * f + fym) / (hstep * hstep);
    const fpp = F([x[0] + hstep, x[1] + hstep]);
    const fmm = F([x[0] - hstep, x[1] - hstep]);
    const hxy = (fpp - fxp - fyp + 2 * f - fxm - fym + fmm) / (2 * hstep * hstep);
    const det = hxx * hyy - hxy * hxy;
    let dx, dy;
    if (det > 0 && hxx > 0) {
      dx = -(hyy * gx - hxy * gy) / det;
      dy = -(-hxy * gx + hxx * gy) / det;
    } else {
      const gn = Math.hypot(gx, gy);
      if (gn === 0) break;
      const step = Math.sqrt(meanL2 / 2) * 0.2;
      dx = (-gx / gn) * step; dy = (-gy / gn) * step;
    }
    let alpha = 1.0, improved = false;
    for (let k = 0; k < 12; k++) {
      const cand = [x[0] + alpha * dx, x[1] + alpha * dy];
      const fn = F(cand);
      if (fn < f) { x = cand; f = fn; improved = true; break; }
      alpha *= 0.5;
    }
    if (!improved || Math.hypot(alpha * dx, alpha * dy) < 1e-12 * Math.sqrt(meanL2)) break;
  }
  return x;
}

// elements: array of arrays of node indices (any polygon size)
// coords:   array of [x, y]
// fixed:    array/Set of node indices that may not move
// slide:    optional map nodeIndex -> [[ax,ay],[bx,by]] segment the node is
//           confined to; a boundary node that is not a real corner may travel
//           along its own edge but must not leave it
function optimize(elements, coords, fixed, { iters = 8, newtonIters = 6, slide = null } = {}) {
  const X = coords.map((p) => [p[0], p[1]]);
  const pinned = fixed instanceof Set ? fixed : new Set(fixed);
  const adjacent = new Map();          // node -> elements containing it
  for (const el of elements) {
    for (const node of el) {
      if (pinned.has(node)) continue;
      if (!adjacent.has(node)) adjacent.set(node, new Set());
      adjacent.get(node).add(el);
    }
  }

  for (let sweep = 0; sweep < iters; sweep++) {
    for (const [node, els] of adjacent) {
      const corners = [];
      for (const el of els) {
        const n = el.length;
        const sin = Math.sin((Math.PI * (n - 2)) / n);
        for (let j = 0; j < n; j++) {
          const trio = [el[(j - 1 + n) % n], el[j], el[(j + 1) % n]];
          corners.push({
            P: [X[trio[0]], X[trio[1]], X[trio[2]]],
            slots: [trio[0] === node, trio[1] === node, trio[2] === node],
            sin,
          });
        }
      }
      if (!corners.length) continue;
      let x = optimizeNode(corners, X[node], newtonIters);
      if (slide && slide.has(node)) {
        // a boundary node that is not a real corner may travel along its own
        // edge but must not leave it
        const [a, b] = slide.get(node);
        const vx = b[0] - a[0], vy = b[1] - a[1];
        const len2 = vx * vx + vy * vy;
        if (len2 > 0) {
          let t = ((x[0] - a[0]) * vx + (x[1] - a[1]) * vy) / len2;
          t = Math.max(0.02, Math.min(0.98, t));
          x = [a[0] + t * vx, a[1] + t * vy];
        } else {
          x = X[node];
        }
      }
      X[node][0] = x[0];
      X[node][1] = x[1];
    }
  }
  return X;
}

function minQuality(elements, coords) {
  let worst = Infinity;
  for (const el of elements) {
    const n = el.length;
    const sin = Math.sin((Math.PI * (n - 2)) / n);
    for (let i = 0; i < n; i++) {
      const P = [coords[el[(i - 1 + n) % n]], coords[el[i]], coords[el[(i + 1) % n]]];
      const { J, L2 } = cornerTerms(P);
      worst = Math.min(worst, (2 * J) / L2 / sin);
    }
  }
  return worst === Infinity ? 1 : worst;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { optimize, minQuality, cornerTerms };
}
// the single-file build concatenates the sources, so the engine finds it here
if (typeof globalThis !== "undefined") {
  globalThis.MeshQuestUntangle = { optimize, minQuality, cornerTerms };
}
