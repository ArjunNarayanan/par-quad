/* Mesh Quest — Copyright 2026 Arjun Narayanan.
 * Licensed under the MIT License (see LICENSE).
 * Released under the MIT License (see LICENSE).
 */
// --- GIF89a encoder ---------------------------------------------------------
//
// Just enough GIF to share a solve. The page ships as a single self-contained
// file and never talks to the network, so this is written from scratch rather
// than pulled from a CDN.
//
// Three things keep the output small: one global palette built by median cut,
// LZW compression, and per-frame diffing — between two moves most of the board
// is unchanged, and unchanged pixels are written as transparent so the previous
// frame shows through.

const GIF_BINS = 32768; // colour histogram at 5 bits per channel
const GIF_TRANSPARENT = 0; // palette slot reserved for "unchanged since last frame"

function gifBin(r, g, b) {
  return ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
}

// Exact colour sums are kept per bin, so a palette entry ends up the true
// average of the pixels behind it rather than the centre of a coarse bin.
function gifHistogram() {
  return { count: new Uint32Array(GIF_BINS), sum: new Float64Array(GIF_BINS * 3) };
}

function gifCount(hist, rgba) {
  const { count, sum } = hist;
  for (let i = 0; i < rgba.length; i += 4) {
    const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
    const bin = gifBin(r, g, b);
    count[bin] += 1;
    sum[bin * 3] += r;
    sum[bin * 3 + 1] += g;
    sum[bin * 3 + 2] += b;
  }
}

// Median cut: repeatedly split a box of colours at its weighted median, along
// its widest channel. Returns the palette plus a bin -> palette index table,
// which makes mapping a pixel a single lookup with no nearest-colour search.
function gifPalette(hist, wanted) {
  const { count, sum } = hist;
  const chan = (bin, axis) => (bin >> (10 - axis * 5)) & 31;

  // Which box to split next is scored by population * spread, not spread alone.
  // A board is mostly flat near-whites separated by a handful of levels: on
  // spread alone that whole cluster looks tiny and never gets split, so the
  // background, the dull faces and the palest tile all collapse into one grey
  // while stray anti-aliasing colours eat the palette.
  const describe = (bins) => {
    let population = 0, span = 0, axis = 0;
    for (const bin of bins) population += count[bin];
    for (let a = 0; a < 3; a++) {
      let lo = 31, hi = 0;
      for (const bin of bins) {
        const c = chan(bin, a);
        if (c < lo) lo = c;
        if (c > hi) hi = c;
      }
      if (hi - lo > span) { span = hi - lo; axis = a; }
    }
    return { bins, population, span, axis, score: population * span };
  };

  const occupied = [];
  for (let bin = 0; bin < GIF_BINS; bin++) if (count[bin]) occupied.push(bin);
  let boxes = [describe(occupied)];
  while (boxes.length < wanted) {
    let pick = -1, best = 0;
    for (let i = 0; i < boxes.length; i++) {
      if (boxes[i].score > best) { best = boxes[i].score; pick = i; }
    }
    if (pick < 0) break; // every box holds a single colour: nothing left to split
    const box = boxes[pick];
    const bins = box.bins;
    bins.sort((a, b) => chan(a, box.axis) - chan(b, box.axis));
    // Cut at the middle of the colour range rather than at the median pixel.
    // One colour here can hold most of the pixels — the paper the board is
    // drawn on — and a median cut then lands just past it every time, peeling
    // off a thin tail while the background, the tiles and the dull faces stay
    // stuck in one box for good.
    const lo = chan(bins[0], box.axis);
    const hi = chan(bins[bins.length - 1], box.axis);
    const mid = (lo + hi) / 2;
    let cut = 0;
    while (cut < bins.length && chan(bins[cut], box.axis) <= mid) cut += 1;
    if (cut === 0 || cut === bins.length) cut = bins.length >> 1; // never split empty
    boxes.splice(pick, 1, describe(bins.slice(0, cut)), describe(bins.slice(cut)));
  }

  const palette = new Uint8Array(256 * 3);
  const binIndex = new Uint8Array(GIF_BINS);
  boxes.forEach((box, i) => {
    const index = i + 1; // slot 0 stays reserved for transparency
    let n = 0, r = 0, g = 0, b = 0;
    for (const bin of box.bins) {
      n += count[bin];
      r += sum[bin * 3];
      g += sum[bin * 3 + 1];
      b += sum[bin * 3 + 2];
      binIndex[bin] = index;
    }
    palette[index * 3] = Math.round(r / n);
    palette[index * 3 + 1] = Math.round(g / n);
    palette[index * 3 + 2] = Math.round(b / n);
  });
  return { palette, binIndex };
}

class GifBytes {
  constructor(capacity = 1 << 16) {
    this.buf = new Uint8Array(capacity);
    this.len = 0;
  }
  _room(n) {
    if (this.len + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.len + n) cap *= 2;
    const grown = new Uint8Array(cap);
    grown.set(this.buf.subarray(0, this.len));
    this.buf = grown;
  }
  byte(v) { this._room(1); this.buf[this.len++] = v & 0xff; }
  bytes(a) { this._room(a.length); this.buf.set(a, this.len); this.len += a.length; }
  short(v) { this.byte(v); this.byte(v >> 8); }
  ascii(s) { for (let i = 0; i < s.length; i++) this.byte(s.charCodeAt(i)); }
  take() { return this.buf.slice(0, this.len); }
}

// LZW as GIF wants it: 8-bit minimum code size, codes packed low-bit-first into
// sub-blocks of at most 255 bytes.
function gifLzw(out, indices) {
  const MIN = 8, CLEAR = 1 << MIN, EOI = CLEAR + 1;
  out.byte(MIN);
  const block = new Uint8Array(255);
  let blockLen = 0;
  const flush = () => {
    if (!blockLen) return;
    out.byte(blockLen);
    out.bytes(block.subarray(0, blockLen));
    blockLen = 0;
  };
  let cur = 0, curBits = 0, codeSize = MIN + 1, next = EOI + 1;
  let table = new Map();
  const emit = (code) => {
    cur |= code << curBits;
    curBits += codeSize;
    while (curBits >= 8) {
      block[blockLen++] = cur & 0xff;
      cur >>= 8;
      curBits -= 8;
      if (blockLen === 255) flush();
    }
  };

  emit(CLEAR);
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const k = indices[i];
    const key = (prefix << 8) | k;
    const known = table.get(key);
    if (known !== undefined) { prefix = known; continue; }
    emit(prefix);
    if (next === 4096) { // dictionary full: start a new one
      emit(CLEAR);
      table = new Map();
      next = EOI + 1;
      codeSize = MIN + 1;
    } else {
      if (next >= (1 << codeSize)) codeSize += 1;
      table.set(key, next++);
    }
    prefix = k;
  }
  emit(prefix);
  emit(EOI);
  if (curBits > 0) {
    block[blockLen++] = cur & 0xff;
    if (blockLen === 255) flush();
  }
  flush();
  out.byte(0); // end of the LZW sub-block chain
}

// Encoding runs as a generator so a browser can drive it a step at a time and
// stay responsive; each yield is a 0..1 progress fraction. Call gifEncode() for
// the plain synchronous version.
//
//   frames  array of RGBA Uint8ClampedArray, each width * height * 4
//   delays  per-frame delay in hundredths of a second
function* gifEncodeSteps(frames, width, height, options = {}) {
  if (!frames.length) throw new Error("gif: no frames");
  const loop = options.loop ?? 0;
  let delays = options.delays || frames.map(() => 60);

  // Identical neighbours are free: drop them and lengthen the delay instead.
  const kept = [], keptDelays = [];
  for (let i = 0; i < frames.length; i++) {
    const same = i > 0 && gifSameFrame(frames[i], kept[kept.length - 1]);
    if (same) keptDelays[keptDelays.length - 1] += delays[i];
    else { kept.push(frames[i]); keptDelays.push(delays[i]); }
  }
  frames = kept;
  delays = keptDelays;

  const hist = gifHistogram();
  for (let i = 0; i < frames.length; i++) {
    gifCount(hist, frames[i]);
    yield (0.45 * (i + 1)) / frames.length;
  }
  const { palette, binIndex } = gifPalette(hist, 255);
  yield 0.5;

  const out = new GifBytes(1 << 18);
  out.ascii("GIF89a");
  out.short(width);
  out.short(height);
  out.byte(0xf7); // global colour table, 8 bits per pixel, 256 entries
  out.byte(GIF_TRANSPARENT);
  out.byte(0);
  out.bytes(palette);
  // NETSCAPE2.0 is how a GIF says "loop"
  out.byte(0x21); out.byte(0xff); out.byte(0x0b);
  out.ascii("NETSCAPE2.0");
  out.byte(0x03); out.byte(0x01); out.short(loop); out.byte(0);

  const pixels = width * height;
  let previous = null;
  for (let f = 0; f < frames.length; f++) {
    const rgba = frames[f];
    const indices = new Uint8Array(pixels);
    for (let p = 0, i = 0; p < pixels; p++, i += 4) {
      indices[p] = binIndex[gifBin(rgba[i], rgba[i + 1], rgba[i + 2])];
    }

    // Only the rectangle that actually changed gets written; inside it,
    // unchanged pixels go out transparent and the previous frame shows through.
    let x0 = 0, y0 = 0, x1 = width - 1, y1 = height - 1, diffed = false;
    if (previous) {
      diffed = true;
      x0 = width; y0 = height; x1 = -1; y1 = -1;
      for (let y = 0, p = 0; y < height; y++) {
        for (let x = 0; x < width; x++, p++) {
          if (indices[p] === previous[p]) continue;
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
      if (x1 < x0) { x0 = y0 = 0; x1 = y1 = 0; } // no change: emit one pixel
    }
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    const sub = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const p = (y0 + y) * width + x0 + x;
        sub[y * w + x] = diffed && indices[p] === previous[p]
          ? GIF_TRANSPARENT : indices[p];
      }
    }

    out.byte(0x21); out.byte(0xf9); out.byte(0x04);
    // disposal 1 (leave the frame in place) is what makes diffing work
    out.byte((1 << 2) | (diffed ? 1 : 0));
    out.short(Math.max(2, Math.round(delays[f])));
    out.byte(GIF_TRANSPARENT);
    out.byte(0);

    out.byte(0x2c);
    out.short(x0); out.short(y0); out.short(w); out.short(h);
    out.byte(0); // no local colour table, not interlaced
    gifLzw(out, sub);

    previous = indices;
    yield 0.5 + (0.5 * (f + 1)) / frames.length;
  }

  out.byte(0x3b);
  return out.take();
}

function gifSameFrame(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 4) {
    if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2]) return false;
  }
  return true;
}

function gifEncode(frames, width, height, options) {
  const steps = gifEncodeSteps(frames, width, height, options);
  let step = steps.next();
  while (!step.done) step = steps.next();
  return step.value;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { gifEncode, gifEncodeSteps };
}
