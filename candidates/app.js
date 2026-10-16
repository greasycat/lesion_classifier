// A1 candidates with concept-model and Mammo-FM second opinions on real mammograms. Data from site/build_candidates.py.
// Heatmaps are calibrated here, as msynth_heatmap / fm_heatmap.py do it: p = bilinear(sigmoid(z + shift)) per pixel,
// q = isotonic(logit p), then a prior post-shift for the declared prevalence. Overlay as the M-SYNTH viewer:
// colour = colormap(norm(q)), opacity = alpha * norm(q)^0.5.
"use strict";
const $ = (id) => document.getElementById(id);
const GAMMA = 0.5, EPS = 1e-7;
let DATA, META, TEXT = null, cur = null, sel = null;
const imgCache = new Map(), qCache = new Map();

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const logit = (p) => { p = Math.min(Math.max(p, EPS), 1 - EPS); return Math.log(p / (1 - p)); };
const fmtP = (p) => (p < 0.001 ? p.toExponential(1) : p.toFixed(3));
const nice = (s) => s.toLowerCase().replace(/_/g, " ");

function loadImage(src) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
}

async function pixels(src) {
  if (imgCache.has(src)) return imgCache.get(src);
  const im = await loadImage(src);
  const c = document.createElement("canvas"); c.width = im.width; c.height = im.height;
  const g = c.getContext("2d"); g.drawImage(im, 0, 0);
  const d = g.getImageData(0, 0, im.width, im.height).data;
  imgCache.set(src, d);
  return d;
}

// ---------------------------------------------------------------- scores
function score(c, name) {
  if (name in c) return c[name];
  const f = META.fusion[name];
  let s = c.z / f.sd.z;
  for (const [k, a] of Object.entries(f.weights)) s += (a * c[k]) / f.sd[k];
  return s;
}

const pts = (it, src) => ({ fm: it.fm_cands, det: it.det_cands || [] }[src] || it.cands);
const native = (src) => ({ fm: "probe_small", det: "det" }[src] || "z");
const SRC_NAME = { a1: "A1 candidate", fm: "Mammo-FM peak", det: "Mammo-FM detector box" };
const SRC_PLURAL = { a1: "A1's candidates", fm: "Mammo-FM's peaks", det: "Mammo-FM's detector boxes" };

function ranked(it, name, src = "a1") {
  const all = pts(it, src);
  if (all.length && !(name in all[0]) && !(name in META.fusion)) name = native(src);   // e.g. detector score on A1's points
  return all.map((c) => ({ c, s: score(c, name) })).sort((a, b) => b.s - a.s).map((o, i) => ({ ...o, rank: i + 1 }));
}

function caseOf(it) {
  if (!it.lesions.length) return null;
  return `${ranked(it, "z")[0].c.hit}${ranked(it, "A1+CBM+probe_mammofm")[0].c.hit}`;
}

// ---------------------------------------------------------------- heatmap
function interp(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const t = (x - xs[lo]) / Math.max(xs[hi] - xs[lo], 1e-12);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

// calibrated pixel probabilities on the display grid (Float32Array, NaN outside the breast)
function calibrated(it, mapKey, view, mask) {
  const key = `${it.id}|${mapKey}|${mapKey === "a1" ? "" : view}`;
  if (qCache.has(key)) return qCache.get(key);
  const isA1 = mapKey === "a1";
  const cal = isA1 ? null : META.calibrators[mapKey][view];      // A1: raw sigmoid(z), the M-SYNTH calibrator flattens real images
  const grid = isA1 ? it.a1 : mapKey.startsWith("fm2040") ? it.fm2040 : it.fm;
  const Z = grid.z, gh = Z.length, gw = Z[0].length;
  const [cy, cx] = isA1 ? [grid.cell_mm, grid.cell_mm] : grid.cell_mm;
  const P = Z.map((row) => row.map((z) => sigmoid(z + (cal ? cal.shift : 0))));
  const W = it.w, H = it.h, q = new Float32Array(W * H).fill(NaN);
  const fy = new Float64Array(H), fx = new Float64Array(W);
  for (let r = 0; r < H; r++) fy[r] = Math.min(Math.max(((r + 0.5) * it.px_mm) / cy - 0.5, 0), gh - 1);
  for (let c = 0; c < W; c++) fx[c] = Math.min(Math.max(((c + 0.5) * it.px_mm) / cx - 0.5, 0), gw - 1);
  for (let r = 0; r < H; r++) {
    const r0 = Math.floor(fy[r]), r1 = Math.min(r0 + 1, gh - 1), ty = fy[r] - r0;
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      if (!mask[i * 4]) continue;
      const c0 = Math.floor(fx[c]), c1 = Math.min(c0 + 1, gw - 1), tx = fx[c] - c0;
      const p = (1 - ty) * ((1 - tx) * P[r0][c0] + tx * P[r0][c1]) + ty * ((1 - tx) * P[r1][c0] + tx * P[r1][c1]);
      let v = p;
      if (cal) { v = interp(cal.iso_x, cal.iso_y, logit(p)); if (cal.post_shift) v = sigmoid(logit(v) + cal.post_shift); }
      q[i] = v;
    }
  }
  let mx = 0;
  for (const v of q) if (v > mx) mx = v;
  const out = { q, max: mx };
  qCache.set(key, out);
  return out;
}

function settings() {
  return {
    map: $("c-map").value, view: $("c-view").value, heat: $("c-heat").value, src: $("c-src").value, score: $("c-score").value, k: +$("c-k").value,
    ghost: $("c-ghost").checked, ranks: $("c-ranks").checked, cmap: $("c-cmap").value, alpha: +$("c-alpha").value,
    scale: $("c-scale").value, floor: +$("c-floor").value, outline: $("c-outline").checked,
  };
}

function normRange(s, mx) {
  if (s.scale === "linear") return [0, 1];
  if (s.scale === "rel") { const hi = Math.log10(Math.max(mx, 1e-6)); return [hi - 2, hi]; }
  return [Math.log10(s.floor), 0];
}

function normFn(s, mx) {
  const [lo, hi] = normRange(s, mx);
  if (s.scale === "linear") return (p) => Math.min(Math.max(p, 0), 1);
  return (p) => (p <= 0 ? 0 : Math.min(Math.max((Math.log10(p) - lo) / (hi - lo), 0), 1));
}

// ---------------------------------------------------------------- places (boxes)
// As msynth_heatmap.top_places: box = bounding box of the connected region around the peak where p >= half the peak's
// p and the pixel is nearer this peak than any other peak of the same source, within 15 mm, at least one 2.7 mm cell
// wide. p = the source's own map (A1: raw sigmoid(z); Mammo-FM: the 5 + 10 mm map, INbreast calibrator, current view).
// In the chosen ranking, a peak whose centre falls inside an earlier box, or inside an earlier peak's half-max blob, is
// dropped (one box per blob).
const BOX_FRAC = 0.5, BOX_MAX_MM = 15, MIN_MM = 2.72;
const boxCache = new Map();

// connected region (8-neighbour) around (r, q) where P >= thr and keep(y, x), inside the window; returns its pixels' bbox
// and a membership test
function region(it, P, r, q, thr, R, keep) {
  const W = it.w, H = it.h;
  const r0 = Math.max(r - R, 0), r1 = Math.min(r + R, H - 1), c0 = Math.max(q - R, 0), c1 = Math.min(q + R, W - 1);
  const w = c1 - c0 + 1, seen = new Uint8Array((r1 - r0 + 1) * w), stack = [[r, q]];
  seen[(r - r0) * w + (q - c0)] = 1;
  let y0 = r, y1 = r, x0 = q, x1 = q;
  while (stack.length) {
    const [y, x] = stack.pop();
    y0 = Math.min(y0, y); y1 = Math.max(y1, y); x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy < r0 || yy > r1 || xx < c0 || xx > c1) continue;
      const k = (yy - r0) * w + (xx - c0);
      if (seen[k] || !(P[yy * W + xx] >= thr) || !keep(yy, xx)) continue;
      seen[k] = 1; stack.push([yy, xx]);
    }
  }
  const has = (y, x) => { y = Math.round(y); x = Math.round(x); return y >= r0 && y <= r1 && x >= c0 && x <= c1 && seen[(y - r0) * w + (x - c0)] === 1; };
  return { bbox: [y0, x0, y1 + 1, x1 + 1], has };
}

function boxOf(it, P, c, all) {
  const W = it.w, H = it.h, R = Math.round(BOX_MAX_MM / it.px_mm), half = Math.ceil(MIN_MM / it.px_mm / 2);
  const r = Math.min(Math.max(Math.round(c.y), 0), H - 1), q = Math.min(Math.max(Math.round(c.x), 0), W - 1);
  let box = [r - half, q - half, r + half, q + half];
  const pk = P[r * W + q];
  let blob = null;
  if (pk > 0) {
    const thr = BOX_FRAC * pk;
    const others = all.filter((o) => o !== c && Math.abs(o.y - r) <= 3 * R && Math.abs(o.x - q) <= 3 * R);
    const own = (y, x) => { const d = (y - r) ** 2 + (x - q) ** 2; return others.every((o) => d <= (y - o.y) ** 2 + (x - o.x) ** 2); };
    const [y0, x0, y1, x1] = region(it, P, r, q, thr, R, own).bbox;
    box = [Math.min(y0, r - half), Math.min(x0, q - half), Math.max(y1, r + half), Math.max(x1, q + half)];
    blob = region(it, P, r, q, thr, R, () => true).has;           // the whole half-max blob, for suppression
  }
  return { box, blob };
}

function places(it, score, src, view, mask) {
  const key = `${it.id}|${score}|${src}|${src === "fm" ? view : ""}`;
  if (boxCache.has(key)) return boxCache.get(key);
  if (src === "det") {           // the detector's own boxes (NMS at IoU 0.5); a box centred inside an earlier box is skipped
    const out = [];
    for (const o of ranked(it, score, src)) {
      if (out.some(({ box: b }) => o.c.y >= b[0] && o.c.y < b[2] && o.c.x >= b[1] && o.c.x < b[3])) continue;
      out.push({ ...o, rank: out.length + 1, box: o.c.box, blob: null });
    }
    boxCache.set(key, out);
    return out;
  }
  const P = calibrated(it, src === "fm" ? "fm_inbreast" : "a1", view, mask).q, all = pts(it, src), out = [];
  for (const o of ranked(it, score, src)) {
    if (out.some(({ box: b, blob }) => (o.c.y >= b[0] && o.c.y < b[2] && o.c.x >= b[1] && o.c.x < b[3]) || (blob && blob(o.c.y, o.c.x)))) continue;
    const { box, blob } = boxOf(it, P, o.c, all);
    out.push({ ...o, rank: out.length + 1, box, blob });
  }
  boxCache.set(key, out);
  return out;
}

// ---------------------------------------------------------------- drawing

async function draw() {
  if (!cur) return;
  const s = settings(), it = cur;
  const [gray, mask] = await Promise.all([pixels(`data/img/${it.id}.jpg`), pixels(`data/mask/${it.id}.png`)]);
  if (cur !== it) return;
  const W = it.w, H = it.h, cv = $("canvas");
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d"), out = g.createImageData(W, H), o = out.data;
  const R = places(it, s.score, s.src, s.view, mask), shown = R.slice(0, s.k);
  const heat = s.map !== "none" ? calibrated(it, s.map, s.view, mask) : null;
  const norm = heat ? normFn(s, heat.max) : null, lut = META.cmaps[s.cmap];
  const [lo] = heat ? normRange(s, heat.max) : [0];
  let inBox = null;
  if (heat && s.heat === "boxes") {
    inBox = new Uint8Array(W * H);
    for (const { box: [r0, c0, r1, c1] } of shown)
      for (let r = Math.max(r0, 0); r < Math.min(r1, H); r++) inBox.fill(1, r * W + Math.max(c0, 0), r * W + Math.min(c1, W));
  }
  for (let i = 0, j = 0; i < W * H; i++, j += 4) {
    const y = gray[j];
    let a = 0, n = 0;
    if (heat) {
      const p = heat.q[i];
      if (p > 0 && (!inBox || inBox[i]) && (s.scale === "linear" || Math.log10(p) >= lo)) { n = norm(p); a = s.alpha * Math.pow(n, GAMMA); }
    }
    const col = lut[Math.round(n * 255)];
    o[j] = (1 - a) * y + a * col[0]; o[j + 1] = (1 - a) * y + a * col[1]; o[j + 2] = (1 - a) * y + a * col[2]; o[j + 3] = 255;
  }
  g.putImageData(out, 0, 0);
  const lw = Math.max(1.5, W / 400), fs = Math.max(12, Math.round(W / 45));
  if (s.outline) {
    g.strokeStyle = "cyan"; g.lineWidth = lw;
    for (const poly of it.outline) { g.beginPath(); poly.forEach(([r, c], k) => (k ? g.lineTo(c, r) : g.moveTo(c, r))); g.closePath(); g.stroke(); }
  }
  if (s.ghost && (s.score !== "z" || s.src !== "a1")) {
    g.setLineDash([5, 4]); g.strokeStyle = "rgba(255,255,255,.75)"; g.lineWidth = lw;
    for (const { box: [r0, c0, r1, c1] } of places(it, "z", "a1", s.view, mask).slice(0, s.k)) g.strokeRect(c0, r0, c1 - c0, r1 - r0);
    g.setLineDash([]);
  }
  g.font = `bold ${fs}px system-ui`;
  for (const { c, rank, box: [r0, c0, r1, c1] } of shown) {
    const hitCol = !it.lesions.length || !s.outline ? "#ff3b30" : c.hit ? "#34c759" : "#ff3b30";
    g.strokeStyle = g.fillStyle = hitCol; g.lineWidth = c === focus(R, s) ? lw * 2.5 : lw;
    g.strokeRect(c0, r0, c1 - c0, r1 - r0);
    g.beginPath(); g.arc(c.x, c.y, lw * 1.5, 0, 2 * Math.PI); g.fill();
    if (s.ranks) g.fillText(String(rank), c1 + 3, r0 + fs);
  }
  drawBar(s, heat);
  info(s, it, R, heat, mask);
  hint(s, it, heat);
  say(s, it, R);
  detail(s, it, R);
  cv.onclick = (e) => {                                  // click a box to select it
    const rc = cv.getBoundingClientRect(), x = ((e.clientX - rc.left) * W) / rc.width, y = ((e.clientY - rc.top) * H) / rc.height;
    const hit = shown.filter(({ box: b }) => y >= b[0] && y < b[2] && x >= b[1] && x < b[3]);
    if (hit.length) { sel = hit[0].c; draw(); }
  };
}

function drawBar(s, heat) {
  const c = $("bar"), g = c.getContext("2d"), lut = META.cmaps[s.cmap], L = $("bar-labels");
  L.innerHTML = "";
  g.clearRect(0, 0, c.width, c.height);
  if (!heat) return;
  for (let y = 0; y < c.height; y++) { const col = lut[Math.round((1 - y / (c.height - 1)) * 255)]; g.fillStyle = `rgb(${col})`; g.fillRect(0, y, c.width, 1); }
  const [lo, hi] = normRange(s, heat.max);
  let ticks = [];
  if (s.scale === "linear") ticks = [0, 0.25, 0.5, 0.75, 1].map((p) => [p, p]);
  else for (let e = Math.ceil(lo); e <= Math.floor(hi + 1e-9); e++) ticks.push([(e - lo) / (hi - lo), Math.pow(10, e)]);
  if (s.scale === "rel") ticks = [[0, Math.pow(10, lo)], [0.5, Math.pow(10, (lo + hi) / 2)], [1, Math.pow(10, hi)]];
  for (const [f, p] of ticks) {
    const sp = document.createElement("span");
    sp.style.top = `${(1 - f) * 100}%`; sp.textContent = p >= 0.01 || p === 0 ? +p.toFixed(3) : p.toExponential(0);
    L.appendChild(sp);
  }
}

function info(s, it, R, heat, mask) {
  const set = META.sets[it.set].name;
  let t = `${set} · ${it.view} · ${it.image}`;
  if (it.lesions.length) {
    const L = it.lesions[0];
    t += ` · mass ${L.r_mm} mm radius` + (L.shape ? `, ${nice(L.shape)}` : "") + (L.margins ? `, ${nice(L.margins)}` : "") +
      (L.pathology ? `, ${nice(L.pathology)}` : "") + (L.label && L.label !== "mass" ? ` (${L.label})` : "");
    const first = R.find((o) => o.c.hit), a1 = places(it, "z", "a1", s.view, mask).find((o) => o.c.hit);
    t += ` · first hit: rank ${first ? first.rank : "none in top 20"}` + (s.score !== "z" || s.src !== "a1" ? ` (A1 alone: ${a1 ? a1.rank : "none"})` : "");
  } else t += " · normal breast";
  if (heat) t += s.map === "a1" ? ` · A1 map max sigmoid(z) ${fmtP(heat.max)} (uncalibrated)` :
    ` · map max ${fmtP(heat.max)} at ${Math.round(META.views[s.view] * 100)} % prevalence`;
  $("info").textContent = t;
}


// LLM paraphrase of the selected place (site/paraphrase_candidates.py), above the image
function say(s, it, R) {
  const S = $("say"), c = focus(R, s);
  if (!c) { S.innerHTML = ""; return; }
  const o = R.find((x) => x.c === c), t = TEXT?.[it.id]?.[s.src]?.[c.k - 1];
  const tag = !t ? "" : t.llm ? '<span class="src">LLM paraphrase · checked against the numbers</span>' : '<span class="src">template</span>';
  const hitTag = it.lesions.length && s.outline ? (c.hit ? '<span class="hit">hits the mass</span>' : '<span class="miss">misses the mass</span>') : "";
  S.innerHTML = `<div class="say-head"><b>Rank ${o.rank}</b> of ${Math.min(s.k, R.length)} · ${SRC_NAME[s.src]} ${hitTag}${tag}</div>` +
    `<p>${t ? t.text : "No text for this box."}</p>` +
    (R.length > 1 && s.k > 1 ? '<p class="muted small">Click another box on the image to read about it.</p>' : "");
}

// the selected place if it is shown, else rank 1
function focus(R, s) {
  const shown = R.slice(0, Math.max(s.k, 1));
  return shown.find((o) => o.c === sel)?.c || shown[0]?.c || null;
}

function bars(names, vals) {
  return names.map((n, i) => `<div class="bar-row"><span>${nice(n)}</span><i style="width:${Math.round(vals[i] * 100)}%"></i><b>${vals[i].toFixed(2)}</b></div>`).join("");
}

// the model that reads each box source: A1's candidates -> the concept model; Mammo-FM's peaks -> Mammo-FM's probes
const READER = {
  a1: { name: "Concept model", p: "pc", shape: "shape", margin: "margin", size: "size_mm", malig: "malig",
        maligNote: "from its shape, margin and size reading", note: "ResNet-50 on a 40 mm crop, trained on CBIS Mass-Training." },
  fm: { name: "Mammo-FM", p: "probe_small", shape: "fm_shape", margin: "fm_margin", size: "fm_size_mm", malig: "fm_malig",
        maligNote: "malignancy probe", note: "Probes on frozen Mammo-FM embeddings, fitted on CBIS Mass-Training: mass present on 5 + 10 mm boxes, the rest on 20 + 40 mm boxes." },
};
READER.det = { ...READER.fm, note: "Box from Mammo-FM's RetinaNet detector (the authors' head on the frozen Mammo-FM encoder, trained on CBIS Mass-Training boxes); read by the Mammo-FM probes at the box centre: " +
  "mass present on 5 + 10 mm boxes, the rest on 20 + 40 mm boxes." };

function trueLabel(it, c) {  // c.box: a detector box (hit = its centre), else a peak
  if (!it.lesions.length) return '<p>No mass annotated (normal breast): every box here is a false alarm.</p>';
  const L = it.lesions[0];
  const desc = [L.shape && nice(L.shape), L.margins && `${nice(L.margins)} margins`, L.pathology && nice(L.pathology),
    L.label && L.label !== "mass" && L.label].filter(Boolean).join(", ");
  return `<p>Annotated mass: ${L.r_mm} mm radius${desc ? ", " + desc : ""}.</p>` +
    `<p class="${c.hit ? "hit" : "miss"}">This box ${c.hit ? "hits the mass (its " + (c.box ? "centre" : "peak") + " is within 4 mm)" : "misses the mass"}.</p>`;
}

function detail(s, it, R) {
  const D = $("detail"), c = focus(R, s);
  if (!c) { D.innerHTML = '<p class="muted">No places.</p>'; return; }
  const o = R.find((x) => x.c === c), M = READER[s.src], b = o.box;
  D.innerHTML = `<h3>Rank ${o.rank} · ${SRC_NAME[s.src]}</h3>` +
    `<h4>True label</h4>${trueLabel(it, c)}` +
    `<h4>${M.name} reading</h4>` +
    (s.src === "det" ? `<p>detector box score <b>${c.det.toFixed(3)}</b> <span class="muted">(focal-loss score, not a probability)</span></p>` : "") +
    `<p>mass probability <b>${fmtP(sigmoid(c[M.p]))}</b> · size ≈ ${c[M.size]} mm radius · malignancy ${c[M.malig].toFixed(2)} ` +
    `<span class="muted">(${M.maligNote})</span> · box ${Math.round((b[2] - b[0]) * it.px_mm)} × ${Math.round((b[3] - b[1]) * it.px_mm)} mm</p>` +
    `<h4>Shape</h4>${bars(META.shapes, c[M.shape])}<h4>Margin</h4>${bars(META.margins, c[M.margin])}` +
    `<p class="muted">${M.note} Probabilities are raw model outputs at their training base rates, not calibrated. ` +
    `Shape, margin, size and malignancy are read as if a mass were there; they mean little when the mass probability is low.</p>`;
}

// where the heatmap's hottest spot is relative to the boxes' source
function hint(s, it, heat) {
  const H = $("hint");
  if (!heat || (s.map === "a1" ? s.src === "a1" : s.src === "fm")) { H.textContent = ""; return; }
  let mi = 0;
  for (let i = 0; i < heat.q.length; i++) if (heat.q[i] > (heat.q[mi] || 0)) mi = i;
  const y = Math.floor(mi / it.w), x = mi % it.w;
  const d = Math.min(...pts(it, s.src).map((c) => Math.hypot(c.y - y, c.x - x))) * it.px_mm;
  const what = s.map === "a1" ? "A1" : "Mammo-FM";
  H.textContent = `Boxes are ${SRC_PLURAL[s.src]}, heatmap is ${what}: ` +
    `${what}'s hottest spot is ${Math.round(d)} mm from the nearest box, so no box can land there.`;
  if (d <= 10) H.textContent = "";
}

// ---------------------------------------------------------------- browser
function fillSelect(sel_, entries) {
  for (const [v, t] of entries) { const o = document.createElement("option"); o.value = v; o.textContent = t; sel_.appendChild(o); }
}

function grid() {
  const f = { set: $("f-set").value, lesion: $("f-lesion").value, kase: $("f-case").value };
  const items = DATA.filter((it) => (f.set === "" || it.set === f.set) && (f.lesion === "" || (it.lesions.length > 0) === (f.lesion === "1")) &&
    (f.kase === "" || it.case === f.kase));
  $("count").textContent = `${items.length} of ${DATA.length} images`;
  const G = $("grid"); G.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    const tag = `${{ test: "CBIS", tompei: "TOMPEI", dmid: "DMID" }[it.set]}${it.lesions.length ? "" : " · normal"}`;
    b.innerHTML = `<img loading="lazy" src="data/img/${it.id}.jpg" alt=""><span class="tag">${tag}</span>`;
    b.onclick = () => { cur = it; sel = null; document.querySelectorAll("#grid button").forEach((x) => x.classList.toggle("on", x === b)); draw(); };
    if (it === cur) b.classList.add("on");
    G.appendChild(b);
  }
  if ((!cur || !items.includes(cur)) && items.length) G.firstChild.click();
}

function setsNote() {
  const rows = Object.values(META.sets).map((x) => `${x.name}: ${x.n_images_set} images (${x.licence})`);
  $("sets-note").innerHTML = "A fixed sample from each set. " + rows.join("; ") +
    ". Candidate AUROC across each whole set is in the table below.";
  const S = Object.keys(META.scorers);
  const T = document.createElement("table");
  T.className = "auroc";
  T.innerHTML = `<tr><th>candidate AUROC</th>${Object.values(META.sets).map((x) => `<th>${x.name.split(" ")[0]}</th>`).join("")}</tr>` +
    S.map((k) => `<tr><td>${META.scorers[k]}</td>${Object.values(META.sets).map((x) => `<td>${x.candidate_auroc[k].toFixed(2)}</td>`).join("")}</tr>`).join("");
  $("sets-note").after(T);
}

function syncSource(map = true, score = true) {
  const src = $("c-src").value, fm = src !== "a1";             // the detector's boxes are drawn on Mammo-FM's map
  if (map) $("c-map").value = fm ? ($("c-map").value.startsWith("fm") ? $("c-map").value : "fm_inbreast") : "a1";
  if (score) $("c-score").value = native(src);
}

async function init() {
  const j = await (await fetch("data/items.json")).json();
  DATA = j.items; META = j.meta;
  try { TEXT = await (await fetch("data/text.json")).json(); } catch (e) { TEXT = null; }
  for (const it of DATA) it.case = caseOf(it);
  fillSelect($("f-set"), Object.entries(META.sets).map(([k, v]) => [k, v.name]));
  fillSelect($("c-view"), Object.entries(META.views).map(([k, v]) => [k, `${Math.round(v * 100)} %`]));
  $("c-view").value = "p10";
  fillSelect($("c-score"), Object.entries(META.scorers).concat([["det", "Mammo-FM detector box score"]]));
  $("c-score").value = "A1+CBM+probe_mammofm";
  fillSelect($("c-cmap"), Object.keys(META.cmaps).map((c) => [c, c]));
  for (const id of ["f-set", "f-lesion", "f-case"]) $(id).onchange = grid;
  $("c-src").addEventListener("change", () => {               // one model per box source: map and ranking follow it
    syncSource();
    sel = null;
    $("c-map").dispatchEvent(new Event("change"));
  });
  $("c-map").addEventListener("change", () => {
    const a1 = $("c-map").value === "a1";
    if (a1) $("c-scale").value = "rel";
    else if ($("c-scale").value === "rel") $("c-scale").value = "log";
    $("c-view").disabled = a1;
  });
  for (const el of document.querySelectorAll(".controls select, .controls input")) el.addEventListener("input", () => {
    $("o-k").textContent = $("c-k").value; $("o-alpha").textContent = (+$("c-alpha").value).toFixed(2); draw();
  });
  const q = new URLSearchParams(location.search);   // e.g. ?id=tompei_00262&src=a1&map=fm_inbreast&view=p10&score=z&k=5&sel=2
  for (const [k, id] of [["src", "c-src"], ["map", "c-map"], ["view", "c-view"], ["score", "c-score"], ["k", "c-k"], ["cmap", "c-cmap"],
                         ["alpha", "c-alpha"], ["scale", "c-scale"], ["floor", "c-floor"], ["heat", "c-heat"]]) if (q.has(k)) $(id).value = q.get(k);
  $("o-k").textContent = $("c-k").value; $("o-alpha").textContent = (+$("c-alpha").value).toFixed(2);
  if (!q.has("map") || !q.has("score")) syncSource(!q.has("map"), !q.has("score"));
  if ($("c-map").value === "a1") $("c-scale").value = q.get("scale") || "rel";
  $("c-view").disabled = $("c-map").value === "a1";
  setsNote();
  cur = DATA.find((x) => x.id === q.get("id")) || null;
  grid();
  if (cur && q.has("sel")) { sel = cur.cands[+q.get("sel") - 1] || null; draw(); }
  if (cur) draw();
}
init();
