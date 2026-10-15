// A1 candidates with concept-model and Mammo-FM second opinions on real mammograms. Data from site/build_candidates.py.
// Heatmaps are calibrated here, as msynth_heatmap / fm_heatmap.py do it: p = bilinear(sigmoid(z + shift)) per pixel,
// q = isotonic(logit p), then a prior post-shift for the declared prevalence. Overlay as the M-SYNTH viewer:
// colour = colormap(norm(q)), opacity = alpha * norm(q)^0.5.
"use strict";
const $ = (id) => document.getElementById(id);
const GAMMA = 0.5, EPS = 1e-7;
let DATA, META, cur = null, sel = null;
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

function ranked(it, name) {
  return it.cands.map((c) => ({ c, s: score(c, name) })).sort((a, b) => b.s - a.s).map((o, i) => ({ ...o, rank: i + 1 }));
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
  const grid = isA1 ? it.a1 : it.fm;
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
    map: $("c-map").value, view: $("c-view").value, heat: $("c-heat").value, score: $("c-score").value, k: +$("c-k").value,
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

// ---------------------------------------------------------------- drawing
const boxPx = (it) => META.box_mm / it.px_mm;

async function draw() {
  if (!cur) return;
  const s = settings(), it = cur;
  const [gray, mask] = await Promise.all([pixels(`data/img/${it.id}.jpg`), pixels(`data/mask/${it.id}.png`)]);
  if (cur !== it) return;
  const W = it.w, H = it.h, cv = $("canvas");
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d"), out = g.createImageData(W, H), o = out.data;
  const R = ranked(it, s.score), shown = R.slice(0, s.k);
  const heat = s.map !== "none" ? calibrated(it, s.map, s.view, mask) : null;
  const norm = heat ? normFn(s, heat.max) : null, lut = META.cmaps[s.cmap];
  const [lo] = heat ? normRange(s, heat.max) : [0];
  let inBox = null;
  if (heat && s.heat === "boxes") {
    inBox = new Uint8Array(W * H);
    const b = boxPx(it) / 2;
    for (const { c } of shown) {
      const r0 = Math.max(Math.round(c.y - b), 0), r1 = Math.min(Math.round(c.y + b), H);
      const c0 = Math.max(Math.round(c.x - b), 0), c1 = Math.min(Math.round(c.x + b), W);
      for (let r = r0; r < r1; r++) inBox.fill(1, r * W + c0, r * W + c1);
    }
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
  const b = boxPx(it);
  if (s.ghost && s.score !== "z") {
    g.setLineDash([5, 4]); g.strokeStyle = "rgba(255,255,255,.75)"; g.lineWidth = lw;
    for (const { c } of ranked(it, "z").slice(0, s.k)) g.strokeRect(c.x - b / 2, c.y - b / 2, b, b);
    g.setLineDash([]);
  }
  g.font = `bold ${fs}px system-ui`;
  for (const { c, rank } of shown) {
    const hitCol = !it.lesions.length || !s.outline ? "#ff3b30" : c.hit ? "#34c759" : "#ff3b30";
    g.strokeStyle = g.fillStyle = hitCol; g.lineWidth = c === sel ? lw * 2.5 : lw;
    g.strokeRect(c.x - b / 2, c.y - b / 2, b, b);
    if (s.ranks) g.fillText(String(rank), c.x + b / 2 + 3, c.y - b / 2 + fs);
  }
  drawBar(s, heat);
  info(s, it, R, heat);
  table(s, it, R);
  detail();
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

function info(s, it, R, heat) {
  const set = META.sets[it.set].name;
  let t = `${set} · ${it.view} · ${it.image}`;
  if (it.lesions.length) {
    const L = it.lesions[0];
    t += ` · mass ${L.r_mm} mm radius` + (L.shape ? `, ${nice(L.shape)}` : "") + (L.margins ? `, ${nice(L.margins)}` : "") +
      (L.pathology ? `, ${nice(L.pathology)}` : "") + (L.label && L.label !== "mass" ? ` (${L.label})` : "");
    const first = R.find((o) => o.c.hit), a1 = ranked(it, "z").find((o) => o.c.hit);
    t += ` · first hit: rank ${first ? first.rank : "none in top 20"}` + (s.score !== "z" ? ` (A1 alone: ${a1 ? a1.rank : "none"})` : "");
  } else t += " · normal breast";
  if (heat) t += s.map === "a1" ? ` · A1 map max sigmoid(z) ${fmtP(heat.max)} (uncalibrated)` :
    ` · map max ${fmtP(heat.max)} at ${Math.round(META.views[s.view] * 100)} % prevalence`;
  $("info").textContent = t;
}

function table(s, it, R) {
  const T = $("places");
  const rows = R.slice(0, Math.max(s.k, 1));
  const a1r = new Map(ranked(it, "z").map((o) => [o.c, o.rank]));
  T.innerHTML = `<tr><th>rank</th><th>A1 rank</th><th>${META.scorers[s.score]}</th><th>A1 z</th><th>concept p(mass)</th>` +
    `<th>Mammo-FM p(mass)</th>${it.lesions.length ? "<th>hit</th>" : ""}</tr>` +
    rows.map(({ c, s: v, rank }) => `<tr data-k="${c.k}" class="${c === sel ? "on" : ""}"><td>${rank}</td><td>${a1r.get(c)}</td>` +
      `<td>${v.toFixed(2)}</td><td>${c.z.toFixed(2)}</td><td>${fmtP(sigmoid(c.pc))}</td><td>${fmtP(sigmoid(c.probe_mammofm))}</td>` +
      (it.lesions.length ? `<td>${c.hit ? "✓" : ""}</td>` : "") + "</tr>").join("");
  for (const tr of T.querySelectorAll("tr[data-k]")) tr.onclick = () => { const c = it.cands[+tr.dataset.k - 1]; sel = sel === c ? null : c; draw(); };
}

function bars(names, vals) {
  return names.map((n, i) => `<div class="bar-row"><span>${nice(n)}</span><i style="width:${Math.round(vals[i] * 100)}%"></i><b>${vals[i].toFixed(2)}</b></div>`).join("");
}

function detail() {
  const D = $("detail");
  if (!sel) { D.innerHTML = '<p class="muted">Click a row for the concept model\'s reading of that region.</p>'; return; }
  const c = sel;
  D.innerHTML = `<h3>Candidate ${c.k} (A1 order)</h3>` +
    `<p>Concept model: mass present ${fmtP(sigmoid(c.pc))}, size ≈ ${c.size_mm} mm radius, malignancy (from concepts) ${c.malig.toFixed(2)}</p>` +
    `<h4>Shape</h4>${bars(META.shapes, c.shape)}<h4>Margin</h4>${bars(META.margins, c.margin)}` +
    `<p class="muted">Concepts are read as if a mass were there. They mean little when mass present is low.</p>`;
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

async function init() {
  const j = await (await fetch("data/items.json")).json();
  DATA = j.items; META = j.meta;
  for (const it of DATA) it.case = caseOf(it);
  fillSelect($("f-set"), Object.entries(META.sets).map(([k, v]) => [k, v.name]));
  fillSelect($("c-view"), Object.entries(META.views).map(([k, v]) => [k, `${Math.round(v * 100)} %`]));
  $("c-view").value = "p10";
  fillSelect($("c-score"), Object.entries(META.scorers));
  $("c-score").value = "A1+CBM+probe_mammofm";
  fillSelect($("c-cmap"), Object.keys(META.cmaps).map((c) => [c, c]));
  for (const id of ["f-set", "f-lesion", "f-case"]) $(id).onchange = grid;
  $("c-map").addEventListener("change", () => {
    const a1 = $("c-map").value === "a1";
    if (a1) $("c-scale").value = "rel";
    $("c-view").disabled = a1;
  });
  for (const el of document.querySelectorAll(".controls select, .controls input")) el.addEventListener("input", () => {
    $("o-k").textContent = $("c-k").value; $("o-alpha").textContent = (+$("c-alpha").value).toFixed(2); draw();
  });
  const q = new URLSearchParams(location.search);   // e.g. ?id=tompei_00262&map=fm_inbreast&view=p10&score=z&k=5&sel=2
  for (const [k, id] of [["map", "c-map"], ["view", "c-view"], ["score", "c-score"], ["k", "c-k"], ["cmap", "c-cmap"],
                         ["alpha", "c-alpha"], ["scale", "c-scale"], ["floor", "c-floor"], ["heat", "c-heat"]]) if (q.has(k)) $(id).value = q.get(k);
  $("o-k").textContent = $("c-k").value; $("o-alpha").textContent = (+$("c-alpha").value).toFixed(2);
  $("c-view").disabled = $("c-map").value === "a1";
  setsNote();
  cur = DATA.find((x) => x.id === q.get("id")) || null;
  grid();
  if (cur && q.has("sel")) { sel = cur.cands[+q.get("sel") - 1] || null; draw(); }
  if (cur) draw();
}
init();
