// M-SYNTH lesion heatmap preview. Maps are precomputed (site/build.py); the overlay follows msynth_heatmap:
// colour = colormap(norm(p)), opacity = alpha * norm(p)^0.5, norm = log10 between the floor and 1 (or linear 0..1).
"use strict";
const $ = (id) => document.getElementById(id);
const GAMMA = 0.5;
let DATA, META, cur = null;
const cache = new Map();

const decode = (v) => (v === 0 ? 0 : Math.pow(10, (v / 255) * 4 - 4));

function loadImage(src) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
}

async function pixels(src) {
  if (cache.has(src)) return cache.get(src);
  const im = await loadImage(src);
  const c = document.createElement("canvas"); c.width = im.width; c.height = im.height;
  const g = c.getContext("2d"); g.drawImage(im, 0, 0);
  const d = g.getImageData(0, 0, im.width, im.height).data;
  cache.set(src, d);
  return d;
}

function settings() {
  return {
    view: $("c-view").value, mode: $("c-mode").value, k: +$("c-k").value, minp: +$("c-minp").value,
    heat: $("c-heat").value, ranks: $("c-ranks").checked, cmap: $("c-cmap").value, alpha: +$("c-alpha").value,
    scale: $("c-scale").value, floor: +$("c-floor").value, outline: $("c-outline").checked,
  };
}

function normFn(s, shown) {
  if (s.scale === "linear") return (p) => Math.min(Math.max(p, 0), 1);
  let lo = Math.log10(s.floor), hi = 0;
  if (s.scale === "rel" && shown.length) {
    lo = Math.log10(Math.min(...shown.map((o) => o.p)) / 2);
    hi = Math.log10(Math.max(...shown.map((o) => o.p)));
    if (hi <= lo) hi = lo + 1e-6;
  }
  return (p) => (p <= 0 ? 0 : Math.min(Math.max((Math.log10(p) - lo) / (hi - lo), 0), 1));
}

async function draw() {
  if (!cur) return;
  const s = settings();
  const it = cur, v = it.views[s.view];
  $("c-mode").value === "topk" ? $("fs-topk").removeAttribute("disabled") : $("fs-topk").setAttribute("disabled", "");
  const shown = s.mode === "topk" ? v.places.filter((o) => o.p >= s.minp).slice(0, s.k) : [];
  const [gray, map] = await Promise.all([pixels(`data/img/${it.id}.jpg`), pixels(`data/map/${it.id}_${s.view}.png`)]);
  if (cur !== it) return;
  const W = it.w, H = it.h, cv = $("canvas");
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d"), out = g.createImageData(W, H), o = out.data;
  const lut = META.cmaps[s.cmap], norm = normFn(s, shown);
  let inBox = null;
  if (s.mode === "topk" && s.heat !== "full") {
    inBox = new Uint8Array(W * H);
    if (s.heat === "boxes") for (const pl of shown) {
      const [r0, c0, r1, c1] = pl.box.map(Math.round);
      for (let r = Math.max(r0, 0); r < Math.min(r1, H); r++) inBox.fill(1, r * W + Math.max(c0, 0), r * W + Math.min(c1, W));
    }
  }
  for (let i = 0, j = 0; i < W * H; i++, j += 4) {
    const y = gray[j];
    let a = 0, n = 0;
    const p = decode(map[j]);
    if (p > 0 && (!inBox || inBox[i]) && (s.scale !== "log" || p >= s.floor)) { n = norm(p); a = s.alpha * Math.pow(n, GAMMA); }
    const col = lut[Math.round(n * 255)];
    o[j] = (1 - a) * y + a * col[0]; o[j + 1] = (1 - a) * y + a * col[1]; o[j + 2] = (1 - a) * y + a * col[2]; o[j + 3] = 255;
  }
  g.putImageData(out, 0, 0);
  const lw = Math.max(1.5, W / 300);
  if (s.outline) {
    g.strokeStyle = "cyan"; g.lineWidth = lw;
    for (const poly of it.outline) { g.beginPath(); poly.forEach(([r, c], k) => (k ? g.lineTo(c, r) : g.moveTo(c, r))); g.closePath(); g.stroke(); }
  }
  g.strokeStyle = g.fillStyle = "#ff2a2a"; g.lineWidth = lw; g.font = `bold ${Math.round(W / 28)}px system-ui`;
  for (const pl of shown) {
    const [r0, c0, r1, c1] = pl.box;
    g.strokeRect(c0, r0, c1 - c0, r1 - r0);
    if (s.ranks) g.fillText(String(pl.rank), c1 + 3, r0 + W / 28);
  }
  drawBar(s, shown);
  info(s, it, v, shown);
}

function drawBar(s, shown) {
  const c = $("bar"), g = c.getContext("2d"), lut = META.cmaps[s.cmap];
  for (let y = 0; y < c.height; y++) { const col = lut[Math.round((1 - y / (c.height - 1)) * 255)]; g.fillStyle = `rgb(${col})`; g.fillRect(0, y, c.width, 1); }
  const L = $("bar-labels"); L.innerHTML = "";
  let ticks;
  if (s.scale === "linear") ticks = [0, 0.25, 0.5, 0.75, 1].map((p) => [p, p]);
  else {
    let lo = Math.log10(s.floor), hi = 0;
    if (s.scale === "rel" && shown.length) { lo = Math.log10(Math.min(...shown.map((o) => o.p)) / 2); hi = Math.log10(Math.max(...shown.map((o) => o.p))); }
    ticks = [];
    for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) ticks.push([(e - lo) / (hi - lo || 1), Math.pow(10, e)]);
    if (s.scale === "rel") ticks = [[0, Math.pow(10, lo)], [1, Math.pow(10, hi)]];
  }
  for (const [f, p] of ticks) {
    const sp = document.createElement("span");
    sp.style.top = `${(1 - f) * 100}%`; sp.textContent = p >= 0.01 || p === 0 ? +p.toFixed(3) : p.toExponential(0);
    L.appendChild(sp);
  }
}

function info(s, it, v, shown) {
  const attrs = it.lesion ? `lesion: density ${it.lesion_density}, radius ${it.radius_mm} mm` : "no lesion";
  $("info").textContent = `${it.image} · breast ${it.breast_density} · ${attrs} · prevalence ${Math.round(META.views[s.view] * 100)} % · max p ${v.max_p}`;
  const T = $("places");
  if (s.mode !== "topk") { T.innerHTML = ""; return; }
  T.innerHTML = "<tr><th>rank</th><th>p</th><th>box (mm)</th>" + (s.outline ? "<th>on lesion</th>" : "") + "</tr>" +
    (shown.map((o) => `<tr><td>${o.rank}</td><td>${o.p < 0.001 ? o.p.toExponential(1) : o.p.toFixed(3)}</td><td>${o.box_mm[0]} × ${o.box_mm[1]}</td>` +
      (s.outline ? `<td>${o.on_lesion === null ? "–" : o.on_lesion ? "yes" : "no"}</td>` : "") + "</tr>").join("") ||
      `<tr><td colspan="4">no place with p ≥ ${s.minp}</td></tr>`);
}

function fillSelect(sel, values, fmt = (x) => x) {
  for (const x of values) { const o = document.createElement("option"); o.value = x; o.textContent = fmt(x); sel.appendChild(o); }
}

function grid() {
  const f = { lesion: $("f-lesion").value, breast: $("f-breast").value, ldens: $("f-ldens").value, radius: $("f-radius").value };
  const items = DATA.filter((it) => (f.lesion === "" || it.lesion === +f.lesion) && (f.breast === "" || it.breast_density === f.breast) &&
    (f.ldens === "" || it.lesion_density === +f.ldens) && (f.radius === "" || it.radius_mm === +f.radius));
  $("count").textContent = `${items.length} of ${DATA.length} test images`;
  const G = $("grid"); G.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    b.innerHTML = `<img loading="lazy" src="data/img/${it.id}.jpg" alt=""><span class="tag">${it.breast_density}${it.lesion ? " · " + it.radius_mm + " mm" : " · none"}</span>`;
    b.onclick = () => { cur = it; document.querySelectorAll("#grid button").forEach((x) => x.classList.toggle("on", x === b)); draw(); };
    if (it === cur) b.classList.add("on");
    G.appendChild(b);
  }
  if (!cur && items.length) G.firstChild.click();
}

async function init() {
  const j = await (await fetch("data/items.json")).json();
  DATA = j.items; META = j.meta;
  fillSelect($("c-view"), Object.keys(META.views), (v) => `${Math.round(META.views[v] * 100)} %`);
  $("c-view").value = "p10";
  fillSelect($("c-cmap"), Object.keys(META.cmaps));
  fillSelect($("f-breast"), ["fatty", "scattered", "hetero", "dense"]);
  fillSelect($("f-ldens"), [...new Set(DATA.filter((x) => x.lesion).map((x) => x.lesion_density))].sort());
  fillSelect($("f-radius"), [...new Set(DATA.filter((x) => x.lesion).map((x) => x.radius_mm))].sort(), (x) => `${x} mm`);
  for (const id of ["f-lesion", "f-breast", "f-ldens", "f-radius"]) $(id).onchange = grid;
  for (const el of document.querySelectorAll(".controls select, .controls input")) el.oninput = () => {
    $("o-k").textContent = $("c-k").value; $("o-alpha").textContent = (+$("c-alpha").value).toFixed(2); draw();
  };
  const q = new URLSearchParams(location.search);   // e.g. ?id=012&view=p10&mode=topk&k=5&minp=0&heat=full&outline=1
  for (const [k, id] of [["view", "c-view"], ["mode", "c-mode"], ["k", "c-k"], ["minp", "c-minp"], ["heat", "c-heat"],
                         ["cmap", "c-cmap"], ["alpha", "c-alpha"], ["scale", "c-scale"], ["floor", "c-floor"]]) if (q.has(k)) $(id).value = q.get(k);
  if (q.has("outline")) $("c-outline").checked = q.get("outline") === "1";
  $("o-k").textContent = $("c-k").value; $("o-alpha").textContent = (+$("c-alpha").value).toFixed(2);
  cur = DATA.find((x) => x.id === q.get("id")) || null;
  grid();
  if (cur) draw();
}
init();
