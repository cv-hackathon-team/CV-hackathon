// DepthGuard 360 front end. No build step: edit and refresh.
const $ = (s) => document.querySelector(s);
const state = { scale: 4, methods: new Set(["Bicubic", "Joint bilateral", "Guided filter"]), rgb: null, depth: null };

// ---------- the six modules (edit text here) ----------
const MODULES = [
  ["Depth Rescue", "The doctor", "Looks at the damaged depth first and labels the problem: holes, noise or broken edges."],
  ["DepthGuard", "The trust checker", "Scores every pixel. Trusted measurements are kept; doubtful ones get the most repair work."],
  ["Occlusion-X", "Who's in front", "At an object edge, each pixel picks the front or the back surface. It never averages the two."],
  ["Texture Trap", "The pattern spotter", "Bricks, tiles and stripes are colour edges, not depth edges. They're ignored."],
  ["Temporal Repair", "The video helper", "For video, borrows missing depth from neighbouring frames.", "Planned"],
  ["Geometry First", "The final checker", "Keeps walls and floors flat while leaving real boundaries sharp."],
];
$("#cards").innerHTML = MODULES.map(([t, s, p, off], i) => `
  <div class="glass card reveal" style="transition-delay:${i * 70}ms">
    <div class="n">0${i + 1} · ${s}</div><h3>${t}</h3><p>${p}</p>${off ? `<div class="off">${off}</div>` : ""}
  </div>`).join("");

// ---------- scroll reveal + scroll-lit text ----------
const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && e.target.classList.add("in")), { threshold: 0.15 });
const observe = () => document.querySelectorAll(".reveal:not(.in)").forEach((el) => io.observe(el));
observe();
const lit = $("[data-lit]");
lit.innerHTML = lit.textContent.trim().split(/\s+/).map((w) => `<span>${w} </span>`).join("");
const words = [...lit.children];
addEventListener("scroll", () => {
  const r = lit.getBoundingClientRect();
  const p = Math.min(1, Math.max(0, (innerHeight * 0.85 - r.top) / (r.height + innerHeight * 0.35)));
  words.forEach((w, i) => w.classList.toggle("on", i / words.length < p));
}, { passive: true });

// ---------- controls ----------
$("#scale").onclick = (e) => {
  const b = e.target.closest("button"); if (!b) return;
  [...$("#scale").children].forEach((x) => x.classList.toggle("on", x === b));
  state.scale = +b.dataset.v;
};
fetch("/api/methods").then((r) => r.json()).then((list) => {
  $("#methods").innerHTML = list.map(({ name }) => name === "DepthGuard 360"
    ? `<span class="chip lock">${name}</span>`
    : `<span class="chip ${state.methods.has(name) ? "on" : ""}" data-n="${name}">${name}</span>`).join("");
});
$("#methods").onclick = (e) => {
  const c = e.target.closest("[data-n]"); if (!c) return;
  const n = c.dataset.n;
  state.methods.has(n) ? state.methods.delete(n) : state.methods.add(n);
  c.classList.toggle("on");
};
for (const k of ["rgb", "depth"]) {
  const drop = $(`#drop-${k}`), inp = $(`#f-${k}`);
  const set = (f) => { state[k] = f; $(`#n-${k}`).textContent = f.name; drop.classList.add("has"); };
  inp.onchange = () => inp.files[0] && set(inp.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && set(e.dataTransfer.files[0]); };
}

// ---------- run ----------
function form(sample) {
  const fd = new FormData();
  fd.append("scale", state.scale);
  fd.append("methods", [...state.methods, "DepthGuard 360"].join(","));
  fd.append("sample", sample);
  if (!sample) { fd.append("rgb", state.rgb); fd.append("depth", state.depth); }
  return fd;
}
async function run(sample) {
  if (!sample && !(state.rgb && state.depth)) { $("#status").textContent = "Add both files first, or try the sample scene."; return; }
  state.lastSample = sample;
  const fd = form(sample);
  $("#run").disabled = $("#sample").disabled = true;
  $("#status").textContent = "Reconstructing…";
  try {
    const res = await fetch("/api/reconstruct", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    render(await res.json());
    $("#sweep-table").innerHTML = "";
    $("#sweep-status").textContent = "Runs the same scene at all three factors.";
    $("#status").textContent = "";
  } catch (err) {
    $("#status").textContent = "Something went wrong: " + err.message.slice(0, 120);
  } finally {
    $("#run").disabled = $("#sample").disabled = false;
  }
}
$("#run").onclick = () => run(false);

// ---------- 2x / 4x / 8x sweep ----------
const SWEEP = ["RMSE", "Boundary F1", "ms", "MB"];
$("#sweep").onclick = async () => {
  $("#sweep").disabled = true;
  $("#sweep-status").textContent = "Running 2×, 4× and 8×…";
  try {
    const res = await fetch("/api/scales", { method: "POST", body: form(state.lastSample) });
    if (!res.ok) throw new Error(await res.text());
    const { methods, scales } = await res.json();
    const S = Object.keys(scales);
    $("#sweep-table").innerHTML =
      `<tr><th></th>${SWEEP.map((k) => `<th colspan="${S.length}">${LABEL[k] || k}</th>`).join("")}</tr>` +
      `<tr><th>Method</th>${SWEEP.map(() => S.map((s) => `<th>${s}×</th>`).join("")).join("")}</tr>` +
      methods.map((m) => `<tr class="${m === "DepthGuard 360" ? "me" : ""}"><td>${m}</td>${SWEEP.map((k) => S.map((s) => {
        const col = methods.map((n) => scales[s][n][k]);
        const b = (HIGHER.has(k) ? Math.max : Math.min)(...col);
        return `<td class="${scales[s][m][k] === b ? "best" : ""}">${fmt(k, scales[s][m][k])}</td>`;
      }).join("")).join("")}</tr>`).join("");
    $("#sweep-status").textContent = "Same scene, same damage. Best value per column in green.";
  } catch (err) {
    $("#sweep-status").textContent = "Something went wrong: " + err.message.slice(0, 120);
  } finally {
    $("#sweep").disabled = false;
  }
};

// real model or stand-in?
fetch("/api/status").then((r) => r.json()).then((s) => {
  $("#model-badge").textContent = s.model === "real"
    ? `DepthGuard 360: trained model on ${s.device}` : "DepthGuard 360: rule-based stand-in (no checkpoint)";
});
$("#sample").onclick = () => run(true);

// ---------- render ----------
const KEYS = ["RMSE", "MAE", "Edge RMSE", "Boundary F1", "PSNR", "SSIM", "ms", "MB"];
const HIGHER = new Set(["PSNR", "SSIM", "Boundary F1"]);
const LABEL = { ms: "Time (ms)", MB: "Peak memory (MB)" };
const DIGITS = { SSIM: 3, "Boundary F1": 3, ms: 0, MB: 1 };
const fmt = (k, v) => (+v).toFixed(DIGITS[k] ?? 3);
const tile = (img, t, s, badge, extra = "") => `
  <div class="glass tile"><img src="${img}" alt="${t}" />
  <div class="tile-c"><div><div class="tile-t">${t}</div><div class="tile-s">${s}</div></div>${badge || ""}</div>${extra}</div>`;

function render(d) {
  $("#out").classList.remove("hidden");
  const R = d.results;
  const best = Object.fromEntries(KEYS.map((k) => [k, (HIGHER.has(k) ? Math.max : Math.min)(...R.map((r) => r[k]))]));

  $("#inputs").innerHTML =
    tile(d.inputs.rgb, "RGB guide", `${d.size[1]} × ${d.size[0]}`) +
    tile(d.inputs.lowres, "Damaged input", `${d.scale}× smaller, noise + holes`) +
    tile(d.inputs.truth, "Ground truth", "What we're aiming for");

  $("#scores").innerHTML = `<tr><th>Method</th>${KEYS.map((k) => `<th>${LABEL[k] || k}</th>`).join("")}</tr>` +
    R.map((r) => `<tr class="${r.name === "DepthGuard 360" ? "me" : ""}"><td>${r.name}</td>${KEYS.map((k) =>
      `<td class="${r[k] === best[k] ? "best" : ""}">${fmt(k, r[k])}</td>`).join("")}</tr>`).join("");

  $("#results-grid").innerHTML = R.map((r) => tile(r.image, r.name, r.group,
    `<span class="badge ${r["Edge RMSE"] === best["Edge RMSE"] ? "best" : ""}">Edge ${r["Edge RMSE"].toFixed(2)}</span>`)).join("");

  $("#maps").innerHTML =
    tile(d.maps.confidence, "Confidence map", "DepthGuard · bright = trusted") +
    tile(d.maps.edges, "Edge decisions", "Texture Trap", "",
      `<div class="legend"><span><i style="background:#28b45a"></i>Real depth edge, kept</span><span><i style="background:#eb5a5a"></i>Texture, ignored</span></div>`);

  const bic = R.find((r) => r.name === "Bicubic") || R[0];
  $("#w-a").src = bic.image;
  $("#w-b").src = R.find((r) => r.name === "DepthGuard 360").image;
  $("#out").scrollIntoView({ behavior: "smooth", block: "start" });
}

// before/after wipe
const wipe = $("#wipe");
const move = (x) => {
  const r = wipe.getBoundingClientRect();
  const p = Math.min(100, Math.max(0, ((x - r.left) / r.width) * 100));
  $("#w-b").style.clipPath = `inset(0 0 0 ${p}%)`;
  wipe.querySelector(".wipe-bar").style.left = p + "%";
};
wipe.onpointerdown = (e) => { wipe.setPointerCapture(e.pointerId); move(e.clientX); };
wipe.onpointermove = (e) => e.buttons && move(e.clientX);

// ---------- benchmark table from outputs/results.csv ----------
fetch("/api/results").then((r) => r.json()).then(({ rows }) => {
  if (!rows.length) { $("#bench").outerHTML = `<p class="empty">No results.csv found yet. Run the evaluation script and it will appear here.</p>`; return; }
  const cols = Object.keys(rows[0]);
  $("#bench").innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>` +
    rows.map((r) => `<tr>${cols.map((c) => { const v = r[c]; const n = parseFloat(v);
      return `<td>${isNaN(n) || /^[a-z]/i.test(v) ? v : n.toFixed(3)}</td>`; }).join("")}</tr>`).join("");
});
