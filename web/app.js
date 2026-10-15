const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const S = 256;
const API = "https://ry2xzlk491.execute-api.us-east-2.amazonaws.com";
const onAws = () => document.querySelector("input[name=runon]:checked")?.value === "aws";

let session = null;
let current = null; // { rgb: ImageData, height: Uint8ClampedArray|null, truth: Uint8ClampedArray|null, label }
let runId = 0;      // guards against a slow network answer overwriting a newer one
let result = null;  // { maps: Float32Array, picks: [...] , inst: Int32Array }

async function loadImageData(url) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const c = new OffscreenCanvas(S, S);
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0, S, S);
  return g.getImageData(0, 0, S, S);
}

async function loadTote(seed) {
  const [rgb, hgt, tru] = await Promise.all(["rgb", "height", "suction"].map((k) => loadImageData(`samples/${seed}_${k}.png`)));
  const chan = (d) => { const o = new Uint8ClampedArray(S * S); for (let i = 0; i < S * S; i++) o[i] = d.data[i * 4]; return o; };
  current = { rgb, height: chan(hgt), truth: chan(tru), label: `Tote ${seed}` };
  document.querySelectorAll(".totes button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.seed == seed));
  await run();
}

const toPng = (imageData) => {
  const c = new OffscreenCanvas(S, S);
  c.getContext("2d").putImageData(imageData, 0, 0);
  return c.convertToBlob({ type: "image/png" });
};

// Same model, served by AWS Lambda behind API Gateway. The browser sends the RGB image and the
// 8-bit height image it already has; the service returns ranked picks (not the full maps).
async function runAws(noDepth, id) {
  const form = new FormData();
  form.append("rgb", await toPng(current.rgb), "rgb.png");
  if (!noDepth) {
    const h = new ImageData(S, S);
    for (let i = 0; i < S * S; i++) h.data.set([current.height[i], current.height[i], current.height[i], 255], i * 4);
    form.append("height", await toPng(h), "height.png");
  }
  $("#status").textContent = `${current.label}: sending to AWS Lambda…`;
  const t0 = performance.now();
  const r = await fetch(`${API}/v1/picks?k=5`, { method: "POST", body: form });
  const body = await r.json();
  const rtt = performance.now() - t0;
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  if (id !== runId) return false;
  result = { maps: null, inst: null, picks: body.picks.map((p) => ({ x: p.x, y: p.y, score: p.score })) };
  const cold = rtt > 1500 ? " (includes a cold start)" : "";
  $("#status").textContent = `${current.label}: ${noDepth ? "RGB only" : "RGB + depth"} on AWS Lambda (arm64, us-east-2). Model and grasp planning ${body.latency_ms.toFixed(0)} ms on the server, ${rtt.toFixed(0)} ms round trip${cold}, ${body.items_detected} items found.`;
  return true;
}

async function run() {
  if (!current) return;
  const noDepth = $("#nodepth").checked || !current.height;
  const id = ++runId;
  if (onAws()) {
    try { if (!(await runAws(noDepth, id))) return; } catch (e) {
      if (id !== runId) return;
      result = null;
      $("#status").textContent = `The AWS endpoint did not answer (${e.message}). It may have been taken down to save cost; switch back to "This browser".`;
      $("#picks").innerHTML = "";
      draw();
      return;
    }
    renderPicks();
    draw();
    return;
  }
  if (!session) return;
  const x = new Float32Array(4 * S * S);
  const d = current.rgb.data;
  for (let i = 0; i < S * S; i++) {
    x[i] = d[i * 4] / 255;
    x[S * S + i] = d[i * 4 + 1] / 255;
    x[2 * S * S + i] = d[i * 4 + 2] / 255;
    x[3 * S * S + i] = noDepth ? 0 : current.height[i] / 255;
  }
  const t0 = performance.now();
  const out = await session.run({ rgbd: new ort.Tensor("float32", x, [1, 4, S, S]) });
  const ms = performance.now() - t0;
  const maps = out.maps.data;
  const inst = instances(maps);
  result = { maps, inst, picks: rankPicks(maps.subarray(3 * S * S, 4 * S * S), inst) };
  $("#status").textContent = `${current.label}: ${noDepth ? "RGB only" : "RGB + depth"}, inference ${ms.toFixed(0)} ms in ONNX Runtime Web, ${result.inst.count} items found.`;
  renderPicks();
  draw();
}

function blur(src) {
  const a = new Float32Array(S * S), b = new Float32Array(S * S), k = [1, 4, 6, 4, 1];
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let s = 0, w = 0;
    for (let j = -2; j <= 2; j++) { const xx = x + j; if (xx >= 0 && xx < S) { s += src[y * S + xx] * k[j + 2]; w += k[j + 2]; } }
    a[y * S + x] = s / w;
  }
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let s = 0, w = 0;
    for (let j = -2; j <= 2; j++) { const yy = y + j; if (yy >= 0 && yy < S) { s += a[yy * S + x] * k[j + 2]; w += k[j + 2]; } }
    b[y * S + x] = s / w;
  }
  return b;
}

// Best point per item first: after choosing a point, suppress its neighbourhood and its whole item,
// so the list offers alternatives rather than five spots on the same box.
function rankPicks(suction, inst, k = 5, nms = 12) {
  const s = blur(suction), work = Float32Array.from(s), picks = [];
  for (let n = 0; n < k; n++) {
    let best = -1, bi = -1;
    for (let i = 0; i < work.length; i++) if (work[i] > best) { best = work[i]; bi = i; }
    if (best < 0.3) break; // below this the robot should not try; a person takes the tote
    const x = bi % S, y = Math.floor(bi / S);
    picks.push({ x, y, score: s[bi] });
    const id = inst[bi];
    if (id >= 0) for (let i = 0; i < work.length; i++) if (inst[i] === id) work[i] = 0;
    for (let yy = Math.max(0, y - nms); yy <= Math.min(S - 1, y + nms); yy++)
      for (let xx = Math.max(0, x - nms); xx <= Math.min(S - 1, x + nms); xx++)
        if ((xx - x) ** 2 + (yy - y) ** 2 <= nms * nms) work[yy * S + xx] = 0;
  }
  return picks;
}

function instances(maps) {
  const lab = new Int32Array(S * S).fill(-1);
  const interior = (i) => maps[S * S + i] > maps[i] && maps[S * S + i] > maps[2 * S * S + i];
  let count = 0;
  const stack = [];
  for (let i = 0; i < S * S; i++) {
    if (lab[i] !== -1 || !interior(i)) continue;
    const members = [];
    stack.push(i); lab[i] = count;
    while (stack.length) {
      const p = stack.pop(); members.push(p);
      const x = p % S, y = (p / S) | 0;
      for (const q of [x > 0 ? p - 1 : -1, x < S - 1 ? p + 1 : -1, y > 0 ? p - S : -1, y < S - 1 ? p + S : -1])
        if (q >= 0 && lab[q] === -1 && interior(q)) { lab[q] = count; stack.push(q); }
    }
    if (members.length < 40) { for (const p of members) lab[p] = -2; } else count++;
  }
  lab.count = count;
  return lab;
}

function renderPicks() {
  const ol = $("#picks");
  if (!result.picks.length) { ol.innerHTML = "<li><span></span><span>No surface scores high enough to try. A real cell would ask a person to look at this tote.</span></li>"; return; }
  ol.innerHTML = result.picks.map((p) => {
    let verdict = "";
    if (current.truth) {
      const ok = current.truth[p.y * S + p.x] > 127;
      verdict = `<span class="verdict ${ok ? "ok" : "bad"}">${ok ? "✓ seals and lifts" : "✗ would fail"}</span>`;
    }
    return `<li><span><span class="score">${Math.round(p.score * 100)}%</span><br><span class="meta">at (${p.x}, ${p.y})</span></span>${verdict}</li>`;
  }).join("");
}

function hue(i) { return `hsl(${(i * 137.5) % 360} 70% 55%)`; }

function draw() {
  const c = $("#view"), g = c.getContext("2d");
  const W = c.width, k = W / S;
  const base = new OffscreenCanvas(S, S);
  base.getContext("2d").putImageData(current.rgb, 0, 0);
  g.imageSmoothingEnabled = true;
  g.drawImage(base, 0, 0, W, W);
  if (!result) return;
  const mode = document.querySelector("input[name=ov]:checked").value;
  const needsMaps = mode === "suction" || mode === "items";
  $("#ovNote").hidden = !(needsMaps && !result.maps);
  if (mode !== "none" && !(needsMaps && !result.maps)) {
    const ov = new ImageData(S, S);
    for (let i = 0; i < S * S; i++) {
      let r = 0, gg = 0, b = 0, a = 0;
      if (mode === "suction") {
        const v = result.maps[3 * S * S + i];
        if (v > 0.15) { r = 30; gg = 220; b = 120; a = Math.min(200, v * 230); }
      } else if (mode === "items") {
        const l = result.inst[i];
        if (l >= 0) { const h = (l * 137.5) % 360; [r, gg, b] = hsl2rgb(h, 0.7, 0.55); a = 120; }
        else if (result.maps[2 * S * S + i] > 0.4) { r = gg = b = 255; a = 200; }
      } else if (mode === "truth" && current.truth) {
        if (current.truth[i] > 127) { r = 255; gg = 210; b = 40; a = 170; }
      }
      ov.data.set([r, gg, b, a], i * 4);
    }
    const oc = new OffscreenCanvas(S, S);
    oc.getContext("2d").putImageData(ov, 0, 0);
    g.drawImage(oc, 0, 0, W, W);
  }
  result.picks.forEach((p, n) => {
    const x = (p.x + 0.5) * k, y = (p.y + 0.5) * k;
    g.lineWidth = 3;
    g.strokeStyle = "#fff";
    g.beginPath(); g.arc(x, y, 13, 0, Math.PI * 2); g.stroke();
    g.fillStyle = n === 0 ? css("--cobalt") || "#2556c9" : "rgba(20,32,41,.85)";
    g.beginPath(); g.arc(x, y, 11, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#fff";
    g.font = "700 13px Archivo, sans-serif";
    g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(String(n + 1), x, y + 1);
  });
}

function hsl2rgb(h, s, l) {
  const f = (n) => { const k = (n + h / 30) % 12; const a = s * Math.min(l, 1 - l); return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))); };
  return [f(0), f(8), f(4)];
}

async function onUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const bmp = await createImageBitmap(file);
  const c = new OffscreenCanvas(S, S), g = c.getContext("2d");
  const side = Math.min(bmp.width, bmp.height);
  g.drawImage(bmp, (bmp.width - side) / 2, (bmp.height - side) / 2, side, side, 0, 0, S, S);
  current = { rgb: g.getImageData(0, 0, S, S), height: null, truth: null, label: "Your photo" };
  document.querySelectorAll(".totes button").forEach((b) => b.setAttribute("aria-pressed", "false"));
  await run();
}

async function loadResults() {
  try {
    const [clear, seg] = await Promise.all(["data/clear_tote.json", "data/seg_metrics.json"].map((u) => fetch(u).then((r) => r.json())));
    const names = Object.keys(clear);
    const isOracle = (n) => n.startsWith("oracle");
    const best = names.filter((n) => !isOracle(n)).reduce((a, b) => (clear[a].items_cleared > clear[b].items_cleared ? a : b));
    const t = clear[names[0]];
    $("#clearNote").textContent += ` ${t.totes} unseen totes, ${t.items.toLocaleString()} items.`;
    $("#clearTable tbody").innerHTML = names.map((n) => {
      const r = clear[n];
      const pct = (v) => `${Math.round(v * 100)}%`;
      return `<tr class="${n === best ? "best" : isOracle(n) ? "ceiling" : ""}"><td>${isOracle(n) ? "Ceiling: perfect knowledge of the ground truth" : n}</td><td>${pct(r.pick_success_rate)}<span class="barfill" style="width:${Math.round(r.pick_success_rate * 120)}px"></span></td><td>${pct(r.first_pick_success)}</td><td>${pct(r.items_cleared)}</td></tr>`;
    }).join("");
    const rows = [
      ["Item mask IoU (RGB-D)", seg.rgbd.iou_item],
      ["Suction average precision (RGB-D)", seg.rgbd.suction_ap],
      ["Top-1 pick precision (RGB-D)", seg.rgbd.top1_pick_precision],
      ["Top-1 pick precision (RGB only)", seg.rgb_only.top1_pick_precision],
    ];
    $("#segTable tbody").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${(v * 100).toFixed(1)}%</td></tr>`).join("");
  } catch {
    $("#clearNote").textContent = "Benchmark results are not available in this build.";
  }
}

async function boot() {
  const index = await (await fetch("samples/index.json")).json();
  $("#totes").innerHTML = index.map((t) =>
    `<button type="button" role="listitem" data-seed="${t.seed}" aria-pressed="false" aria-label="Tote ${t.seed}, ${t.items} items"><img src="samples/${t.seed}_rgb.png" alt="" loading="lazy"></button>`).join("");
  $("#totes").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) loadTote(b.dataset.seed); });
  document.querySelectorAll("input[name=ov]").forEach((r) => r.addEventListener("change", () => current && draw()));
  $("#nodepth").addEventListener("change", run);
  document.querySelectorAll("input[name=runon]").forEach((r) => r.addEventListener("change", run));
  $("#file").addEventListener("change", onUpload);
  loadResults();
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create("model/picknet.onnx", { executionProviders: ["wasm"] });
  } catch (e) {
    $("#status").textContent = "The model could not be loaded in this browser. The results below still apply.";
    console.error(e);
    return;
  }
  await loadTote(index[0].seed);
}

boot();
