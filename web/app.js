/**
 * Vigil — browser drowsiness detector.
 *
 * Detection is a faithful port of the Python package:
 *   - MediaPipe FaceMesh landmarks (same six EAR indices per eye)   [landmarks.py]
 *   - EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)                  [ear.py]
 *   - FSM: AWAKE --(EAR<thr for N frames)--> DROWSY                  [detector.py]
 *          DROWSY --(EAR>=thr for M frames)--> AWAKE
 *   - Defaults mirror DrowsinessConfig: 0.25 / 20 / 5.
 *
 * Everything runs client-side; no frame leaves the browser.
 * @mediapipe/face_mesh is a UMD bundle (not an ES module): loaded via a deferred
 * classic <script> in index.html, read off window lazily inside start().
 */

// Same indices as drowsiness/landmarks.py (_MP_LEFT_EYE / _MP_RIGHT_EYE).
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [263, 387, 385, 362, 380, 373];

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function eyeAspectRatio(eye) {
  const [p1, p2, p3, p4, p5, p6] = eye;
  const h = dist(p1, p4);
  if (h <= 1e-6) return 0;
  return (dist(p2, p6) + dist(p3, p5)) / (2 * h);
}
const averageEar = (l, r) => (eyeAspectRatio(l) + eyeAspectRatio(r)) / 2;

class FSM {
  constructor(cfg) { this.cfg = cfg; this.reset(); }
  reset() { this.state = "awake"; this.closed = 0; this.open = 0; }
  update(ear) {
    const c = this.cfg;
    let fired = false;
    if (ear < c.thr) {
      this.closed++; this.open = 0;
      if (this.state === "awake" && this.closed >= c.frames) { this.state = "drowsy"; fired = true; }
    } else {
      this.open++;
      if (this.state === "drowsy" && this.open >= c.clear) { this.state = "awake"; this.closed = 0; }
      else if (this.state === "awake") this.closed = 0;
    }
    return fired;
  }
}

class Alarm {
  constructor() { this.ctx = null; this.last = 0; this.cooldown = 3000; }
  trigger() {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    const now = performance.now();
    if (now - this.last < this.cooldown) return;
    this.last = now;
    this.ctx = this.ctx || new AC();
    const t = this.ctx.currentTime;
    for (let i = 0; i < 3; i++) {
      const o = this.ctx.createOscillator(), g = this.ctx.createGain();
      o.type = "sine"; o.frequency.value = 760;
      const s = t + i * 0.24;
      g.gain.setValueAtTime(0.0001, s);
      g.gain.exponentialRampToValueAtTime(0.2, s + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, s + 0.2);
      o.connect(g).connect(this.ctx.destination);
      o.start(s); o.stop(s + 0.22);
    }
  }
}

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const octx = overlay.getContext("2d");
const graph = $("graph");
const gctx = graph.getContext("2d");

const el = {
  start: $("startBtn"), stop: $("stopBtn"), load: $("loadCover"), loadText: $("loadText"),
  camStatus: $("camStatus"), statusLine: $("statusLine"), statusText: $("statusText"),
  rEar: $("rEar"), rClosed: $("rClosed"), rEvents: $("rEvents"), rFps: $("rFps"),
  thr: $("threshold"), thrOut: $("thrOut"), frames: $("frames"), framesOut: $("framesOut"),
  graphThr: $("graphThr"), alarm: $("alarmToggle"), mesh: $("meshToggle"),
};

const cfg = { thr: 0.25, frames: 20, clear: 5 };
const fsm = new FSM(cfg);
const alarm = new Alarm();

function bind(input, out, key, parse, fmt) {
  const sync = () => {
    cfg[key] = parse(input.value);
    out.textContent = fmt ? fmt(input.value) : input.value;
    if (key === "thr") el.graphThr.textContent = `threshold ${cfg.thr.toFixed(2)}`;
  };
  input.addEventListener("input", sync); sync();
}
bind(el.thr, el.thrOut, "thr", parseFloat, (v) => parseFloat(v).toFixed(2));
bind(el.frames, el.framesOut, "frames", (v) => parseInt(v, 10));

const sess = { events: 0 };
const HIST = 200;
const hist = new Array(HIST).fill(null);

function fit(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.max(1, Math.round(r.width * dpr));
  cv.height = Math.max(1, Math.round(r.height * dpr));
  return dpr;
}

// EAR graph on a light background
function drawGraph() {
  const dpr = fit(graph);
  const W = graph.width, H = graph.height;
  gctx.clearRect(0, 0, W, H);
  const min = 0.05, max = 0.45;
  const y = (e) => H - ((e - min) / (max - min)) * H;

  // threshold line
  const ty = y(cfg.thr);
  gctx.strokeStyle = "#d9b8a0";
  gctx.lineWidth = 1 * dpr;
  gctx.setLineDash([5 * dpr, 4 * dpr]);
  gctx.beginPath(); gctx.moveTo(0, ty); gctx.lineTo(W, ty); gctx.stroke();
  gctx.setLineDash([]);

  // trace
  gctx.beginPath();
  let started = false;
  hist.forEach((e, i) => {
    if (e == null) { started = false; return; }
    const x = (i / (HIST - 1)) * W;
    const yy = y(Math.max(min, Math.min(max, e)));
    started ? gctx.lineTo(x, yy) : gctx.moveTo(x, yy);
    started = true;
  });
  gctx.strokeStyle = fsm.state === "drowsy" ? "#c4452f" : "#b87a2b";
  gctx.lineWidth = 1.8 * dpr;
  gctx.lineJoin = "round";
  gctx.stroke();
}

function drawOverlay(lm) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!lm) return;
  const W = overlay.width, H = overlay.height;
  const drowsy = fsm.state === "drowsy";
  const col = drowsy ? "#ff7a5c" : "#ffd9a0";
  if (el.mesh.checked) {
    octx.fillStyle = "rgba(255,255,255,0.4)";
    for (const p of lm) { octx.beginPath(); octx.arc(p.x * W, p.y * H, 1, 0, Math.PI * 2); octx.fill(); }
  }
  for (const idx of [LEFT_EYE, RIGHT_EYE]) {
    octx.beginPath();
    idx.forEach((i, k) => { const p = lm[i]; const x = p.x * W, y = p.y * H; k ? octx.lineTo(x, y) : octx.moveTo(x, y); });
    octx.closePath();
    octx.lineWidth = 2.4; octx.strokeStyle = col; octx.stroke();
    octx.fillStyle = col + "30"; octx.fill();
  }
}

function setStatus(cls, text, cam) {
  el.statusLine.className = "status-line" + (cls ? " " + cls : "");
  el.statusText.textContent = text;
  if (cam !== undefined) el.camStatus.textContent = cam;
}

let frames = 0, fpsAt = performance.now();
function onResults(res) {
  const lm = res.multiFaceLandmarks && res.multiFaceLandmarks[0];
  let e = 0;
  if (lm) {
    e = averageEar(LEFT_EYE.map((i) => lm[i]), RIGHT_EYE.map((i) => lm[i]));
    if (fsm.update(e)) { sess.events++; el.rEvents.textContent = sess.events; if (el.alarm.checked) alarm.trigger(); }
  }
  hist.push(lm ? e : null); hist.shift();

  el.rEar.textContent = lm ? e.toFixed(3) : "—";
  el.rClosed.textContent = fsm.closed;

  const drowsy = fsm.state === "drowsy";
  if (!lm) setStatus("is-warn", "Looking for a face", "No face");
  else if (drowsy) setStatus("is-alert", "Drowsy — wake up", "Alert");
  else if (e < cfg.thr) setStatus("is-warn", "Eyes closing", "Live");
  else setStatus("is-ok", "Alert and awake", "Live");

  drawOverlay(lm);

  frames++;
  const now = performance.now();
  if (now - fpsAt >= 500) { el.rFps.innerHTML = `${Math.round((frames * 1000) / (now - fpsAt))}<small>fps</small>`; frames = 0; fpsAt = now; }
}

let mesh = null, stream = null, running = false, raf = null;

async function start() {
  el.start.hidden = true;
  el.load.hidden = false;
  try {
    el.loadText.textContent = "Requesting camera";
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    video.srcObject = stream; await video.play();
  } catch (err) {
    el.load.hidden = true; el.start.hidden = false;
    setStatus("is-alert", "Camera blocked", "Blocked");
    return;
  }

  el.loadText.textContent = "Loading model";
  const FaceMesh = window.FaceMesh;
  if (typeof FaceMesh !== "function") { el.loadText.textContent = "Model unavailable. Check the network."; return; }
  mesh = new FaceMesh({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}` });
  mesh.setOptions({ maxNumFaces: 1, refineLandmarks: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  mesh.onResults(onResults);

  fit(overlay);
  await mesh.send({ image: video });
  el.load.hidden = true;

  running = true;
  fsm.reset();
  el.stop.disabled = false;
  setStatus("is-ok", "Alert and awake", "Live");
  loop();
}

async function loop() {
  if (!running) return;
  if (video.readyState >= 2) { try { await mesh.send({ image: video }); } catch (_) {} }
  drawGraph();
  raf = requestAnimationFrame(loop);
}

function stop() {
  running = false;
  if (raf) cancelAnimationFrame(raf);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  octx.clearRect(0, 0, overlay.width, overlay.height);
  el.stop.disabled = true;
  el.start.hidden = false;
  el.start.childNodes[2] && (el.start.childNodes[2].textContent = " Resume camera");
  setStatus("", "Standby", "Camera off");
}

el.start.addEventListener("click", start);
el.stop.addEventListener("click", stop);
window.addEventListener("resize", drawGraph);
drawGraph();
