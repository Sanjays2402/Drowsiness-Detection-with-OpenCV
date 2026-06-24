/**
 * Drowsiness Detection — browser demo.
 *
 * Faithful JS port of the Python package's detection pipeline:
 *   - MediaPipe FaceMesh landmarks (same six EAR indices per eye)
 *   - Eye Aspect Ratio:  EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
 *   - Finite-state machine: AWAKE --(EAR<thr for N frames)--> DROWSY
 *                           DROWSY --(EAR>=thr for M frames)--> AWAKE
 *
 * Defaults mirror drowsiness/detector.py:DrowsinessConfig
 *   ear_threshold = 0.25, closed_frames_to_alarm = 20, open_frames_to_clear = 5.
 *
 * Everything runs client-side. No frame ever leaves the browser.
 *
 * NOTE: @mediapipe/face_mesh ships as a UMD bundle (not an ES module), so it is
 * loaded via a classic <script> tag in index.html and exposed as window.FaceMesh.
 */

const FaceMesh = window.FaceMesh;

// Same indices as drowsiness/landmarks.py (_MP_LEFT_EYE / _MP_RIGHT_EYE):
// order = outer, upper-outer, upper-inner, inner, lower-inner, lower-outer.
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [263, 387, 385, 362, 380, 373];

// ---- EAR (ports drowsiness/ear.py) ---------------------------------------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function eyeAspectRatio(eye) {
  const [p1, p2, p3, p4, p5, p6] = eye;
  const horizontal = dist(p1, p4);
  if (horizontal <= 1e-6) return 0.0;
  return (dist(p2, p6) + dist(p3, p5)) / (2.0 * horizontal);
}
const averageEar = (l, r) => (eyeAspectRatio(l) + eyeAspectRatio(r)) / 2.0;

// ---- finite-state machine (ports DrowsinessDetector._update_state) -------
class DrowsinessFSM {
  constructor(cfg) {
    this.cfg = cfg;
    this.state = "awake";
    this.closedFrames = 0;
    this.openFrames = 0;
  }
  reset() { this.state = "awake"; this.closedFrames = 0; this.openFrames = 0; }
  /** @returns {boolean} true exactly on an AWAKE->DROWSY transition */
  update(ear) {
    const c = this.cfg;
    let transitioned = false;
    if (ear < c.earThreshold) {
      this.closedFrames += 1;
      this.openFrames = 0;
      if (this.state === "awake" && this.closedFrames >= c.closedFramesToAlarm) {
        this.state = "drowsy";
        transitioned = true;
      }
    } else {
      this.openFrames += 1;
      if (this.state === "drowsy" && this.openFrames >= c.openFramesToClear) {
        this.state = "awake";
        this.closedFrames = 0;
      } else if (this.state === "awake") {
        this.closedFrames = 0;
      }
    }
    return transitioned;
  }
}

// ---- tiny WebAudio alarm (no asset needed; mirrors the alarm trigger) ----
class Beeper {
  constructor() { this.ctx = null; this.lastAt = 0; this.cooldownMs = 3000; }
  trigger() {
    if (typeof AudioContext === "undefined" && typeof webkitAudioContext === "undefined") return;
    const now = performance.now();
    if (now - this.lastAt < this.cooldownMs) return;
    this.lastAt = now;
    this.ctx = this.ctx || new (window.AudioContext || window.webkitAudioContext)();
    const t = this.ctx.currentTime;
    for (let i = 0; i < 3; i++) {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = "square";
      osc.frequency.value = 880;
      const start = t + i * 0.22;
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(0.25, start + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.18);
      osc.connect(gain).connect(this.ctx.destination);
      osc.start(start);
      osc.stop(start + 0.2);
    }
  }
}

// ---- DOM refs ------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const octx = overlay.getContext("2d");
const earGraph = $("earGraph");
const gctx = earGraph.getContext("2d");

const els = {
  banner: $("banner"), bannerText: $("banner-text"),
  ear: $("earValue"), state: $("stateValue"), closed: $("closedValue"), fps: $("fpsValue"),
  startCover: $("startCover"), startBtn: $("startBtn"), loadCover: $("loadCover"), loadText: $("loadText"),
  stopBtn: $("stopBtn"), eventCount: $("eventCount"), minEar: $("minEar"), uptime: $("uptime"),
  threshold: $("threshold"), thresholdOut: $("thresholdOut"),
  closedFrames: $("closedFrames"), closedFramesOut: $("closedFramesOut"),
  openFrames: $("openFrames"), openFramesOut: $("openFramesOut"),
  alarmToggle: $("alarmToggle"), meshToggle: $("meshToggle"),
};

// ---- config wired to the sliders -----------------------------------------
const cfg = { earThreshold: 0.25, closedFramesToAlarm: 20, openFramesToClear: 5 };
const fsm = new DrowsinessFSM(cfg);
const beeper = new Beeper();

function bindRange(input, out, key, parse) {
  const sync = () => { const v = parse(input.value); cfg[key] = v; out.textContent = input.value; };
  input.addEventListener("input", sync); sync();
}
bindRange(els.threshold, els.thresholdOut, "earThreshold", parseFloat);
bindRange(els.closedFrames, els.closedFramesOut, "closedFramesToAlarm", (v) => parseInt(v, 10));
bindRange(els.openFrames, els.openFramesOut, "openFramesToClear", (v) => parseInt(v, 10));

// ---- session stats + EAR history -----------------------------------------
const session = { events: 0, minEar: Infinity, startedAt: 0, lastLm: null };
const earHistory = new Array(160).fill(null);

// ---- rendering -----------------------------------------------------------
function fitCanvas(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(r.width * dpr);
  cv.height = Math.round(r.height * dpr);
  return dpr;
}

function drawOverlay(landmarks, ear, drowsy) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!landmarks) return;
  const W = overlay.width, H = overlay.height;

  if (els.meshToggle.checked) {
    octx.fillStyle = "rgba(91,140,255,0.35)";
    for (const p of landmarks) { octx.beginPath(); octx.arc(p.x * W, p.y * H, 1.1, 0, Math.PI * 2); octx.fill(); }
  }

  const color = drowsy ? "#ff5a5f" : ear < cfg.earThreshold ? "#ffb454" : "#2fd47a";
  for (const idx of [LEFT_EYE, RIGHT_EYE]) {
    octx.beginPath();
    idx.forEach((i, k) => {
      const p = landmarks[i];
      const x = p.x * W, y = p.y * H;
      k === 0 ? octx.moveTo(x, y) : octx.lineTo(x, y);
    });
    octx.closePath();
    octx.lineWidth = 2.2;
    octx.strokeStyle = color;
    octx.stroke();
    octx.fillStyle = color + "22";
    octx.fill();
  }
}

function drawGraph() {
  const dpr = fitCanvas(earGraph);
  const W = earGraph.width, H = earGraph.height;
  gctx.clearRect(0, 0, W, H);
  const min = 0.05, max = 0.42;
  const y = (e) => H - ((e - min) / (max - min)) * H;

  // threshold line
  gctx.strokeStyle = "rgba(255,180,84,0.55)";
  gctx.lineWidth = 1 * dpr;
  gctx.setLineDash([4 * dpr, 4 * dpr]);
  gctx.beginPath(); gctx.moveTo(0, y(cfg.earThreshold)); gctx.lineTo(W, y(cfg.earThreshold)); gctx.stroke();
  gctx.setLineDash([]);

  // EAR trace
  gctx.beginPath();
  let started = false;
  earHistory.forEach((e, i) => {
    if (e == null) { started = false; return; }
    const x = (i / (earHistory.length - 1)) * W;
    const yy = y(Math.max(min, Math.min(max, e)));
    started ? gctx.lineTo(x, yy) : gctx.moveTo(x, yy);
    started = true;
  });
  gctx.strokeStyle = "#5b8cff";
  gctx.lineWidth = 1.6 * dpr;
  gctx.stroke();
}

// ---- per-frame result handler --------------------------------------------
let frames = 0, fpsAt = performance.now();

function onResults(results) {
  const lm = results.multiFaceLandmarks && results.multiFaceLandmarks[0];
  let ear = 0, faceFound = !!lm;

  if (faceFound) {
    const left = LEFT_EYE.map((i) => lm[i]);
    const right = RIGHT_EYE.map((i) => lm[i]);
    ear = averageEar(left, right);
    const transitioned = fsm.update(ear);
    if (transitioned) {
      session.events += 1;
      els.eventCount.textContent = session.events;
      if (els.alarmToggle.checked) beeper.trigger();
    }
    if (ear < session.minEar) { session.minEar = ear; els.minEar.textContent = ear.toFixed(3); }
    session.lastLm = lm;
  }

  // history + readouts
  earHistory.push(faceFound ? ear : null);
  earHistory.shift();
  els.ear.textContent = faceFound ? ear.toFixed(3) : "—";
  els.state.textContent = fsm.state === "drowsy" ? "DROWSY" : "AWAKE";
  els.state.style.color = fsm.state === "drowsy" ? "var(--drowsy)" : "var(--awake)";
  els.closed.textContent = fsm.closedFrames;

  // banner
  const drowsy = fsm.state === "drowsy";
  let cls = "banner--awake", txt = "Awake";
  if (!faceFound) { cls = "banner--noface"; txt = "No face detected"; }
  else if (drowsy) { cls = "banner--drowsy"; txt = "DROWSY — wake up!"; }
  els.banner.className = "banner " + cls;
  els.bannerText.textContent = txt;
  overlay.parentElement.classList.toggle("is-drowsy", drowsy);

  drawOverlay(faceFound ? lm : null, ear, drowsy);

  // fps
  frames++;
  const now = performance.now();
  if (now - fpsAt >= 500) {
    els.fps.textContent = Math.round((frames * 1000) / (now - fpsAt));
    frames = 0; fpsAt = now;
  }
}

// ---- camera + mediapipe loop ---------------------------------------------
let faceMesh = null, stream = null, running = false, rafId = null;

async function start() {
  els.startCover.hidden = true;
  els.loadCover.hidden = false;

  try {
    els.loadText.textContent = "Requesting camera…";
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    els.loadCover.hidden = true;
    els.startCover.hidden = false;
    els.banner.className = "banner banner--noface";
    els.bannerText.textContent = "Camera blocked";
    els.startBtn.textContent = "Camera permission denied — retry";
    return;
  }

  els.loadText.textContent = "Loading FaceMesh model…";
  faceMesh = new FaceMesh({
    locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}`,
  });
  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  faceMesh.onResults(onResults);

  fitCanvas(overlay);
  await faceMesh.send({ image: video }); // warm up / triggers wasm fetch
  els.loadCover.hidden = true;

  running = true;
  session.startedAt = performance.now();
  els.stopBtn.disabled = false;
  fsm.reset();
  loop();
  tickUptime();
}

async function loop() {
  if (!running) return;
  if (video.readyState >= 2) {
    try { await faceMesh.send({ image: video }); } catch (_) { /* keep looping */ }
    drawGraph();
  }
  rafId = requestAnimationFrame(loop);
}

function stop() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  octx.clearRect(0, 0, overlay.width, overlay.height);
  els.stopBtn.disabled = true;
  els.startCover.hidden = false;
  els.startBtn.textContent = "Resume camera";
  els.banner.className = "banner banner--idle";
  els.bannerText.textContent = "Camera off";
  overlay.parentElement.classList.remove("is-drowsy");
}

function tickUptime() {
  if (!running) return;
  const s = Math.floor((performance.now() - session.startedAt) / 1000);
  els.uptime.textContent = s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
  setTimeout(tickUptime, 1000);
}

els.startBtn.addEventListener("click", start);
els.stopBtn.addEventListener("click", stop);
window.addEventListener("resize", () => { fitCanvas(overlay); drawGraph(); });
drawGraph();
