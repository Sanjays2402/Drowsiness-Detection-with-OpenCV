# Live EAR Demo (web)

A browser **drowsiness-detection demo** — the same Eye Aspect Ratio (EAR)
pipeline as the Python package, running entirely client-side on your webcam via
MediaPipe FaceMesh. Nothing is uploaded; every frame is processed in the tab.

## Run it

It's a static site — just serve the folder:

```bash
cd web
python3 -m http.server 8730
# open http://localhost:8730 and click "Start camera"
```

> A webcam + camera permission are required. Browsers only allow `getUserMedia`
> on `https://` or `http://localhost`, so serving locally (not `file://`) is
> required. The FaceMesh model is fetched from the jsDelivr CDN on first run.

## Faithful to the Python detector

The JS port mirrors `drowsiness/` exactly so the demo behaves like the library:

| Concept | Python source | Web port |
|---|---|---|
| EAR formula | `drowsiness/ear.py` | `eyeAspectRatio()` in `app.js` |
| Eye landmark indices | `_MP_LEFT_EYE` / `_MP_RIGHT_EYE` in `landmarks.py` | `LEFT_EYE` / `RIGHT_EYE` |
| FSM (AWAKE↔DROWSY) | `DrowsinessDetector._update_state` | `DrowsinessFSM.update()` |
| Defaults | `DrowsinessConfig` (`0.25`, `20`, `5`) | slider defaults |

The threshold, frames-to-alarm and frames-to-clear are exposed as live sliders
so you can see the state machine react in real time. The EAR trace graph plots
the ratio against the threshold line; an AWAKE→DROWSY transition flashes the
banner and (optionally) sounds a WebAudio alarm with the same cooldown idea as
`drowsiness/alarm.py`.

## What's in here

| File | Purpose |
|---|---|
| `index.html` | Viewport, readouts, control panel |
| `style.css` | Dark premium UI, responsive, reduced-motion aware |
| `app.js` | Camera + FaceMesh loop, EAR, FSM, overlay & graph rendering |

## Relationship to the package

This is a **showcase surface**, not a replacement. The Python package remains the
source of truth (it's the IEEE-published implementation with the CLI, batch
`analyze` mode and tests). The web demo reuses the same math so what you see in
the browser matches what the library computes from a video file.
