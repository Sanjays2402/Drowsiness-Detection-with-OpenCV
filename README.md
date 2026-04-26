# Drowsiness Detection

[![Tests](https://github.com/Sanjays2402/Drowsiness-Detection-with-OpenCV/actions/workflows/tests.yml/badge.svg)](https://github.com/Sanjays2402/Drowsiness-Detection-with-OpenCV/actions/workflows/tests.yml)
[![IEEE](https://img.shields.io/badge/IEEE-Published%20Paper-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/9532758)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-0097A7)
![License](https://img.shields.io/badge/license-MIT-green)

Real-time driver drowsiness detection using the **Eye Aspect Ratio (EAR)**. Originally an IEEE conference paper; this repo packages it as a production-shaped Python library with a CLI, two interchangeable landmark backends, a stateful detector with event hooks, and a headless batch-analysis mode.

> S. Santhanam et al., *"Real-Time Drowsiness Detection using Computer Vision"*, **2021 5th ICCMC (IEEE)**. [Read the paper →](https://ieeexplore.ieee.org/document/9532758)

---

## What's new vs. the original script

| | Old script | This package |
|---|---|---|
| Architecture | Single file, globals, blocking alarm | Stateful `DrowsinessDetector` with FSM (`AWAKE`/`DROWSY`), event callbacks |
| Alarm | Blocking — stalled the detection loop on every trigger | Background thread + cooldown (`Alarm`); never blocks frames |
| Alarm spam | Fired every frame while eyes closed | Fires once per AWAKE→DROWSY transition |
| Landmarks | dlib + `face_recognition` (CMake hell) | **MediaPipe FaceMesh by default** (pure-pip, faster, GPU-friendly) — `face_recognition` still available |
| API | None | `DrowsinessDetector`, `DrowsinessConfig`, `DrowsyEvent`, `DetectionResult` |
| CLI | Single mode | `drowsy run` (live) and `drowsy analyze` (headless batch → CSV/JSONL/JSON) |
| Tests | None | 14 pytest cases, FSM and alarm fully covered, runs in CI on Python 3.9/3.11/3.12 |
| Robustness | Crashed on missing alarm file; reset on lost detection | Soft-fails on missing audio; preserves state across brief detection dropouts |
| Visualization | EAR text only | EAR vs threshold, FPS counter, color-coded eye polylines, drowsy banner |

---

## Install

```bash
# Default install: package + OpenCV + NumPy
pip install -e .

# Recommended: add MediaPipe (no native build, very fast)
pip install -e ".[mediapipe,audio]"

# Or, original dlib-based backend (requires CMake)
pip install -e ".[legacy,audio]"
```

---

## Quick start

### CLI

```bash
# Live webcam detection with overlay
drowsy run 0

# Headless batch analysis of a video — emits CSV/JSONL/JSON report
drowsy analyze drive.mp4 --output-dir report/

# Tune sensitivity
drowsy run 0 --ear-threshold 0.22 --frame-count 24

# Use the legacy dlib backend
drowsy run 0 --backend face_recognition
```

Press **`q`** or **Esc** to quit live mode.

### Python API

```python
import cv2
from drowsiness import DrowsinessDetector, DrowsinessConfig

cfg = DrowsinessConfig(ear_threshold=0.23, closed_frames_to_alarm=24)

with DrowsinessDetector(cfg, on_event=lambda e: print("DROWSY!", e)) as detector:
 cap = cv2.VideoCapture(0)
 while True:
 ok, frame = cap.read()
 if not ok:
 break
 result = detector.process(frame)
 # result.state: EyeState.AWAKE | EyeState.DROWSY
 # result.ear: current Eye Aspect Ratio
 # result.event: DrowsyEvent or None (fires once per drowsy episode)
```

---

## How it works

### Eye Aspect Ratio

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)

 p2 p3
 •--------•
 / \
p1 • • p4
 \ /
 •--------•
 p6 p5
```

- **Eyes open** → EAR ≈ 0.30
- **Eyes closed** → EAR ≈ 0.05
- EAR below threshold for *N* consecutive frames → **drowsy event**

### Detection FSM

```
 EAR < threshold (N frames)
 ┌──────────┐ ─────────────────────────▶ ┌──────────┐
 │ AWAKE │ │ DROWSY │
 └──────────┘ ◀───────────────────────── └──────────┘
 EAR ≥ threshold (M frames)
```

The state machine de-bounces both directions: a brief blink doesn't trigger an alarm, and a single open frame doesn't immediately clear a drowsy state.

### Pipeline

```
Webcam / Video
 │
 ▼
 Landmark backend ──▶ 6 EAR points per eye
 (MediaPipe FaceMesh
 or face_recognition)
 │
 ▼
 EAR computation ──▶ average EAR
 │
 ▼
 Drowsy FSM ──▶ state, closed_frames, event?
 │
 ├─ Alarm (background, debounced)
 ├─ on_event callback (your code)
 └─ Visualization overlay
```

---

## Configuration

```python
DrowsinessConfig(
 ear_threshold=0.25, # below = closed eye
 closed_frames_to_alarm=20, # frames below threshold → DROWSY
 open_frames_to_clear=5, # frames above threshold → AWAKE
 alarm_sound="assets/alert1.mp3",
 alarm_cooldown_s=3.0, # min seconds between alarm playbacks
 backend="mediapipe", # or "face_recognition"
 enable_alarm=True,
)
```

| CLI flag | Default | Notes |
|---|---|---|
| `--ear-threshold` | `0.25` | Tune lower for darker/glasses-heavy footage. |
| `--frame-count` | `20` | At 30 FPS this is ~0.7s of closed eyes. |
| `--open-frames` | `5` | Clears the drowsy state after a sustained reopen. |
| `--alarm-cooldown` | `3.0` | Seconds. Prevents back-to-back alarms. |
| `--backend` | `mediapipe` | `mediapipe` or `face_recognition`. |
| `--no-alarm` | off | Disables audio (good for benchmarking). |
| `--no-window` | off | Headless `run` (no `cv2.imshow`). |

---

## Batch analysis

`drowsy analyze` runs a video through the same detector with audio disabled and produces a small report:

```
report/
├── ear.csv # frame-by-frame EAR & state
├── events.jsonl # one record per AWAKE→DROWSY transition
└── summary.json # counts, parameters, timestamps
```

Useful for fleet review, evaluating threshold changes, or feeding downstream analytics without a GUI.

---

## Project structure

```
drowsiness/
├── __init__.py # public API
├── ear.py # eye_aspect_ratio() + helpers
├── landmarks.py # MediaPipe + face_recognition backends
├── alarm.py # background, cooldown-debounced audio
├── detector.py # DrowsinessDetector + FSM + events
├── visualization.py # cv2 overlay (EAR, FPS, banner)
└── cli.py # `drowsy` entrypoint
tests/
└── test_detector.py # 14 cases, mocked landmark backend
assets/
└── alert1.mp3
```

---

## Limitations

This is still an **EAR-based** system. Known weaknesses:

- Sunglasses / heavy bangs / extreme head pose break landmark detection.
- A single threshold doesn't fit every driver — production deployments should calibrate per-driver during the first ~30s of awake driving.
- No yawn / head-nod / gaze-direction signal yet (PRs welcome — `DrowsyEvent` is the right place to fan out into multi-cue fusion).
- No tracking across multiple faces; the first detected face is used. Cabin-facing cameras typically only see one driver.

---

## Citation

```bibtex
@inproceedings{santhanam2021drowsiness,
 title = {Real-Time Drowsiness Detection using Computer Vision},
 author = {Santhanam, S. and others},
 booktitle = {2021 5th International Conference on Computing Methodologies and Communication (ICCMC)},
 year = {2021},
 doi = {10.1109/ICCMC51019.2021.9418325}
}
```

## License

[MIT](LICENSE) © Sanjay Santhanam
