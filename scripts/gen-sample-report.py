"""Synthesize a realistic report directory and render report.html."""
import csv
import json
import math
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drowsiness.report import render_report

random.seed(42)

out = Path("scripts/sample-report")
out.mkdir(parents=True, exist_ok=True)

fps = 30.0
duration_s = 180.0  # 3-minute drive clip
n_frames = int(duration_s * fps)
threshold = 0.25
closed_frames_to_alarm = 20

# Generate a realistic EAR signal: baseline 0.30 with sinusoidal blinks
# and three drowsy stretches (sustained low EAR) where events fire.
drowsy_windows = [(35.0, 41.0), (94.0, 102.0), (148.0, 155.0)]

ears = []
state = "awake"
closed = 0
events = []
closed_total = 0

for i in range(n_frames):
    t = i / fps
    base = 0.30 + 0.02 * math.sin(t * 0.7)
    blink = -0.15 if (i % 90) < 3 else 0.0  # brief blinks every 3s
    drowsy_pull = 0.0
    for a, b in drowsy_windows:
        if a <= t <= b:
            drowsy_pull = -0.12
            break
    ear = max(0.05, base + blink + drowsy_pull + random.gauss(0, 0.004))
    if ear < threshold:
        closed += 1
        closed_total += 1
        if state == "awake" and closed >= closed_frames_to_alarm:
            state = "drowsy"
            events.append({"frame": i, "t_seconds": t, "ear": ear, "closed_frames": closed})
    else:
        if state == "drowsy":
            state = "awake"
            closed = 0
        else:
            closed = 0
    ears.append((i, t, ear, state, closed))

with (out / "ear.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "t_seconds", "ear", "state", "closed_frames"])
    for frame, t, ear, st, cf in ears:
        w.writerow([frame, f"{t:.3f}", f"{ear:.4f}", st, cf])

with (out / "events.jsonl").open("w") as f:
    for ev in events:
        f.write(json.dumps(ev) + "\n")

summary = {
    "source": "samples/long-drive.mp4",
    "frames": n_frames,
    "input_fps": fps,
    "closed_frames": closed_total,
    "drowsy_events": len(events),
    "ear_threshold": threshold,
    "closed_frames_to_alarm": closed_frames_to_alarm,
    "wall_seconds": 7.2,
    "events": events,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))

html_path = render_report(out, version="0.3.0")
print(f"wrote {html_path}")
print(f"events: {len(events)}, closed_frames: {closed_total}/{n_frames}")
