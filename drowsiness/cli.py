"""Command-line entry point: ``drowsy``.

Subcommands
-----------

* ``drowsy run`` — live webcam or video-file detection with overlay window.
* ``drowsy analyze`` — headless processing of a video file. Emits a JSONL
  report and a per-frame EAR CSV. No GUI required, no audio. Useful for
  fleet review / batch evaluation.

Run ``drowsy run --help`` or ``drowsy analyze --help`` for full options.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2

from .detector import DrowsinessConfig, DrowsinessDetector
from .visualization import draw_overlay

logger = logging.getLogger("drowsy")


def _open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {source!r}")
    return cap


def _config_from_args(args: argparse.Namespace) -> DrowsinessConfig:
    return DrowsinessConfig(
        ear_threshold=args.ear_threshold,
        closed_frames_to_alarm=args.frame_count,
        open_frames_to_clear=args.open_frames,
        alarm_sound=args.alarm_sound,
        alarm_cooldown_s=args.alarm_cooldown,
        backend=args.backend,
        enable_alarm=not args.no_alarm,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    cap = _open_capture(args.source)

    fps_window: deque[float] = deque(maxlen=30)
    last_t = time.monotonic()

    with DrowsinessDetector(cfg) as detector:
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if args.width:
                    h, w = frame.shape[:2]
                    new_w = args.width
                    new_h = int(h * (new_w / w))
                    frame = cv2.resize(frame, (new_w, new_h))

                result = detector.process(frame)
                now = time.monotonic()
                fps_window.append(1.0 / max(now - last_t, 1e-6))
                last_t = now
                fps = sum(fps_window) / len(fps_window)

                if result.event:
                    logger.warning(
                        "DROWSY at t=%.2fs ear=%.3f closed_frames=%d",
                        result.event.timestamp, result.event.ear,
                        result.event.closed_frames,
                    )

                if not args.no_window:
                    draw_overlay(frame, result, fps=fps, threshold=cfg.ear_threshold)
                    cv2.imshow("Drowsiness Detection", frame)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
        finally:
            cap.release()
            if not args.no_window:
                cv2.destroyAllWindows()
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    cfg.enable_alarm = False  # never play audio in batch mode
    cap = _open_capture(args.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_dir = Path(args.output_dir or "drowsy-report")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ear.csv"
    events_path = output_dir / "events.jsonl"
    summary_path = output_dir / "summary.json"

    events = []
    frames = 0
    closed = 0
    started = time.monotonic()

    with DrowsinessDetector(cfg) as detector, \
         csv_path.open("w", newline="") as csv_f, \
         events_path.open("w") as events_f:
        writer = csv.writer(csv_f)
        writer.writerow(["frame", "t_seconds", "ear", "state", "closed_frames"])

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            result = detector.process(frame)
            t = frames / fps_in
            writer.writerow([frames, f"{t:.3f}", f"{result.ear:.4f}",
                             result.state.value, result.closed_frames])
            if result.ear and result.ear < cfg.ear_threshold:
                closed += 1
            if result.event:
                payload = {
                    "frame": frames,
                    "t_seconds": t,
                    "ear": result.event.ear,
                    "closed_frames": result.event.closed_frames,
                }
                events_f.write(json.dumps(payload) + "\n")
                events.append(payload)
            frames += 1

    elapsed = time.monotonic() - started
    summary = {
        "source": args.source,
        "frames": frames,
        "input_fps": fps_in,
        "closed_frames": closed,
        "drowsy_events": len(events),
        "ear_threshold": cfg.ear_threshold,
        "closed_frames_to_alarm": cfg.closed_frames_to_alarm,
        "wall_seconds": round(elapsed, 3),
        "events": events,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"frames={frames} drowsy_events={len(events)} report={output_dir}")
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", help="Video source: device index (e.g. 0) or path.")
    parser.add_argument("--ear-threshold", type=float, default=0.25)
    parser.add_argument("--frame-count", type=int, default=20,
                        help="Closed frames before raising alarm.")
    parser.add_argument("--open-frames", type=int, default=5,
                        help="Open frames before clearing the drowsy state.")
    parser.add_argument("--alarm-sound", default="assets/alert1.mp3")
    parser.add_argument("--alarm-cooldown", type=float, default=3.0)
    parser.add_argument("--no-alarm", action="store_true")
    parser.add_argument("--backend", default="mediapipe",
                        choices=["mediapipe", "face_recognition"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drowsy",
        description="Real-time drowsiness detection (Eye Aspect Ratio).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Live detection with overlay window.")
    _add_common(run)
    run.add_argument("--width", type=int, default=640,
                     help="Resize input to this width (preserves aspect).")
    run.add_argument("--no-window", action="store_true",
                     help="Headless mode (no cv2.imshow).")
    run.set_defaults(func=_cmd_run)

    analyze = sub.add_parser("analyze",
                             help="Headless batch processing of a video file.")
    _add_common(analyze)
    analyze.add_argument("--output-dir", default=None,
                         help="Directory for ear.csv / events.jsonl / summary.json.")
    analyze.set_defaults(func=_cmd_analyze)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
