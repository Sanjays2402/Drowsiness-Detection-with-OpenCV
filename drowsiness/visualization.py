"""Frame visualization helpers (cv2-based)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .detector import DetectionResult, EyeState

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (100, 250, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)


def draw_overlay(
    frame_bgr: np.ndarray,
    result: DetectionResult,
    *,
    fps: Optional[float] = None,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Draw EAR, FPS, eye contours, and any drowsy banner onto ``frame_bgr``.

    Modifies the frame in place and returns it for chaining.
    """
    color = _RED if result.state is EyeState.DROWSY else _GREEN
    for face in result.faces:
        left = np.array(face.left_eye, dtype=np.int32)
        right = np.array(face.right_eye, dtype=np.int32)
        cv2.polylines(frame_bgr, [left], True, color, 1)
        cv2.polylines(frame_bgr, [right], True, color, 1)
        x, y, w, h = face.bbox
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 1)

    h, _ = frame_bgr.shape[:2]
    text = f"EAR: {result.ear:.2f}"
    if threshold is not None:
        text += f" / {threshold:.2f}"
    cv2.putText(frame_bgr, text, (10, h - 14), _FONT, 0.6, _WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, (10, h - 14), _FONT, 0.6, color, 1, cv2.LINE_AA)

    if fps is not None:
        fps_text = f"{fps:5.1f} fps"
        cv2.putText(frame_bgr, fps_text, (10, 22), _FONT, 0.6, _WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, fps_text, (10, 22), _FONT, 0.6, color, 1, cv2.LINE_AA)

    if result.state is EyeState.DROWSY:
        banner = "DROWSINESS ALERT!"
        size = cv2.getTextSize(banner, _FONT, 1.0, 3)[0]
        x = (frame_bgr.shape[1] - size[0]) // 2
        cv2.putText(frame_bgr, banner, (x, 50), _FONT, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame_bgr, banner, (x, 50), _FONT, 1.0, _RED, 2, cv2.LINE_AA)
    return frame_bgr
