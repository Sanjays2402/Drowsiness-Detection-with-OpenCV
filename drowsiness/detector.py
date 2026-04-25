"""Stateful drowsiness detector.

The detector consumes BGR frames one at a time and maintains a small
finite-state machine:

    AWAKE  --(EAR < threshold for N frames)-->   DROWSY
    DROWSY --(EAR >= threshold for M frames)-->  AWAKE

A ``DrowsyEvent`` callback fires exactly once on each AWAKE→DROWSY
transition, which is the right place to wire up alarms, logging,
fleet telemetry, etc.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from .alarm import Alarm
from .ear import Point, average_ear, eye_aspect_ratio
from .landmarks import FaceLandmarks, LandmarkBackend, make_backend

logger = logging.getLogger(__name__)


class EyeState(str, Enum):
    AWAKE = "awake"
    DROWSY = "drowsy"


@dataclass
class DrowsyEvent:
    """Fired on every AWAKE → DROWSY transition."""

    timestamp: float
    ear: float
    closed_frames: int
    face_index: int


@dataclass
class DetectionResult:
    """Per-frame output from ``DrowsinessDetector.process``."""

    state: EyeState
    ear: float
    closed_frames: int
    faces: List[FaceLandmarks] = field(default_factory=list)
    event: Optional[DrowsyEvent] = None


@dataclass
class DrowsinessConfig:
    ear_threshold: float = 0.25
    closed_frames_to_alarm: int = 20
    open_frames_to_clear: int = 5
    alarm_sound: str = "assets/alert1.mp3"
    alarm_cooldown_s: float = 3.0
    backend: str = "mediapipe"  # "mediapipe" | "face_recognition"
    enable_alarm: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.ear_threshold < 1.0:
            raise ValueError("ear_threshold must be in (0, 1)")
        if self.closed_frames_to_alarm < 1:
            raise ValueError("closed_frames_to_alarm must be >= 1")
        if self.open_frames_to_clear < 1:
            raise ValueError("open_frames_to_clear must be >= 1")


class DrowsinessDetector:
    """Frame-by-frame drowsiness detector.

    Example:

        detector = DrowsinessDetector()
        while True:
            frame = camera.read()
            result = detector.process(frame)
            if result.event:
                logger.warning("Driver drowsy: EAR=%.2f", result.event.ear)
    """

    def __init__(
        self,
        config: Optional[DrowsinessConfig] = None,
        backend: Optional[LandmarkBackend] = None,
        on_event: Optional[Callable[[DrowsyEvent], None]] = None,
        alarm: Optional[Alarm] = None,
    ) -> None:
        self.config = config or DrowsinessConfig()
        self._backend = backend or make_backend(self.config.backend)
        self._on_event = on_event
        if self.config.enable_alarm:
            self._alarm = alarm or Alarm(
                sound_path=self.config.alarm_sound,
                cooldown=self.config.alarm_cooldown_s,
            )
        else:
            self._alarm = None

        self._state = EyeState.AWAKE
        self._closed_frames = 0
        self._open_frames = 0

    # --- public API ----------------------------------------------------

    @property
    def state(self) -> EyeState:
        return self._state

    @property
    def closed_frames(self) -> int:
        return self._closed_frames

    def reset(self) -> None:
        self._state = EyeState.AWAKE
        self._closed_frames = 0
        self._open_frames = 0
        if self._alarm is not None:
            self._alarm.reset()

    def process(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Process a single BGR frame and update internal state."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("process() requires a non-empty BGR frame")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("process() expects an (H, W, 3) BGR frame")

        faces = self._backend.detect(frame_bgr)
        if not faces:
            # No face → don't accumulate either way. We deliberately
            # avoid resetting closed_frames so brief detection dropouts
            # mid-yawn don't reset the alarm countdown.
            return DetectionResult(
                state=self._state,
                ear=0.0,
                closed_frames=self._closed_frames,
                faces=[],
            )

        # Use the first detected face. Most cabin-facing cameras only
        # see the driver; backends can be configured for max_faces=1.
        primary = faces[0]
        ear = average_ear(primary.left_eye, primary.right_eye)
        event = self._update_state(ear, face_index=0)
        return DetectionResult(
            state=self._state,
            ear=ear,
            closed_frames=self._closed_frames,
            faces=faces,
            event=event,
        )

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "DrowsinessDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- internals -----------------------------------------------------

    def _update_state(self, ear: float, face_index: int) -> Optional[DrowsyEvent]:
        cfg = self.config
        if ear < cfg.ear_threshold:
            self._closed_frames += 1
            self._open_frames = 0
            if (
                self._state is EyeState.AWAKE
                and self._closed_frames >= cfg.closed_frames_to_alarm
            ):
                self._state = EyeState.DROWSY
                event = DrowsyEvent(
                    timestamp=time.time(),
                    ear=ear,
                    closed_frames=self._closed_frames,
                    face_index=face_index,
                )
                if self._alarm is not None:
                    self._alarm.trigger()
                if self._on_event is not None:
                    try:
                        self._on_event(event)
                    except Exception:  # pragma: no cover - user callback
                        logger.exception("on_event callback raised")
                return event
        else:
            self._open_frames += 1
            if (
                self._state is EyeState.DROWSY
                and self._open_frames >= cfg.open_frames_to_clear
            ):
                self._state = EyeState.AWAKE
                self._closed_frames = 0
            elif self._state is EyeState.AWAKE:
                self._closed_frames = 0
        return None


__all__ = [
    "DrowsinessConfig",
    "DrowsinessDetector",
    "DrowsyEvent",
    "DetectionResult",
    "EyeState",
    "eye_aspect_ratio",
    "Point",
]
