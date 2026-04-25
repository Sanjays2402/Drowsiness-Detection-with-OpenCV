"""Tests for the drowsiness detector core.

These tests rely only on numpy and pytest. The landmark backend is
mocked so we never need a real face or webcam.
"""

from __future__ import annotations

import math
import time
from typing import List

import numpy as np
import pytest

from drowsiness import DrowsinessConfig, DrowsinessDetector, EyeState
from drowsiness.alarm import Alarm
from drowsiness.ear import average_ear, eye_aspect_ratio
from drowsiness.landmarks import FaceLandmarks


# ---- EAR math --------------------------------------------------------


def _open_eye():
    # A wide eye: horizontal span 100, vertical span 30.
    return [(0, 50), (25, 35), (75, 35), (100, 50), (75, 65), (25, 65)]


def _closed_eye():
    return [(0, 50), (25, 49), (75, 49), (100, 50), (75, 51), (25, 51)]


def test_eye_aspect_ratio_open_vs_closed():
    open_ear = eye_aspect_ratio(_open_eye())
    closed_ear = eye_aspect_ratio(_closed_eye())
    assert open_ear > 0.25
    assert closed_ear < 0.05
    assert open_ear > closed_ear


def test_eye_aspect_ratio_handles_degenerate_horizontal():
    # All points collapsed to one location → horizontal == 0; should
    # return 0.0 instead of raising.
    ear = eye_aspect_ratio([(0, 0)] * 6)
    assert ear == 0.0


def test_eye_aspect_ratio_rejects_wrong_count():
    with pytest.raises(ValueError):
        eye_aspect_ratio([(0, 0), (1, 1)])


def test_average_ear_is_mean():
    avg = average_ear(_open_eye(), _closed_eye())
    assert math.isclose(
        avg, (eye_aspect_ratio(_open_eye()) + eye_aspect_ratio(_closed_eye())) / 2
    )


# ---- Detector FSM with a fake backend --------------------------------


class FakeBackend:
    """Backend stub that replays a queue of EAR values per call.

    The eye points it returns *always* produce the requested EAR via
    a tiny synthetic eye where horizontal=100 and vertical=ear*100.
    """

    name = "fake"

    def __init__(self, ear_values: List[float], face_present: List[bool] = None):
        self._ears = list(ear_values)
        self._face_present = list(face_present or [True] * len(ear_values))
        self.calls = 0

    @staticmethod
    def _eye(ear: float):
        v = ear * 100.0
        return [
            (0, 50),
            (25, 50 - v / 2),
            (75, 50 - v / 2),
            (100, 50),
            (75, 50 + v / 2),
            (25, 50 + v / 2),
        ]

    def detect(self, frame_bgr):
        idx = self.calls
        self.calls += 1
        if idx >= len(self._ears) or not self._face_present[idx]:
            return []
        ear = self._ears[idx]
        eye = self._eye(ear)
        return [FaceLandmarks(left_eye=eye, right_eye=eye, bbox=(0, 0, 100, 100))]


@pytest.fixture
def blank_frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def _silent_detector(config: DrowsinessConfig, backend: FakeBackend, **kwargs):
    """Build a detector with a no-op alarm so tests stay quiet."""
    plays: list[str] = []
    alarm = Alarm(sound_path=__file__, cooldown=0.0, player=lambda p: plays.append(p))
    detector = DrowsinessDetector(config=config, backend=backend, alarm=alarm, **kwargs)
    return detector, plays


def test_drowsy_event_fires_after_n_closed_frames(blank_frame):
    cfg = DrowsinessConfig(closed_frames_to_alarm=3, open_frames_to_clear=2)
    backend = FakeBackend([0.10, 0.10, 0.10, 0.10, 0.40, 0.40, 0.40])
    detector, plays = _silent_detector(cfg, backend)

    states = []
    events = []
    for _ in range(7):
        result = detector.process(blank_frame)
        states.append(result.state)
        if result.event:
            events.append(result.event)

    # After 3 closed frames we should be DROWSY.
    assert states[:2] == [EyeState.AWAKE, EyeState.AWAKE]
    assert states[2] is EyeState.DROWSY
    # Subsequent closed frames must NOT fire additional events.
    assert len(events) == 1
    # 2 open frames clear it.
    assert states[-1] is EyeState.AWAKE
    # Alarm played exactly once.
    assert len(plays) == 1


def test_short_blink_does_not_alarm(blank_frame):
    cfg = DrowsinessConfig(closed_frames_to_alarm=5, open_frames_to_clear=2)
    backend = FakeBackend([0.10, 0.10, 0.40, 0.40, 0.40, 0.10, 0.40])
    detector, plays = _silent_detector(cfg, backend)

    events = []
    for _ in range(7):
        r = detector.process(blank_frame)
        if r.event:
            events.append(r.event)

    assert events == []
    assert plays == []
    assert detector.state is EyeState.AWAKE


def test_no_face_does_not_reset_closed_counter(blank_frame):
    cfg = DrowsinessConfig(closed_frames_to_alarm=3, open_frames_to_clear=2)
    # 2 closed frames, then a frame with no face, then another closed.
    backend = FakeBackend(
        [0.10, 0.10, 0.10, 0.10],
        face_present=[True, True, False, True],
    )
    detector, _ = _silent_detector(cfg, backend)

    results = [detector.process(blank_frame) for _ in range(4)]
    # When no face is detected, state and closed_frames are preserved.
    assert results[2].closed_frames == 2
    # Next closed frame brings us to threshold.
    assert results[3].state is EyeState.DROWSY


def test_callback_receives_event(blank_frame):
    cfg = DrowsinessConfig(closed_frames_to_alarm=2, open_frames_to_clear=2)
    backend = FakeBackend([0.10, 0.10, 0.10])
    received = []
    plays: list[str] = []
    alarm = Alarm(sound_path=__file__, cooldown=0.0, player=lambda p: plays.append(p))
    detector = DrowsinessDetector(
        cfg, backend=backend, alarm=alarm, on_event=received.append,
    )
    for _ in range(3):
        detector.process(blank_frame)
    assert len(received) == 1
    assert received[0].closed_frames >= cfg.closed_frames_to_alarm


def test_reset_clears_state(blank_frame):
    cfg = DrowsinessConfig(closed_frames_to_alarm=2, open_frames_to_clear=5)
    backend = FakeBackend([0.10, 0.10])
    detector, _ = _silent_detector(cfg, backend)
    detector.process(blank_frame)
    detector.process(blank_frame)
    assert detector.state is EyeState.DROWSY
    detector.reset()
    assert detector.state is EyeState.AWAKE
    assert detector.closed_frames == 0


def test_invalid_frame_shape_raises(blank_frame):
    backend = FakeBackend([0.4])
    detector, _ = _silent_detector(DrowsinessConfig(), backend)
    with pytest.raises(ValueError):
        detector.process(np.zeros((10, 10), dtype=np.uint8))


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        DrowsinessConfig(ear_threshold=2.0)
    with pytest.raises(ValueError):
        DrowsinessConfig(closed_frames_to_alarm=0)


# ---- Alarm cooldown --------------------------------------------------


def test_alarm_cooldown_drops_repeat_triggers(tmp_path):
    sound = tmp_path / "fake.mp3"
    sound.write_bytes(b"\x00")
    plays: list[str] = []
    alarm = Alarm(sound_path=str(sound), cooldown=10.0, player=lambda p: plays.append(p))
    assert alarm.trigger() is True
    assert alarm.trigger() is False  # within cooldown
    assert alarm.trigger() is False
    # Wait for background thread to actually run the player.
    time.sleep(0.05)
    assert len(plays) == 1


def test_alarm_missing_sound_returns_false(tmp_path, caplog):
    plays: list[str] = []
    alarm = Alarm(
        sound_path=str(tmp_path / "nope.mp3"),
        cooldown=0.0,
        player=lambda p: plays.append(p),
    )
    assert alarm.trigger() is False
    assert plays == []


def test_alarm_reset_allows_immediate_retrigger(tmp_path):
    sound = tmp_path / "fake.mp3"
    sound.write_bytes(b"\x00")
    plays: list[str] = []
    alarm = Alarm(sound_path=str(sound), cooldown=10.0, player=lambda p: plays.append(p))
    assert alarm.trigger() is True
    assert alarm.trigger() is False
    alarm.reset()
    assert alarm.trigger() is True
    time.sleep(0.05)
    assert len(plays) == 2
