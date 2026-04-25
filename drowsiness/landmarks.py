"""Landmark detection backends.

Two backends are supported:

- ``mediapipe`` (default): FaceMesh, no native compile step, very fast.
- ``face_recognition``: dlib-based, matches the original IEEE paper.

Each backend is wrapped in a tiny adapter that exposes a single method:

    detect(frame_bgr) -> list[FaceLandmarks]

``FaceLandmarks`` carries the six EAR points per eye plus a bounding
box for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

import numpy as np

from .ear import Point


@dataclass(frozen=True)
class FaceLandmarks:
    left_eye: Sequence[Point]
    right_eye: Sequence[Point]
    bbox: Tuple[int, int, int, int]  # x, y, w, h in pixels


class LandmarkBackend(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray) -> List[FaceLandmarks]: ...


# MediaPipe FaceMesh indices for the six EAR points per eye.
# Order matches the EAR definition: outer, upper-outer, upper-inner,
# inner, lower-inner, lower-outer.
_MP_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_MP_RIGHT_EYE = (263, 387, 385, 362, 380, 373)


class MediaPipeBackend:
    """FaceMesh-based backend. Default."""

    name = "mediapipe"

    def __init__(
        self,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = False,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "MediaPipe backend requires the 'mediapipe' package. "
                "Install with: pip install drowsiness-detector[mediapipe]"
            ) from err
        self._mp = mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> List[FaceLandmarks]:
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []
        faces: List[FaceLandmarks] = []
        for landmark_set in result.multi_face_landmarks:
            pts = [(lm.x * w, lm.y * h) for lm in landmark_set.landmark]
            left = [pts[i] for i in _MP_LEFT_EYE]
            right = [pts[i] for i in _MP_RIGHT_EYE]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
            x1, y1 = min(w, int(max(xs))), min(h, int(max(ys)))
            faces.append(
                FaceLandmarks(
                    left_eye=left,
                    right_eye=right,
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                )
            )
        return faces

    def close(self) -> None:
        self._mesh.close()


class FaceRecognitionBackend:
    """dlib/face_recognition backend (matches the IEEE paper)."""

    name = "face_recognition"

    def __init__(self, model: str = "small") -> None:
        try:
            import face_recognition  # noqa: F401
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "face_recognition backend requires the 'face_recognition' package. "
                "Install with: pip install drowsiness-detector[legacy]"
            ) from err
        self._face_recognition = __import__("face_recognition")
        self._model = model

    def detect(self, frame_bgr: np.ndarray) -> List[FaceLandmarks]:
        rgb = frame_bgr[:, :, ::-1]
        landmarks_list = self._face_recognition.face_landmarks(rgb, model=self._model)
        faces: List[FaceLandmarks] = []
        for landmarks in landmarks_list:
            left = landmarks.get("left_eye")
            right = landmarks.get("right_eye")
            if not left or not right or len(left) != 6 or len(right) != 6:
                continue
            all_points = []
            for region in landmarks.values():
                all_points.extend(region)
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            x0, y0 = int(min(xs)), int(min(ys))
            x1, y1 = int(max(xs)), int(max(ys))
            faces.append(
                FaceLandmarks(
                    left_eye=[(float(x), float(y)) for x, y in left],
                    right_eye=[(float(x), float(y)) for x, y in right],
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                )
            )
        return faces

    def close(self) -> None:
        return None


def make_backend(name: str = "mediapipe", **kwargs) -> LandmarkBackend:
    """Create a backend by name.

    Args:
        name: Either ``"mediapipe"`` (default) or ``"face_recognition"``.
        **kwargs: Forwarded to the backend constructor.
    """
    name = name.lower().replace("-", "_")
    if name == "mediapipe":
        return MediaPipeBackend(**kwargs)
    if name in {"face_recognition", "dlib"}:
        return FaceRecognitionBackend(**kwargs)
    raise ValueError(f"Unknown landmark backend: {name!r}")
