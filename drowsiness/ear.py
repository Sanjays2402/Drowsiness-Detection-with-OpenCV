"""Eye Aspect Ratio (EAR) computation.

Reference:
    Soukupová & Čech, "Real-Time Eye Blink Detection using Facial
    Landmarks", 21st Computer Vision Winter Workshop (2016).

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

The function accepts any iterable of six (x, y) points; it does NOT
care which landmark backend produced them. Backends are responsible
for selecting the six canonical eye points and feeding them in the
expected order (outer → upper → lower → ...).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

Point = Tuple[float, float]


def _euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(eye: Sequence[Point]) -> float:
    """Compute the EAR for a single eye described by 6 ordered points.

    Args:
        eye: Sequence of exactly 6 ``(x, y)`` points in the order
            ``[outer, upper-outer, upper-inner, inner, lower-inner, lower-outer]``.

    Returns:
        The EAR. Higher = eye more open. Returns ``0.0`` when the eye
        is degenerate (zero horizontal span) instead of raising, so
        the caller can keep its detection loop running.
    """
    if len(eye) != 6:
        raise ValueError(f"eye_aspect_ratio expects 6 points, got {len(eye)}")
    p1, p2, p3, p4, p5, p6 = eye
    horizontal = _euclid(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    vertical_a = _euclid(p2, p6)
    vertical_b = _euclid(p3, p5)
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def average_ear(left: Sequence[Point], right: Sequence[Point]) -> float:
    """Mean EAR across both eyes."""
    return (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0


def points_from_iterable(points: Iterable[Iterable[float]]) -> list[Point]:
    """Coerce arbitrary nested numerics to a list of ``(float, float)`` tuples."""
    out: list[Point] = []
    for p in points:
        x, y = list(p)[:2]
        out.append((float(x), float(y)))
    return out
