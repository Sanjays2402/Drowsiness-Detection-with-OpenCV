"""Real-time driver drowsiness detection.

Public API:

    from drowsiness import DrowsinessDetector, DrowsinessConfig
    from drowsiness import EyeState, DrowsyEvent

The detector is backend-agnostic and stateful. See `drowsiness.cli` for
the bundled `drowsy` command-line entry point.
"""

from __future__ import annotations

from .detector import (
    DrowsinessConfig,
    DrowsinessDetector,
    DrowsyEvent,
    DetectionResult,
    EyeState,
)

__all__ = [
    "DrowsinessConfig",
    "DrowsinessDetector",
    "DrowsyEvent",
    "DetectionResult",
    "EyeState",
    "__version__",
]

__version__ = "1.0.0"
