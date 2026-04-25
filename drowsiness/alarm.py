"""Non-blocking, debounced audio alarm.

Fires once per drowsiness *event* on a background thread so the
detection loop never stalls. Successive triggers within ``cooldown``
seconds are silently dropped — you don't get overlapping playbacks
piling on top of one another.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _default_player(path: str) -> None:
    """Play ``path`` synchronously using whichever lib is available."""
    try:
        from playsound import playsound
        playsound(path)
        return
    except ImportError:
        pass
    except Exception as err:  # pragma: no cover - playback errors
        logger.warning("playsound failed: %s", err)
        return

    # Fallback: macOS 'afplay', Linux 'aplay/paplay', Windows winsound.
    if os.name == "nt":  # pragma: no cover - Windows-only
        try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
            return
        except Exception as err:
            logger.warning("winsound failed: %s", err)
            return

    for cmd in ("afplay", "paplay", "aplay"):
        if _has_binary(cmd):
            os.system(f"{cmd} {path!r} > /dev/null 2>&1")  # noqa: S605
            return
    logger.warning("No audio player available; alarm path=%s not played", path)


def _has_binary(name: str) -> bool:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if os.path.isfile(os.path.join(directory, name)):
            return True
    return False


class Alarm:
    """Background, debounced alarm.

    Args:
        sound_path: Path to a playable audio file (mp3/wav/...).
        cooldown: Seconds between successive alarms. Triggers inside
            the cooldown window are dropped.
        player: Optional override for the audio player. Mostly used
            for tests. Receives the sound path string.
    """

    def __init__(
        self,
        sound_path: str = "assets/alert1.mp3",
        cooldown: float = 3.0,
        player: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._sound_path = sound_path
        self._cooldown = float(cooldown)
        self._player = player or _default_player
        self._last_played = 0.0
        self._lock = threading.Lock()
        self._missing_logged = False

    @property
    def sound_path(self) -> str:
        return self._sound_path

    def _resolve_path(self) -> Optional[str]:
        path = Path(self._sound_path)
        if path.exists():
            return str(path)
        if not self._missing_logged:
            logger.warning("Alarm sound not found at %s; alarms will be silent", path)
            self._missing_logged = True
        return None

    def trigger(self) -> bool:
        """Schedule a single alarm playback.

        Returns:
            ``True`` if the playback was scheduled, ``False`` if it
            was suppressed by the cooldown or because the file is
            missing.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_played < self._cooldown:
                return False
            self._last_played = now

        path = self._resolve_path()
        if path is None:
            return False

        thread = threading.Thread(
            target=self._player, args=(path,), daemon=True, name="drowsy-alarm"
        )
        thread.start()
        return True

    def reset(self) -> None:
        with self._lock:
            self._last_played = 0.0
