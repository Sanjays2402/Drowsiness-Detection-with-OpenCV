"""Non-blocking, debounced audio alarm.

Fires once per drowsiness *event* on a background thread so the
detection loop never stalls. Successive triggers within ``cooldown``
seconds are silently dropped — you don't get overlapping playbacks
piling on top of one another.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _try_playsound(path: str) -> bool:
    """Return True iff playsound is installed and played without raising."""
    try:
        from playsound import playsound  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        playsound(path)
        return True
    except Exception as err:  # pragma: no cover - playback errors
        # Previously a runtime failure here returned silently and never
        # fell through to the OS players. Now we log and fall through
        # so a flaky playsound install doesn't disable the alarm.
        logger.warning("playsound failed (%s); falling back to OS player", err)
        return False


def _try_winsound(path: str) -> bool:  # pragma: no cover - Windows-only
    if os.name != "nt":
        return False
    try:
        import winsound  # type: ignore[import-not-found]
        winsound.PlaySound(path, winsound.SND_FILENAME)
        return True
    except Exception as err:
        logger.warning("winsound failed: %s", err)
        return False


def _try_subprocess(path: str) -> bool:
    """Try common OS audio players with safe arg passing (no shell)."""
    for cmd in ("afplay", "paplay", "aplay"):
        exe = shutil.which(cmd)
        if not exe:
            continue
        try:
            subprocess.run(
                [exe, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return True
        except OSError as err:  # pragma: no cover - exec failure
            logger.warning("%s exec failed: %s", cmd, err)
            continue
    return False


def _default_player(path: str) -> None:
    """Play ``path`` synchronously using whichever backend is available.

    Order: playsound (if importable) → winsound (Windows) → afplay/paplay/
    aplay via ``subprocess.run`` (no shell, so paths with shell-metachars
    like apostrophes or spaces are safe).
    """
    if _try_playsound(path):
        return
    if _try_winsound(path):
        return
    if _try_subprocess(path):
        return
    logger.warning("No audio player available; alarm path=%s not played", path)


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

        path = self._resolve_path()
        if path is None:
            return False

        # Only burn the cooldown slot once we've confirmed the file
        # exists. Otherwise a missing alarm file plus a permanent
        # cooldown lockout would mean the very first time the file
        # appeared, we'd still suppress the playback.
        with self._lock:
            self._last_played = now

        thread = threading.Thread(
            target=self._player, args=(path,), daemon=True, name="drowsy-alarm"
        )
        thread.start()
        return True

    def reset(self) -> None:
        with self._lock:
            self._last_played = 0.0
