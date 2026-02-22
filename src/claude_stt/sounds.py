"""Audio feedback sounds.

Bundled sounds are from Kenney Interface Sounds (CC0 public domain).
See assets/SOUNDS_LICENSE.txt for details.
"""

import logging
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Literal

SoundEvent = Literal["start", "stop", "complete", "error", "warning"]
_logger = logging.getLogger(__name__)

# Bundled sounds (CC0, Kenney Interface Sounds)
_ASSETS_DIR = Path(__file__).parent / "assets"
SOUNDS: dict[str, Path] = {
    "start": _ASSETS_DIR / "start.ogg",
    "stop": _ASSETS_DIR / "stop.ogg",
    "complete": _ASSETS_DIR / "complete.ogg",
    "error": _ASSETS_DIR / "error.ogg",
    "warning": _ASSETS_DIR / "warning.ogg",
}

# Linux desktop notification messages (requires notify-send / libnotify-bin)
LINUX_NOTIFICATIONS: dict[str, tuple[str, str]] = {
    "start": ("Claude STT", "Recording..."),
    "stop": ("Claude STT", "Recording stopped"),
    "complete": ("Claude STT", "Text injected"),
    "error": ("Claude STT", "Error"),
    "warning": ("Claude STT", "Warning — no speech detected"),
}


def play_sound(event: SoundEvent) -> None:
    """Play a bundled sound for the given event.

    Args:
        event: The type of sound event to play.
    """
    try:
        system = platform.system()

        if system == "Windows":
            _play_windows_sound(event)
            return

        sound_file = SOUNDS.get(event)
        if not sound_file or not sound_file.exists():
            _logger.debug("Sound file missing for event: %s", event)
            return

        _play_sound_file(str(sound_file), system)

        if system == "Linux":
            _send_linux_notification(event)
    except Exception:
        _logger.debug("Sound playback failed for event: %s", event, exc_info=True)


def _play_sound_file(sound_file: str, system: str) -> None:
    """Play a sound file using the best available player.

    Args:
        sound_file: Path to the sound file.
        system: Platform name from platform.system().
    """
    if system == "Darwin":
        if shutil.which("afplay"):
            subprocess.Popen(
                ["afplay", sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return

    # Linux: try paplay (PulseAudio/PipeWire), then aplay (ALSA)
    if shutil.which("paplay"):
        subprocess.Popen(
            ["paplay", sound_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif shutil.which("aplay"):
        subprocess.Popen(
            ["aplay", "-q", sound_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _send_linux_notification(event: SoundEvent) -> None:
    """Send a desktop notification on Linux via notify-send."""
    if not shutil.which("notify-send"):
        return
    notification = LINUX_NOTIFICATIONS.get(event)
    if not notification:
        return
    title, body = notification
    urgency = "critical" if event in ("error", "warning") else "low"
    subprocess.Popen(
        ["notify-send", "--urgency", urgency, "--expire-time", "2000",
         "--app-name", "claude-stt", title, body],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _play_windows_sound(event: SoundEvent) -> None:
    """Play sound on Windows using winsound."""
    try:
        import winsound

        sound_map = {
            "start": winsound.MB_OK,
            "stop": winsound.MB_OK,
            "complete": winsound.MB_OK,
            "error": winsound.MB_ICONHAND,
            "warning": winsound.MB_ICONEXCLAMATION,
        }

        sound_type = sound_map.get(event, winsound.MB_OK)
        winsound.MessageBeep(sound_type)
    except ImportError:
        _logger.debug("winsound not available")
