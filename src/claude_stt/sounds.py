"""Audio feedback sounds.

Bundled sounds are from Kenney Interface Sounds (CC0 public domain).
See assets/SOUNDS_LICENSE.txt for details.
"""

import logging
import platform
import shutil
import subprocess
from enum import Enum
from pathlib import Path

_logger = logging.getLogger(__name__)
_ASSETS_DIR = Path(__file__).parent / "assets"


class SoundEvent(Enum):
    START = "start"
    STOP = "stop"
    COMPLETE = "complete"
    ERROR = "error"
    WARNING = "warning"
    READY = "ready"
    SHUTDOWN = "shutdown"


SOUNDS: dict[SoundEvent, Path] = {
    SoundEvent.START: _ASSETS_DIR / "start.ogg",
    SoundEvent.STOP: _ASSETS_DIR / "stop.ogg",
    SoundEvent.COMPLETE: _ASSETS_DIR / "complete.ogg",
    SoundEvent.ERROR: _ASSETS_DIR / "error.ogg",
    SoundEvent.WARNING: _ASSETS_DIR / "warning.ogg",
    SoundEvent.READY: _ASSETS_DIR / "ready.ogg",
    SoundEvent.SHUTDOWN: _ASSETS_DIR / "shutdown.ogg",
}

# Linux desktop notification messages (requires notify-send / libnotify-bin)
LINUX_NOTIFICATIONS: dict[SoundEvent, tuple[str, str]] = {
    SoundEvent.START: ("Claude STT", "Recording..."),
    SoundEvent.STOP: ("Claude STT", "Recording stopped"),
    SoundEvent.COMPLETE: ("Claude STT", "Text injected"),
    SoundEvent.ERROR: ("Claude STT", "Error"),
    SoundEvent.WARNING: ("Claude STT", "Warning — no speech detected"),
    SoundEvent.READY: ("Claude STT", "Ready for voice input"),
    SoundEvent.SHUTDOWN: ("Claude STT", "Daemon stopped"),
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
            _logger.debug("Sound file missing for event: %s", event.value)
            return

        _play_sound_file(str(sound_file), system)

        if system == "Linux":
            _send_linux_notification(event)
    except Exception:
        _logger.debug("Sound playback failed for event: %s", event.value, exc_info=True)


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
    urgency = "critical" if event in (SoundEvent.ERROR, SoundEvent.WARNING) else "low"
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
            SoundEvent.START: winsound.MB_OK,
            SoundEvent.STOP: winsound.MB_OK,
            SoundEvent.COMPLETE: winsound.MB_OK,
            SoundEvent.ERROR: winsound.MB_ICONHAND,
            SoundEvent.WARNING: winsound.MB_ICONEXCLAMATION,
        }

        sound_type = sound_map.get(event, winsound.MB_OK)
        winsound.MessageBeep(sound_type)
    except ImportError:
        _logger.debug("winsound not available")
