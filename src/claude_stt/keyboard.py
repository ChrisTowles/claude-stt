"""Keyboard output: direct injection or clipboard fallback."""

import logging
import shutil
import subprocess
import time
from typing import Optional

try:
    from pynput.keyboard import Controller, Key
    _PYNPUT_AVAILABLE = True
    _PYNPUT_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    Controller = None
    Key = None
    _PYNPUT_AVAILABLE = False
    _PYNPUT_IMPORT_ERROR = exc

from .config import Config, is_wayland
from .sounds import play_sound
from .window import WindowInfo, restore_focus

# Global keyboard controller
_keyboard: Optional[Controller] = None
_injection_capable: Optional[bool] = None
_injection_checked_at: Optional[float] = None
_injection_cache_ttl = 300.0
_logger = logging.getLogger(__name__)
_pynput_warned = False


def get_keyboard() -> Controller:
    """Get the global keyboard controller."""
    global _keyboard
    if not _PYNPUT_AVAILABLE:
        raise RuntimeError("pynput unavailable; keyboard injection disabled")
    if _keyboard is None:
        _keyboard = Controller()
    return _keyboard


def _warn_pynput_missing() -> None:
    global _pynput_warned
    if _pynput_warned:
        return
    message = "pynput unavailable; falling back to clipboard output"
    if _PYNPUT_IMPORT_ERROR:
        message = f"{message} ({_PYNPUT_IMPORT_ERROR})"
    _logger.warning(message)
    _pynput_warned = True


def _has_wtype() -> bool:
    """Check if wtype is available for Wayland text input."""
    return shutil.which("wtype") is not None


def _has_wl_copy() -> bool:
    """Check if wl-copy is available for Wayland clipboard."""
    return shutil.which("wl-copy") is not None


def _output_via_wtype(text: str, config: Config) -> bool:
    """Output text on Wayland using wl-copy + Ctrl+V paste.

    Direct wtype character injection is broken on some compositors (e.g.
    COSMIC) so we copy text to the Wayland clipboard with wl-copy and
    then simulate Ctrl+V to paste it. Falls back to raw wtype typing if
    wl-copy is not available.

    Args:
        text: The text to type.
        config: Configuration.

    Returns:
        True if successful, False otherwise.
    """
    _logger.debug("Injecting text via wtype (%d chars)", len(text))
    # Brief delay to let modifier keys from the hotkey fully release
    time.sleep(0.2)

    # Preferred path: wl-copy + Ctrl+V paste (works on all compositors)
    if _has_wl_copy():
        return _output_via_wl_clipboard_paste(text, config)

    # Fallback: raw wtype character injection
    _logger.debug("wl-copy not available; falling back to raw wtype")
    return _output_via_wtype_raw(text, config)


def _output_via_wl_clipboard_paste(text: str, config: Config) -> bool:
    """Copy text to Wayland clipboard and paste with Ctrl+V.

    wl-copy stays alive as the clipboard owner until another copy occurs,
    so we use Popen without waiting for exit, allowing the process to run
    in the background.

    Args:
        text: The text to paste.
        config: Configuration.

    Returns:
        True if successful, False otherwise.
    """
    # Launch wl-copy to set clipboard (runs in background)
    try:
        _logger.debug("wl-copy: copying text to clipboard")
        proc = subprocess.Popen(
            ["wl-copy", "--", text],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Give wl-copy a moment to register with the compositor
        time.sleep(0.1)
        # Check it didn't fail immediately
        rc = proc.poll()
        if rc is not None and rc != 0:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            _logger.warning("wl-copy failed (rc=%d): %s", rc, stderr)
            return False
        _logger.debug("wl-copy: process running (pid=%d)", proc.pid)
    except Exception:
        _logger.warning("wl-copy launch failed", exc_info=True)
        return False

    # Small delay to ensure clipboard is ready
    time.sleep(0.05)

    if not _send_paste_key_command():
        return False

    if config.sound_effects:
        play_sound("complete")
    return True


def _send_paste_key_command() -> bool:
    """Send Ctrl+V using xdotool or wtype.

    Tries xdotool first (works reliably on COSMIC/XWayland), then falls
    back to wtype if xdotool is unavailable.

    Returns:
        True if successful, False otherwise.
    """
    if shutil.which("xdotool"):
        _logger.debug("xdotool: sending Ctrl+V")
        result = subprocess.run(
            ["xdotool", "key", "ctrl+v"],
            capture_output=True,
            timeout=3,
        )
        if result.returncode == 0:
            _logger.debug("xdotool: paste succeeded")
            return True
        _logger.warning(
            "xdotool paste failed: %s",
            result.stderr.decode(errors="replace"),
        )

    _logger.debug("wtype: sending Ctrl+V")
    result = subprocess.run(
        ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"],
        capture_output=True,
        timeout=5,
    )
    if result.returncode == 0:
        _logger.debug("wtype: paste succeeded")
        return True

    _logger.warning(
        "wtype paste failed: %s",
        result.stderr.decode(errors="replace"),
    )
    return False


def _output_via_wtype_raw(text: str, config: Config) -> bool:
    """Output text using raw wtype character injection (legacy fallback).

    Args:
        text: The text to type.
        config: Configuration.

    Returns:
        True if successful, False otherwise.
    """
    use_soft_newlines = config.soft_newlines and "\n" in text
    ok = _type_with_wtype_soft_newlines(text) if use_soft_newlines else _type_with_wtype(text)
    if not ok:
        return False

    if config.sound_effects:
        play_sound("complete")
    return True


def _type_with_wtype(text: str) -> bool:
    """Type text using wtype without newline handling.

    Args:
        text: The text to type.

    Returns:
        True if successful, False otherwise.
    """
    result = subprocess.run(
        ["wtype", "--", text],
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        _logger.warning("wtype failed: %s", result.stderr.decode(errors="replace"))
        return False
    return True


def _type_with_wtype_soft_newlines(text: str) -> bool:
    """Type text using wtype with soft newlines (Shift+Enter).

    Intermediate newlines are sent as Shift+Enter (for text editors), while
    trailing newlines are sent as Enter.

    Args:
        text: The text to type, may contain newlines.

    Returns:
        True if successful, False otherwise.
    """
    has_trailing = text.endswith("\n")
    lines = text.split("\n")
    # Remove empty trailing element if text ended with newline
    if has_trailing and lines and lines[-1] == "":
        lines = lines[:-1]

    for i, line in enumerate(lines):
        if line:
            if not _type_with_wtype(line):
                return False

        is_last = i == len(lines) - 1
        if not is_last:
            # Soft newline: Shift+Enter
            if not _send_wtype_key("shift+Return"):
                _logger.warning("wtype shift+Return failed")
                return False
        elif has_trailing:
            # Final trailing newline: Enter
            if not _send_wtype_key("Return"):
                _logger.warning("wtype Return failed")
                return False

    return True


def _send_wtype_key(key: str) -> bool:
    """Send a key combination using wtype.

    Args:
        key: The key combination to send (e.g., "shift+Return").

    Returns:
        True if successful, False otherwise.
    """
    result = subprocess.run(
        ["wtype", "-k", key],
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0


def test_injection() -> bool:
    """Test if keyboard injection works.

    This is a lightweight probe that presses/release a modifier key.
    If it fails, we know injection doesn't work.

    Returns:
        True if injection appears to work, False otherwise.
    """
    global _injection_capable, _injection_checked_at
    now = time.monotonic()

    # Return cached result if still valid
    if (
        _injection_capable is not None
        and _injection_checked_at is not None
        and now - _injection_checked_at < _injection_cache_ttl
    ):
        return _injection_capable

    def cache_result(capable: bool) -> bool:
        global _injection_capable, _injection_checked_at
        _injection_capable = capable
        _injection_checked_at = now
        return capable

    # On Wayland, check for wtype
    if is_wayland():
        return cache_result(_has_wtype())

    if not _PYNPUT_AVAILABLE:
        _warn_pynput_missing()
        return cache_result(False)

    try:
        kb = get_keyboard()
        kb.press(Key.shift)
        kb.release(Key.shift)
        return cache_result(True)
    except Exception:
        return cache_result(False)


def output_text(
    text: str,
    window_info: Optional[WindowInfo] = None,
    config: Optional[Config] = None,
) -> bool:
    """Output transcribed text using the best available method.

    Args:
        text: The text to output.
        window_info: Optional window to restore focus to before typing.
        config: Configuration (uses default if not provided).

    Returns:
        True if text was output successfully, False otherwise.
    """
    if config is None:
        config = Config.load().validate()

    # Determine output mode
    mode = config.output_mode
    if mode == "auto":
        mode = "injection" if test_injection() else "clipboard"
        _logger.debug("Output mode auto-selected: %s", mode)

    if mode != "injection":
        return _output_via_clipboard(text, config)

    if not _PYNPUT_AVAILABLE:
        _warn_pynput_missing()
        return _output_via_clipboard(text, config)

    return _output_via_injection(text, window_info, config)


def _type_with_soft_newlines(kb: Controller, text: str) -> None:
    """Type text using Shift+Enter for intermediate newlines.

    Args:
        kb: Keyboard controller.
        text: Text to type, may contain newlines.
    """
    has_trailing = text.endswith("\n")
    lines = text.split("\n")
    # Remove empty trailing element if text ended with newline
    if has_trailing and lines and lines[-1] == "":
        lines = lines[:-1]

    for i, line in enumerate(lines):
        if line:
            kb.type(line)

        is_last = i == len(lines) - 1
        if not is_last:
            # Soft newline: Shift+Enter
            kb.press(Key.shift)
            kb.press(Key.enter)
            kb.release(Key.enter)
            kb.release(Key.shift)
        elif has_trailing:
            # Final trailing newline: real Enter
            kb.press(Key.enter)
            kb.release(Key.enter)


def _output_via_injection(
    text: str,
    window_info: Optional[WindowInfo],
    config: Config,
) -> bool:
    """Output text by simulating keyboard input.

    Args:
        text: The text to type.
        window_info: Window to restore focus to before typing.
        config: Configuration.

    Returns:
        True if successful, False otherwise.
    """
    # On Wayland, use wtype
    if is_wayland():
        if _has_wtype():
            return _output_via_wtype(text, config)
        _logger.warning("wtype not available; falling back to clipboard")
        return _output_via_clipboard(text, config)

    # Restore focus to original window if provided
    if window_info is not None:
        if not restore_focus(window_info):
            _logger.warning("Focus restore failed; falling back to clipboard")
            return _output_via_clipboard(text, config)

    try:
        kb = get_keyboard()
        if config.soft_newlines and "\n" in text:
            _type_with_soft_newlines(kb, text)
        else:
            kb.type(text)

        if config.sound_effects:
            play_sound("complete")

        return True
    except Exception:
        _logger.warning("Injection failed; falling back to clipboard", exc_info=True)
        return _output_via_clipboard(text, config)


def _output_via_clipboard(text: str, config: Config) -> bool:
    """Output text by copying to clipboard.

    Args:
        text: The text to copy.
        config: Configuration.

    Returns:
        True if successful, False otherwise.
    """
    try:
        try:
            import pyperclip
        except ImportError:
            _logger.error("pyperclip not installed; clipboard output unavailable")
            return False
        if hasattr(pyperclip, "is_available") and not pyperclip.is_available():
            _logger.error("No clipboard mechanism available")
            return False

        pyperclip.copy(text)

        if config.sound_effects:
            play_sound("complete")

        return True
    except Exception:
        if config.sound_effects:
            play_sound("error")
        _logger.warning("Clipboard output failed", exc_info=True)
        return False


def type_text_streaming(text: str) -> bool:
    """Type text character by character for streaming output.

    This is used during live transcription to show words as they're recognized.

    Args:
        text: The text to type.

    Returns:
        True if successful, False otherwise.
    """
    try:
        kb = get_keyboard()
        kb.type(text)
        return True
    except Exception:
        return False
