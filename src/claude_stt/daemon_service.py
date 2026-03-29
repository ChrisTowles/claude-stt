"""Runtime daemon service for claude-stt."""

from __future__ import annotations

import logging
import signal
import shutil
import subprocess
import threading
import time
from typing import Optional

from .config import Config
from .errors import HotkeyError
from .hotkey import HotkeyListener
from .keyboard import delete_chars, type_text_streaming
from .sounds import SoundEvent, play_sound
from .web.server import WebSpeechServer
from .window import get_active_window, WindowInfo


class STTDaemon:
    """Main daemon that coordinates all STT components."""

    def __init__(self, config: Optional[Config] = None):
        self.config = (config or Config.load()).validate()
        self._running = False
        self._recording = False

        # Components
        self._server: Optional[WebSpeechServer] = None
        self._hotkey: Optional[HotkeyListener] = None

        # Recording state
        self._record_start_time: float = 0
        self._original_window: Optional[WindowInfo] = None
        self._typed_text: str = ""
        self._chars_typed: int = 0

        # Threading
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def _init_components(self) -> bool:
        try:
            self._server = WebSpeechServer(
                port=self.config.ws_port,
                on_interim=self._on_text,
                on_final=self._on_text,
                on_end=self._on_recognition_end,
                on_error=self._on_recognition_error,
                on_ready=self._on_chrome_ready,
            )

            self._hotkey = HotkeyListener(
                hotkey=self.config.hotkey,
                on_start=self._on_recording_start,
                on_stop=self._on_recording_stop,
                mode=self.config.mode,
            )
        except HotkeyError as exc:
            self._logger.warning("Hotkey listener failed: %s (SIGUSR1 toggle still works)", exc)
            self._hotkey = None
        except Exception as exc:
            self._logger.error("%s", exc)
            return False

        return True

    def _launch_chrome(self):
        chrome = shutil.which("google-chrome") or shutil.which("chromium-browser") or shutil.which("chromium")
        if not chrome:
            self._logger.warning("Chrome not found — open http://127.0.0.1:%d manually", self.config.ws_port)
            return
        url = self._server.url
        try:
            subprocess.Popen(
                [chrome, f"--app={url}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._logger.info("Launched Chrome app window: %s", url)
        except Exception:
            self._logger.warning("Failed to launch Chrome — open %s manually", url)

    def _on_chrome_ready(self):
        self._logger.info("Chrome Web Speech API connected and ready")

    def _on_text(self, text: str):
        """Handle interim or final text from Chrome."""
        if not self._recording:
            return
        text = text.strip()
        if not text or text == self._typed_text:
            return

        with self._lock:
            if text.startswith(self._typed_text):
                # Common case: new text extends what we typed — just append
                addition = text[len(self._typed_text):]
                type_text_streaming(addition)
            else:
                # Correction: find common prefix, delete the tail, type new tail
                common = 0
                for i in range(min(len(self._typed_text), len(text))):
                    if self._typed_text[i] == text[i]:
                        common = i + 1
                    else:
                        break
                chars_to_delete = self._chars_typed - common
                new_tail = text[common:]
                if chars_to_delete > 0:
                    delete_chars(chars_to_delete)
                if new_tail:
                    type_text_streaming(new_tail)
            self._chars_typed = len(text)
            self._typed_text = text

    def _on_recognition_end(self):
        if self._recording:
            self._logger.debug("Recognition ended but still recording (Chrome will auto-restart)")

    def _on_recognition_error(self, error: str):
        self._logger.warning("Speech recognition error: %s", error)
        if error == "not-allowed":
            self._logger.error("Microphone access denied in Chrome — grant permission and reload")
        if self.config.sound_effects and error != "no-speech":
            play_sound(SoundEvent.WARNING)

    def _on_recording_start(self):
        with self._lock:
            if self._recording:
                return

            self._recording = True
            self._record_start_time = time.time()
            self._typed_text = ""
            self._chars_typed = 0
            self._original_window = get_active_window()

            if self._original_window and self._original_window.app_name:
                app = self._original_window.app_name.lower()
                for excluded in self.config.excluded_apps:
                    if excluded.lower() in app:
                        self._logger.info("Skipping: %s is excluded", self._original_window.app_name)
                        self._recording = False
                        if self._hotkey:
                            self._hotkey.reset_recording()
                        return

            if not self._server or not self._server.is_connected():
                self._logger.error("Chrome not connected — open %s", self._server.url)
                self._recording = False
                if self.config.sound_effects:
                    play_sound(SoundEvent.ERROR)
                return

            self._server.send_start()
            self._logger.info("Recording started")
            if self.config.sound_effects:
                play_sound(SoundEvent.COMPLETE)

    def _on_recording_stop(self):
        with self._lock:
            if not self._recording:
                return

            self._recording = False
            elapsed = time.time() - self._record_start_time
            self._logger.info("Recording stopped (%.1fs)", elapsed)

            if self._server and self._server.is_connected():
                self._server.send_stop()

            if self.config.sound_effects:
                play_sound(SoundEvent.STOP)

            if self._typed_text:
                display = self._typed_text[:100] + "..." if len(self._typed_text) > 100 else self._typed_text
                self._logger.info("Final transcription: %s", display)

    def _check_max_recording_time(self) -> None:
        if not self._recording:
            return

        elapsed = time.time() - self._record_start_time
        max_seconds = self.config.max_recording_seconds

        if max_seconds > 30 and max_seconds - 30 <= elapsed < max_seconds - 29:
            if self.config.sound_effects:
                play_sound(SoundEvent.WARNING)

        if elapsed >= max_seconds:
            self._on_recording_stop()

    def run(self):
        self._logger.info("claude-stt daemon starting...")
        self._logger.info("Hotkey: %s", self.config.hotkey)
        self._logger.info("Mode: %s", self.config.mode)
        self._logger.info("Web server port: %d", self.config.ws_port)

        if not self._init_components():
            raise SystemExit(1)

        if not self._server.start():
            self._logger.error("Failed to start web server")
            raise SystemExit(1)

        self._launch_chrome()

        self._logger.info("Ready. Waiting for Chrome to connect at %s", self._server.url)
        if self.config.sound_effects:
            play_sound(SoundEvent.READY)

        if self._hotkey and not self._hotkey.start():
            self._logger.warning("Hotkey listener failed to start (SIGUSR1 toggle still works)")

        self._running = True

        def shutdown(signum, frame):
            self._logger.info("Shutting down...")
            self._running = False

        _last_toggle = [0.0]

        def toggle_recording(signum, frame):
            now = time.monotonic()
            if now - _last_toggle[0] < 1.0:
                return
            _last_toggle[0] = now
            if self._recording:
                self._logger.info("SIGUSR1: stopping recording")
                self._on_recording_stop()
            else:
                self._logger.info("SIGUSR1: starting recording")
                self._on_recording_start()

        try:
            signal.signal(signal.SIGINT, shutdown)
            signal.signal(signal.SIGTERM, shutdown)
            signal.signal(signal.SIGUSR1, toggle_recording)
        except Exception:
            self._logger.debug("Signal handlers unavailable", exc_info=True)

        try:
            while self._running:
                self._check_max_recording_time()
                time.sleep(0.1)
        finally:
            self.stop()

    def stop(self):
        self._running = False

        if self._recording and self._server:
            self._server.send_stop()

        if self._hotkey:
            self._hotkey.stop()

        if self._server:
            self._server.stop()

        if self.config.sound_effects:
            play_sound(SoundEvent.SHUTDOWN)
        self._logger.info("claude-stt daemon stopped.")
