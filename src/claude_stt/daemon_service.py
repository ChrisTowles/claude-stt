"""Runtime daemon service for claude-stt."""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Optional

from .asr import ASREngine, create_engine
from .config import Config
from .errors import HotkeyError
from .hotkey import HotkeyListener
from .keyboard import delete_chars, type_text_streaming
from .sounds import SoundEvent, play_sound
from .window import get_active_window, WindowInfo


class STTDaemon:
    """Main daemon that coordinates all STT components."""

    def __init__(self, config: Optional[Config] = None):
        self.config = (config or Config.load()).validate()
        self._running = False
        self._recording = False

        # Components
        self._engine: Optional[ASREngine] = None
        self._hotkey: Optional[HotkeyListener] = None

        # Recording state
        self._record_start_time: float = 0
        # Updated on recording start and on every emitted-text change so the
        # silence auto-stop check can tell when the mic last produced anything.
        self._last_text_time: float = 0
        self._original_window: Optional[WindowInfo] = None
        self._typed_text: str = ""
        self._chars_typed: int = 0

        # Threading
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def _init_components(self) -> bool:
        try:
            self._engine = create_engine(
                model_id=self.config.model,
                chunk_ms=self.config.chunk_ms,
                context_seconds=self.config.context_seconds,
                silence_threshold_dbfs=self.config.silence_threshold_dbfs,
                silence_reset_seconds=self.config.silence_reset_seconds,
                input_device=self.config.input_device,
                on_text=self._on_text,
                on_error=self._on_recognition_error,
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

    def _on_text(self, text: str):
        """Handle interim/final text from the ASR engine."""
        if not self._recording:
            return
        text = text.strip()
        if not text or text == self._typed_text:
            return

        self._last_text_time = time.time()

        with self._lock:
            if text.startswith(self._typed_text):
                # Common case: new text extends what we typed — just append
                addition = text[len(self._typed_text):]
                type_text_streaming(addition)
            else:
                # Correction: find common prefix, delete the tail, type new tail.
                # NOTE: once recording exceeds `context_seconds` the ASR's
                # audio buffer rolls and `text` no longer covers the whole
                # utterance. In that regime this branch will incorrectly
                # delete earlier text. context_seconds defaults to 30s, which
                # covers typical dictations; for longer utterances pause and
                # resume to start a fresh recording.
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

    def _on_recognition_error(self, error: str):
        self._logger.warning("Speech recognition error: %s", error)
        if self.config.sound_effects:
            play_sound(SoundEvent.WARNING)

    def _on_recording_start(self):
        with self._lock:
            if self._recording:
                return

            self._typed_text = ""
            self._chars_typed = 0
            self._original_window = get_active_window()

            if self._original_window and self._original_window.app_name:
                app = self._original_window.app_name.lower()
                for excluded in self.config.excluded_apps:
                    if excluded.lower() in app:
                        self._logger.info("Skipping: %s is excluded", self._original_window.app_name)
                        if self._hotkey:
                            self._hotkey.reset_recording()
                        return

            if not self._engine or not self._engine.start():
                self._logger.error("Failed to start ASR engine")
                if self.config.sound_effects:
                    play_sound(SoundEvent.ERROR)
                return

            self._recording = True
            self._record_start_time = time.time()
            self._last_text_time = self._record_start_time
            self._logger.info("Recording started")
            if self.config.sound_effects:
                play_sound(SoundEvent.START)

    def _on_recording_stop(self):
        # Snapshot state under the lock, then release it before calling
        # `engine.stop()`. The engine's stop() blocks until its flush
        # (silence-pad + final emission) completes, and that flush calls
        # back into `self._on_text` which re-acquires the lock — so we
        # must not be holding it. We also keep `_recording=True` across
        # the flush so the callback's `if not self._recording: return`
        # gate lets the final text through.
        with self._lock:
            if not self._recording:
                return
            elapsed = time.time() - self._record_start_time
            self._logger.info("Recording stopped (%.1fs)", elapsed)
            engine = self._engine

        if engine is not None:
            engine.stop()

        with self._lock:
            self._recording = False
            if self.config.sound_effects:
                play_sound(SoundEvent.STOP)
            if self._typed_text:
                display = self._typed_text[:100] + "..." if len(self._typed_text) > 100 else self._typed_text
                self._logger.info("Final transcription: %s", display)

    def _check_max_recording_time(self) -> None:
        if not self._recording:
            return

        now = time.time()
        elapsed = now - self._record_start_time
        max_seconds = self.config.max_recording_seconds

        if max_seconds > 30 and max_seconds - 30 <= elapsed < max_seconds - 29:
            if self.config.sound_effects:
                play_sound(SoundEvent.WARNING)

        if elapsed >= max_seconds:
            self._on_recording_stop()
            return

        # Idle auto-stop: if no transcribed text has arrived for
        # `silence_auto_stop_seconds`, treat the session as forgotten
        # (toggle-mode "I walked away" case) and stop. The timer is
        # reset on every meaningful _on_text emission.
        idle_limit = self.config.silence_auto_stop_seconds
        if idle_limit > 0 and now - self._last_text_time >= idle_limit:
            self._logger.info(
                "Auto-stop: no new text in %ds (idle limit)", idle_limit
            )
            self._on_recording_stop()

    def run(self):
        self._logger.info("claude-stt daemon starting...")
        self._logger.info("Hotkey: %s", self.config.hotkey)
        self._logger.info("Mode: %s", self.config.mode)
        self._logger.info("Model: %s", self.config.model)

        if not self._init_components():
            raise SystemExit(1)

        if self._engine is not None:
            try:
                self._logger.info("Input device: %s", self._engine.describe_input_device())
            except Exception:
                self._logger.debug("could not resolve input device for logging", exc_info=True)

        # Pre-load the model so the first hotkey press is instant.
        try:
            self._engine.load()
        except Exception:
            self._logger.exception("Failed to load ASR model")
            raise SystemExit(1)

        self._logger.info("Ready.")
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

        if self._recording and self._engine:
            self._engine.stop()
            self._recording = False

        if self._hotkey:
            self._hotkey.stop()

        if self.config.sound_effects:
            play_sound(SoundEvent.SHUTDOWN)
        self._logger.info("claude-stt daemon stopped.")
