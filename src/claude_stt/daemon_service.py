"""Runtime daemon service for claude-stt."""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Optional

import numpy as np

from .config import Config
from .engines.qwen_asr import QwenASREngine
from .errors import EngineError, HotkeyError, RecorderError
from .hotkey import HotkeyListener
from .keyboard import delete_chars, type_text_streaming, output_text
from .recorder import AudioRecorder, RecorderConfig
from .sounds import SoundEvent, play_sound
from .window import get_active_window, WindowInfo


class STTDaemon:
    """Main daemon that coordinates all STT components."""

    def __init__(self, config: Optional[Config] = None):
        self.config = (config or Config.load()).validate()
        self._running = False
        self._recording = False

        # Components
        self._recorder: Optional[AudioRecorder] = None
        self._engine: Optional[QwenASREngine] = None
        self._hotkey: Optional[HotkeyListener] = None

        # Recording/streaming state
        self._record_start_time: float = 0
        self._original_window: Optional[WindowInfo] = None
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_streaming = threading.Event()
        self._chars_typed: int = 0

        # Threading
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._logger = logging.getLogger(__name__)

    def _init_components(self) -> bool:
        try:
            self._recorder = AudioRecorder(
                RecorderConfig(
                    sample_rate=self.config.sample_rate,
                    max_recording_seconds=self.config.max_recording_seconds,
                    device=self.config.audio_device,
                )
            )
            if not self._recorder.is_available():
                raise RecorderError("No audio input device available")

            # Log audio device
            try:
                import sounddevice as sd
                if self.config.audio_device is not None:
                    device_info = sd.query_devices(self.config.audio_device)
                    self._logger.info(
                        "Audio input: [%s] %s", self.config.audio_device, device_info['name']
                    )
                else:
                    device_info = sd.query_devices(kind='input')
                    self._logger.info("Audio input: %s (default)", device_info['name'])
            except Exception:
                self._logger.debug("Could not query audio device info", exc_info=True)

            self._engine = QwenASREngine(model_name=self.config.stt_model)
            if not self._engine.is_available():
                raise EngineError(
                    "STT engine not available. Run setup to install dependencies."
                )

            self._hotkey = HotkeyListener(
                hotkey=self.config.hotkey,
                on_start=self._on_recording_start,
                on_stop=self._on_recording_stop,
                mode=self.config.mode,
            )

        except (RecorderError, EngineError, HotkeyError) as exc:
            self._logger.error("%s", exc)
            return False

        return True

    def _play_warning(self) -> None:
        if self.config.sound_effects:
            play_sound(SoundEvent.WARNING)

    def _on_recording_start(self):
        with self._lock:
            if self._recording:
                return

            self._recording = True
            self._record_start_time = time.time()
            self._original_window = get_active_window()

            # Check if the focused app is excluded
            if self._original_window and self._original_window.app_name:
                app = self._original_window.app_name.lower()
                for excluded in self.config.excluded_apps:
                    if excluded.lower() in app:
                        self._logger.info(
                            "Skipping recording: %s is excluded", self._original_window.app_name
                        )
                        self._recording = False
                        if self._hotkey:
                            self._hotkey.reset_recording()
                        return

            # Start recording
            if self._recorder and self._recorder.start():
                self._logger.info("Recording started")
                if self.config.sound_effects:
                    play_sound(SoundEvent.COMPLETE)
            else:
                self._logger.error("Audio recorder failed to start")
                self._recording = False
                if self.config.sound_effects:
                    play_sound(SoundEvent.ERROR)
                return

        # Start streaming transcription thread
        self._stop_streaming.clear()
        self._chars_typed = 0
        self._streaming_thread = threading.Thread(
            target=self._streaming_worker,
            name="claude-stt-streaming",
            daemon=True,
        )
        self._streaming_thread.start()

    def _streaming_worker(self):
        """Stream audio chunks to engine and type partial results in real-time."""
        state = self._engine.init_streaming()
        if state is None:
            self._logger.error("Failed to init streaming state")
            return

        prev_text = ""

        while not self._stop_streaming.is_set():
            chunk = self._recorder.get_chunk(timeout=0.1)
            if chunk is None:
                continue

            chunk = np.squeeze(chunk)
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)

            new_text = self._engine.stream_chunk(chunk, state)

            if new_text and new_text != prev_text:
                # Delete previous partial text and type new
                if self._chars_typed > 0:
                    delete_chars(self._chars_typed)
                type_text_streaming(new_text)
                self._chars_typed = len(new_text)
                prev_text = new_text

        # Finalize: flush remaining audio
        final_text = self._engine.finish_stream(state)
        if final_text:
            if self._chars_typed > 0:
                delete_chars(self._chars_typed)
            self._chars_typed = 0

            final_text = final_text.strip()
            if final_text:
                display = final_text[:100] + "..." if len(final_text) > 100 else final_text
                self._logger.info("Final transcription: %s", display)
                output_text(final_text, self._original_window, self.config)
            else:
                self._logger.info("No speech detected")
                self._play_warning()
        else:
            if self._chars_typed > 0:
                delete_chars(self._chars_typed)
                self._chars_typed = 0
            self._logger.info("No speech detected")
            self._play_warning()

    def _on_recording_stop(self):
        with self._lock:
            if not self._recording:
                return

            self._recording = False
            elapsed = time.time() - self._record_start_time

            # Stop recording
            if self._recorder:
                self._recorder.stop()

            self._logger.info("Recording stopped (%.1fs)", elapsed)
            if self.config.sound_effects:
                play_sound(SoundEvent.STOP)

        # Signal streaming thread to finalize
        self._stop_streaming.set()
        if self._streaming_thread:
            self._streaming_thread.join(timeout=10.0)
            if self._streaming_thread.is_alive():
                self._logger.warning("Streaming thread did not exit cleanly")
            self._streaming_thread = None

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
        self._logger.info("Engine: qwen-asr (%s)", self.config.stt_model)
        self._logger.info("Mode: %s", self.config.mode)

        if not self._init_components():
            raise SystemExit(1)

        self._logger.info("Loading STT model...")
        if not self._engine.load_model():
            self._logger.error("Failed to load STT model")
            raise SystemExit(1)

        self._logger.info(
            "Model loaded. Ready for voice input. Hotkey: %s",
            self.config.hotkey,
        )
        if self.config.sound_effects:
            play_sound(SoundEvent.READY)

        if not self._hotkey.start():
            self._logger.error("Failed to start hotkey listener")
            raise SystemExit(1)

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
        self._stop_event.set()
        self._stop_streaming.set()

        if self._streaming_thread:
            self._streaming_thread.join(timeout=5.0)

        if self._recording and self._recorder:
            self._recorder.stop()

        if self._hotkey:
            self._hotkey.stop()

        if self.config.sound_effects:
            play_sound(SoundEvent.SHUTDOWN)
        self._logger.info("claude-stt daemon stopped.")
