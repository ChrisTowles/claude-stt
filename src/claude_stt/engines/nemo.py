"""Local streaming ASR engine using NVIDIA Parakeet via NeMo + CUDA.

Captures audio from the default microphone and runs Parakeet inference
on a rolling audio buffer. Emits the growing transcript via the
`on_text` callback after every inference pass, the same way the old
Chrome Web Speech path emitted interim results.

Notes
-----
Parakeet-TDT is an offline RNN-T model — there is no native cache-aware
streaming path. We approximate live behaviour by re-transcribing the
last `context_seconds` of audio after every chunk. With a 320 ms chunk
size on an RTX 3060, total perceived latency lands around 350 ms.

Silence-based buffer reset
--------------------------
The rolling raw-audio buffer would otherwise let background audio in a
long pause (TV across the room, music, hallway chatter) accumulate and
get re-transcribed alongside the user's actual speech. After
`silence_reset_seconds` of continuous below-gate audio we purge the
buffer and re-enter "waiting for speech" mode. Text already emitted is
preserved by carrying it forward in `_committed_text`, so the daemon's
prefix-diff typing logic never sees the transcript shrink.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

import numpy as np

from ._audio import resolve_input_device, rms_dbfs

_logger = logging.getLogger(__name__)


class ParakeetEngine:
    """Streaming Parakeet ASR engine driven by a local microphone."""

    def __init__(
        self,
        model_id: str = "nvidia/parakeet-tdt-0.6b-v2",
        sample_rate: int = 16000,
        chunk_ms: int = 320,
        context_seconds: float = 10.0,
        silence_threshold_dbfs: float = -45.0,
        silence_reset_seconds: float = 1.5,
        input_device: str | None = None,
        on_text: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        self.context_samples = int(sample_rate * context_seconds)
        self.silence_threshold_dbfs = silence_threshold_dbfs
        self.input_device = input_device
        self._silence_reset_chunks = max(1, int(silence_reset_seconds * 1000 / chunk_ms))
        self._on_text = on_text or (lambda _t: None)
        self._on_error = on_error or (lambda _e: None)

        self._model = None
        self._stream = None
        self._infer_thread: threading.Thread | None = None
        self._audio_q: queue.Queue = queue.Queue()
        self._active = False
        self._buffer = np.zeros(0, dtype=np.float32)
        self._speech_started = False
        self._silent_chunks = 0
        # Frozen transcript from previous segments (before the most recent
        # silence-triggered buffer reset). Emissions are `committed + " " +
        # current_buffer_text` so the daemon only ever sees the transcript
        # grow, even though our rolling buffer gets purged on long pauses.
        self._committed_text: str = ""

    def describe_input_device(self) -> str:
        """Resolve the configured input device to a friendly name (for logging)."""
        _, name = resolve_input_device(self.input_device)
        return name

    def load(self) -> None:
        """Load the Parakeet model. Slow on first call (~30s)."""
        if self._model is not None:
            return

        import torch
        from nemo.collections.asr.models import ASRModel

        _logger.info("Loading Parakeet model %s ...", self.model_id)
        model = ASRModel.from_pretrained(model_name=self.model_id).eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
            _logger.info("Parakeet running on CUDA: %s", torch.cuda.get_device_name(0))
        else:
            _logger.warning("CUDA unavailable; running Parakeet on CPU (will be slow)")
        self._model = model

    def start(self) -> bool:
        """Begin a recording session (open mic, spawn inference thread)."""
        if self._active:
            return True
        if self._model is None:
            self.load()

        import sounddevice as sd

        self._buffer = np.zeros(0, dtype=np.float32)
        self._speech_started = False
        self._silent_chunks = 0
        self._committed_text = ""
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

        device_index, device_name = resolve_input_device(self.input_device)
        _logger.info("Opening input device: %s", device_name)

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                device=device_index,
                callback=self._mic_callback,
            )
            self._stream.start()
        except Exception as exc:
            _logger.exception("Failed to open microphone")
            self._on_error(f"microphone: {exc}")
            return False

        self._active = True
        self._infer_thread = threading.Thread(
            target=self._infer_loop,
            name="parakeet-infer",
            daemon=True,
        )
        self._infer_thread.start()
        return True

    def stop(self) -> None:
        """End the current recording session."""
        if not self._active:
            return
        self._active = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                _logger.debug("error closing mic stream", exc_info=True)
            self._stream = None

        self._audio_q.put(None)
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=5.0)
            self._infer_thread = None

    def _mic_callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            _logger.debug("sounddevice status: %s", status)
        if not self._active:
            return
        self._audio_q.put(indata[:, 0].copy())

    def _infer_loop(self) -> None:
        import torch

        while self._active:
            try:
                chunk = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if chunk is None:
                break

            while True:
                try:
                    extra = self._audio_q.get_nowait()
                except queue.Empty:
                    break
                if extra is None:
                    self._active = False
                    break
                chunk = np.concatenate([chunk, extra])

            chunk_db = rms_dbfs(chunk)
            is_silent = chunk_db <= self.silence_threshold_dbfs

            # Energy gate: don't transcribe until we hear something above the
            # noise floor. Parakeet hallucinates ("yeah", "you", "thank you")
            # on pure silence, which leaks into the typed output.
            if not self._speech_started:
                if is_silent:
                    continue
                self._speech_started = True
                self._silent_chunks = 0

            self._buffer = np.concatenate([self._buffer, chunk])
            if len(self._buffer) > self.context_samples:
                self._buffer = self._buffer[-self.context_samples :]

            # Track post-speech silence so we can purge the rolling buffer
            # on long pauses (e.g. TV bleed during a thinking pause).
            if is_silent:
                self._silent_chunks += 1
            else:
                self._silent_chunks = 0

            try:
                out = self._model.transcribe(
                    [self._buffer], batch_size=1, verbose=False
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as exc:
                _logger.exception("Parakeet inference failed")
                self._on_error(str(exc))
                continue

            text = _extract_text(out)
            if text:
                full = _join(self._committed_text, text)
                try:
                    self._on_text(full)
                except Exception:
                    _logger.exception("on_text callback failed")

            if self._silent_chunks >= self._silence_reset_chunks:
                # Sustained silence -> freeze the current segment into
                # committed text and start fresh. The next non-silent
                # chunk will begin a new segment whose emissions extend
                # `committed`, so the daemon's prefix-diff appends
                # cleanly without rewriting earlier text.
                if text:
                    self._committed_text = _join(self._committed_text, text)
                self._buffer = np.zeros(0, dtype=np.float32)
                self._silent_chunks = 0
                self._speech_started = False
                _logger.debug(
                    "nemo: buffer reset after %d silent chunks; committed=%r",
                    self._silence_reset_chunks,
                    self._committed_text[-60:],
                )


def _join(committed: str, new: str) -> str:
    if not committed:
        return new
    if not new:
        return committed
    return f"{committed} {new}".strip()


def _extract_text(transcribe_output) -> str:
    """Pull text out of NeMo's varied transcribe() return shapes."""
    if not transcribe_output:
        return ""
    if isinstance(transcribe_output, tuple):
        transcribe_output = transcribe_output[0]
    if not transcribe_output:
        return ""
    first = transcribe_output[0]
    if isinstance(first, str):
        return first
    text = getattr(first, "text", None)
    if text is not None:
        return text
    return str(first)
