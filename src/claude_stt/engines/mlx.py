"""Local streaming ASR engine using Parakeet-MLX on Apple Silicon.

Captures audio from the default microphone and feeds chunks into the
parakeet-mlx streaming context, which keeps a stateful encoder cache and
exposes the growing transcript via `transcriber.result.text` after each
chunk.

Threading note
--------------
MLX arrays are bound to the thread that created them — using a model
loaded on thread A from thread B raises "There is no Stream(gpu, 0) in
current thread." The engine therefore runs a single long-lived worker
thread that owns the model: it loads on first `load()`, then services
each recording session by entering a fresh `transcribe_stream` context.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

import numpy as np

_logger = logging.getLogger(__name__)


class ParakeetMLXEngine:
    """Streaming Parakeet-MLX ASR engine driven by a local microphone."""

    def __init__(
        self,
        model_id: str = "mlx-community/parakeet-tdt-0.6b-v3",
        chunk_ms: int = 320,
        context_seconds: float = 30.0,  # noqa: ARG002 — kept for parity, not used by MLX backend
        silence_threshold_dbfs: float = -45.0,
        on_text: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.chunk_ms = chunk_ms
        self.silence_threshold_dbfs = silence_threshold_dbfs
        self._on_text = on_text or (lambda _t: None)
        self._on_error = on_error or (lambda _e: None)

        self.sample_rate: int = 16000
        self.chunk_size: int = int(self.sample_rate * chunk_ms / 1000)

        self._stream = None  # sounddevice InputStream
        self._audio_q: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._session_active = threading.Event()
        self._shutdown = threading.Event()
        self._load_done = threading.Event()
        self._load_error: BaseException | None = None
        self._speech_started = False
        self._last_emitted = ""

    def load(self) -> None:
        """Spin up the worker thread (which loads the model). Blocks until ready."""
        if self._worker_thread is None:
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="parakeet-mlx-worker",
                daemon=True,
            )
            self._worker_thread.start()
        # Model download is ~600 MB on a cold cache; allow generous time.
        if not self._load_done.wait(timeout=600):
            raise RuntimeError("Parakeet-MLX model load timed out")
        if self._load_error is not None:
            raise self._load_error

    def start(self) -> bool:
        if self._session_active.is_set():
            return True
        if not self._load_done.is_set():
            try:
                self.load()
            except Exception as exc:
                self._on_error(f"model load failed: {exc}")
                return False

        import sounddevice as sd

        # Drain any residual audio from a prior aborted session.
        self._drain_audio_q()
        self._speech_started = False
        self._last_emitted = ""

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=self._mic_callback,
            )
            self._stream.start()
        except Exception as exc:
            _logger.exception("Failed to open microphone")
            self._on_error(f"microphone: {exc}")
            return False

        self._session_active.set()
        return True

    def stop(self) -> None:
        if not self._session_active.is_set():
            return
        self._session_active.clear()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                _logger.debug("error closing mic stream", exc_info=True)
            self._stream = None

        # Wake the worker out of its blocking get() so it can exit the
        # transcribe_stream context cleanly and return to the idle wait.
        self._audio_q.put(None)

    def _drain_audio_q(self) -> None:
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

    def _mic_callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            _logger.debug("sounddevice status: %s", status)
        if not self._session_active.is_set():
            return
        self._audio_q.put(indata[:, 0].copy())

    def _worker_loop(self) -> None:
        import mlx.core as mx
        from parakeet_mlx import from_pretrained

        try:
            with mx.stream(mx.gpu):
                _logger.info("Loading Parakeet-MLX model %s ...", self.model_id)
                model = from_pretrained(self.model_id)
                self.sample_rate = int(model.preprocessor_config.sample_rate)
                self.chunk_size = int(self.sample_rate * self.chunk_ms / 1000)
                _logger.info("Parakeet-MLX ready (sample_rate=%d Hz)", self.sample_rate)
                self._load_done.set()

                while not self._shutdown.is_set():
                    if not self._session_active.wait(timeout=0.5):
                        continue
                    self._run_session(mx, model)
        except Exception as exc:
            _logger.exception("Parakeet-MLX worker crashed")
            self._load_error = exc
            self._load_done.set()
            self._on_error(str(exc))

    def _run_session(self, mx, model) -> None:
        with model.transcribe_stream(context_size=(256, 256)) as transcriber:
            while self._session_active.is_set():
                try:
                    chunk = self._audio_q.get(timeout=0.5)
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
                        break
                    chunk = np.concatenate([chunk, extra])

                # Energy gate: silence-pad before first speech to avoid
                # the same hallucinated wake-words ("you", "yeah") that
                # plague the NeMo backend on quiet leading audio.
                if not self._speech_started:
                    if _rms_dbfs(chunk) > self.silence_threshold_dbfs:
                        self._speech_started = True
                    else:
                        continue

                try:
                    transcriber.add_audio(mx.array(chunk))
                    text = transcriber.result.text
                except Exception as exc:
                    _logger.exception("Parakeet-MLX inference failed")
                    self._on_error(str(exc))
                    continue

                if text and text != self._last_emitted:
                    self._last_emitted = text
                    try:
                        self._on_text(text)
                    except Exception:
                        _logger.exception("on_text callback failed")


def _rms_dbfs(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))) + 1e-12)
    return 20.0 * np.log10(rms)
