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
        # Set by the worker once the per-session flush (silence pad +
        # final emission) has completed. `stop()` blocks on this so the
        # daemon doesn't drop the flush by clearing `_recording` first.
        self._flush_done = threading.Event()
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

        # Close the mic first so no more audio chunks queue up.
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                _logger.debug("error closing mic stream", exc_info=True)
            self._stream = None

        # Clear the session flag *and* push a None sentinel — the worker's
        # inner loop exits on either, runs its silence-pad flush, then
        # sets `_flush_done`. We block here so the daemon doesn't clear
        # `_recording=False` (and start dropping callbacks) before the
        # final emission lands.
        self._session_active.clear()
        self._audio_q.put(None)
        if not self._flush_done.wait(timeout=5.0):
            _logger.warning("MLX flush did not complete within 5s")

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
        # Append-only emission: we feed the daemon `result.text` (finalized
        # + draft) but only when it extends what we last emitted. If a
        # draft churn rewrites earlier text we drop that emission rather
        # than triggering delete-and-retype, which would clobber any
        # manual edits the user made mid-dictation.
        #
        # Reset session-scoped state here (not just in start()) so an
        # abnormal exit from a prior session can't leave stale prefix
        # state that would silently freeze the new session's emissions.
        self._last_emitted = ""
        self._speech_started = False
        self._flush_done.clear()
        chunks_seen = 0
        try:
            with model.transcribe_stream(context_size=(256, 256)) as transcriber:
                # Inner-loop guard checks `_session_active` so a `stop()`
                # that clears the flag (after waking us with the None
                # sentinel) can't leave a phantom _run_session running
                # past the end of the user's recording.
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
                            self._audio_q.put(None)
                            break
                        chunk = np.concatenate([chunk, extra])

                    chunks_seen += 1
                    chunk_db = _rms_dbfs(chunk)

                    # Energy gate: silence-pad before first speech to avoid
                    # the same hallucinated wake-words ("you", "yeah") that
                    # plague the NeMo backend on quiet leading audio.
                    if not self._speech_started:
                        if chunk_db > self.silence_threshold_dbfs:
                            self._speech_started = True
                            _logger.debug(
                                "mlx: speech detected (chunk #%d, %.1f dBFS > %.1f)",
                                chunks_seen, chunk_db, self.silence_threshold_dbfs,
                            )
                        else:
                            if chunks_seen % 10 == 1:
                                # Periodic ping so users can see whether
                                # their mic audio is actually below the
                                # gate without flooding the log.
                                _logger.debug(
                                    "mlx: below silence gate (chunk #%d, %.1f dBFS, gate %.1f)",
                                    chunks_seen, chunk_db, self.silence_threshold_dbfs,
                                )
                            continue

                    try:
                        transcriber.add_audio(mx.array(chunk))
                    except Exception as exc:
                        _logger.exception("Parakeet-MLX inference failed")
                        self._on_error(str(exc))
                        continue

                    _logger.debug(
                        "mlx: add_audio chunk #%d (%.1f dBFS) -> result=%r",
                        chunks_seen, chunk_db, transcriber.result.text[:80],
                    )
                    self._emit_if_extending(transcriber)

                # Session ending — pad silence so the encoder's right-context
                # window can stabilize the trailing draft tokens before we
                # exit the streaming context, then do a final emission.
                try:
                    silence = mx.zeros(self.sample_rate, dtype=mx.float32)
                    transcriber.add_audio(silence)
                except Exception:
                    _logger.debug("flush silence pad failed", exc_info=True)
                self._emit_if_extending(transcriber)
        finally:
            # Clear `_session_active` here (instead of relying solely on
            # the caller of stop()) so the worker's outer loop can't
            # immediately re-enter _run_session — that race would either
            # eat the next session's chunks or set _flush_done stale and
            # confuse a waiter that's just about to start a new session.
            self._session_active.clear()
            self._flush_done.set()

    def _emit_if_extending(self, transcriber) -> None:
        """Bounded-retype emission gate.

        Parakeet's streaming output keeps revising the trailing draft
        tokens chunk-to-chunk. A strict append-only gate freezes after
        the first word, because every subsequent draft revises a few
        characters back (capitalization, punctuation, last-word
        substitution). A naive "always emit" path lets the daemon's
        prefix-diff retype unboundedly, clobbering manual edits made
        seconds ago.

        Compromise: emit whatever the model says, but only when the
        retype window (chars we'd rewrite) is small. Edits beyond that
        window are protected — the engine drops the emission rather
        than clobber. Recent text (~last word) may still flicker.
        """
        text = transcriber.result.text  # AlignedResult already strips
        if not text or text == self._last_emitted:
            return

        prefix_len = _case_insensitive_lcp_len(self._last_emitted, text)
        retype_chars = len(self._last_emitted) - prefix_len

        if retype_chars > MAX_RETYPE_CHARS:
            _logger.debug(
                "mlx skip: retype too far (%d > %d) last=%r new=%r lcp=%d",
                retype_chars, MAX_RETYPE_CHARS,
                self._last_emitted[-60:], text[:60], prefix_len,
            )
            return

        if prefix_len == len(text):
            return  # text is a case-only prefix of _last_emitted; nothing new

        # Preserve the casing of what's already typed up to the LCP, then
        # append the model's new content past the LCP. The daemon's
        # prefix-diff will issue at most `retype_chars` backspaces.
        new_emitted = self._last_emitted[:prefix_len] + text[prefix_len:]
        if new_emitted == self._last_emitted:
            return
        self._last_emitted = new_emitted
        try:
            self._on_text(new_emitted)
        except Exception:
            _logger.exception("on_text callback failed")


# Words rarely run longer than ~20 chars; leave headroom for short
# multi-word draft revisions ("the test" → "a test" is 8 chars). Bigger
# than this means the model is rewriting something the user has likely
# moved past — protect their edits instead.
MAX_RETYPE_CHARS = 40


def _case_insensitive_lcp_len(a: str, b: str) -> int:
    """Length of the longest common prefix of `a` and `b`, case-insensitive."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i].lower() != b[i].lower():
            return i
    return n


def _rms_dbfs(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))) + 1e-12)
    return 20.0 * np.log10(rms)
