"""Audio fixture helpers for MLX streaming tests.

Synthesizes test audio via macOS `say`, plus utility builders for
silence, noise, and mixing — so tests can drive the engine through
controlled scenarios (background noise, mid-utterance pauses, trailing
silence) without depending on a live microphone.

Sample rate is fixed at 16000 Hz to match Parakeet's preprocessor.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def have_say() -> bool:
    return shutil.which("say") is not None


def synth_speech(phrase: str, voice: str | None = None) -> np.ndarray:
    """Render `phrase` to a 16 kHz mono float32 numpy array via macOS `say`.

    The output is normalized to roughly speech RMS (the `say` voice ships
    around -20 dBFS peak, so the energy gate at -45 dBFS clears easily).
    """
    if not have_say():
        raise RuntimeError("`say` binary not available — required for synth_speech")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        cmd = ["say", "-o", str(wav_path), "--data-format=LEI16@16000"]
        if voice:
            cmd += ["-v", voice]
        cmd += [phrase]
        subprocess.run(cmd, check=True, capture_output=True)
        return _read_wav_float32(wav_path)
    finally:
        wav_path.unlink(missing_ok=True)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def noise(seconds: float, dbfs: float, *, seed: int = 0) -> np.ndarray:
    """White noise at approximately the given dBFS RMS level."""
    rms = 10.0 ** (dbfs / 20.0)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, rms, size=int(SAMPLE_RATE * seconds)).astype(np.float32)


def concat(*signals: np.ndarray) -> np.ndarray:
    """Concatenate audio segments end-to-end."""
    return np.concatenate(signals).astype(np.float32, copy=False)


def mix(*signals: np.ndarray) -> np.ndarray:
    """Sum signals sample-wise (truncating to the shortest length)."""
    n = min(s.size for s in signals)
    out = np.zeros(n, dtype=np.float32)
    for s in signals:
        out += s[:n]
    return out


def _read_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SAMPLE_RATE:
            raise RuntimeError(f"unexpected sample rate: {w.getframerate()}")
        if w.getnchannels() != 1:
            raise RuntimeError(f"unexpected channels: {w.getnchannels()}")
        if w.getsampwidth() != 2:
            raise RuntimeError(f"unexpected sample width: {w.getsampwidth()}")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def run_session(
    engine,
    audio: np.ndarray,
    *,
    timeout: float = 30.0,
    pace: bool = True,
) -> list[str]:
    """Drive `engine` through one streaming session with `audio` as input.

    Bypasses the microphone by injecting chunks directly into the audio
    queue. Returns the list of strings the engine emitted via `_on_text`,
    in order. Engine must already have `load()`'d its model.

    `pace=True` (default) sleeps `chunk_ms` between chunks so the
    worker's drain logic doesn't concatenate everything into one
    mega-chunk and emit only once. Set False for "blast" tests that
    want to exercise the drain path explicitly.
    """
    emitted: list[str] = []
    original_on_text = engine._on_text
    engine._on_text = emitted.append
    chunk_interval_s = engine.chunk_ms / 1000.0 if pace else 0.0
    try:
        # Clear any stale flush_done from a prior session BEFORE we set
        # _session_active, so our `_flush_done.wait()` below can only
        # return when this session's flush actually completes.
        engine._flush_done.clear()
        engine._session_active.set()
        for start in range(0, len(audio), engine.chunk_size):
            engine._audio_q.put(
                audio[start : start + engine.chunk_size].astype(np.float32, copy=False)
            )
            if chunk_interval_s:
                time.sleep(chunk_interval_s)
        # The None sentinel signals the worker to break the inner loop,
        # run its silence-pad flush, and set `_flush_done`. We do NOT
        # clear `_session_active` here — _run_session clears it itself
        # in its finally block to prevent phantom restarts.
        engine._audio_q.put(None)
        if not engine._flush_done.wait(timeout=timeout):
            raise RuntimeError(f"flush_done not set within {timeout}s")
    finally:
        engine._on_text = original_on_text
        engine._session_active.clear()
    return list(emitted)


class EngineHarness:
    """One-shot lifecycle helper for use in unittest setUpClass / tearDownClass."""

    def __init__(self) -> None:
        from claude_stt.engines.mlx import ParakeetMLXEngine

        self._engine = ParakeetMLXEngine()
        self._loaded = False

    @property
    def engine(self):
        return self._engine

    def load(self, timeout: float = 600.0) -> None:
        if self._loaded:
            return
        # Run load in a thread because ParakeetMLXEngine.load() blocks on
        # the worker thread reaching its loaded-event; raise if too slow.
        done = threading.Event()
        err: list[BaseException] = []

        def _do() -> None:
            try:
                self._engine.load()
            except BaseException as exc:  # noqa: BLE001 — propagate
                err.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=_do, name="harness-load", daemon=True)
        t.start()
        if not done.wait(timeout=timeout):
            raise RuntimeError(f"engine load did not complete within {timeout}s")
        if err:
            raise err[0]
        self._loaded = True

    def shutdown(self) -> None:
        self._engine._shutdown.set()
        # The worker thread is a daemon, so it'll die with the process.
        # No explicit join — keeps tearDownClass fast.
        self._loaded = False
