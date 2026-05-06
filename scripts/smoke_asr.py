"""Smoke test for the integrated ParakeetEngine.

Loads the engine, feeds it audio from a wav file (bypassing the mic), and
verifies that on_text fires with reasonable transcripts. Run from repo root:

    uv run python scripts/smoke_asr.py experiments/parakeet/audio/sample.wav
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from claude_stt.asr import ParakeetEngine


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: smoke_asr.py <16kHz mono wav>", file=sys.stderr)
        return 2

    wav = Path(sys.argv[1])
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"want 16kHz, got {sr}"

    received: list[tuple[float, str]] = []
    start = time.perf_counter()
    engine = ParakeetEngine(
        chunk_ms=320,
        context_seconds=10.0,
        on_text=lambda t: received.append((time.perf_counter() - start, t)),
        on_error=lambda e: print(f"ERROR: {e}", file=sys.stderr),
    )
    engine.load()

    # Skip the mic stream; feed the queue directly so the test is hermetic.
    chunk_n = engine.chunk_size
    engine._active = True
    import threading
    t = threading.Thread(target=engine._infer_loop, name="parakeet-infer", daemon=True)
    t.start()

    for offset in range(0, len(audio), chunk_n):
        chunk = audio[offset : offset + chunk_n]
        engine._audio_q.put(chunk.copy())
        # simulate live audio arrival
        time.sleep(engine.chunk_ms / 1000.0)

    engine._active = False
    engine._audio_q.put(None)
    t.join(timeout=10.0)

    if not received:
        print("FAIL: no transcripts received", file=sys.stderr)
        return 1

    print(f"received {len(received)} transcripts:")
    for ts, text in received[-5:]:
        print(f"  t+{ts:5.2f}s  {text}")

    final = received[-1][1].lower()
    expect = "phebe"
    if expect not in final:
        print(f"FAIL: expected {expect!r} in final transcript, got {final!r}", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
