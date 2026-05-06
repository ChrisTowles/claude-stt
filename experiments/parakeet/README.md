# Parakeet research spike

Goal: decide whether to replace Chrome Web Speech with NVIDIA Parakeet-TDT-0.6B-v2
running locally on this Linux box (Ryzen 9800X3D + RTX 3060 12GB).

## Verdict: yes — clear win on this hardware.

Run on 2026-05-05. NeMo 2.x, torch 2.6+cu124, RTX 3060 12GB.

## Batch numbers (`bench.py` on 7.43s LibriSpeech sample)

| metric        | value                       |
|---------------|-----------------------------|
| avg latency   | **60.8 ms** for 7.43s audio |
| RTF           | **0.008** (122× real-time)  |
| VRAM (peak)   | 5.0 GB                      |
| model load    | ~35 s (cold), cached after  |
| transcript    | perfect on first try        |

The first run after a cold start pays a one-time ~35s model load + cuDNN
autotune. After that, every inference is ~60ms. VRAM headroom is fine on
the 12GB 3060 — the model itself is ~600MB, the rest is PyTorch's
allocator.

## Streaming numbers (`stream.py`, naive rolling-buffer approach)

320 ms audio chunks, 4 s context window:

| metric                | value           |
|-----------------------|-----------------|
| avg inference / chunk | **34.1 ms**     |
| p95 inference / chunk | 37.3 ms         |
| perceived latency     | ~350 ms total   |
| consistency           | very tight      |

End-to-end "audio captured → text appears" is `chunk_size + infer_time`,
i.e. ~350ms with these settings. That's on par with what the Chrome
Web Speech API gives today, with the nice properties that it's local,
free at runtime, and doesn't depend on Chrome being open.

## Caveats from the streaming spike

The script is a naive rolling-window approach (re-transcribe the last
4 s every chunk). Real production streaming should use NeMo's
**cache-aware RNN-T streaming** API — that keeps encoder state across
chunks and avoids the "earlier text drops out as the window slides"
problem visible in the demo output (chunks 14+).

Two specific things to clean up before integrating into the daemon:

1. **Commit-on-aging-out**: as audio leaves the rolling window, treat
   that prefix as committed and never re-emit it. The current daemon
   already has prefix-diffing logic in `keyboard.py` for this kind of
   incremental text — it would slot in cleanly.
2. **VAD-driven boundaries**: chunk 19 emitted an empty string mid-utterance.
   A simple energy-based VAD (or `webrtcvad`) would skip silent chunks and
   reduce noise.

## Comparison with current Chrome path

| dimension          | Chrome Web Speech                    | Parakeet on RTX 3060               |
|--------------------|--------------------------------------|------------------------------------|
| latency            | ~300–500 ms (network)                | ~350 ms (local)                    |
| accuracy           | OK; struggles on names/technical     | top-of-leaderboard English         |
| privacy            | audio to Google                      | 100% local                         |
| dependencies       | Chrome must be running               | python + 5 GB VRAM                 |
| recurring cost     | free (rate-limited)                  | free                               |
| offline            | no                                   | yes                                |

## Files

- `pyproject.toml` — isolated uv project, pins torch+cu124 and NeMo
- `bench.py` — batch transcription benchmark
- `stream.py` — naive rolling-buffer streaming simulator
- `audio/sample.wav` — 7.43 s LibriSpeech utterance for tests

## Recreate the run

```bash
cd experiments/parakeet
uv sync             # ~5–10 min on first install
uv run python bench.py audio/sample.wav
uv run python stream.py audio/sample.wav --chunk-ms 320 --context-ms 4000
```

## Recommended next step

Build a real streaming integration using NeMo's `transcribe_simulate_cache_aware_streaming`
(or the underlying cache-aware streaming API) and wire it into
`daemon_service.py` behind a config flag, so the existing Chrome path
stays as a fallback while Parakeet bakes.
