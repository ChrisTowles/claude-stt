"""Simulated streaming Parakeet inference.

Feeds a wav file to the model in fixed-size chunks and measures the
per-chunk wall-clock latency you would see in a live-mic dictation loop.

This is a quick look at the "live feel" — not a production streaming
pipeline.  It uses NeMo's `transcribe` on a rolling buffer; with chunk
sizes around 200-500ms it gives a usable approximation of the perceived
latency the user will see while typing.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel


def chunked(samples: np.ndarray, chunk: int):
    for start in range(0, len(samples), chunk):
        yield samples[start : start + chunk]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--chunk-ms", type=int, default=320, help="Streaming chunk size")
    parser.add_argument(
        "--context-ms",
        type=int,
        default=4000,
        help="Trailing audio context retained between chunks",
    )
    args = parser.parse_args()

    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"expected 16 kHz, got {sr}"

    chunk_n = int(sr * args.chunk_ms / 1000)
    ctx_n = int(sr * args.context_ms / 1000)

    print(f"Loading {args.model} ...")
    model = ASRModel.from_pretrained(model_name=args.model).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Warm-up
    _ = model.transcribe([str(args.audio)], batch_size=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Streaming {len(audio)/sr:.2f}s in {args.chunk_ms} ms chunks "
          f"(context {args.context_ms} ms) ...\n")

    buffer = np.zeros(0, dtype=np.float32)
    last_text = ""
    per_chunk: list[float] = []

    for i, chunk in enumerate(chunked(audio, chunk_n)):
        buffer = np.concatenate([buffer, chunk])
        if len(buffer) > ctx_n:
            buffer = buffer[-ctx_n:]
        t0 = time.perf_counter()
        out = model.transcribe([buffer], batch_size=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        per_chunk.append(dt)
        if isinstance(out, tuple):
            out = out[0]
        first = out[0]
        text = getattr(first, "text", first if isinstance(first, str) else str(first))
        new_suffix = text[len(last_text) :] if text.startswith(last_text) else f"[REWRITE] {text}"
        last_text = text
        print(f"  chunk {i:>3}  +{args.chunk_ms}ms audio   "
              f"infer={dt*1000:6.1f}ms   text={text!r}")

    avg = sum(per_chunk) / len(per_chunk)
    p95 = sorted(per_chunk)[int(0.95 * len(per_chunk))]
    print()
    print(f"chunks:         {len(per_chunk)}")
    print(f"avg latency:    {avg*1000:.1f} ms per {args.chunk_ms} ms chunk")
    print(f"p95 latency:    {p95*1000:.1f} ms")
    print(f"final text:     {last_text}")


if __name__ == "__main__":
    main()
