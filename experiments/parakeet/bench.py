"""Batch-transcription benchmark for Parakeet-TDT-0.6B-v2 on RTX 3060."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="Path to a 16kHz mono wav file")
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="HuggingFace/NGC model id",
    )
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
    print(f"Loading model:  {args.model}")

    t0 = time.perf_counter()
    model = ASRModel.from_pretrained(model_name=args.model)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    load_s = time.perf_counter() - t0
    print(f"Model loaded in {load_s:.2f}s")

    info = sf.info(str(args.audio))
    audio_s = info.duration
    print(f"Audio:          {args.audio.name}  ({audio_s:.2f}s @ {info.samplerate}Hz)")

    # Warm-up
    print("Warmup ...")
    _ = model.transcribe([str(args.audio)], batch_size=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings: list[float] = []
    text = ""
    for i in range(args.runs):
        t0 = time.perf_counter()
        out = model.transcribe([str(args.audio)], batch_size=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        timings.append(dt)
        # NeMo's output shape varies between versions; normalise
        if isinstance(out, tuple):
            out = out[0]
        first = out[0]
        text = getattr(first, "text", first if isinstance(first, str) else str(first))
        print(f"  run {i+1}: {dt*1000:.1f} ms   RTF={dt/audio_s:.3f}")

    avg = sum(timings) / len(timings)
    rtf = avg / audio_s
    print()
    print(f"avg latency:    {avg*1000:.1f} ms")
    print(f"avg RTF:        {rtf:.3f}   (1/RTF = {1/rtf:.1f}x real-time)")
    print(f"VRAM used:      {torch.cuda.max_memory_allocated()/1e9:.2f} GB" if torch.cuda.is_available() else "")
    print()
    print(f"Transcript:     {text}")


if __name__ == "__main__":
    main()
