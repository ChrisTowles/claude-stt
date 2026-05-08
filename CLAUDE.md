# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# New machine setup (installs uv if needed + all deps)
./install.sh

# Or manually
uv sync --python 3.12

# Run tests
uv run python -m unittest discover -s tests

# Run single test
uv run python -m unittest tests.test_config

# Lint (ruff)
uv run ruff check src/
```

## Conventions

- Hard cutover only — no backwards compatibility. English-only support is fine.

## Architecture

**Daemon + local Parakeet ASR**: A background daemon manages hotkey events, captures audio from the default microphone, and runs streaming inference against NVIDIA Parakeet-TDT (via NeMo) on the local CUDA GPU. Text is typed into the focused app as recognition progresses.

### Core Components

- `daemon.py` - Process management (start/stop/status, PID file handling, background spawning)
- `daemon_service.py` - Runtime orchestration (`STTDaemon` class coordinates all components)
- `asr.py` - `ParakeetEngine`: mic capture (sounddevice) + rolling-buffer streaming inference, emits text via callback
- `hotkey.py` - Global hotkey listener using pynput (supports toggle and push-to-talk modes)
- `keyboard.py` - Text output via ydotool (Wayland), pynput (X11), or clipboard fallback
- `window.py` - Platform-specific window tracking to restore focus after transcription
- `config.py` - TOML-based config with validation, stored in `~/.config/claude-stt/`

### Flow

```
Hotkey press → daemon opens mic → audio chunks queued → Parakeet inference on rolling buffer
    → growing transcript emitted every ~320 ms → typed live via ydotool (with prefix-diff backspaces)
Hotkey release → daemon closes mic → final inference flushes → final transcript remains typed
```

Parakeet-TDT-0.6B-v2 is an offline RNN-T model. We get the "live feel" by re-transcribing the last `context_seconds` of audio after each chunk; the daemon's prefix-diff logic in `_on_text` handles the rare interim corrections by deleting and retyping the differing tail.

### Hardware

Tested on RTX 3060 12 GB + Ryzen 9800X3D. Model uses ~5 GB VRAM (with PyTorch allocator overhead), batch RTF 0.008 (122× real-time), per-chunk inference ~34 ms. macOS / Apple Silicon support is on the roadmap (Parakeet-MLX backend).

## Task Tracking

Always use the task system (TaskCreate/TaskUpdate) to track objectives, especially when handling multiple tasks simultaneously. Create tasks before starting work and mark them completed when done.

## Git Identity

This repo uses the `ChrisTowles` GitHub account. **Pushing to this repo** — use `./scripts/git-push.sh` instead of `git push`. On macOS it switches to the `ChrisTowles` gh account, pushes, and switches back. On Linux it's a passthrough.

## Version Bumps

Update version in `pyproject.toml`.
