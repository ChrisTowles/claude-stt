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

**Daemon-based design**: A background process (`STTDaemon`) runs continuously, listening for hotkey events and coordinating audio capture, transcription, and text output.

### Core Components

- `daemon.py` - Process management (start/stop/status, PID file handling, background spawning)
- `daemon_service.py` - Runtime orchestration (`STTDaemon` class coordinates all components)
- `hotkey.py` - Global hotkey listener using pynput (supports toggle and push-to-talk modes)
- `recorder.py` - Audio capture via sounddevice
- `engines/qwen_asr.py` - Qwen3-ASR STT engine (Qwen/Qwen3-ASR-1.7B via qwen-asr). Auto-detects device: CUDA → MPS → CPU. Override with `CLAUDE_STT_DEVICE` env var.
- `keyboard.py` - Text output via ydotool (Wayland), pynput (X11), or clipboard fallback
- `window.py` - Platform-specific window tracking to restore focus after transcription
- `config.py` - TOML-based config with validation, stored in `~/.config/claude-stt/`

### Flow

```
Hotkey press → AudioRecorder.start() → [user speaks] → Hotkey release
    → AudioRecorder.stop() → Engine.transcribe() → output_text()
```

Transcription runs in a dedicated worker thread to avoid blocking the hotkey listener.

## Task Tracking

Always use the task system (TaskCreate/TaskUpdate) to track objectives, especially when handling multiple tasks simultaneously. Create tasks before starting work and mark them completed when done.

## Git Identity

This repo uses the `ChrisTowles` GitHub account. **Pushing to this repo** — use `./scripts/git-push.sh` instead of `git push`. On macOS it switches to the `ChrisTowles` gh account, pushes, and switches back. On Linux it's a passthrough.

## Version Bumps

Update version in `pyproject.toml`.
