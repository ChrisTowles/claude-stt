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

**Daemon + Chrome Web Speech API**: A background daemon manages hotkey events and text output. Chrome's Web Speech API handles speech recognition via a localhost page served by the daemon.

### Core Components

- `daemon.py` - Process management (start/stop/status, PID file handling, background spawning)
- `daemon_service.py` - Runtime orchestration (`STTDaemon` class coordinates all components)
- `hotkey.py` - Global hotkey listener using pynput (supports toggle and push-to-talk modes)
- `web/server.py` - HTTP + WebSocket server (aiohttp). Serves the HTML page and bridges Chrome ↔ daemon
- `web/index.html` - Chrome page with `webkitSpeechRecognition` (streaming, interim results)
- `keyboard.py` - Text output via ydotool (Wayland), pynput (X11), or clipboard fallback
- `window.py` - Platform-specific window tracking to restore focus after transcription
- `config.py` - TOML-based config with validation, stored in `~/.config/claude-stt/`

### Flow

```
Hotkey press → daemon sends "start" via WebSocket → Chrome starts speech recognition
    → interim results streamed back → text typed in real-time via ydotool
Hotkey release → daemon sends "stop" → Chrome stops recognition → final correction if needed
```

Communication runs over WebSocket (`ws://localhost:PORT/ws`). Chrome auto-restarts recognition on silence timeout.

## Task Tracking

Always use the task system (TaskCreate/TaskUpdate) to track objectives, especially when handling multiple tasks simultaneously. Create tasks before starting work and mark them completed when done.

## Git Identity

This repo uses the `ChrisTowles` GitHub account. **Pushing to this repo** — use `./scripts/git-push.sh` instead of `git push`. On macOS it switches to the `ChrisTowles` gh account, pushes, and switches back. On Linux it's a passthrough.

## Version Bumps

Update version in `pyproject.toml`.
