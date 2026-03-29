# Claude STT (Fork)

Speech-to-text input for Claude Code. Hold a hotkey, speak, and your words appear in the focused app.

> **Note**: This is a diverged fork of [jarrodwatts/claude-stt](https://github.com/jarrodwatts/claude-stt).

## Quick Start

```bash
git clone https://github.com/ChrisTowles/claude-stt
cd claude-stt
uv sync
uv run claude-stt run
```

A Chrome window opens automatically. Press **Ctrl+Shift+Space** to start recording, press again to stop. Words appear in real-time as you speak.

## How It Works

```
Press hotkey → daemon sends "start" via WebSocket → Chrome starts listening
    → words streamed back in real-time → typed into focused app via ydotool
Press hotkey → daemon sends "stop" → Chrome stops listening
```

The daemon serves a localhost page (`http://localhost:18333`) that Chrome opens in app mode. The page uses Chrome's Web Speech API for recognition and communicates with the daemon over WebSocket. Text is injected into the focused application via ydotool (Wayland) or pynput (X11).

## Configuration

Settings stored in `~/.config/claude-stt/config.toml`.

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `hotkey` | Key combo | `ctrl+shift+space` | Trigger recording |
| `mode` | `toggle`, `push-to-talk` | `toggle` | Press to toggle vs hold to record |
| `ws_port` | Port number | `18333` | WebSocket/HTTP server port |
| `output_mode` | `auto`, `injection`, `clipboard` | `auto` | How text is inserted |
| `sound_effects` | `true`, `false` | `true` | Play audio feedback |
| `soft_newlines` | `true`, `false` | `true` | Use Shift+Enter for intermediate newlines |
| `max_recording_seconds` | 1-600 | 300 | Maximum recording duration |
| `excluded_apps` | List of app names | `[]` | Skip hotkey when these apps are focused |

## Requirements

- **Python 3.10-3.13** with **uv**
- **Google Chrome** or Chromium
- **Internet connection** (Chrome sends audio to Google for recognition)

### Platform-Specific

| Platform | Requirements |
|----------|-------------|
| **Linux (Wayland)** | `ydotool` for text injection |
| **Linux (X11)** | `xdotool` for window management |
| **macOS** | Accessibility permissions |

## CLI Commands

```bash
claude-stt run                 # Run daemon in foreground
claude-stt start --background  # Run daemon in background
claude-stt stop                # Stop daemon
claude-stt status              # Show daemon status
claude-stt toggle              # Toggle recording via SIGUSR1
```

## Wayland / COSMIC Setup

On Wayland, pynput can't capture global hotkeys. Instead, register the hotkey in your compositor's settings to run:

```
/path/to/claude-stt/.venv/bin/python -m claude_stt.daemon toggle
```

This sends SIGUSR1 to the running daemon to toggle recording.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Chrome window doesn't open | Open `http://localhost:18333` manually |
| "Disconnected" in Chrome | Make sure daemon is running (`stt-run`) |
| No audio / mic denied | Grant mic permission when Chrome prompts |
| Text not appearing (Wayland) | Install ydotool: `sudo apt install ydotool` |

Set `CLAUDE_STT_LOG_LEVEL=DEBUG` for verbose logs.

## Engine History

This project tried several local STT approaches before settling on Chrome Web Speech API:

1. **faster-whisper (Whisper medium)** — Original engine. Reliable but slow (~7.4% WER), batch-only.
2. **Cohere Transcribe** — #1 on Open ASR Leaderboard (5.42% WER). Gated repo, legacy API produced garbled output, no streaming.
3. **Qwen3-ASR-1.7B via vLLM** — Streaming support but too slow for real-time dictation on a 12GB GPU. Multi-second latency, heavy resource usage.
4. **Chrome Web Speech API** (current) — Instant streaming, no GPU, no model loading. Audio processed by Google's servers.

## License

MIT — see [LICENSE](LICENSE)
