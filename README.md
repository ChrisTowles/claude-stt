# Claude STT (Fork)

Speech-to-text input for Claude Code. Hold a hotkey, speak, and your words appear in the focused app — all processed locally.

> **Note**: This is a diverged fork of [jarrodwatts/claude-stt](https://github.com/jarrodwatts/claude-stt). While a PR was contributed back upstream, this fork has diverged significantly: switched to uv, removed Moonshine in favor of Whisper-only, removed the Claude Code plugin integration, added ydotool support for Wayland/COSMIC, audio cues for daemon lifecycle, configurable language, optional text improvement via Claude CLI (grammar/punctuation cleanup before typing), and other changes that differ from upstream goals.

## Quick Start

```bash
# Clone and install
git clone https://github.com/ChrisTowles/claude-stt
cd claude-stt
uv sync --extra dev

# Run the daemon
uv run claude-stt run
```

Press **Ctrl+Shift+Space** to start recording, press again to stop and transcribe.

## How It Works

```
Press Ctrl+Shift+Space -> start recording
        |
Audio captured from microphone
        |
Press Ctrl+Shift+Space -> stop recording
        |
Whisper STT processes locally
        |
Text injected into focused app (ydotool/pynput)
```

- Audio is processed in memory and immediately discarded
- Uses faster-whisper for local inference
- Text injection via ydotool (Wayland) or pynput (X11), with clipboard fallback
- Audio feedback for recording start/stop, transcription complete, daemon ready/shutdown

## Configuration

Settings stored in `~/.config/claude-stt/config.toml`.

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `hotkey` | Key combo | `ctrl+shift+space` | Trigger recording |
| `mode` | `toggle`, `push-to-talk` | `toggle` | Press to toggle vs hold to record |
| `whisper_model` | Whisper model name | `medium` | Model size (tiny, base, small, medium, large) |
| `language` | ISO 639-1 code or `auto` | `auto` | Transcription language (`en`, `fr`, etc.) |
| `output_mode` | `auto`, `injection`, `clipboard` | `auto` | How text is inserted |
| `sound_effects` | `true`, `false` | `true` | Play audio feedback |
| `soft_newlines` | `true`, `false` | `true` | Use Shift+Enter for intermediate newlines |
| `improve_text` | `true`, `false` | `false` | Fix grammar/punctuation via Claude CLI |
| `max_recording_seconds` | 1-600 | 300 | Maximum recording duration |
| `audio_device` | Device index or null | null | Audio input device (null = system default) |

## Requirements

- **Python 3.10-3.13**
- **uv** (package manager)
- **Microphone access**

### Platform-Specific

| Platform | Requirements |
|----------|-------------|
| **Linux (Wayland)** | `ydotool` for text injection (required) |
| **Linux (X11)** | `xdotool` for window management |
| **macOS** | Accessibility permissions |
| **Windows** | pywin32 for window tracking |

## CLI Commands

```bash
claude-stt run              # Run daemon in foreground
claude-stt start --background  # Run daemon in background
claude-stt stop             # Stop daemon
claude-stt status           # Show daemon status
claude-stt setup            # First-time setup
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No audio input | Check microphone permissions |
| Text not appearing (Wayland) | Install ydotool: `sudo apt install ydotool` |
| Text not appearing (X11) | Grant Accessibility permissions (macOS) or install xdotool (Linux) |
| Garbled text on COSMIC | Ensure ydotool is installed (wtype has keymap bugs on COSMIC) |
| Model not loading | Run `claude-stt setup` to download. Check disk space |

Set `CLAUDE_STT_LOG_LEVEL=DEBUG` for verbose logs.

## Privacy

All processing is local. No audio or text is sent to external services. No telemetry.

## License

MIT — see [LICENSE](LICENSE)
