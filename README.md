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

Press **Ctrl+Shift+Space** to start recording, press again to stop and transcribe.

## How It Works

```
Press hotkey -> start recording
       |
Audio captured from microphone
       |
Press hotkey -> stop recording
       |
Speech-to-text via Chrome Web Speech API
       |
Text injected into focused app (ydotool/pynput)
```

- Text injection via ydotool (Wayland) or pynput (X11), with clipboard fallback
- Audio feedback for recording start/stop, transcription complete, daemon ready/shutdown

## Configuration

Settings stored in `~/.config/claude-stt/config.toml`.

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `hotkey` | Key combo | `ctrl+shift+space` | Trigger recording |
| `mode` | `toggle`, `push-to-talk` | `toggle` | Press to toggle vs hold to record |
| `stt_model` | Model name | `Qwen/Qwen3-ASR-1.7B` | STT model (HuggingFace ID) |
| `output_mode` | `auto`, `injection`, `clipboard` | `auto` | How text is inserted |
| `sound_effects` | `true`, `false` | `true` | Play audio feedback |
| `soft_newlines` | `true`, `false` | `true` | Use Shift+Enter for intermediate newlines |
| `max_recording_seconds` | 1-600 | 300 | Maximum recording duration |
| `audio_device` | Device name/index or null | null | Audio input device (null = system default) |
| `excluded_apps` | List of app names | `[]` | Skip hotkey when these apps are focused |

## Requirements

- **Python 3.10-3.13**
- **uv** (package manager)
- **Microphone access**

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

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No audio input | Check microphone permissions |
| Text not appearing (Wayland) | Install ydotool: `sudo apt install ydotool` |
| Text not appearing (X11) | Install xdotool (Linux) or grant Accessibility permissions (macOS) |

Set `CLAUDE_STT_LOG_LEVEL=DEBUG` for verbose logs.

## Engine History

This project tried several local STT approaches before settling on the current architecture:

1. **faster-whisper (Whisper medium)** — Original engine. Reliable but slow (~7.4% WER), batch-only, no streaming.

2. **Cohere Transcribe (cohere-transcribe-03-2026)** — #1 on the Open ASR Leaderboard (5.42% WER). Required a gated HuggingFace repo and `trust_remote_code`. The legacy `model.transcribe()` API produced garbled output; switching to the native `CohereAsrForConditionalGeneration` + `model.generate()` API fixed it, but it was batch-only with no streaming support.

3. **Qwen3-ASR-1.7B** — #6 on the Open ASR Leaderboard (5.76% WER). Supports streaming via vLLM backend. vLLM required significant GPU memory tuning (`max_model_len`, `gpu_memory_utilization`) to fit on a 12GB RTX 3060. Streaming worked but was too slow for real-time dictation — partial results arrived with multi-second latency, and the append-only typing approach couldn't keep up with engine corrections. The vLLM process also consumed substantial system resources.

4. **Chrome Web Speech API** (current) — Chrome's built-in speech recognition provides instant streaming transcription with no model loading, no GPU usage, and no dependency management. Accuracy is comparable to the best local models for English dictation. The tradeoff is that audio is processed by Google's servers rather than locally.

## License

MIT — see [LICENSE](LICENSE)
