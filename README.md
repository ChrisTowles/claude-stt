# Claude STT (Fork)

Local speech-to-text input for Claude Code. Hold a hotkey, speak, and your words appear in the focused app.

> **Note**: This is a diverged fork of [jarrodwatts/claude-stt](https://github.com/jarrodwatts/claude-stt).

## Quick Start

```bash
git clone https://github.com/ChrisTowles/claude-stt
cd claude-stt
./install.sh
uv run claude-stt run
```

The daemon loads NVIDIA Parakeet-TDT (~30s on first start), then plays a ready chime. Press **Ctrl+Shift+Space** to start recording, press again to stop. Words appear in real-time as you speak.

## How It Works

```
Press hotkey → daemon opens mic → Parakeet streaming inference (rolling 10s buffer)
    → interim text emitted every ~320ms → typed into focused app via ydotool
Press hotkey → daemon closes mic → final transcript stays typed
```

Recognition runs entirely on your local GPU via NVIDIA's [Parakeet-TDT-0.6B-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (currently #1 on the HuggingFace English ASR leaderboard) through the [NeMo toolkit](https://github.com/NVIDIA/NeMo). No audio leaves your machine.

On an RTX 3060, this hits ~350ms perceived latency with batch RTF around 0.008 (122× real-time).

## Configuration

Settings stored in `~/.config/claude-stt/config.toml`.

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `hotkey` | Key combo | `ctrl+shift+space` | Trigger recording |
| `mode` | `toggle`, `push-to-talk` | `toggle` | Press to toggle vs hold to record |
| `model` | HuggingFace model id | `nvidia/parakeet-tdt-0.6b-v2` | NeMo ASR model |
| `chunk_ms` | 80–2000 | `320` | Streaming chunk size in ms |
| `context_seconds` | 1–60 | `10.0` | Rolling audio buffer length |
| `output_mode` | `auto`, `injection`, `clipboard` | `auto` | How text is inserted |
| `sound_effects` | `true`, `false` | `true` | Play audio feedback |
| `soft_newlines` | `true`, `false` | `true` | Use Shift+Enter for intermediate newlines |
| `max_recording_seconds` | 1-600 | 300 | Maximum recording duration |
| `excluded_apps` | List of app names | `[]` | Skip hotkey when these apps are focused |

## Requirements

- **Linux** with a **CUDA-capable NVIDIA GPU** (≥4 GB VRAM)
- **Python 3.12** with **uv**
- A working microphone (`sounddevice` / PortAudio)

### Platform-Specific

| Platform | Requirements |
|----------|-------------|
| **Linux (Wayland)** | `ydotool` for text injection |
| **Linux (X11)** | `xdotool` for window management |
| **macOS** | Not yet supported — Parakeet-MLX backend on the roadmap |

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

### Recommended: Caps Lock as the toggle (via keyd)

Caps Lock is the ideal dictation key — it's a single press, near the home row, and most people don't use it. The trick is keeping the OS from toggling its caps-lock state when you press it (otherwise typed text comes out case-inverted).

Use [keyd](https://github.com/rvaiya/keyd) to remap Caps Lock to **F13** (an unused keysym), then bind F13 to `claude-stt toggle` in COSMIC.

```bash
# 1. Install keyd (Pop!_OS / Ubuntu):
sudo apt install keyd
sudo systemctl enable --now keyd

# 2. Install the claude-stt remap (or merge `capslock = f13` into your existing
#    /etc/keyd/default.conf under [main] — see configs/keyd/claude-stt.conf):
sudo install -m 0644 configs/keyd/claude-stt.conf /etc/keyd/default.conf
sudo systemctl restart keyd

# 3. Verify Caps Lock now reports as F13:
sudo keyd monitor   # press Caps Lock — you should see "f13"
```

Then in **COSMIC Settings → Keyboard → Custom Shortcuts**, add:

| Field | Value |
|-------|-------|
| Name | claude-stt toggle |
| Command | `/home/<you>/code/f/claude-stt/.venv/bin/python -m claude_stt.daemon toggle` |
| Shortcut | F13 (press Caps Lock to capture) |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA unavailable` warning | Install NVIDIA driver + CUDA-capable GPU; CPU is far too slow |
| First start slow (~30s) | Model is downloading + warming up; cached after |
| `microphone:` error | Run `arecord -l` to confirm a default capture device exists |
| Text not appearing (Wayland) | Install ydotool: `sudo apt install ydotool` |
| Stutter / garbled text mid-utterance | Increase `chunk_ms` to 480–640 in config |

Set `CLAUDE_STT_LOG_LEVEL=DEBUG` for verbose logs.

## Engine History

This project tried several local STT approaches before settling on Parakeet:

1. **faster-whisper (Whisper medium)** — Reliable but slow (~7.4% WER), batch-only.
2. **Cohere Transcribe** — #1 on Open ASR Leaderboard but gated, no streaming.
3. **Qwen3-ASR-1.7B via vLLM** — Streaming support but too slow on a 12 GB GPU.
4. **Chrome Web Speech API** — Streamed nicely but required Chrome running and sent audio to Google.
5. **NVIDIA Parakeet-TDT-0.6B-v2 via NeMo** (current) — Top of the leaderboard for English, runs locally on a single GPU at 122× real-time, ~350 ms perceived latency.

## License

MIT — see [LICENSE](LICENSE)
