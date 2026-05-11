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
| `context_seconds` | 1–60 | `30.0` | Rolling audio buffer length (cap on per-recording dictation length) |
| `input_device` | Mic name substring or empty | `""` | Case-insensitive substring match against PortAudio input devices. Empty = system default. The resolved device is logged at daemon startup. |
| `silence_reset_seconds` | 0.5–10 | `1.5` | NeMo backend only — purge the rolling audio buffer after this many seconds of continuous silence so background audio in a pause can't carry forward into the next inference pass. |
| `output_mode` | `auto`, `injection`, `clipboard` | `auto` | How text is inserted |
| `sound_effects` | `true`, `false` | `true` | Play audio feedback |
| `soft_newlines` | `true`, `false` | `true` | Use Shift+Enter for intermediate newlines |
| `max_recording_seconds` | 1-600 | 300 | Maximum recording duration |
| `silence_auto_stop_seconds` | 0-600 | 60 | In toggle mode, auto-stop when no new text has been transcribed for this long (so a forgotten mic doesn't sit open). `0` disables. |
| `excluded_apps` | List of app names | `[]` | Skip hotkey when these apps are focused |

### Picking a specific microphone

Background noise (TVs, music, hallway chatter) is much easier to keep out of transcripts with a near-field mic than with any software filter. If you have a headset or USB cardioid, set `input_device` to a substring of its name:

```toml
[claude-stt]
input_device = "HyperX"
```

To see what's available:

```bash
uv run python -c "import sounddevice as sd; print(sd.query_devices())"
```

The daemon logs the resolved device at startup (`Input device: HyperX QuadCast`). If the configured device isn't present, it falls back to the system default and logs a warning.

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

## Wayland Setup

On Wayland, pynput can't capture global hotkeys, so the daemon also accepts a `SIGUSR1` toggle. Either register `python -m claude_stt.daemon toggle` as a custom shortcut in your compositor, or use the keyd setup below — which bypasses the compositor entirely and is the most reliable option.

### Recommended: Caps Lock as the toggle (via keyd)

Caps Lock is the ideal dictation key — a single press near the home row that most people don't use. The trick is keeping the OS from toggling its caps-lock state when you press it (otherwise typed text comes out case-inverted).

Use [keyd](https://github.com/rvaiya/keyd) to intercept Caps Lock and run a `pkill -USR1` against the daemon directly. No compositor binding required.

```bash
# 1. Install keyd (Pop!_OS / Ubuntu):
sudo apt install keyd
sudo systemctl enable --now keyd

# 2. Install the claude-stt remap (or merge the [main] line from
#    configs/keyd/claude-stt.conf into your existing /etc/keyd/default.conf):
sudo install -m 0644 configs/keyd/claude-stt.conf /etc/keyd/default.conf
sudo systemctl restart keyd

# 3. Verify — pressing Caps Lock should toggle recording in the daemon log:
tail -f ~/.config/claude-stt/daemon.log
```

We previously routed Caps Lock → Ctrl+Alt+F13 → a COSMIC custom shortcut, but COSMIC's dispatch for F13–F24 doesn't fire reliably. The `command()` path skips the compositor and signals the daemon directly.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA unavailable` warning | Install NVIDIA driver + CUDA-capable GPU; CPU is far too slow |
| First start slow (~30s) | Model is downloading + warming up; cached after |
| `microphone:` error | Run `arecord -l` to confirm a default capture device exists |
| Text not appearing (Wayland) | Install ydotool: `sudo apt install ydotool` |
| Stutter / garbled text mid-utterance | Increase `chunk_ms` to 480–640 in config |
| Earlier text gets erased / duplicated on long dictations | Recording exceeded `context_seconds` (default 30s); the rolling audio buffer dropped earlier audio. Pause and start a new recording, or raise `context_seconds` (max 60s). |

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
