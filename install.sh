#!/usr/bin/env bash
set -euo pipefail

# Detect OS-specific extras
case "$(uname -s)" in
    Darwin)
        echo "ERROR: claude-stt currently requires Linux + a CUDA-capable NVIDIA GPU."
        echo "macOS support (Parakeet-MLX or WhisperKit) is on the roadmap."
        exit 1
        ;;
    MINGW*|MSYS*|CYGWIN*) OS_EXTRA="--extra windows" ;;
    *) OS_EXTRA="" ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found. claude-stt needs a CUDA GPU to run Parakeet."
    echo "         CPU inference works but is far too slow for live dictation."
fi

# Install dependencies (~5–10 min on first run; pulls torch+cu124 and NeMo)
uv sync --python 3.12 $OS_EXTRA

echo
echo "Done. First run will download the Parakeet model (~600 MB) on demand."
echo "Run tests with: uv run python -m unittest discover -s tests"
