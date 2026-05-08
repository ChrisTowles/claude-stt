#!/usr/bin/env bash
set -euo pipefail

case "$(uname -s)" in
    Darwin)
        if [[ "$(uname -m)" != "arm64" ]]; then
            echo "ERROR: macOS support requires Apple Silicon (arm64). Detected: $(uname -m)"
            exit 1
        fi
        ;;
    Linux)
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "WARNING: nvidia-smi not found. claude-stt needs a CUDA GPU to run Parakeet."
            echo "         CPU inference works but is far too slow for live dictation."
        fi
        ;;
esac

# Install dependencies. Platform-specific deps are gated by PEP 508 markers
# in pyproject.toml — Linux/Windows pull torch+cu124+NeMo, macOS pulls parakeet-mlx.
uv sync --python 3.12

echo
echo "Done. First run will download the Parakeet model (~600 MB) on demand."
echo "Run tests with: uv run python -m unittest discover -s tests"
