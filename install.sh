#!/usr/bin/env bash
set -euo pipefail

# Detect OS-specific extras
case "$(uname -s)" in
    Darwin) OS_EXTRA="--extra macos" ;;
    MINGW*|MSYS*|CYGWIN*) OS_EXTRA="--extra windows" ;;
    *) OS_EXTRA="" ;;
esac

# Install dependencies
uv sync --python 3.12 --extra dev $OS_EXTRA

echo "Done. Run tests with: uv run python -m unittest discover -s tests"
